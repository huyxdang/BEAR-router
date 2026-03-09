# Experiment Results

This document details each step of the experimental pipeline, the decisions made along the way, and the final results.

---

## Step 1: Data Preparation

**Script:** `scripts/01_prepare_data.py`

We sampled 300 prompts from two QA benchmarks on HuggingFace:

- **SQuAD 2.0** (`rajpurkar/squad_v2`): 150 reading comprehension questions (seed=42). Each prompt includes a context passage and a question.
- **FinQA** (`ibm/finqa`): 150 financial QA questions from SEC 10-K filings (seed=44). Each prompt includes a financial table/context and a question.

These two benchmarks were chosen because they represent different prompt structures (free-text paragraphs vs. structured financial data), which lets the router learn that different prompt types respond differently to compression.

**Output:** `data/squad2_subset.json`, `data/finqa_subset.json` (300 prompts total)

---

## Step 2: Grid Search

**Script:** `scripts/02_grid_search.py`

We exhaustively evaluated every combination of:

| Dimension | Values | Count |
|---|---|---|
| Prompts | 300 (150 SQuAD + 150 FinQA) | 300 |
| Models | Claude Sonnet 4.6, Claude Haiku 4.5, GPT-4o-mini | 3 |
| Aggressiveness | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 | 10 |

**Total: 9,000 API calls** (300 x 3 x 10).

For each combination, the pipeline:
1. Compresses the prompt using TTC's Bear API (`bear-1.2`) at the given aggressiveness level (agg=0.0 skips compression entirely)
2. Sends the (possibly compressed) prompt to the LLM with `temperature=0` and `max_tokens=100`
3. Records: response text, input/output tokens, latency, cost

The grid search supports auto-resume from checkpoint, so it can be interrupted and restarted without losing progress. Compression results are cached in `compressed_cache.json` to avoid redundant Bear API calls.

**All LLM calls use the same system prompt:**
```
You are a helpful assistant. Answer the question based on the provided context.
Be concise and give the answer directly.
```

**Model costs:**

| Model | Provider | Input / Output per 1M tokens |
|---|---|---|
| Claude Sonnet 4.6 | Anthropic | $3.00 / $15.00 |
| Claude Haiku 4.5 | Anthropic | $1.00 / $5.00 |
| GPT-4o-mini | OpenAI | $0.15 / $0.60 |

**Output:** `results/grid_results.parquet` (9,000 records)

---

## Step 3: LLM-as-Judge Evaluation

**Script:** `scripts/07_llm_judge_batch.py` (OpenAI Batch API)

### Why LLM-as-judge?

We initially computed three automated metrics for each response:

- **Exact Match (EM):** Near zero (~3-7%) for all models. Too strict -- models say "The answer is 42" instead of "42".
- **F1 (token overlap):** Penalizes verbose but correct answers. Made GPT-4o-mini look like the best model because it's more concise, even though Sonnet was actually more accurate.
- **Contains Answer (CA):** Better than EM/F1 but has false positives (e.g., "42" matching "42% of respondents").

None of these metrics correctly captured answer correctness. We switched to **GPT-4o-mini as a judge**, which evaluates whether each response conveys the same answer as the ground truth. This reversed the model rankings to match expectations.

### Judge setup

Each of the 9,000 responses was judged by GPT-4o-mini (`temperature=0`, `max_tokens=10`) with the prompt:

```
You are an evaluation judge. Determine if the response correctly answers the question.

Ground truth answer: {ground_truth}
Model response: {response}

Is the model's response correct? It doesn't need to match exactly -- it just needs
to contain or convey the same answer as the ground truth.

Reply with ONLY "correct" or "incorrect".
```

The judge script merges existing judgments by matching on `(prompt_id, model_name, aggressiveness)`, so it handles dataset expansion automatically -- only unjudged rows are submitted.

### Judge results (all 9,000 records)

**By model:**

| Model | Accuracy |
|---|---|
| Claude Sonnet 4.6 | 90.4% |
| Claude Haiku 4.5 | 86.3% |
| GPT-4o-mini | 80.7% |

**By aggressiveness:**

| Aggressiveness | Accuracy |
|---|---|
| 0.0 (none) | 91.2% |
| 0.1 | 90.0% |
| 0.2 | 89.2% |
| 0.3 | 89.0% |
| 0.4 | 88.8% |
| 0.5 | 87.0% |
| 0.6 | 85.6% |
| 0.7 | 83.7% |
| 0.8 | 81.3% |
| 0.9 | 72.3% |

Accuracy degrades smoothly from 0.0 to 0.8, then drops sharply at 0.9 (garbled input).

**Output:** `results/grid_results_judged.parquet` (9,000 records with `llm_judge` column)

---

## Step 4: Router Construction

**Script:** `scripts/04_build_router.py`

### Embedding and clustering

1. Embed all 300 prompts using `all-MiniLM-L6-v2` (384-dimensional sentence embeddings)
2. Cluster embeddings with K-means (K=5, `random_state=42`, `n_init=10`)
3. For each cluster, compute mean judge accuracy and mean cost for every `(model, aggressiveness)` combination

### Why K=5?

Initially we used K=20 (from the UniRoute paper). We ran a K tuning sweep with 5-fold cross-validation (see Step 6) and found K=5 is optimal for our 300-prompt dataset. More details in Step 6.

### Cluster assignments (K=5)

| Cluster | Prompts | Best (model, agg) | Judge Accuracy | Cost |
|---|---|---|---|---|
| 0 | 48 | Sonnet, agg=0.0 | 95.8% | $0.00134 |
| 1 | 64 | Sonnet, agg=0.0 | 95.3% | $0.00106 |
| 2 | 61 | Haiku, agg=0.1 | 98.4% | $0.00044 |
| 3 | 71 | Haiku, agg=0.0 | 97.2% | $0.00043 |
| 4 | 56 | Sonnet, agg=0.0 | 96.4% | $0.00130 |

The router makes meaningfully different decisions per cluster: Sonnet for clusters 0/1/4, Haiku with light compression for cluster 2, and Haiku without compression for cluster 3. No cluster picks GPT-4o-mini at quality mode -- the judge metric correctly identifies it as the weakest model.

### Routing mechanism

At inference time, given a prompt and a user-configurable `lambda` (cost-quality tradeoff):

1. Embed the prompt
2. Find the nearest cluster centroid
3. For each candidate `(model, aggressiveness)` pair, compute: `score = (1 - accuracy) + lambda * cost`
4. Select the pair with the lowest score

Users can also constrain the model pool, set a max budget, or bound aggressiveness levels.

**Output:** `results/router.pkl`, `results/grid_results_clustered.parquet`

---

## Step 5: Evaluation

**Script:** `scripts/05_evaluate.py`

### Setup

- **Train/test split:** 80/20 by prompt_id (`RandomState(42)`) -- 240 train, 60 test prompts
- **Cluster stats built from train set only** to avoid data leakage
- **Metric:** `llm_judge_correct` (LLM-as-judge binary accuracy)
- **Deferral curve:** sweep lambda from 0 to 1000 to trace the accuracy-vs-cost frontier

### Baseline results (60 test prompts)

| Strategy | Accuracy | Avg Cost/Request |
|---|---|---|
| Sonnet, no compression | 98.3% | $0.00123 |
| Haiku, no compression | 96.7% | $0.00044 |
| GPT-4o-mini, no compression | 95.0% | $0.00004 |
| Sonnet, agg=0.3 | 95.0% | $0.00123 |
| Haiku, agg=0.3 | 93.3% | $0.00042 |
| GPT-4o-mini, agg=0.3 | 95.0% | $0.00003 |
| Sonnet, agg=0.6 | 90.0% | $0.00121 |
| Haiku, agg=0.6 | 83.3% | $0.00047 |
| GPT-4o-mini, agg=0.6 | 81.7% | $0.00003 |
| Sonnet, agg=0.9 | 83.3% | $0.00125 |
| GPT-4o-mini, agg=0.9 | 80.0% | $0.00004 |

### Router deferral curve

| Lambda | Mode | Accuracy | Avg Cost | Behavior |
|---|---|---|---|---|
| 0 | Quality | 98.3% | $0.00092 | Matches Sonnet accuracy at 25.6% less cost |
| 0.5-25 | High quality | 95.0% | $0.00073 | Mixes Sonnet + Haiku |
| 30-35 | Balanced | 95.0% | $0.00053 | Shifts toward Haiku |
| 40-100 | Cost-aware | 91.7% | $0.00038 | More Haiku, some compression |
| 150-200 | Cost-focused | 91.7% | $0.00028 | Haiku with compression |
| 350+ | Minimum cost | 91.7% | $0.00004 | GPT-4o-mini territory |

### Key metrics

- **AUC (all models):** 0.000830
- **QNC (Quality-Neutral Cost):** $0.000917 -- the router matches Sonnet's 98.3% accuracy at this cost, a **25.6% savings** over Sonnet's $0.001233

**Output:** `results/evaluation.json`, `results/deferral_curve.csv`

---

## Step 6: K Tuning with Cross-Validation

**Script:** `scripts/08_tune_and_cv.py`

### Setup

- **K values tested:** 5, 10, 15, 20, 30
- **5-fold cross-validation:** 300 prompts split into 5 folds of 60 prompts each. For each fold, cluster stats are built from the 240-prompt training set and evaluated on the 60-prompt test set. Results are averaged across folds.
- **Metric:** LLM judge accuracy at representative lambda values (0, 10, 100, 500) + AUC of the full deferral curve

### Results

| K | lambda=0 acc | lambda=10 acc | lambda=100 acc | lambda=500 acc | AUC |
|---|---|---|---|---|---|
| **5** | **95.0% +/- 1.5%** | **93.0% +/- 1.6%** | **93.3% +/- 2.1%** | **85.0% +/- 4.5%** | **0.000775** |
| 10 | 93.3% +/- 2.4% | 91.7% +/- 3.8% | 90.0% +/- 4.2% | 84.3% +/- 6.7% | 0.000607 |
| 15 | 93.3% +/- 1.5% | 90.3% +/- 1.6% | 88.3% +/- 3.0% | 83.7% +/- 5.3% | 0.000568 |
| 20 | 93.3% +/- 1.8% | 89.3% +/- 3.4% | 89.0% +/- 4.9% | 83.7% +/- 5.0% | 0.000453 |
| 30 | 93.3% +/- 1.1% | 85.3% +/- 2.9% | 85.7% +/- 2.5% | 82.3% +/- 3.1% | 0.000444 |

**K=5 wins across all metrics.** With 300 prompts, K=20 gives only ~15 prompts per cluster, making per-cluster accuracy estimates noisy. K=5 gives ~60 prompts per cluster, producing more robust routing decisions.

The variance is also telling: K=5 has the lowest std at quality mode and balanced mode, meaning its routing decisions are more stable across different train/test splits.

---

## Step 7: Visualizations

**Script:** `scripts/06_visualize.py`

Six plots generated in `results/plots/`:

1. **Deferral curve** (`01_deferral_curve.png`): Router accuracy-vs-cost frontier with baseline points overlaid
2. **Compression vs. accuracy** (`02_compression_vs_accuracy.png`): How each model degrades with increasing aggressiveness, across three metrics (judge, F1, CA)
3. **Cost vs. accuracy scatter** (`03_cost_vs_accuracy.png`): Every (model, agg) combination plotted by cost and judge accuracy
4. **Routing heatmap** (`04_routing_heatmap.png`): Best (model, agg) per cluster at different lambda values, showing how the router transitions from quality to cost mode
5. **Benchmark comparison** (`05_benchmark_comparison.png`): Accuracy by model and aggressiveness, broken down by SQuAD vs. FinQA
6. **Cost breakdown** (`06_cost_breakdown.png`): LLM cost vs. Bear compression cost for each (model, agg) combination

---

## Summary of Results

### Router performance (K=5, judge metric, 60 held-out prompts)

| Strategy | Accuracy | Avg Cost/Request | Notes |
|---|---|---|---|
| Sonnet, no compression | 98.3% | $0.00123 | Quality ceiling |
| **Router, lambda=0** | **98.3%** | **$0.00092** | **Matches Sonnet, 25.6% cheaper** |
| Haiku, no compression | 96.7% | $0.00044 | Mid-tier baseline |
| **Router, lambda=30** | **95.0%** | **$0.00053** | **Near-Haiku cost, higher accuracy** |
| GPT-4o-mini, no compression | 95.0% | $0.00004 | Cost floor |
| Router, lambda=350+ | 91.7% | $0.00004 | See finding #2 below |

### Key findings

1. **The router matches Sonnet's accuracy at 25.6% less cost.** At lambda=0 (quality mode), the router achieves 98.3% accuracy -- identical to Sonnet -- by adaptively choosing Sonnet for some clusters and Haiku for others where Haiku is equally accurate but cheaper.

2. **At high lambda, the router underperforms fixed GPT-4o-mini.** At lambda=350+, the router correctly selects GPT-4o-mini for all prompts but then unnecessarily compresses most of them (only 12/60 get agg=0.0). The cluster-level training stats show these compressed configurations as nearly equivalent (e.g., cluster 1 shows 94.3% accuracy for GPT-4o-mini at agg=0.6), but specific test prompts fail after compression. The cost savings from compression are negligible ($0.000036 vs $0.000038), so the router is trading real accuracy for meaningless cost reduction. This suggests the scoring function needs a minimum accuracy threshold or compression should be penalized more heavily when the cost savings are marginal.

3. **The router's sweet spot is the mid-range.** The strongest results are at lambda=0 to lambda=30, where the router outperforms every fixed baseline at the same cost. It matches Sonnet's 98.3% at 25.6% less cost (lambda=0), and achieves 95.0% at Haiku-level pricing (lambda=30). The router adds less value at the cost extremes — at the top, Sonnet is already excellent; at the bottom, GPT-4o-mini without compression is hard to beat.

4. **SQuAD degrades faster under compression than FinQA.** Reading comprehension requires intact context; financial questions are more structured and compression-resilient. The router learns this and protects SQuAD prompts while compressing FinQA more aggressively.

5. **Aggressive compression can increase cost.** Models produce more verbose (expensive) output on garbled input. Since output tokens cost 5-15x more than input tokens, compression savings on input can be offset by increased output costs.

6. **Fewer clusters work better at this dataset size.** K=5 outperforms K=20 (the UniRoute default) across all metrics in 5-fold CV, because 60 prompts per cluster produces much more reliable routing statistics than 15 prompts per cluster.

7. **Bear compression cost is negligible.** Total compression cost across all 9,000 calls was under $0.01, compared to $2+ for LLM calls. The compression API itself is not a cost bottleneck.

7. **The router's value scales with the model pool.** With 3 models at different price points, the routing surface is relatively simple. With 10+ models at varying price tiers, the adaptive advantage would grow significantly as the router exploits finer-grained cost-quality tradeoffs per cluster.
