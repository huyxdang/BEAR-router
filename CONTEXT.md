# Context for Future Sessions

This file captures key knowledge, gotchas, and decisions for future Claude sessions working on this project.

---

## Project Status

**What's done:**
- Full pipeline: data prep → grid search → LLM judge → router build → evaluation → visualization
- 3,600 grid search calls completed (4 agg levels × 300 prompts × 3 models)
- LLM-as-judge evaluation on all 3,600 results
- Router built with 20 clusters, evaluated against baselines
- 6 visualization plots generated

**In progress:**
- Expanded grid search: 6 new aggressiveness levels (0.1, 0.2, 0.4, 0.5, 0.7, 0.8) = 5,400 additional LLM calls
- After that: need to judge new results, rebuild router, re-evaluate

**Not yet done:**
- Streamlit demo app
- K tuning (try K=5, 10, 15, 30 instead of 20)
- Cross-validation (currently single 80/20 split)
- Rebuild router.pkl with judge metric (currently uses F1)

---

## API Gotchas

### TTC Bear API
- **Endpoint**: `POST https://api.thetokencompany.com/v1/compress`
- **Model**: `bear-1.2`
- **Auth**: Bearer token via `TTC_API_KEY`
- **Aggressiveness bounds**: Must be >0 AND <1.0. Both 0.0 and 1.0 return HTTP 422.
- **Handling agg=0.0**: Code skips the API call entirely and returns original text with zeroed token counts. Real token counts come from the LLM API response, not bear.
- **Cost**: $0.05 per 1M tokens removed — negligible.

### Anthropic API
- **Empty responses**: Sonnet sometimes returns empty `content` list (especially at agg=0.9 on garbled input). Code handles this by returning `""` instead of crashing on `response.content[0].text`.
- **Credit balance**: If you get 400 "credit balance too low", top up the account.

### OpenAI API
- **Batch API**: `07_llm_judge_batch.py` uses it. Submit a JSONL file, poll for status, download results. Usually completes in 10-30 minutes for small batches. 50% cheaper than real-time.
- **Response content can be None**: Code uses `response.choices[0].message.content or ""` as guard.

---

## Key Technical Decisions

### Why LLM-as-judge instead of F1/EM/CA
- **EM (Exact Match)**: Near zero (~3-7%) for all models. Too strict — models say "The answer is 42" instead of "42".
- **F1 (token overlap)**: Penalizes verbose but correct answers. Made GPT-4o-mini look best because it's more concise, even though Sonnet was actually more accurate.
- **CA (Contains Answer)**: Better than EM/F1 but has false positives (e.g., "42" matching "42% of respondents").
- **LLM Judge**: GPT-4o-mini evaluates correctness. Reversed the model rankings — Sonnet (89%) > Haiku (85%) > GPT-4o-mini (80%). This matches expectations and creates a real cost-quality tradeoff for the router.

### Why these 3 models
- Originally included Mistral Large 3, but the API key was unstable (401 errors). Replaced with GPT-4o-mini.
- The 3 models create a nice spread: cheap (GPT-4o-mini, $0.15/$0.60), mid (Haiku, $1/$5), expensive (Sonnet, $3/$15).

### Compression findings
- **Aggressive compression can INCREASE cost**: Models produce more verbose output on garbled input. Since output tokens cost 5-15x more than input tokens, saving input tokens via compression can backfire.
- **SQuAD2 degrades faster than FinQA**: Reading comprehension needs intact context; financial questions are more structured and resilient.
- **Compression is essentially free**: Bear cost is <1% of total cost.

### System prompt
```python
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based on the provided context. "
    "Be concise and give the answer directly."
)
```
We considered making it more restrictive ("answer in 1-3 words only") but decided against it — the verbose behavior is a real-world signal the router should learn to handle.

---

## Environment

- **Python 3.9** (macOS system Python)
- **No `X | None` syntax** — use `from __future__ import annotations` or `Optional[X]`
- **`.env` file** at project root with: `TTC_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- **Dependencies**: see `requirements.txt`. Key ones: `sentence-transformers`, `scikit-learn`, `pandas`, `anthropic`, `openai`, `httpx`, `matplotlib`
- **`python3` not `python`**: macOS uses `python3` command

---

## File Map

| File | What it does |
|---|---|
| `config.py` | Central config: API keys, model definitions, agg levels, clustering params |
| `router/compress.py` | Bear API wrapper. Skips API at agg=0.0. |
| `router/llm.py` | Sync + async LLM callers. Lazy-initialized singleton clients. Handles empty responses. |
| `router/evaluate.py` | EM, F1, CA metrics + cost calculation with bear cost. |
| `router/router.py` | `Router` class: loads router.pkl, embeds prompt, finds cluster, scores candidates. |
| `scripts/01_prepare_data.py` | Loads SQuAD 2.0 (seed=42) and FinQA (seed=44) from HuggingFace, 150 each. |
| `scripts/02_grid_search.py` | 3-phase grid search: compress → async LLM calls → retry failures. Auto-resume. |
| `scripts/03_validate.py` | Smoke test: 2 prompts × all agg levels × all models. Uses sync `call_llm`. |
| `scripts/04_build_router.py` | Embed prompts → K-means(20) → compute cluster stats → save router.pkl. |
| `scripts/05_evaluate.py` | 80/20 split, deferral curves, AUC, QNC. Uses `llm_judge_correct` metric. |
| `scripts/06_visualize.py` | 6 plots: deferral curve, compression impact, scatter, heatmap, benchmarks, cost. |
| `scripts/07_llm_judge.py` | Direct async API calls to GPT-4o-mini for judging. |
| `scripts/07_llm_judge_batch.py` | OpenAI Batch API version: submit / status / download subcommands. |

---

## Data Files

| File | Records | Description |
|---|---|---|
| `results/grid_results.parquet` | 3,600 (expanding to 9,000) | Raw grid search results |
| `results/grid_results_judged.parquet` | 3,600 | + `llm_judge` and `llm_judge_correct` columns |
| `results/grid_results_clustered.parquet` | 3,600 | + `cluster_id` and judge columns |
| `results/compressed_cache.json` | ~1,230 entries | Keyed by `{prompt_id}_agg{level}` |
| `results/router.pkl` | 1 | K-means model, embeddings, cluster_stats, prompt mappings |

---

## Reproducibility

- SQuAD 2.0 sampling: `seed=42`, 150 samples from HuggingFace `rajpurkar/squad_v2`
- FinQA sampling: `seed=44`, 150 samples from HuggingFace `ibm/finqa`
- K-means: `random_state=42`, `n_init=10`, `n_clusters=20`
- Train/test split: `RandomState(42)`, 20% test
- All LLM calls: `temperature=0`, `max_tokens=100`

---

## Common Tasks

**Re-run grid search after interruption:**
```bash
python3 scripts/02_grid_search.py  # auto-resumes from checkpoint
```

**Judge new results:**
```bash
python3 scripts/07_llm_judge.py  # direct async API
# OR
python3 scripts/07_llm_judge_batch.py submit  # batch API (cheaper, slower)
python3 scripts/07_llm_judge_batch.py status
python3 scripts/07_llm_judge_batch.py download
```

**Rebuild router after new data:**
```bash
# First merge judge results into clustered parquet (see 05_evaluate.py pattern)
python3 scripts/04_build_router.py
python3 scripts/05_evaluate.py
python3 scripts/06_visualize.py
```

### FEEDBACK: 
Honest take:                                                                                                                                                           
                                                                                                                                                                       
  The core idea is genuinely interesting. Joint routing over models AND compression is a real extension of UniRoute that nobody's published. The insight that different  
  prompt types tolerate compression differently — and that this interacts with model choice — is non-obvious and backed by your data. The finding that aggressive
  compression can increase cost (via verbose output) is a real discovery.                                                                                                
                                                                                                                                                                      
  What's strong:                                                                                                                                                         
  - The theoretical grounding from UniRoute gives it credibility
  - The routing heatmap proves the router makes meaningfully different decisions per cluster                                                                             
  - Dynamic model pools at inference time (no retraining) is a clean design                 
  - The pipeline engineering is solid — auto-resume, caching, batch judging                                                                                              
                                                                                                                                                                         
  What's a bit weak right now:                                                                                                                                           
  - The test set is small (60 prompts) — results could shift with more data                                                                                              
  - K=20 was never tuned — the clustering might be noisy                                                                                                                 
  - The router can't beat "just use GPT-4o-mini" on cost, and can't match "just use Sonnet" on accuracy. It lives in the middle. That's useful but not dramatic.         
  - The model pool is lopsided — GPT-4o-mini is so cheap it dominates the cost axis. Models closer in price (e.g., Haiku vs GPT-4o vs Gemini Flash) would make the model 
  routing dimension more interesting.                                                                                                                                    
                                                                                                                                                                         
  For a TTC work trial though? It's strong. You built a working system end-to-end in a day, grounded it in a real paper, and produced quantitative results with real API 
  calls. The CONTEXT.md and pipeline design show engineering maturity. Most work trial submissions are either pure theory or pure hacking — this is both.                
                                                                                                                                                                      
  The main thing I'd emphasize in a writeup: the router's value scales with the model pool. With 3 models it's decent; with 10+ models at varying price points, the      
  routing surface gets much richer and the adaptive advantage grows.  


  ### TODO
  ❯ okay. Pause for now. I'll continue it later. If i want to continue, it's just python3 scripts/02_grid_search.py, right?                                                
                                                                                                                                                                         
⏺ Yep, exactly. python3 scripts/02_grid_search.py — it'll auto-resume from the checkpoint and pick up the remaining ~1,100 calls.                                        
                                                                                                                                                                       
  After it finishes, the next steps are:                                                                                                                                 
  1. python3 scripts/07_llm_judge.py — judge the new results                                                                                                             
  2. Merge judge data into clustered parquet                                                                                                                             
  3. python3 scripts/04_build_router.py — rebuild with all 10 agg levels                                                                                              
  4. python3 scripts/05_evaluate.py — re-evaluate                                                                                                                        
  5. python3 scripts/06_visualize.py — regenerate plots 