# Context for Future Sessions

This file captures key knowledge, gotchas, and decisions for future Claude sessions working on this project.

---

## Project Status

**What's done:**
- Full pipeline: data prep → grid search → LLM judge → router build → evaluation → visualization
- 18,000 grid search calls completed (10 agg levels × 300 prompts × 6 models)
- LLM-as-judge evaluation on all 18,000 results
- Router built with K=5 clusters, evaluated against baselines
- Per-benchmark evaluation breakdown (SQuAD2 vs FinQA)
- 6 visualization plots generated

**Model expansion history:**
- Started with 3 models (claude-haiku, claude-sonnet, gpt-4o-mini)
- Added 3 OpenAI models (gpt-5.4, gpt-4.1, gpt-4.1-nano)
- Attempted Google Gemini models but removed due to strict daily rate limits (250-1,500 RPD on paid tier 1)
- Replaced reasoning models (gpt-5, gpt-5-mini) with non-reasoning alternatives (gpt-4.1, gpt-4.1-nano) — reasoning models don't support temperature=0 and waste tokens on hidden reasoning

**Not yet done:**
- Streamlit demo app
- Update RESULTS.md with 6-model findings

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
- **`max_completion_tokens` not `max_tokens`**: Newer OpenAI models (gpt-5.4, gpt-4.1, etc.) require `max_completion_tokens`. The code uses this for all OpenAI models.
- **Reasoning models (gpt-5, gpt-5-mini)**: Don't support `temperature=0`, waste tokens on hidden reasoning. We replaced them with gpt-4.1 and gpt-4.1-nano to avoid these issues.

### Google Gemini API
- **Removed from project** due to strict rate limits (250 RPD for pro models, ~1,500 RPD for flash on paid tier 1). Would need ~3,000 calls per model.
- **SDK**: `google-genai` package, `genai.Client(api_key=...)`. Sync: `client.models.generate_content(...)`, Async: `client.aio.models.generate_content(...)`.
- **Token counting**: `response.usage_metadata.prompt_token_count` and `candidates_token_count`.

---

## Key Technical Decisions

### Why LLM-as-judge instead of F1/EM/CA
- **EM (Exact Match)**: Near zero (~3-7%) for all models. Too strict — models say "The answer is 42" instead of "42".
- **F1 (token overlap)**: Penalizes verbose but correct answers. Made GPT-4o-mini look best because it's more concise, even though Sonnet was actually more accurate.
- **CA (Contains Answer)**: Better than EM/F1 but has false positives (e.g., "42" matching "42% of respondents").
- **LLM Judge**: GPT-4o-mini evaluates correctness. Reversed the model rankings — Sonnet (89%) > Haiku (85%) > GPT-4o-mini (80%). This matches expectations and creates a real cost-quality tradeoff for the router.

### Why these 6 models
- **Claude Haiku 4.5** ($1/$5): Mid-tier Anthropic, good balance
- **Claude Sonnet 4.6** ($3/$15): Quality ceiling
- **GPT-4o-mini** ($0.15/$0.60): Legacy cheap model
- **GPT-4.1-nano** ($0.10/$0.40): Ultra-cheap, fastest model
- **GPT-4.1** ($2/$8): High-quality non-reasoning OpenAI
- **GPT-5.4** ($2.50/$15): Premium OpenAI

The 6 models span a 25x cost range ($0.10-$2.50 input), giving the router a rich routing surface.

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

- **Python 3.11** (via venv: `venv/bin/python`)
- **No `X | None` syntax** — use `from __future__ import annotations` or `Optional[X]`
- **`.env` file** at project root with: `TTC_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- **Dependencies**: see `requirements.txt`. Key ones: `sentence-transformers`, `scikit-learn`, `pandas`, `anthropic`, `openai`, `httpx`, `matplotlib`, `google-genai`
- **`venv/bin/python`**: Use this to run scripts, not system python

---

## File Map

| File | What it does |
|---|---|
| `config.py` | Central config: API keys, model definitions (6 models), agg levels, clustering params |
| `router/compress.py` | Bear API wrapper. Skips API at agg=0.0. |
| `router/llm.py` | Sync + async LLM callers for Anthropic + OpenAI. Lazy-initialized singleton clients. Google support present but unused. |
| `router/evaluate.py` | EM, F1, CA metrics + cost calculation with bear cost. |
| `router/router.py` | `Router` class: loads router.pkl, embeds prompt, finds cluster, scores candidates. |
| `scripts/01_prepare_data.py` | Loads SQuAD 2.0 (seed=42) and FinQA (seed=44) from HuggingFace, 150 each. |
| `scripts/02_grid_search.py` | 3-phase grid search: compress → async LLM calls → retry failures. Auto-resume. Rate limit detection with exponential backoff. |
| `scripts/03_validate.py` | Smoke test: 1 prompt per benchmark × 3 agg levels × all models. Uses sync `call_llm`. |
| `scripts/04_build_router.py` | Embed prompts → K-means(5) → compute cluster stats → save router.pkl. |
| `scripts/05_evaluate.py` | 80/20 split, deferral curves, AUC, QNC. Per-benchmark breakdown. Auto-generates baselines for all models. |
| `scripts/06_visualize.py` | 6 plots: deferral curve, compression impact, scatter, heatmap, benchmarks, cost. Supports dynamic model sets. |
| `scripts/07_llm_judge.py` | Direct async API calls to GPT-4o-mini for judging. |
| `scripts/07_llm_judge_batch.py` | OpenAI Batch API version: submit / status / download subcommands. |

---

## Data Files

| File | Records | Description |
|---|---|---|
| `results/grid_results.parquet` | 18,000 | Raw grid search results (6 models × 300 prompts × 10 agg) |
| `results/grid_results_judged.parquet` | 18,000 | + `llm_judge` column |
| `results/grid_results_clustered.parquet` | 18,000 | + `cluster_id` and `llm_judge_correct` columns |
| `results/compressed_cache.json` | 3,000 entries | Keyed by `{prompt_id}_agg{level}` |
| `results/router.pkl` | 1 | K-means model, embeddings, cluster_stats, prompt mappings |

---

## Reproducibility

- SQuAD 2.0 sampling: `seed=42`, 150 samples from HuggingFace `rajpurkar/squad_v2`
- FinQA sampling: `seed=44`, 150 samples from HuggingFace `ibm/finqa`
- K-means: `random_state=42`, `n_init=10`, `n_clusters=5`
- Train/test split: `RandomState(42)`, 20% test
- All LLM calls: `temperature=0`, `max_output_tokens=256`

---

## Common Tasks

**Re-run grid search after interruption:**
```bash
venv/bin/python scripts/02_grid_search.py  # auto-resumes from checkpoint
```

**Judge new results:**
```bash
venv/bin/python scripts/07_llm_judge_batch.py submit  # batch API (cheaper, slower)
venv/bin/python scripts/07_llm_judge_batch.py status
venv/bin/python scripts/07_llm_judge_batch.py download
```

**Rebuild router after new data:**
```bash
venv/bin/python scripts/04_build_router.py
venv/bin/python scripts/05_evaluate.py
venv/bin/python scripts/06_visualize.py
```
