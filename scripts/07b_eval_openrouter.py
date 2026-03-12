"""Evaluate OpenRouter auto-router baseline on test prompts (no compression).

Sends each test prompt to OpenRouter with no model restrictions — OpenRouter's
auto-router picks freely from its full pool. Judges via Modal (Qwen2.5-7B)
and merges results into evaluation.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODELS, RESULTS_DIR, BATCH_SIZE,
    OPENROUTER_MODEL_IDS, SYSTEM_PROMPTS, DEFAULT_SYSTEM_PROMPT,
)
# Cost lookup: map known OpenRouter model IDs back to our configs for cost calc
_OR_TO_MODEL_CONFIG = {
    OPENROUTER_MODEL_IDS[m["id"]]: m
    for m in MODELS
    if m["id"] in OPENROUTER_MODEL_IDS
}
from router.data import load_prompts, load_ground_truths
from router.evaluate import compute_cost
from router.llm import call_openrouter_async
from router.judge import judge_responses_async

SPLITS_PATH = os.path.join(str(RESULTS_DIR), "router_splits.json")
GRID_PATH = os.path.join(str(RESULTS_DIR), "grid_results_clustered.parquet")
EVAL_PATH = os.path.join(str(RESULTS_DIR), "evaluation.json")
OR_RESULTS_PATH = os.path.join(str(RESULTS_DIR), "openrouter_results.json")

_api_semaphore = None


async def run_openrouter_eval():
    global _api_semaphore
    _api_semaphore = asyncio.Semaphore(BATCH_SIZE)

    print("=" * 80)
    print("OPENROUTER BASELINE EVALUATION")
    print("=" * 80)

    # Load test split
    print("\n[1] Loading test split...")
    if not os.path.exists(GRID_PATH):
        fallback = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")
        if os.path.exists(fallback):
            df = pd.read_parquet(fallback)
        else:
            raise SystemExit("Missing grid results. Run 06_build_router.py first.")
    else:
        df = pd.read_parquet(GRID_PATH)

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_ids = set(splits["test_ids"])
    df_test = df[df["prompt_id"].isin(test_ids)]
    test_prompt_ids = df_test["prompt_id"].unique()
    print(f"  {len(test_prompt_ids)} test prompts")

    # Build lookups
    prompt_lookup = {p["id"]: p for p in load_prompts()}
    ground_truths = load_ground_truths()

    # Phase 1: Call OpenRouter (unrestricted — no model list)
    print(f"\n[2] Calling OpenRouter for {len(test_prompt_ids)} prompts (unrestricted)...")

    async def _call_one(pid):
        prompt = prompt_lookup.get(pid)
        if prompt is None:
            return None
        text = prompt["text"]
        sys_prompt = SYSTEM_PROMPTS.get(prompt.get("benchmark", ""), DEFAULT_SYSTEM_PROMPT)
        async with _api_semaphore:
            try:
                result = await call_openrouter_async(text, None, sys_prompt)
                return {
                    "prompt_id": pid,
                    "response_text": result["response_text"],
                    "model_used": result["model_used"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency": result["latency"],
                }
            except Exception as e:
                print(f"  ERROR {pid}: {e}")
                return None

    tasks = [_call_one(pid) for pid in test_prompt_ids]
    raw_results = await asyncio.gather(*tasks)
    or_results = [r for r in raw_results if r is not None]
    print(f"  Got {len(or_results)}/{len(test_prompt_ids)} responses")

    if not or_results:
        print("  No results — aborting.")
        return

    # Show model distribution
    model_counts = {}
    for r in or_results:
        m = r.get("model_used", "unknown")
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n  Model distribution:")
    for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {m}: {c}")

    # Phase 2: Judge via Modal
    print(f"\n[3] Judging {len(or_results)} responses via Modal...")
    gt_list = [ground_truths.get(r["prompt_id"], "") for r in or_results]
    resp_list = [r["response_text"] for r in or_results]
    verdicts = await judge_responses_async(gt_list, resp_list)

    # Phase 3: Compute metrics
    print("\n[4] Computing metrics...")
    correct = sum(1 for v in verdicts if v == "correct")
    total = len(or_results)

    total_cost = 0.0
    unknown_model_count = 0
    unknown_model_names = set()
    # Per-model tracking
    model_correct = {}
    model_total = {}
    for r, v in zip(or_results, verdicts):
        m = r.get("model_used", "unknown")
        model_total[m] = model_total.get(m, 0) + 1
        if v == "correct":
            model_correct[m] = model_correct.get(m, 0) + 1

        model_used = (m or "").split(":")[0]
        model_config = _OR_TO_MODEL_CONFIG.get(model_used)
        if model_config is None:
            unknown_model_count += 1
            unknown_model_names.add(m)
            continue
        cost = compute_cost(model_config, r["input_tokens"], r["output_tokens"], tokens_removed=0)
        total_cost += cost["total_cost_usd"]

    cost_count = total - unknown_model_count
    mean_cost = total_cost / cost_count if cost_count else 0.0

    accuracy = correct / total if total else 0.0
    print(f"\n  Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"  Mean cost: ${mean_cost:.6f}")

    # Per-model accuracy breakdown
    print(f"\n  Per-model accuracy:")
    for m, cnt in sorted(model_total.items(), key=lambda x: -x[1]):
        acc = model_correct.get(m, 0) / cnt
        print(f"    {m}: {acc:.3f} ({model_correct.get(m, 0)}/{cnt})")

    if unknown_model_count:
        print(f"\n  *** WARNING: {unknown_model_count}/{total} responses used unknown models (excluded from cost) ***")
        print(f"  Unknown models: {', '.join(sorted(unknown_model_names))}")
        if unknown_model_count > total * 0.5:
            print(f"  *** COST METRIC IS UNRELIABLE — majority ({unknown_model_count}/{total}) of responses used unknown models ***")

    baseline_result = {
        "accuracy": accuracy,
        "cost": mean_cost,
        "count": total,
        "model_distribution": model_counts,
    }

    # Save detailed results
    detailed = []
    for r, v in zip(or_results, verdicts):
        detailed.append({**r, "verdict": v})

    with open(OR_RESULTS_PATH, "w") as f:
        json.dump({"summary": baseline_result, "details": detailed}, f, indent=2, default=str)
    print(f"\n  Saved detailed results to {OR_RESULTS_PATH}")

    # Merge into evaluation.json
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            eval_data = json.load(f)
        eval_data["baselines"]["openrouter"] = baseline_result
        with open(EVAL_PATH, "w") as f:
            json.dump(eval_data, f, indent=2, default=str)
        print(f"  Updated OpenRouter baseline in {EVAL_PATH}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_openrouter_eval())
