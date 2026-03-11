"""Evaluate the router against baselines: deferral curves, AUC, QNC."""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODELS, AGGRESSIVENESS_LEVELS, RESULTS_DIR,
    TEST_FRACTION, RANDOM_SEED, BENCHMARKS, DATA_DIR,
)

ROUTER_PATH = os.path.join(str(RESULTS_DIR), "router.pkl")
GRID_PATH = os.path.join(str(RESULTS_DIR), "grid_results_clustered.parquet")

# Lambda values to sweep for deferral curves
LAMBDA_VALUES = np.concatenate([
    np.arange(0, 10, 0.5),
    np.arange(10, 100, 5),
    np.arange(100, 1001, 50),
])


def split_prompts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by prompt_id."""
    prompt_ids = df["prompt_id"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(prompt_ids)

    n_test = max(1, int(len(prompt_ids) * TEST_FRACTION))
    test_ids = set(prompt_ids[:n_test])
    train_ids = set(prompt_ids[n_test:])

    df_train = df[df["prompt_id"].isin(train_ids)]
    df_test = df[df["prompt_id"].isin(test_ids)]

    print(f"  Train: {len(train_ids)} prompts ({len(df_train)} records)")
    print(f"  Test:  {len(test_ids)} prompts ({len(df_test)} records)")
    return df_train, df_test


def build_cluster_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """Build cluster stats from training data only."""
    return df_train.groupby(["cluster_id", "model_name", "aggressiveness"]).agg(
        mean_judge=("llm_judge_correct", "mean"),
        mean_cost=("total_cost_usd", "mean"),
    ).reset_index()


def evaluate_fixed_strategy(df_test: pd.DataFrame, model_name: str,
                            aggressiveness: float) -> dict:
    """Evaluate a fixed (model, agg) strategy on test data."""
    sub = df_test[
        (df_test["model_name"] == model_name) &
        (df_test["aggressiveness"] == aggressiveness)
    ]
    if len(sub) == 0:
        return {"accuracy": 0.0, "cost": 0.0, "count": 0}

    return {
        "accuracy": sub["llm_judge_correct"].mean(),
        "cost": sub["total_cost_usd"].mean(),
        "count": len(sub),
    }


def evaluate_router(df_test: pd.DataFrame, cluster_stats: pd.DataFrame,
                    lambda_: float,
                    models: list[str] | None = None,
                    agg_filter: list[float] | None = None) -> dict:
    """Evaluate the adaptive router at a given λ on test data.

    Args:
        agg_filter: If set, restrict to these aggressiveness levels only.
                    Use [0.0] for the no-compression ablation.
    """
    if models is None:
        models = cluster_stats["model_name"].unique().tolist()

    # For each test prompt, the router picks the best (model, agg) for its cluster
    accuracies = []
    costs = []

    for prompt_id in df_test["prompt_id"].unique():
        prompt_rows = df_test[df_test["prompt_id"] == prompt_id]
        cluster_id = prompt_rows["cluster_id"].iloc[0]

        # Get candidates for this cluster
        candidates = cluster_stats[
            (cluster_stats["cluster_id"] == cluster_id) &
            (cluster_stats["model_name"].isin(models))
        ].copy()

        if agg_filter is not None:
            candidates = candidates[candidates["aggressiveness"].isin(agg_filter)]

        if len(candidates) == 0:
            continue

        # Score and pick best
        candidates["score"] = (1 - candidates["mean_judge"]) + lambda_ * candidates["mean_cost"]
        best = candidates.loc[candidates["score"].idxmin()]
        chosen_model = best["model_name"]
        chosen_agg = best["aggressiveness"]

        # Look up actual test performance for this prompt with that (model, agg)
        actual = prompt_rows[
            (prompt_rows["model_name"] == chosen_model) &
            (prompt_rows["aggressiveness"] == chosen_agg)
        ]
        if len(actual) == 0:
            continue

        accuracies.append(actual["llm_judge_correct"].iloc[0])
        costs.append(actual["total_cost_usd"].iloc[0])

    return {
        "accuracy": np.mean(accuracies) if accuracies else 0.0,
        "cost": np.mean(costs) if costs else 0.0,
        "count": len(accuracies),
    }


def compute_deferral_curve(df_test, cluster_stats, models=None, agg_filter=None):
    """Sweep λ to trace accuracy vs cost curve."""
    points = []
    for lam in LAMBDA_VALUES:
        result = evaluate_router(df_test, cluster_stats, lam, models, agg_filter)
        points.append({
            "lambda": lam,
            "accuracy": result["accuracy"],
            "cost": result["cost"],
        })
    return pd.DataFrame(points)


def compute_auc(curve: pd.DataFrame) -> float:
    """Area under the deferral curve (accuracy vs cost)."""
    # Sort by cost ascending
    curve = curve.sort_values("cost")
    # Remove duplicate cost points
    curve = curve.drop_duplicates(subset="cost", keep="last")
    if len(curve) < 2:
        return 0.0
    return float(np.trapezoid(curve["accuracy"], curve["cost"]))


def compute_qnc(curve: pd.DataFrame, target_accuracy: float) -> float | None:
    """Quality-Neutral Cost: minimum cost to reach target accuracy."""
    curve = curve.sort_values("cost")
    above = curve[curve["accuracy"] >= target_accuracy]
    if len(above) == 0:
        return None
    return float(above["cost"].iloc[0])


def evaluate_openrouter_baseline(df_test: pd.DataFrame) -> dict:
    """Evaluate OpenRouter auto-router on test prompts (no compression).

    Sends each test prompt to OpenRouter restricted to our 6-model pool.
    Returns accuracy and cost. Requires live API calls.
    """
    import asyncio
    from router.llm import call_openrouter_async

    allowed_model_ids = [m["id"] for m in MODELS]

    # Load prompts for text lookup
    prompt_texts = {}
    for bench in BENCHMARKS:
        path = os.path.join(str(DATA_DIR), f"{bench}_subset.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for p in json.load(f):
                prompt_texts[p["id"]] = p["text"]

    # Load ground truths
    ground_truths = {}
    for bench in BENCHMARKS:
        path = os.path.join(str(DATA_DIR), f"{bench}_subset.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for p in json.load(f):
                ground_truths[p["id"]] = p["ground_truth"]

    test_prompt_ids = df_test["prompt_id"].unique()

    async def _run():
        results = []
        for pid in test_prompt_ids:
            text = prompt_texts.get(pid)
            if text is None:
                continue
            try:
                result = await call_openrouter_async(text, allowed_model_ids)
                results.append({
                    "prompt_id": pid,
                    "response_text": result["response_text"],
                    "model_used": result["model_used"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency": result["latency"],
                })
            except Exception as e:
                print(f"    OpenRouter error for {pid}: {e}")
        return results

    print("    Calling OpenRouter API for test prompts...")
    or_results = asyncio.run(_run())

    if not or_results:
        return {"accuracy": 0.0, "cost": 0.0, "count": 0}

    # Judge OpenRouter responses using the same LLM judge
    from router.llm import call_openrouter
    import openai as oai
    from config import OPENAI_API_KEY, JUDGE_MODEL

    judge_client = oai.OpenAI(api_key=OPENAI_API_KEY)

    JUDGE_PROMPT = """You are an evaluation judge. Determine if the response correctly answers the question.

Ground truth answer: {ground_truth}

Model response: {response}

Is the model's response correct? It doesn't need to match exactly — it just needs to contain or convey the same answer as the ground truth.

Reply with ONLY "correct" or "incorrect"."""

    correct = 0
    total = 0
    total_cost = 0.0

    for r in or_results:
        gt = ground_truths.get(r["prompt_id"], "")
        try:
            judge_resp = judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    ground_truth=gt, response=r["response_text"]
                )}],
                max_tokens=10,
                temperature=0,
            )
            verdict = judge_resp.choices[0].message.content.strip().lower()
            if "correct" in verdict and "incorrect" not in verdict:
                correct += 1
        except Exception as e:
            print(f"    Judge error: {e}")

        # Estimate cost from OpenRouter (use input/output tokens with model pricing)
        # OpenRouter charges at model rates, approximate with average pool cost
        total_cost += (r["input_tokens"] + r["output_tokens"]) * 1.0 / 1_000_000
        total += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "cost": total_cost / total if total else 0.0,
        "count": total,
    }


def main():
    print("=" * 80)
    print("ROUTER EVALUATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(GRID_PATH)
    print(f"  {len(df)} records, {df['prompt_id'].nunique()} prompts")

    # Train/test split
    print("\n[2] Train/test split...")
    df_train, df_test = split_prompts(df)

    # Build cluster stats from TRAIN only
    print("\n[3] Building cluster stats from train set...")
    cluster_stats = build_cluster_stats(df_train)
    print(f"  {len(cluster_stats)} (cluster, model, agg) combos")

    print(f"\n  Evaluation metric: LLM-as-judge")

    # ===== BASELINE 1: GPT-5.4 Only (no compression) =====
    print("\n[4] Evaluating baselines...")

    print("\n  --- Baseline 1: GPT-5.4 Only (no router, no compression) ---")
    gpt54_baseline = evaluate_fixed_strategy(df_test, "gpt-5.4", 0.0)
    print(f"  GPT-5.4 acc={gpt54_baseline['accuracy']:.3f}  cost=${gpt54_baseline['cost']:.6f}")

    # ===== BASELINE 2: OpenRouter (same 6-model pool, no compression) =====
    print("\n  --- Baseline 2: OpenRouter (same pool, no compression) ---")
    openrouter_baseline = evaluate_openrouter_baseline(df_test)
    print(f"  OpenRouter acc={openrouter_baseline['accuracy']:.3f}  "
          f"cost=${openrouter_baseline['cost']:.6f}  "
          f"n={openrouter_baseline['count']}")

    # ===== BASELINE 3: UniRoute No Compression (our router, agg=0.0 only) =====
    print("\n  --- Baseline 3: UniRoute No Compression (agg=0.0 only) ---")
    no_compress_curve = compute_deferral_curve(
        df_test, cluster_stats, models=None, agg_filter=[0.0]
    )
    auc_no_compress = compute_auc(no_compress_curve)
    # Pick the λ=0 point as representative
    nc_quality = evaluate_router(df_test, cluster_stats, 0.0, agg_filter=[0.0])
    print(f"  No-compress (λ=0) acc={nc_quality['accuracy']:.3f}  "
          f"cost=${nc_quality['cost']:.6f}")
    print(f"  No-compress AUC={auc_no_compress:.6f}")

    # ===== OUR ROUTER (full: all models × all compression tiers) =====
    print("\n[5] Computing router deferral curve (full)...")
    router_curve = compute_deferral_curve(df_test, cluster_stats)
    auc_router = compute_auc(router_curve)

    print(f"\n[6] Results summary...")
    print(f"  Router AUC (full):          {auc_router:.6f}")
    print(f"  Router AUC (no compress):   {auc_no_compress:.6f}")
    if auc_no_compress > 0:
        auc_gain = (auc_router - auc_no_compress) / auc_no_compress * 100
        print(f"  Compression AUC gain:       {auc_gain:+.1f}%")

    # QNC — minimum cost to match GPT-5.4 baseline accuracy
    qnc = compute_qnc(router_curve, gpt54_baseline["accuracy"])
    print(f"\n  GPT-5.4 baseline acc:       {gpt54_baseline['accuracy']:.3f}")
    print(f"  GPT-5.4 baseline cost:      ${gpt54_baseline['cost']:.6f}")
    if qnc is not None:
        print(f"  QNC (router cost to match): ${qnc:.6f}")
        if gpt54_baseline["cost"] > 0:
            savings = (1 - qnc / gpt54_baseline["cost"]) * 100
            print(f"  Cost savings vs GPT-5.4:    {savings:.1f}%")
    else:
        print(f"  QNC: router cannot match GPT-5.4 accuracy")

    # Print router curve summary
    print(f"\n  Router deferral curve (sampled):")
    print(f"  {'λ':>8s} {'Acc':>6s} {'Cost':>10s}")
    print("  " + "-" * 30)
    for _, row in router_curve.iloc[::5].iterrows():
        print(f"  {row['lambda']:>8.1f} {row['accuracy']:>6.3f} ${row['cost']:>9.6f}")

    # Per-benchmark breakdown
    print("\n[6b] Per-benchmark breakdown...")
    benchmarks = df_test["benchmark"].unique()
    for bench in sorted(benchmarks):
        df_bench = df_test[df_test["benchmark"] == bench]
        df_train_bench = df_train[df_train["benchmark"] == bench]
        cs_bench = build_cluster_stats(df_train_bench)

        print(f"\n  --- {bench} ({df_bench['prompt_id'].nunique()} test prompts) ---")
        gpt54_bench = evaluate_fixed_strategy(df_bench, "gpt-5.4", 0.0)
        print(f"  GPT-5.4:  acc={gpt54_bench['accuracy']:.3f}  cost=${gpt54_bench['cost']:.6f}")

        for lam in [0, 1, 10, 100, 500]:
            r = evaluate_router(df_bench, cs_bench, lam)
            if r["count"] > 0:
                print(f"  {'router λ=' + str(lam):<20s} acc={r['accuracy']:.3f}  cost=${r['cost']:.6f}")

    # Save results
    print("\n[7] Saving evaluation results...")
    eval_results = {
        "baselines": {
            "gpt54_only": gpt54_baseline,
            "openrouter": openrouter_baseline,
        },
        "router_curve": router_curve.to_dict("records"),
        "no_compress_curve": no_compress_curve.to_dict("records"),
        "auc_router": auc_router,
        "auc_no_compress": auc_no_compress,
        "qnc": qnc,
    }

    eval_path = os.path.join(str(RESULTS_DIR), "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"  Saved to {eval_path}")

    router_curve.to_csv(os.path.join(str(RESULTS_DIR), "deferral_curve.csv"), index=False)
    no_compress_curve.to_csv(os.path.join(str(RESULTS_DIR), "deferral_curve_no_compress.csv"), index=False)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
