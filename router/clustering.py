"""Clustering and per-cluster stats computation."""

import pandas as pd


def compute_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(cluster, model, aggressiveness) statistics.

    Expects df to have columns:
        cluster_id, model_name, aggressiveness, llm_judge_correct,
        total_cost_usd, latency_seconds, compression_ratio
    """
    return df.groupby(["cluster_id", "model_name", "aggressiveness"]).agg(
        mean_judge=("llm_judge_correct", "mean"),
        mean_cost=("total_cost_usd", "mean"),
        mean_latency=("latency_seconds", "mean"),
        mean_compression_ratio=("compression_ratio", "mean"),
        count=("prompt_id", "count"),
    ).reset_index()


def compute_cluster_stats_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight version for evaluation (only judge + cost)."""
    return df.groupby(["cluster_id", "model_name", "aggressiveness"]).agg(
        mean_judge=("llm_judge_correct", "mean"),
        mean_cost=("total_cost_usd", "mean"),
    ).reset_index()
