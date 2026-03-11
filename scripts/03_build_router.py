"""Build the routing table: embed prompts, cluster, compute per-cluster stats."""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    EMBEDDING_MODEL, K_VALUES, AGGRESSIVENESS_LEVELS, MODELS,
    DATA_DIR, RESULTS_DIR, BENCHMARKS, OPENAI_API_KEY,
)

GRID_RESULTS_PATH = os.path.join(str(RESULTS_DIR), "grid_results_judged.parquet")
ROUTER_PATH = os.path.join(str(RESULTS_DIR), "router.pkl")


def load_prompts() -> list[dict]:
    prompts = []
    for bench in BENCHMARKS:
        path = os.path.join(str(DATA_DIR), f"{bench}_subset.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            prompts.extend(json.load(f))
    return prompts


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using OpenAI text-embedding-3-small."""
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # OpenAI embeddings API supports batches up to 2048
    all_embeddings = []
    batch_size = 2048
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([e.embedding for e in response.data])
    return np.array(all_embeddings)


def main():
    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    print("=" * 80)
    print("BUILD ROUTER")
    print("=" * 80)

    # Step 1: Load grid results
    print("\n[1] Loading grid results...")
    df = pd.read_parquet(GRID_RESULTS_PATH)
    print(f"  {len(df)} records")

    # Step 2: Load prompts and embed
    print(f"\n[2] Embedding prompts with {EMBEDDING_MODEL}...")
    prompts = load_prompts()
    prompt_texts = [p["text"] for p in prompts]
    prompt_ids = [p["id"] for p in prompts]

    embeddings = embed_texts(prompt_texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 3: K-means clustering — sweep K values, pick best by AUC later
    print(f"\n[3] Clustering with K values: {K_VALUES}")

    best_k = K_VALUES[0]
    best_kmeans = None
    best_labels = None

    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  K={k:3d}: min_cluster={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}")

        # Default to first K; 05_evaluate.py will select best K by AUC
        if best_kmeans is None:
            best_kmeans = kmeans
            best_labels = labels
            best_k = k

    # Use middle K as default; final selection happens in evaluation
    mid_idx = len(K_VALUES) // 2
    default_k = K_VALUES[mid_idx]
    print(f"\n  Default K={default_k} (final selection by AUC in evaluation)")
    kmeans = KMeans(n_clusters=default_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Map prompt_id → cluster_id
    prompt_to_cluster = dict(zip(prompt_ids, cluster_labels))

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, median={np.median(counts):.0f}")

    # Step 4: Assign cluster IDs to grid results
    print("\n[4] Assigning cluster IDs to grid results...")
    df["cluster_id"] = df["prompt_id"].map(prompt_to_cluster)
    missing = df["cluster_id"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} records have no cluster assignment")
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Step 5: Compute per-cluster stats
    print("\n[5] Computing per-cluster statistics...")

    # Add binary judge correctness column
    df["llm_judge_correct"] = (df["llm_judge"] == "correct").astype(float)

    cluster_stats = df.groupby(["cluster_id", "model_name", "aggressiveness"]).agg(
        mean_judge=("llm_judge_correct", "mean"),
        mean_cost=("total_cost_usd", "mean"),
        mean_latency=("latency_seconds", "mean"),
        mean_compression_ratio=("compression_ratio", "mean"),
        count=("prompt_id", "count"),
    ).reset_index()

    n_combos = len(cluster_stats)
    print(f"  {n_combos} (cluster, model, agg) combinations")
    print(f"  Expected: {default_k} x {len(MODELS)} x {len(AGGRESSIVENESS_LEVELS)} = "
          f"{default_k * len(MODELS) * len(AGGRESSIVENESS_LEVELS)}")

    # Step 6: Quick sanity check — print best combo per cluster (by judge accuracy)
    print("\n[6] Best (model, agg) per cluster (by judge accuracy):")
    for cid in sorted(cluster_stats["cluster_id"].unique()):
        sub = cluster_stats[cluster_stats["cluster_id"] == cid]
        best = sub.loc[sub["mean_judge"].idxmax()]
        print(f"  Cluster {cid:2d} ({int(best['count']):2d} prompts): "
              f"{best['model_name']:<16s} agg={best['aggressiveness']:.1f}  "
              f"judge={best['mean_judge']:.3f}  cost=${best['mean_cost']:.5f}")

    # Step 7: Save everything
    print("\n[7] Saving router artifacts...")
    router_data = {
        "kmeans": kmeans,
        "embedder_name": EMBEDDING_MODEL,
        "cluster_stats": cluster_stats,
        "embeddings": embeddings,
        "prompt_ids": prompt_ids,
        "prompt_to_cluster": prompt_to_cluster,
        "models_available": [m["name"] for m in MODELS],
        "agg_levels": AGGRESSIVENESS_LEVELS,
        "n_clusters": default_k,
    }
    with open(ROUTER_PATH, "wb") as f:
        pickle.dump(router_data, f)
    print(f"  Saved to {ROUTER_PATH}")

    # Also save enriched grid results with cluster IDs
    enriched_path = os.path.join(str(RESULTS_DIR), "grid_results_clustered.parquet")
    df.to_parquet(enriched_path, index=False)
    print(f"  Enriched results saved to {enriched_path}")

    print("\n" + "=" * 80)
    print("ROUTER BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
