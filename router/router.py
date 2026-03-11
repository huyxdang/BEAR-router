"""Adaptive compression router: embed → cluster → route to optimal (model, agg)."""

import pickle

import numpy as np
import openai

from config import OPENAI_API_KEY, EMBEDDING_MODEL


class Router:
    """Routes prompts to optimal (model, aggressiveness) pairs based on cluster stats."""

    def __init__(self, router_path: str):
        with open(router_path, "rb") as f:
            data = pickle.load(f)

        self.kmeans = data["kmeans"]
        self.cluster_stats = data["cluster_stats"]
        self.models_available = data["models_available"]
        self.agg_levels = data["agg_levels"]
        self._client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL, input=[text]
        )
        return np.array(response.data[0].embedding)

    def route(self, prompt: str, user_config: dict | None = None) -> dict:
        """Route a prompt to the optimal (model, aggressiveness).

        user_config options:
            models: list[str]           — restrict to these models
            max_cost_per_request: float  — hard budget ceiling
            min_aggressiveness: float    — compression floor
            max_aggressiveness: float    — compression ceiling
            lambda_: float              — cost-quality tradeoff (default 1.0)
        """
        if user_config is None:
            user_config = {}

        # 1. Embed and find nearest cluster
        embedding = self.embed(prompt)
        cluster_id = int(self.kmeans.predict(embedding.reshape(1, -1))[0])

        # 2. Get stats for this cluster
        stats = self.cluster_stats
        candidates = stats[stats["cluster_id"] == cluster_id].copy()

        # 3. Filter by user constraints
        models = user_config.get("models", self.models_available)
        candidates = candidates[candidates["model_name"].isin(models)]

        min_agg = user_config.get("min_aggressiveness", min(self.agg_levels))
        max_agg = user_config.get("max_aggressiveness", max(self.agg_levels))
        candidates = candidates[
            (candidates["aggressiveness"] >= min_agg) &
            (candidates["aggressiveness"] <= max_agg)
        ]

        if "max_cost_per_request" in user_config:
            candidates = candidates[
                candidates["mean_cost"] <= user_config["max_cost_per_request"]
            ]

        if len(candidates) == 0:
            return {"error": "No valid (model, aggressiveness) pair satisfies all constraints"}

        # 4. Score: minimize error + λ * cost
        lam = user_config.get("lambda_", 1.0)

        candidates = candidates.copy()
        candidates["score"] = (1 - candidates["mean_judge"]) + lam * candidates["mean_cost"]

        # 5. Pick the best
        best = candidates.loc[candidates["score"].idxmin()]

        return {
            "model": best["model_name"],
            "aggressiveness": float(best["aggressiveness"]),
            "expected_accuracy": float(best["mean_judge"]),
            "expected_cost": float(best["mean_cost"]),
            "expected_latency": float(best["mean_latency"]),
            "cluster_id": cluster_id,
            "score": float(best["score"]),
        }

    def route_batch(self, prompts: list[str], user_config: dict | None = None) -> list[dict]:
        """Route multiple prompts."""
        if user_config is None:
            user_config = {}

        # Batch embed
        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL, input=prompts
        )
        embeddings = np.array([e.embedding for e in response.data])
        cluster_ids = self.kmeans.predict(embeddings)

        results = []
        for i, prompt in enumerate(prompts):
            cluster_id = int(cluster_ids[i])
            result = self._route_by_cluster(cluster_id, user_config)
            results.append(result)

        return results

    def _route_by_cluster(self, cluster_id: int, user_config: dict) -> dict:
        """Route given a known cluster ID (avoids re-embedding)."""
        stats = self.cluster_stats
        candidates = stats[stats["cluster_id"] == cluster_id].copy()

        models = user_config.get("models", self.models_available)
        candidates = candidates[candidates["model_name"].isin(models)]

        min_agg = user_config.get("min_aggressiveness", min(self.agg_levels))
        max_agg = user_config.get("max_aggressiveness", max(self.agg_levels))
        candidates = candidates[
            (candidates["aggressiveness"] >= min_agg) &
            (candidates["aggressiveness"] <= max_agg)
        ]

        if "max_cost_per_request" in user_config:
            candidates = candidates[
                candidates["mean_cost"] <= user_config["max_cost_per_request"]
            ]

        if len(candidates) == 0:
            return {"error": "No valid (model, aggressiveness) pair satisfies all constraints"}

        lam = user_config.get("lambda_", 1.0)

        candidates = candidates.copy()
        candidates["score"] = (1 - candidates["mean_judge"]) + lam * candidates["mean_cost"]

        best = candidates.loc[candidates["score"].idxmin()]

        return {
            "model": best["model_name"],
            "aggressiveness": float(best["aggressiveness"]),
            "expected_accuracy": float(best["mean_judge"]),
            "expected_cost": float(best["mean_cost"]),
            "expected_latency": float(best["mean_latency"]),
            "cluster_id": cluster_id,
            "score": float(best["score"]),
        }
