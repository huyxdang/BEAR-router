"""Prompt embedding via OpenAI text-embedding-3-small with caching."""

import json
import os

import numpy as np
import openai

from config import OPENAI_API_KEY, EMBEDDING_MODEL, RESULTS_DIR

CACHE_PATH = os.path.join(str(RESULTS_DIR), "embeddings_cache.npz")
CACHE_IDS_PATH = os.path.join(str(RESULTS_DIR), "embeddings_ids.json")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str], batch_size: int = 2048) -> np.ndarray:
    """Embed texts using OpenAI API. Handles batching automatically."""
    client = _get_client()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([e.embedding for e in response.data])
    return np.array(all_embeddings)


def embed_single(text: str) -> np.ndarray:
    """Embed a single text."""
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return np.array(response.data[0].embedding)


def embed_and_cache(prompt_ids: list[str], prompt_texts: list[str]) -> np.ndarray:
    """Embed prompts, using cache if available.

    Returns embeddings array aligned with prompt_ids order.
    """
    # Try loading cache
    if os.path.exists(CACHE_PATH) and os.path.exists(CACHE_IDS_PATH):
        cached = np.load(CACHE_PATH)["embeddings"]
        with open(CACHE_IDS_PATH) as f:
            cached_ids = json.load(f)

        # Check if cache covers all requested prompts
        cached_map = {pid: emb for pid, emb in zip(cached_ids, cached)}
        missing = [i for i, pid in enumerate(prompt_ids) if pid not in cached_map]

        if not missing:
            print(f"  Loaded {len(cached_ids)} embeddings from cache")
            return np.array([cached_map[pid] for pid in prompt_ids])

        # Partial cache hit — embed missing ones
        print(f"  Cache hit: {len(prompt_ids) - len(missing)}/{len(prompt_ids)}, "
              f"embedding {len(missing)} new prompts...")
        missing_texts = [prompt_texts[i] for i in missing]
        missing_ids = [prompt_ids[i] for i in missing]
        new_embeddings = embed_texts(missing_texts)

        # Merge into cache
        for pid, emb in zip(missing_ids, new_embeddings):
            cached_map[pid] = emb

        # Save updated cache
        all_ids = list(cached_map.keys())
        all_embs = np.array([cached_map[pid] for pid in all_ids])
        _save_cache(all_ids, all_embs)

        return np.array([cached_map[pid] for pid in prompt_ids])

    # No cache — embed everything
    print(f"  Embedding {len(prompt_ids)} prompts with {EMBEDDING_MODEL}...")
    embeddings = embed_texts(prompt_texts)
    _save_cache(prompt_ids, embeddings)
    return embeddings


def _save_cache(prompt_ids: list[str], embeddings: np.ndarray):
    """Save embeddings cache to disk."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH, embeddings=embeddings)
    with open(CACHE_IDS_PATH, "w") as f:
        json.dump(prompt_ids, f)
