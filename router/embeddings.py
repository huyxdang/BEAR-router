"""Prompt embedding with local caching.

Embeddings are computed via Modal GPU (scripts/embed.py) and cached locally.
All other scripts just read from cache via embed_and_cache().
"""

import json
import os

import numpy as np

from config import EMBEDDING_MODEL, RESULTS_DIR

CACHE_PATH = os.path.join(str(RESULTS_DIR), "embeddings_cache.npz")
CACHE_IDS_PATH = os.path.join(str(RESULTS_DIR), "embeddings_ids.json")


def embed_and_cache(prompt_ids: list[str], prompt_texts: list[str]) -> np.ndarray:
    """Load embeddings from cache. Raises if cache is missing or incomplete.

    Run `modal run scripts/embed.py` first to populate the cache.
    """
    if not os.path.exists(CACHE_PATH) or not os.path.exists(CACHE_IDS_PATH):
        raise SystemExit(
            f"Embeddings cache not found. Run: modal run scripts/embed.py"
        )

    cached = np.load(CACHE_PATH)["embeddings"]
    with open(CACHE_IDS_PATH) as f:
        cached_ids = json.load(f)

    cached_map = {pid: emb for pid, emb in zip(cached_ids, cached)}
    missing = [pid for pid in prompt_ids if pid not in cached_map]

    if missing:
        raise SystemExit(
            f"{len(missing)} prompts not in cache. Re-run: modal run scripts/embed.py"
        )

    print(f"  Loaded {len(prompt_ids)} embeddings from cache ({cached.shape[1]}-dim)")
    return np.array([cached_map[pid] for pid in prompt_ids])


def _save_cache(prompt_ids: list[str], embeddings: np.ndarray):
    """Save embeddings cache to disk."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH, embeddings=embeddings)
    with open(CACHE_IDS_PATH, "w") as f:
        json.dump(prompt_ids, f)


# Lazy-loaded local embedding model
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_single(text: str) -> np.ndarray:
    """Encode a single string, return 1-D embedding array."""
    model = _get_model()
    return model.encode(text, show_progress_bar=False)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a batch of strings, return 2-D array (len(texts), dim)."""
    model = _get_model()
    return model.encode(texts, show_progress_bar=False)
