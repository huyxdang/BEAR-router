"""Embed all prompts via Modal GPU and cache locally.

Usage:
  modal run scripts/embed.py
"""

from __future__ import annotations

import modal

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_DIR = "/models"

app = modal.App("bear-router-embed")

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers>=2.7.0", "numpy")
)

model_volume = modal.Volume.from_name(
    "bear-embed-model-cache", create_if_missing=True
)


@app.cls(
    image=embed_image,
    gpu="T4",
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    scaledown_window=120,
)
class Embedder:
    @modal.enter()
    def load_model(self):
        from pathlib import Path
        from sentence_transformers import SentenceTransformer

        model_path = Path(MODEL_DIR) / MODEL_ID
        if not model_path.exists():
            print(f"Downloading {MODEL_ID}...")
            self.model = SentenceTransformer(
                MODEL_ID, device="cuda", cache_folder=MODEL_DIR
            )
            model_volume.commit()
        else:
            self.model = SentenceTransformer(str(model_path), device="cuda")

    @modal.method()
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embs.tolist()


@app.local_entrypoint()
def main(batch_size: int = 512):
    import json
    import os
    import sys

    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config import RESULTS_DIR
    from router.data import load_prompts
    from router.embeddings import CACHE_PATH, CACHE_IDS_PATH, _save_cache

    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    print("[1] Loading prompts...")
    prompts = load_prompts()
    prompt_ids = [p["id"] for p in prompts]
    prompt_texts = [p["text"] for p in prompts]
    print(f"  {len(prompts)} prompts")

    # Check cache for partial hits
    already_done = set()
    if os.path.exists(CACHE_PATH) and os.path.exists(CACHE_IDS_PATH):
        cached = np.load(CACHE_PATH)["embeddings"]
        with open(CACHE_IDS_PATH) as f:
            cached_ids = json.load(f)
        already_done = set(cached_ids)
        print(f"  Cache has {len(already_done)} embeddings already")

    to_embed_idx = [i for i, pid in enumerate(prompt_ids) if pid not in already_done]
    if not to_embed_idx:
        print("  All prompts already cached!")
        return

    texts_to_embed = [prompt_texts[i] for i in to_embed_idx]
    ids_to_embed = [prompt_ids[i] for i in to_embed_idx]
    print(f"  Embedding {len(texts_to_embed)} new prompts...")

    batches = [
        texts_to_embed[i:i + batch_size]
        for i in range(0, len(texts_to_embed), batch_size)
    ]

    print(f"[2] Embedding on Modal ({len(batches)} batches)...")
    embedder = Embedder()

    all_embs = []
    for i, batch_embs in enumerate(embedder.embed_batch.map(batches)):
        all_embs.extend(batch_embs)
        print(f"  Batch {i + 1}/{len(batches)} | {len(all_embs)}/{len(texts_to_embed)} done")

    new_embs = np.array(all_embs)

    # Merge with existing cache
    if already_done:
        cached = np.load(CACHE_PATH)["embeddings"]
        with open(CACHE_IDS_PATH) as f:
            cached_ids = json.load(f)
        all_ids = cached_ids + ids_to_embed
        all_embeddings = np.vstack([cached, new_embs])
    else:
        all_ids = ids_to_embed
        all_embeddings = new_embs

    print(f"[3] Saving cache ({len(all_ids)} embeddings, {all_embeddings.shape[1]}-dim)...")
    _save_cache(all_ids, all_embeddings)
    print(f"  Saved to {CACHE_PATH}")
    print("DONE")
