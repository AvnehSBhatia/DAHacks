"""
Load SentenceTransformer without blocking on Hugging Face Hub metadata calls.

When the model is already in the HF cache, ``local_files_only=True`` avoids extra
HTTP requests during tokenizer init (e.g. ``list_repo_templates``) that can stall
on slow or rate-limited networks. If the snapshot is missing, we fall back to a
normal online load.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


def load_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
    print(
        "[models] Loading SentenceTransformer (local cache first — "
        "avoids Hub stalls when the model is already downloaded)…"
    )
    try:
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception as e:
        print(
            f"[models] local_files_only failed ({type(e).__name__}: {e!s}). "
            "Falling back to Hub (network; set HF_TOKEN for better limits)…"
        )
        return SentenceTransformer(model_name, device=device, local_files_only=False)
