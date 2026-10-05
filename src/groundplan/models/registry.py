"""Model weights and a deterministic output cache.

Weights are fetched from the Hugging Face hub on first use (or ahead of time with
``scripts/fetch_models.py``) into the standard HF cache; nothing is fetched from our own
infrastructure. Model outputs are cached under ``~/.cache/groundplan/model_outputs`` keyed by the
SHA-256 of the exact input bytes, the model id and its revision, so a cached run replays
bit-identically and ``--no-cache`` runs the live path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("GROUNDPLAN_CACHE_DIR", Path.home() / ".cache" / "groundplan"))

# model key -> (hub id, pinned revision). Revisions are pinned once verified; see the README disclosure.
MODELS: dict[str, tuple[str, str]] = {
    "clip": ("openai/clip-vit-base-patch32", "main"),
    "depth_da2": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "main"),
    "depth_moge2": ("Ruicheng/moge-2-vits-normal", "main"),
}

_USE_CACHE = True
_STATS = {"hit": 0, "miss": 0}


def set_cache(enabled: bool) -> None:
    global _USE_CACHE
    _USE_CACHE = enabled


def cache_stats() -> str:
    if not _USE_CACHE:
        return "off"
    if _STATS["hit"] == 0 and _STATS["miss"] == 0:
        return "n/a"
    if _STATS["miss"] == 0:
        return "hit"
    if _STATS["hit"] == 0:
        return "miss"
    return "mixed"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def output_key(model: str, *parts: bytes | str) -> str:
    h = hashlib.sha256()
    hub, rev = MODELS.get(model, (model, ""))
    h.update(f"{hub}@{rev}".encode())
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode())
    return h.hexdigest()


def cached(model: str, key: str, compute) -> dict[str, np.ndarray]:
    """Return the cached arrays for ``key`` or compute, store and return them."""
    path = CACHE_DIR / "model_outputs" / model / f"{key[:2]}" / f"{key}.npz"
    if _USE_CACHE and path.exists():
        _STATS["hit"] += 1
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    _STATS["miss"] += 1
    out = compute()
    if _USE_CACHE:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
    return out


def set_threads() -> None:
    try:
        import torch

        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
