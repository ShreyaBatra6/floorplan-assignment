"""Metric monocular depth (photo and video tiers).

Backends are pip-installable through ``transformers`` so a clean install needs no compilers:

* ``da2-indoor-small``: Depth Anything V2 Metric-Indoor Small (Apache-2.0, 25 M parameters), the
  default: fast on a laptop CPU.
* ``da2-indoor-base``: the Base variant (CC-BY-NC-4.0), optional.

Images must be upright (the model has learned that floors are at the bottom). The model was
trained with a fixed field of view, so its metric scale is corrected for the actual focal length
by ``focal_correction`` (depth scales with focal length for a given apparent size); the residual
per-scene scale error is measured on the benchmark and carried as the tier's log-scale sigma.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from groundplan.models.registry import cached, output_key, set_threads, torch_available

BACKENDS = {
    "da2-indoor-small": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "main"),
    "da2-indoor-base": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", "main"),
}
DEFAULT_BACKEND = "da2-indoor-small"
TRAIN_HFOV_DEG = 60.0  # Hypersim-style training camera; see focal_correction()


def available() -> bool:
    return torch_available()


@lru_cache(maxsize=2)
def _load(backend: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    set_threads()
    hub, rev = BACKENDS[backend]
    proc = AutoImageProcessor.from_pretrained(hub, revision=rev)
    model = AutoModelForDepthEstimation.from_pretrained(hub, revision=rev).eval()
    return proc, model


def predict_depth(image_bgr: np.ndarray, backend: str = DEFAULT_BACKEND, max_side: int = 518) -> np.ndarray:
    """Metric depth (metres) at the input image's resolution, as predicted (no focal correction)."""
    h, w = image_bgr.shape[:2]
    s = max_side / max(h, w)
    small = cv2.resize(image_bgr, (max(int(round(w * s)), 14), max(int(round(h * s)), 14)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    key = output_key(f"depth_{backend}", rgb.tobytes(), str(rgb.shape))

    def compute():
        import torch

        proc, model = _load(backend)
        with torch.no_grad():
            inputs = proc(images=rgb, return_tensors="pt")
            out = model(**inputs).predicted_depth
            d = torch.nn.functional.interpolate(out[:, None], size=rgb.shape[:2], mode="bicubic",
                                                align_corners=False)[0, 0]
        return {"depth": d.numpy().astype(np.float32)}

    d = cached(f"depth_{backend}", key, compute)["depth"]
    return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)


def focal_correction(fx: float, width: int) -> float:
    """Multiplicative depth correction for a camera whose focal length differs from training.

    For the same apparent size an object is farther away when the focal length is longer, so
    depth predicted by a model that assumes the training field of view is rescaled by
    ``fx / fx_train``. Measured on the assessor captures (see technical report), not assumed.
    """
    fx_train = (width / 2) / np.tan(np.radians(TRAIN_HFOV_DEG) / 2)
    return float(fx / fx_train)
