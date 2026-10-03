"""Drift correction for multi-room LiDAR walks (pose graph over submaps). Implemented in a later step."""

from __future__ import annotations

import numpy as np


def correct_drift(cap, keyframes: np.ndarray, poses: np.ndarray, depth_scale: float = 1.0,
                  depth_offset: float = 0.0) -> tuple[np.ndarray, dict]:
    return poses, {"enabled": False, "method": "not yet implemented: ARKit poses used as-is"}
