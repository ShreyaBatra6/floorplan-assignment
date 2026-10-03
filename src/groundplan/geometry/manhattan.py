"""Dominant wall orientation (Manhattan alignment) in the floor plane."""

from __future__ import annotations

import numpy as np


def dominant_yaw(normals_xz: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Yaw (radians, in [-pi/4, pi/4)) of the dominant pair of orthogonal wall directions.

    ``normals_xz`` are horizontal normal components (plan x, plan y) of vertical surfaces. Angles
    are folded modulo 90 degrees with the 4-theta trick, so opposite and perpendicular walls all
    vote for the same yaw. Returns (yaw, strength) where strength in [0, 1] is the resultant length
    (1 = perfectly Manhattan).
    """
    if len(normals_xz) == 0:
        return 0.0, 0.0
    theta = np.arctan2(normals_xz[:, 1], normals_xz[:, 0])
    w = np.ones(len(theta)) if weights is None else np.asarray(weights, float)
    # Coarse histogram first (robust to clutter), then a weighted circular mean near the peak.
    folded = np.mod(theta, np.pi / 2)
    hist, edges = np.histogram(folded, bins=180, range=(0, np.pi / 2), weights=w)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    smooth = np.convolve(np.concatenate([hist[-6:], hist, hist[:6]]), kernel, mode="same")[6:-6]
    peak = (edges[np.argmax(smooth)] + edges[np.argmax(smooth) + 1]) / 2
    near = np.abs(_wrap_quarter(folded - peak)) < np.radians(8)
    z = np.sum(w[near] * np.exp(4j * theta[near]))
    yaw = np.angle(z) / 4.0
    strength = float(np.abs(np.sum(w * np.exp(4j * theta))) / max(np.sum(w), 1e-9))
    yaw = float(_wrap_quarter(yaw))
    return yaw, strength


def _wrap_quarter(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles into [-pi/4, pi/4)."""
    return (np.asarray(a) + np.pi / 4) % (np.pi / 2) - np.pi / 4


def snap_direction(angle: float, yaw: float, tol_deg: float = 12.0) -> tuple[float, bool]:
    """Snap a line direction to the nearest Manhattan axis if within ``tol_deg``."""
    rel = _wrap_quarter(angle - yaw)
    if abs(rel) <= np.radians(tol_deg):
        return float(angle - rel), True
    return float(angle), False
