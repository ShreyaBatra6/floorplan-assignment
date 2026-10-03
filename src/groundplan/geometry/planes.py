"""Robust plane and line fitting with honest uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fit1D:
    """Robust location estimate of a 1D sample (e.g. wall offsets along its normal)."""

    value: float
    sigma: float  # standard error of ``value``
    spread: float  # robust standard deviation of the samples
    n: int
    n_eff: float


def robust_location(x: np.ndarray, weights: np.ndarray | None = None, groups: np.ndarray | None = None,
                    c: float = 2.5, iters: int = 10) -> Fit1D:
    """Huber M-estimate of location with a standard error that respects correlated samples.

    Points from one depth frame are not independent (they share that frame's pose and depth bias),
    so the effective sample size is the number of distinct ``groups`` (frames), not the number of
    points. Without groups every sample counts.
    """
    x = np.asarray(x, float)
    if len(x) == 0:
        return Fit1D(float("nan"), float("inf"), float("inf"), 0, 0.0)
    w0 = np.ones_like(x) if weights is None else np.asarray(weights, float)
    mu = float(np.median(x))
    mad = float(np.median(np.abs(x - mu))) * 1.4826 + 1e-6
    for _ in range(iters):
        r = (x - mu) / (c * mad)
        w = w0 * np.where(np.abs(r) <= 1, 1.0, 1.0 / np.maximum(np.abs(r), 1e-9))
        mu_new = float(np.sum(w * x) / np.sum(w))
        if abs(mu_new - mu) < 1e-7:
            mu = mu_new
            break
        mu = mu_new
    resid = x - mu
    spread = float(np.median(np.abs(resid))) * 1.4826 + 1e-6
    if groups is not None and len(groups) == len(x):
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        # per-group means: their scatter captures pose/depth-bias variation between frames
        sums = np.bincount(inv, weights=x)
        cnt = np.bincount(inv)
        gm = sums / np.maximum(cnt, 1)
        n_eff = float(len(uniq))
        if len(uniq) >= 3:
            gspread = float(np.median(np.abs(gm - np.median(gm)))) * 1.4826 + 1e-6
            sigma = max(gspread / np.sqrt(n_eff), spread / np.sqrt(len(x)))
        else:
            sigma = spread / np.sqrt(max(n_eff, 1.0))
    else:
        n_eff = float(len(x))
        sigma = spread / np.sqrt(n_eff)
    return Fit1D(mu, float(sigma), spread, len(x), n_eff)


@dataclass
class Plane:
    normal: np.ndarray  # unit
    d: float  # plane: normal . x + d = 0
    rms: float
    n: int

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d


def fit_plane(pts: np.ndarray, weights: np.ndarray | None = None) -> Plane:
    w = np.ones(len(pts)) if weights is None else weights
    c = np.average(pts, axis=0, weights=w)
    X = (pts - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[-1]
    d = -float(n @ c)
    rms = float(np.sqrt(np.average((pts @ n + d) ** 2, weights=w)))
    return Plane(n, d, rms, len(pts))


def ransac_plane(pts: np.ndarray, thresh: float = 0.02, iters: int = 300, seed: int = 0,
                 axis: np.ndarray | None = None, max_angle_deg: float = 10.0) -> tuple[Plane | None, np.ndarray]:
    """RANSAC plane, optionally constrained to normals within ``max_angle_deg`` of ``axis``."""
    n = len(pts)
    if n < 3:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    best_cnt = 0
    cos_lim = np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = pts[idx]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm /= ln
        if axis is not None and abs(nrm @ axis) < cos_lim:
            continue
        d = -nrm @ a
        inl = np.abs(pts @ nrm + d) < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_cnt < 3:
        return None, best_inl
    plane = fit_plane(pts[best_inl])
    inl = np.abs(plane.distance(pts)) < thresh
    return fit_plane(pts[inl]), inl
