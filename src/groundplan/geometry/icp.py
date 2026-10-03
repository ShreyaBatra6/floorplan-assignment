"""Point-to-plane ICP restricted to 4 degrees of freedom (x, y, z, yaw about gravity).

ARKit observes gravity directly, so roll and pitch do not drift measurably; restricting the
alignment to yaw + translation removes two poorly constrained unknowns and makes loop closures
between partially overlapping submaps far more reliable. The solver reports how well each
direction is constrained, so a closure that only sees a floor and one wall (unconstrained along
the wall) can be rejected instead of silently sliding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ICPResult:
    yaw: float  # radians, about the world y axis through ``center``
    t: np.ndarray  # (3,) translation
    center: np.ndarray  # (3,) rotation centre
    fitness: float  # fraction of source points with a close target match
    rmse: float  # point-to-plane RMS of inliers (m)
    min_eig: float  # smallest eigenvalue of the normal matrix (per point): degeneracy measure
    n_inliers: int

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return yaw_rotate(pts - self.center, self.yaw) + self.center + self.t


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def yaw_rotate(pts: np.ndarray, yaw: float) -> np.ndarray:
    return pts @ yaw_matrix(yaw).T


def icp_4dof(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray, src_normals: np.ndarray | None = None,
             max_dist: tuple[float, ...] = (0.10, 0.06, 0.04, 0.03), iters_per_level: int = 6,
             inlier_dist: float = 0.03, max_points: int = 4000, seed: int = 0) -> ICPResult:
    rng = np.random.default_rng(seed)
    if len(src) > max_points:
        keep = rng.choice(len(src), max_points, replace=False)
        src = src[keep]
        if src_normals is not None:
            src_normals = src_normals[keep]
    center = src.mean(axis=0)
    tree = cKDTree(dst)
    yaw, t = 0.0, np.zeros(3)
    yhat = np.array([0.0, 1.0, 0.0])
    A = np.zeros((0, 4))
    for md in max_dist:
        for _ in range(iters_per_level):
            p = yaw_rotate(src - center, yaw) + center + t
            d, idx = tree.query(p, k=1, distance_upper_bound=md)
            ok = np.isfinite(d)
            if src_normals is not None:
                ns = yaw_rotate(src_normals, yaw)
                ok &= np.einsum("ij,ij->i", ns, dst_normals[np.minimum(idx, len(dst) - 1)]) > 0.8
            if ok.sum() < 30:
                break
            q = dst[idx[ok]]
            n = dst_normals[idx[ok]]
            pp = p[ok]
            r = np.einsum("ij,ij->i", n, pp - q)
            lever = np.cross(yhat, pp - center)  # d(R p)/d(yaw) at yaw increment 0
            A = np.column_stack([np.einsum("ij,ij->i", n, lever), n])
            # Huber weights
            s = 1.4826 * np.median(np.abs(r)) + 1e-4
            w = np.where(np.abs(r) <= 1.5 * s, 1.0, 1.5 * s / np.abs(r))
            H = (A * w[:, None]).T @ A
            g = (A * w[:, None]).T @ r
            try:
                dx = -np.linalg.solve(H + 1e-9 * np.eye(4), g)
            except np.linalg.LinAlgError:
                break
            yaw += dx[0]
            # rotate the existing translation consistently with the incremental rotation about centre
            t = yaw_rotate(t, dx[0]) + dx[1:]
            if np.abs(dx[0]) < 1e-6 and np.linalg.norm(dx[1:]) < 1e-5:
                break
    p = yaw_rotate(src - center, yaw) + center + t
    d, idx = tree.query(p, k=1, distance_upper_bound=inlier_dist)
    ok = np.isfinite(d)
    if ok.sum() >= 10:
        n = dst_normals[idx[ok]]
        r = np.einsum("ij,ij->i", n, p[ok] - dst[idx[ok]])
        rmse = float(np.sqrt(np.mean(r**2)))
        lever = np.cross(yhat, p[ok] - center)
        A = np.column_stack([np.einsum("ij,ij->i", n, lever), n])
        # scale the yaw column by 1 m lever so eigenvalues compare like translations
        H = A.T @ A / ok.sum()
        min_eig = float(np.linalg.eigvalsh(H)[0])
    else:
        rmse, min_eig = float("inf"), 0.0
    return ICPResult(float(yaw), t, center, float(ok.mean()), rmse, min_eig, int(ok.sum()))
