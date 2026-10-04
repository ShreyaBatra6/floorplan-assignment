"""Orthomosaics of every surface: wall elevations and floor/ceiling plans at a fixed metric scale.

Damage is found where it lives: on a surface. Each wall (and the floor and ceiling of each room)
is rasterised in its own metric (u, v) coordinates, and every raster pixel takes its colour from
the views that see it best:

* only views in front of the surface, inside the image, closer than ``max_dist`` and not grazing;
* occlusion is tested against the view's depth map, so furniture in front of a wall is not painted
  onto it;
* the per-pixel median over the best ``k`` views removes specular highlights and moving shadows
  that a single photo would show as false "stains".

A region found on the mosaic therefore has its metric extent directly (pixel = ``res`` metres).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class View:
    image: np.ndarray  # BGR uint8
    K: np.ndarray  # intrinsics for ``image``
    T_wc: np.ndarray  # OpenCV camera -> world
    depth: np.ndarray | None = None  # metres, any resolution
    K_depth: np.ndarray | None = None
    key: str = ""

    @property
    def center(self) -> np.ndarray:
        return self.T_wc[:3, 3]


@dataclass
class SurfaceGeom:
    id: str
    room_id: str
    kind: str  # wall | floor | ceiling
    origin: np.ndarray  # world point at (u, v) = (u0, v0)
    u_axis: np.ndarray  # world unit vector of +u
    v_axis: np.ndarray  # world unit vector of +v
    normal: np.ndarray  # world unit normal pointing into the room
    u0: float
    u1: float
    v0: float
    v1: float
    exclude: list[tuple[float, float, float, float]] = field(default_factory=list)  # (u0, u1, v0, v1) openings
    polygon_uv: np.ndarray | None = None  # floor/ceiling outline in (u, v)

    def grid(self, res: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points of the raster (rows from v1 down to v0, so walls read upright) and (u, v)."""
        nu = max(int(np.ceil((self.u1 - self.u0) / res)), 1)
        nv = max(int(np.ceil((self.v1 - self.v0) / res)), 1)
        u = self.u0 + (np.arange(nu) + 0.5) * res
        v = self.v1 - (np.arange(nv) + 0.5) * res
        uu, vv = np.meshgrid(u, v)
        pts = self.origin + (uu[..., None] - self.u0) * self.u_axis + (vv[..., None] - self.v0) * self.v_axis
        return pts, uu, vv


@dataclass
class Mosaic:
    surface: SurfaceGeom
    image: np.ndarray  # BGR uint8 (nv, nu, 3)
    valid: np.ndarray  # bool: observed and not excluded
    views: np.ndarray  # int: number of views that contributed
    protrusion: np.ndarray | None  # metres the measured surface sits in front of (+) / behind (-) the plane
    res: float
    uu: np.ndarray
    vv: np.ndarray


def _project(K: np.ndarray, T_wc: np.ndarray, pts: np.ndarray):
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    cam = (pts - t) @ R
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[..., 0] / z + K[0, 2]
        v = K[1, 1] * cam[..., 1] / z + K[1, 2]
    return u, v, z


def select_views(surface: SurfaceGeom, views: list[View], max_views: int = 12, max_dist: float = 4.0,
                 min_cos: float = 0.3) -> list[int]:
    """Views that face the surface, ranked by how squarely and closely they see its centre."""
    uc, vc = (surface.u0 + surface.u1) / 2, (surface.v0 + surface.v1) / 2
    corners = [(surface.u0, surface.v0), (surface.u1, surface.v0), (surface.u0, surface.v1), (surface.u1, surface.v1),
               (uc, vc)]
    pts = np.array([surface.origin + (a - surface.u0) * surface.u_axis + (b - surface.v0) * surface.v_axis
                    for a, b in corners])
    scored = []
    for k, view in enumerate(views):
        d = view.center - pts[-1]
        dist = np.linalg.norm(d)
        cos = float(d @ surface.normal / max(dist, 1e-6))
        if cos < min_cos or dist > max_dist * 1.6:
            continue
        u, v, z = _project(view.K, view.T_wc, pts)
        h, w = view.image.shape[:2]
        inside = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if inside.sum() == 0:
            continue
        scored.append((cos / max(dist, 0.5) * (0.5 + inside.mean()), k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:max_views]]


def build_mosaic(surface: SurfaceGeom, views: list[View], res: float = 0.01, top_k: int = 3, max_dist: float = 4.0,
                 min_cos: float = 0.3, occlusion_margin: float = 0.06, max_views: int = 12) -> Mosaic | None:
    pts, uu, vv = surface.grid(res)
    nv, nu = uu.shape
    if nu * nv > 1_500_000:  # very large surfaces: coarsen to keep memory bounded
        return build_mosaic(surface, views, res * 1.5, top_k, max_dist, min_cos, occlusion_margin, max_views)
    chosen = select_views(surface, views, max_views, max_dist, min_cos)
    if not chosen:
        return None
    flat = pts.reshape(-1, 3)
    n = len(flat)
    scores = np.full((len(chosen), n), -1.0, np.float32)
    colors = np.zeros((len(chosen), n, 3), np.uint8)
    offsets = np.full((len(chosen), n), np.nan, np.float32)
    for j, k in enumerate(chosen):
        view = views[k]
        h, w = view.image.shape[:2]
        u, v, z = _project(view.K, view.T_wc, flat)
        ok = (z > 0.2) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        d = view.center[None, :] - flat
        dist = np.linalg.norm(d, axis=1)
        cos = (d @ surface.normal) / np.maximum(dist, 1e-6)
        ok &= (cos >= min_cos) & (dist <= max_dist)
        if view.depth is not None and view.K_depth is not None:
            ud, vd, zd = _project(view.K_depth, view.T_wc, flat)
            dh, dw = view.depth.shape
            iu = np.clip(np.round(ud).astype(int), 0, dw - 1)
            iv = np.clip(np.round(vd).astype(int), 0, dh - 1)
            obs = view.depth[iv, iu]
            # something measured clearly in front of the surface point: occluded
            ok &= ~((obs > 0) & (obs < zd - occlusion_margin))
            near = ok & (obs > 0) & (np.abs(zd - obs) < occlusion_margin)
            offsets[j, near] = (zd - obs)[near]
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        px = cv2.remap(view.image, u[idx].astype(np.float32).reshape(-1, 1), v[idx].astype(np.float32).reshape(-1, 1),
                       interpolation=cv2.INTER_LINEAR)
        colors[j, idx] = px.reshape(-1, 3)
        scores[j, idx] = (cos[idx] / np.maximum(dist[idx], 0.5)).astype(np.float32)
    count = (scores > 0).sum(axis=0)
    k_eff = min(top_k, len(chosen))
    top = np.argsort(-scores, axis=0)[:k_eff]  # (k, n)
    gathered = np.take_along_axis(colors, top[..., None], axis=0).astype(np.float32)
    good = np.take_along_axis(scores, top, axis=0) > 0
    gathered[~good] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(gathered, axis=0)
    off_top = np.take_along_axis(offsets, top, axis=0)
    off_top[~good] = np.nan
    with np.errstate(all="ignore"):
        protrusion = np.nanmedian(off_top, axis=0)
    valid = count > 0
    med = np.where(valid[:, None], med, 0)
    image = np.clip(med, 0, 255).astype(np.uint8).reshape(nv, nu, 3)
    valid = valid.reshape(nv, nu)
    for (a0, a1, b0, b1) in surface.exclude:
        valid &= ~((uu >= a0) & (uu <= a1) & (vv >= b0) & (vv <= b1))
    if surface.polygon_uv is not None:
        from matplotlib.path import Path as MplPath

        inside = MplPath(surface.polygon_uv).contains_points(np.column_stack([uu.ravel(), vv.ravel()]))
        valid &= inside.reshape(nv, nu)
    prot = protrusion.reshape(nv, nu) if np.isfinite(protrusion).any() else None
    return Mosaic(surface, image, valid, count.reshape(nv, nu), prot, res, uu, vv)
