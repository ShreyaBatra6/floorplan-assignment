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


def _sample_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear colour lookup for many points (remap's maps must stay below 32767 per side)."""
    n = len(u)
    cols = 1024
    rows = int(np.ceil(n / cols))
    mu = np.zeros(rows * cols, np.float32)
    mv = np.zeros(rows * cols, np.float32)
    mu[:n], mv[:n] = u, v
    out = cv2.remap(image, mu.reshape(rows, cols), mv.reshape(rows, cols), interpolation=cv2.INTER_LINEAR)
    return out.reshape(-1, 3)[:n]


def select_views(surface: SurfaceGeom, views: list[View], max_views: int = 16, max_dist: float = 4.0,
                 min_cos: float = 0.3, target_cover: int = 3, step: float = 0.2) -> list[int]:
    """Views chosen greedily to see every part of the surface ``target_cover`` times, best-facing first."""
    nu = max(int(np.ceil((surface.u1 - surface.u0) / step)), 1)
    nv = max(int(np.ceil((surface.v1 - surface.v0) / step)), 1)
    us = surface.u0 + (np.arange(nu) + 0.5) * (surface.u1 - surface.u0) / nu
    vs = surface.v0 + (np.arange(nv) + 0.5) * (surface.v1 - surface.v0) / nv
    uu, vv = np.meshgrid(us, vs)
    pts = (surface.origin + (uu.ravel()[:, None] - surface.u0) * surface.u_axis
           + (vv.ravel()[:, None] - surface.v0) * surface.v_axis)
    if surface.polygon_uv is not None:
        from matplotlib.path import Path as MplPath

        keep = MplPath(surface.polygon_uv).contains_points(np.column_stack([uu.ravel(), vv.ravel()]))
        pts = pts[keep] if keep.any() else pts
    sees, quality = [], []
    for view in views:
        u, v, z = _project(view.K, view.T_wc, pts)
        h, w = view.image.shape[:2]
        d = view.center - pts
        dist = np.linalg.norm(d, axis=1)
        cos = (d @ surface.normal) / np.maximum(dist, 1e-6)
        ok = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (cos >= min_cos) & (dist <= max_dist)
        if view.depth is not None and view.K_depth is not None:
            ud, vd, zd = _project(view.K_depth, view.T_wc, pts)
            dh, dw = view.depth.shape
            iu = np.clip(np.round(np.nan_to_num(ud)).astype(int), 0, dw - 1)
            iv = np.clip(np.round(np.nan_to_num(vd)).astype(int), 0, dh - 1)
            obs = view.depth[iv, iu]
            ok &= ~((obs > 0) & (obs < zd - 0.1))
        sees.append(ok)
        quality.append(np.where(ok, cos / np.maximum(dist, 0.5), 0.0))
    if not sees:
        return []
    sees = np.array(sees)
    quality = np.array(quality)
    cover = np.zeros(len(pts), int)
    chosen: list[int] = []
    for _ in range(max_views):
        need = cover < target_cover
        if not need.any():
            break
        gain = (quality * need[None, :]).sum(axis=1)
        gain[chosen] = -1
        k = int(np.argmax(gain))
        if gain[k] <= 0:
            break
        chosen.append(k)
        cover += sees[k]
    return chosen


def build_mosaic(surface: SurfaceGeom, views: list[View], res: float = 0.01, top_k: int = 3, max_dist: float = 4.0,
                 min_cos: float = 0.3, occlusion_margin: float = 0.06, max_views: int = 16) -> Mosaic | None:
    pts, uu, vv = surface.grid(res)
    nv, nu = uu.shape
    if nu * nv > 1_500_000:  # very large surfaces: coarsen to keep memory bounded
        return build_mosaic(surface, views, res * 1.5, top_k, max_dist, min_cos, occlusion_margin, max_views)
    chosen = select_views(surface, views, max_views, max_dist, min_cos)
    if not chosen:
        return None
    flat = pts.reshape(-1, 3).astype(np.float32)
    n = len(flat)
    k_eff = min(top_k, len(chosen))
    # running top-k per pixel: memory O(k * pixels) however many views are considered
    best_s = np.full((k_eff, n), -1.0, np.float32)
    best_c = np.zeros((k_eff, n, 3), np.uint8)
    best_o = np.full((k_eff, n), np.nan, np.float32)
    count = np.zeros(n, np.int16)
    for k in chosen:
        view = views[k]
        h, w = view.image.shape[:2]
        u, v, z = _project(view.K, view.T_wc, flat)
        ok = (z > 0.2) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        d = view.center[None, :].astype(np.float32) - flat
        dist = np.linalg.norm(d, axis=1)
        cos = (d @ surface.normal.astype(np.float32)) / np.maximum(dist, 1e-6)
        del d
        ok &= (cos >= min_cos) & (dist <= max_dist)
        off = None
        if view.depth is not None and view.K_depth is not None:
            ud, vd, zd = _project(view.K_depth, view.T_wc, flat)
            dh, dw = view.depth.shape
            iu = np.clip(np.round(ud).astype(np.int32), 0, dw - 1)
            iv = np.clip(np.round(vd).astype(np.int32), 0, dh - 1)
            obs = view.depth[iv, iu]
            ok &= ~((obs > 0) & (obs < zd - occlusion_margin))  # something clearly in front: occluded
            off = np.where(ok & (obs > 0) & (np.abs(zd - obs) < occlusion_margin), zd - obs, np.nan)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        count[idx] += 1
        sc = (cos[idx] / np.maximum(dist[idx], 0.5)).astype(np.float32)
        col = _sample_bilinear(view.image, u[idx], v[idx])
        of = off[idx].astype(np.float32) if off is not None else np.full(len(idx), np.nan, np.float32)
        # insert into the running top-k (slot k_eff-1 is the weakest)
        for slot in range(k_eff):
            better = sc > best_s[slot, idx]
            if not better.any():
                continue
            sel = idx[better]
            for lower in range(k_eff - 1, slot, -1):
                best_s[lower, sel] = best_s[lower - 1, sel]
                best_c[lower, sel] = best_c[lower - 1, sel]
                best_o[lower, sel] = best_o[lower - 1, sel]
            best_s[slot, sel] = sc[better]
            best_c[slot, sel] = col[better]
            best_o[slot, sel] = of[better]
            keep = ~better
            idx, sc, col, of = idx[keep], sc[keep], col[keep], of[keep]
            if len(idx) == 0:
                break
    good = best_s > 0
    gathered = best_c.astype(np.float32)
    gathered[~good] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(gathered, axis=0)
        best_o[~good] = np.nan
        protrusion = np.nanmedian(best_o, axis=0)
    del gathered
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
