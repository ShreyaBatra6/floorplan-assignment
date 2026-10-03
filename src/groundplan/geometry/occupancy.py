"""2D plan-view evidence grids: walls (by vertical span), observed floor, carved free space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Grid:
    origin: np.ndarray  # plan xy of the lower-left corner of cell (0, 0)
    res: float
    shape: tuple[int, int]  # (ny, nx)

    @staticmethod
    def around(xy: np.ndarray, res: float = 0.02, margin: float = 0.6) -> Grid:
        lo = xy.min(axis=0) - margin
        hi = xy.max(axis=0) + margin
        nx = int(np.ceil((hi[0] - lo[0]) / res))
        ny = int(np.ceil((hi[1] - lo[1]) / res))
        return Grid(lo.astype(float), res, (ny, nx))

    def index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ij = np.floor((xy - self.origin) / self.res).astype(np.int64)
        ix, iy = ij[:, 0], ij[:, 1]
        ok = (ix >= 0) & (iy >= 0) & (ix < self.shape[1]) & (iy < self.shape[0])
        return iy, ix, ok

    def centers(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        return np.stack([self.origin[0] + (ix + 0.5) * self.res, self.origin[1] + (iy + 0.5) * self.res], axis=-1)

    def count(self, xy: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        iy, ix, ok = self.index(xy)
        out = np.zeros(self.shape, np.float32)
        w = None if weights is None else weights[ok]
        np.add.at(out, (iy[ok], ix[ok]), 1.0 if w is None else w)
        return out


@dataclass
class WallEvidence:
    count: np.ndarray
    span: np.ndarray  # max - min height of vertical-surface points per cell (m)
    top: np.ndarray  # highest such point per cell (m above floor)


def wall_evidence(grid: Grid, xy: np.ndarray, h: np.ndarray) -> WallEvidence:
    iy, ix, ok = grid.index(xy)
    iy, ix, hh = iy[ok], ix[ok], h[ok]
    lin = iy * grid.shape[1] + ix
    size = grid.shape[0] * grid.shape[1]
    cnt = np.bincount(lin, minlength=size).astype(np.float32)
    hmax = np.full(size, -np.inf)
    hmin = np.full(size, np.inf)
    np.maximum.at(hmax, lin, hh)
    np.minimum.at(hmin, lin, hh)
    span = np.where(cnt > 0, hmax - hmin, 0.0)
    top = np.where(cnt > 0, hmax, 0.0)
    return WallEvidence(cnt.reshape(grid.shape), span.reshape(grid.shape).astype(np.float32),
                        top.reshape(grid.shape).astype(np.float32))


def carve_free_space(grid: Grid, starts: np.ndarray, ends: np.ndarray, stop_short: float = 0.06,
                     step: float | None = None, max_rays: int = 400_000, seed: int = 0) -> np.ndarray:
    """Count, per cell, how many observation rays pass through it (projected to the plan).

    A ray from the camera to a surface it observed proves the space in between is empty; the last
    ``stop_short`` metres are left alone so the observed surface itself is not carved away.
    """
    n = len(starts)
    if n == 0:
        return np.zeros(grid.shape, np.float32)
    if n > max_rays:
        idx = np.random.default_rng(seed).choice(n, max_rays, replace=False)
        starts, ends = starts[idx], ends[idx]
    step = step or grid.res * 0.9
    vec = ends - starts
    length = np.linalg.norm(vec, axis=1)
    usable = length > stop_short + step
    starts, vec, length = starts[usable], vec[usable], length[usable]
    out = np.zeros(grid.shape[0] * grid.shape[1], np.float32)
    # process in chunks to bound memory
    chunk = 20000
    for a in range(0, len(starts), chunk):
        s, v, L = starts[a : a + chunk], vec[a : a + chunk], length[a : a + chunk]
        nsteps = np.ceil((L - stop_short) / step).astype(int)
        total = int(nsteps.sum())
        if total == 0:
            continue
        ray_id = np.repeat(np.arange(len(s)), nsteps)
        offs = np.arange(total) - np.repeat(np.cumsum(nsteps) - nsteps, nsteps)
        t = (offs * step) / L[ray_id]
        pts = s[ray_id] + v[ray_id] * t[:, None]
        iy, ix, ok = grid.index(pts)
        lin = iy[ok] * grid.shape[1] + ix[ok]
        # count each ray at most once per cell
        key = np.unique(lin * len(s) + ray_id[ok])
        cells = key // len(s)
        out += np.bincount(cells, minlength=out.size).astype(np.float32)
    return out.reshape(grid.shape)


def fill_small_holes(mask: np.ndarray, max_area_cells: int) -> np.ndarray:
    lab, n = ndimage.label(~mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    small = np.zeros(n + 1, bool)
    small[1:] = sizes <= max_area_cells
    # never fill the region touching the border (outside)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small[border] = False
    return mask | small[lab]
