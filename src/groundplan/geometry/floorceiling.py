"""Floor and ceiling levels from the heights of horizontal surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Level:
    y: float  # world height of the surface
    support: int  # points supporting it
    spread: float  # robust std of their heights (m)
    area_m2: float  # observed area (10 cm cells)


def _peaks(h: np.ndarray, lo: float, hi: float, bin_m: float = 0.01) -> list[tuple[float, float]]:
    if hi <= lo or len(h) == 0:
        return []
    bins = max(int(np.ceil((hi - lo) / bin_m)), 1)
    hist, edges = np.histogram(h, bins=bins, range=(lo, hi))
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2)
    s = np.convolve(hist, k / k.sum(), mode="same")
    out = []
    for i in range(1, len(s) - 1):
        if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] > 0:
            out.append(((edges[i] + edges[i + 1]) / 2, float(s[i])))
    return out


def _level(h: np.ndarray, xz: np.ndarray, y0: float, band: float = 0.04) -> Level:
    sel = np.abs(h - y0) < band
    hs = h[sel]
    y = float(np.median(hs))
    spread = float(np.median(np.abs(hs - y)) * 1.4826)
    cells = np.unique(np.floor(xz[sel] / 0.1).astype(np.int64), axis=0)
    return Level(y=y, support=int(sel.sum()), spread=spread, area_m2=float(len(cells)) * 0.01)


def find_floor(xyz: np.ndarray, nrm: np.ndarray, cam_y: np.ndarray, min_rel: float = 0.25) -> Level | None:
    """Lowest strong up-facing horizontal surface below the cameras.

    The lowest peak holding at least ``min_rel`` of the strongest peak wins: furniture tops sit
    above the floor, and sparse virtual points from reflective floors sit below it.
    """
    up = nrm[:, 1] > 0.95
    h = xyz[up, 1]
    top = float(np.median(cam_y)) - 0.35
    peaks = [(y, s) for y, s in _peaks(h, float(h.min()) if len(h) else 0.0, top)] if len(h) else []
    if not peaks:
        return None
    smax = max(s for _, s in peaks)
    strong = [(y, s) for y, s in peaks if s >= min_rel * smax]
    y0 = min(strong)[0]
    return _level(h, xyz[up][:, [0, 2]], y0)


def find_ceiling(xyz: np.ndarray, nrm: np.ndarray, cam_y: np.ndarray, floor_y: float,
                 min_area_m2: float = 0.5) -> tuple[Level | None, list[Level]]:
    """Main ceiling (most observed area) plus secondary levels (bulkheads, soffits)."""
    down = nrm[:, 1] < -0.95
    h = xyz[down, 1]
    lo = max(float(np.median(cam_y)) + 0.2, floor_y + 1.9)
    if len(h) == 0 or h.max() <= lo:
        return None, []
    peaks = _peaks(h, lo, float(h.max()) + 0.02)
    levels = []
    for y, _ in sorted(peaks, key=lambda p: -p[1])[:6]:
        lev = _level(h, xyz[down][:, [0, 2]], y)
        if lev.area_m2 >= min_area_m2 and all(abs(lev.y - o.y) > 0.06 for o in levels):
            levels.append(lev)
    if not levels:
        return None, []
    levels.sort(key=lambda lv: -lv.area_m2)
    return levels[0], levels[1:]
