"""Openings (doors, windows, open passages) from rays that pass through a wall plane.

For every wall face we rasterise a (s, h) grid in the wall's own coordinates (s along the wall from
its start corner, h above the floor):

* ``solid``: points on the face itself (within a few cm of the plane, normal facing the room);
* ``through``: observation rays that cross the plane inside the wall's extent and end well beyond
  it. A ray can only get through where there is no wall.

Cells with through-evidence and little solid evidence form opening candidates; the rectangle of
each component gives width, height and sill. A mirror also lets "rays through" (the depth sensor
measures the reflected room), so each candidate is tested for reflection consistency: if the
points behind the plane, reflected back across it, land on surfaces observed inside the room, the
candidate is a mirror, not an opening.

Width precision here is bounded by the 256x192 depth resolution and the ray density; the reported
interval says so (see the error budget in the technical report).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class OpeningCandidate:
    s0: float
    s1: float
    h0: float
    h1: float
    kind: str  # door | window | open_passage
    width: float
    width_sigma: float
    height: float
    height_sigma: float
    sill: float | None
    through_rays: int
    solid_fraction: float
    confidence: float
    beyond_xy: np.ndarray  # plan points seen through the opening (to find the room behind)
    mirror: bool = False


@dataclass
class OpeningParams:
    res: float = 0.02
    solid_band: float = 0.04
    beyond_min: float = 0.15
    min_width: float = 0.45
    min_height: float = 0.45
    max_width: float = 3.0
    open_ratio: float = 0.65
    door_bottom_max: float = 0.12
    door_top_min: float = 1.6
    window_bottom_min: float = 0.2
    passage_width_min: float = 1.25
    mirror_nn_dist: float = 0.04
    mirror_fraction: float = 0.6


def wall_frame(axis: int, offset: float, inward: int, start: np.ndarray, end: np.ndarray):
    """Unit along-wall vector, the start corner, and wall length."""
    d = end - start
    length = float(np.linalg.norm(d))
    return d / max(length, 1e-9), start, length


def detect_openings(
    axis: int,
    offset: float,
    inward: int,
    start: np.ndarray,
    end: np.ndarray,
    wall_height: float,
    solid_xy: np.ndarray,
    solid_h: np.ndarray,
    ray_start_xy: np.ndarray,
    ray_start_h: np.ndarray,
    ray_end_xy: np.ndarray,
    ray_end_h: np.ndarray,
    room_surface_pts: np.ndarray | None,
    p: OpeningParams | None = None,
) -> list[OpeningCandidate]:
    p = p or OpeningParams()
    along, origin, length = wall_frame(axis, offset, inward, start, end)
    if length < p.min_width:
        return []
    ns = int(np.ceil(length / p.res))
    nh = int(np.ceil(wall_height / p.res))
    if ns < 2 or nh < 2:
        return []

    # solid hits on the face
    s_sol = (solid_xy - origin) @ along
    ok = (s_sol >= 0) & (s_sol < length) & (solid_h >= 0) & (solid_h < wall_height)
    solid = np.zeros((nh, ns), np.float32)
    np.add.at(solid, ((solid_h[ok] / p.res).astype(int), (s_sol[ok] / p.res).astype(int)), 1)

    # rays crossing the plane from inside to beyond
    a0 = ray_start_xy[:, axis] - offset
    a1 = ray_end_xy[:, axis] - offset
    inside0 = a0 * inward > 0.02
    beyond1 = a1 * inward < -p.beyond_min
    cand = inside0 & beyond1
    t = np.zeros(len(a0))
    t[cand] = a0[cand] / (a0[cand] - a1[cand])
    cross_xy = ray_start_xy + (ray_end_xy - ray_start_xy) * t[:, None]
    cross_h = ray_start_h + (ray_end_h - ray_start_h) * t
    s_cr = (cross_xy - origin) @ along
    cand &= (s_cr >= 0) & (s_cr < length) & (cross_h >= 0) & (cross_h < wall_height)
    through = np.zeros((nh, ns), np.float32)
    np.add.at(through, ((cross_h[cand] / p.res).astype(int), (s_cr[cand] / p.res).astype(int)), 1)

    # smooth a little: rays are sparse
    th = ndimage.uniform_filter(through, size=3) * 9
    so = ndimage.uniform_filter(solid, size=3) * 9
    ratio = th / (th + so + 1e-6)
    open_mask = (ratio >= p.open_ratio) & (th >= 1.0)
    open_mask = ndimage.binary_closing(open_mask, structure=np.ones((5, 5)), iterations=2)
    open_mask = ndimage.binary_opening(open_mask, structure=np.ones((3, 3)))
    lab, n = ndimage.label(open_mask)
    out: list[OpeningCandidate] = []
    beyond_idx = np.flatnonzero(cand)
    for k in range(1, n + 1):
        comp = lab == k
        rows = np.flatnonzero(comp.any(axis=1))
        cols = np.flatnonzero(comp.any(axis=0))
        h0, h1 = rows[0] * p.res, (rows[-1] + 1) * p.res
        if h1 - h0 < p.min_height:
            continue
        # width from the middle 60% of the opening's rows (jambs are cleanest there)
        r_lo = rows[0] + int(0.2 * len(rows))
        r_hi = rows[0] + max(int(0.8 * len(rows)), 1)
        lefts, rights = [], []
        for r in range(r_lo, r_hi):
            c = np.flatnonzero(comp[r])
            if len(c):
                lefts.append(c[0])
                rights.append(c[-1] + 1)
        if not lefts:
            continue
        s0 = float(np.median(lefts)) * p.res
        s1 = float(np.median(rights)) * p.res
        width = s1 - s0
        if width < p.min_width or width > p.max_width:
            continue
        jitter = (np.std(lefts) + np.std(rights)) * p.res / 2
        width_sigma = float(np.hypot(p.res, jitter) + 0.01)
        bottom_on_floor = h0 <= p.door_bottom_max
        if bottom_on_floor and (width >= p.passage_width_min or h1 >= wall_height - 0.05):
            kind = "open_passage"
        elif bottom_on_floor and h1 >= p.door_top_min:
            kind = "door"
        elif h0 >= p.window_bottom_min:
            kind = "window"
        else:
            continue
        height = (h1 - (0.0 if bottom_on_floor else h0))
        n_through = int(through[comp].sum())
        solid_frac = float((solid[comp] > 0).mean())
        sel = beyond_idx[(s_cr[beyond_idx] >= s0) & (s_cr[beyond_idx] < s1) &
                         (cross_h[beyond_idx] >= h0) & (cross_h[beyond_idx] < h1)]
        cand_out = OpeningCandidate(
            s0=s0, s1=s1, h0=0.0 if bottom_on_floor else h0, h1=h1, kind=kind,
            width=width, width_sigma=width_sigma, height=height, height_sigma=float(p.res + 0.015),
            sill=None if bottom_on_floor else h0, through_rays=n_through, solid_fraction=solid_frac,
            confidence=float(np.clip(n_through / 200.0, 0.2, 1.0) * (1 - solid_frac)),
            beyond_xy=ray_end_xy[sel],
        )
        if room_surface_pts is not None and len(sel) >= 20:
            cand_out.mirror = _looks_like_mirror(ray_end_xy[sel], ray_end_h[sel], axis, offset,
                                                 room_surface_pts, p)
        out.append(cand_out)
    return out


def _looks_like_mirror(beyond_xy: np.ndarray, beyond_h: np.ndarray, axis: int, offset: float,
                       room_pts: np.ndarray, p: OpeningParams) -> bool:
    """Reflect the points seen 'through' the plane back across it; a mirror maps them onto the room."""
    refl = beyond_xy.copy()
    refl[:, axis] = 2 * offset - refl[:, axis]
    q = np.column_stack([refl, beyond_h])
    tree = cKDTree(room_pts)
    d, _ = tree.query(q, k=1, distance_upper_bound=0.5)
    return float(np.mean(d < p.mirror_nn_dist)) >= p.mirror_fraction
