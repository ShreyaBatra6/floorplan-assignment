"""Openings (doors, windows, open passages) as holes in a wall face, confirmed by rays through it.

For every wall face we work in the wall's own coordinates (``s`` along the wall from its start
corner, ``h`` above the floor):

* **solid**: points on the face itself (within a few cm of the plane, normal facing the room);
* **through**: observation rays that cross the plane inside the wall's extent and end well beyond
  it (doorways), plus **miss** rays: pixels that returned no depth although the plane was well
  within sensor range (glass, open windows; a plastered wall always returns).

A candidate is a region with through-evidence and no solid points. Its rectangle is then located
on the *solid* evidence, which is far sharper than the ray evidence: walking outward from the
candidate's centre along a solid-occupancy profile, the jamb is where the wall face resumes, the
head is where it resumes above, and the sill is where it resumes below (no wall below means a door
or passage). Each candidate is finally tested for reflection consistency: a mirror "lets rays
through" too, but the points behind it, reflected back across the plane, land on surfaces observed
inside the room.

Precision is bounded by the depth map: 256x192 depth with 2 cm voxels localises a jamb edge to
about a centimetre in good conditions, worse at grazing angles and near trim; the interval says so.
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
    min_width: float = 0.40
    min_height: float = 0.40
    max_width: float = 3.2
    min_through: int = 25
    profile_bin: float = 0.01
    profile_solid: float = 0.35  # fraction of rows (or columns) with a solid point -> wall resumes
    door_bottom_max: float = 0.15
    door_top_min: float = 1.6
    passage_width_min: float = 1.25
    miss_max_range: float = 3.2  # a plane closer than this must have returned depth if it were solid
    mirror_nn_dist: float = 0.04
    mirror_fraction: float = 0.6


def _profile_edge(coords: np.ndarray, start: float, stop: float, n_lines: int, p: OpeningParams) -> float | None:
    """Walk from ``start`` towards ``stop`` and return where the solid profile first exceeds the threshold."""
    if n_lines <= 0:
        return None
    step = p.profile_bin if stop > start else -p.profile_bin
    lo, hi = sorted((start, stop))
    sel = coords[(coords >= lo) & (coords <= hi)]
    if len(sel) == 0:
        return None
    bins = np.arange(lo, hi + p.profile_bin, p.profile_bin)
    hist, _ = np.histogram(sel, bins=bins)
    frac = np.minimum(hist / n_lines, 1.0)
    order = range(len(frac)) if step > 0 else range(len(frac) - 1, -1, -1)
    run = 0
    for i in order:
        if frac[i] >= p.profile_solid:
            run += 1
            if run >= 2:  # two consecutive solid bins: the face really resumes
                j = i - 1 if step > 0 else i + 1
                return float(bins[j] if step > 0 else bins[j + 1])
        else:
            run = 0
    return None


def _profile_last_open(coords: np.ndarray, start: float, stop: float, n_lines: int, p: OpeningParams,
                       thresh: float = 0.12, max_gap: float = 0.04) -> tuple[float | None, float]:
    """Walk outward from ``start`` along the open-evidence profile; return the outer edge of the last
    bin that is still open (gaps up to ``max_gap`` tolerated) and the open fraction near that edge."""
    if n_lines <= 0 or len(coords) == 0:
        return None, 0.0
    lo, hi = sorted((start, stop))
    bins = np.arange(lo, hi + p.profile_bin, p.profile_bin)
    hist, _ = np.histogram(coords[(coords >= lo) & (coords <= hi)], bins=bins)
    frac = np.minimum(hist / n_lines, 1.0)
    order = list(range(len(frac))) if stop > start else list(range(len(frac) - 1, -1, -1))
    last, gap_bins, edge_frac = None, 0, 0.0
    max_gap_bins = int(round(max_gap / p.profile_bin))
    for i in order:
        if frac[i] >= thresh:
            last, gap_bins = i, 0
            edge_frac = float(np.mean(frac[max(i - 2, 0): i + 3]))
        else:
            gap_bins += 1
            if last is not None and gap_bins > max_gap_bins:
                break
    if last is None:
        return None, 0.0
    return (float(bins[last + 1]) if stop > start else float(bins[last])), edge_frac


def _locate_edge(open_coords, solid_coords, start, stop, n_lines, p) -> tuple[float | None, float]:
    """Jamb (or head/sill) position between certain-open and certain-solid evidence, with its sigma."""
    open_edge, density = _profile_last_open(open_coords, start, stop, n_lines, p)
    solid_edge = _profile_edge(solid_coords, start, stop, n_lines, p)
    direction = 1.0 if stop > start else -1.0
    if open_edge is None and solid_edge is None:
        return None, 0.05
    if open_edge is None:
        return solid_edge, 0.02
    if solid_edge is None or (solid_edge - open_edge) * direction < -0.01:
        return open_edge, 0.03 if density < 0.4 else 0.012
    gap = abs(solid_edge - open_edge)
    if gap <= 0.05:
        return (open_edge + solid_edge) / 2, max(gap / np.sqrt(12), 0.004)
    if density >= 0.4:
        # dense rays stop exactly where the wall begins; the band beyond is merely unobserved wall
        return open_edge + direction * 0.01, 0.012
    return open_edge + direction * min(gap, 0.1) / 2, gap / 4 + 0.005


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
    miss_start_xy: np.ndarray | None = None,
    miss_start_h: np.ndarray | None = None,
    miss_dir_xy: np.ndarray | None = None,
    miss_dir_h: np.ndarray | None = None,
) -> list[OpeningCandidate]:
    p = p or OpeningParams()
    d = end - start
    length = float(np.linalg.norm(d))
    if length < p.min_width:
        return []
    along = d / length
    ns = int(np.ceil(length / p.res))
    nh = int(np.ceil(wall_height / p.res))
    if ns < 3 or nh < 3:
        return []

    s_sol = (solid_xy - start) @ along
    ok = (s_sol >= -0.05) & (s_sol < length + 0.05) & (solid_h >= 0) & (solid_h < wall_height)
    s_sol, h_sol = s_sol[ok], solid_h[ok]
    solid = np.zeros((nh, ns), np.float32)
    inb = (s_sol >= 0) & (s_sol < length)
    np.add.at(solid, (np.clip((h_sol[inb] / p.res).astype(int), 0, nh - 1),
                      np.clip((s_sol[inb] / p.res).astype(int), 0, ns - 1)), 1)

    # rays that end beyond the plane
    a0 = ray_start_xy[:, axis] - offset
    a1 = ray_end_xy[:, axis] - offset
    cand = (a0 * inward > 0.02) & (a1 * inward < -p.beyond_min)
    t = np.zeros(len(a0))
    t[cand] = a0[cand] / (a0[cand] - a1[cand])
    cross_xy = ray_start_xy + (ray_end_xy - ray_start_xy) * t[:, None]
    cross_h = ray_start_h + (ray_end_h - ray_start_h) * t
    s_cr = (cross_xy - start) @ along
    cand &= (s_cr >= 0) & (s_cr < length) & (cross_h >= 0) & (cross_h < wall_height)
    through = np.zeros((nh, ns), np.float32)
    np.add.at(through, ((cross_h[cand] / p.res).astype(int), (s_cr[cand] / p.res).astype(int)), 1)

    open_s, open_h = [s_cr[cand]], [cross_h[cand]]
    # miss rays (no depth returned) crossing the plane within reliable range
    if miss_start_xy is not None and len(miss_start_xy):
        m0 = miss_start_xy[:, axis] - offset
        dm = miss_dir_xy[:, axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            tm = -m0 / dm
        mok = (m0 * inward > 0.02) & np.isfinite(tm) & (tm > 0)
        hit_xy = miss_start_xy + miss_dir_xy * tm[:, None]
        hit_h = miss_start_h + miss_dir_h * tm
        dist = tm * np.sqrt(np.sum(miss_dir_xy**2, axis=1) + miss_dir_h**2)
        sm = (hit_xy - start) @ along
        mok &= (dist < p.miss_max_range) & (sm >= 0) & (sm < length) & (hit_h >= 0) & (hit_h < wall_height)
        np.add.at(through, ((hit_h[mok] / p.res).astype(int), (sm[mok] / p.res).astype(int)), 1)
        open_s.append(sm[mok])
        open_h.append(hit_h[mok])
    open_s = np.concatenate(open_s)
    open_h = np.concatenate(open_h)

    open_seed = ndimage.binary_dilation(through > 0, structure=np.ones((3, 3)), iterations=2)
    open_seed &= ~ndimage.binary_dilation(solid > 0, iterations=1)
    open_seed = ndimage.binary_opening(open_seed, structure=np.ones((3, 3)))
    lab, n = ndimage.label(open_seed)
    out: list[OpeningCandidate] = []
    beyond_idx = np.flatnonzero(cand)
    for k in range(1, n + 1):
        comp = lab == k
        if through[comp].sum() < p.min_through:
            continue
        rows = np.flatnonzero(comp.any(axis=1))
        cols = np.flatnonzero(comp.any(axis=0))
        sc = (cols[0] + cols[-1] + 1) / 2 * p.res
        hc = (rows[0] + rows[-1] + 1) / 2 * p.res
        h_lo_c, h_hi_c = rows[0] * p.res, (rows[-1] + 1) * p.res
        s_lo_c, s_hi_c = cols[0] * p.res, (cols[-1] + 1) * p.res

        # jambs: between the last open and the first solid evidence, over the middle of the height
        hb0, hb1 = h_lo_c + 0.2 * (h_hi_c - h_lo_c), h_hi_c - 0.2 * (h_hi_c - h_lo_c)
        n_rows = max(int(round((hb1 - hb0) / p.res)), 1)
        so_band = (h_sol >= hb0) & (h_sol < hb1)
        op_band = (open_h >= hb0) & (open_h < hb1)
        left, sig_l = _locate_edge(open_s[op_band], s_sol[so_band], sc, max(sc - p.max_width, -0.05), n_rows, p)
        right, sig_r = _locate_edge(open_s[op_band], s_sol[so_band], sc, min(sc + p.max_width, length + 0.05),
                                    n_rows, p)
        s0 = left if left is not None else s_lo_c
        s1 = right if right is not None else s_hi_c
        s0, s1 = max(s0, 0.0), min(s1, length)
        width = s1 - s0
        if width < p.min_width or width > p.max_width:
            continue
        # head and sill over the middle of the width
        sb0, sb1 = s0 + 0.2 * width, s1 - 0.2 * width
        n_cols = max(int(round((sb1 - sb0) / p.res)), 1)
        so_cols = (s_sol >= sb0) & (s_sol < sb1)
        op_cols = (open_s >= sb0) & (open_s < sb1)
        head, sig_h = _locate_edge(open_h[op_cols], h_sol[so_cols], hc, wall_height, n_cols, p)
        h1 = head if head is not None else min(h_hi_c, wall_height)
        solid_below = _profile_edge(h_sol[so_cols], hc, 0.0, n_cols, p)
        low_open = open_h[op_cols].min() if op_cols.any() else hc
        if solid_below is not None and solid_below > p.door_bottom_max:
            sill, sig_s = _locate_edge(open_h[op_cols], h_sol[so_cols], hc, 0.0, n_cols, p)
            h0 = sill if sill is not None else solid_below
        elif solid_below is None and low_open > 0.6:
            h0, sig_s = float(low_open), 0.05  # wall below hidden (furniture): sill bounded by open evidence
        else:
            h0, sig_s = 0.0, 0.0
        if h1 - h0 < p.min_height:
            continue
        sel = beyond_idx[(s_cr[beyond_idx] >= s0) & (s_cr[beyond_idx] < s1) &
                         (cross_h[beyond_idx] >= h0) & (cross_h[beyond_idx] < h1)]
        if h0 == 0.0 and (width >= p.passage_width_min or h1 >= wall_height - 0.05):
            kind = "open_passage"
        elif h0 == 0.0 and h1 >= p.door_top_min:
            kind = "door"
        elif h0 > 0.0:
            kind = "window"
        else:
            continue
        n_through = int(through[comp].sum())
        cand_out = OpeningCandidate(
            s0=s0, s1=s1, h0=h0, h1=h1, kind=kind, width=width, width_sigma=float(np.hypot(sig_l, sig_r)),
            height=h1 - h0, height_sigma=float(np.hypot(sig_h, sig_s)), sill=None if h0 == 0.0 else h0,
            through_rays=n_through, solid_fraction=float((solid[comp] > 0).mean()),
            confidence=float(np.clip(n_through / 150.0, 0.2, 1.0)), beyond_xy=ray_end_xy[sel],
        )
        if room_surface_pts is not None and len(sel) >= 20:
            cand_out.mirror = _looks_like_mirror(ray_end_xy[sel], ray_end_h[sel], axis, offset, room_surface_pts, p)
        out.append(cand_out)
    return _dedupe(out)


def _edge_sigma(s: np.ndarray, h: np.ndarray, s0: float, s1: float, p: OpeningParams) -> float:
    """Per-row jamb positions near each edge; their scatter (and the bin size) bound the width error."""
    sig = []
    for edge, side in ((s0, -1), (s1, 1)):
        near = (np.abs(s - edge) < 0.06)
        if near.sum() < 10:
            sig.append(0.02)
            continue
        rows = np.floor(h[near] / 0.05).astype(int)
        per_row = []
        for r in np.unique(rows):
            ss = s[near][rows == r]
            per_row.append(ss.max() if side < 0 else ss.min())
        per_row = np.array(per_row)
        if len(per_row) < 3:
            sig.append(0.02)
            continue
        spread = 1.4826 * np.median(np.abs(per_row - np.median(per_row)))
        sig.append(float(np.hypot(spread / np.sqrt(len(per_row)), p.profile_bin / np.sqrt(12))))
    return float(np.hypot(*sig))


def _dedupe(cands: list[OpeningCandidate]) -> list[OpeningCandidate]:
    """Candidates whose rectangles overlap substantially are one opening (keep the best supported)."""
    keep: list[OpeningCandidate] = []
    for c in sorted(cands, key=lambda c: -c.through_rays):
        if all(min(c.s1, o.s1) - max(c.s0, o.s0) < 0.5 * min(c.width, o.width) for o in keep):
            keep.append(c)
    return sorted(keep, key=lambda c: c.s0)


def _looks_like_mirror(beyond_xy: np.ndarray, beyond_h: np.ndarray, axis: int, offset: float,
                       room_pts: np.ndarray, p: OpeningParams) -> bool:
    """Reflect the points seen 'through' the plane back across it; a mirror maps them onto the room."""
    refl = beyond_xy.copy()
    refl[:, axis] = 2 * offset - refl[:, axis]
    q = np.column_stack([refl, beyond_h])
    tree = cKDTree(room_pts)
    dd, _ = tree.query(q, k=1, distance_upper_bound=0.5)
    return float(np.mean(dd < p.mirror_nn_dist)) >= p.mirror_fraction
