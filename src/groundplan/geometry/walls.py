"""Wall lines, room polygons and precise wall offsets.

1. ``detect_wall_lines``: axis-aligned wall faces from the per-facing wall evidence grids. Each
   face is a separate line, so the two sides of a thin partition are never confused, and a face
   carries the direction it looks into (towards the room it bounds).
2. ``room_polygon``: the room is the union of cells of the arrangement of nearby wall lines that are
   mostly covered by the room's interior region. This yields a clean rectilinear outline bounded by
   real walls (L-shapes included), and the same wall lines give the same outline from capture to
   capture, which is what makes the plan repeatable.
3. ``refine_offsets``: each polygon edge is re-fitted to the 3D points of its wall face (robust
   location along the normal, effective sample size = number of frames), giving millimetre-level
   offsets with honest standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from groundplan.geometry.occupancy import Grid
from groundplan.geometry.planes import Fit1D, robust_location

# facing index -> (axis of the wall line, sign of the face normal along the perpendicular axis)
# axis 0: line of constant x (normal along +-x); axis 1: line of constant y (normal along +-y)
FACINGS = {0: (0, +1), 1: (0, -1), 2: (1, +1), 3: (1, -1)}


def facing_of(normals_xy: np.ndarray, min_cos: float = 0.85) -> np.ndarray:
    """Facing index 0..3 (+x, -x, +y, -y) of horizontal unit normals; -1 if not axis-aligned."""
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
    dots = normals_xy @ dirs.T
    best = np.argmax(dots, axis=1)
    ok = dots[np.arange(len(best)), best] >= min_cos
    return np.where(ok, best, -1)


@dataclass
class WallLine:
    facing: int
    axis: int  # 0: constant x; 1: constant y
    sign: int  # direction the face looks into
    offset: float  # plan coordinate of the face
    lo: float  # extent along the line
    hi: float
    support: float  # metres of the line backed by wall evidence

    @property
    def length(self) -> float:
        return self.hi - self.lo


def detect_wall_lines(masks: dict[int, np.ndarray], grid: Grid, min_len: float = 0.25, max_gap: float = 0.3,
                      merge_dist: float = 0.05) -> list[WallLine]:
    lines: list[WallLine] = []
    gap = max(int(round(max_gap / grid.res)), 1)
    for facing, mask in masks.items():
        axis, sign = FACINGS[facing]
        # constant-x lines live in columns; constant-y lines in rows
        M = mask.T if axis == 0 else mask
        origin_off = grid.origin[0] if axis == 0 else grid.origin[1]
        origin_along = grid.origin[1] if axis == 0 else grid.origin[0]
        segs = []
        for r in np.flatnonzero(M.any(axis=1)):
            idx = np.flatnonzero(M[r])
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1):
                length = (run[-1] - run[0] + 1) * grid.res
                if length >= min_len and len(run) * grid.res >= 0.5 * length:
                    segs.append((origin_off + (r + 0.5) * grid.res, origin_along + run[0] * grid.res,
                                 origin_along + (run[-1] + 1) * grid.res, len(run) * grid.res))
        lines += _cluster_segments(segs, facing, axis, sign, merge_dist)
    return lines


def _cluster_segments(segs, facing, axis, sign, merge_dist) -> list[WallLine]:
    """Merge row/column runs belonging to the same face (adjacent offsets, overlapping extents)."""
    segs = sorted(segs)
    clusters: list[list] = []
    for s in segs:
        placed = False
        for c in clusters:
            off = np.average([x[0] for x in c], weights=[x[3] for x in c])
            lo, hi = min(x[1] for x in c), max(x[2] for x in c)
            if abs(s[0] - off) <= merge_dist and s[1] <= hi + 0.3 and s[2] >= lo - 0.3:
                c.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    out = []
    for c in clusters:
        w = np.array([x[3] for x in c])
        off = float(np.average([x[0] for x in c], weights=w))
        lo, hi = min(x[1] for x in c), max(x[2] for x in c)
        support = float(min(w.max() * 1.0, hi - lo))
        out.append(WallLine(facing, axis, sign, off, lo, hi, support))
    return out


@dataclass
class Edge:
    axis: int  # 0: constant x, 1: constant y
    offset: float
    inward: int  # sign of the room interior relative to the edge along the perpendicular axis
    line: WallLine | None = None  # supporting wall face, None if unobserved
    fit: Fit1D | None = None
    sigma: float = 0.05
    observed_fraction: float = 0.0


@dataclass
class RoomOutline:
    vertices: np.ndarray  # (k, 2) CCW
    edges: list[Edge] = field(default_factory=list)  # edge i runs vertices[i] -> vertices[i+1]


def room_polygon(region: np.ndarray, grid: Grid, lines: list[WallLine], min_cover: float = 0.5,
                 search: float = 0.6) -> RoomOutline | None:
    """Rectilinear outline of one room region, bounded by nearby wall faces where they exist."""
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return None
    x0, x1 = grid.origin[0] + xs.min() * grid.res, grid.origin[0] + (xs.max() + 1) * grid.res
    y0, y1 = grid.origin[1] + ys.min() * grid.res, grid.origin[1] + (ys.max() + 1) * grid.res

    def nearby(ax):
        lo_b, hi_b = (x0, x1) if ax == 0 else (y0, y1)
        lo_a, hi_a = (y0, y1) if ax == 0 else (x0, x1)
        return [ln for ln in lines if ln.axis == ax and lo_b - search <= ln.offset <= hi_b + search
                and ln.hi >= lo_a - 0.2 and ln.lo <= hi_a + 0.2]

    xs_cut = sorted({x0, x1} | {ln.offset for ln in nearby(0)})
    ys_cut = sorted({y0, y1} | {ln.offset for ln in nearby(1)})
    xs_cut = _dedupe(xs_cut, 0.03)
    ys_cut = _dedupe(ys_cut, 0.03)

    # coverage of each arrangement cell by the region
    cov = np.zeros((len(ys_cut) - 1, len(xs_cut) - 1))
    integral = np.pad(region.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def to_idx(v, o):
        return int(np.clip(round((v - o) / grid.res), 0, None))

    for j in range(len(ys_cut) - 1):
        for i in range(len(xs_cut) - 1):
            ix0 = min(to_idx(xs_cut[i], grid.origin[0]), grid.shape[1])
            ix1 = min(to_idx(xs_cut[i + 1], grid.origin[0]), grid.shape[1])
            iy0 = min(to_idx(ys_cut[j], grid.origin[1]), grid.shape[0])
            iy1 = min(to_idx(ys_cut[j + 1], grid.origin[1]), grid.shape[0])
            area = max((ix1 - ix0) * (iy1 - iy0), 0)
            if area == 0:
                continue
            s = integral[iy1, ix1] - integral[iy0, ix1] - integral[iy1, ix0] + integral[iy0, ix0]
            cov[j, i] = s / area
    sel = cov >= min_cover
    if not sel.any():
        return None
    # keep the largest 4-connected component of selected cells
    lab, n = ndimage.label(sel)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        sel = lab == (1 + int(np.argmax(sizes)))
    sel = ndimage.binary_fill_holes(sel)
    verts = _cells_to_polygon(sel, xs_cut, ys_cut)
    if verts is None or len(verts) < 4:
        return None
    outline = RoomOutline(vertices=verts)
    outline.edges = _edges_from_vertices(verts, lines)
    return outline


def _dedupe(vals: list[float], tol: float) -> list[float]:
    out: list[float] = []
    for v in vals:
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _cells_to_polygon(sel: np.ndarray, xs: list[float], ys: list[float]) -> np.ndarray | None:
    """Trace the outer boundary of a set of arrangement cells into a CCW rectilinear polygon."""
    ny, nx = sel.shape
    padded = np.pad(sel, 1)
    # directed boundary edges between grid nodes (node (i, j) = (xs[i], ys[j]))
    nxt: dict[tuple[int, int], tuple[int, int]] = {}
    for j in range(ny):
        for i in range(nx):
            if not sel[j, i]:
                continue
            pj, pi = j + 1, i + 1
            if not padded[pj - 1, pi]:  # below empty: edge left->right along bottom
                nxt[(i, j)] = (i + 1, j)
            if not padded[pj, pi + 1]:  # right empty: edge bottom->top along right side
                nxt[(i + 1, j)] = (i + 1, j + 1)
            if not padded[pj + 1, pi]:  # above empty: edge right->left along top
                nxt[(i + 1, j + 1)] = (i, j + 1)
            if not padded[pj, pi - 1]:  # left empty: edge top->bottom along left side
                nxt[(i, j + 1)] = (i, j)
    if not nxt:
        return None
    start = min(nxt)
    loop = [start]
    cur = nxt[start]
    guard = 0
    while cur != start and guard < 100000:
        loop.append(cur)
        cur = nxt.get(cur)
        if cur is None:
            return None
        guard += 1
    pts = np.array([[xs[i], ys[j]] for i, j in loop], float)
    # drop collinear vertices
    keep = []
    k = len(pts)
    for t in range(k):
        a, b, c = pts[t - 1], pts[t], pts[(t + 1) % k]
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9:
            keep.append(b)
    return np.array(keep)


def _edges_from_vertices(verts: np.ndarray, lines: list[WallLine], snap_tol: float = 0.04) -> list[Edge]:
    edges = []
    k = len(verts)
    for t in range(k):
        a, b = verts[t], verts[(t + 1) % k]
        if abs(a[0] - b[0]) < 1e-9:  # constant x
            axis, offset = 0, float(a[0])
            # CCW polygon: going up (b.y > a.y) the interior is on the left (-x)
            inward = -1 if b[1] > a[1] else 1
            lo, hi = sorted((a[1], b[1]))
        else:
            axis, offset = 1, float(a[1])
            inward = 1 if b[0] > a[0] else -1
            lo, hi = sorted((a[0], b[0]))
        best = None
        for ln in lines:
            if ln.axis != axis or ln.sign != inward or abs(ln.offset - offset) > snap_tol:
                continue
            overlap = min(hi, ln.hi) - max(lo, ln.lo)
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, ln)
        line = best[1] if best else None
        frac = (min(best[0] / max(hi - lo, 1e-6), 1.0)) if best else 0.0
        edges.append(Edge(axis=axis, offset=offset, inward=inward, line=line, observed_fraction=frac))
    return edges


def refine_offsets(outline: RoomOutline, pts_xy: np.ndarray, h: np.ndarray, facing: np.ndarray,
                   groups: np.ndarray, height_lo: float, height_hi: float, band: float = 0.08,
                   corner_margin: float = 0.12, unobserved_sigma: float = 0.08) -> RoomOutline:
    """Re-fit each edge's offset from the 3D points of the matching wall face."""
    verts = outline.vertices
    k = len(verts)
    hband = (h > height_lo) & (h < height_hi)
    for t, e in enumerate(outline.edges):
        a, b = verts[t], verts[(t + 1) % k]
        along_axis = 1 - e.axis
        lo, hi = sorted((a[along_axis], b[along_axis]))
        if hi - lo <= 2 * corner_margin:
            lo_m, hi_m = lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)
        else:
            lo_m, hi_m = lo + corner_margin, hi - corner_margin
        want_facing = [f for f, (ax, sg) in FACINGS.items() if ax == e.axis and sg == e.inward][0]
        sel = (
            hband
            & (facing == want_facing)
            & (np.abs(pts_xy[:, e.axis] - e.offset) < band)
            & (pts_xy[:, along_axis] > lo_m)
            & (pts_xy[:, along_axis] < hi_m)
        )
        if sel.sum() >= 30:
            fit = robust_location(pts_xy[sel, e.axis], groups=groups[sel])
            e.fit = fit
            e.offset = fit.value
            e.sigma = fit.sigma
            covered = np.unique(np.floor(pts_xy[sel, along_axis] / 0.05)).size * 0.05
            e.observed_fraction = float(min(covered / max(hi_m - lo_m, 1e-6), 1.0))
        else:
            e.sigma = unobserved_sigma
            e.observed_fraction = 0.0
    outline.vertices = vertices_from_edges(outline.edges, verts)
    return outline


def simplify_outline(outline: RoomOutline, min_edge: float = 0.18, keep_observed: float = 0.6,
                     min_keep_len: float = 0.08) -> RoomOutline:
    """Remove short jogs from a rectilinear outline.

    A short edge between two parallel edges is a step; dropping it merges its neighbours into one
    wall at the offset of the better-observed (then longer) of the two, so precise wall positions
    survive and the outline stays rectilinear. Short edges that are themselves well observed (a real
    pier or nib wall) are kept unless shorter than ``min_keep_len``.
    """
    edges = list(outline.edges)
    verts = outline.vertices.astype(float)
    changed = True
    while changed and len(edges) > 4:
        changed = False
        k = len(edges)
        lengths = np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1)
        order = np.argsort(lengths)
        for t in order:
            if lengths[t] >= min_edge:
                break
            e = edges[t]
            if e.observed_fraction >= keep_observed and lengths[t] >= min_keep_len:
                continue
            prev_i, next_i = (t - 1) % k, (t + 1) % k
            ep, en = edges[prev_i], edges[next_i]
            if ep.axis != en.axis or ep.axis == e.axis:
                continue
            lp, ln = lengths[prev_i], lengths[next_i]
            score_p = (ep.observed_fraction, lp)
            score_n = (en.observed_fraction, ln)
            winner = ep if score_p >= score_n else en
            merged = Edge(axis=ep.axis, offset=winner.offset, inward=winner.inward, line=winner.line, fit=winner.fit,
                          sigma=winner.sigma,
                          observed_fraction=float((ep.observed_fraction * lp + en.observed_fraction * ln) / max(lp + ln, 1e-9)))
            if ep.inward != en.inward:
                continue  # a U-turn (spike); leave it for the coverage step to resolve
            # replace prev, t, next by merged
            idx = sorted({prev_i, t, next_i})
            new_edges = []
            new_verts = []
            for j in range(k):
                if j == prev_i:
                    new_edges.append(merged)
                    new_verts.append(verts[j])
                elif j in (t, next_i):
                    continue
                else:
                    new_edges.append(edges[j])
                    new_verts.append(verts[j])
            if len(new_edges) < 4:
                continue
            edges = new_edges
            verts = vertices_from_edges(edges, np.array(new_verts))
            changed = True
            _ = idx
            break
    outline.edges = edges
    outline.vertices = vertices_from_edges(edges, verts)
    return outline


def vertices_from_edges(edges: list[Edge], old: np.ndarray) -> np.ndarray:
    """Corner i is the intersection of edge i-1 (incoming) and edge i (outgoing)."""
    k = len(edges)
    out = old.copy().astype(float)
    for t in range(k):
        e_in, e_out = edges[t - 1], edges[t]
        if e_in.axis == e_out.axis:  # degenerate (should not happen for a rectilinear outline)
            continue
        x = e_in.offset if e_in.axis == 0 else e_out.offset
        y = e_in.offset if e_in.axis == 1 else e_out.offset
        out[t] = (x, y)
    return out


def polygon_area(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
