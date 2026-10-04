"""Room segmentation on the plan-view grids.

Interior space = cells crossed by observation rays or where floor was seen, minus wall cells.
Rooms are the basins of the interior's distance transform (watershed), and two basins stay
separate rooms only where the boundary between them is a constriction no wider than a door. A
long shared boundary means one space (an L-shaped room, an open-plan area) split by noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import disk
from skimage.segmentation import watershed

from groundplan.geometry.occupancy import Grid, WallEvidence, fill_small_holes


@dataclass
class RoomSegmentation:
    labels: np.ndarray  # (ny, nx) int, 0 = not interior, k = room k
    interior: np.ndarray  # bool
    walls: np.ndarray  # bool
    dist: np.ndarray  # distance transform of the interior (m)
    n_rooms: int


@dataclass
class SegParams:
    min_wall_span: float = 0.9  # m of vertical extent for a cell to count as wall
    min_wall_count: int = 3
    min_free_rays: float = 2.0
    open_radius_m: float = 0.08  # removes thin carving streaks through glass and windows
    marker_min_dist: float = 0.35  # m; a basin must be at least this far from walls at its core
    marker_h: float = 0.12  # h-maxima depth separating basins
    max_door_width: float = 1.6  # m; wider shared boundary -> same room
    min_room_area: float = 1.0  # m2


def wall_mask(we: WallEvidence, p: SegParams) -> np.ndarray:
    m = (we.span >= p.min_wall_span) & (we.count >= p.min_wall_count)
    return ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=1)


def interior_mask(grid: Grid, free: np.ndarray, floor: np.ndarray, walls: np.ndarray, p: SegParams) -> np.ndarray:
    inside = (free >= p.min_free_rays) | (floor >= 1)
    inside &= ~ndimage.binary_dilation(walls, iterations=1)
    r = max(int(round(p.open_radius_m / grid.res)), 1)
    inside = ndimage.binary_opening(inside, structure=disk(r))
    min_cells = int(p.min_room_area / grid.res**2)
    lab, n = ndimage.label(inside)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= max(min_cells, 1)
        inside = keep[lab]
    inside = fill_small_holes(inside, max_area_cells=int(1.5 / grid.res**2))
    return inside


def segment_rooms(grid: Grid, free: np.ndarray, floor: np.ndarray, we: WallEvidence,
                  p: SegParams | None = None) -> RoomSegmentation:
    p = p or SegParams()
    walls = wall_mask(we, p)
    inside = interior_mask(grid, free, floor, walls, p)
    dist = ndimage.distance_transform_edt(inside) * grid.res

    # Markers: h-maxima of the distance map with a minimum core size.
    h_steps = max(int(round(p.marker_h / grid.res)), 1)
    dq = np.round(dist / grid.res).astype(np.int32)
    recon = _reconstruct(np.maximum(dq - h_steps, 0), dq)
    hmax = (dq - recon) >= h_steps - 1e-9
    hmax &= dist >= p.marker_min_dist
    markers, nmark = ndimage.label(hmax)
    if nmark == 0:
        markers, nmark = ndimage.label(inside)
    labels = watershed(-dist, markers=markers, mask=inside)
    labels = _merge_wide_boundaries(labels, grid, p)
    labels = _absorb_small(labels, grid, p)
    labels, n = _relabel(labels)
    return RoomSegmentation(labels, inside, walls, dist, n)


def _reconstruct(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import reconstruction

    return reconstruction(marker, mask, method="dilation").astype(np.int32)


def boundary_lengths(labels: np.ndarray, res: float) -> dict[tuple[int, int], float]:
    """Length (m) of the shared boundary between each pair of touching labels."""
    out: dict[tuple[int, int], float] = {}
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        touch = (a != b) & (a > 0) & (b > 0)
        if not touch.any():
            continue
        pa, pb = a[touch], b[touch]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        keys, counts = np.unique(np.stack([lo, hi], 1), axis=0, return_counts=True)
        for (x, y), c in zip(keys, counts):
            out[(int(x), int(y))] = out.get((int(x), int(y)), 0.0) + c * res
    return out


def _merge_wide_boundaries(labels: np.ndarray, grid: Grid, p: SegParams) -> np.ndarray:
    labels = labels.copy()
    while True:
        bl = boundary_lengths(labels, grid.res)
        wide = [(L, k) for k, L in bl.items() if L > p.max_door_width]
        if not wide:
            return labels
        _, (a, b) = max(wide)
        labels[labels == b] = a


def _absorb_small(labels: np.ndarray, grid: Grid, p: SegParams) -> np.ndarray:
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        areas = counts * grid.res**2
        small = [(a, i) for i, a in zip(ids, areas) if a < p.min_room_area]
        if not small:
            return labels
        _, sid = min(small)
        bl = boundary_lengths(labels, grid.res)
        nbrs = [(L, (k[0] if k[1] == sid else k[1])) for k, L in bl.items() if sid in k]
        if nbrs:
            labels[labels == sid] = max(nbrs)[1]
        else:
            labels[labels == sid] = 0


def _relabel(labels: np.ndarray) -> tuple[np.ndarray, int]:
    ids = [i for i in np.unique(labels) if i > 0]
    # order rooms by area, largest first, for stable naming
    areas = {i: int((labels == i).sum()) for i in ids}
    order = sorted(ids, key=lambda i: -areas[i])
    out = np.zeros_like(labels)
    for k, i in enumerate(order, start=1):
        out[labels == i] = k
    return out, len(order)
