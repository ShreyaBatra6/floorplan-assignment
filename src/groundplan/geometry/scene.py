"""The shared geometric core: fused cloud + camera rays -> rooms, walls, openings, adjacency.

Every tier ends here. The LiDAR tier feeds sensor depth with (drift-corrected) ARKit poses; the
video tier feeds monocular depth aligned to an SfM reconstruction; the photo tier feeds per-photo
monocular depth registered within each room. What differs per tier is only the noise model
(``CoreParams``) and the metric-scale uncertainty, which is carried separately (``sigma_log``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from groundplan.geometry.floorceiling import Level, find_ceiling, find_floor
from groundplan.geometry.manhattan import dominant_yaw
from groundplan.geometry.occupancy import Grid, WallEvidence, carve_free_space, wall_evidence
from groundplan.geometry.openings import OpeningCandidate, OpeningParams, detect_openings
from groundplan.geometry.planes import robust_location
from groundplan.geometry.rooms import RoomSegmentation, SegParams, segment_rooms
from groundplan.geometry.transforms import PlanFrame
from groundplan.geometry.walls import (
    FACINGS,
    RoomOutline,
    WallLine,
    detect_wall_lines,
    facing_of,
    polygon_area,
    refine_offsets,
    room_polygon,
    simplify_outline,
)


@dataclass
class SceneInput:
    xyz: np.ndarray  # (M, 3) world, y up, metres (or model-metric)
    normal: np.ndarray  # (M, 3)
    group: np.ndarray  # (M,) observation group (frame) per point, for effective sample sizes
    cams: np.ndarray  # (K, 3) camera centres
    ray_cam: np.ndarray  # (R,) index into cams per ray
    ray_end: np.ndarray  # (R, 3) ray endpoints (observed surfaces)


@dataclass
class CoreParams:
    grid_res: float = 0.02
    wall_band_bottom: float = 0.08  # ignore skirting / floor junction
    wall_band_top_margin: float = 0.08
    default_room_height: float = 3.0  # used only to bound the wall band when no ceiling is seen
    min_wall_span: float = 0.9
    refine_band: float = 0.08
    refine_h_lo: float = 0.25
    refine_h_top_margin: float = 0.2
    unobserved_wall_sigma: float = 0.08
    seg: SegParams = field(default_factory=SegParams)
    openings: OpeningParams = field(default_factory=OpeningParams)


@dataclass
class RoomGeom:
    label: int
    outline: RoomOutline
    region: np.ndarray  # bool grid mask
    floor_y: float
    floor_sigma: float
    ceiling_y: float | None
    ceiling_sigma: float | None
    ceiling_levels: list[tuple[float, float]]  # (y, area fraction)
    ceiling_lower_bound: float | None  # highest wall point seen, when the ceiling itself was not
    floor_observed_fraction: float
    openings: list[tuple[int, OpeningCandidate]] = field(default_factory=list)  # (edge index, cand)

    @property
    def area(self) -> float:
        return polygon_area(self.outline.vertices)


@dataclass
class SceneLayout:
    frame: PlanFrame
    grid: Grid
    floor: Level
    ceiling: Level | None
    manhattan_strength: float
    seg: RoomSegmentation
    lines: list[WallLine]
    rooms: list[RoomGeom]
    wall_evidence: WallEvidence
    free: np.ndarray
    floor_seen: np.ndarray
    notes: list[str] = field(default_factory=list)


def build_layout(scene: SceneInput, p: CoreParams | None = None) -> SceneLayout:
    p = p or CoreParams()
    notes: list[str] = []
    xyz, nrm = scene.xyz, scene.normal

    floor = find_floor(xyz, nrm, scene.cams[:, 1])
    if floor is None:
        raise ValueError("no floor found: the capture must show the floor (see the capture protocol)")
    ceiling, _ = find_ceiling(xyz, nrm, scene.cams[:, 1], floor.y)
    if ceiling is None:
        notes.append("ceiling not observed in this capture; ceiling heights are bounded by priors")

    vertical = np.abs(nrm[:, 1]) < 0.25
    frame0 = PlanFrame(0.0, floor_y=floor.y)
    nxy0 = frame0.dir_to_plan(nrm[vertical])
    nxy0 /= np.linalg.norm(nxy0, axis=1, keepdims=True) + 1e-9
    yaw, strength = dominant_yaw(nxy0)
    if strength < 0.35:
        notes.append(f"weak Manhattan structure (strength {strength:.2f}); non-orthogonal walls are approximated")
    frame = PlanFrame(yaw, floor_y=floor.y)

    xy = frame.to_plan(xyz)
    h = frame.height(xyz)
    nplan = frame.dir_to_plan(nrm)
    nplan /= np.linalg.norm(nplan, axis=1, keepdims=True) + 1e-9
    facing = np.where(vertical, facing_of(nplan), -1)

    cams_xy = frame.to_plan(scene.cams)
    grid = Grid.around(np.vstack([xy, cams_xy]), p.grid_res, 0.6)
    top = (ceiling.y - floor.y - p.wall_band_top_margin) if ceiling else p.default_room_height
    band = (h > p.wall_band_bottom) & (h < top)

    we_all = wall_evidence(grid, xy[vertical & band], h[vertical & band])
    masks = {}
    for f in FACINGS:
        sel = (facing == f) & band
        we = wall_evidence(grid, xy[sel], h[sel])
        masks[f] = (we.span >= p.min_wall_span) & (we.count >= 2)
    lines = detect_wall_lines(masks, grid)

    floor_seen = grid.count(xy[(nrm[:, 1] > 0.9) & (np.abs(h) < 0.04)])
    ray_end_xy = frame.to_plan(scene.ray_end)
    ray_start_xy = cams_xy[scene.ray_cam]
    free = carve_free_space(grid, ray_start_xy, ray_end_xy)

    seg = segment_rooms(grid, free, floor_seen, we_all, p.seg)

    ray_start_h = frame.height(scene.cams)[scene.ray_cam]
    ray_end_h = frame.height(scene.ray_end)

    rooms: list[RoomGeom] = []
    for k in range(1, seg.n_rooms + 1):
        region = seg.labels == k
        outline = room_polygon(region, grid, lines)
        if outline is None:
            notes.append(f"room region {k} has no closed outline; skipped")
            continue
        outline = refine_offsets(outline, xy, h, facing, scene.group, p.refine_h_lo,
                                 top - p.refine_h_top_margin, band=p.refine_band,
                                 unobserved_sigma=p.unobserved_wall_sigma)
        outline = simplify_outline(outline)
        if polygon_area(outline.vertices) < p.seg.min_room_area:
            continue
        rooms.append(_room_levels(k, outline, region, grid, xy, h, nrm, scene.group, floor, ceiling))

    # openings on every observed wall
    room_pts = np.column_stack([xy, h])
    for room in rooms:
        verts = room.outline.vertices
        nv = len(verts)
        wall_h = (room.ceiling_y - room.floor_y) if room.ceiling_y is not None else top
        for t, e in enumerate(room.outline.edges):
            a, b = verts[t], verts[(t + 1) % nv]
            want = [f for f, (ax, sg) in FACINGS.items() if ax == e.axis and sg == e.inward][0]
            sel = (facing == want) & (np.abs(xy[:, e.axis] - e.offset) < p.openings.solid_band)
            cands = detect_openings(e.axis, e.offset, e.inward, a, b, wall_h, xy[sel], h[sel],
                                    ray_start_xy, ray_start_h, ray_end_xy, ray_end_h, room_pts, p.openings)
            for c in cands:
                room.openings.append((t, c))

    return SceneLayout(frame, grid, floor, ceiling, strength, seg, lines, rooms, we_all, free, floor_seen, notes)


def _room_levels(label: int, outline: RoomOutline, region: np.ndarray, grid: Grid, xy: np.ndarray,
                 h: np.ndarray, nrm: np.ndarray, group: np.ndarray, floor: Level,
                 ceiling: Level | None) -> RoomGeom:
    """Per-room floor and ceiling heights from the points inside the room's outline."""
    from matplotlib.path import Path as MplPath

    inside = MplPath(outline.vertices).contains_points(xy)
    y_abs = h + floor.y
    fsel = inside & (nrm[:, 1] > 0.95) & (np.abs(h) < 0.06)
    if fsel.sum() >= 50:
        ff = robust_location(y_abs[fsel], groups=group[fsel])
        floor_y, floor_sigma = ff.value, ff.sigma
    else:
        floor_y, floor_sigma = floor.y, max(floor.spread, 0.005)

    cells = np.unique(np.floor(xy[fsel] / 0.1).astype(np.int64), axis=0) if fsel.any() else np.zeros((0, 2))
    area = max(polygon_area(outline.vertices), 1e-6)
    floor_obs = float(min(len(cells) * 0.01 / area, 1.0))

    ceiling_y = ceiling_sigma = lower = None
    levels: list[tuple[float, float]] = []
    csel = inside & (nrm[:, 1] < -0.95) & (h > 1.8)
    if csel.sum() >= 50:
        hs = y_abs[csel]
        # main level = densest 2 cm band; secondary levels reported separately
        hist, edges = np.histogram(hs, bins=np.arange(hs.min() - 0.01, hs.max() + 0.03, 0.02))
        y0 = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
        main = csel & (np.abs(y_abs - y0) < 0.04)
        cf = robust_location(y_abs[main], groups=group[main])
        ceiling_y, ceiling_sigma = cf.value, cf.sigma
        frac_main = float(main.sum() / csel.sum())
        levels.append((ceiling_y, frac_main))
        rest = csel & ~main
        if rest.sum() > 0.15 * csel.sum():
            y1 = float(np.median(y_abs[rest]))
            levels.append((y1, 1 - frac_main))
    else:
        wall_pts = inside & (np.abs(nrm[:, 1]) < 0.3)
        # walls near this room (inside test fails for wall points exactly on the boundary)
        if not wall_pts.any():
            wall_pts = np.abs(nrm[:, 1]) < 0.3
        lower = float(np.percentile(h[wall_pts], 99.5)) if wall_pts.any() else None
    return RoomGeom(label, outline, region, floor_y, floor_sigma, ceiling_y, ceiling_sigma, levels, lower, floor_obs)
