"""Turn a SceneLayout into the contract Plan: every number becomes a Measurement.

This is where the error budget meets the geometry. Each quantity is built as an ``Estimate`` from
(a) the statistical uncertainty of the fits that produced it and (b) the tier's systematic budget
(``config.BUDGETS``), plus the tier's metric-scale uncertainty as a log-scale term. ``finalize``
then applies the calibration multipliers and emits the interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from groundplan.config import BUDGETS, CEILING_PRIOR, ErrorBudget
from groundplan.contract import (
    Adjacency,
    CeilingLevel,
    DriftReport,
    Measurement,
    Opening,
    Overlap,
    Room,
    RoomCoverage,
    RoomType,
    StitchedPlan,
    Surface,
    Wall,
)
from groundplan.geometry.scene import RoomGeom, SceneLayout
from groundplan.geometry.walls import FACINGS, polygon_area
from groundplan.measure import Estimate, finalize

MC_SAMPLES = 400


@dataclass
class AssembleContext:
    tier: str
    scale_sigma: float = 0.0  # residual log-scale uncertainty of the capture
    budget: ErrorBudget = field(default_factory=lambda: BUDGETS["lidar"])
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))  # subtracted from plan coordinates

    def m(self, est: Estimate, kind: str) -> Measurement:
        return finalize(est, kind, self.tier)


def _edge_sigma(edge, budget: ErrorBudget) -> float:
    """Total 1-sigma uncertainty of one wall face's position."""
    if edge.fit is None:
        return math.hypot(edge.sigma, budget.unobserved_wall)
    return math.sqrt(edge.sigma**2 + budget.face_bias**2 + budget.face_drift**2)


def _mc_vertices(geom: RoomGeom, ctx: AssembleContext, rng: np.random.Generator) -> np.ndarray:
    """Monte-Carlo samples of the outline (S, k, 2), perturbing each wall face independently."""
    edges = geom.outline.edges
    k = len(edges)
    sig = np.array([_edge_sigma(e, ctx.budget) for e in edges])
    offs = np.array([e.offset for e in edges])
    samples = offs[None, :] + rng.normal(size=(MC_SAMPLES, k)) * sig[None, :]
    verts = np.repeat(geom.outline.vertices[None].astype(float), MC_SAMPLES, axis=0)
    for t in range(k):
        e_in, e_out = edges[t - 1], edges[t]
        if e_in.axis == e_out.axis:
            continue
        xin = samples[:, t - 1] if e_in.axis == 0 else samples[:, t]
        yin = samples[:, t - 1] if e_in.axis == 1 else samples[:, t]
        verts[:, t, 0] = xin
        verts[:, t, 1] = yin
    return verts


def _areas(verts: np.ndarray) -> np.ndarray:
    x, y = verts[..., 0], verts[..., 1]
    return 0.5 * (np.sum(x * np.roll(y, -1, axis=-1), axis=-1) - np.sum(y * np.roll(x, -1, axis=-1), axis=-1))


def ceiling_measurement(geom: RoomGeom, ctx: AssembleContext) -> Measurement:
    b = ctx.budget
    if geom.ceiling_y is not None:
        sigma = math.sqrt(geom.floor_sigma**2 + (geom.ceiling_sigma or 0.0) ** 2 + 2 * b.level**2)
        est = Estimate(geom.ceiling_y - geom.floor_y, sigma, ctx.scale_sigma, "m", "floor-to-ceiling plane levels")
        return ctx.m(est, "ceiling_height")
    lo = CEILING_PRIOR[0]
    if geom.ceiling_lower_bound is not None:
        lo = max(lo, geom.ceiling_lower_bound)
    hi = max(CEILING_PRIOR[1], lo + 0.1)
    return Measurement(value=round((lo + hi) / 2, 6), lo=round(lo, 6), hi=round(hi, 6), unit="m", confidence=0.9,
                       sigma_abs=round((hi - lo) / (2 * 1.645), 6), sigma_log=0.0,
                       method="unobserved: bounded prior (ceiling never seen; lower bound = highest wall point)",
                       calibrated=False)


def build_rooms(layout: SceneLayout, ctx: AssembleContext) -> tuple[list[Room], dict[int, str]]:
    rng = np.random.default_rng(0)
    label_to_id: dict[int, str] = {}
    rooms: list[Room] = []
    for idx, geom in enumerate(layout.rooms, start=1):
        label_to_id[geom.label] = f"R{idx}"

    for idx, geom in enumerate(layout.rooms, start=1):
        rid = f"R{idx}"
        verts = geom.outline.vertices - ctx.origin
        edges = geom.outline.edges
        k = len(edges)
        sig = [_edge_sigma(e, ctx.budget) for e in edges]
        ceiling = ceiling_measurement(geom, ctx)

        walls: list[Wall] = []
        surfaces: list[Surface] = []
        for t, e in enumerate(edges):
            a, b = verts[t], verts[(t + 1) % k]
            length = float(np.linalg.norm(b - a))
            prev_s, next_s = sig[t - 1], sig[(t + 1) % k]
            sigma_len = math.sqrt(prev_s**2 + next_s**2 + ctx.budget.definition**2)
            m_len = ctx.m(Estimate(length, sigma_len, ctx.scale_sigma, "m", "wall-face plane intersections"),
                          "wall_length")
            ax, sg = e.axis, e.inward
            outward = np.array([-sg, 0.0]) if ax == 0 else np.array([0.0, -sg])
            wid = f"{rid}-W{t + 1}"
            walls.append(Wall(
                id=wid, room_id=rid, index=t + 1, surface_id=wid,
                start=(round(float(a[0]), 4), round(float(a[1]), 4)),
                end=(round(float(b[0]), 4), round(float(b[1]), 4)),
                length=m_len, height=ceiling,
                outward_normal_deg=round(float(np.degrees(np.arctan2(outward[1], outward[0]))) % 360, 2),
                observed_fraction=round(float(e.observed_fraction), 3),
            ))
            wall_area = Estimate(length * ceiling.value,
                                 math.hypot(sigma_len * ceiling.value, length * ceiling.sigma_abs),
                                 2 * ctx.scale_sigma, "m2", "wall length x wall height")
            surfaces.append(Surface(id=wid, room_id=rid, kind="wall", wall_id=wid, area=ctx.m(wall_area, "default")))

        samples = _mc_vertices(geom, ctx, rng)
        areas = _areas(samples)
        area = polygon_area(geom.outline.vertices)
        m_area = ctx.m(Estimate(area, float(np.std(areas)), 2 * ctx.scale_sigma, "m2", "polygon area of wall faces"),
                       "floor_area")
        perims = np.sum(np.linalg.norm(np.roll(samples, -1, axis=1) - samples, axis=-1), axis=1)
        perim = float(np.sum(np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1)))
        m_perim = ctx.m(Estimate(perim, float(np.std(perims)), ctx.scale_sigma, "m", "sum of wall lengths"), "default")
        surfaces.append(Surface(id=f"{rid}-FLOOR", room_id=rid, kind="floor", area=m_area))
        surfaces.append(Surface(id=f"{rid}-CEILING", room_id=rid, kind="ceiling", area=m_area))

        openings = _openings(geom, layout, rid, walls, label_to_id, ctx)
        levels = [CeilingLevel(height=ctx.m(Estimate(y - geom.floor_y, ctx.budget.level * 2, ctx.scale_sigma, "m",
                                                     "ceiling level"), "ceiling_height"),
                               area_fraction=round(float(f), 3))
                  for y, f in geom.ceiling_levels]
        wall_obs = float(np.average([w.observed_fraction for w in walls], weights=[w.length.value for w in walls]))
        rooms.append(Room(
            id=rid, name=f"Room {idx}", type=RoomType(label="room", confidence=0.0, source="unclassified"),
            is_connector=_is_connector(geom), polygon=[(round(float(x), 4), round(float(y), 4)) for x, y in verts],
            floor_area=m_area, perimeter=m_perim, ceiling_height=ceiling, ceiling_levels=levels,
            walls=walls, openings=openings, surfaces=surfaces,
            coverage=RoomCoverage(floor_observed_fraction=round(geom.floor_observed_fraction, 3),
                                  walls_observed_fraction=round(wall_obs, 3),
                                  ceiling_observed=geom.ceiling_y is not None),
        ))
    return rooms, label_to_id


def _is_connector(geom: RoomGeom) -> bool:
    v = geom.outline.vertices
    w, h = np.ptp(v[:, 0]), np.ptp(v[:, 1])
    short, long_ = min(w, h), max(w, h)
    return bool(short < 1.5 and long_ / max(short, 1e-6) > 2.2)


def _openings(geom: RoomGeom, layout: SceneLayout, rid: str, walls: list[Wall], label_to_id: dict[int, str],
              ctx: AssembleContext) -> list[Opening]:
    out: list[Opening] = []
    b = ctx.budget
    for n, (t, c) in enumerate(sorted(geom.openings, key=lambda x: (x[0], x[1].s0)), start=1):
        if c.mirror:
            continue
        other = _room_behind(c.beyond_xy, layout, geom.label)
        width = Estimate(c.width, math.sqrt(c.width_sigma**2 + 2 * b.opening_jamb**2), ctx.scale_sigma, "m",
                         "ray-crossing gap in the wall face (depth resolution)")
        height = Estimate(c.height, math.hypot(c.height_sigma, b.opening_jamb), ctx.scale_sigma, "m",
                          "ray-crossing gap in the wall face")
        offset = Estimate(c.s0, math.hypot(c.width_sigma, b.opening_jamb), ctx.scale_sigma, "m",
                          "distance from wall start corner to near jamb")
        sill = None
        if c.sill is not None:
            sill = ctx.m(Estimate(c.sill, math.hypot(c.height_sigma, b.opening_jamb), ctx.scale_sigma, "m",
                                  "floor to sill"), "opening_height")
        out.append(Opening(
            id=f"{rid}-O{n}", room_id=rid, wall_id=walls[t].id, kind=c.kind,  # type: ignore[arg-type]
            offset=ctx.m(offset, "opening_width"), width=ctx.m(width, "opening_width"),
            height=ctx.m(height, "opening_height"), sill_height=sill,
            connects_to_room_id=label_to_id.get(other) if other else None,
            detection_confidence=round(float(c.confidence), 3),
            evidence=f"{c.through_rays} rays through; solid fraction {c.solid_fraction:.2f}",
        ))
    return out


def _room_behind(beyond_xy: np.ndarray, layout: SceneLayout, own: int) -> int | None:
    if len(beyond_xy) == 0:
        return None
    iy, ix, ok = layout.grid.index(beyond_xy)
    labs = layout.seg.labels[iy[ok], ix[ok]]
    labs = labs[(labs > 0) & (labs != own)]
    if len(labs) < 5:
        return None
    vals, counts = np.unique(labs, return_counts=True)
    return int(vals[np.argmax(counts)])


def build_stitched(layout: SceneLayout, rooms: list[Room], ctx: AssembleContext, drift: dict,
                   stitch_method: str) -> StitchedPlan:
    adjacency: list[Adjacency] = []
    seen: set[tuple[str, str]] = set()
    by_id = {r.id: r for r in rooms}
    for r in rooms:
        for op in r.openings:
            if op.kind != "window" and op.connects_to_room_id and op.connects_to_room_id in by_id:
                key = tuple(sorted((r.id, op.connects_to_room_id)))
                vias = [op.id] + [o.id for o in by_id[op.connects_to_room_id].openings if o.connects_to_room_id == r.id]
                if key in seen:
                    continue
                seen.add(key)
                adjacency.append(Adjacency(room_a=key[0], room_b=key[1],
                                           kind="door" if op.kind == "door" else "open_passage",
                                           via_opening_ids=sorted(set(vias)), evidence="rays through a shared opening"))
    # shared walls (parallel faces 3-45 cm apart, facing away from each other, overlapping)
    for i, ra in enumerate(layout.rooms):
        for rb in layout.rooms[i + 1:]:
            ida, idb = f"R{layout.rooms.index(ra) + 1}", f"R{layout.rooms.index(rb) + 1}"
            best = _shared_wall(ra, rb)
            if best is None:
                continue
            thickness, overlap = best
            key = tuple(sorted((ida, idb)))
            if key in seen:
                continue
            seen.add(key)
            adjacency.append(Adjacency(
                room_a=key[0], room_b=key[1], kind="shared_wall",
                shared_wall_length=ctx.m(Estimate(overlap, 0.02, ctx.scale_sigma, "m", "overlap of facing walls"),
                                         "wall_length"),
                evidence=f"parallel wall faces {thickness * 100:.0f} cm apart",
            ))

    # an outline can touch itself after jog simplification; GEOS refuses set operations on it
    polys = [make_valid(Polygon(r.polygon)) for r in rooms]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = polys[i].intersection(polys[j]).area
            if inter > 0.01:
                overlaps.append(Overlap(room_a=rooms[i].id, room_b=rooms[j].id, area_m2=round(inter, 4)))
    union = unary_union(polys) if polys else Polygon()
    total_area = sum(r.floor_area.value for r in rooms)
    area_sig = math.sqrt(sum(r.floor_area.sigma_abs**2 for r in rooms))
    footprint = ctx.m(Estimate(total_area, area_sig, 2 * ctx.scale_sigma, "m2", "sum of room floor areas"),
                      "footprint_area")
    if rooms:
        minx, miny, maxx, maxy = union.bounds
        edge_sig = math.hypot(ctx.budget.face_bias, ctx.budget.face_drift)
        bbw = ctx.m(Estimate(maxx - minx, math.sqrt(2) * edge_sig, ctx.scale_sigma, "m", "extent of stitched plan"),
                    "wall_length")
        bbd = ctx.m(Estimate(maxy - miny, math.sqrt(2) * edge_sig, ctx.scale_sigma, "m", "extent of stitched plan"),
                    "wall_length")
    else:
        bbw = bbd = ctx.m(Estimate(0.0, 0.0, 0.0, "m", "empty"), "wall_length")
    return StitchedPlan(
        frame="plan metres, x right / y up, Manhattan-aligned to the dominant wall direction",
        room_ids=[r.id for r in rooms], adjacency=adjacency, footprint_area=footprint,
        bounding_box_width=bbw, bounding_box_depth=bbd, overlaps=overlaps, stitch_method=stitch_method,
        drift=DriftReport(**{k: v for k, v in drift.items() if k in DriftReport.model_fields}),
    )


def _shared_wall(ra: RoomGeom, rb: RoomGeom, tmin: float = 0.03, tmax: float = 0.45, min_overlap: float = 0.3):
    best = None
    va, vb = ra.outline.vertices, rb.outline.vertices
    for ta, ea in enumerate(ra.outline.edges):
        a0, a1 = va[ta], va[(ta + 1) % len(va)]
        for tb, eb in enumerate(rb.outline.edges):
            if ea.axis != eb.axis or ea.inward == eb.inward:
                continue
            # faces look away from each other: room A's interior on one side, B's on the other
            gap = (ea.offset - eb.offset) * ea.inward
            if not (tmin <= gap <= tmax):
                continue
            b0, b1 = vb[tb], vb[(tb + 1) % len(vb)]
            along = 1 - ea.axis
            lo = max(min(a0[along], a1[along]), min(b0[along], b1[along]))
            hi = min(max(a0[along], a1[along]), max(b0[along], b1[along]))
            if hi - lo >= min_overlap and (best is None or hi - lo > best[1]):
                best = (gap, hi - lo)
    return best


__all__ = ["AssembleContext", "build_rooms", "build_stitched", "FACINGS"]
