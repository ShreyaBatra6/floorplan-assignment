"""Damage, concealed-damage flags and scope for one capture (tier-independent)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from groundplan.assemble import AssembleContext
from groundplan.contract import ConcealedFlag, DamageRegion, Room, RoomType, ScopeItem
from groundplan.damage.detect import DetectParams, Detection, classify, find_candidates
from groundplan.damage.ortho import Mosaic, SurfaceGeom, View, build_mosaic
from groundplan.damage.rules import evaluate
from groundplan.damage.scope import build_scope
from groundplan.geometry.transforms import PlanFrame
from groundplan.measure import Estimate


@dataclass
class DamageResult:
    regions: list[DamageRegion] = field(default_factory=list)
    flags: list[ConcealedFlag] = field(default_factory=list)
    scope: list[ScopeItem] = field(default_factory=list)
    room_types: dict[str, RoomType] = field(default_factory=dict)
    mosaics: list[Mosaic] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def surfaces_for_room(room: Room, frame: PlanFrame, origin: np.ndarray, floor_offset: float) -> list[SurfaceGeom]:
    """Metric surface frames (world) for the walls, floor and ceiling of one contract Room."""

    def W(xy, h):
        return frame.to_world(np.asarray(xy, float) + origin, h + floor_offset)

    out = []
    H = room.ceiling_height.value
    for wall in room.walls:
        a, b = np.array(wall.start), np.array(wall.end)
        p0, p1 = W(a, 0.0), W(b, 0.0)
        u_axis = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        ang = math.radians(wall.outward_normal_deg)
        inward_plan = -np.array([math.cos(ang), math.sin(ang)])
        n = W(a + inward_plan, 0.0) - p0
        n /= np.linalg.norm(n)
        excl = []
        for op in room.openings:
            if op.wall_id == wall.id:
                s0 = op.sill_height.value if op.sill_height else 0.0
                excl.append((op.offset.value - 0.03, op.offset.value + op.width.value + 0.03, s0 - 0.03,
                             s0 + op.height.value + 0.03))
        out.append(SurfaceGeom(wall.surface_id, room.id, "wall", p0, u_axis, np.array([0.0, 1.0, 0.0]), n,
                               0.0, wall.length.value, 0.0, H, excl))
    poly = np.array(room.polygon)
    (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
    ex = W((x0 + 1, y0), 0.0) - W((x0, y0), 0.0)
    ey = W((x0, y0 + 1), 0.0) - W((x0, y0), 0.0)
    out.append(SurfaceGeom(f"{room.id}-FLOOR", room.id, "floor", W((x0, y0), 0.0), ex, ey, np.array([0.0, 1.0, 0.0]),
                           x0, x1, y0, y1, polygon_uv=poly))
    if room.coverage.ceiling_observed:
        out.append(SurfaceGeom(f"{room.id}-CEILING", room.id, "ceiling", W((x0, y0), H), ex, ey,
                               np.array([0.0, -1.0, 0.0]), x0, x1, y0, y1, polygon_uv=poly))
    return out


def classify_rooms(rooms: list[Room], views: list[View], frame: PlanFrame, origin: np.ndarray,
                   preset: dict[str, str] | None = None) -> dict[str, RoomType]:
    from matplotlib.path import Path as MplPath

    from groundplan.models import clip

    types: dict[str, RoomType] = {}
    for room in rooms:
        if preset and room.id in preset:
            types[room.id] = RoomType(label=preset[room.id], confidence=1.0, source="folder name")
            continue
        if not clip.available() or not views:
            types[room.id] = RoomType(label="room", confidence=0.0, source="unclassified (no model)")
            continue
        path = MplPath(np.array(room.polygon))
        inside = [v for v in views if path.contains_point(frame.to_plan(v.center) - origin)]
        level = [v for v in inside if abs(v.T_wc[1, 2]) < 0.5]  # optical axis roughly horizontal
        pick = (level or inside)[:: max(len(level or inside) // 4, 1)][:4]
        if not pick:
            types[room.id] = RoomType(label="room", confidence=0.0, source="no view inside the room")
            continue
        label, p = clip.classify_room([cv2.resize(v.image, (336, 252)) for v in pick])
        types[room.id] = RoomType(label=label, confidence=round(p, 3), source="CLIP zero-shot on views in the room")
    return types


def run_damage(rooms: list[Room], views: list[View], frame: PlanFrame, ctx: AssembleContext,
               floor_offsets: dict[str, float], out_dir: Path | None = None,
               room_presets: dict[str, str] | None = None, params: DetectParams | None = None) -> DamageResult:
    res = DamageResult()
    p = params or DetectParams()
    if not views:
        res.notes.append("no images available for damage detection")
        return res
    res.room_types = classify_rooms(rooms, views, frame, ctx.origin, room_presets)
    budget = ctx.budget
    n = 0
    for room in rooms:
        for surf in surfaces_for_room(room, frame, ctx.origin, floor_offsets.get(room.id, 0.0)):
            mosaic = build_mosaic(surf, views, res=0.005 if surf.kind == "wall" else 0.015)
            if mosaic is None:
                continue
            res.mosaics.append(mosaic)
            for det in classify(mosaic, find_candidates(mosaic, p), p, views):
                n += 1
                res.regions.append(_to_region(f"D{n}", room, surf, det, ctx, budget, out_dir))
    res.flags = evaluate(rooms, res.regions, {rid: {t.label} for rid, t in res.room_types.items()})
    res.scope = build_scope(rooms, res.regions, res.flags)
    res.notes.append("floor surfaces are not assessed visually (rugs, mats and furniture make floor staining "
                     "unreliable); floor-level water is flagged through wall-base staining (rule CDR-02)")
    if not any(m.surface.kind == "ceiling" for m in res.mosaics):
        res.notes.append("ceilings not observed: no ceiling damage could be assessed")
    return res


def _to_region(rid: str, room: Room, surf: SurfaceGeom, det: Detection, ctx: AssembleContext, budget,
               out_dir: Path | None) -> DamageRegion:
    geo = math.hypot(budget.face_bias, budget.face_drift)
    perim = 2 * (det.width + det.height)
    area = ctx.m(Estimate(det.area, math.hypot(det.area_spread, perim * geo), 2 * ctx.scale_sigma, "m2",
                          "pixels of the region on the metric surface mosaic"), "damage_extent")
    width = ctx.m(Estimate(det.width, math.sqrt(det.extent_spread**2 + 2 * geo**2), ctx.scale_sigma, "m",
                           "extent on the surface mosaic"), "damage_extent")
    height = ctx.m(Estimate(det.height, math.sqrt(det.extent_spread**2 + 2 * geo**2), ctx.scale_sigma, "m",
                            "extent on the surface mosaic"), "damage_extent")
    centre_h = None
    if surf.kind == "wall":
        vs = [v for _, v in det.polygon_uv]
        centre_h = ctx.m(Estimate(float(np.mean(vs)), geo, ctx.scale_sigma, "m", "region centre above the floor"),
                         "damage_extent")
    evidence = None
    if out_dir is not None:
        (out_dir / "damage").mkdir(parents=True, exist_ok=True)
        evidence = f"damage/{rid}.jpg"
        cv2.imwrite(str(out_dir / evidence), det.crop)
    return DamageRegion(id=rid, room_id=room.id, surface_id=surf.id, damage_class=det.damage_class,  # type: ignore[arg-type]
                        confidence=round(det.confidence, 3),
                        polygon_uv=[(round(u, 4), round(v, 4)) for u, v in det.polygon_uv],
                        area=area, width=width, height=height, center_height_above_floor=centre_h,
                        views=det.views, evidence_image=evidence)
