"""Assembling separately reconstructed rooms (photo tier, or per-room video clips) into one Plan."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from groundplan.assemble import AssembleContext
from groundplan.contract import Adjacency, DriftReport, Opening, Overlap, Room, StitchedPlan, Wall
from groundplan.geometry.layout_solver import Door, Placement, RoomIn, StitchResult, stitch, transform_points
from groundplan.measure import Estimate

ROOM_WORDS = {
    "kitchen": "kitchen", "bath": "bathroom", "wc": "bathroom", "toilet": "bathroom", "shower": "bathroom",
    "bed": "bedroom", "living": "living room", "lounge": "living room", "hall": "hallway", "corridor": "hallway",
    "entry": "hallway", "dining": "dining room", "office": "office", "study": "office", "laundry": "laundry",
    "utility": "laundry", "closet": "closet", "store": "closet",
}


def room_type_from_name(name: str) -> str | None:
    low = name.lower()
    for word, label in ROOM_WORDS.items():
        if word in low:
            return label
    return None


def relabel(room: Room, new_id: str) -> Room:
    """Rename a room and every id derived from it (walls, surfaces, openings)."""
    old = room.id
    js = room.model_dump_json().replace(f'"{old}-', f'"{new_id}-').replace(f'"{old}"', f'"{new_id}"')
    return Room.model_validate_json(js)


def place(room: Room, p: Placement) -> Room:
    def T(pt):
        q = transform_points(np.array([pt]), p)[0]
        return (round(float(q[0]), 4), round(float(q[1]), 4))

    walls = [w.model_copy(update={"start": T(w.start), "end": T(w.end),
                                  "outward_normal_deg": round((w.outward_normal_deg + p.rotation_deg) % 360, 2)})
             for w in room.walls]
    return room.model_copy(update={"polygon": [T(v) for v in room.polygon], "walls": walls})


def doors_of(room: Room) -> list[Door]:
    out = []
    walls = {w.id: w for w in room.walls}
    for op in room.openings:
        w = walls[op.wall_id]
        a, b = np.array(w.start), np.array(w.end)
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        c = a + u * (op.offset.value + op.width.value / 2)
        ang = math.radians(w.outward_normal_deg)
        out.append(Door(c, np.array([math.cos(ang), math.sin(ang)]), op.width.value, op.kind, op.id))
    return out


def stitch_rooms(rooms: list[Room], north: dict[str, float | None], ctx_by_room: dict[str, AssembleContext],
                 method: str) -> tuple[list[Room], StitchedPlan, StitchResult]:
    ins = [RoomIn(r.id, np.array(r.polygon), doors_of(r), north.get(r.id)) for r in rooms]
    res = stitch(ins)
    placed = [place(r, res.placements[r.id]) for r in rooms]
    by_id = {r.id: r for r in placed}
    adjacency = []
    for a, da, b, db in res.links:
        ra, rb = by_id[a], by_id[b]
        ra.openings = [o.model_copy(update={"connects_to_room_id": b}) if o.id == da else o for o in ra.openings]
        rb.openings = [o.model_copy(update={"connects_to_room_id": a}) if o.id == db else o for o in rb.openings]
        kind = "door" if next(o for o in ra.openings if o.id == da).kind == "door" else "open_passage"
        adjacency.append(Adjacency(room_a=a, room_b=b, kind=kind, via_opening_ids=[da, db],
                                   evidence="door pairing: equal widths on facing walls, compass-consistent, no overlap"))
    polys = [Polygon(r.polygon) for r in placed]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = polys[i].intersection(polys[j]).area
            if inter > 0.01:
                overlaps.append(Overlap(room_a=placed[i].id, room_b=placed[j].id, area_m2=round(inter, 4)))
    total = sum(r.floor_area.value for r in placed)
    # rooms carry independent scale errors: combine additive and per-room scale parts in quadrature
    var = 0.0
    for r in placed:
        a = r.floor_area
        var += a.sigma_abs**2 + (a.value * a.sigma_log) ** 2
    any_ctx = next(iter(ctx_by_room.values()))
    footprint = any_ctx.m(Estimate(total, math.sqrt(var), 0.0, "m2", "sum of room floor areas (stitched)"),
                          "footprint_area")
    union = unary_union(polys)
    minx, miny, maxx, maxy = union.bounds
    sig_len = max((math.hypot(c.budget.face_bias, c.budget.face_drift) for c in ctx_by_room.values()), default=0.02)
    slog = max((c.scale_sigma for c in ctx_by_room.values()), default=0.0)
    bbw = any_ctx.m(Estimate(maxx - minx, 2 * sig_len, slog, "m", "extent of stitched plan"), "wall_length")
    bbd = any_ctx.m(Estimate(maxy - miny, 2 * sig_len, slog, "m", "extent of stitched plan"), "wall_length")
    plan = StitchedPlan(
        frame="plan metres; reference room frame, rooms placed by the layout solver",
        room_ids=[r.id for r in placed], adjacency=adjacency, footprint_area=footprint,
        bounding_box_width=bbw, bounding_box_depth=bbd, overlaps=overlaps, stitch_method=method,
        drift=DriftReport(enabled=True, method="per-room reconstructions (no accumulated drift); rooms joined by "
                                               "door pairing with compass and no-overlap constraints",
                          notes="; ".join(res.notes)),
    )
    return placed, plan, res


__all__ = ["stitch_rooms", "relabel", "room_type_from_name", "Opening", "Wall"]
