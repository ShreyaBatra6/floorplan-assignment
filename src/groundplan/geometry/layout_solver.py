"""Stitching separately captured rooms into one plan.

Each room arrives in its own frame (rectilinear outline, doors on its walls, optional compass
heading of the frame). Unknown per room: a rotation (multiple of 90 degrees once the frames are
Manhattan-aligned) and a translation. Evidence:

* **compass**: photo EXIF headings give each room frame's rotation to north (+-15 deg indoors),
  enough to pick the right multiple of 90 degrees;
* **doors**: a door seen from both rooms has the same width and sits on anti-parallel walls; pairing
  two doors fixes the translation (door centres coincide across a wall of plausible thickness);
* **no overlap**: rooms cannot occupy the same floor; a pairing that overlaps another room is wrong;
* **shared walls**: parallel walls of neighbouring rooms end up one wall thickness apart.

Search: best-first over door pairings growing a spanning tree from the room with most doors;
every candidate is scored (width match, compass agreement, overlap) and the placement with no
overlap and the best score wins. Rooms that cannot be connected are placed apart and reported, so
the plan never invents an adjacency it has no evidence for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.affinity import translate as shp_translate
from shapely.geometry import Polygon


@dataclass
class Door:
    center: np.ndarray  # local plan coordinates of the door centre on the wall's interior face
    outward: np.ndarray  # local outward unit normal of its wall
    width: float
    kind: str
    id: str


@dataclass
class RoomIn:
    id: str
    polygon: np.ndarray  # local plan polygon (CCW)
    doors: list[Door]
    north_deg: float | None = None  # rotation of the local frame relative to north, if known


@dataclass
class Placement:
    rotation_deg: float  # multiple of 90 (plus the common Manhattan offset)
    translation: np.ndarray
    placed_by: str


@dataclass
class StitchResult:
    placements: dict[str, Placement]
    links: list[tuple[str, str, str, str]] = field(default_factory=list)  # (room a, door a, room b, door b)
    unplaced: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _rot(v: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _placed_polygon(room: RoomIn, p: Placement) -> Polygon:
    poly = shp_rotate(Polygon(room.polygon), p.rotation_deg, origin=(0, 0))
    return shp_translate(poly, p.translation[0], p.translation[1])


def _rotation_options(room: RoomIn, ref_north: float | None) -> list[tuple[float, float]]:
    """Candidate rotations (deg) with a compass penalty (0 if unknown)."""
    opts = []
    for k in range(4):
        r = 90.0 * k
        if room.north_deg is not None and ref_north is not None:
            # rotating the room by r should bring its north into the reference room's north
            diff = (room.north_deg - r - ref_north + 180) % 360 - 180
            opts.append((r, (diff / 20.0) ** 2))
        else:
            opts.append((r, 0.0))
    return opts


def stitch(rooms: list[RoomIn], wall_t: float = 0.12, max_overlap_m2: float = 0.05,
           width_tol: float = 0.15) -> StitchResult:
    if not rooms:
        return StitchResult({})
    order = sorted(rooms, key=lambda r: (-len([d for d in r.doors if d.kind != "window"]), -Polygon(r.polygon).area))
    root = order[0]
    res = StitchResult({root.id: Placement(0.0, np.zeros(2), "reference room")})
    ref_north = root.north_deg
    placed = {root.id}
    by_id = {r.id: r for r in rooms}
    used_doors: set[tuple[str, str]] = set()

    while len(placed) < len(rooms):
        best = None
        for a_id in list(placed):
            A = by_id[a_id]
            pa = res.placements[a_id]
            for da in A.doors:
                if da.kind == "window" or (a_id, da.id) in used_doors:
                    continue
                ca = _rot(da.center, pa.rotation_deg) + pa.translation
                na = _rot(da.outward, pa.rotation_deg)
                for B in rooms:
                    if B.id in placed:
                        continue
                    for db in B.doors:
                        if db.kind == "window" or (B.id, db.id) in used_doors:
                            continue
                        dw = abs(da.width - db.width)
                        if dw > max(width_tol, 0.18 * max(da.width, db.width)):
                            continue
                        for rot, compass_pen in _rotation_options(B, ref_north):
                            nb = _rot(db.outward, rot)
                            if float(np.dot(na, nb)) > -0.95:  # walls must face each other
                                continue
                            cb = _rot(db.center, rot)
                            t = ca + na * wall_t - cb
                            cand = Placement(rot, t, f"door {da.id} of {a_id} <-> door {db.id} of {B.id}")
                            polyb = _placed_polygon(B, cand)
                            overlap = sum(polyb.intersection(_placed_polygon(by_id[o], res.placements[o])).area
                                          for o in placed)
                            if overlap > max_overlap_m2:
                                continue
                            score = (dw / 0.05) ** 2 + compass_pen + overlap * 10
                            if best is None or score < best[0]:
                                best = (score, B.id, cand, (a_id, da.id, B.id, db.id))
        if best is None:
            break
        _, b_id, cand, link = best
        res.placements[b_id] = cand
        res.links.append(link)
        used_doors.add((link[0], link[1]))
        used_doors.add((link[2], link[3]))
        placed.add(b_id)

    # rooms with no usable door: place to the right of the plan, flagged
    if len(placed) < len(rooms):
        xmax = max(_placed_polygon(by_id[r], res.placements[r]).bounds[2] for r in placed)
        for r in rooms:
            if r.id in placed:
                continue
            rot = 0.0
            if r.north_deg is not None and ref_north is not None:
                rot = min(_rotation_options(r, ref_north), key=lambda o: o[1])[0]
            poly = shp_rotate(Polygon(r.polygon), rot, origin=(0, 0))
            minx, miny, _, _ = poly.bounds
            res.placements[r.id] = Placement(rot, np.array([xmax + 1.0 - minx, -miny]), "unconnected: no door match")
            xmax = _placed_polygon(r, res.placements[r.id]).bounds[2]
            res.unplaced.append(r.id)
            res.notes.append(f"{r.id}: no door could be matched to an already placed room; placed apart, "
                             "adjacency not claimed")
    return res


def transform_points(xy: np.ndarray, p: Placement) -> np.ndarray:
    a = math.radians(p.rotation_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.asarray(xy) @ R.T + p.translation
