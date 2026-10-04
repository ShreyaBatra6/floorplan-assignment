"""Synthetic flats with exact ground truth, built from axis-aligned boxes.

A flat is a set of rectangular rooms (interior faces given in plan coordinates), walls of a fixed
thickness around every room, doors and windows cut through the walls, a floor, a ceiling and some
furniture. World coordinates follow ARKit (y up) with plan ``(x, y) = (world x, -world z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoomSpec:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    ceiling: float | None = None  # defaults to the flat's ceiling


@dataclass
class OpeningSpec:
    """A door or window cut through the wall on one side of a room.

    ``side`` is the room side the opening is on (``N`` = +y, ``S`` = -y, ``E`` = +x, ``W`` = -x);
    ``start`` is the distance of the near jamb from the side's start corner (counter-clockwise
    order: S runs west->east, E south->north, N east->west, W north->south).
    """

    room: str
    side: str
    start: float
    width: float
    kind: str = "door"  # door | window | open_passage
    height: float = 2.03  # head height for doors; head minus sill for windows
    sill: float = 0.0


@dataclass
class BoxSpec:
    lo: tuple[float, float, float]  # plan x, plan y, height
    hi: tuple[float, float, float]
    label: str = "furniture"


@dataclass
class FlatSpec:
    rooms: list[RoomSpec]
    openings: list[OpeningSpec] = field(default_factory=list)
    furniture: list[BoxSpec] = field(default_factory=list)
    wall_t: float = 0.12
    ceiling: float = 2.60

    def room(self, name: str) -> RoomSpec:
        return next(r for r in self.rooms if r.name == name)


# ---------------------------------------------------------------- geometry helpers
def _box_world(lo_plan, hi_plan) -> np.ndarray:
    """Plan-space box (x, y, h) -> world box (x, y_up, z) as [[min], [max]]."""
    (x0, y0, h0), (x1, y1, h1) = lo_plan, hi_plan
    return np.array([[x0, h0, -y1], [x1, h1, -y0]], float)


def _subtract(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Axis-aligned box difference a - b, as up to six boxes."""
    if np.any(a[1] <= b[0]) or np.any(a[0] >= b[1]):
        return [a]
    out = []
    cur = a.copy()
    for ax in range(3):
        if cur[0, ax] < b[0, ax]:
            piece = cur.copy()
            piece[1, ax] = b[0, ax]
            out.append(piece)
            cur[0, ax] = b[0, ax]
        if cur[1, ax] > b[1, ax]:
            piece = cur.copy()
            piece[0, ax] = b[1, ax]
            out.append(piece)
            cur[1, ax] = b[1, ax]
    return [p for p in out if np.all(p[1] - p[0] > 1e-6)]


def side_segment(r: RoomSpec, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Start and end corner (plan) of a room side in counter-clockwise order."""
    corners = {"S": ((r.x0, r.y0), (r.x1, r.y0)), "E": ((r.x1, r.y0), (r.x1, r.y1)),
               "N": ((r.x1, r.y1), (r.x0, r.y1)), "W": ((r.x0, r.y1), (r.x0, r.y0))}
    a, b = corners[side]
    return np.array(a, float), np.array(b, float)


def opening_cutter(spec: FlatSpec, op: OpeningSpec) -> np.ndarray:
    r = spec.room(op.room)
    a, b = side_segment(r, op.side)
    u = (b - a) / np.linalg.norm(b - a)
    p0 = a + u * op.start
    p1 = p0 + u * op.width
    t = spec.wall_t * 2.5  # cut through this room's wall and any wall of the room behind it
    out = {"S": np.array([0, -1.0]), "E": np.array([1.0, 0]), "N": np.array([0, 1.0]), "W": np.array([-1.0, 0])}[op.side]
    q = np.array([p0, p1, p0 + out * t, p1 + out * t, p0 - out * 0.01, p1 - out * 0.01])
    lo, hi = q.min(axis=0), q.max(axis=0)
    h0 = op.sill if op.kind == "window" else -0.05
    h1 = (op.sill + op.height) if op.kind == "window" else op.height
    return _box_world((lo[0], lo[1], h0), (hi[0], hi[1], h1))


def build_boxes(spec: FlatSpec) -> tuple[np.ndarray, list[str]]:
    """All solids of the flat as (B, 2, 3) world boxes plus a label per box."""
    t = spec.wall_t
    walls = []
    for r in spec.rooms:
        H = (r.ceiling or spec.ceiling) + 0.1
        walls += [
            _box_world((r.x0 - t, r.y0 - t, -0.1), (r.x1 + t, r.y0, H)),  # S
            _box_world((r.x0 - t, r.y1, -0.1), (r.x1 + t, r.y1 + t, H)),  # N
            _box_world((r.x0 - t, r.y0, -0.1), (r.x0, r.y1, H)),  # W
            _box_world((r.x1, r.y0, -0.1), (r.x1 + t, r.y1, H)),  # E
        ]
    for op in spec.openings:
        cut = opening_cutter(spec, op)
        nxt = []
        for w in walls:
            nxt += _subtract(w, cut)
        walls = nxt
    boxes = list(walls)
    labels = ["wall"] * len(walls)
    xs = [r.x0 for r in spec.rooms] + [r.x1 for r in spec.rooms]
    ys = [r.y0 for r in spec.rooms] + [r.y1 for r in spec.rooms]
    lo = (min(xs) - 1.0, min(ys) - 1.0)
    hi = (max(xs) + 1.0, max(ys) + 1.0)
    boxes.append(_box_world((lo[0], lo[1], -0.2), (hi[0], hi[1], 0.0)))
    labels.append("floor")
    for r in spec.rooms:
        H = r.ceiling or spec.ceiling
        boxes.append(_box_world((r.x0 - t, r.y0 - t, H), (r.x1 + t, r.y1 + t, H + 0.2)))
        labels.append("ceiling")
    for f in spec.furniture:
        boxes.append(_box_world(f.lo, f.hi))
        labels.append(f.label)
    # outer shell far away so window rays hit something (the "outdoors")
    boxes.append(_box_world((lo[0] - 8, lo[1] - 8, -0.2), (lo[0] - 7.9, hi[1] + 8, 6)))
    boxes.append(_box_world((hi[0] + 7.9, lo[1] - 8, -0.2), (hi[0] + 8, hi[1] + 8, 6)))
    boxes.append(_box_world((lo[0] - 8, lo[1] - 8, -0.2), (hi[0] + 8, lo[1] - 7.9, 6)))
    boxes.append(_box_world((lo[0] - 8, hi[1] + 7.9, -0.2), (hi[0] + 8, hi[1] + 8, 6)))
    labels += ["outdoor"] * 4
    return np.stack(boxes), labels


def _counterpart(spec: FlatSpec, op: OpeningSpec, other: RoomSpec) -> dict | None:
    """The opening ``op`` expressed on the wall of ``other`` it passes through, if it does."""
    a, b = side_segment(spec.room(op.room), op.side)
    u = (b - a) / np.linalg.norm(b - a)
    outn = {"S": (0, -1), "E": (1, 0), "N": (0, 1), "W": (-1, 0)}[op.side]
    p0 = a + u * op.start + np.array(outn) * spec.wall_t
    p1 = a + u * (op.start + op.width) + np.array(outn) * spec.wall_t
    opposite = {"S": "N", "N": "S", "E": "W", "W": "E"}[op.side]
    oa, ob = side_segment(other, opposite)
    ou = (ob - oa) / np.linalg.norm(ob - oa)
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]

    on_line = abs(cross2(ou, p0 - oa)) < 1e-6 and abs(cross2(ou, p1 - oa)) < 1e-6
    s0, s1 = sorted(((p0 - oa) @ ou, (p1 - oa) @ ou))
    if not on_line or s0 < -1e-6 or s1 > np.linalg.norm(ob - oa) + 1e-6:
        return None
    return {"wall": opposite, "kind": op.kind, "width": op.width, "height": op.height, "sill": None,
            "offset": float(s0)}


def ground_truth(spec: FlatSpec) -> dict:
    """Exact measurements, in the benchmark ground-truth layout (see groundplan.bench.gt)."""
    rooms = []
    for r in spec.rooms:
        walls = []
        for side in ("S", "E", "N", "W"):
            a, b = side_segment(r, side)
            walls.append({"id": side, "length": float(np.linalg.norm(b - a))})
        ops = []
        for op in spec.openings:
            if op.room == r.name:
                ops.append({"wall": op.side, "kind": op.kind, "width": op.width,
                            "height": op.height, "sill": op.sill if op.kind == "window" else None,
                            "offset": op.start})
            elif op.kind != "window":
                # the same door seen from the room behind it: it is an opening of both rooms
                mirrored = _counterpart(spec, op, r)
                if mirrored is not None:
                    ops.append(mirrored)
        rooms.append({
            "name": r.name,
            "polygon": [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]],
            "walls": walls,
            "ceiling_height": float(r.ceiling or spec.ceiling),
            "floor_area": float((r.x1 - r.x0) * (r.y1 - r.y0)),
            "openings": ops,
        })
    adjacency = []
    for op in spec.openings:
        if op.kind == "window":
            continue
        cut = opening_cutter(spec, op)
        for other in spec.rooms:
            if other.name == op.room:
                continue
            # the room behind the opening contains the cutter's far face
            ox = (cut[0, 0] + cut[1, 0]) / 2
            oy = -(cut[0, 2] + cut[1, 2]) / 2
            a, b = side_segment(spec.room(op.room), op.side)
            outn = {"S": (0, -1), "E": (1, 0), "N": (0, 1), "W": (-1, 0)}[op.side]
            px, py = ox + outn[0] * spec.wall_t * 1.5, oy + outn[1] * spec.wall_t * 1.5
            if other.x0 <= px <= other.x1 and other.y0 <= py <= other.y1:
                adjacency.append(sorted([op.room, other.name]))
    uniq = sorted({tuple(a) for a in adjacency})
    return {"rooms": rooms, "adjacency": [list(a) for a in uniq],
            "footprint_area": float(sum(x["floor_area"] for x in rooms))}


def demo_flat() -> FlatSpec:
    """Three rooms and a hallway connector (the brief's minimum multi-room composition)."""
    t = 0.12
    hall = RoomSpec("hall", 0.0, 0.0, 1.10, 5.20)
    living = RoomSpec("living", 1.10 + t, 0.0, 1.10 + t + 4.35, 3.60)
    bed = RoomSpec("bedroom", 1.10 + t, 3.60 + t, 1.10 + t + 3.20, 3.60 + t + 3.05)
    bath = RoomSpec("bath", -t - 2.15, 2.40, -t, 2.40 + 2.05)
    return FlatSpec(
        rooms=[hall, living, bed, bath],
        openings=[
            OpeningSpec("hall", "E", 0.80, 0.86),  # hall -> living
            OpeningSpec("hall", "E", 4.10, 0.80),  # hall -> bedroom (shared wall section y 3.72..)
            OpeningSpec("hall", "W", 1.20, 0.76),  # hall -> bath
            OpeningSpec("living", "S", 1.20, 1.50, kind="window", height=1.25, sill=0.90),
            OpeningSpec("bedroom", "N", 0.90, 1.20, kind="window", height=1.30, sill=0.85),
        ],
        furniture=[
            BoxSpec((2.0, 0.3, 0.0), (4.0, 1.2, 0.85), "sofa"),
            BoxSpec((4.6, 2.9, 0.0), (5.5, 3.5, 2.0), "wardrobe"),
            BoxSpec((1.6, 4.4, 0.0), (3.4, 6.4, 0.55), "bed"),
        ],
        wall_t=t,
        ceiling=2.62,
    )
