"""Ground truth: what was measured by laser or tape, in a format a person fills in on site.

One YAML file per site (property). Walls are listed counter-clockwise as seen from above,
starting from the wall that contains the room's entrance door (see docs/BENCHMARK_PROTOCOL.md for
how to measure each quantity). Repeated readings are kept so the measurement's own uncertainty is
known; the value used is their mean.

    site: flat_a
    instrument: "Bosch GLM 50-27 laser, +-1.5 mm"
    measured_by: Shreya
    date: 2026-10-11
    rooms:
      - name: living
        walls:
          - {id: W1, readings: [4.352, 4.351]}
          - {id: W2, readings: [3.598, 3.601]}
        ceiling: {readings: [2.618, 2.616, 2.619]}
        diagonals: [5.646, 5.649]          # optional, checks squareness and area
        openings:
          - {id: D1, wall: W1, kind: door, width: [0.862, 0.861], height: [2.031], offset: 0.42,
             connects_to: hall}
        damage:
          - {id: X1, class: water_stain, surface: W3, width: 0.30, height: 0.22, bottom: 0.95, offset: 1.10}
    adjacency:
      - [living, hall]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


def _value(x) -> tuple[float | None, list[float]]:
    if x is None:
        return None, []
    if isinstance(x, (int, float)):
        return float(x), [float(x)]
    if isinstance(x, dict):
        if "readings" in x:
            r = [float(v) for v in x["readings"]]
            return float(np.mean(r)), r
        if "value" in x:
            return float(x["value"]), [float(x["value"])]
    if isinstance(x, (list, tuple)):
        r = [float(v) for v in x]
        return float(np.mean(r)), r
    raise ValueError(f"cannot read a measurement from {x!r}")


@dataclass
class GTWall:
    id: str
    length: float
    readings: list[float]


@dataclass
class GTOpening:
    id: str
    wall: str
    kind: str
    width: float
    height: float | None
    offset: float | None
    sill: float | None
    connects_to: str | None


@dataclass
class GTDamage:
    id: str
    damage_class: str
    surface: str
    width: float
    height: float
    bottom: float | None = None
    offset: float | None = None


@dataclass
class GTRoom:
    name: str
    walls: list[GTWall]
    ceiling: float | None
    ceiling_readings: list[float]
    openings: list[GTOpening] = field(default_factory=list)
    damage: list[GTDamage] = field(default_factory=list)
    diagonals: list[float] = field(default_factory=list)
    floor_area: float | None = None
    connector: bool = False

    def area(self) -> float | None:
        """Floor area: given, or from a rectangle's walls (cross-checked with the diagonals)."""
        if self.floor_area is not None:
            return self.floor_area
        if len(self.walls) == 4:
            a = (self.walls[0].length + self.walls[2].length) / 2
            b = (self.walls[1].length + self.walls[3].length) / 2
            return a * b
        return None


@dataclass
class GroundTruth:
    site: str
    rooms: list[GTRoom]
    adjacency: list[tuple[str, str]]
    instrument: str = ""
    footprint_area: float | None = None
    source: str = ""

    def room(self, name: str) -> GTRoom:
        return next(r for r in self.rooms if r.name == name)

    def footprint(self) -> float | None:
        if self.footprint_area is not None:
            return self.footprint_area
        areas = [r.area() for r in self.rooms]
        return float(sum(areas)) if all(a is not None for a in areas) else None


def parse_ground_truth(doc: dict, source: str = "") -> GroundTruth:
    rooms = []
    for r in doc.get("rooms", []):
        walls = []
        for w in r.get("walls", []):
            v, rd = _value(w.get("readings", w.get("length")))
            walls.append(GTWall(str(w["id"]), v, rd))
        ceil, ceil_r = _value(r.get("ceiling", r.get("ceiling_height")))
        ops = []
        for o in r.get("openings", []):
            ops.append(GTOpening(
                id=str(o["id"]), wall=str(o["wall"]), kind=o.get("kind", "door"),
                width=_value(o["width"])[0], height=_value(o.get("height"))[0],
                offset=_value(o.get("offset"))[0], sill=_value(o.get("sill"))[0],
                connects_to=o.get("connects_to"),
            ))
        dmg = [GTDamage(str(d["id"]), d["class"], str(d["surface"]), float(d["width"]), float(d["height"]),
                        d.get("bottom"), d.get("offset")) for d in r.get("damage", [])]
        rooms.append(GTRoom(
            name=str(r["name"]), walls=walls, ceiling=ceil, ceiling_readings=ceil_r, openings=ops, damage=dmg,
            diagonals=[float(x) for x in r.get("diagonals", [])],
            floor_area=_value(r.get("floor_area"))[0], connector=bool(r.get("connector", False)),
        ))
    adj = [tuple(sorted((str(a), str(b)))) for a, b in doc.get("adjacency", [])]
    return GroundTruth(site=str(doc.get("site", "site")), rooms=rooms, adjacency=adj,
                       instrument=str(doc.get("instrument", "")),
                       footprint_area=_value(doc.get("footprint_area"))[0], source=source)


def load_ground_truth(path: Path) -> GroundTruth:
    with open(path, encoding="utf-8") as fh:
        return parse_ground_truth(yaml.safe_load(fh), str(path))


def from_sim(gt_json: dict) -> GroundTruth:
    """The simulator's exact ground truth in the same structure (walls S, E, N, W = CCW)."""
    doc = {"site": "sim", "rooms": [], "adjacency": gt_json.get("adjacency", [])}
    for r in gt_json["rooms"]:
        doc["rooms"].append({
            "name": r["name"],
            "walls": [{"id": w["id"], "length": w["length"]} for w in r["walls"]],
            "ceiling": r["ceiling_height"],
            "floor_area": r["floor_area"],
            "openings": [{"id": f"{o['kind'][0].upper()}{k + 1}", "wall": o["wall"], "kind": o["kind"],
                          "width": o["width"], "height": o["height"], "offset": o["offset"], "sill": o.get("sill")}
                         for k, o in enumerate(r["openings"])],
        })
    return parse_ground_truth(doc, "simulator")
