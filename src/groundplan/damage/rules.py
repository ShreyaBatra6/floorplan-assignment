"""Concealed-damage rule engine: deterministic, auditable, one flag per (rule, region)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import LineString, Polygon

from groundplan.contract import ConcealedFlag, DamageRegion, Room

RULES_PATH = Path(__file__).with_name("rules.yaml")


@lru_cache(maxsize=1)
def load_rules() -> dict:
    with open(RULES_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _surface_kind(region: DamageRegion) -> str:
    if region.surface_id.endswith("-CEILING"):
        return "ceiling"
    if region.surface_id.endswith("-FLOOR"):
        return "floor"
    return "wall"


def _bottom(region: DamageRegion) -> float:
    return float(min(v for _, v in region.polygon_uv))


def _u_range(region: DamageRegion) -> tuple[float, float]:
    us = [u for u, _ in region.polygon_uv]
    return float(min(us)), float(max(us))


def _check(cond: dict, region: DamageRegion, room: Room, room_types: set[str]) -> tuple[bool, list[str]]:
    """Evaluate a rule's conditions; return (fired, list of the facts that satisfied it)."""
    facts: list[str] = []
    kind = _surface_kind(region)
    if "classes" in cond:
        if region.damage_class not in cond["classes"]:
            return False, []
        facts.append(f"class={region.damage_class}")
    if "surfaces" in cond:
        if kind not in cond["surfaces"]:
            return False, []
        facts.append(f"surface={kind}")
    if "bottom_below_m" in cond:
        if kind != "wall" or _bottom(region) > cond["bottom_below_m"]:
            return False, []
        facts.append(f"bottom {_bottom(region):.2f} m <= {cond['bottom_below_m']} m")
    if "room_types" in cond:
        hit = room_types & set(cond["room_types"])
        if not hit:
            return False, []
        facts.append(f"room type {sorted(hit)[0]}")
    if "min_area_m2" in cond:
        if region.area.value < cond["min_area_m2"]:
            return False, []
        facts.append(f"area {region.area.value:.2f} m2 >= {cond['min_area_m2']} m2")
    if "near_window_m" in cond or "near_opening_top_m" in cond:
        if kind != "wall":
            return False, []
        u0, u1 = _u_range(region)
        vmax = max(v for _, v in region.polygon_uv)
        ops = [o for o in room.openings if o.wall_id == region.surface_id]
        if "near_window_m" in cond:
            d = cond["near_window_m"]
            near = [o for o in ops if o.kind == "window" and u1 >= o.offset.value - d
                    and u0 <= o.offset.value + o.width.value + d]
            if not near:
                return False, []
            facts.append(f"within {d} m of window {near[0].id}")
        if "near_opening_top_m" in cond:
            d = cond["near_opening_top_m"]
            hit = None
            for o in ops:
                head = (o.sill_height.value if o.sill_height else 0.0) + o.height.value
                for cu in (o.offset.value, o.offset.value + o.width.value):
                    dist = np.hypot(max(u0 - cu, 0, cu - u1), max(head - vmax, 0, _bottom(region) - head))
                    if dist <= d:
                        hit = o
            if hit is None:
                return False, []
            facts.append(f"within {d} m of the head corner of {hit.id}")
    return True, facts


def evaluate(rooms: list[Room], regions: list[DamageRegion], room_types: dict[str, set[str]]) -> list[ConcealedFlag]:
    rules = load_rules()
    by_room = {r.id: r for r in rooms}
    flags: list[ConcealedFlag] = []
    for region in regions:
        room = by_room[region.room_id]
        types = room_types.get(room.id, set())
        for rule in rules.get("rules", []):
            fired, facts = _check(rule["when"], region, room, types)
            if fired:
                flags.append(_flag(len(flags) + 1, rule, room.id, [region], facts))
    for rule in rules.get("pair_rules", []):
        flags += _pairs(rule, rooms, regions, start=len(flags) + 1)
    return flags


def _flag(n: int, rule: dict, room_id: str, regions: list[DamageRegion], facts: list[str]) -> ConcealedFlag:
    return ConcealedFlag(
        id=f"F{n}", rule_id=rule["id"], rule=f"{rule['name']}: if {_describe(rule['when'])}",
        condition="; ".join(facts) + f" [{', '.join(r.id for r in regions)}]",
        suspected=" ".join(rule["suspected"].split()), room_id=room_id,
        surface_ids=sorted({r.surface_id for r in regions}), region_ids=[r.id for r in regions],
        severity=rule["severity"], recommended_verification=rule["verify"], reference=rule.get("reference"),
    )


def _describe(cond: dict) -> str:
    parts = []
    for k, v in cond.items():
        label = {"classes": "class in", "surfaces": "surface in", "bottom_below_m": "bottom height <= m",
                 "room_types": "room type in", "near_window_m": "distance to a window <= m",
                 "near_opening_top_m": "distance to an opening head corner <= m", "min_area_m2": "area >= m2",
                 "max_gap_m": "gap between regions <= m"}.get(k, k)
        parts.append(f"{label} {v}")
    return ", ".join(parts)


def _pairs(rule: dict, rooms: list[Room], regions: list[DamageRegion], start: int) -> list[ConcealedFlag]:
    out = []
    cond = rule["when"]
    by_room = {r.id: r for r in rooms}
    for room in rooms:
        walls = {w.id: w for w in room.walls}
        rs = [r for r in regions if r.room_id == room.id and r.damage_class in cond["classes"]]
        wall_rs = [r for r in rs if _surface_kind(r) == "wall"]
        ceil_rs = [r for r in rs if _surface_kind(r) == "ceiling"]
        for wr in wall_rs:
            w = walls.get(wr.surface_id)
            if w is None:
                continue
            top = max(v for _, v in wr.polygon_uv)
            if w.height.value - top > cond["max_gap_m"]:
                continue
            line = LineString([w.start, w.end])
            for cr in ceil_rs:
                if Polygon(cr.polygon_uv).distance(line) <= cond["max_gap_m"]:
                    facts = [f"{wr.id} reaches {w.height.value - top:.2f} m from the ceiling",
                             f"{cr.id} within {cond['max_gap_m']} m of wall {w.id}"]
                    out.append(_flag(start + len(out), rule, room.id, [wr, cr], facts))
    _ = by_room
    return out
