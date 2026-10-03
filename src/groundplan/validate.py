"""Validation of plan documents against the published JSON Schema (and the pydantic models)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from groundplan.contract import Plan, json_schema


def validate_document(doc: dict) -> list[str]:
    validator = jsonschema.Draft202012Validator(json_schema())
    errors = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator.iter_errors(doc)]
    if errors:
        return errors
    # Semantic checks the schema cannot express.
    plan = Plan.model_validate(doc)
    return semantic_errors(plan)


def semantic_errors(plan: Plan) -> list[str]:
    errors: list[str] = []
    room_ids = {r.id for r in plan.rooms}
    surface_ids = {s.id for r in plan.rooms for s in r.surfaces}
    for room in plan.rooms:
        for wall in room.walls:
            if wall.surface_id not in surface_ids:
                errors.append(f"wall {wall.id} references unknown surface {wall.surface_id}")
        for op in room.openings:
            if op.connects_to_room_id and op.connects_to_room_id not in room_ids:
                errors.append(f"opening {op.id} connects to unknown room {op.connects_to_room_id}")
    for adj in plan.stitched.adjacency:
        if adj.room_a not in room_ids or adj.room_b not in room_ids:
            errors.append(f"adjacency {adj.room_a}-{adj.room_b} references an unknown room")
    region_ids = {d.id for d in plan.damage_regions}
    for d in plan.damage_regions:
        if d.surface_id not in surface_ids:
            errors.append(f"damage region {d.id} on unknown surface {d.surface_id}")
    flag_ids = {f.id for f in plan.concealed_damage_flags}
    for f in plan.concealed_damage_flags:
        for rid in f.region_ids:
            if rid not in region_ids:
                errors.append(f"flag {f.id} cites unknown region {rid}")
    for item in plan.scope:
        if item.surface_id not in surface_ids:
            errors.append(f"scope item {item.id} keyed to unknown surface {item.surface_id}")
        for rid in item.region_ids:
            if rid not in region_ids:
                errors.append(f"scope item {item.id} cites unknown region {rid}")
        for fid in item.flag_ids:
            if fid not in flag_ids:
                errors.append(f"scope item {item.id} cites unknown flag {fid}")
    return errors


def validate_file(path: Path) -> list[str]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read: {exc}"]
    return validate_document(doc)
