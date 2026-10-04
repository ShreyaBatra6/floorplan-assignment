"""Head-to-head against a consumer scanning app on the same rooms, dimension by dimension.

The app's numbers are entered from its own export / on-screen plan into a CSV next to the exported
file itself (both are submitted)::

    room,dimension,app_value,source
    living,W1,4.36,Polycam floor plan PDF p.1
    living,ceiling,2.61,Polycam room details
    living,D1,0.84,Polycam floor plan PDF p.1
    living,floor_area,15.71,Polycam room details

``dimension`` is a ground-truth wall id, ``ceiling``, ``floor_area`` or a ground-truth opening id.
A dimension is a *tie* when both errors are within the measurement's own resolution: 5 mm for
lengths (laser +-1.5 mm, plus where exactly each side placed the corner) and 1 % for areas.
"""

from __future__ import annotations

import csv
from pathlib import Path

from groundplan.bench.gt import GroundTruth
from groundplan.bench.match import match_rooms
from groundplan.contract import Plan

TIE_LENGTH_M = 0.005
TIE_AREA_REL = 0.01
PASS_RATE = 0.70  # brief: beat or tie on >= 70% of shared dimensions


def _ours(plan: Plan, gt: GroundTruth, room_map) -> dict[tuple[str, str], float]:
    pm = match_rooms(plan, gt, room_map)
    out = {}
    for rm in pm.rooms:
        name = rm.gt.name
        for w in rm.walls:
            out[(name, w.gt_id)] = w.pred.length.value
        out[(name, "ceiling")] = rm.pred.ceiling_height.value
        out[(name, "floor_area")] = rm.pred.floor_area.value
        for op in rm.openings:
            if op.gt is not None and op.pred is not None:
                out[(name, op.gt.id)] = op.pred.width.value
    return out


def _truth(gt: GroundTruth) -> dict[tuple[str, str], float]:
    out = {}
    for r in gt.rooms:
        for w in r.walls:
            out[(r.name, w.id)] = w.length
        if r.ceiling is not None:
            out[(r.name, "ceiling")] = r.ceiling
        if r.area() is not None:
            out[(r.name, "floor_area")] = r.area()
        for o in r.openings:
            out[(r.name, o.id)] = o.width
    return out


def head_to_head(plan: Plan, gt: GroundTruth, app_csv: Path, app: str, version: str, room_map=None) -> dict:
    ours = _ours(plan, gt, room_map)
    truth = _truth(gt)
    rows = []
    with open(app_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["room"].strip(), r["dimension"].strip())
            if key not in truth:
                continue
            app_v = float(r["app_value"])
            our_v = ours.get(key)
            t = truth[key]
            area = key[1] == "floor_area"
            tie = TIE_AREA_REL * t if area else TIE_LENGTH_M
            row = {"room": key[0], "dimension": key[1], "truth": t, "app": app_v, "app_err": app_v - t,
                   "ours": our_v, "ours_err": None if our_v is None else our_v - t, "source": r.get("source", "")}
            if our_v is None:
                row["result"] = "loss (not reported by us)"
            elif abs(our_v - t) <= tie and abs(app_v - t) <= tie:
                row["result"] = "tie"
            elif abs(our_v - t) <= abs(app_v - t):
                row["result"] = "beat"
            else:
                row["result"] = "loss"
            rows.append(row)
    # dimensions the app did not report but we did count for us only if the app could not measure them
    wins = sum(r["result"] in ("beat", "tie") for r in rows)
    return {"app": app, "app_version": version, "rows": rows, "shared": len(rows), "beat_or_tie": wins,
            "rate": wins / len(rows) if rows else None,
            "passed": (wins / len(rows) >= PASS_RATE) if rows else None,
            "tie_rule": f"tie when both errors <= {TIE_LENGTH_M * 1000:.0f} mm (lengths) or {TIE_AREA_REL:.0%} (areas)"}
