"""Scoring a Plan against ground truth, and the gates of the brief.

Thresholds are quoted from the brief (Round 2 additions) or marked as inferred where they come
from the Round 1 targets that the brief says still apply but does not restate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from groundplan.bench.gt import GroundTruth
from groundplan.bench.match import PlanMatch, match_rooms
from groundplan.contract import Measurement, Plan

# --- gate thresholds -------------------------------------------------------------------------
OPENING_TOL_M = 0.02  # brief: opening widths <= 2 cm ...
OPENING_PASS_RATE = 0.85  # ... on >= 85% of openings; missed and phantom openings count as misses
CEILING_TOL_M = 0.015  # brief: ceiling height <= 1.5 cm per room
CEILING_SPREAD_M = 0.010  # brief: spread across repeated captures <= 1 cm
REPEAT_ABS_M = 0.010  # brief: two captures agree within 1 cm ...
REPEAT_REL = 0.005  # ... or 0.5% per wall
WALL_REL = {"photo": 0.08, "video": 0.03}  # brief: photo +-8 %, video +-3 %
WALL_LIDAR = (0.02, 0.01)  # inferred Round 1 LiDAR target: max(2 cm, 1 %)
PHOTO_FOOTPRINT_REL = 0.08  # brief: photo-tier stitched footprint within +-8 %
COVERAGE_BAND = (0.85, 0.95)  # calibration: nominal 90% intervals should cover 85-95% of truths


@dataclass
class Item:
    kind: str  # wall_length | ceiling_height | opening_width | floor_area | footprint_area
    room: str
    ref: str
    truth: float
    value: float
    lo: float
    hi: float

    @property
    def err(self) -> float:
        return self.value - self.truth

    @property
    def rel(self) -> float:
        return self.err / self.truth if self.truth else float("nan")

    @property
    def covered(self) -> bool:
        return self.lo <= self.truth <= self.hi


@dataclass
class CaptureScore:
    capture: str
    tier: str
    items: list[Item] = field(default_factory=list)
    openings_gt: int = 0
    openings_matched_within: int = 0
    openings_missed: int = 0
    openings_phantom: int = 0
    adjacency_found: int = 0
    adjacency_gt: int = 0
    adjacency_extra: int = 0
    overlaps: int = 0
    rooms_matched: int = 0
    rooms_gt: int = 0
    rooms_pred: int = 0
    wall_count_mismatch: int = 0
    runtime_s: float = 0.0
    manual_match: bool = False

    def by_kind(self, kind: str) -> list[Item]:
        return [i for i in self.items if i.kind == kind]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["items"] = [dict(asdict(i), err=i.err, rel=i.rel, covered=i.covered) for i in self.items]
        return d


def _item(kind, room, ref, truth, m: Measurement) -> Item:
    return Item(kind, room, ref, float(truth), m.value, m.lo, m.hi)


def score_capture(plan: Plan, gt: GroundTruth, capture_id: str, manual: dict[str, str] | None = None) -> CaptureScore:
    pm: PlanMatch = match_rooms(plan, gt, manual)
    sc = CaptureScore(capture=capture_id, tier=plan.capture.tier, rooms_gt=len(gt.rooms), rooms_pred=len(plan.rooms),
                      rooms_matched=len(pm.rooms), runtime_s=plan.runtime.total_s, manual_match=pm.manual)
    name_of = {}
    for rm in pm.rooms:
        name_of[rm.pred.id] = rm.gt.name
        if not rm.wall_count_ok:
            sc.wall_count_mismatch += 1
        for wp in rm.walls:
            sc.items.append(_item("wall_length", rm.gt.name, wp.gt_id, wp.truth, wp.pred.length))
        if rm.gt.ceiling is not None:
            sc.items.append(_item("ceiling_height", rm.gt.name, "ceiling", rm.gt.ceiling, rm.pred.ceiling_height))
        if rm.gt.area() is not None:
            sc.items.append(_item("floor_area", rm.gt.name, "floor", rm.gt.area(), rm.pred.floor_area))
        for op in rm.openings:
            if op.gt is not None:
                sc.openings_gt += 1
                if op.pred is None:
                    sc.openings_missed += 1
                else:
                    it = _item("opening_width", rm.gt.name, op.gt.id, op.gt.width, op.pred.width)
                    sc.items.append(it)
                    if abs(it.err) <= OPENING_TOL_M + 1e-9:
                        sc.openings_matched_within += 1
            else:
                sc.openings_phantom += 1
    # rooms the plan missed entirely: their openings are all misses
    for name in pm.unmatched_gt:
        sc.openings_gt += len(gt.room(name).openings)
        sc.openings_missed += len(gt.room(name).openings)

    # adjacency, in ground-truth room names
    pred_adj = set()
    for a in plan.stitched.adjacency:
        if a.kind == "shared_wall":
            continue
        if a.room_a in name_of and a.room_b in name_of:
            pred_adj.add(tuple(sorted((name_of[a.room_a], name_of[a.room_b]))))
    gt_adj = set(gt.adjacency)
    sc.adjacency_gt = len(gt_adj)
    sc.adjacency_found = len(gt_adj & pred_adj)
    sc.adjacency_extra = len(pred_adj - gt_adj)
    sc.overlaps = len(plan.stitched.overlaps)
    fp = gt.footprint()
    if fp is not None and len(pm.rooms) == len(gt.rooms):
        sc.items.append(_item("footprint_area", "*", "footprint", fp, plan.stitched.footprint_area))
    return sc


# --- gate evaluation ----------------------------------------------------------------------------
@dataclass
class Gate:
    name: str
    tier: str
    threshold: str
    value: str
    passed: bool | None  # None = not measured
    detail: str = ""


def opening_gate(scores: list[CaptureScore], tier: str) -> Gate:
    s = [x for x in scores if x.tier == tier]
    total = sum(x.openings_gt + x.openings_phantom for x in s)
    good = sum(x.openings_matched_within for x in s)
    if total == 0:
        return Gate("opening widths", tier, "<=2 cm on >=85%, misses+phantoms count", "n/a", None)
    rate = good / total
    return Gate("opening widths", tier, "<=2 cm on >=85%, misses+phantoms count", f"{rate:.0%} ({good}/{total})",
                rate >= OPENING_PASS_RATE,
                f"missed {sum(x.openings_missed for x in s)}, phantom {sum(x.openings_phantom for x in s)}")


def ceiling_gate(scores: list[CaptureScore], tier: str) -> Gate:
    items = [i for x in scores if x.tier == tier for i in x.by_kind("ceiling_height")]
    if not items:
        return Gate("ceiling height", tier, "<=1.5 cm per room", "n/a", None)
    errs = np.array([i.err for i in items])
    ok = int((np.abs(errs) <= CEILING_TOL_M + 1e-9).sum())
    return Gate("ceiling height", tier, "<=1.5 cm per room", f"{ok}/{len(items)} rooms; max {np.abs(errs).max() * 100:.1f} cm",
                ok == len(items), f"bias {errs.mean() * 100:+.1f} cm, sd {errs.std() * 100:.1f} cm")


def wall_gate(scores: list[CaptureScore], tier: str) -> Gate:
    items = [i for x in scores if x.tier == tier for i in x.by_kind("wall_length")]
    if not items:
        return Gate("wall lengths", tier, "", "n/a", None)
    if tier == "lidar":
        tol = [max(WALL_LIDAR[0], WALL_LIDAR[1] * i.truth) for i in items]
        thr = "max(2 cm, 1%) (Round 1, inferred)"
    else:
        tol = [WALL_REL[tier] * i.truth for i in items]
        thr = f"+-{WALL_REL[tier]:.0%} with calibrated intervals"
    ok = sum(abs(i.err) <= t for i, t in zip(items, tol))
    cov = np.mean([i.covered for i in items])
    return Gate("wall lengths", tier, thr, f"{ok}/{len(items)} within; coverage {cov:.0%}",
                ok == len(items) and COVERAGE_BAND[0] <= cov, f"median |err| {np.median([abs(i.err) for i in items]) * 100:.1f} cm")


def calibration_table(scores: list[CaptureScore]) -> list[dict]:
    rows = []
    for tier in ("lidar", "video", "photo"):
        for kind in ("wall_length", "ceiling_height", "opening_width", "floor_area", "footprint_area"):
            items = [i for x in scores if x.tier == tier for i in x.by_kind(kind)]
            if not items:
                continue
            cov = float(np.mean([i.covered for i in items]))
            width = float(np.median([i.hi - i.lo for i in items]))
            rows.append({"tier": tier, "quantity": kind, "n": len(items), "coverage": cov,
                         "median_interval_width": width,
                         "median_abs_err": float(np.median([abs(i.err) for i in items])),
                         "in_band": COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1]})
    return rows


def repeatability(plan_a: Plan, plan_b: Plan, gt: GroundTruth | None = None) -> dict:
    """Per-wall agreement between two captures of the same room(s) at the same tier."""
    from groundplan.bench.match import match_rooms as _mr

    if gt is not None:
        ma, mb = _mr(plan_a, gt), _mr(plan_b, gt)
        rows = []
        for ra in ma.rooms:
            rb = next((r for r in mb.rooms if r.gt.name == ra.gt.name), None)
            if rb is None:
                continue
            wb = {w.gt_id: w.pred for w in rb.walls}
            for w in ra.walls:
                if w.gt_id in wb:
                    a, b = w.pred.length.value, wb[w.gt_id].length.value
                    tol = max(REPEAT_ABS_M, REPEAT_REL * (a + b) / 2)
                    rows.append({"room": ra.gt.name, "wall": w.gt_id, "a": a, "b": b, "diff": a - b,
                                 "tol": tol, "ok": abs(a - b) <= tol})
            ca, cb = ra.pred.ceiling_height.value, rb.pred.ceiling_height.value
            rows.append({"room": ra.gt.name, "wall": "ceiling", "a": ca, "b": cb, "diff": ca - cb,
                         "tol": CEILING_SPREAD_M, "ok": abs(ca - cb) <= CEILING_SPREAD_M})
        return {"rows": rows, "passed": bool(rows) and all(r["ok"] for r in rows)}
    raise ValueError("repeatability needs ground truth to put walls in correspondence")
