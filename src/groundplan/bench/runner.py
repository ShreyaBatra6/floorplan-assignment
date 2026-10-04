"""Benchmark runner: every capture in the manifest, run live, scored against its ground truth.

    groundplan bench run    [--manifest benchmark/manifest.yaml] [--out benchmark/results/<tag>]
    groundplan bench score  <results dir>          # rescore saved plans (no re-run)

The manifest lists captures (raw data paths are relative to ``GROUNDPLAN_DATA``):

    sites:
      flat_a: benchmark/ground_truth/flat_a.yaml
    captures:
      - id: flat_a_lidar_1
        site: flat_a
        tier: lidar
        path: flat_a/lidar_1                 # Stray Scanner export
        repeat_of: null
      - id: flat_a_lidar_2
        site: flat_a
        tier: lidar
        path: flat_a/lidar_2
        repeat_of: flat_a_lidar_1            # same rooms, same tier -> repeatability
      - id: flat_a_photo
        site: flat_a
        tier: photo
        path: flat_a/photos                  # one folder per room
    drift_ablation: [flat_a_lidar_1]         # also run with --no-drift
    head_to_head:
      app: "Polycam (free tier), Room mode"
      app_version: "x.y.z"
      dimensions: benchmark/app_exports/polycam_dimensions.csv
      ours: flat_a_lidar_1

Every reported number in the benchmark report is regenerated from these inputs by this module.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from groundplan.bench.gates import (
    CaptureScore,
    calibration_table,
    ceiling_gate,
    opening_gate,
    repeatability,
    score_capture,
    wall_gate,
)
from groundplan.bench.gt import GroundTruth, load_ground_truth
from groundplan.calib.intervals import load_calibration
from groundplan.contract import Plan

REPO = Path(__file__).resolve().parents[3]


@dataclass
class CaptureSpec:
    id: str
    site: str
    tier: str
    path: str
    repeat_of: str | None = None
    room_map: dict[str, str] | None = None


@dataclass
class Manifest:
    sites: dict[str, str]
    captures: list[CaptureSpec]
    drift_ablation: list[str] = field(default_factory=list)
    head_to_head: dict | None = None
    source: Path | None = None


def load_manifest(path: Path) -> Manifest:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    caps = [CaptureSpec(id=c["id"], site=c["site"], tier=c["tier"], path=c["path"], repeat_of=c.get("repeat_of"),
                        room_map=c.get("room_map")) for c in doc.get("captures", [])]
    return Manifest(sites=doc.get("sites", {}), captures=caps, drift_ablation=doc.get("drift_ablation", []),
                    head_to_head=doc.get("head_to_head"), source=Path(path))


def data_root() -> Path:
    return Path(os.environ.get("GROUNDPLAN_DATA", REPO / "data"))


def run_benchmark(manifest_path: Path, out: Path, only: list[str] | None = None, live: bool = True) -> dict:
    from groundplan.pipeline import RunOptions, run_capture

    man = load_manifest(manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    gts = {name: load_ground_truth(REPO / p) for name, p in man.sites.items()}
    timing = {}
    for cap in man.captures:
        if only and cap.id not in only:
            continue
        cap_out = out / cap.id
        if live or not (cap_out / "plan.json").exists():
            t = time.perf_counter()
            run_capture(data_root() / cap.path, cap_out, RunOptions(tier=cap.tier))
            timing[cap.id] = time.perf_counter() - t
        if cap.id in man.drift_ablation:
            abl = out / f"{cap.id}__no_drift"
            if live or not (abl / "plan.json").exists():
                run_capture(data_root() / cap.path, abl, RunOptions(tier=cap.tier, drift=False, damage=False))
    (out / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return score_benchmark(man, out, gts)


def _load_plan(p: Path) -> Plan:
    return Plan.model_validate_json(p.read_text(encoding="utf-8"))


def score_benchmark(man: Manifest, out: Path, gts: dict[str, GroundTruth] | None = None) -> dict:
    gts = gts or {name: load_ground_truth(REPO / p) for name, p in man.sites.items()}
    scores: list[CaptureScore] = []
    plans: dict[str, Plan] = {}
    for cap in man.captures:
        pj = out / cap.id / "plan.json"
        if not pj.exists():
            continue
        plan = _load_plan(pj)
        plans[cap.id] = plan
        scores.append(score_capture(plan, gts[cap.site], cap.id, cap.room_map))

    gates = []
    for tier in ("lidar", "video", "photo"):
        if any(s.tier == tier for s in scores):
            gates += [wall_gate(scores, tier), ceiling_gate(scores, tier), opening_gate(scores, tier)]
    gates += _photo_stitch_gates(scores)

    repeat = []
    for cap in man.captures:
        if cap.repeat_of and cap.id in plans and cap.repeat_of in plans:
            r = repeatability(plans[cap.repeat_of], plans[cap.id], gts[cap.site])
            repeat.append({"pair": [cap.repeat_of, cap.id], "tier": cap.tier, **r})

    drift = []
    for cid in man.drift_ablation:
        on, off = out / cid / "plan.json", out / f"{cid}__no_drift" / "plan.json"
        if on.exists() and off.exists():
            spec = next(c for c in man.captures if c.id == cid)
            row = {"capture": cid}
            for label, pth in (("on", on), ("off", off)):
                pl = _load_plan(pth)
                sc = score_capture(pl, gts[spec.site], cid)
                walls = sc.by_kind("wall_length")
                fp = sc.by_kind("footprint_area")
                row[label] = {
                    "footprint_m2": pl.stitched.footprint_area.value,
                    "footprint_err_pct": (100 * fp[0].rel) if fp else None,
                    "walls_within": sum(abs(i.err) <= max(0.02, 0.01 * i.truth) for i in walls),
                    "walls": len(walls),
                    "median_wall_err_cm": (100 * sorted(abs(i.err) for i in walls)[len(walls) // 2]) if walls else None,
                    "adjacency": f"{sc.adjacency_found}/{sc.adjacency_gt}",
                    "drift_notes": pl.stitched.drift.notes,
                }
            drift.append(row)

    timing = {s.capture: s.runtime_s for s in scores}
    result = {
        "captures": [s.to_dict() for s in scores],
        "gates": [g.__dict__ for g in gates],
        "repeatability": repeat,
        "calibration": calibration_table(scores),
        "corrections_used": load_calibration().corrections,  # what bench calibrate fits relative to
        "drift_ablation": drift,
        "timing_s": timing,
    }
    if man.head_to_head:
        from groundplan.bench.h2h import head_to_head

        h = man.head_to_head
        if h.get("ours") in plans and Path(REPO / h["dimensions"]).exists():
            spec = next(c for c in man.captures if c.id == h["ours"])
            result["head_to_head"] = head_to_head(plans[h["ours"]], gts[spec.site], REPO / h["dimensions"],
                                                  h.get("app", "app"), h.get("app_version", "?"), spec.room_map)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    from groundplan.bench.report import write_report

    write_report(result, out / "benchmark_report.md")
    return result


def _photo_stitch_gates(scores: list[CaptureScore]):
    from groundplan.bench.gates import PHOTO_FOOTPRINT_REL, Gate

    out = []
    for s in scores:
        if s.tier != "photo":
            continue
        fp = s.by_kind("footprint_area")
        ok_adj = s.adjacency_found == s.adjacency_gt and s.adjacency_extra == 0
        ok_fp = bool(fp) and abs(fp[0].rel) <= PHOTO_FOOTPRINT_REL and fp[0].covered
        val = (f"adjacency {s.adjacency_found}/{s.adjacency_gt} (+{s.adjacency_extra} extra), overlaps {s.overlaps}, "
               f"footprint {('%+.1f%%' % (100 * fp[0].rel)) if fp else 'n/a'}")
        out.append(Gate("photo whole-property stitch", "photo",
                        "one plan, correct adjacency, no overlaps, footprint +-8% with calibrated interval", val,
                        ok_adj and s.overlaps == 0 and ok_fp, s.capture))
    return out
