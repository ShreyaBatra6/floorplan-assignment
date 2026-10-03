"""One capture in, one Plan out: tier detection, the tier front-end, assembly, validation, outputs."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from groundplan import SCHEMA_VERSION, __version__
from groundplan.assemble import AssembleContext, build_rooms, build_stitched
from groundplan.calib.intervals import load_calibration
from groundplan.config import BUDGETS
from groundplan.contract import (
    CalibrationInfo,
    Capture,
    CaptureQuality,
    Plan,
    Runtime,
    ScaleInfo,
)
from groundplan.io.detect import detect, unpack_if_zip
from groundplan.validate import semantic_errors

CACHE_DIR = Path(os.environ.get("GROUNDPLAN_CACHE_DIR", Path.home() / ".cache" / "groundplan"))


@dataclass
class RunOptions:
    tier: str | None = None
    drift: bool = True
    damage: bool = True
    use_cache: bool = True
    render: bool = True


@dataclass
class RunResult:
    plan: Plan
    out_dir: Path | None
    files: list[Path] = field(default_factory=list)


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=5)
        if out.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _machine() -> str:
    return f"{platform.system()} {platform.machine()} · {platform.processor() or 'cpu'} · {os.cpu_count()} threads"


def run_capture(path: Path, out_dir: Path | None = None, opts: RunOptions | None = None) -> RunResult:
    opts = opts or RunOptions()
    t_start = time.perf_counter()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    path = unpack_if_zip(Path(path), CACHE_DIR)
    det = detect(path)
    tier = opts.tier or det.tier
    stages: dict[str, float] = {}
    warnings: list[str] = []
    if opts.tier and opts.tier != det.tier:
        warnings.append(f"tier forced to {opts.tier} (detected {det.tier}: {det.detail})")

    if tier == "lidar":
        from groundplan.tiers.lidar import LidarOptions, run_lidar

        res = run_lidar(det.root, LidarOptions(drift=opts.drift))
        stages.update(res.timings)
        warnings += res.notes
        layout = res.layout
        cap = res.capture
        capture = Capture(
            id=cap.root.name, tier="lidar", source=str(cap.root), app="Stray Scanner",
            duration_s=round(cap.duration_s, 2), frames_total=cap.n, frames_used=len(res.keyframes),
            quality=CaptureQuality(),
        )
        scale = ScaleInfo(source="lidar", sigma_log=0.0, cues=[])
        ctx = AssembleContext(tier="lidar", scale_sigma=0.0, budget=BUDGETS["lidar"])
        drift = res.drift_report
        stitch_method = "single continuous capture: rooms share one (drift-corrected) world frame"
    else:
        raise NotImplementedError(f"the {tier} tier is not wired up yet")

    t = time.perf_counter()
    import numpy as np

    all_v = np.vstack([g.outline.vertices for g in layout.rooms]) if layout.rooms else np.zeros((1, 2))
    ctx.origin = np.floor(all_v.min(axis=0) * 10) / 10  # plan origin at the plan's lower-left
    rooms, _ = build_rooms(layout, ctx)
    stitched = build_stitched(layout, rooms, ctx, drift, stitch_method)
    for r in rooms:
        if not r.coverage.ceiling_observed:
            warnings.append(f"{r.id}: ceiling not observed; reported as a bounded prior interval")
    stages["assemble"] = time.perf_counter() - t

    cal = load_calibration()
    plan = Plan(
        schema_version=SCHEMA_VERSION,
        capture=capture,
        calibration=CalibrationInfo(version=cal.version, fitted=cal.fitted, source=cal.source),
        scale=scale,
        rooms=rooms,
        stitched=stitched,
        warnings=sorted(set(warnings), key=warnings.index),
        runtime=Runtime(groundplan_version=__version__, git_commit=_git_commit(), started_at=started,
                        total_s=0.0, stages={}, machine=_machine()),
    )
    errors = semantic_errors(plan)
    if errors:
        raise RuntimeError("plan failed semantic validation: " + "; ".join(errors[:5]))

    files: list[Path] = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if opts.render:
            t = time.perf_counter()
            from groundplan.render.plan import render_plan

            files += render_plan(plan, out_dir / "plan")
            for r in plan.rooms:
                files += render_plan(plan, out_dir / "rooms" / r.id, room_id=r.id, formats=("svg",))
            stages["render"] = time.perf_counter() - t
    stages = {k: round(v, 3) for k, v in stages.items()}
    plan.runtime.stages = stages
    plan.runtime.total_s = round(time.perf_counter() - t_start, 3)
    if out_dir is not None:
        p = out_dir / "plan.json"
        p.write_text(plan.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        files.insert(0, p)
        from groundplan.summary import write_summary

        files.append(write_summary(plan, out_dir / "summary.md"))
    return RunResult(plan, out_dir, files)


def plan_to_dict(plan: Plan) -> dict:
    return json.loads(plan.model_dump_json())
