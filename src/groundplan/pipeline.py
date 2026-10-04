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


def _cache_state() -> str:
    from groundplan.models.registry import cache_stats

    return cache_stats()


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
        capture, scale, rooms, stitched, regions, flags, scope_items = _lidar_branch(det, opts, out_dir, warnings, stages)
    elif tier == "photo":
        capture, scale, rooms, stitched, regions, flags, scope_items = _photo_branch(det, opts, out_dir, warnings, stages)
    elif tier == "video":
        from groundplan.tiers.video import video_branch

        capture, scale, rooms, stitched, regions, flags, scope_items = video_branch(det, opts, out_dir, warnings, stages)
    else:
        raise ValueError(f"unknown tier {tier!r}")
    for r in rooms:
        if not r.coverage.ceiling_observed:
            warnings.append(f"{r.id}: ceiling not observed; reported as a bounded prior interval")

    cal = load_calibration()
    plan = Plan(
        schema_version=SCHEMA_VERSION,
        capture=capture,
        calibration=CalibrationInfo(version=cal.version, fitted=cal.fitted, source=cal.source),
        scale=scale,
        rooms=rooms,
        stitched=stitched,
        damage_regions=regions,
        concealed_damage_flags=flags,
        scope=scope_items,
        warnings=sorted(set(warnings), key=warnings.index),
        runtime=Runtime(groundplan_version=__version__, git_commit=_git_commit(), started_at=started,
                        total_s=0.0, stages={}, machine=_machine(), model_cache=_cache_state()),
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
        if opts.render:
            from groundplan.render.report import write_report

            files.append(write_report(plan, out_dir))
    return RunResult(plan, out_dir, files)


def plan_to_dict(plan: Plan) -> dict:
    return json.loads(plan.model_dump_json())


def _apply_room_types(rooms, types) -> None:
    for r in rooms:
        if r.id in types:
            r.type = types[r.id]
            if types[r.id].confidence > 0 and types[r.id].source != "folder name":
                r.name = types[r.id].label.title()


def _lidar_branch(det, opts: RunOptions, out_dir, warnings: list[str], stages: dict):
    import numpy as np

    from groundplan.tiers.lidar import LidarOptions, collect_views, run_lidar

    res = run_lidar(det.root, LidarOptions(drift=opts.drift))
    stages.update(res.timings)
    warnings += res.notes
    res.scene = None  # free the fused cloud before damage detection
    layout, cap = res.layout, res.capture
    capture = Capture(id=cap.root.name, tier="lidar", source=str(cap.root), app="Stray Scanner",
                      duration_s=round(cap.duration_s, 2), frames_total=cap.n, frames_used=len(res.keyframes),
                      quality=CaptureQuality())
    scale = ScaleInfo(source="lidar", sigma_log=0.0, cues=[])
    ctx = AssembleContext(tier="lidar", scale_sigma=0.0, budget=BUDGETS["lidar"])
    t = time.perf_counter()
    all_v = np.vstack([g.outline.vertices for g in layout.rooms]) if layout.rooms else np.zeros((1, 2))
    ctx.origin = np.floor(all_v.min(axis=0) * 10) / 10  # plan origin at the plan's lower-left
    rooms, _ = build_rooms(layout, ctx)
    stitched = build_stitched(layout, rooms, ctx, res.drift_report,
                              "single continuous capture: rooms share one (drift-corrected) world frame")
    stages["assemble"] = time.perf_counter() - t
    regions, flags, scope_items = [], [], []
    if opts.damage:
        t = time.perf_counter()
        from groundplan.damage.pipeline import run_damage
        from groundplan.models.registry import set_cache

        set_cache(opts.use_cache)
        views = collect_views(cap, res.keyframes, res.T_wc_used)
        offsets = {f"R{k}": g.floor_y - layout.frame.floor_y for k, g in enumerate(layout.rooms, start=1)}
        dmg = run_damage(rooms, views, layout.frame, ctx, offsets, out_dir)
        _apply_room_types(rooms, dmg.room_types)
        regions, flags, scope_items = dmg.regions, dmg.flags, dmg.scope
        warnings += dmg.notes
        stages["damage"] = time.perf_counter() - t
    return capture, scale, rooms, stitched, regions, flags, scope_items


def _photo_branch(det, opts: RunOptions, out_dir, warnings: list[str], stages: dict):
    import math

    import numpy as np

    from groundplan.contract import DeviceInfo, RoomType, ScaleCue
    from groundplan.damage.pipeline import run_damage
    from groundplan.damage.rules import evaluate
    from groundplan.damage.scope import build_scope
    from groundplan.geometry.layout_solver import transform_points
    from groundplan.models.registry import set_cache
    from groundplan.stitching import relabel, room_type_from_name, stitch_rooms
    from groundplan.tiers.photo import photo_views, run_photos

    set_cache(opts.use_cache)
    t = time.perf_counter()
    recs = run_photos(det.root)
    stages["reconstruct"] = time.perf_counter() - t
    rooms, ctxs, north, local = [], {}, {}, {}
    cue_rows, sigmas = [], []
    for k, rec in enumerate(recs, start=1):
        warnings += rec.notes
        if rec.layout is None or not rec.layout.rooms:
            warnings.append(f"{rec.name}: no room outline could be built from its photos")
            continue
        rid = f"R{k}"
        ctx = AssembleContext(tier="photo", scale_sigma=rec.sigma_log, budget=BUDGETS["photo"])
        r_local, _ = build_rooms(rec.layout, ctx)
        room = relabel(r_local[0], rid)
        room.name = rec.name
        preset = room_type_from_name(rec.name)
        if preset:
            room.type = RoomType(label=preset, confidence=1.0, source="folder name")
        ctxs[rid] = ctx
        local[rid] = (rec, room)
        sigmas.append(rec.sigma_log)
        for c in rec.scale_info.cues:
            cue_rows.append(ScaleCue(name=f"{rec.name}:{c.name}", log_scale=c.log_scale, sigma_log=c.sigma_log,
                                     used=c.used, detail=c.detail))
        # compass: rotation of this room's plan frame relative to north-up
        rots = []
        for v, idx in zip(photo_views(rec), rec.poses.keys()):
            h = rec.geoms[idx].photo.heading_deg
            if h is None:
                continue
            d = rec.layout.frame.dir_to_plan(v.T_wc[:3, 2])
            phi = math.degrees(math.atan2(d[1], d[0]))
            rots.append(math.radians(90.0 - h - phi))
        north[rid] = math.degrees(math.atan2(np.mean(np.sin(rots)), np.mean(np.cos(rots)))) if rots else None
        rooms.append(room)
    if not rooms:
        raise RuntimeError("no room could be reconstructed from the photos")

    # damage per room in its own frame (wall regions live in wall coordinates and need no transform)
    regions, room_types = [], {}
    if opts.damage:
        t = time.perf_counter()
        for rid, (rec, room) in local.items():
            ctx = ctxs[rid]
            dmg = run_damage([room], photo_views(rec), rec.layout.frame, ctx,
                             {rid: rec.layout.rooms[0].floor_y - rec.layout.frame.floor_y}, out_dir,
                             room_presets={rid: room.type.label} if room.type.source == "folder name" else None)
            room_types.update(dmg.room_types)
            for reg in dmg.regions:
                regions.append((rid, reg))
            warnings += [n for n in dmg.notes if n not in warnings]
        stages["damage"] = time.perf_counter() - t
    _apply_room_types(rooms, room_types)

    t = time.perf_counter()
    placed, stitched, sres = stitch_rooms(rooms, north, ctxs,
                                          "photo tier: per-room folders joined by door pairing, compass and no overlap")
    stages["stitch"] = time.perf_counter() - t
    warnings += sres.notes
    final_regions = []
    for n, (rid, reg) in enumerate(regions, start=1):
        upd = {"id": f"D{n}"}
        if reg.surface_id.endswith(("-FLOOR", "-CEILING")):
            upd["polygon_uv"] = [tuple(map(float, q)) for q in transform_points(np.array(reg.polygon_uv),
                                                                               sres.placements[rid])]
        final_regions.append(reg.model_copy(update=upd))
    flags = evaluate(placed, final_regions, {r.id: {r.type.label} for r in placed})
    scope_items = build_scope(placed, final_regions, flags)

    first = recs[0].geoms[0].photo if recs and recs[0].geoms else None
    capture = Capture(id=det.root.name, tier="photo", source=str(det.root), app="iPhone Camera (stills)",
                      device=DeviceInfo(make=first.make, model=first.model, lens=first.lens) if first else None,
                      photos_per_room={r.name: len(r.geoms) for r in recs}, quality=CaptureQuality())
    scale = ScaleInfo(source="per-room fusion of model, ceiling, door and camera-height cues",
                      sigma_log=round(float(np.median(sigmas)), 4), cues=cue_rows)
    return capture, scale, placed, stitched, final_regions, flags, scope_items
