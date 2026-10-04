"""Staged damage on real footage: procedural decals projected onto a wall of the assessor capture.

Needs the sample captures: set GROUNDPLAN_SAMPLE_DATA to the folder holding single_room/.
"""

import os
from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path(os.environ.get("GROUNDPLAN_SAMPLE_DATA", Path(__file__).resolve().parents[2] / "SampleData"))
CAPTURE = SAMPLE / "single_room" / "c00a170fe1"

pytestmark = [pytest.mark.sample, pytest.mark.slow,
              pytest.mark.skipif(not (CAPTURE / "depth").is_dir(), reason="assessor sample capture not available")]


@pytest.fixture(scope="module")
def staged():
    pytest.importorskip("torch")
    from groundplan.assemble import AssembleContext, build_rooms
    from groundplan.config import BUDGETS
    from groundplan.damage.pipeline import run_damage, surfaces_for_room
    from groundplan.sim.decals import Decal, apply_decals
    from groundplan.tiers.lidar import LidarOptions, collect_views, run_lidar

    res = run_lidar(CAPTURE, LidarOptions(drift=False))
    res.scene = None
    layout = res.layout
    ctx = AssembleContext(tier="lidar", budget=BUDGETS["lidar"])
    ctx.origin = np.floor(np.vstack([g.outline.vertices for g in layout.rooms]).min(0) * 10) / 10
    rooms, _ = build_rooms(layout, ctx)
    views = collect_views(res.capture, res.keyframes, res.T_wc_used)
    target = max((w for r in rooms for w in r.walls if w.observed_fraction > 0.9), key=lambda w: w.length.value)
    room = next(r for r in rooms if r.id == target.room_id)
    offs = {f"R{k}": g.floor_y - layout.frame.floor_y for k, g in enumerate(layout.rooms, start=1)}
    surf = next(s for s in surfaces_for_room(room, layout.frame, ctx.origin, offs[room.id]) if s.id == target.id)
    L = target.length.value
    decals = [Decal("water_stain", 0.25 * L, 1.0, 0.36, 0.26, seed=1), Decal("crack", 0.6 * L, 1.2, 0.40, 0.12, seed=2)]
    dmg = run_damage(rooms, apply_decals(views, surf, decals), layout.frame, ctx, offs, None)
    return target, decals, dmg


def test_both_classes_found_with_extent_inside_interval(staged):
    target, decals, dmg = staged
    for d in decals:
        hits = [r for r in dmg.regions if r.surface_id == target.id and r.damage_class == d.kind]
        assert hits, f"staged {d.kind} not detected"
        r = hits[0]
        us = [p[0] for p in r.polygon_uv]
        assert abs(min(us) - d.u0) < 0.05
        assert r.width.lo <= d.width <= r.width.hi
        assert r.height.lo <= d.height <= r.height.hi


def test_no_detections_off_the_staged_wall(staged):
    target, _, dmg = staged
    assert [r.id for r in dmg.regions if r.surface_id != target.id] == []


def test_scope_keyed_to_the_damaged_wall(staged):
    target, _, dmg = staged
    codes = {s.code for s in dmg.scope if s.surface_id == target.id}
    assert {"PNT-SEAL", "DRY-CRACK", "PNT-WALL"} <= codes
