"""End-to-end LiDAR tier on a simulated flat with exact ground truth."""

import json

import pytest

from groundplan.bench.gates import score_capture
from groundplan.bench.gt import from_sim
from groundplan.pipeline import RunOptions, run_capture
from groundplan.sim.capture import SimOptions, simulate_capture
from groundplan.sim.scene import FlatSpec, OpeningSpec, RoomSpec


def two_rooms() -> FlatSpec:
    t = 0.12
    a = RoomSpec("a", 0.0, 0.0, 3.40, 2.90)
    b = RoomSpec("b", 3.40 + t, 0.0, 3.40 + t + 2.60, 2.90)
    return FlatSpec(rooms=[a, b], wall_t=t, ceiling=2.55,
                    openings=[OpeningSpec("a", "E", 1.0, 0.84),
                              OpeningSpec("a", "N", 1.0, 1.2, kind="window", height=1.2, sill=0.9)])


@pytest.fixture(scope="module")
def sim_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    gt = simulate_capture(two_rooms(), root / "cap", SimOptions(seed=5, fps=3.0))
    res = run_capture(root / "cap", root / "out", RunOptions(drift=False, damage=False, render=False))
    return gt, res.plan


@pytest.mark.slow
def test_walls_ceiling_area(sim_run):
    gt, plan = sim_run
    sc = score_capture(plan, from_sim(gt), "sim")
    assert sc.rooms_matched == 2 and sc.rooms_pred == 2
    walls = sc.by_kind("wall_length")
    assert len(walls) == 8
    assert max(abs(i.err) for i in walls) < 0.01
    assert all(abs(i.err) < 0.005 for i in sc.by_kind("ceiling_height"))
    assert all(abs(i.rel) < 0.01 for i in sc.by_kind("floor_area"))
    assert all(i.covered for i in walls)


@pytest.mark.slow
def test_adjacency_and_openings(sim_run):
    gt, plan = sim_run
    sc = score_capture(plan, from_sim(gt), "sim")
    assert sc.adjacency_found == sc.adjacency_gt == 1
    assert sc.overlaps == 0
    assert sc.openings_missed == 0 and sc.openings_phantom == 0
    assert all(abs(i.err) < 0.06 for i in sc.by_kind("opening_width"))


@pytest.mark.slow
def test_plan_is_schema_valid(sim_run):
    from groundplan.validate import validate_document

    _, plan = sim_run
    assert validate_document(json.loads(plan.model_dump_json())) == []
