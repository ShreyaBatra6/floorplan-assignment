"""A complete synthetic benchmark (captures + exact ground truth + manifest) for demonstrating and
testing the benchmark and fix-loop tooling before real captures exist.

    python scripts/make_sim_benchmark.py <out dir>

Creates, in <out dir>:
    data/sim_a_lidar_1/          simulated Stray Scanner capture of the demo flat (sensor noise only)
    data/sim_a_lidar_2/          the same flat again, with incremental pose drift (repeat + drift case)
    ground_truth/sim_a.yaml      exact measurements, in the same format a person fills in on site
    manifest.yaml                both captures, a repeatability pair and a drift ablation

Then: GROUNDPLAN_DATA=<out>/data groundplan bench run --manifest <out>/manifest.yaml --out <out>/results
Every number this produces is synthetic and must never be reported as benchmark accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from groundplan.bench.gt import sim_truth_doc
from groundplan.sim.capture import SimOptions, simulate_capture
from groundplan.sim.scene import demo_flat

CAPTURES = {
    "sim_a_lidar_1": SimOptions(seed=1),
    "sim_a_lidar_2": SimOptions(seed=3, drift_pos_per_m=0.01, drift_yaw_deg_per_m=0.4),
}


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sim_benchmark").resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(exist_ok=True)
    gt = None
    for cid, opts in CAPTURES.items():
        cap = out / "data" / cid
        if (cap / "ground_truth.json").exists():
            print(f"reusing {cap}")
            gt = json.loads((cap / "ground_truth.json").read_text(encoding="utf-8"))
        else:
            print(f"simulating {cid} ...")
            gt = simulate_capture(demo_flat(), cap, opts)
    gt_path = out / "ground_truth" / "sim_a.yaml"
    gt_path.write_text("# SYNTHETIC: exact truth of the simulated flat (not a measurement)\n"
                       + yaml.safe_dump(sim_truth_doc(gt, "sim_a"), sort_keys=False), encoding="utf-8")
    manifest = {
        "sites": {"sim_a": str(gt_path)},
        "captures": [
            {"id": "sim_a_lidar_1", "site": "sim_a", "tier": "lidar", "path": "sim_a_lidar_1"},
            {"id": "sim_a_lidar_2", "site": "sim_a", "tier": "lidar", "path": "sim_a_lidar_2",
             "repeat_of": "sim_a_lidar_1"},
        ],
        "drift_ablation": ["sim_a_lidar_2"],
    }
    (out / "manifest.yaml").write_text("# SYNTHETIC benchmark (demonstration of the tooling)\n"
                                       + yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"ok: {out / 'manifest.yaml'}\n  set GROUNDPLAN_DATA={out / 'data'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
