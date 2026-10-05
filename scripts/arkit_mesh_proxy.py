"""Head-to-head PROXY on the public stand-in: our LiDAR tier vs Apple's on-device ARKit mesh.

    python scripts/arkit_mesh_proxy.py [--data D] [--results benchmark/public/results/full_6b78d8d]

NOT the brief's head-to-head, which needs a named consumer app's export on our own rooms (no public
dataset has app exports, raw LiDAR and laser truth for the same rooms). ARKitScenes ships, for each
recording, the mesh ARKit itself reconstructed on the iPad: the on-device reconstruction consumer
LiDAR apps build on. Each mesh is measured with the same procedure that measured the laser truth
(``bench/arkitscenes.py``: room box around the walk, walls must reach the top of the observed room,
floor and ceiling as the extreme dense layers), and compared with our pipeline's numbers from the
benchmark run, against the laser truth, with the brief's rule: beat or tie (both within 5 mm).
Writes benchmark/public/h2h_proxy/{RESULT.md, dimensions.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundplan.bench import arkitscenes as A  # noqa: E402

TIE_M = 0.005
TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4",
         "uint": "<u4", "short": "<i2", "ushort": "<u2", "char": "i1"}


def mesh_vertices(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        head = b""
        while b"end_header" not in head:
            head += fh.readline()
    lines = head.decode(errors="replace").splitlines()
    n, props, in_vertex = 0, [], False
    for ln in lines:
        t = ln.split()
        if t and t[0] == "element":
            in_vertex = t[1] == "vertex"
            n = int(t[2]) if in_vertex else n
        elif t and t[0] == "property" and in_vertex and t[1] != "list":
            props.append((t[2], TYPES[t[1]]))
    v = np.memmap(path, dtype=np.dtype(props), mode="r", offset=len(head), shape=(n,))
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def voxel(xyz: np.ndarray, size: float = 0.02) -> np.ndarray:
    k = np.floor(xyz / size).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    return np.stack([np.bincount(inv, weights=xyz[:, d]) for d in range(3)], 1) / cnt[:, None]


def mesh_room(rec_dir: Path) -> dict | None:
    """Room box, and ceiling height when the mesh holds a ceiling, from the ARKit mesh (z up)."""
    V = voxel(mesh_vertices(rec_dir / f"{rec_dir.name}_3dod_mesh.ply"))
    _, T = A.read_traj(rec_dir / "lowres_wide.traj")  # y-up; the mesh is z-up
    c = np.median(T[:, :3, 3], axis=0)
    cam = np.array([c[0], -c[2], c[1]])
    sc = A.prepare_scene(V)
    horiz = np.abs(sc.normals[:, 2]) > 0.95
    near = horiz & (np.linalg.norm(V[:, :2] - cam[:2], axis=1) < 1.5)
    lv = A.levels(V[near, 2])
    if lv is None:
        return None
    floor = lv[0]
    above = horiz & (V[:, 2] > floor + 2.0)
    ceiling = A.levels(V[above, 2])[1] if above.sum() > 2000 else None  # a dense layer, not a few points
    vert = np.abs(sc.normals[:, 2]) < 0.15
    top = ceiling if ceiling is not None else float(np.percentile(V[vert, 2], 99.5))
    m = A.measure_room(sc, cam, floor, top)
    if m is None:
        return None
    x0, x1, y0, y1 = m["box"]
    return {"dims": sorted([x1 - x0, y1 - y0]), "ceiling": None if ceiling is None else ceiling - floor}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(os.environ.get("GROUNDPLAN_DATA", ROOT / "data")) / "arkitscenes"))
    ap.add_argument("--results", default=str(ROOT / "benchmark" / "public" / "results" / "full_6b78d8d"))
    a = ap.parse_args()
    import yaml

    data, res = Path(a.data), Path(a.results)
    man = yaml.safe_load((ROOT / "benchmark" / "public" / "manifest.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((res / "metrics.json").read_text(encoding="utf-8"))
    ours = {c["capture"]: c["items"] for c in metrics["captures"]}
    rows = []
    for cap in man["captures"]:
        if cap["tier"] != "lidar" or cap["id"] not in ours:
            continue
        visit, vid = Path(cap["path"]).name.split("_")
        mr = mesh_room(data / "raw" / visit / vid)
        items = [it for it in ours[cap["id"]] if it["kind"] in ("wall_length", "ceiling_height")]
        for it in items:
            if it["kind"] == "ceiling_height":
                app = None if mr is None else mr["ceiling"]
            else:  # the mesh box has no wall labels: take its extent closest to this wall's true length
                app = None if mr is None else min(mr["dims"], key=lambda d: abs(d - it["truth"]))
            e_ours = abs(it["value"] - it["truth"])
            e_app = None if app is None else abs(app - it["truth"])
            if e_app is None:
                outcome = "app n/a"
            elif e_ours <= TIE_M and e_app <= TIE_M:
                outcome = "tie"
            else:
                outcome = "beat" if e_ours < e_app else "lose"
            rows.append({"capture": cap["id"], "dimension": f"{it['room']}:{it['ref']}", "truth": it["truth"],
                         "ours": it["value"], "arkit_mesh": app, "err_ours_cm": 100 * (it["value"] - it["truth"]),
                         "err_mesh_cm": None if app is None else 100 * (app - it["truth"]), "outcome": outcome})
        print(cap["id"], "mesh", None if mr is None else [round(d, 3) for d in mr["dims"]],
              "ceiling", None if mr is None or mr["ceiling"] is None else round(mr["ceiling"], 3), flush=True)
    out = ROOT / "benchmark" / "public" / "h2h_proxy"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dimensions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shared = [r for r in rows if r["outcome"] != "app n/a"]
    won = sum(r["outcome"] in ("beat", "tie") for r in shared)
    lines = ["# Head-to-head PROXY: our LiDAR tier vs the on-device ARKit mesh (public stand-in)", "",
             "**Not the brief's head-to-head** (that needs a named consumer app's export on our own rooms). "
             "The comparison is Apple's on-device ARKit reconstruction shipped with each ARKitScenes recording, "
             "measured by the same procedure as the laser truth (`scripts/arkit_mesh_proxy.py`). "
             f"Tie: both errors within {TIE_M * 1000:.0f} mm.", "",
             f"**Beat or tie on {won}/{len(shared)} shared dimensions ({100 * won / max(len(shared), 1):.0f} %)**; "
             f"{len(rows) - len(shared)} dimension(s) the mesh could not give (no ceiling in it).", "",
             "| recording | dimension | truth (m) | ours (m) | ARKit mesh (m) | ours err (cm) | mesh err (cm) | outcome |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        app = "-" if r["arkit_mesh"] is None else f"{r['arkit_mesh']:.3f}"
        em = "-" if r["err_mesh_cm"] is None else f"{r['err_mesh_cm']:+.1f}"
        lines.append(f"| {r['capture']} | {r['dimension']} | {r['truth']:.3f} | {r['ours']:.3f} | {app} | "
                     f"{r['err_ours_cm']:+.1f} | {em} | {r['outcome']} |")
    lines += ["", "The mesh box has no wall labels, so each true wall is compared with the mesh extent closest "
              "to it; where a room is nearly square that choice favours the mesh, never us."]
    (out / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
