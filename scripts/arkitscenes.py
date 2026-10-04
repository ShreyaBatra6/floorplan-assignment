"""Public stand-in benchmark from ARKitScenes (see docs/PUBLIC_BENCHMARK.md).

    python scripts/arkitscenes.py fetch   --visits 466183 422378 471948 437450 [--data D]
    python scripts/arkitscenes.py convert [--data D]   recordings -> Stray Scanner folders (LiDAR tier)
    python scripts/arkitscenes.py truth   [--data D]   laser scans -> ground-truth YAML + review renders
    python scripts/arkitscenes.py tiers   [--data D]   a video clip and a photo folder per home
    python scripts/arkitscenes.py manifest [--data D]  benchmark/public/manifest.yaml

D defaults to $GROUNDPLAN_DATA/arkitscenes (or ./data/arkitscenes). Downloads come straight from
Apple's servers; ARKitScenes is licensed for non-commercial use (Apple's ARKitScenes license), so
the data is fetched, never redistributed with this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundplan.bench import arkitscenes as A  # noqa: E402

PUBLIC = ROOT / "benchmark" / "public"
DIAG_35MM = math.hypot(36.0, 24.0)


def data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return Path(os.environ.get("GROUNDPLAN_DATA", ROOT / "data")) / "arkitscenes"


def curl(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for _ in range(8):  # Apple's CDN drops long transfers now and then; resume where it stopped
        r = subprocess.run(["curl", "-sS", "-f", "-C", "-", "--retry", "5", "--retry-delay", "3", "-o", str(tmp), url])
        if r.returncode in (0, 33):  # 33: range not satisfiable, i.e. the partial file is complete
            break
    else:
        raise SystemExit(f"download failed after retries: {url}")
    tmp.replace(dst)


def metadata(d: Path) -> tuple[dict, dict]:
    curl(f"{A.BASE_URL}/raw/metadata.csv", d / "raw_metadata.csv")
    curl(f"{A.BASE_URL}/raw/laser_scanner_point_clouds/laser_scanner_point_clouds_mapping.csv", d / "laser_mapping.csv")
    videos = defaultdict(list)
    for r in csv.DictReader(open(d / "raw_metadata.csv")):
        videos[r["visit_id"]].append(r)
    scans = defaultdict(list)
    for r in csv.DictReader(open(d / "laser_mapping.csv")):
        scans[r["visit_id"]].append(r["laser_scanner_point_clouds_id"])
    return videos, scans


def visits(d: Path) -> list[str]:
    return json.loads((d / "selection.json").read_text())["visits"]


def recordings(d: Path, visit: str) -> list[Path]:
    return sorted(p for p in (d / "raw" / visit).iterdir() if p.is_dir() and p.name != "laser")


def cmd_fetch(a) -> None:
    d = data_dir(a.data)
    videos, scans = metadata(d)
    (d / "selection.json").parent.mkdir(parents=True, exist_ok=True)
    (d / "selection.json").write_text(json.dumps({"visits": a.visits}, indent=1))
    for v in a.visits:
        for r in videos[v]:
            vid, fold = r["video_id"], r["fold"]
            dst = d / "raw" / v / vid
            for f in A.LIDAR_ASSETS + [f"{vid}.mov"]:
                curl(f"{A.BASE_URL}/raw/{fold}/{vid}/{f}", dst / f)
                if f.endswith(".zip") and not (dst / f[:-4]).is_dir():
                    import zipfile

                    zipfile.ZipFile(dst / f).extractall(dst)
            print(f"  {v}/{vid} ok", flush=True)
        for sid in scans[v]:
            for ext in (".ply", "_pose.txt"):
                curl(f"{A.BASE_URL}/raw/laser_scanner_point_clouds/{v}/{sid}{ext}", d / "raw" / v / "laser" / f"{sid}{ext}")
        print(f"{v}: {len(videos[v])} recordings, {len(scans[v])} laser scans", flush=True)


def cmd_convert(a) -> None:
    d = data_dir(a.data)
    for visit in visits(d):
        for vdir in recordings(d, visit):
            out = d / "stray" / f"{visit}_{vdir.name}"
            if (out / "rgb.mp4").exists():
                continue
            print(A.convert_recording(vdir, out), flush=True)


def _review(visit: str, sc, rooms: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = sc.plan(sc.xyz[:, :2])
    vert = np.abs(sc.normals[:, 2]) < 0.15
    if rooms:  # a slice through the walls at 0.5-2.0 m above the floor
        z0 = rooms[0]["floor"]
        vert &= (sc.xyz[:, 2] > z0 + 0.5) & (sc.xyz[:, 2] < z0 + 2.0)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(P[vert][:, 0], P[vert][:, 1], s=0.3, c="0.35")
    for r in rooms:
        x0, x1, y0, y1 = r["box"]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c="tab:blue", lw=1.2)
        ax.text((x0 + x1) / 2, y0 - 0.15, f"{x1 - x0:.3f}", ha="center", va="top", color="tab:blue")
        ax.text(x1 + 0.1, (y0 + y1) / 2, f"{y1 - y0:.3f}", rotation=90, va="center", color="tab:blue")
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{r['name']}\nceiling {r['ceiling_height']:.3f}", ha="center", color="tab:blue")
        s = sc.plan(r["scanner_xy"][None])[0]
        ax.plot(*s, "rx", ms=10)
    if rooms:
        bx = np.array([r["box"] for r in rooms])
        ax.set_xlim(bx[:, 0].min() - 2.0, bx[:, 1].max() + 2.0)
        ax.set_ylim(bx[:, 2].min() - 2.0, bx[:, 3].max() + 2.0)
    ax.set_aspect("equal")
    ax.set_title(f"ARKitScenes visit {visit}: laser-measured rooms (blue), wall points (grey)")
    fig.savefig(out / f"{visit}_plan.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    for r in rooms:
        fig, axes = plt.subplots(1, 4, figsize=(20, 3.6))
        for ax, w in zip(axes, ("S", "E", "N", "W")):
            ras = r["rasters"][w]
            img = np.ones(ras["solid"].shape + (3,))
            img[ras["beyond"]] = (0.75, 0.85, 1.0)
            img[ras["solid"]] = (0.25, 0.25, 0.25)
            ax.imshow(img, origin="lower", extent=(0, ras["length"], 0, ras["height"]))
            for o in r["openings"][w]:
                s0, s1, b, t = o["rect"]
                ax.plot([s0, s1, s1, s0, s0], [b, b, t, t, b], c="tab:red", lw=1.5)
                ax.text((s0 + s1) / 2, t + 0.05, f"{o.get('id', '')} {o['kind']} {o['width']:.3f}", ha="center",
                        color="tab:red", fontsize=8)
            ax.set_title(f"{r['name']} wall {w} ({r['walls'][w]:.3f} m)", fontsize=9)
        fig.savefig(out / f"{visit}_{r['name']}_walls.png", dpi=70, bbox_inches="tight")
        plt.close(fig)


def cmd_truth(a) -> None:
    d = data_dir(a.data)
    gt_dir, rev_dir = PUBLIC / "ground_truth", PUBLIC / "review"
    gt_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)
    dec_file = PUBLIC / "review" / "decisions.yaml"
    decisions = yaml.safe_load(dec_file.read_text(encoding="utf-8")) or {} if dec_file.exists() else {}
    decisions = {str(k): v for k, v in decisions.items()}
    for visit in visits(d):
        if (gt_dir / f"ark_{visit}.yaml").exists() and not a.force:
            continue
        plys = sorted((d / "raw" / visit / "laser").glob("*.ply"))
        xyz, origin = A.fuse_laser(plys, voxel=0.02)
        scanners = np.array([np.loadtxt(p.with_name(p.stem + "_pose.txt"), delimiter=",").T[:3, 3] - origin
                             for p in plys])
        sc = A.prepare_scene(xyz.astype(np.float64))
        rooms = A.visit_truth(sc, scanners)
        doc = {"site": f"ark_{visit}", "instrument": "Faro Focus S70 laser scans from ARKitScenes, measured by "
               "scripts/arkitscenes.py truth (planes fitted to the scans; see the review renders)",
               "source_scans": [p.stem for p in plys], "rooms": [], "adjacency": []}
        review = decisions.get(str(visit), {})
        for k, r in enumerate(rooms):
            r["name"] = f"room{k + 1}"
            r["scanner_xy"] = scanners[r["scanner"], :2]
            dec = review.get(r["name"], {})
            if dec.get("exclude_room"):
                r["excluded"] = dec["exclude_room"]
                continue
            ops, n = [], 0
            for w in ("S", "E", "N", "W"):
                for o in r["openings"][w]:
                    n += 1
                    prefix = {"door": "D", "window": "N", "open_passage": "P"}[o["kind"]]
                    o["id"] = f"{prefix}{n}"
                    if o["id"] in dec.get("exclude_openings", []):
                        continue
                    ops.append({"id": o["id"], "wall": w, "kind": o["kind"], "width": o["width"],
                                "height": o["height"], "offset": o["offset"], "sill": o["sill"]})
            keep = [w for w in ("S", "E", "N", "W") if w not in dec.get("exclude_walls", [])]
            room = {"name": r["name"],
                    "walls": [{"id": w, "length": round(float(r["walls"][w]), 4)} for w in keep],
                    "ceiling": round(float(r["ceiling_height"]), 4)}
            if len(keep) == 4:
                room["floor_area"] = round(float(r["walls"]["S"] * r["walls"]["E"]), 3)
            room["openings"] = ops
            if dec.get("openings_assessed") is False:
                room["openings"], room["openings_assessed"] = [], False
            if dec.get("note"):
                room["review"] = dec["note"]
            doc["rooms"].append(room)
        for i, ri in enumerate(rooms):  # rooms whose boxes touch through a wall (<= 35 cm apart)
            for rj in rooms[i + 1:]:
                ax0, ax1, ay0, ay1 = ri["box"]
                bx0, bx1, by0, by1 = rj["box"]
                gap_x = max(bx0 - ax1, ax0 - bx1)
                gap_y = max(by0 - ay1, ay0 - by1)
                if (0 <= gap_x <= 0.35 and gap_y < -0.5) or (0 <= gap_y <= 0.35 and gap_x < -0.5):
                    doc["adjacency"].append([ri["name"], rj["name"]])
        header = (f"# PUBLIC STAND-IN: ground truth of ARKitScenes visit {visit}, measured on its laser scans\n"
                  f"# (not tape or laser on site). Review: benchmark/public/review/{visit}_*.png\n")
        (gt_dir / f"ark_{visit}.yaml").write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        _review(visit, sc, rooms, rev_dir)
        print(f"{visit}: {len(rooms)} room(s): " + "; ".join(
            f"{r['name']} {r['walls']['S']:.3f} x {r['walls']['E']:.3f} m, ceiling {r['ceiling_height']:.3f}, "
            f"{sum(len(v) for v in r['openings'].values())} opening(s)" for r in rooms), flush=True)


def _pick_photos(T: np.ndarray, n: int = 6) -> list[int]:
    """Frames that look like protocol photos: camera near level, views spread round the room."""
    fwd = T[:, :3, 2]  # y-up world after conversion
    level = np.abs(fwd[:, 1]) < 0.25
    yaw = np.degrees(np.arctan2(fwd[:, 0], -fwd[:, 2])) % 360
    picks = []
    for b in range(n):
        lo, hi = b * 360 / n, (b + 1) * 360 / n
        idx = np.flatnonzero(level & (yaw >= lo) & (yaw < hi))
        if len(idx):
            picks.append(int(idx[len(idx) // 2]))
    return picks


def cmd_tiers(a) -> None:
    """Per home: the first recording's video as the video tier, stills of the second as the photo tier."""
    import av
    from PIL import Image

    d = data_dir(a.data)
    for visit in visits(d):
        recs = recordings(d, visit)
        vdir = d / "tiers" / visit / "video"
        vdir.mkdir(parents=True, exist_ok=True)
        mov = recs[0] / f"{recs[0].name}.mov"
        dst = vdir / mov.name
        if not dst.exists():
            try:
                os.link(mov, dst)
            except OSError:
                import shutil

                shutil.copy2(mov, dst)
        src = recs[1] if len(recs) > 1 else recs[0]
        stray = d / "stray" / f"{visit}_{src.name}"
        pdir = d / "tiers" / visit / "photos" / "01 Room"
        if pdir.exists() and any(pdir.iterdir()):
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        from groundplan.io.stray import load_stray

        cap = load_stray(stray, require_depth=False)
        picks = _pick_photos(cap.T_wc)
        with av.open(str(stray / "rgb.mp4")) as c:
            for i, frame in enumerate(c.decode(video=0)):
                if i in picks:
                    img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    K = cap.K_rgb[i]
                    f35 = int(round(K[0, 0] * DIAG_35MM / math.hypot(*img.size)))
                    exif = img.getexif()
                    exif[271], exif[272] = "Apple", "iPad Pro 2020 (ARKitScenes)"
                    exif.get_ifd(0x8769)[41989] = f35
                    img.save(pdir / f"IMG_{i:04d}.jpg", quality=92, exif=exif)
        print(f"{visit}: video {mov.name}; {len(picks)} photos from {src.name}", flush=True)


def cmd_manifest(a) -> None:
    d = data_dir(a.data)
    # no damage truth exists for these homes, so the damage stage would only cost time
    man = {"sites": {}, "options": {"damage": False}, "captures": [], "drift_ablation": []}
    for visit in visits(d):
        site = f"ark_{visit}"
        gt = PUBLIC / "ground_truth" / f"{site}.yaml"
        if not gt.exists():
            continue
        man["sites"][site] = str(gt.relative_to(ROOT)).replace("\\", "/")
        first = None
        for k, vdir in enumerate(recordings(d, visit), start=1):
            cid = f"{site}_lidar_{k}"
            entry = {"id": cid, "site": site, "tier": "lidar", "path": f"arkitscenes/stray/{visit}_{vdir.name}"}
            if first:
                entry["repeat_of"] = first
            else:
                first = cid
                man["drift_ablation"].append(cid)
            man["captures"].append(entry)
        mov = next((d / "tiers" / visit / "video").glob("*.mov"), None)
        if mov:
            man["captures"].append({"id": f"{site}_video_1", "site": site, "tier": "video",
                                    "path": f"arkitscenes/tiers/{visit}/video/{mov.name}"})
        if (d / "tiers" / visit / "photos").exists():
            man["captures"].append({"id": f"{site}_photo_1", "site": site, "tier": "photo",
                                    "path": f"arkitscenes/tiers/{visit}/photos"})
    header = ("# PUBLIC STAND-IN benchmark: ARKitScenes homes (iPad Pro 2020) with laser-scan ground truth.\n"
              "# Raw data: python scripts/arkitscenes.py fetch/convert/tiers; set GROUNDPLAN_DATA to the folder\n"
              "# that contains arkitscenes/. Not the case study's own captures (see docs/PUBLIC_BENCHMARK.md).\n")
    (PUBLIC / "manifest.yaml").write_text(header + yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    print(f"{len(man['captures'])} captures over {len(man['sites'])} homes -> {PUBLIC / 'manifest.yaml'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"fetch": cmd_fetch, "convert": cmd_convert, "truth": cmd_truth, "tiers": cmd_tiers, "manifest": cmd_manifest}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("--data", default=None)
        if name == "fetch":
            p.add_argument("--visits", nargs="+", required=True)
        if name == "truth":
            p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cmds[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
