"""Development inputs for the photo and video tiers, derived from a Stray Scanner (LiDAR) capture.

Not a substitute for real captures (the benchmark uses real iPhone photos and videos taken with
the capture protocol); it lets the photo/video paths be developed and checked against the LiDAR
result of the very same rooms.

* photos/<room>/<frame>.jpg: upright frames chosen per room (camera inside the room per the LiDAR
  plan, roughly level, spread over headings), with EXIF focal length (35 mm equivalent, from the
  capture's intrinsics) and a compass heading (pose heading + a global offset + noise, like a phone
  compass indoors);
* video/walkthrough.mp4: the capture's video, upright, at a reduced frame rate.

Usage: python scripts/make_dev_tiers.py <stray capture> <out dir> [--photos-per-room 5] [--fps 6]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from groundplan.io.stray import load_stray
from groundplan.tiers.lidar import LidarOptions, run_lidar


def upright_k(T_wc: np.ndarray) -> int:
    up_c = T_wc[:3, :3].T @ np.array([0, 1.0, 0])
    return int(np.round(np.degrees(np.arctan2(up_c[0], -up_c[1])) / 90.0)) % 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--photos-per-room", type=int, default=5)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    cap = load_stray(Path(a.capture), require_depth=False)
    res = run_lidar(Path(a.capture), LidarOptions(drift=False))
    layout = res.layout
    from matplotlib.path import Path as MplPath

    # which room is each frame's camera in, and is the view roughly level?
    cams_xy = layout.frame.to_plan(cap.T_wc[:, :3, 3])
    fwd = cap.T_wc[:, :3, 2]
    level = np.abs(fwd[:, 1]) < math.sin(math.radians(38))
    heading = np.degrees(np.arctan2(fwd[:, 0], -fwd[:, 2]))  # world-frame heading
    compass_offset = rng.uniform(0, 360)
    chosen: dict[str, list[int]] = {}
    sectors: dict[str, list[np.ndarray]] = {}
    for k, room in enumerate(layout.rooms, start=1):
        inside = MplPath(room.outline.vertices).contains_points(cams_xy) & level & cap.has_depth
        idx = np.flatnonzero(inside)
        if len(idx) < a.photos_per_room:
            continue
        bins = np.linspace(-180, 180, a.photos_per_room + 1)
        sectors[f"{k:02d} Room {k}"] = [
            sel[np.linspace(0, len(sel) - 1, min(len(sel), 12)).round().astype(int)]
            for b0, b1 in zip(bins[:-1], bins[1:])
            if len(sel := idx[(heading[idx] >= b0) & (heading[idx] < b1)])
        ]
    # one decoding pass for every candidate, then the sharpest frame per heading sector
    allc = sorted({int(i) for v in sectors.values() for c in v for i in c})
    sharp = {fi: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
             for fi, im in cap.rgb_frames(allc, max_side=640)}
    for room, cands in sectors.items():
        chosen[room] = sorted({int(max(c, key=lambda i: sharp.get(int(i), 0))) for c in cands})
    frames = {i for v in chosen.values() for i in v}
    imgs = dict(cap.rgb_frames(frames))
    for room, idx in chosen.items():
        d = out / "photos" / room
        d.mkdir(parents=True, exist_ok=True)
        for i in idx:
            k = upright_k(cap.T_wc[i])
            img = np.rot90(imgs[i], k=k).copy()
            h, w = img.shape[:2]
            fx = float(cap.K_rgb[i][0, 0])
            f35 = fx * math.hypot(36, 24) / math.hypot(w, h)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            exif = Image.Exif()
            exif[0x010F] = "Apple"
            exif[0x0110] = "derived from Stray Scanner frame (dev)"
            ifd = exif.get_ifd(0x8769)
            ifd[41989] = int(round(f35))
            ifd[36867] = "2026:10:04 12:%02d:%02d" % ((i // 60) % 60, i % 60)
            gps = exif.get_ifd(0x8825)
            gps[16] = "M"
            gps[17] = float((heading[i] + compass_offset + rng.normal(0, 8)) % 360)
            pil.save(d / f"frame_{i:06d}.jpg", quality=92, exif=exif)
    # upright, reduced-rate video
    if cap.video_path is not None:
        step = max(int(round((cap.n / max(cap.duration_s, 1e-6)) / a.fps)), 1)
        idx = list(range(0, cap.n, step))
        vid_dir = out / "video"
        vid_dir.mkdir(parents=True, exist_ok=True)
        k = upright_k(cap.T_wc[len(cap.T_wc) // 2])
        writer = None
        for i, img in cap.rgb_frames(idx, max_side=1280):
            img = np.rot90(img, k=k).copy()
            if writer is None:
                writer = cv2.VideoWriter(str(vid_dir / "walkthrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
        if writer is not None:
            writer.release()
    print(f"photos: {', '.join(f'{r} ({len(v)})' for r, v in chosen.items())}; video at {a.fps} fps")


if __name__ == "__main__":
    main()
