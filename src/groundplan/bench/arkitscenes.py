"""ARKitScenes (Apple, 2021) as a public stand-in benchmark: iPad Pro LiDAR recordings of real homes
with a survey-grade Faro laser scan of each home.

Used where the case study's own captures are not available yet. Everything here is clearly a
proxy: the device is a 2020 iPad Pro (LiDAR, 1920x1440 wide camera), not an iPhone 15, the
recordings follow Apple's capture procedure rather than ours, and the ground truth is measured on
the laser scans (``laser_truth``), not with a tape on site.

Data facts this module relies on (checked on visit 421063; see docs/PUBLIC_BENCHMARK.md):
* ``lowres_wide.traj``: one line per pose at 10 Hz, ``t rx ry rz tx ty tz`` = world-to-camera
  (axis-angle, metres); inverted it maps OpenCV-convention camera points into a z-up world.
* ``lowres_depth/<vid>_<t>.png``: uint16 mm, 256x192, 60 Hz; ``confidence/`` the same, 0-2;
  ``lowres_wide_intrinsics/<vid>_<t>.pincam``: ``w h fx fy cx cy`` at 256x192.
* ``<vid>.mov``: 1920x1440 HEVC at 60 Hz; colour frame j of ``lowres_wide`` is video frame j-1.
* Laser ``<id>.ply``: binary, double x y z (metres, already registered across the visit's scans,
  z up with a site offset), uchar rgba, double quality, radius; ``<id>_pose.txt`` is the scanner
  pose (translation in the last row).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
LIDAR_ASSETS = ["lowres_wide.traj", "lowres_wide_intrinsics.zip", "lowres_depth.zip", "confidence.zip",
                "lowres_wide.zip"]
Z_UP_TO_Y_UP = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])  # (x, y, z) -> (x, z, -y)
PLY_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1"),
                      ("a", "u1"), ("q", "<f8"), ("rad", "<f8")])


def read_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(timestamps, T_wc) with T_wc mapping OpenCV camera points into a y-up world."""
    stamps, poses = [], []
    for line in Path(path).read_text().split("\n"):
        t = line.split()
        if len(t) != 7:
            continue
        E = np.eye(4)
        E[:3, :3] = Rotation.from_rotvec([float(v) for v in t[1:4]]).as_matrix()
        E[:3, 3] = [float(v) for v in t[4:7]]
        stamps.append(float(t[0]))
        poses.append(Z_UP_TO_Y_UP @ np.linalg.inv(E))
    return np.array(stamps), np.stack(poses)


def _stamp(p: Path) -> float:
    return float(p.stem.rsplit("_", 1)[1])


def convert_recording(video_dir: Path, out_dir: Path, tolerance: float = 0.005) -> dict:
    """One ARKitScenes recording -> a Stray Scanner folder (rgb.mp4, depth/, confidence/,
    odometry.csv, camera_matrix.csv), keeping the depth frames that have a pose within ``tolerance``
    seconds (as Apple's own loader does): about 10 frames per second."""
    import av

    video_dir, out_dir = Path(video_dir), Path(out_dir)
    vid = video_dir.name
    stamps, T = read_traj(video_dir / "lowres_wide.traj")
    depth = {round(_stamp(p), 3): p for p in (video_dir / "lowres_depth").glob("*.png")}
    conf = {round(_stamp(p), 3): p for p in (video_dir / "confidence").glob("*.png")}
    intr = {round(_stamp(p), 3): p for p in (video_dir / "lowres_wide_intrinsics").glob("*.pincam")}
    colour = sorted((video_dir / "lowres_wide").glob("*.png"), key=_stamp)
    colour_index = {round(_stamp(p), 3): j for j, p in enumerate(colour)}
    keys = np.array(sorted(depth))
    rows = []
    for t, P in zip(stamps, T):
        k = keys[np.argmin(np.abs(keys - t))]
        if abs(k - t) <= tolerance and k in colour_index and k in intr:
            rows.append((float(k), P, depth[k], conf.get(k), intr[k], colour_index[k] - 1))
    rows = [r for r in rows if r[5] >= 0]
    if len(rows) < 30:
        raise RuntimeError(f"{vid}: only {len(rows)} frames with depth, pose and colour")
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence").mkdir(exist_ok=True)
    s = 1920 / 256  # colour comes from the 1920x1440 video; intrinsics are given at 256x192
    with open(out_dir / "odometry.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"])
        for i, (t, P, dp, cp, ip, _) in enumerate(rows):
            _, _, fx, fy, cx, cy = np.loadtxt(ip)
            q = Rotation.from_matrix(P[:3, :3]).as_quat()
            w.writerow([f"{t:.6f}", i, *(f"{v:.6f}" for v in P[:3, 3]), *(f"{v:.8f}" for v in q),
                        f"{fx * s:.4f}", f"{fy * s:.4f}", f"{(cx + 0.5) * s - 0.5:.4f}", f"{(cy + 0.5) * s - 0.5:.4f}"])
            (out_dir / "depth" / f"{i:06d}.png").write_bytes(dp.read_bytes())
            if cp is not None:
                (out_dir / "confidence" / f"{i:06d}.png").write_bytes(cp.read_bytes())
    _, _, fx, fy, cx, cy = np.loadtxt(rows[0][4])
    K = np.array([[fx * s, 0, (cx + 0.5) * s - 0.5], [0, fy * s, (cy + 0.5) * s - 0.5], [0, 0, 1]])
    np.savetxt(out_dir / "camera_matrix.csv", K, delimiter=",", fmt="%.4f")
    # one video frame per odometry row, from the full-resolution recording
    wanted = {r[5]: i for i, r in enumerate(rows)}
    src = av.open(str(video_dir / f"{vid}.mov"))
    src.streams.video[0].thread_type = "AUTO"
    dst = av.open(str(out_dir / "rgb.mp4"), "w")
    stream = dst.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = 1920, 1440, "yuv420p"
    stream.options = {"crf": "20", "preset": "veryfast"}
    written = 0
    for j, frame in enumerate(src.decode(video=0)):
        if j in wanted:
            img = frame.to_ndarray(format="rgb24")
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                dst.mux(pkt)
            written += 1
        if j > max(wanted):
            break
    for pkt in stream.encode():
        dst.mux(pkt)
    dst.close()
    src.close()
    if written != len(rows):
        raise RuntimeError(f"{vid}: wrote {written} video frames for {len(rows)} odometry rows")
    return {"video_id": vid, "frames": len(rows), "duration_s": round(rows[-1][0] - rows[0][0], 1)}


# ---------------------------------------------------------------- laser scans -> fused 1 cm cloud

def ply_points(path: Path) -> np.memmap:
    with open(path, "rb") as fh:
        head = b""
        while b"end_header" not in head:
            head += fh.readline()
    lines = head.decode(errors="replace").splitlines()
    n = int(next(ln for ln in lines if ln.startswith("element vertex")).split()[-1])
    props = [ln.split()[-1] for ln in lines if ln.startswith("property") and "list" not in ln]
    if props[:3] != ["x", "y", "z"] or len(props) != len(PLY_DTYPE.names):
        raise ValueError(f"{path}: unexpected PLY layout {props}")
    return np.memmap(path, dtype=PLY_DTYPE, mode="r", offset=len(head), shape=(n,))


def fuse_laser(paths: list[Path], voxel: float = 0.01, chunk: int = 4_000_000,
               max_range: float | None = None, centre: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-averaged laser cloud of several registered scans (streamed in chunks; a scan is ~2 GB).

    Returns (xyz in metres relative to ``origin``, origin). The site offset in z is removed by
    subtracting the first chunk's median position, so float32 keeps sub-millimetre precision."""
    origin = None
    keys_all, sums_all, counts_all = [], [], []
    for path in paths:
        pts = ply_points(Path(path))
        for a in range(0, len(pts), chunk):
            c = pts[a:a + chunk]
            xyz = np.stack([c["x"], c["y"], c["z"]], 1)
            if origin is None:
                origin = np.median(xyz, axis=0).round(2)
            xyz = xyz - origin
            if max_range is not None and centre is not None:
                xyz = xyz[np.linalg.norm(xyz[:, :2] - centre[:2], axis=1) < max_range]
            k = np.floor(xyz / voxel).astype(np.int64) + 2**20
            lin = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
            u, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
            keys_all.append(u)
            sums_all.append(np.stack([np.bincount(inv, weights=xyz[:, d], minlength=len(u)) for d in range(3)], 1))
            counts_all.append(cnt)
    keys = np.concatenate(keys_all)
    u, inv = np.unique(keys, return_inverse=True)
    sums = np.stack([np.bincount(inv, weights=np.concatenate(sums_all)[:, d], minlength=len(u)) for d in range(3)], 1)
    cnt = np.bincount(inv, weights=np.concatenate(counts_all), minlength=len(u))
    return (sums / cnt[:, None]).astype(np.float32), origin


# ---------------------------------------------------------------- ground truth from the laser cloud
#
# Deliberately simple and separate from the pipeline it judges (no shared geometry code): planes
# are found by histogramming survey-grade points, a room is the box of the four nearest
# full-height walls around a laser scanner position, and openings are holes in a wall plane
# through which the laser saw beyond it. Every value is drawn on a review render.

@dataclass
class WallPlane:
    axis: int  # 0: plane x = offset, 1: plane y = offset (room frame, after the Manhattan yaw)
    offset: float
    inward: int  # +1 if the room lies on the +axis side
    support: int


@dataclass
class LaserScene:
    xyz: np.ndarray  # fused laser points (m), z up
    normals: np.ndarray
    yaw: float  # Manhattan yaw: plan = R @ xy
    R: np.ndarray

    def plan(self, xy: np.ndarray) -> np.ndarray:
        return xy @ self.R.T


def _normals(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(xyz).query(xyz, k=k, workers=-1)
    out = np.empty_like(xyz)
    step = 200_000
    for a in range(0, len(xyz), step):  # bounded memory
        nb = xyz[idx[a:a + step]]
        nb = nb - nb.mean(axis=1, keepdims=True)
        _, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
        out[a:a + step] = vec[:, :, 0]
    return out


def prepare_scene(xyz: np.ndarray) -> LaserScene:
    n = _normals(xyz)
    vert = np.abs(n[:, 2]) < 0.15
    ang = np.mod(np.arctan2(n[vert, 1], n[vert, 0]), np.pi / 2)
    h, e = np.histogram(ang, bins=180, range=(0, np.pi / 2))
    k = int(np.argmax(np.convolve(np.r_[h[-3:], h, h[:3]], np.ones(7), "valid")))
    yaw = float((e[k] + e[k + 1]) / 2)
    R = np.array([[math.cos(-yaw), -math.sin(-yaw)], [math.sin(-yaw), math.cos(-yaw)]])
    return LaserScene(xyz, n, yaw, R)


def levels(z: np.ndarray) -> tuple[float, float] | None:
    """Lowest and highest dense horizontal layers (floor, ceiling)."""
    if len(z) < 500:
        return None
    hist, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.01, 0.01))
    c = (e[:-1] + e[1:]) / 2
    strong = np.flatnonzero(hist > 0.15 * hist.max())
    out = []
    for near in (c[strong[0]], c[strong[-1]]):
        out.append(float(np.median(z[np.abs(z - near) < 0.02])))
    return out[0], out[1]


def measure_room(sc: LaserScene, scanner: np.ndarray, floor: float, ceiling: float) -> dict | None:
    """The box of the nearest wall in each of the four directions around ``scanner``.

    A wall plane reaches to within 25 cm of the ceiling (a cornice may hide the last few cm;
    furniture, however tall, stops lower), runs for at least 80 cm and lies across the scanner's
    line of sight. Of the planes within 50 cm of the nearest one, the best supported is the wall
    surface (panelling or a sill in front of it is smaller). If another plane with at least 30 % of
    its support sits more than 15 cm from it (alcoves beside a chimney breast, a recess), the room
    is not a box and is reported as irregular instead of being forced into one."""
    height = ceiling - floor
    n, xyz = sc.normals, sc.xyz
    vert = (np.abs(n[:, 2]) < 0.15) & (xyz[:, 2] > floor + 0.15) & (xyz[:, 2] < ceiling - 0.02)
    P, N, Z = sc.plan(xyz[vert, :2]), sc.plan(n[vert, :2]), xyz[vert, 2]
    s = sc.plan(scanner[None, :2])[0]
    walls, irregular = {}, []
    for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
        other = 1 - axis
        facing = np.abs(N[:, axis]) > 0.9
        coord = P[:, axis]
        bins = np.arange(s[axis] - 10, s[axis] + 10, 0.01)
        hh, _ = np.histogram(coord[facing], bins=bins)
        planes = []
        for b in np.argsort(np.abs(bins[:-1] - s[axis])):
            dist = (bins[b] - s[axis]) * sign
            if hh[b] < 60 or dist <= 0.25:
                continue
            if planes and dist > planes[0][0] + 0.5:
                break
            if any(abs(dist - q[0]) < 0.06 for q in planes):
                continue
            on = facing & (np.abs(coord - (bins[b] + 0.005)) < 0.02)
            if on.sum() < 150:
                continue
            lo, hi = np.percentile(P[on, other], [2, 98])
            top = np.percentile(Z[on], 99)
            if top > ceiling - 0.25 and hi - lo > 0.8 and lo - 0.3 < s[other] < hi + 0.3:
                planes.append((dist, WallPlane(axis, float(np.median(coord[on])), -sign, int(on.sum()))))
        if not planes:
            return None
        best = max(planes, key=lambda q: q[1].support)  # the wall surface itself, not panelling or a sill
        walls[(axis, sign)] = best[1]
        if any(q[1].support > 0.3 * best[1].support and abs(q[0] - best[0]) > 0.15 for q in planes):
            irregular.append(("x", "y")[axis] + ("-" if sign < 0 else "+"))
    box = (walls[(0, -1)].offset, walls[(0, 1)].offset, walls[(1, -1)].offset, walls[(1, 1)].offset)
    return {"box": box, "walls": walls, "irregular": irregular}


WALL_SPEC = {  # wall -> (plane axis, which box offset, outward sign, start, end along the other axis)
    "S": (1, 2, -1, 0, 1), "E": (0, 1, +1, 2, 3), "N": (1, 3, +1, 1, 0), "W": (0, 0, -1, 3, 2),
}


def wall_openings(sc: LaserScene, box: tuple, wall: str, floor: float, ceiling: float,
                  res: float = 0.01) -> tuple[list[dict], dict]:
    """Holes in one wall plane through which the laser saw beyond it (doors, windows, passages).

    Wall coordinates: ``s`` from the wall's start corner (counter-clockwise walk), ``h`` above the
    floor. A hole without points beyond it is a laser shadow (furniture in front), not an opening."""
    from scipy import ndimage

    axis, oi, out_sign, ai, bi = WALL_SPEC[wall]
    off, a, b = box[oi], box[ai], box[bi]
    other = 1 - axis
    P = sc.plan(sc.xyz[:, :2])
    Z = sc.xyz[:, 2] - floor
    d = (P[:, axis] - off) * out_sign  # distance beyond the plane (positive = outside the room)
    along = (P[:, other] - a) * (1 if b > a else -1)
    length, height = abs(b - a), ceiling - floor
    inside = (along > 0) & (along < length) & (Z > 0.03) & (Z < height - 0.03)
    plane_pts = inside & (np.abs(d) < 0.03)
    beyond = inside & (d > 0.08) & (d < 4.0)
    ns, nh = int(length / res) + 1, int(height / res) + 1
    solid = np.zeros((nh, ns), bool)
    solid[(Z[plane_pts] / res).astype(int), (along[plane_pts] / res).astype(int)] = True
    seen_beyond = np.zeros((nh, ns), bool)
    seen_beyond[(Z[beyond] / res).astype(int), (along[beyond] / res).astype(int)] = True
    solid_f = ndimage.binary_closing(solid, np.ones((5, 5)))  # 2 cm voxels leave pin holes at 1 cm
    # candidates: where the laser saw beyond the plane, closed across window bars and mullions
    through = ndimage.binary_closing(seen_beyond, np.ones((9, 9))) & ~solid_f
    through = ndimage.binary_closing(through, np.ones((11, 11)))  # bridges bars up to ~10 cm, never past a jamb
    lab, k = ndimage.label(through)
    found = []
    for i in range(1, k + 1):
        comp = lab == i
        rows, cols = np.flatnonzero(comp.any(1)), np.flatnonzero(comp.any(0))
        w_cells, h_cells = cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1
        if w_cells * res < 0.4 or h_cells * res < 0.5 or w_cells * res > 3.0:
            continue
        if cols[0] * res < 0.05 or (ns - 1 - cols[-1]) * res < 0.05:
            continue  # runs into a corner: a recess or the room continuing, not an opening in this wall
        bottom, top = rows[0] * res, (rows[-1] + 1) * res
        # jambs: per row of the middle half, where the wall plane resumes beside the opening
        lefts, rights = [], []
        reach = int(0.15 / res)
        for r in range(rows[0] + int(0.25 * h_cells), rows[-1] + 1 - int(0.25 * h_cells)):
            on = np.flatnonzero(comp[r])
            if not len(on):
                continue
            left = np.flatnonzero(solid_f[r, max(on[0] - reach, 0):on[0] + 1])
            right = np.flatnonzero(solid_f[r, on[-1]:on[-1] + reach + 1])
            if len(left) and len(right):
                lefts.append(max(on[0] - reach, 0) + left[-1] + 1)
                rights.append(on[-1] + right[0])
        if len(lefts) < 10:
            continue
        s0, s1 = float(np.median(lefts)) * res, float(np.median(rights)) * res
        door, w = bottom < 0.06, s1 - s0
        if door and 1.8 <= top <= 2.6 and 0.55 <= w <= 1.25:
            kind = "door"
        elif door and top >= 1.9 and 1.25 < w <= 3.0:
            kind = "open_passage"
        elif not door and 0.3 <= bottom <= 1.6 and top - bottom >= 0.4:
            kind = "window"
        else:
            continue  # a fireplace, a niche, a low cabinet seen into: not an opening
        found.append({"kind": kind, "width": round(float(s1 - s0), 3), "offset": round(float(s0), 3),
                      "height": round(float(top - (0.0 if door else bottom)), 3),
                      "sill": None if door else round(float(bottom), 3), "rect": (s0, s1, bottom, top)})
    return found, {"solid": solid_f, "beyond": seen_beyond, "res": res, "length": length, "height": height}


def visit_truth(sc: LaserScene, scanners: np.ndarray) -> list[dict]:
    """One measured room per laser scanner position (scanners in the same room give one room)."""
    rooms, skipped = [], []
    P = sc.plan(sc.xyz[:, :2])
    horiz = np.abs(sc.normals[:, 2]) > 0.95
    for k, scan in enumerate(scanners):
        s = sc.plan(scan[None, :2])[0]
        near = horiz & (np.abs(P[:, 0] - s[0]) < 1.0) & (np.abs(P[:, 1] - s[1]) < 1.0)
        lv = levels(sc.xyz[near, 2])
        if lv is None:
            continue
        floor, ceiling = lv
        m = measure_room(sc, scan, floor, ceiling)
        if m is None:
            continue
        x0, x1, y0, y1 = m["box"]
        if m["irregular"]:
            skipped.append({"scanner": k, "box": m["box"], "irregular": m["irregular"]})
            continue
        if any(abs(x0 - r["box"][0]) < 0.05 and abs(x1 - r["box"][1]) < 0.05 and abs(y0 - r["box"][2]) < 0.05
               and abs(y1 - r["box"][3]) < 0.05 for r in rooms):
            continue  # same room seen from another scanner position
        # floor and ceiling over the room's own footprint (inset 30 cm)
        inner = horiz & (P[:, 0] > x0 + 0.3) & (P[:, 0] < x1 - 0.3) & (P[:, 1] > y0 + 0.3) & (P[:, 1] < y1 - 0.3)
        lv = levels(sc.xyz[inner, 2]) or (floor, ceiling)
        floor, ceiling = lv
        walls = {"S": x1 - x0, "E": y1 - y0, "N": x1 - x0, "W": y1 - y0}
        openings, rasters = {}, {}
        for w in walls:
            openings[w], rasters[w] = wall_openings(sc, m["box"], w, floor, ceiling)
        rooms.append({"scanner": k, "box": m["box"], "floor": floor, "ceiling": ceiling,
                      "ceiling_height": ceiling - floor, "walls": walls, "openings": openings, "rasters": rasters})
    if skipped and not rooms:
        print(f"  no box-shaped room: walls with a recess on {[x['irregular'] for x in skipped]}")
    return rooms
