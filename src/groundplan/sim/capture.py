"""Render a synthetic flat into a Stray Scanner capture (depth, confidence, odometry).

The depth sensor is simulated at 256x192 with iPhone-like intrinsics and noise; the recorded poses
can carry injected drift (the depth is rendered from the true poses, the odometry written with
the drifted ones), and the depth can carry a scale/offset bias. Ground truth is written next to
the capture as ``ground_truth.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from groundplan.sim.scene import FlatSpec, build_boxes, ground_truth

RGB_W, RGB_H = 1920, 1440
DEPTH_W, DEPTH_H = 256, 192
FX_RGB = 1600.0


@dataclass
class SimOptions:
    seed: int = 0
    noise_rel: float = 0.003  # depth noise sigma = noise_abs + noise_rel * depth
    noise_abs: float = 0.001
    depth_scale_bias: float = 0.0  # measured = true * (1 + scale_bias) + offset_bias
    depth_offset_bias: float = 0.0
    drift_pos_per_m: float = 0.0  # random-walk position drift, metres per sqrt(metre walked)
    drift_yaw_deg_per_m: float = 0.0
    fps: float = 5.0
    max_range: float = 5.0


def camera_rotation(yaw: float, pitch: float) -> np.ndarray:
    """OpenCV camera axes in the y-up world for a heading (plan angle) and pitch (up positive)."""
    f = np.array([np.cos(pitch) * np.cos(yaw), np.sin(pitch), -np.cos(pitch) * np.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.column_stack([right, down, f])


def walk_trajectory(spec: FlatSpec, fps: float, rng: np.random.Generator) -> list[tuple[np.ndarray, float, float]]:
    """(plan position, yaw, pitch) per frame: a sweep in every room, walking through doorways."""
    frames: list[tuple[np.ndarray, float, float]] = []
    order = [r.name for r in spec.rooms]
    centers = {r.name: np.array([(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2]) for r in spec.rooms}

    def sweep(center, radius_xy):
        n = int(fps * 14)
        for k in range(n):
            s = k / n
            yaw = 2 * np.pi * s * 1.15
            pitch = np.radians(50) * np.sin(2 * np.pi * s * 3.0)
            pos = center + radius_xy * np.array([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)])
            frames.append((pos + rng.normal(0, 0.01, 2), yaw, pitch))

    def walk(a, b):
        d = np.linalg.norm(b - a)
        n = max(int(d / 0.5 * fps), 2)
        yaw = np.arctan2(b[1] - a[1], b[0] - a[0])
        for k in range(n):
            s = k / n
            frames.append((a + (b - a) * s, yaw + 0.3 * np.sin(6 * s), np.radians(-10)))

    doors = {}
    for op in spec.openings:
        if op.kind == "window":
            continue
        r = spec.room(op.room)
        from groundplan.sim.scene import side_segment

        a, b = side_segment(r, op.side)
        u = (b - a) / np.linalg.norm(b - a)
        doors.setdefault(op.room, []).append(a + u * (op.start + op.width / 2))

    for i, name in enumerate(order):
        r = spec.room(name)
        rad = np.array([max((r.x1 - r.x0) / 2 - 0.6, 0.05), max((r.y1 - r.y0) / 2 - 0.6, 0.05)])
        sweep(centers[name], rad)
        if i + 1 < len(order):
            nxt = order[i + 1]
            # walk via the hall door shared with the next room when there is one (else straight)
            via = None
            for room_name in (name, nxt):
                for d in doors.get(room_name, []):
                    for other in (name, nxt):
                        ro = spec.room(other)
                        if ro.x0 - 0.3 <= d[0] <= ro.x1 + 0.3 and ro.y0 - 0.3 <= d[1] <= ro.y1 + 0.3:
                            via = d
            path = [centers[name]] + ([via] if via is not None else []) + [centers[nxt]]
            for a, b in zip(path[:-1], path[1:]):
                walk(a, b)
    # return to the start for a loop closure
    walk(centers[order[-1]], centers[order[0]])
    sweep(centers[order[0]], np.array([0.2, 0.2]))
    return frames


_DIRS: dict[tuple, np.ndarray] = {}


def _pixel_dirs(K: np.ndarray, w: int, h: int) -> np.ndarray:
    key = (w, h, *np.round(K.ravel(), 6))
    if key not in _DIRS:
        v, u = np.mgrid[0:h, 0:w]
        _DIRS[key] = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)],
                              -1).reshape(-1, 3).astype(np.float32)
    return _DIRS[key]


def raycast(boxes: np.ndarray, C: np.ndarray, R_wc: np.ndarray, K: np.ndarray, w: int, h: int) -> np.ndarray:
    """Depth (z along the optical axis) for every pixel; inf where nothing is hit (slab method)."""
    dw = _pixel_dirs(K, w, h) @ R_wc.T.astype(np.float32)
    dw = np.where(np.abs(dw) < 1e-9, np.float32(1e-9), dw)
    inv = (1.0 / dw).astype(np.float32)
    best = np.full(h * w, np.inf, np.float32)
    corner_sel = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    pix = np.arange(h * w).reshape(h, w)
    for box in boxes:
        corners = np.stack([box[corner_sel[:, 0], 0], box[corner_sel[:, 1], 1], box[corner_sel[:, 2], 2]], 1)
        cam = (corners - C) @ R_wc
        if np.all(cam[:, 2] <= 0.05):
            continue
        if np.all(cam[:, 2] > 0.05):
            uu = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
            vv = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
            u0, u1 = int(max(np.floor(uu.min()), 0)), int(min(np.ceil(uu.max()) + 1, w))
            v0, v1 = int(max(np.floor(vv.min()), 0)), int(min(np.ceil(vv.max()) + 1, h))
            if u0 >= u1 or v0 >= v1:
                continue
            idx = pix[v0:v1, u0:u1].ravel()
        else:
            idx = pix.ravel()
        iv = inv[idx]
        t1 = iv * (box[0] - C).astype(np.float32)
        t2 = iv * (box[1] - C).astype(np.float32)
        tmin = np.minimum(t1, t2).max(axis=1)
        tmax = np.maximum(t1, t2).min(axis=1)
        hit = (tmax >= tmin) & (tmin > 0.05)
        cur = best[idx]
        best[idx] = np.where(hit & (tmin < cur), tmin, cur)
    return best.reshape(h, w)


def simulate_capture(spec: FlatSpec, out_dir: Path, opts: SimOptions | None = None) -> dict:
    opts = opts or SimOptions()
    rng = np.random.default_rng(opts.seed)
    out_dir = Path(out_dir)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence").mkdir(parents=True, exist_ok=True)
    boxes, _ = build_boxes(spec)
    traj = walk_trajectory(spec, opts.fps, rng)

    K_rgb = np.array([[FX_RGB, 0, 959.5], [0, FX_RGB, 719.5], [0, 0, 1]])
    s = DEPTH_W / RGB_W
    K_d = K_rgb.copy()
    K_d[0, 0] *= s
    K_d[1, 1] *= s
    K_d[0, 2] = (K_d[0, 2] + 0.5) * s - 0.5
    K_d[1, 2] = (K_d[1, 2] + 0.5) * s - 0.5

    np.savetxt(out_dir / "camera_matrix.csv", K_rgb, delimiter=",", fmt="%.4f")
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    true_rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    C_rec = np.zeros(3)
    drift_yaw = 0.0
    prev = None
    cam_h = 1.40
    for i, (pos, yaw, pitch) in enumerate(traj):
        C = np.array([pos[0], cam_h + 0.05 * np.sin(i / 17.0), -pos[1]])
        R = camera_rotation(yaw, pitch)
        depth = raycast(boxes, C, R, K_d, DEPTH_W, DEPTH_H)
        valid = np.isfinite(depth) & (depth < opts.max_range)
        noisy = depth * (1 + opts.depth_scale_bias) + opts.depth_offset_bias
        noisy = noisy + rng.normal(0, 1, depth.shape) * (opts.noise_abs + opts.noise_rel * np.where(valid, depth, 0))
        d_mm = np.where(valid, np.clip(noisy * 1000, 0, 65535), 0).astype(np.uint16)
        conf = np.where(valid & (depth < 3.5), 2, np.where(valid, 1, 0)).astype(np.uint8)
        cv2.imwrite(str(out_dir / "depth" / f"{i:06d}.png"), d_mm)
        cv2.imwrite(str(out_dir / "confidence" / f"{i:06d}.png"), conf)

        # visual-inertial drift is incremental: each true step is integrated under a slowly
        # wandering heading, plus a small translation random walk
        if prev is None:
            C_rec = C.copy()
        else:
            step = np.linalg.norm(C - prev)
            drift_yaw += rng.normal() * np.radians(opts.drift_yaw_deg_per_m) * np.sqrt(step)
            Dy_step = Rotation.from_euler("y", drift_yaw).as_matrix()
            C_rec = C_rec + Dy_step @ (C - prev)
            C_rec = C_rec + rng.normal(0, 1, 3) * opts.drift_pos_per_m * np.sqrt(step) * np.array([1, 0.3, 1])
        prev = C
        Dy = Rotation.from_euler("y", drift_yaw).as_matrix()
        R_rec = Dy @ R
        qt = Rotation.from_matrix(R).as_quat()
        true_rows.append(f"{i / opts.fps:.6f}, {i:06d}, {C[0]:.6f}, {C[1]:.6f}, {C[2]:.6f}, "
                         f"{qt[0]:.7f}, {qt[1]:.7f}, {qt[2]:.7f}, {qt[3]:.7f}")
        q = Rotation.from_matrix(R_rec).as_quat()
        rows.append(f"{i / opts.fps:.6f}, {i:06d}, {C_rec[0]:.6f}, {C_rec[1]:.6f}, {C_rec[2]:.6f}, "
                    f"{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}, {FX_RGB:.4f}, {FX_RGB:.4f}, 959.5, 719.5, , ")
    (out_dir / "odometry.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_dir / "odometry_true.csv").write_text("\n".join(true_rows) + "\n", encoding="utf-8")
    gt = ground_truth(spec)
    gt["sim"] = {k: getattr(opts, k) for k in opts.__dataclass_fields__}
    gt["frames"] = len(traj)
    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return gt
