"""LiDAR tier: Stray Scanner depth + ARKit poses -> fused cloud -> shared core."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from groundplan.calib.intervals import load_calibration
from groundplan.geometry.pointcloud import backproject_frame, miss_rays, ray_samples, select_keyframes, voxel_fuse
from groundplan.geometry.scene import CoreParams, SceneInput, SceneLayout, build_layout
from groundplan.io.stray import StrayCapture, load_stray


@dataclass
class LidarOptions:
    drift: bool = True
    voxel: float = 0.02
    pixel_stride: int = 2
    ray_stride: int = 4
    min_conf: int = 2
    max_depth: float = 4.5


@dataclass
class LidarResult:
    capture: StrayCapture
    keyframes: np.ndarray
    T_wc_used: np.ndarray  # (K, 4, 4) poses after drift correction, keyframes only
    layout: SceneLayout
    scene: SceneInput
    drift_report: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def fuse_lidar(cap: StrayCapture, keyframes: np.ndarray, poses: np.ndarray, opts: LidarOptions,
               depth_scale: float = 1.0, depth_offset: float = 0.0) -> SceneInput:
    X, N, G, RE, RC, MD, MC = [], [], [], [], [], [], []
    for j, (i, (d, c)) in enumerate(zip(keyframes, cap.load_depth_batch(keyframes))):
        K = cap.K_depth(i)
        p, n = backproject_frame(d, c, K, poses[j], min_conf=opts.min_conf, dmax=opts.max_depth,
                                 stride=opts.pixel_stride, depth_scale=depth_scale, depth_offset=depth_offset)
        X.append(p)
        N.append(n)
        G.append(np.full(len(p), j, np.int32))
        d_corr = d * depth_scale + np.where(d > 0, depth_offset, 0.0)
        r = ray_samples(d_corr, c, K, poses[j], stride=opts.ray_stride)
        RE.append(r)
        RC.append(np.full(len(r), j, np.int32))
        m = miss_rays(d_corr, c, K, poses[j], stride=opts.ray_stride * 2)
        MD.append(m)
        MC.append(np.full(len(m), j, np.int32))
    X, N, G = np.concatenate(X), np.concatenate(N), np.concatenate(G)
    xyz, nrm, _, grp = voxel_fuse(X, N, G, opts.voxel)
    return SceneInput(xyz=xyz, normal=nrm, group=grp, cams=poses[:, :3, 3].astype(np.float32),
                      ray_cam=np.concatenate(RC), ray_end=np.concatenate(RE),
                      miss_cam=np.concatenate(MC), miss_dir=np.concatenate(MD))


def run_lidar(path: Path, opts: LidarOptions | None = None, core: CoreParams | None = None) -> LidarResult:
    opts = opts or LidarOptions()
    timings: dict[str, float] = {}
    t = time.perf_counter()
    cap = load_stray(path)
    keyframes = select_keyframes(cap.T_wc, cap.timestamps, cap.has_depth)
    timings["ingest"] = time.perf_counter() - t

    cal = load_calibration()
    scale = cal.correction("lidar", "depth_scale", 1.0)
    offset = cal.correction("lidar", "depth_offset_m", 0.0)

    poses = cap.T_wc[keyframes]
    drift_report: dict = {"enabled": False, "method": "none (ARKit poses as-is)"}
    if opts.drift:
        from groundplan.geometry.drift import correct_drift

        t = time.perf_counter()
        poses, drift_report = correct_drift(cap, keyframes, poses, depth_scale=scale, depth_offset=offset)
        timings["drift"] = time.perf_counter() - t

    t = time.perf_counter()
    scene = fuse_lidar(cap, keyframes, poses, opts, scale, offset)
    timings["fuse"] = time.perf_counter() - t

    t = time.perf_counter()
    layout = build_layout(scene, core or CoreParams())
    timings["layout"] = time.perf_counter() - t

    notes = list(cap.notes) + list(layout.notes)
    return LidarResult(cap, keyframes, poses, layout, scene, drift_report, timings, notes)
