"""Video tier: plain walkthrough clip -> structure from motion -> metric depth -> shared core.

1. Sharp frames at ~3 fps (upright, focal length from QuickTime metadata).
2. pycolmap: SIFT, sequential matching, incremental mapping with global bundle adjustment
   (the bundle adjustment is this tier's drift handling: the whole walk is one optimised map).
3. Gravity: a phone is filmed without roll, so every camera's x-axis is horizontal and "up" is the
   direction perpendicular to all of them (smallest eigenvector of their scatter); the floor decides
   the sign.
4. Scale: for every registered frame the depth model is compared with the SfM depths of the points
   that frame sees; each frame's depth map is aligned to the reconstruction (removing the model's
   per-frame scale noise) and the reconstruction itself gets a metric scale from the fused cues
   (``tiers/scale.py``: model factor, ceiling, door heads, camera height).
5. The aligned depth maps are fused and go through the same geometric core as the LiDAR tier.
If structure from motion fails, the frames are processed like photos of one space, with intervals
that say so.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from groundplan.damage.ortho import View
from groundplan.geometry.pointcloud import flying_pixel_mask, image_normals, voxel_fuse
from groundplan.geometry.rooms import SegParams
from groundplan.geometry.scene import CoreParams, SceneInput, SceneLayout, build_layout
from groundplan.io.video import VideoClip, read_video
from groundplan.models.depth import predict_depth
from groundplan.tiers import scale as S

SFM_THREADS = 2

VIDEO_CORE = CoreParams(grid_res=0.025, min_wall_span=0.6, refine_band=0.12, refine_h_lo=0.2,
                        unobserved_wall_sigma=0.12, wall_band_bottom=0.1,
                        seg=SegParams(min_wall_span=0.6, min_free_rays=1.5, marker_min_dist=0.35, min_room_area=1.2))


@dataclass
class SfMResult:
    poses: dict[int, np.ndarray]  # frame position in clip.frames -> T_wc (4x4, OpenCV camera -> SfM world)
    K: np.ndarray
    points: np.ndarray  # (P, 3) SfM world
    obs: dict[int, tuple[np.ndarray, np.ndarray]]  # frame -> (uv (M,2), point ids (M,))
    registered: int
    notes: str


def run_sfm(clip: VideoClip, workdir: Path) -> SfMResult | None:
    import pycolmap

    img_dir = workdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, f in enumerate(clip.frames):
        name = f"{i:05d}.jpg"
        cv2.imwrite(str(img_dir / name), f.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        names.append(name)
    db = workdir / "database.db"
    h, w = clip.frames[0].image.shape[:2]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{clip.K[0, 0]},{clip.K[0, 2]},{clip.K[1, 2]},0"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = 1280
    ext.sift.max_num_features = 4000
    ext.use_gpu = False
    ext.num_threads = SFM_THREADS  # all-core SIFT exhausts memory on 8 GB laptops and COLMAP aborts
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    pair = pycolmap.SequentialPairingOptions()
    pair.overlap = 12
    pair.quadratic_overlap = True
    pair.num_threads = SFM_THREADS
    match = pycolmap.FeatureMatchingOptions()
    match.num_threads = SFM_THREADS
    match.use_gpu = False
    pycolmap.match_sequential(db, matching_options=match, pairing_options=pair)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = 8
    opts.multiple_models = True
    opts.ba_refine_focal_length = True
    opts.random_seed = 0
    opts.num_threads = SFM_THREADS
    out = workdir / "sparse"
    out.mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(db, img_dir, out, options=opts)
    if not maps:
        return None
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    cam = next(iter(rec.cameras.values()))
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    ids = list(rec.point3D_ids()) if callable(getattr(rec, "point3D_ids", None)) else list(rec.points3D.keys())
    pid_index = {pid: k for k, pid in enumerate(ids)}
    points = np.array([rec.points3D[pid].xyz for pid in ids]) if ids else np.zeros((0, 3))
    poses, obs = {}, {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        k = int(Path(img.name).stem)
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        Rcw = np.array(cfw.rotation.matrix())
        tcw = np.array(cfw.translation)
        T = np.eye(4)
        T[:3, :3] = Rcw.T
        T[:3, 3] = -Rcw.T @ tcw
        poses[k] = T
        uv, pid = [], []
        for p2 in img.points2D:
            if p2.has_point3D():
                uv.append(p2.xy)
                pid.append(pid_index[p2.point3D_id])
        obs[k] = (np.array(uv).reshape(-1, 2), np.array(pid, int))
    note = f"structure from motion: {len(poses)}/{len(clip.frames)} frames registered in the largest of {len(maps)} model(s)"
    return SfMResult(poses, K, points, obs, len(poses), note)


def gravity_up(poses: dict[int, np.ndarray]) -> np.ndarray:
    xs = np.array([T[:3, 0] for T in poses.values()])
    ys = np.array([T[:3, 1] for T in poses.values()])
    w, V = np.linalg.eigh(xs.T @ xs)
    up = V[:, 0]
    if up @ (-ys.mean(axis=0)) < 0:  # camera +y points down in the image
        up = -up
    return up / np.linalg.norm(up)


def _align_world(up: np.ndarray) -> np.ndarray:
    """Rotation taking the SfM world to a y-up world."""
    y = up
    x = np.cross(y, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(y, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z])


def frame_scales(clip: VideoClip, sfm: SfMResult, frames: list[int]) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    """Per frame: ratio of model depth to SfM depth (model units per SfM unit), and the model depth map."""
    ratios, depths = {}, {}
    for k in frames:
        T = sfm.poses[k]
        uv, pid = sfm.obs.get(k, (np.zeros((0, 2)), np.zeros(0, int)))
        if len(pid) < 20:
            continue
        d_model = predict_depth(clip.frames[k].image)
        P = sfm.points[pid]
        cam = (P - T[:3, 3]) @ T[:3, :3]
        z = cam[:, 2]
        h, w = d_model.shape
        iu = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        iv = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        dm = d_model[iv, iu]
        ok = (z > 0.05) & (dm > 0.1)
        if ok.sum() < 15:
            continue
        ratios[k] = float(np.median(dm[ok] / z[ok]))
        depths[k] = d_model
    return ratios, depths


def build_scene(clip: VideoClip, sfm: SfMResult, ratios: dict[int, float], depths: dict[int, np.ndarray],
                R_align: np.ndarray, metric: float, grid: int = 4) -> tuple[SceneInput, dict[int, np.ndarray]]:
    """Fused cloud in a y-up metric world from depth maps aligned per frame to the reconstruction."""
    X, N, G, cams, RE, RC = [], [], [], [], [], []
    world_T = {}
    K = sfm.K
    for j, (k, r) in enumerate(sorted(ratios.items())):
        T = sfm.poses[k].copy()
        Tw = np.eye(4)
        Tw[:3, :3] = R_align @ T[:3, :3]
        Tw[:3, 3] = (R_align @ T[:3, 3]) * metric
        world_T[k] = Tw
        d = depths[k][::grid, ::grid] / r * metric  # aligned to SfM, then metric
        d = np.where(flying_pixel_mask(d, 0.05) | (d > 7.0), 0.0, d).astype(np.float32)
        vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        uu, vv = uu * grid, vv * grid
        P = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1).astype(np.float32)
        nrm, nok = image_normals(P, d > 0, step=2)
        ok = (d > 0) & nok
        pts = P[ok] @ Tw[:3, :3].T + Tw[:3, 3]
        X.append(pts)
        N.append(nrm[ok] @ Tw[:3, :3].T)
        G.append(np.full(len(pts), j, np.int32))
        cams.append(Tw[:3, 3])
        sub = pts[:: max(len(pts) // 3000, 1)]
        RE.append(sub)
        RC.append(np.full(len(sub), j, np.int32))
    X, N, G = np.concatenate(X), np.concatenate(N), np.concatenate(G)
    xyz, nrm, _, grp = voxel_fuse(X.astype(np.float32), N.astype(np.float32), G, 0.025)
    scene = SceneInput(xyz=xyz, normal=nrm, group=grp, cams=np.array(cams, np.float32),
                       ray_cam=np.concatenate(RC), ray_end=np.concatenate(RE).astype(np.float32))
    return scene, world_T


@dataclass
class VideoResult:
    clip: VideoClip
    sfm: SfMResult | None
    layout: SceneLayout
    views: list[View]
    scale_info: object
    sigma_log: float
    timings: dict
    notes: list[str]


def run_video(path: Path, max_dense: int = 60) -> VideoResult:
    timings, notes = {}, []
    t = time.perf_counter()
    clip = read_video(path)
    notes += clip.notes
    timings["decode"] = time.perf_counter() - t
    work = Path(tempfile.mkdtemp(prefix="groundplan_sfm_"))
    try:
        t = time.perf_counter()
        sfm = run_sfm(clip, work)
        timings["sfm"] = time.perf_counter() - t
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if sfm is None or sfm.registered < 8:
        raise RuntimeError("structure from motion failed on this video (too fast, too dark or too little texture); "
                           "re-record following the capture protocol or use the photo tier")
    notes.append(sfm.notes)
    t = time.perf_counter()
    keys = sorted(sfm.poses)
    pick = [keys[i] for i in np.linspace(0, len(keys) - 1, min(len(keys), max_dense)).round().astype(int)]
    ratios, depths = frame_scales(clip, sfm, sorted(set(pick)))
    timings["depth"] = time.perf_counter() - t
    if len(ratios) < 5:
        raise RuntimeError("too few frames with usable depth and reconstruction overlap")
    raw = float(np.median(list(ratios.values())))  # model units per SfM unit
    R_align = _align_world(gravity_up(sfm.poses))
    t = time.perf_counter()
    # pass 1: geometry in model units -> scale cues; pass 2: metric
    scene1, _ = build_scene(clip, sfm, ratios, depths, R_align, raw)
    lay1 = build_layout(scene1, VIDEO_CORE)
    cues = [S.model_cue()]
    ceilings = [r.ceiling_y - r.floor_y for r in lay1.rooms if r.ceiling_y is not None]
    cues.append(S.ceiling_cue(ceilings))
    cues.append(S.door_cue([c.h1 for r in lay1.rooms for _, c in r.openings if c.kind == "door"]))
    cam_h = lay1.frame.height(scene1.cams)
    cues.append(S.camera_cue(list(np.clip(cam_h, 0.3, 2.5)), "video"))
    factor, sigma, info = S.fuse([c for c in cues if c is not None])
    scene, world_T = build_scene(clip, sfm, ratios, depths, R_align, raw * factor)
    layout = build_layout(scene, VIDEO_CORE)
    timings["layout"] = time.perf_counter() - t
    notes += layout.notes
    views = []
    for k, Tw in world_T.items():
        d = depths[k] / ratios[k] * raw * factor
        views.append(View(clip.frames[k].image, sfm.K, Tw, d.astype(np.float32), sfm.K, f"frame{clip.frames[k].index}"))
    return VideoResult(clip, sfm, layout, views, info, sigma, timings, notes)


def video_branch(det, opts, out_dir, warnings: list[str], stages: dict):
    """Pipeline branch: same assembly as the LiDAR tier, with the video error budget and scale sigma."""
    from groundplan.assemble import AssembleContext, build_rooms, build_stitched
    from groundplan.config import BUDGETS
    from groundplan.contract import Capture, CaptureQuality, DeviceInfo
    from groundplan.models.registry import set_cache

    set_cache(opts.use_cache)
    path = det.root if det.root.is_file() else next(p for p in sorted(det.root.iterdir())
                                                   if p.suffix.lower() in {".mov", ".mp4", ".m4v"})
    res = run_video(path)
    stages.update(res.timings)
    warnings += res.notes
    ctx = AssembleContext(tier="video", scale_sigma=res.sigma_log, budget=BUDGETS["video"])
    layout = res.layout
    all_v = np.vstack([g.outline.vertices for g in layout.rooms]) if layout.rooms else np.zeros((1, 2))
    ctx.origin = np.floor(all_v.min(axis=0) * 10) / 10
    t = time.perf_counter()
    rooms, _ = build_rooms(layout, ctx)
    drift = {"enabled": True, "method": "structure from motion with global bundle adjustment over the whole walk",
             "notes": res.sfm.notes if res.sfm else ""}
    stitched = build_stitched(layout, rooms, ctx, drift, "single continuous video: one reconstruction for all rooms")
    stages["assemble"] = time.perf_counter() - t
    regions, flags, scope_items = [], [], []
    if opts.damage:
        t = time.perf_counter()
        from groundplan.damage.pipeline import run_damage
        from groundplan.pipeline import _apply_room_types

        offsets = {f"R{k}": g.floor_y - layout.frame.floor_y for k, g in enumerate(layout.rooms, start=1)}
        dmg = run_damage(rooms, res.views, layout.frame, ctx, offsets, out_dir)
        _apply_room_types(rooms, dmg.room_types)
        regions, flags, scope_items = dmg.regions, dmg.flags, dmg.scope
        warnings += dmg.notes
        stages["damage"] = time.perf_counter() - t
    clip = res.clip
    capture = Capture(id=path.stem, tier="video", source=str(path), app="iPhone Camera (video)",
                      device=DeviceInfo(make=clip.make, model=clip.model), duration_s=round(clip.duration_s, 2),
                      frames_total=clip.n_frames, frames_used=len(clip.frames), quality=CaptureQuality())
    return capture, res.scale_info, rooms, stitched, regions, flags, scope_items


__all__ = ["run_video", "video_branch", "math"]
