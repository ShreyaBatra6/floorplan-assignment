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
   (``tiers/scale.py``: model factor, ceiling, door heads, camera height, a sheet of paper on
   the floor when one is visible, ``tiers/paper.py``).
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
from groundplan.geometry.openings import OpeningParams
from groundplan.geometry.rooms import SegParams
from groundplan.geometry.scene import CoreParams, SceneInput, SceneLayout, build_layout
from groundplan.io.video import VideoClip, read_video
from groundplan.models.depth import predict_depth
from groundplan.tiers import scale as S
from groundplan.tiers.paper import find_sheet

SFM_THREADS = 2

VIDEO_CORE = CoreParams(openings=OpeningParams(voxel=0.025), grid_res=0.025, min_wall_span=0.6, refine_band=0.12, refine_h_lo=0.2,
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
    pair.overlap = 15
    pair.quadratic_overlap = True
    pair.num_threads = SFM_THREADS
    match = pycolmap.FeatureMatchingOptions()
    match.num_threads = SFM_THREADS
    match.use_gpu = False
    pycolmap.match_sequential(db, matching_options=match, pairing_options=pair)
    # revisit pairs: frames that look alike but are far apart in time (coming back into a room,
    # looking at the same doorway twice). Sequential matching alone cannot connect them, and a walk
    # whose pieces never meet again ends up as several separate models.
    D = frame_descriptors(clip)
    revisit = revisit_pairs(D, min_gap=pair.overlap + 1)
    if revisit:
        pairs_file = workdir / "revisit_pairs.txt"
        pairs_file.write_text("\n".join(f"{i:05d}.jpg {j:05d}.jpg" for i, j in revisit) + "\n", encoding="utf-8")
        imported = pycolmap.ImportedPairingOptions()
        imported.match_list_path = str(pairs_file)
        pycolmap.match_image_pairs(db, matching_options=match, pairing_options=imported)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = 8
    opts.multiple_models = True
    opts.ba_refine_focal_length = True
    opts.random_seed = 0
    opts.num_threads = SFM_THREADS
    # a handheld walkthrough turns quickly in doorways: accept smaller (still verified) overlaps so the
    # walk stays in one model instead of fragmenting at every turn
    opts.min_num_matches = 10
    opts.mapper.init_min_num_inliers = 50
    opts.mapper.abs_pose_min_num_inliers = 15
    opts.mapper.abs_pose_min_inlier_ratio = 0.15
    opts.mapper.max_reg_trials = 5
    out = workdir / "sparse"
    out.mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(db, img_dir, out, options=opts)
    if not maps:
        return None
    models = [m for m in (_model(rec, len(clip.frames)) for rec in maps.values()) if m.registered >= 6]
    if not models:
        return None
    if len(models) == 1:
        models[0].notes = (f"structure from motion: {models[0].registered}/{len(clip.frames)} frames in one model"
                           + _coverage_note(models[0].registered, len(clip.frames)))
        return models[0]
    return bridge_models(clip, models, D)


def frame_descriptors(clip: VideoClip) -> np.ndarray:
    """One global appearance vector per frame (CLIP image embedding; tiny-thumbnail fallback)."""
    from groundplan.models import clip as clip_model

    if clip_model.available():
        return clip_model.image_features([f.image for f in clip.frames])
    thumbs = []
    for f in clip.frames:
        g = cv2.cvtColor(cv2.resize(f.image, (32, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        v = g.astype(np.float32).ravel()
        v -= v.mean()
        thumbs.append(v / (np.linalg.norm(v) + 1e-6))
    return np.array(thumbs)


def revisit_pairs(D: np.ndarray, min_gap: int, per_frame: int = 3, max_pairs: int = 600) -> list[tuple[int, int]]:
    """Non-sequential frame pairs worth matching: each frame's most similar frames outside its time window.

    ``D`` holds one unit descriptor per frame. Similarity must also stand out from the clip's typical
    similarity (above the 97th percentile of all non-adjacent pairs), so plain walls that all look
    alike do not flood the matcher.
    """
    n = len(D)
    if n <= min_gap + 2:
        return []
    S = D @ D.T
    idx = np.arange(n)
    far = np.abs(idx[:, None] - idx[None, :]) >= min_gap
    if not far.any():
        return []
    thr = float(np.percentile(S[far], 97))
    pairs = set()
    for i in range(n):
        cand = np.flatnonzero(far[i] & (S[i] >= thr))
        for j in cand[np.argsort(-S[i, cand])][:per_frame]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    ranked = sorted(pairs, key=lambda p: -S[p[0], p[1]])
    return ranked[:max_pairs]


def _model(rec, n_frames: int) -> SfMResult:
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
    return SfMResult(poses, K, points, obs, len(poses), "")


def _metric_normalise(clip: VideoClip, m: SfMResult, samples: int = 6) -> float:
    """Scale a model to the depth model's units (median model-depth / SfM-depth over a few frames)."""
    keys = sorted(m.poses)
    pick = [keys[i] for i in np.linspace(0, len(keys) - 1, min(samples, len(keys))).round().astype(int)]
    ratios, _ = frame_scales(clip, m, sorted(set(pick)))
    r = float(np.median(list(ratios.values()))) if ratios else 1.0
    for k in m.poses:
        m.poses[k] = m.poses[k].copy()
        m.poses[k][:3, 3] *= r
    m.points = m.points * r
    return r


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _pnp_step(clip: VideoClip, K: np.ndarray, T_a: np.ndarray, a: int, b: int, depth_a: np.ndarray,
              min_inliers: int = 10):
    """Pose of frame b from frame a (known pose, model-unit depth) by SIFT matches + PnP RANSAC.

    Tuned for the frames where reconstruction broke, which are mostly plain walls: contrast is
    equalised and the SIFT threshold halved. A low inlier count is acceptable because the caller
    accepts a transform only when a second frame pair agrees with it.
    Returns (T_wc of b, inlier count) or None."""
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    ga = _CLAHE.apply(cv2.cvtColor(clip.frames[a].image, cv2.COLOR_BGR2GRAY))
    gb = _CLAHE.apply(cv2.cvtColor(clip.frames[b].image, cv2.COLOR_BGR2GRAY))
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < 12:
        return None
    ua = np.array([ka[g.queryIdx].pt for g in good])
    ub = np.array([kb[g.trainIdx].pt for g in good], np.float32)
    h, w = depth_a.shape
    d = depth_a[np.clip(ua[:, 1].astype(int), 0, h - 1), np.clip(ua[:, 0].astype(int), 0, w - 1)]
    ok = (d > 0.2) & (d < 8)
    if ok.sum() < min_inliers:
        return None
    Pc = np.stack([(ua[ok, 0] - K[0, 2]) / K[0, 0] * d[ok], (ua[ok, 1] - K[1, 2]) / K[1, 1] * d[ok], d[ok]], 1)
    Pw = Pc @ T_a[:3, :3].T + T_a[:3, 3]
    found, rvec, tvec, inl = cv2.solvePnPRansac(Pw.astype(np.float32), ub[ok], K, None, reprojectionError=3.0,
                                                iterationsCount=500, confidence=0.999)
    if not found or inl is None or len(inl) < min_inliers:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(Pw[inl].astype(np.float32), ub[ok][inl], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    # RANSAC can return a degenerate pose (camera "at infinity") that still counts inliers: the
    # matched points must lie in front of b at depths like those seen from a, and b must be near a
    zb = (Pw[inl] @ R.T + tvec.ravel())[:, 2]
    za = d[ok][inl]
    if (zb <= 0.1).any() or not (0.33 < np.median(zb) / np.median(za) < 3.0):
        return None
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = (-R.T @ tvec).ravel()
    if np.linalg.norm(T[:3, 3] - T_a[:3, 3]) > 2.0 * float(np.median(za)) + 0.5:
        return None
    return T, len(inl)


def _same_transform(A: np.ndarray, B: np.ndarray, max_deg: float = 4.0, max_shift: float = 0.2) -> bool:
    dR = A[:3, :3].T @ B[:3, :3]
    ang = math.degrees(math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return ang < max_deg and float(np.linalg.norm(A[:3, 3] - B[:3, 3])) < max_shift


def _plausible(M: np.ndarray, base: SfMResult, m: SfMResult, margin: float = 3.0) -> bool:
    """Model m's cameras, moved by M, must stay within reach of the base walk (same property):
    within ``margin`` plus m's own extent of the base cameras' bounding box."""
    cb = np.array([T[:3, 3] for T in base.poses.values()])
    cm = np.array([(M @ T)[:3, 3] for T in m.poses.values()])
    reach = margin + float(np.ptp(cm, axis=0).max())
    lo, hi = cb.min(axis=0) - reach, cb.max(axis=0) + reach
    return bool(((cm >= lo) & (cm <= hi)).all())


def _relocalise(clip: VideoClip, base: SfMResult, m: SfMResult, S: np.ndarray, tries: int = 10):
    """Rigid transform from model m's world into base's world, or None.

    Frames of m are posed against the most similar base frames (appearance similarity, with a bonus
    for neighbours in time) by depth-aided PnP; each success implies one transform. The transform
    is accepted when two different frame pairs agree on it, or one pair has a large inlier set.
    """
    a_keys, b_keys = np.array(sorted(base.poses)), np.array(sorted(m.poses))
    score = S[np.ix_(a_keys, b_keys)] + 0.05 * (np.abs(a_keys[:, None] - b_keys[None, :]) <= 3)
    order = np.argsort(-score, axis=None)
    found: list[tuple[np.ndarray, int]] = []
    used_a, used_b = set(), set()
    for flat in order:
        if len(used_b) >= tries:
            break
        ia, ib = np.unravel_index(flat, score.shape)
        a, b = int(a_keys[ia]), int(b_keys[ib])
        if a in used_a or b in used_b:  # spread the attempts over different frames
            continue
        used_a.add(a)
        used_b.add(b)
        r = _pnp_step(clip, base.K, base.poses[a], a, b, predict_depth(clip.frames[a].image))
        if r is None:
            continue
        M = r[0] @ np.linalg.inv(m.poses[b])
        if not _plausible(M, base, m):
            continue
        for M2, n2 in found:
            if _same_transform(M, M2):
                return M if r[1] >= n2 else M2
        found.append((M, r[1]))
    strong = [x for x in found if x[1] >= 60]
    return max(strong, key=lambda x: x[1])[0] if strong else None


def bridge_models(clip: VideoClip, models: list[SfMResult], D: np.ndarray | None = None) -> SfMResult:
    """Join SfM models that a fast turn or a bland wall split apart.

    Every model is first scaled to the depth model's units, so models differ only by a rigid
    transform. Starting from the largest model, each remaining model is relocalised against the
    growing base (``_relocalise``); models that fail are retried after others have joined, since a
    bigger base offers more frames to match against.
    """
    for m in models:
        _metric_normalise(clip, m)
    D = frame_descriptors(clip) if D is None else D
    S = D @ D.T
    models.sort(key=lambda m: -m.registered)
    base, rest = models[0], models[1:]
    joined = 1
    progress = True
    while rest and progress:
        progress = False
        for m in list(rest):
            M = _relocalise(clip, base, m, S)
            if M is None:
                continue
            offset = len(base.points)
            base.points = np.vstack([base.points, m.points @ M[:3, :3].T + M[:3, 3]])
            for k, Tk in m.poses.items():
                if k not in base.poses:
                    base.poses[k] = M @ Tk
                    uv, pid = m.obs[k]
                    base.obs[k] = (uv, pid + offset)
            rest.remove(m)
            joined += 1
            progress = True
    base.registered = len(base.poses)
    dropped = sum(m.registered for m in rest)
    base.notes = (f"structure from motion split the walk into {len(models)} pieces; {joined - 1} of the "
                  f"{len(models) - 1} smaller ones were joined to the largest (depth-aided PnP, two agreeing "
                  f"frame pairs)"
                  f"{f', {len(rest)} with {dropped} frames could not be placed and were dropped' if rest else ''}"
                  f"; {base.registered}/{len(clip.frames)} frames posed" + _coverage_note(base.registered, len(clip.frames)))
    return base


def _coverage_note(posed: int, total: int) -> str:
    if posed >= 0.6 * total:
        return ""
    return (f". Only {100 * posed / total:.0f} % of the walk is reconstructed: rooms outside it are missing and "
            "rooms at its edge may be cut short (re-record with slower turns, protocol B)")


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
    scene1, world_T1 = build_scene(clip, sfm, ratios, depths, R_align, raw)
    lay1 = build_layout(scene1, VIDEO_CORE)
    cues = [S.model_cue()]
    ceilings = [r.ceiling_y - r.floor_y for r in lay1.rooms if r.ceiling_y is not None]
    cues.append(S.ceiling_cue(ceilings))
    cues.append(S.door_cue([c.h1 for r in lay1.rooms for _, c in r.openings if c.kind == "door"]))
    cam_h = lay1.frame.height(scene1.cams)
    cues.append(S.camera_cue(list(np.clip(cam_h, 0.3, 2.5)), "video"))
    views1 = [View(clip.frames[k].image, sfm.K, Tw, (depths[k] / ratios[k] * raw).astype(np.float32), sfm.K,
                   f"frame{clip.frames[k].index}") for k, Tw in world_T1.items()]
    sheet = find_sheet(views1, lay1.frame.floor_y, max_views=24)
    if sheet is not None:
        cues.append(S.paper_cue(sheet.long_raw, sheet.paper, sheet.sigma, sheet.detail))
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
