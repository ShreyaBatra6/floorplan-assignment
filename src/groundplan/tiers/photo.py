"""Photo tier: per-room folders of 2-8 stills -> per-room layouts -> one stitched plan.

Per photo:   metric monocular depth -> points and normals -> gravity from the floor plane ->
             Manhattan yaw from the walls.
Per room:    photos registered pairwise with SIFT matches lifted to 3D (gravity-aligned similarity
             RANSAC: yaw, translation and relative scale, because every photo's depth has its own
             scale error); the registered points go through the same geometric core as the other
             tiers, at a coarser grid and with the photo tier's error budget.
Scale:       fused from the calibrated model factor, the room's ceiling, door heads and camera
             heights (``tiers/scale.py``); the fused log-sigma widens every interval.
Property:    rooms placed by the layout solver (compass headings + door pairing + no overlap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from groundplan.damage.ortho import View
from groundplan.geometry.manhattan import dominant_yaw
from groundplan.geometry.planes import ransac_plane
from groundplan.geometry.pointcloud import flying_pixel_mask, image_normals, voxel_fuse
from groundplan.geometry.rooms import SegParams
from groundplan.geometry.scene import CoreParams, SceneInput, SceneLayout, build_layout
from groundplan.io.photos import Photo, RoomPhotos, load_photo_folders
from groundplan.models.depth import predict_depth
from groundplan.tiers import scale as S

PHOTO_CORE = CoreParams(grid_res=0.03, min_wall_span=0.5, refine_band=0.15, refine_h_lo=0.2,
                        unobserved_wall_sigma=0.15, wall_band_bottom=0.1,
                        seg=SegParams(min_wall_span=0.5, min_free_rays=1.0, marker_min_dist=0.4, max_door_width=1.6,
                                      min_room_area=1.5))


@dataclass
class PhotoGeom:
    photo: Photo
    depth_raw: np.ndarray  # model depth at photo resolution (no scale correction)
    pts: np.ndarray  # (N, 3) points in the photo's gravity+Manhattan frame (raw units)
    nrm: np.ndarray  # (N, 3)
    R: np.ndarray  # camera -> photo frame rotation
    cam_height: float | None
    yaw_strength: float
    room_height: float | None = None  # floor-to-ceiling distance in this photo's depth units


@dataclass
class RoomRecon:
    name: str
    geoms: list[PhotoGeom]
    poses: dict[int, tuple[np.ndarray, np.ndarray, float]]  # photo idx -> (R cam->room, t, scale)
    layout: SceneLayout | None = None
    scale: float = 1.0
    sigma_log: float = 0.5
    scale_info: object = None
    notes: list[str] = field(default_factory=list)


def _gravity_rotation(up_c: np.ndarray) -> np.ndarray:
    """Rotation (camera -> gravity frame, y up) keeping the camera's heading along world -z."""
    Y = up_c / np.linalg.norm(up_c)
    f = np.array([0.0, 0.0, 1.0])
    Zf = f - (f @ Y) * Y
    if np.linalg.norm(Zf) < 1e-6:
        Zf = np.array([0.0, 1.0, 0.0]) - (np.array([0.0, 1.0, 0.0]) @ Y) * Y
    Zf /= np.linalg.norm(Zf)
    Z = -Zf
    X = np.cross(Y, Z)
    return np.stack([X, Y, Z])


def _yaw(R: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ R


def photo_geometry(photo: Photo, grid: int = 4) -> PhotoGeom:
    depth = predict_depth(photo.image)
    h, w = depth.shape
    K = photo.K
    d = depth[::grid, ::grid].astype(np.float32)
    d = np.where(flying_pixel_mask(d, 0.05) | (d > 9.0), 0.0, d)
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    uu, vv = uu * grid, vv * grid
    P = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1).astype(np.float32)
    valid = d > 0
    N, nok = image_normals(P, valid, step=2)
    ok = valid & nok
    pts, nrm = P[ok], N[ok]
    rows = vv[ok]
    # gravity: floor plane among roughly up-facing points in the lower part of the image
    up_guess = np.array([0.0, -1.0, 0.0])
    cand = (nrm @ up_guess > math.cos(math.radians(35))) & (rows > 0.35 * h)
    up = up_guess
    cam_h = None
    if cand.sum() > 200:
        plane, inl = ransac_plane(pts[cand], thresh=0.03 * float(np.median(d[valid])), iters=200,
                                  axis=up_guess, max_angle_deg=35)
        if plane is not None and inl.sum() > 150:
            n = plane.normal if plane.normal @ up_guess > 0 else -plane.normal
            up = n
            cam_h = abs(plane.d) if plane.normal @ up_guess > 0 else abs(plane.d)
    room_h = None
    if cam_h is not None:
        down_c = -up
        cc = (nrm @ down_c > math.cos(math.radians(25))) & (rows < 0.55 * h)
        if cc.sum() > 150:
            cplane, cin = ransac_plane(pts[cc], thresh=0.03 * float(np.median(d[valid])), iters=200,
                                       axis=up, max_angle_deg=20)
            if cplane is not None and cin.sum() > 100:
                room_h = cam_h + abs(cplane.d)
    Rg = _gravity_rotation(up)
    pg = pts @ Rg.T
    ng = nrm @ Rg.T
    vert = np.abs(ng[:, 1]) < 0.25
    yaw, strength = (0.0, 0.0)
    if vert.sum() > 100:
        nxy = np.stack([ng[vert, 0], -ng[vert, 2]], 1)
        nxy /= np.linalg.norm(nxy, axis=1, keepdims=True) + 1e-9
        yaw, strength = dominant_yaw(nxy)
    R = _yaw(Rg, -yaw)
    return PhotoGeom(photo, depth, pts @ R.T, nrm @ R.T, R, cam_h, strength, room_h)


def _sift(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 1024 / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=4000)
    kp, des = sift.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kp], np.float32) / (s if s < 1 else 1.0)
    return pts, des


def _lift(g: PhotoGeom, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = g.depth_raw.shape
    iu = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    iv = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    d = g.depth_raw[iv, iu]
    K = g.photo.K
    P = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * d, (uv[:, 1] - K[1, 2]) / K[1, 1] * d, d], 1)
    return P @ g.R.T, (d > 0.2) & (d < 9.0)


def _yaw_from_compass(a: PhotoGeom, b: PhotoGeom) -> float | None:
    """Relative yaw (radians, b into a) implied by the two photos' compass headings, if both have one."""
    if a.photo.heading_deg is None or b.photo.heading_deg is None:
        return None
    # optical axes in each photo frame (frames differ by the yaw we want)
    fa = a.R[:, 2]
    fb = b.R[:, 2]
    phi_a = math.atan2(-fa[2], fa[0])
    phi_b = math.atan2(-fb[2], fb[0])
    # compass is clockwise from north, plan angles counter-clockwise: d(plan) = -d(compass)
    d_compass = math.radians(b.photo.heading_deg - a.photo.heading_deg)
    return (phi_a - d_compass) - phi_b


def register_pair(a: PhotoGeom, b: PhotoGeom, feats, seed: int = 0):
    """Similarity (yaw, t, scale) mapping b's frame into a's.

    Matches are verified in 2D first (fundamental-matrix RANSAC: robust to the depth model's noise),
    then the surviving matches are lifted to 3D and a gravity-constrained similarity is fitted
    robustly. Both frames are Manhattan-aligned, so the yaw is near a multiple of 90 degrees; the
    compass (when present) picks which multiple and rejects registrations that contradict it.
    """
    (pa, da), (pb, db) = feats
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    m1 = matcher.knnMatch(da, db, k=2)
    good = [m[0] for m in m1 if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 12:
        return None
    ua = pa[[m.queryIdx for m in good]]
    ub = pb[[m.trainIdx for m in good]]
    F, mask = cv2.findFundamentalMat(ua, ub, cv2.FM_RANSAC, 2.0, 0.999)
    if F is None or mask is None or mask.sum() < 12:
        return None
    keep = mask.ravel().astype(bool)
    A, okA = _lift(a, ua[keep])
    B, okB = _lift(b, ub[keep])
    ok = okA & okB
    A, B = A[ok], B[ok]
    if len(A) < 10:
        return None
    compass = _yaw_from_compass(a, b)
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    for _ in range(300):
        i, j = rng.choice(len(A), 2, replace=False)
        va, vb = A[j] - A[i], B[j] - B[i]
        if np.hypot(va[0], va[2]) < 0.25 or np.hypot(vb[0], vb[2]) < 0.25:
            continue
        s = np.linalg.norm(va) / max(np.linalg.norm(vb), 1e-6)
        if not (0.5 < s < 2.0):
            continue
        yaw = math.atan2(vb[2], vb[0]) - math.atan2(va[2], va[0])
        if compass is not None and abs((yaw - compass + math.pi) % (2 * math.pi) - math.pi) > math.radians(35):
            continue
        Ry = _yaw(np.eye(3), yaw)
        t = A[i] - s * (Ry @ B[i])
        res = np.linalg.norm(A - (s * (B @ Ry.T) + t), axis=1)
        tol = np.maximum(0.10 * np.linalg.norm(A, axis=1), 0.08)
        score = float(np.sum(np.clip(1 - (res / tol) ** 2, 0, None)))  # MSAC-style truncated score
        if score > best_score:
            best, best_score = (yaw, s, t), score
    if best is None:
        return None
    yaw, s, t = best
    Ry = _yaw(np.eye(3), yaw)
    res = np.linalg.norm(A - (s * (B @ Ry.T) + t), axis=1)
    inl = res < np.maximum(0.10 * np.linalg.norm(A, axis=1), 0.08)
    if inl.sum() < 10 or inl.mean() < 0.3:
        return None
    Ai, Bi = A[inl], B[inl]
    ca, cb = Ai.mean(0), Bi.mean(0)
    a0, b0 = Ai - ca, Bi - cb
    num = np.sum(a0[:, 2] * b0[:, 0] - a0[:, 0] * b0[:, 2])
    den = np.sum(a0[:, 0] * b0[:, 0] + a0[:, 2] * b0[:, 2])
    yaw = math.atan2(-num, den)
    # snap to the Manhattan grid when close: both frames are wall-aligned
    snapped = round(yaw / (math.pi / 2)) * (math.pi / 2)
    if abs(yaw - snapped) < math.radians(8):
        yaw = snapped
    Ry = _yaw(np.eye(3), yaw)
    s = float(np.sum(a0 * (b0 @ Ry.T)) / max(np.sum(b0 * b0), 1e-9))
    t = ca - s * (Ry @ cb)
    return yaw, s, t, int(inl.sum())


def _structural_poses(geoms: list[PhotoGeom]):
    """Poses from the walls each photo sees (needs compass headings; see tiers/structural.py)."""
    from groundplan.tiers.structural import solve_room, wall_observations

    if len(geoms) < 2 or any(g.photo.heading_deg is None for g in geoms):
        return None, "structural registration skipped (needs a compass heading on every photo)"
    rot = []
    for g in geoms:
        f = g.R[:, 2]  # optical axis in the photo frame
        phi = math.degrees(math.atan2(-f[2], f[0]))
        rot.append(90.0 - g.photo.heading_deg - phi)
    ks = [int(round(((r - rot[0] + 180) % 360 - 180) / 90.0)) % 4 for r in rot]
    obs = []
    for g in geoms:
        plan_p = np.stack([g.pts[:, 0], -g.pts[:, 2]], 1)
        plan_n = np.stack([g.nrm[:, 0], -g.nrm[:, 2]], 1)
        plan_n /= np.linalg.norm(plan_n, axis=1, keepdims=True) + 1e-9
        vert = np.abs(g.nrm[:, 1]) < 0.3
        obs.append(wall_observations(plan_p[vert], plan_n[vert]))
    from groundplan.tiers.structural import rotate_facing

    backs = []
    for g, k in zip(geoms, ks):
        f = g.R[:, 2]
        dx, dy = f[0], -f[2]
        if g.room_height is None and max(abs(dx), abs(dy)) > math.cos(math.radians(25)) * math.hypot(dx, dy):
            # looking along a wall axis with the photographer's back to a wall: that wall faces the view direction
            face = (0 if dx > 0 else 1) if abs(dx) > abs(dy) else (2 if dy > 0 else 3)
            backs.append(rotate_facing(face, k))
        else:
            backs.append(None)
    res = solve_room(obs, ks, room_heights=[g.room_height for g in geoms], back_walls=backs)
    if not res.ok:
        return None, f"structural registration rejected (rms {res.rms:.2f}, walls seen {len(res.walls)})"
    poses = {}
    h0 = geoms[0].cam_height or 0.0
    for i, (g, k) in enumerate(zip(geoms, ks)):
        R = _yaw(np.eye(3), k * math.pi / 2)
        y = (res.scales[i] * (g.cam_height or h0) - h0) if g.cam_height else 0.0
        t = np.array([res.positions[i, 0], y, -res.positions[i, 1]])
        poses[i] = (R, t, float(res.scales[i]))
    return poses, f"structural registration: {len(geoms)} photos, wall rms {res.rms:.3f}"


def reconstruct_room(room: RoomPhotos) -> RoomRecon:
    geoms = [photo_geometry(p) for p in room.photos]
    poses, note = _structural_poses(geoms)
    if poses is not None:
        rec = RoomRecon(room.name, geoms, poses)
        rec.notes.append(f"{room.name}: {note}")
        return rec
    feats = [_sift(g.photo.image) for g in geoms]
    n = len(geoms)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            r = register_pair(geoms[i], geoms[j], (feats[i], feats[j]), seed=i * 31 + j)
            if r is not None:
                edges.append((r[3], i, j, r))
    # maximum spanning tree from the photo with most links
    poses: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    deg = np.zeros(n)
    for w, i, j, _ in edges:
        deg[i] += w
        deg[j] += w
    root = int(np.argmax(deg)) if edges else 0
    poses[root] = (np.eye(3), np.zeros(3), 1.0)
    remaining = sorted(edges, reverse=True)
    grew = True
    while grew:
        grew = False
        for w, i, j, (yaw, s, t, _) in remaining:
            if (i in poses) == (j in poses):
                continue
            Ry = _yaw(np.eye(3), yaw)
            if i in poses:  # b=j into a=i's frame, then into the room frame
                Ri, ti, si = poses[i]
                poses[j] = (Ri @ Ry, si * (Ri @ t) + ti, si * s)
            else:  # a=i from b=j: invert the similarity
                Rj, tj, sj = poses[j]
                Rinv = Ry.T
                poses[i] = (Rj @ Rinv, sj * (Rj @ (-(Rinv @ t) / s)) + tj, sj / s)
            grew = True
    rec = RoomRecon(room.name, geoms, poses)
    rec.notes.append(f"{room.name}: {note}; feature-based registration used")
    if len(poses) < n:
        rec.notes.append(f"{room.name}: {n - len(poses)} of {n} photos could not be registered (too little "
                         "overlap or texture) and were not used")
    return rec


def room_scene(rec: RoomRecon, scale: float = 1.0) -> SceneInput:
    X, N, G, cams, RE, RC = [], [], [], [], [], []
    for k, (Rr, t, s) in rec.poses.items():
        g = rec.geoms[k]
        pts = (g.pts @ Rr.T) * s + t
        X.append(pts * scale)
        N.append(g.nrm @ Rr.T)
        G.append(np.full(len(pts), k, np.int32))
        cams.append(t * scale)
        sub = pts[:: max(len(pts) // 4000, 1)] * scale
        RE.append(sub)
        RC.append(np.full(len(sub), len(cams) - 1, np.int32))
    X, N, G = np.concatenate(X), np.concatenate(N), np.concatenate(G)
    xyz, nrm, _, grp = voxel_fuse(X.astype(np.float32), N.astype(np.float32), G, 0.03)
    return SceneInput(xyz=xyz, normal=nrm, group=grp, cams=np.array(cams, np.float32),
                      ray_cam=np.concatenate(RC), ray_end=np.concatenate(RE).astype(np.float32))


def _main_room(layout: SceneLayout, cams_xy: np.ndarray):
    """The room region holding most camera positions (doorway views also see the next room)."""
    if not layout.rooms:
        return None
    from matplotlib.path import Path as MplPath

    best = max(layout.rooms, key=lambda r: (MplPath(r.outline.vertices).contains_points(cams_xy).sum(), r.area))
    layout.rooms = [best]
    return best


def scale_room(rec: RoomRecon) -> None:
    """Two passes: geometry in raw model units -> cues -> fused scale -> metric geometry."""
    raw = build_layout(room_scene(rec, 1.0), PHOTO_CORE)
    cams_xy = raw.frame.to_plan(room_scene(rec, 1.0).cams)
    main = _main_room(raw, cams_xy)
    cues = [S.model_cue()]
    if main is not None and main.ceiling_y is not None:
        cues.append(S.ceiling_cue([main.ceiling_y - main.floor_y]))
    if main is not None:
        heads = [c.h1 for _, c in main.openings if c.kind == "door"]
        cues.append(S.door_cue(heads))
    cam_h = []
    for k, (Rr, t, s) in rec.poses.items():
        if main is not None:
            cam_h.append(float(t[1] - main.floor_y))
    cues.append(S.camera_cue(cam_h, "photo"))
    rec.scale, rec.sigma_log, rec.scale_info = S.fuse([c for c in cues if c is not None])
    scene = room_scene(rec, rec.scale)
    rec.layout = build_layout(scene, PHOTO_CORE)
    if _main_room(rec.layout, rec.layout.frame.to_plan(scene.cams)) is None:
        _fallback_room(rec, scene)


def _fallback_room(rec: RoomRecon, scene: SceneInput) -> None:
    """The photos never closed an outline: report the observed footprint as a box, every wall unobserved."""
    from groundplan.geometry.walls import Edge, RoomOutline
    from groundplan.geometry.scene import RoomGeom

    lay = rec.layout
    xy = lay.frame.to_plan(scene.xyz)
    h = lay.frame.height(scene.xyz)
    sel = (h > 0.05) & (h < 2.2)
    if sel.sum() < 50:
        return
    (x0, y0), (x1, y1) = np.percentile(xy[sel], 5, axis=0), np.percentile(xy[sel], 95, axis=0)
    if x1 - x0 < 0.8 or y1 - y0 < 0.8:
        return
    verts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    edges = [Edge(1, y0, 1, sigma=0.3), Edge(0, x1, -1, sigma=0.3), Edge(1, y1, -1, sigma=0.3), Edge(0, x0, 1, sigma=0.3)]
    outline = RoomOutline(verts, edges)
    lay.rooms = [RoomGeom(0, outline, np.zeros(lay.grid.shape, bool), lay.floor.y, 0.02, None, None, [], None, 0.0)]
    rec.notes.append(f"{rec.name}: photos did not close a room outline; the observed footprint is reported as a "
                     "box with every wall unobserved (wide intervals)")


def photo_views(rec: RoomRecon) -> list[View]:
    """The room's photos as posed views (room frame, metric) for damage detection."""
    views = []
    for k, (Rr, t, s) in rec.poses.items():
        g = rec.geoms[k]
        T = np.eye(4)
        T[:3, :3] = Rr @ g.R
        T[:3, 3] = t * rec.scale
        depth = g.depth_raw * s * rec.scale
        views.append(View(g.photo.image, g.photo.K, T, depth, g.photo.K, g.photo.path.name))
    return views


def run_photos(root: Path) -> list[RoomRecon]:
    recs = []
    for room in load_photo_folders(root):
        if not room.photos:
            continue
        rec = reconstruct_room(room)
        scale_room(rec)
        recs.append(rec)
    return recs
