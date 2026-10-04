"""Drift correction for LiDAR walks: a 4-DoF submap pose graph with loop closures and plane anchoring.

ARKit's visual-inertial odometry is good but not perfect over a multi-room walk: a few millimetres
to centimetres of translation and a fraction of a degree of heading accumulate, which shows up as
doubled walls when a room is seen again at the end of the walk, and as rooms that do not quite
line up along a shared wall.

Method
------
1. The keyframes are cut into contiguous submaps (~1 m of travel or ~40 degrees of turn).
2. Each submap gets an unknown correction (yaw about its centre + 3D translation).
3. Odometry edges tie consecutive submaps together with a drift budget (1 %/m + 2 mm in
   translation, 0.25 deg/m in heading), i.e. corrections may only change slowly along the walk.
4. Loop-closure edges come from point-to-plane 4-DoF ICP between non-consecutive submaps that see
   the same surfaces; closures that are degenerate (e.g. only a floor and one wall in common),
   poorly fitting, or implausibly large are rejected, and accepted ones enter with a robust
   (Cauchy) loss so a wrong closure cannot drag the whole walk.
5. Plane anchoring: walls are vertical planes on two orthogonal directions, so a submap whose
   walls are rotated against the walk's dominant Manhattan direction carries heading drift; a soft
   prior pulls its yaw back.
6. The solved corrections are interpolated in time onto every keyframe pose.

``--no-drift`` skips all of this (poses as-is) for the ablation the brief requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from groundplan.geometry.icp import icp_4dof, yaw_matrix, yaw_rotate
from groundplan.geometry.manhattan import _wrap_quarter, dominant_yaw
from groundplan.geometry.pointcloud import backproject_frame, voxel_fuse


@dataclass
class DriftParams:
    submap_dist: float = 1.0
    submap_rot_deg: float = 40.0
    submap_max_frames: int = 30
    voxel: float = 0.04
    odo_trans_per_m: float = 0.01
    odo_trans_floor: float = 0.002
    odo_yaw_deg_per_m: float = 0.25
    loop_min_gap: int = 3
    loop_min_time_gap: float = 8.0
    loop_per_submap: int = 3
    loop_max_center_dist: float = 2.5
    loop_max_trans: float = 0.35
    loop_max_yaw_deg: float = 6.0
    loop_min_fitness: float = 0.35
    loop_max_rmse: float = 0.02
    loop_min_eig: float = 8e-3
    loop_cauchy_c: float = 4.0  # whitened residual scale of the per-closure robust weight
    loop_sigma_point: float = 0.005  # per-point plane distance noise for the closure information
    loop_max_eff_points: int = 150  # cap: neighbouring points share pose/depth errors
    loop_expected_drift_per_sqrt_m: float = 0.012  # plausibility gate on a closure's size
    loop_gate_sigmas: float = 3.0
    manhattan_sigma_deg: float = 0.7
    manhattan_min_strength: float = 0.7
    entropy_min_gain: float = 0.005
    max_loops: int = 160


@dataclass
class Submap:
    frames: np.ndarray  # keyframe positions (indices into the keyframe arrays)
    xyz: np.ndarray
    normal: np.ndarray
    center: np.ndarray
    t_mid: float
    path_pos: float  # metres walked at the submap's middle frame


def _submaps(cap, keyframes, poses, p: DriftParams, depth_scale, depth_offset) -> list[Submap]:
    subs: list[list[int]] = []
    cur: list[int] = []
    anchor = None
    for j in range(len(keyframes)):
        if anchor is None:
            anchor = j
        moved = np.linalg.norm(poses[j, :3, 3] - poses[anchor, :3, 3])
        Rrel = poses[anchor, :3, :3].T @ poses[j, :3, :3]
        turned = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        if cur and (moved > p.submap_dist or turned > p.submap_rot_deg or len(cur) >= p.submap_max_frames):
            subs.append(cur)
            cur = []
            anchor = j
        cur.append(j)
    if cur:
        subs.append(cur)

    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))])
    out = []
    depth = cap.load_depth_batch(keyframes)
    for idx in subs:
        X, N, G = [], [], []
        for j in idx:
            d, c = depth[j]
            pts, nrm = backproject_frame(d, c, cap.K_depth(keyframes[j]), poses[j], stride=3,
                                         depth_scale=depth_scale, depth_offset=depth_offset, dmax=4.0)
            X.append(pts)
            N.append(nrm)
            G.append(np.full(len(pts), j, np.int32))
        X, N, G = np.concatenate(X), np.concatenate(N), np.concatenate(G)
        xyz, nrm, _, _ = voxel_fuse(X, N, G, p.voxel)
        mid = idx[len(idx) // 2]
        out.append(Submap(np.array(idx), xyz, nrm, poses[mid, :3, 3].copy(),
                          float(cap.timestamps[keyframes[mid]]), float(path[mid])))
    return out


def _anchors(center: np.ndarray) -> np.ndarray:
    """Four points around a submap centre: residuals on them weigh rotation by a 1 m lever."""
    return center + np.array([[0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], [-1.0, 0, -1.0]])


def correct_drift(cap, keyframes: np.ndarray, poses: np.ndarray, depth_scale: float = 1.0,
                  depth_offset: float = 0.0, params: DriftParams | None = None) -> tuple[np.ndarray, dict]:
    p = params or DriftParams()
    subs = _submaps(cap, keyframes, poses, p, depth_scale, depth_offset)
    n = len(subs)
    report = {"enabled": True, "method": "4-DoF submap pose graph: odometry + ICP loop closures + "
                                         "Manhattan plane anchoring", "submaps": n}
    if n < 3:
        report.update(loop_closures=0, max_correction_m=0.0, mean_correction_m=0.0,
                      notes="walk too short for a pose graph; poses unchanged")
        return poses, report

    # --- loop closures: revisits (long time gap) first, a few nearest partners per submap
    loops = []
    centers = np.array([s.center for s in subs])
    times = np.array([s.t_mid for s in subs])
    cand = []
    for j in range(n):
        partners = []
        for i in range(0, j - p.loop_min_gap + 1):
            if times[j] - times[i] < p.loop_min_time_gap:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist <= p.loop_max_center_dist:
                partners.append((dist, i))
        for _, i in sorted(partners)[: p.loop_per_submap]:
            cand.append((times[j] - times[i], i, j))
    cand.sort(reverse=True)
    if len(cand) > p.max_loops:  # keep an even spread along the walk
        pick = np.linspace(0, len(cand) - 1, p.max_loops).round().astype(int)
        cand = [cand[k] for k in pick]
    tried = 0
    for _, i, j in cand:
        a, b = subs[i], subs[j]
        if len(a.xyz) < 200 or len(b.xyz) < 200:
            continue
        tried += 1
        res = icp_4dof(b.xyz, a.xyz, a.normal, b.normal, max_dist=(0.30, 0.15, 0.08, 0.04, 0.03),
                       max_points=2500, seed=i * 1000 + j)
        walked = abs(subs[j].path_pos - subs[i].path_pos)
        expected = p.loop_expected_drift_per_sqrt_m * np.sqrt(max(walked, 1.0)) + 0.01
        shift = np.linalg.norm(res.apply(subs[j].center[None])[0] - subs[j].center)
        if (res.fitness >= p.loop_min_fitness and res.rmse <= p.loop_max_rmse and res.min_eig >= p.loop_min_eig
                and res.H is not None and shift <= min(p.loop_gate_sigmas * expected, p.loop_max_trans)
                and abs(np.degrees(res.yaw)) <= p.loop_max_yaw_deg):
            loops.append((i, j, res))

    # --- Manhattan heading per submap
    global_yaw, strength = _manhattan_yaw(np.concatenate([s.normal for s in subs]))
    yaw_obs = []
    for s in subs:
        y, st = _manhattan_yaw(s.normal)
        if st >= p.manhattan_min_strength and strength >= p.manhattan_min_strength:
            yaw_obs.append(float(_wrap_quarter(y - global_yaw)))
        else:
            yaw_obs.append(None)

    # --- solve: x = [yaw_i, tx_i, ty_i, tz_i] for i in 1..n-1, then the walk's Manhattan heading.
    # Submap 0 is fixed (gauge); the Manhattan heading is a free unknown, so the plane-anchoring
    # prior pulls submaps towards *each other's* wall direction instead of fighting the gauge.
    def unpack(x):
        X = np.zeros((n, 4))
        X[1:] = x[: (n - 1) * 4].reshape(n - 1, 4)
        return X

    def apply(X, i, pts):
        return yaw_rotate(pts - subs[i].center, X[i, 0]) + subs[i].center + X[i, 1:]

    odo_edges = []
    for i in range(n - 1):
        dist = max(subs[i + 1].path_pos - subs[i].path_pos, 0.05)
        st = p.odo_trans_floor + p.odo_trans_per_m * dist
        sy = np.radians(p.odo_yaw_deg_per_m) * max(dist, 0.2)
        odo_edges.append((i, st, sy))

    def residuals(x):
        X = unpack(x)
        r = []
        for i, st, sy in odo_edges:
            q = _anchors(subs[i + 1].center)
            r.append(((apply(X, i + 1, q) - apply(X, i, q)) / st).ravel())
            r.append(np.array([(X[i + 1, 0] - X[i, 0]) / sy]))
        for (i, j, res), W, wr in zip(loops, loop_w, robust_w):
            r.append(np.sqrt(wr) * (W @ loop_error(X, i, j, res)))
        sm = np.radians(p.manhattan_sigma_deg)
        for i, yo in enumerate(yaw_obs):
            if yo is not None:
                r.append(np.array([(X[i, 0] + yo - x[-1]) / sm]))
        return np.concatenate(r)

    def loop_weights(lps):
        out = []
        for _, _, res in lps:
            n_eff = min(res.n_inliers, p.loop_max_eff_points)
            sp = max(res.rmse, p.loop_sigma_point)
            info = res.H * n_eff / sp**2
            w, V = np.linalg.eigh(info)
            out.append((V * np.sqrt(np.maximum(w, 0.0))).T)  # W^T W = info
        return out

    def loop_error(X, i, j, res):
        c = res.center[None]
        e_t = (apply(X, j, c) - apply(X, i, res.apply(c)))[0]
        return np.concatenate([[X[j, 0] - X[i, 0] - res.yaw], e_t])

    x0 = np.zeros((n - 1) * 4 + 1)
    all_loops = list(loops)
    loop_w = loop_weights(loops)
    # odometry + plane anchoring only, then iteratively reweighted closures (Cauchy weight per
    # closure): closures consistent with the rest of the graph converge to weight 1, false ones
    # (an ICP that slid along a corridor) decay towards 0 instead of bending the walk.
    robust_w = [0.0] * len(loops)
    sparsity = _sparsity(n, odo_edges, loops, yaw_obs, len(residuals(x0)))
    sol = least_squares(residuals, x0, jac_sparsity=sparsity, loss="linear", max_nfev=50,
                        x_scale=np.full(len(x0), 0.01))
    for _round in range(6 if loops else 0):
        X = unpack(sol.x)
        z = np.array([np.linalg.norm(W @ loop_error(X, i, j, res)) for (i, j, res), W in zip(loops, loop_w)])
        robust_w = list(1.0 / (1.0 + (z / p.loop_cauchy_c) ** 2))
        sol = least_squares(residuals, sol.x, jac_sparsity=sparsity, loss="linear", max_nfev=50,
                            x_scale=np.full(len(x0), 0.01))
    X = unpack(sol.x)
    keep = [k for k, w in enumerate(robust_w) if w >= 0.3]
    rejected = len(loops) - len(keep)
    loops = [loops[k] for k in keep]
    _ = all_loops
    report["loop_closures_rejected"] = rejected
    _ = all_loops

    # --- self-check: keep the correction only if it makes the map measurably sharper
    h_raw = map_entropy(subs, np.zeros((n, 4)), apply, global_yaw)
    h_cor = map_entropy(subs, X, apply, global_yaw)
    report["map_entropy_raw"] = round(h_raw, 4)
    report["map_entropy_corrected"] = round(h_cor, 4)
    if not (h_cor < h_raw - p.entropy_min_gain):
        report.update(loop_closures=len(loops), max_correction_m=0.0, mean_correction_m=0.0,
                      notes=(f"correction computed ({len(loops)} closures) but rejected: wall-map entropy "
                             f"{h_raw:.3f} -> {h_cor:.3f} did not improve, so ARKit poses are kept as the "
                             "more consistent map"))
        report["method"] += " (self-checked by wall-map entropy)"
        return poses, report

    # --- interpolate corrections onto keyframes (by walked distance) and apply
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))])
    sp = np.array([s.path_pos for s in subs])
    yaw_k = np.interp(path, sp, X[:, 0])
    t_k = np.stack([np.interp(path, sp, X[:, c]) for c in (1, 2, 3)], axis=1)
    c_k = np.stack([np.interp(path, sp, centers[:, c]) for c in range(3)], axis=1)
    new = poses.copy()
    moved = []
    for k in range(len(poses)):
        R = yaw_matrix(yaw_k[k])
        C = poses[k, :3, 3]
        new[k, :3, 3] = R @ (C - c_k[k]) + c_k[k] + t_k[k]
        new[k, :3, :3] = R @ poses[k, :3, :3]
        moved.append(np.linalg.norm(new[k, :3, 3] - C))
    moved = np.array(moved)
    loop_before = _loop_residual(subs, loops, np.zeros((n, 4)), apply)
    report["notes_rejected"] = rejected
    loop_after = _loop_residual(subs, loops, X, apply)
    report.update(
        loop_closures=len(loops),
        max_correction_m=round(float(moved.max()), 4),
        mean_correction_m=round(float(moved.mean()), 4),
        notes=(f"{tried} closure candidates tested, {len(loops)} accepted ({rejected} rejected as inconsistent); loop residual "
               f"{loop_before * 1000:.1f} mm -> {loop_after * 1000:.1f} mm; max heading correction "
               f"{np.degrees(np.abs(X[:, 0]).max()):.2f} deg; Manhattan anchors on "
               f"{sum(y is not None for y in yaw_obs)}/{n} submaps"),
    )
    return new, report


def map_entropy(subs, X, apply, yaw: float, bin_m: float = 0.01) -> float:
    """Entropy of wall-point coordinates in the Manhattan frame (lower = walls stack more sharply).

    A consistent map puts every observation of a wall on the same plane, giving narrow histogram
    peaks; drift doubles or smears them. Used to accept or reject a correction without ground truth.
    """
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    total = 0.0
    fam = {0: [], 1: []}
    for i, sm in enumerate(subs):
        vert = np.abs(sm.normal[:, 1]) < 0.2
        if vert.sum() < 20:
            continue
        pts = apply(X, i, sm.xyz[vert])
        nr = yaw_rotate(sm.normal[vert], X[i, 0])
        px, py = pts[:, 0], -pts[:, 2]
        nx, ny = nr[:, 0], -nr[:, 2]
        u = c * px - s_ * py
        v = s_ * px + c * py
        nu = c * nx - s_ * ny
        nv = s_ * nx + c * ny
        fam[0].append(u[np.abs(nu) > 0.9])
        fam[1].append(v[np.abs(nv) > 0.9])
    for k in (0, 1):
        if not fam[k]:
            continue
        x = np.concatenate(fam[k])
        if len(x) < 100:
            continue
        hist, _ = np.histogram(x, bins=max(int((x.max() - x.min()) / bin_m), 1))
        pr = hist[hist > 0] / hist.sum()
        total += float(-(pr * np.log(pr)).sum())
    return total


def _loop_residual(subs, loops, X, apply) -> float:
    if not loops:
        return 0.0
    errs = []
    for i, j, res in loops:
        q = subs[j].center[None]
        errs.append(np.linalg.norm(apply(X, j, q) - apply(X, i, res.apply(q))))
    return float(np.median(errs))


def _manhattan_yaw(normals: np.ndarray) -> tuple[float, float]:
    vert = np.abs(normals[:, 1]) < 0.25
    if vert.sum() < 50:
        return 0.0, 0.0
    nxy = np.stack([normals[vert, 0], -normals[vert, 2]], axis=1)
    nxy /= np.linalg.norm(nxy, axis=1, keepdims=True) + 1e-9
    return dominant_yaw(nxy)


def _sparsity(n, odo_edges, loops, yaw_obs, m):
    S = lil_matrix((m, (n - 1) * 4 + 1), dtype=int)
    row = 0

    def mark(r0, nrows, subs_idx):
        for s in subs_idx:
            if s == 0:
                continue
            S[r0 : r0 + nrows, (s - 1) * 4 : s * 4] = 1

    for i, _, _ in odo_edges:
        mark(row, 12, (i, i + 1))
        row += 12
        mark(row, 1, (i, i + 1))
        row += 1
    for i, j, _ in loops:
        mark(row, 4, (i, j))
        row += 4
    for i, yo in enumerate(yaw_obs):
        if yo is not None:
            mark(row, 1, (i,))
            S[row, (n - 1) * 4] = 1
            row += 1
    return S
