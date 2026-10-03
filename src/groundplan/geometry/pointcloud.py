"""Depth back-projection, per-pixel normals and voxel fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from groundplan.geometry.transforms import transform_points


@dataclass
class Cloud:
    """A fused point cloud in world coordinates (y up)."""

    xyz: np.ndarray  # (M, 3) float32
    normal: np.ndarray  # (M, 3) float32, unit, oriented towards the observing cameras
    count: np.ndarray  # (M,) int32, observations merged into each voxel
    frame: np.ndarray  # (M,) int32, index into ``cams`` of one observing keyframe
    cams: np.ndarray  # (K, 3) camera centres of the keyframes
    rays_end: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    rays_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask: np.ndarray) -> Cloud:
        return Cloud(self.xyz[mask], self.normal[mask], self.count[mask], self.frame[mask], self.cams,
                     self.rays_end, self.rays_frame)


def depth_to_points(depth: np.ndarray, K: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame (OpenCV) points for every ``stride``-th pixel. Returns (H', W', 3) and valid mask."""
    d = depth[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pts = np.stack([x, y, d], axis=-1).astype(np.float32)
    return pts, d > 0


def flying_pixel_mask(depth: np.ndarray, rel_jump: float = 0.04) -> np.ndarray:
    """True where a pixel sits on a depth discontinuity (mixed foreground/background returns)."""
    d = depth
    pad = np.pad(d, 1, mode="edge")
    worst = np.zeros_like(d)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = pad[1 + dy : 1 + dy + d.shape[0], 1 + dx : 1 + dx + d.shape[1]]
            worst = np.maximum(worst, np.abs(nb - d))
    return worst > rel_jump * np.maximum(d, 1e-3)


def image_normals(pts: np.ndarray, valid: np.ndarray, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel normals from central differences of the point image, oriented towards the camera."""
    h, w, _ = pts.shape
    n = np.zeros_like(pts)
    ok = np.zeros((h, w), bool)
    s = step
    if h <= 2 * s or w <= 2 * s:
        return n, ok
    du = pts[s:-s, 2 * s :] - pts[s:-s, : -2 * s]
    dv = pts[2 * s :, s:-s] - pts[: -2 * s, s:-s]
    cr = np.cross(du, dv)
    norm = np.linalg.norm(cr, axis=-1, keepdims=True)
    good = (
        valid[s:-s, 2 * s :] & valid[s:-s, : -2 * s] & valid[2 * s :, s:-s] & valid[: -2 * s, s:-s] & valid[s:-s, s:-s]
    ) & (norm[..., 0] > 1e-9)
    cr = cr / np.maximum(norm, 1e-12)
    centre = pts[s:-s, s:-s]
    flip = np.sum(cr * centre, axis=-1) > 0  # normal must face the camera (origin)
    cr[flip] *= -1
    n[s:-s, s:-s] = cr
    ok[s:-s, s:-s] = good
    return n, ok


def backproject_frame(
    depth: np.ndarray,
    conf: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    min_conf: int = 2,
    dmin: float = 0.25,
    dmax: float = 4.5,
    stride: int = 2,
    depth_scale: float = 1.0,
    depth_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """World points and normals of the trustworthy pixels of one depth frame."""
    d = depth * depth_scale + np.where(depth > 0, depth_offset, 0.0)
    good = (d > dmin) & (d < dmax) & (conf >= min_conf) & ~flying_pixel_mask(d)
    d = np.where(good, d, 0.0).astype(np.float32)
    pts, valid = depth_to_points(d, K, 1)
    nrm, nok = image_normals(pts, valid, step=2)
    keep = valid & nok
    if stride > 1:
        sub = np.zeros_like(keep)
        sub[::stride, ::stride] = True
        keep &= sub
    P = pts[keep]
    N = nrm[keep]
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    return (P @ R.T + t).astype(np.float32), (N @ R.T).astype(np.float32)


def voxel_fuse(
    xyz: np.ndarray, normal: np.ndarray, frame: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average points falling in the same voxel; sum normals; count observations."""
    if len(xyz) == 0:
        return xyz, normal, np.zeros(0, np.int32), frame
    keys = np.floor(xyz / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    m = len(uniq)
    out_xyz = np.stack([np.bincount(inv, weights=xyz[:, k], minlength=m) for k in range(3)], axis=1)
    out_n = np.stack([np.bincount(inv, weights=normal[:, k], minlength=m) for k in range(3)], axis=1)
    out_xyz /= counts[:, None]
    nn = np.linalg.norm(out_n, axis=1, keepdims=True)
    out_n = out_n / np.maximum(nn, 1e-12)
    first = np.full(m, -1, np.int64)
    first[inv[::-1]] = frame[::-1]
    return out_xyz.astype(np.float32), out_n.astype(np.float32), counts.astype(np.int32), first.astype(np.int32)


def select_keyframes(
    T_wc: np.ndarray,
    timestamps: np.ndarray,
    usable: np.ndarray,
    min_trans: float = 0.04,
    min_rot_deg: float = 4.0,
    max_gap_s: float = 0.6,
    max_ang_vel_deg: float = 90.0,
) -> np.ndarray:
    """Indices of frames that add new viewpoint, skipping frames during very fast rotation."""
    n = len(T_wc)
    if n == 0:
        return np.zeros(0, int)
    ang_vel = np.zeros(n)
    for i in range(1, n):
        dt = max(timestamps[i] - timestamps[i - 1], 1e-3)
        Rrel = T_wc[i - 1, :3, :3].T @ T_wc[i, :3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        ang_vel[i] = ang / dt
    chosen: list[int] = []
    last = None
    for i in range(n):
        if not usable[i] or ang_vel[i] > max_ang_vel_deg:
            continue
        if last is None:
            chosen.append(i)
            last = i
            continue
        dtr = np.linalg.norm(T_wc[i, :3, 3] - T_wc[last, :3, 3])
        Rrel = T_wc[last, :3, :3].T @ T_wc[i, :3, :3]
        drot = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        if dtr >= min_trans or drot >= min_rot_deg or timestamps[i] - timestamps[last] >= max_gap_s:
            chosen.append(i)
            last = i
    return np.array(chosen, dtype=int)


def ray_samples(depth: np.ndarray, conf: np.ndarray, K: np.ndarray, T_wc: np.ndarray, stride: int = 8,
                dmax: float = 6.0) -> np.ndarray:
    """Sparse world endpoints of observed rays (used for free-space carving and opening tests)."""
    d = np.where((depth > 0.25) & (depth < dmax) & (conf >= 1), depth, 0.0)
    pts, valid = depth_to_points(d, K, stride)
    return transform_points(T_wc, pts[valid]).astype(np.float32)
