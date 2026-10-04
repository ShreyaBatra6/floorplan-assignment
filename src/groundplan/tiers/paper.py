"""A printed A4 / US Letter sheet on the floor as a metric scale reference (photo and video tiers).

A sheet of printer paper is the one object in a home whose size is known to a fraction of a
millimetre, so when the capture protocol's sheet is visible it dominates the scale fusion.
Detection runs on the floor plane, where the sheet is seen without perspective:

1. Each view's floor is resampled to a top-down raster through the homography of the plane
   y = floor_y, in the capture's own (not yet metric) units. The raster resolution follows the
   camera height, so it is about 2 mm whatever the units are.
2. Bright, low-chroma blobs (a sweep of brightness thresholds) whose outline is a clean convex
   quadrilateral are kept when their side ratio matches A4 (sqrt 2) or Letter (11 / 8.5) within
   3 %. The ratio does not depend on the unknown scale, so it is the main test; size is only
   checked loosely against the camera height.
3. Each side is relocated to sub-pixel accuracy at the strongest brightness step across it. Long
   and short side give two scale estimates; if they disagree the floor plane (or a curled sheet)
   is suspect and the detection is dropped. The depth map must also put the sheet on the floor,
   not on a table top, where the same shape would come out too large.
4. Detections of the same sheet from several views are combined by median; a sheet seen in one
   view only gets a wider sigma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

PAPERS = {"A4": (0.297, 0.210), "Letter": (0.2794, 0.2159)}
RATIO_TOL = 0.03
SIDE_AGREEMENT = 0.03
CAMERA_HEIGHT_RANGE = (0.9, 2.0)  # metres; bounds the sheet's size in raster units, nothing more
SIGMA_MULTI, SIGMA_SINGLE = 0.015, 0.025


@dataclass
class SheetDetection:
    view: str
    paper: str
    long_raw: float  # long side in capture units, from both sides (short side x nominal ratio)
    center: np.ndarray  # (x, z) on the floor, capture units
    agreement: float  # |log(long-side scale / short-side scale)|
    contrast: float


@dataclass
class SheetFinding:
    paper: str
    long_raw: float
    sigma: float
    views: int
    detail: str


def _floor_raster(view, floor_y: float, max_px: int = 1800):
    """Top-down resampling of the floor seen by ``view``: (raster BGR, valid mask, A raster->floor, res)."""
    T = view.T_wc
    C, R = T[:3, 3], T[:3, :3]
    hc = float(C[1] - floor_y)
    if hc <= 0:
        return None
    h, w = view.image.shape[:2]
    us, vs = np.meshgrid(np.linspace(0, w - 1, 24), np.linspace(0, h - 1, 18))
    rays = np.linalg.inv(view.K) @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
    d = R @ rays
    down = d[1] < -1e-3
    if down.sum() < 20:
        return None
    t = -hc / d[1, down]
    hit = C[None, [0, 2]] + (d[[0, 2]][:, down] * t).T
    near = np.hypot(*(hit - C[[0, 2]]).T) < 2.2 * hc
    if near.sum() < 20:
        return None
    lo, hi = hit[near].min(axis=0), hit[near].max(axis=0)
    res = max(hc / 700.0, float(np.max(hi - lo)) / max_px)
    W, H = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    A = np.array([[res, 0, lo[0]], [0, res, lo[1]], [0, 0, 1.0]])
    Hfloor = view.K @ R.T @ np.column_stack([[1, 0, 0], [0, 0, 1], np.array([0, floor_y, 0]) - C])
    M = Hfloor @ A  # raster pixel -> image pixel
    raster = cv2.warpPerspective(view.image, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    p = M @ np.stack([cols.ravel(), rows.ravel(), np.ones(cols.size)])
    in_front = p[2] > 1e-6
    u = np.where(in_front, p[0] / np.where(in_front, p[2], 1), -1)
    v = np.where(in_front, p[1] / np.where(in_front, p[2], 1), -1)
    valid = (in_front & (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)).reshape(H, W)
    return raster, valid, A, res, hc, M


def _edge_offset(L: np.ndarray, c: np.ndarray, n: np.ndarray, t_dir: np.ndarray, nominal: float,
                 half_len: float) -> float | None:
    """Distance from ``c`` along ``n`` to the bright-to-dark step nearest ``nominal`` (sub-pixel)."""
    s = np.arange(nominal - 6, nominal + 6.01, 0.25)
    ts = np.linspace(-0.35, 0.35, 15) * 2 * half_len
    found = []
    for tv in ts:
        pts = c[None] + tv * t_dir[None] + s[:, None] * n[None]
        prof = map_coordinates(L, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
        g = -np.diff(prof)
        k = int(np.argmax(g))
        if g[k] < 0.75 or k == 0 or k == len(g) - 1:  # under 3 L units per pixel: no edge here
            continue
        den = g[k - 1] - 2 * g[k] + g[k + 1]
        dk = 0.5 * (g[k - 1] - g[k + 1]) / den if den < 0 else 0.0
        found.append(s[0] + (k + 0.5 + dk) * 0.25)
    if len(found) < 8:
        return None
    return float(np.median(found))


def _measure(L: np.ndarray, rect) -> tuple[float, float] | None:
    """Refined (long, short) side lengths in raster pixels for a sheet found as ``rect``."""
    (cx, cy), (rw, rh), ang = rect
    a = math.radians(ang)
    ax_w, ax_h = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.array([cx, cy])
    off = []
    for n, t_dir, half, other in ((ax_w, ax_h, rw / 2, rh / 2), (-ax_w, ax_h, rw / 2, rh / 2),
                                  (ax_h, ax_w, rh / 2, rw / 2), (-ax_h, ax_w, rh / 2, rw / 2)):
        e = _edge_offset(L, c, n, t_dir, half, other)
        if e is None:
            return None
        off.append(e)
    side_w, side_h = off[0] + off[1], off[2] + off[3]
    return (max(side_w, side_h), min(side_w, side_h))


def _depth_on_floor(view, rect, A: np.ndarray, floor_y: float) -> bool:
    """The depth map, where available, must put the sheet's centre on the floor plane (+-15 %)."""
    if view.depth is None:
        return True
    X = A @ np.array([rect[0][0], rect[0][1], 1.0])
    T = view.T_wc
    pc = T[:3, :3].T @ (np.array([X[0], floor_y, X[1]]) - T[:3, 3])
    z_pred = float(pc[2])
    if z_pred <= 0:
        return False
    q = (view.K_depth if view.K_depth is not None else view.K) @ pc
    ud, vd = int(round(q[0] / q[2])), int(round(q[1] / q[2]))
    hd, wd = view.depth.shape
    if not (2 <= ud < wd - 2 and 2 <= vd < hd - 2):
        return True
    z = float(np.median(view.depth[vd - 2:vd + 3, ud - 2:ud + 3]))
    if not np.isfinite(z) or z <= 0:
        return True
    return abs(math.log(z / z_pred)) < 0.15


def detect_in_view(view, floor_y: float) -> list[SheetDetection]:
    fr = _floor_raster(view, floor_y)
    if fr is None:
        return []
    raster, valid, A, res, hc, M = fr
    lab = cv2.cvtColor(raster, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[..., 0].astype(np.float32), (0, 0), 1.0)
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    pale = (chroma < 22) & valid
    if pale.sum() < 500:
        return []
    # expected long side in raster pixels, from a hand-held camera height between 0.9 and 2.0 m
    lo_px = 0.297 * hc / CAMERA_HEIGHT_RANGE[1] / res * 0.85
    hi_px = 0.297 * hc / CAMERA_HEIGHT_RANGE[0] / res * 1.15
    valid_er = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    hi_L = float(np.percentile(L[valid], 99.5))
    out, seen = [], []
    for thr in np.arange(hi_L - 6, max(hi_L - 110, 60), -8):
        mask = ((L >= thr) & pale).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            if len(cnt) < 40:
                continue
            rect = cv2.minAreaRect(cnt)
            long_px, short_px = max(rect[1]), min(rect[1])
            if not (lo_px <= long_px <= hi_px) or short_px < 10:
                continue
            ratio = long_px / short_px
            paper = next((p for p, (a, b) in PAPERS.items() if abs(ratio / (a / b) - 1) < RATIO_TOL), None)
            if paper is None:
                continue
            if cv2.contourArea(cnt) / (long_px * short_px) < 0.92:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            box = cv2.boxPoints(((rect[0]), (rect[1][0] + 12, rect[1][1] + 12), rect[2])).round().astype(int)
            if (box < 0).any() or (box[:, 0] >= valid.shape[1]).any() or (box[:, 1] >= valid.shape[0]).any():
                continue
            if not valid_er[box[:, 1], box[:, 0]].all():  # the whole sheet must be inside the view
                continue
            if any(np.hypot(*(np.array(rect[0]) - s)) < 0.3 * short_px for s in seen):
                continue
            inner = np.zeros(valid.shape, np.uint8)
            cv2.drawContours(inner, [cnt], -1, 1, -1)
            ring = cv2.dilate(inner, np.ones((15, 15), np.uint8)) - cv2.dilate(inner, np.ones((5, 5), np.uint8))
            core = cv2.erode(inner, np.ones((7, 7), np.uint8))
            if core.sum() < 100 or ring.sum() < 100:
                continue
            contrast = float(L[core > 0].mean() - L[(ring > 0) & valid].mean())
            if contrast < 20 or float(L[core > 0].std()) > 18:
                continue
            m = _measure(L, rect)
            if m is None:
                continue
            a, b = PAPERS[paper]
            s_long, s_short = a / m[0], b / m[1]
            agree = abs(math.log(s_long / s_short))
            if agree > SIDE_AGREEMENT:
                continue
            if not _depth_on_floor(view, rect, A, floor_y):
                continue
            seen.append(np.array(rect[0]))
            long_eq = math.sqrt(m[0] * m[1] * a / b) * res
            ctr = A @ np.array([rect[0][0], rect[0][1], 1.0])
            out.append(SheetDetection(view.key, paper, long_eq, ctr[:2], agree, contrast))
    return out


def find_sheet(views, floor_y: float, max_views: int = 40) -> SheetFinding | None:
    """The best-supported sheet over ``views`` (all in one frame and unit, y up), or None."""
    if not views:
        return None
    step = max(1, len(views) // max_views)
    dets = [d for v in views[::step] for d in detect_in_view(v, floor_y)]
    if not dets:
        return None
    groups: list[list[SheetDetection]] = []
    for d in sorted(dets, key=lambda d: -d.contrast):
        for g in groups:
            if g[0].paper == d.paper and np.hypot(*(g[0].center - d.center)) < 0.5 * g[0].long_raw:
                g.append(d)
                break
        else:
            groups.append([d])
    best = max(groups, key=lambda g: (len({d.view for d in g}), sum(d.contrast for d in g)))
    lengths = np.array([d.long_raw for d in best])
    med = float(np.median(lengths))
    n = len({d.view for d in best})
    spread = float(np.median(np.abs(np.log(lengths / med)))) if n > 1 else 0.0
    sigma = SIGMA_MULTI if n > 1 and spread < 0.015 else SIGMA_SINGLE
    others = len(groups) - 1
    detail = (f"{best[0].paper} sheet on the floor in {n} view(s), long side {med:.4f} (raw), "
              f"view-to-view spread {spread * 100:.1f} %" + (f"; {others} other candidate(s) ignored" if others else ""))
    return SheetFinding(best[0].paper, med, sigma, n, detail)
