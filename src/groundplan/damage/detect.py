"""Damage candidates on a surface mosaic, verified, measured in metres.

Candidates (cheap, high recall):
* discolouration: Lab colour distance from the surface's own large-scale background (a stain,
  mould, peeling or a hole all differ from the plaster around them);
* cracks: thin dark ridges (Sato ridge filter) long enough to matter.

Verification (precision): each candidate crop is scored by CLIP against the damage classes and
against things that look like candidates but are not damage (picture frames, switches, shadows,
furniture, tiles). Without the learned model installed a conservative shape/colour heuristic is
used and its confidence is capped, so the output says how much it can be trusted.

Extent: the region's mask is on a metric raster, so area, width and height follow directly; their
intervals combine raster resolution, the sensitivity of the boundary to the detection threshold,
and the tier's geometric error budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import sato
from skimage.morphology import skeletonize

from groundplan.damage.ortho import Mosaic


@dataclass
class Candidate:
    mask: np.ndarray  # bool, mosaic-sized
    kind_hint: str  # discoloration | ridge
    contrast: float
    bbox: tuple[int, int, int, int]  # r0, r1, c0, c1 (inclusive-exclusive)


@dataclass
class Detection:
    damage_class: str
    confidence: float
    mask: np.ndarray
    polygon_uv: list[tuple[float, float]]
    area: float
    width: float
    height: float
    area_spread: float  # area sensitivity to the threshold (m2)
    extent_spread: float  # width/height sensitivity (m)
    crop: np.ndarray
    views: int
    source: str


@dataclass
class DetectParams:
    bg_kernel_m: float = 0.41
    de_min: float = 9.0
    de_mad_k: float = 4.5
    min_area_m2: float = 0.0025  # 25 cm2
    max_area_frac: float = 0.5
    edge_margin_m: float = 0.04
    crack_min_len_m: float = 0.20
    crack_ridge_k: float = 6.0
    clip_min_prob: float = 0.40
    clip_margin: float = 1.5  # damage probability must exceed the best negative x this
    heuristic_conf_cap: float = 0.45
    min_views: int = 2
    skirting_band_m: float = 0.12  # wall base: skirting boards
    cornice_band_m: float = 0.06  # wall top: cornice / ceiling shadow line
    band_fraction: float = 0.7
    max_border_contact: float = 0.3  # fraction of the region outline touching unobserved pixels
    max_protrusion_m: float = 0.012  # region standing proud of the surface: an object, not damage
    min_stain_area_m2: float = 0.01  # stains, mould and peeling smaller than 10 x 10 cm are not reported
    crack_max_straightness: float = 0.97  # perfectly straight, axis-aligned lines are joints/trim/handles
    crack_max_thickness_m: float = 0.012


def _background(lab: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    filled = lab.copy()
    for c in range(3):
        ch = filled[..., c]
        med = float(np.median(ch[valid])) if valid.any() else 0.0
        ch[~valid] = med
        filled[..., c] = cv2.medianBlur(np.clip(ch, 0, 255).astype(np.uint8), k).astype(np.float32)
    return filled


def find_candidates(m: Mosaic, p: DetectParams | None = None) -> list[Candidate]:
    p = p or DetectParams()
    valid = m.valid & (m.views >= 1)
    if valid.sum() < 100:
        return []
    erode_px = max(int(round(p.edge_margin_m / m.res)), 1)
    core = ndimage.binary_erosion(valid, iterations=erode_px)
    lab = cv2.cvtColor(m.image, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = int(round(p.bg_kernel_m / m.res)) | 1
    k = max(min(k, 2 * (min(lab.shape[:2]) // 2) - 1), 3)
    bg = _background(lab, valid, k)
    d = lab - bg
    de = np.sqrt((0.7 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    vals = de[core]
    if vals.size == 0:
        return []
    thr = max(p.de_min, float(np.median(vals) + p.de_mad_k * 1.4826 * np.median(np.abs(vals - np.median(vals)))))
    mask = (de > thr) & core
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    out: list[Candidate] = []
    lab_c, n = ndimage.label(mask)
    min_px = p.min_area_m2 / m.res**2
    for i, sl in enumerate(ndimage.find_objects(lab_c), start=1):
        comp = lab_c == i
        area_px = int(comp.sum())
        if area_px < min_px or area_px > p.max_area_frac * valid.sum():
            continue
        if not _plausible(m, comp, valid, p) or m.surface.kind == "floor":
            continue
        out.append(Candidate(comp, "discoloration", float(de[comp].mean()),
                             (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)))
    if m.surface.kind == "floor" or _looks_tiled(m, valid):
        return out  # grout lines and floor joints are not cracks
    # cracks: dark thin ridges
    L = lab[..., 0] / 255.0
    L = np.where(valid, L, float(np.median(L[valid])))
    ridge = sato(L, sigmas=[1, 2], black_ridges=True)
    rv = ridge[core]
    if rv.size:
        rthr = max(0.05, float(np.percentile(rv, 99)),
                   float(np.median(rv) + p.crack_ridge_k * 1.4826 * np.median(np.abs(rv - np.median(rv)))))
        rmask = (ridge > rthr) & core
        skel = skeletonize(rmask)
        # each skeleton piece is judged on its own: two parallel handle bars are two straight lines
        lab_r, _ = ndimage.label(skel, structure=np.ones((3, 3)))
        for i, sl in enumerate(ndimage.find_objects(lab_r), start=1):
            piece = lab_r == i
            length_m = piece.sum() * m.res
            h = (sl[0].stop - sl[0].start) * m.res
            w = (sl[1].stop - sl[1].start) * m.res
            if length_m < p.crack_min_len_m or max(h, w) < p.crack_min_len_m or not _tortuous(piece, p):
                continue
            comp = ndimage.binary_dilation(piece, iterations=1)
            if comp.sum() * m.res**2 < 0.02 and _plausible(m, comp, valid, p, min_views=1):
                out.append(Candidate(comp, "ridge", float(ridge[comp].mean()),
                                     (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)))
    return out


def _plausible(m: Mosaic, comp: np.ndarray, valid: np.ndarray, p: DetectParams, min_views: int | None = None) -> bool:
    """Reject regions that are artefacts of where the surface was (not) observed or of its trim.

    Discolouration must be confirmed by ``min_views`` views (a glare or reflection is view-dependent);
    a thin crack is not produced by reflections and may come from a single view."""
    if np.median(m.views[comp]) < (p.min_views if min_views is None else min_views):
        return False
    if m.protrusion is not None:
        pr = m.protrusion[comp]
        pr = pr[np.isfinite(pr)]
        if pr.size >= 5 and float(np.median(pr)) > p.max_protrusion_m:
            return False
    ring = ndimage.binary_dilation(comp, iterations=2) & ~comp
    if ring.any() and (~valid[ring]).mean() > p.max_border_contact:
        return False
    if m.surface.kind == "wall":
        vv = m.vv[comp]
        if (vv < m.surface.v0 + p.skirting_band_m).mean() >= p.band_fraction:
            return False
        if (vv > m.surface.v1 - p.cornice_band_m).mean() >= p.band_fraction:
            return False
    return True


def _tortuous(skel: np.ndarray, p: DetectParams) -> bool:
    """Cracks wander; a straight line aligned with the surface axes is a joint, trim edge or handle."""
    ys, xs = np.nonzero(skel)
    if len(xs) < 5:
        return False
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts, full_matrices=False)
    straightness = sv[0] / max(np.sqrt((sv**2).sum()), 1e-9)
    ang = abs(np.degrees(np.arctan2(vt[0, 1], vt[0, 0]))) % 90
    axis_aligned = min(ang, 90 - ang) < 6
    return not (straightness > p.crack_max_straightness and axis_aligned)


def _thickness(mask: np.ndarray, res: float) -> float:
    """Mean width of a line-like region: area over skeleton length."""
    length = max(int(skeletonize(mask).sum()), 1)
    return float(mask.sum()) / length * res


def _elongation(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return 1.0
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    sv = np.linalg.svd(pts, compute_uv=False)
    return float(sv[0] / max(sv[-1], 1e-6))


def _looks_tiled(m: Mosaic, valid: np.ndarray) -> bool:
    """Long straight joints repeating across the surface: tiles or panelling."""
    gray = cv2.cvtColor(m.image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges[~valid] = 0
    span = min(gray.shape)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(int(0.5 * span), 30), maxLineGap=8)
    return lines is not None and len(lines) >= 6


def _crop(m: Mosaic, c: Candidate) -> np.ndarray:
    r0, r1, c0, c1 = c.bbox
    pad_r = max(int(0.4 * (r1 - r0)), 8)
    pad_c = max(int(0.4 * (c1 - c0)), 8)
    side = max(r1 - r0 + 2 * pad_r, c1 - c0 + 2 * pad_c)
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    a0, a1 = max(rc - side // 2, 0), min(rc + side // 2 + 1, m.image.shape[0])
    b0, b1 = max(cc - side // 2, 0), min(cc + side // 2 + 1, m.image.shape[1])
    img = m.image.copy()
    if m.valid.any():
        img[~m.valid] = np.median(m.image[m.valid], axis=0).astype(np.uint8)
    return img[a0:a1, b0:b1]


def _heuristic(m: Mosaic, c: Candidate) -> tuple[str | None, float]:
    lab = cv2.cvtColor(m.image, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = lab[c.mask]
    ring = ndimage.binary_dilation(c.mask, iterations=6) & ~c.mask & m.valid
    outer = lab[ring] if ring.any() else inner
    dL = float(inner[:, 0].mean() - outer[:, 0].mean())
    db = float(inner[:, 2].mean() - outer[:, 2].mean())
    texture = float(inner[:, 0].std())
    if c.kind_hint == "ridge":
        return "crack", 0.4
    if dL < -12 and texture > 18:
        return "mold", 0.35
    if dL < -6 and db > 3:
        return "water_stain", 0.4
    return None, 0.0


def _view_crop(m: Mosaic, c: Candidate, views) -> np.ndarray | None:
    """The candidate as a natural photo crop from the view that sees it most squarely."""
    if not views:
        return None
    r0, r1, c0, c1 = c.bbox
    us = [m.uu[0, min(c0, m.uu.shape[1] - 1)], m.uu[0, min(c1 - 1, m.uu.shape[1] - 1)]]
    vs = [m.vv[min(r0, m.vv.shape[0] - 1), 0], m.vv[min(r1 - 1, m.vv.shape[0] - 1), 0]]
    s = m.surface
    pts = np.array([s.origin + (u - s.u0) * s.u_axis + (v - s.v0) * s.v_axis for u in us for v in vs])
    centre = pts.mean(axis=0)
    best, best_score = None, 0.0
    for view in views:
        d = view.center - centre
        dist = float(np.linalg.norm(d))
        cos = float(d @ s.normal) / max(dist, 1e-6)
        if cos < 0.4 or dist > 3.5:
            continue
        R, t = view.T_wc[:3, :3], view.T_wc[:3, 3]
        cam = (pts - t) @ R
        if np.any(cam[:, 2] < 0.2):
            continue
        u = view.K[0, 0] * cam[:, 0] / cam[:, 2] + view.K[0, 2]
        v = view.K[1, 1] * cam[:, 1] / cam[:, 2] + view.K[1, 2]
        h, w = view.image.shape[:2]
        if u.min() < 0 or v.min() < 0 or u.max() >= w or v.max() >= h:
            continue
        score = cos / max(dist, 0.5)
        if score > best_score:
            best_score, best = score, (view, u, v)
    if best is None:
        return None
    view, u, v = best
    cx, cy = u.mean(), v.mean()
    side = max(u.max() - u.min(), v.max() - v.min()) * 1.8 + 24
    h, w = view.image.shape[:2]
    x0, x1 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, w))
    y0, y1 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, h))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return view.image[y0:y1, x0:x1].copy()


def classify(m: Mosaic, cands: list[Candidate], p: DetectParams | None = None, views=None) -> list[Detection]:
    p = p or DetectParams()
    if not cands:
        return []
    crops = []
    for c in cands:
        vc = _view_crop(m, c, views)
        crops.append(vc if vc is not None else _crop(m, c))
    from groundplan.models import clip

    use_clip = clip.available()
    verdicts = clip.classify_damage(crops) if use_clip else [None] * len(cands)
    out = []
    for c, crop, v in zip(cands, crops, verdicts):
        if use_clip:
            cls, prob, neg = v
            if c.kind_hint == "ridge" and (cls != "crack" or prob < 0.5):
                continue
            if cls in ("water_stain", "mold", "peeling_paint") and c.mask.sum() * m.res**2 < p.min_stain_area_m2:
                continue
            if cls == "crack" and c.kind_hint != "ridge" and (
                    _elongation(c.mask) < 4.0 or _thickness(c.mask, m.res) > p.crack_max_thickness_m):
                continue  # a crack is a thin line; a dark bar or blob is something else
            if prob < p.clip_min_prob or prob < neg * p.clip_margin:
                continue
            conf, source = float(min(prob / (prob + neg + 1e-6), 0.95)), "clip"
        else:
            cls, conf = _heuristic(m, c)
            if cls is None:
                continue
            conf, source = min(conf, p.heuristic_conf_cap), "heuristic"
        out.append(_measure(m, c, cls, conf, crop, source))
    return out


def _measure(m: Mosaic, c: Candidate, cls: str, conf: float, crop: np.ndarray, source: str) -> Detection:
    mask = c.mask
    res = m.res
    rows, cols = np.nonzero(mask)
    area = float(mask.sum()) * res**2
    width = float((cols.max() - cols.min() + 1) * res)
    height = float((rows.max() - rows.min() + 1) * res)
    # boundary sensitivity: one-pixel erosion/dilation of the region
    grown = ndimage.binary_dilation(mask)
    shrunk = ndimage.binary_erosion(mask)
    area_spread = float(grown.sum() - max(shrunk.sum(), 0)) * res**2 / 2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    eps = max(0.01 * cv2.arcLength(cnt, True), 1.0)
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(poly) < 3:
        x, y, w, h = cv2.boundingRect(cnt)
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    uv = [(float(m.uu[0, 0] + (px - 0) * res), float(m.vv[0, 0] - (py - 0) * res)) for px, py in poly]
    views = int(np.median(m.views[mask])) if mask.any() else 1
    return Detection(cls, conf, mask, uv, area, width, height, area_spread, res, crop, max(views, 1), source)
