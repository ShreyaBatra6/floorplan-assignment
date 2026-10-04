"""Floor-sheet scale reference: rendered views of a textured floor with a sheet of known size."""

import math

import numpy as np
import pytest

from groundplan.damage.ortho import View
from groundplan.tiers import scale as S
from groundplan.tiers.paper import find_sheet

UNIT = 1 / 0.684  # capture units per metre: the raw depth model's over-estimate


def _view(cam_xz, yaw_deg, pitch_deg, sheet, cam_h=1.4, seed=0, w=1280, h=960, f=1050.0, ss=2):
    """Render one photo (supersampled) of the floor y=0; ``sheet`` = (cx, cz, long, short, angle_deg) in m."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    fwd = np.array([math.sin(y) * math.cos(p), -math.sin(p), -math.cos(y) * math.cos(p)])
    right = np.array([math.cos(y), 0.0, math.sin(y)])
    down = np.cross(fwd, right)
    R = np.column_stack([right, down, fwd])  # OpenCV camera -> world (y up)
    C = np.array([cam_xz[0], cam_h, cam_xz[1]])
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    us, vs = np.meshgrid((np.arange(w * ss) + 0.5) / ss - 0.5, (np.arange(h * ss) + 0.5) / ss - 0.5)
    d = R @ (np.linalg.inv(K) @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)]))
    t = np.where(d[1] < -1e-6, -cam_h / np.minimum(d[1], -1e-6), np.inf)
    x, z = C[0] + d[0] * t, C[2] + d[2] * t
    rng = np.random.default_rng(seed)
    grain = 0.5 + 0.5 * np.sin(x * 40 + 3 * np.sin(z * 7)) * np.cos(z * 2.3)
    img = np.stack([55 + 25 * grain, 85 + 30 * grain, 120 + 35 * grain], axis=-1)  # warm wood floor, BGR
    cx, cz, L, Wd, ang = sheet
    a = math.radians(ang)
    lu = (x - cx) * math.cos(a) + (z - cz) * math.sin(a)
    lv = -(x - cx) * math.sin(a) + (z - cz) * math.cos(a)
    inside = (np.abs(lu) <= L / 2) & (np.abs(lv) <= Wd / 2)
    img[inside] = (232, 236, 238)
    img[~np.isfinite(t)] = (200, 200, 200)
    img = img.reshape(h * ss, w * ss, 3).reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
    img = np.clip(img + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, C * UNIT
    depth = (np.where(np.isfinite(t), t * (d.T @ R)[:, 2], 0).reshape(h * ss, w * ss)[::ss, ::ss] * UNIT)
    return View(img, K, T, depth.astype(np.float32), K, f"v{seed}")


@pytest.mark.parametrize("paper,dims", [("A4", (0.297, 0.210)), ("Letter", (0.2794, 0.2159))])
def test_sheet_recovers_scale(paper, dims):
    sheet = (0.15, -1.6, dims[0], dims[1], 27.0)
    views = [_view((0.0, 0.0), 0, 48, sheet, seed=1), _view((0.6, -0.3), -20, 55, sheet, seed=2)]
    found = find_sheet(views, floor_y=0.0)
    assert found is not None and found.paper == paper and found.views == 2
    assert abs(found.long_raw / (dims[0] * UNIT) - 1) < 0.006
    cue = S.paper_cue(found.long_raw, found.paper, found.sigma)
    assert abs(math.exp(cue.log_scale) - 1 / UNIT) / (1 / UNIT) < 0.006
    assert cue.sigma == pytest.approx(0.015)


def test_square_tile_is_not_a_sheet():
    views = [_view((0.0, 0.0), 0, 50, (0.1, -1.5, 0.25, 0.25, 10.0), seed=3)]
    assert find_sheet(views, floor_y=0.0) is None


def test_sheet_on_a_table_is_rejected_by_depth():
    sheet = (0.15, -1.6, 0.297, 0.210, 15.0)
    v = _view((0.0, 0.0), 0, 48, sheet, seed=4)
    # the same image, but the depth map says the surface is 0.7 m closer to the camera (a table top)
    v.depth = np.clip(v.depth - 0.7 * UNIT, 0.05, None)
    assert find_sheet([v], floor_y=0.0) is None
