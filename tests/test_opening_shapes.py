"""Opening shape rules: what reaches the floor must be tall enough to walk through."""

import numpy as np

from groundplan.geometry.openings import OpeningParams, detect_openings


def _wall_with_holes(holes, length=4.0, height=2.5, step=0.02):
    s, h = np.meshgrid(np.arange(0, length, step), np.arange(0, height, step))
    s, h = s.ravel(), h.ravel()
    keep = np.ones(len(s), bool)
    for s0, s1, h0, h1 in holes:
        keep &= ~((s > s0) & (s < s1) & (h > h0) & (h < h1))
    return s[keep], h[keep]


def _rays_through(holes, cam=(2.0, 2.0), cam_h=1.4, step=0.02):
    ends_xy, ends_h = [], []
    for s0, s1, h0, h1 in holes:
        ss, hh = np.meshgrid(np.arange(s0 + step, s1, step), np.arange(h0 + step, h1, step))
        for sv, hv in zip(ss.ravel(), hh.ravel()):
            t = 1.6  # the ray crosses the wall plane (y = 0) and ends 1.2 m beyond it
            ends_xy.append([cam[0] + (sv - cam[0]) * t, cam[1] + (0.0 - cam[1]) * t])
            ends_h.append(cam_h + (hv - cam_h) * t)
    n = len(ends_xy)
    return (np.tile(cam, (n, 1)).astype(float), np.full(n, cam_h), np.array(ends_xy), np.array(ends_h))


def test_low_floor_level_gap_is_not_an_opening_but_the_door_is():
    door = (3.0, 3.9, -0.1, 2.05)
    low_gap = (0.6, 2.4, -0.1, 0.7)  # 1.8 m wide, 0.7 m tall, at floor level (under furniture)
    s, h = _wall_with_holes([door, low_gap])
    solid_xy = np.c_[s, np.zeros_like(s)]
    rs_xy, rs_h, re_xy, re_h = _rays_through([door, low_gap])
    cands = detect_openings(1, 0.0, 1, np.array([0.0, 0.0]), np.array([4.0, 0.0]), 2.5, solid_xy, h,
                            rs_xy, rs_h, re_xy, re_h, None, OpeningParams(reject_low_gaps=True))
    assert len(cands) == 1
    c = cands[0]
    assert c.kind == "door" and abs(c.width - 0.9) < 0.03 and c.s0 > 2.8
