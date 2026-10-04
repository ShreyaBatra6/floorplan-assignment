"""Video-tier helpers that do not need structure from motion: revisit pairs, frame thinning,
view-change tracking and the joining safeguards."""

import math

import cv2
import numpy as np

from groundplan.io.video import VideoFrame, _thin, _ViewChange
from groundplan.tiers.video import SfMResult, _plausible, _same_transform, revisit_pairs


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_revisit_pairs_link_a_return_to_the_same_place():
    rng = np.random.default_rng(0)
    places = _unit(rng.normal(size=(6, 64)))
    # 60 frames walking through places 0..5, then back to place 1 at frames 50-59
    seq = [places[min(i // 10, 5)] for i in range(50)] + [places[1]] * 10
    D = _unit(np.array(seq) + 0.05 * rng.normal(size=(60, 64)))
    pairs = revisit_pairs(D, min_gap=16)
    assert pairs, "the return to place 1 must produce revisit pairs"
    assert all(abs(i - j) >= 16 for i, j in pairs)
    assert sum(10 <= i < 20 and j >= 50 for i, j in pairs) >= 0.8 * len(pairs)


def test_thin_drops_the_frames_that_moved_least():
    frames = [VideoFrame(i, i / 30, np.zeros((2, 2, 3), np.uint8), 1.0, m)
              for i, m in enumerate([0, 0.2, 0.01, 0.2, 0.02, 0.2, 0.2])]
    kept = _thin(frames, 5)
    assert [f.index for f in kept] == [0, 1, 3, 5, 6]


def test_view_change_grows_with_camera_motion():
    rng = np.random.default_rng(1)
    big = cv2.GaussianBlur((rng.random((400, 700)) * 255).astype(np.uint8), (0, 0), 2)
    tr = _ViewChange()
    tr.reset(big[:240, :320])
    small, _ = tr.update(big[:240, 8:328])  # one frame of a pan: 8 px of 320
    for x in range(16, 57, 8):  # the pan continues frame by frame to 56 px
        large, tracked = tr.update(big[:240, x:x + 320])
    assert abs(small - 8 / 320) < 0.005 and abs(large - 56 / 320) < 0.01 and tracked > 0.6


def _T(yaw_deg=0.0, t=(0, 0, 0)):
    a = math.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    T[:3, 3] = t
    return T


def test_transform_agreement_and_plausibility():
    assert _same_transform(_T(10, (1, 0, 2)), _T(12, (1.1, 0, 2)))
    assert not _same_transform(_T(10, (1, 0, 2)), _T(20, (1, 0, 2)))
    assert not _same_transform(_T(0, (1, 0, 2)), _T(0, (1.5, 0, 2)))
    base = SfMResult({k: _T(0, (0.3 * k, 0, 0)) for k in range(10)}, np.eye(3), np.zeros((0, 3)), {}, 10, "")
    piece = SfMResult({k: _T(0, (0.2 * (k - 20), 0, 0)) for k in range(20, 26)}, np.eye(3), np.zeros((0, 3)), {}, 6, "")
    assert _plausible(_T(0, (2.0, 0, 0)), base, piece)
    assert not _plausible(_T(0, (1e13, 0, 0)), base, piece)
