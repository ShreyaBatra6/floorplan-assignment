"""Fix-loop evidence: floor-to-ceiling height straight from raw LiDAR depth and ARKit poses (no
pipeline code), for several ways of reading the floor and ceiling layers.

    python fixloop/evidence/level_bands.py <Stray Scanner folder>

Normals come from neighbouring depth pixels; the floor and ceiling are the lowest and highest
dense horizontal layers (5 mm histogram); the height is printed for medians over bands of
+-6, 4, 2 and 1 cm around them, plus the mode and skew of each layer's points."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from groundplan.io.stray import load_stray  # noqa: E402

cap = load_stray(Path(sys.argv[1]))
Y, NY = [], []
for i in range(0, cap.n, 6):
    if not cap.has_depth[i]:
        continue
    d = cv2.imread(str(cap.depth_path(i)), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    K = cap.K_depth(i)
    hgt, wid = d.shape
    vv, uu = np.mgrid[0:hgt, 0:wid]
    X = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1)
    W = X @ cap.T_wc[i][:3, :3].T + cap.T_wc[i][:3, 3]
    dx = W[1:-1, 2:] - W[1:-1, :-2]
    dy = W[2:, 1:-1] - W[:-2, 1:-1]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    ok = (d[1:-1, 1:-1] > 0.2) & (d[1:-1, 1:-1] < 4.0)
    Y.append(W[1:-1, 1:-1][ok][:, 1])
    NY.append(n[ok][:, 1])
y, ny = np.concatenate(Y), np.concatenate(NY)
horiz = np.abs(ny) > 0.95
hist, e = np.histogram(y[horiz], bins=np.arange(y.min(), y.max() + 0.005, 0.005))
c = (e[:-1] + e[1:]) / 2
strong = np.flatnonzero(hist > 0.1 * hist.max())
f0, c0 = c[strong[0]], c[strong[-1]]
print(Path(sys.argv[1]).name)
for band in (0.06, 0.04, 0.02, 0.01):
    fl = np.median(y[horiz & (np.abs(y - f0) < band)])
    ce = np.median(y[horiz & (np.abs(y - c0) < band)])
    print(f"  band +-{band * 100:.0f} cm: floor {fl:.4f} ceiling {ce:.4f} height {ce - fl:.4f}")
# mode of a fine histogram (the surface itself, ignoring rugs / fixtures to one side)
for lab, y0 in (("floor", f0), ("ceiling", c0)):
    s = y[horiz & (np.abs(y - y0) < 0.06)]
    hh, ee = np.histogram(s, bins=np.arange(y0 - 0.06, y0 + 0.0605, 0.0025))
    k = np.argmax(np.convolve(hh, np.ones(3) / 3, "same"))
    print(f"  {lab}: mode {(ee[k] + ee[k + 1]) / 2:.4f}, mean {s.mean():.4f}, skew sign {np.sign(s.mean() - np.median(s)):+.0f}")
