"""Texture-free registration of a room's photos from the walls they see.

In a rectangular room there is exactly one wall facing each of the four Manhattan directions, so
every photo that sees "the wall facing +x" sees the same wall. Each photo is already gravity- and
Manhattan-aligned; its compass heading picks which of the four 90-degree rotations brings it into
the room's frame. A wall seen at offset ``o`` (in the photo's own depth units) then gives one
equation that is *linear* in the unknowns:

    x_i + s_i * o  =  X_wall        (and the same along y for the other two facings)

with camera position (x_i, y_i), relative depth scale s_i of photo i and the wall position X_wall.
All photos and walls are solved together by weighted least squares (photo 0 fixes the gauge:
x = y = 0, s = 1); far walls are down-weighted because monocular depth error grows with
distance. Residuals report whether the rectangle assumption held; if it did not (an L-shaped room,
a photo assigned the wrong rotation) the caller falls back to feature-based registration.

The capture protocol makes this well posed: from the middle of each wall the opposite wall and
both side walls are in view, so one photo already spans the room's width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# facing index -> (axis, sign): axis 0 = x (plan x), axis 1 = y (plan y = -world z)
FACING_AXES = {0: (0, +1), 1: (0, -1), 2: (1, +1), 3: (1, -1)}


@dataclass
class WallObs:
    facing: int  # in the photo's own frame
    offset: float  # plan coordinate of the wall face relative to the camera (photo depth units)
    weight: float  # inverse variance proxy


@dataclass
class StructuralResult:
    positions: np.ndarray  # (n, 2) plan positions of the cameras in the room frame
    scales: np.ndarray  # (n,) relative depth scales
    walls: dict[int, float]  # room facing -> wall coordinate
    rms: float
    ok: bool


def rotate_facing(f: int, k: int) -> int:
    """Facing index after rotating the photo frame by k * 90 degrees counter-clockwise."""
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dx, dy = dirs[f]
    for _ in range(k % 4):
        dx, dy = -dy, dx
    return dirs.index((dx, dy))


def rotate_offset(offset_xy: tuple[float, float], k: int) -> tuple[float, float]:
    x, y = offset_xy
    for _ in range(k % 4):
        x, y = -y, x
    return x, y


def wall_observations(pts_plan: np.ndarray, normals_plan: np.ndarray, min_frac: float = 0.03,
                      heights: np.ndarray | None = None) -> list[WallObs]:
    """The room's wall per facing in one photo (plan coordinates relative to the camera).

    Points sharing a facing form layers along that axis: the wall itself, furniture fronts in front
    of it, and walls of the next room seen through a doorway behind it. With ``heights`` each layer
    is scored by its support times its vertical extent, because only the room's own wall spans
    (nearly) floor to ceiling across the whole view; without heights the densest layer is used.
    """
    from groundplan.geometry.walls import facing_of

    fac = facing_of(normals_plan)
    out = []
    n = len(pts_plan)
    for f, (axis, _sign) in FACING_AXES.items():
        sel = fac == f
        if sel.sum() < max(min_frac * n, 40):
            continue
        coord = pts_plan[sel, axis]
        hist, edges = np.histogram(coord, bins=60)
        centres = (edges[:-1] + edges[1:]) / 2
        cand = [int(np.argmax(hist))]
        if heights is not None:
            hs = heights[sel]
            peaks = [k for k in range(len(hist)) if hist[k] >= 0.3 * hist.max()
                     and hist[k] >= hist[max(k - 1, 0)] and hist[k] >= hist[min(k + 1, len(hist) - 1)]]
            scored = []
            for k in peaks:
                layer = np.abs(coord - centres[k]) < 0.12 * max(abs(centres[k]), 0.5)
                if layer.sum() < 20:
                    continue
                span = float(np.percentile(hs[layer], 95) - np.percentile(hs[layer], 5))
                scored.append((layer.sum() * span, k))
            if scored:
                cand = [max(scored)[1]]
        o = float(centres[cand[0]])
        near = np.abs(coord - o) < 0.12 * max(abs(o), 0.5)
        o = float(np.median(coord[near]))
        dist = max(abs(o), 0.3)
        out.append(WallObs(f, o, float(near.sum()) / (dist**2)))
    return out


def solve_room(obs: list[list[WallObs]], rotations: list[int], scale_prior_sigma: float = 0.15,
               max_rms: float = 0.12, room_heights: list[float | None] | None = None,
               back_walls: list[int | None] | None = None, back_gap: float = 0.45,
               back_sigma: float = 0.15, height_rel_sigma: float = 0.03) -> StructuralResult:
    """Weighted least squares over camera positions, relative scales and the four wall positions.

    Two further constraints make relative scales observable even when photos share no pair of
    parallel walls: the floor-to-ceiling height is the same physical length in every photo
    (``room_heights`` in each photo's own depth units: s_i * h_i = h_0), and a photo taken with
    the photographer's back to a wall has that wall about ``back_gap`` behind the camera
    (``back_walls``: the room facing of the wall behind photo i).
    """
    n = len(obs)
    # unknown layout: x_1..x_{n-1}, y_1..y_{n-1}, s_1..s_{n-1}, X_0..X_3 (photo 0 fixed)
    nx = n - 1
    idx_x = lambda i: i - 1  # noqa: E731
    idx_y = lambda i: nx + i - 1  # noqa: E731
    idx_s = lambda i: 2 * nx + i - 1  # noqa: E731
    idx_w = lambda f: 3 * nx + f  # noqa: E731
    m = 3 * nx + 4
    rows, rhs, wts = [], [], []
    seen = set()
    for i, (photo_obs, k) in enumerate(zip(obs, rotations)):
        for o in photo_obs:
            f = rotate_facing(o.facing, k)
            axis, _ = FACING_AXES[f]
            # rotate the observed offset vector into the room frame: only its component on `axis` matters
            vec = [0.0, 0.0]
            vec[FACING_AXES[o.facing][0]] = o.offset
            ox, oy = rotate_offset(tuple(vec), k)
            off = ox if axis == 0 else oy
            row = np.zeros(m)
            b = 0.0
            if i > 0:
                row[idx_x(i) if axis == 0 else idx_y(i)] = 1.0
                row[idx_s(i)] = off
            else:
                b -= off  # x_0 = y_0 = 0, s_0 = 1
            row[idx_w(f)] = -1.0
            rows.append(row)
            rhs.append(b)
            wts.append(o.weight)
            seen.add(f)
    wmax = max(wts) if wts else 1.0
    n_obs = len(rows)
    if room_heights and room_heights[0]:
        h0 = room_heights[0]
        for i in range(1, n):
            hi = room_heights[i]
            if hi:
                row = np.zeros(m)
                row[idx_s(i)] = hi
                rows.append(row)
                rhs.append(h0)
                wts.append(wmax * (0.05 / (height_rel_sigma * h0)) ** 2)
    if back_walls:
        for i, f in enumerate(back_walls):
            if f is None:
                continue
            axis, sign = FACING_AXES[f]
            # the wall behind faces into the room (sign) and lies back_gap behind the camera:
            # cam_axis - X_f = sign * back_gap
            row = np.zeros(m)
            b = sign * back_gap
            if i > 0:
                row[idx_x(i) if axis == 0 else idx_y(i)] = 1.0
            row[idx_w(f)] = -1.0
            rows.append(row)
            rhs.append(b)
            wts.append(wmax * (0.05 / back_sigma) ** 2)
            seen.add(f)
    n_constraints = len(rows)
    for i in range(1, n):  # tie-breaking prior only: relative scales near 1 (matters if a scale is unconstrained)
        row = np.zeros(m)
        row[idx_s(i)] = 1.0
        rows.append(row)
        rhs.append(1.0)
        wts.append(wmax * 1e-6 / scale_prior_sigma**2)
    for f in range(4):  # walls never observed: pin them weakly so the system is solvable
        if f not in seen:
            row = np.zeros(m)
            row[idx_w(f)] = 1.0
            rows.append(row)
            rhs.append(0.0)
            wts.append(1e-6)
    A = np.array(rows)
    b = np.array(rhs)
    w = np.sqrt(np.array(wts) / max(np.max(wts), 1e-12))
    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    resid = (A @ sol - b)[:n_obs]
    rms = float(np.sqrt(np.mean(resid**2))) if len(resid) else float("inf")
    pos = np.zeros((n, 2))
    sc = np.ones(n)
    for i in range(1, n):
        pos[i] = (sol[idx_x(i)], sol[idx_y(i)])
        sc[i] = sol[idx_s(i)]
    walls = {f: float(sol[idx_w(f)]) for f in seen}
    # observability: the real constraints alone must fix every unknown (unseen walls excepted);
    # otherwise a perfect fit is meaningless and the tie-breaking prior would be deciding
    A_real = np.array(rows[:n_constraints])
    keep_cols = [c for c in range(m) if not (c >= 3 * nx and (c - 3 * nx) not in seen)]
    rank = int(np.linalg.matrix_rank(A_real[:, keep_cols], tol=1e-6)) if len(A_real) else 0
    observable = rank >= len(keep_cols)
    ok = bool(observable and rms <= max_rms and np.all(sc > 0.5) and np.all(sc < 2.0) and len(seen) >= 3)
    return StructuralResult(pos, sc, walls, rms, ok)
