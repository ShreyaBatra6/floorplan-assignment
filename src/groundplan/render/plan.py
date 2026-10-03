"""Rendered floor plans (SVG / PNG / PDF) from a Plan document.

The stitched plan is drawn the way a homeowner knows from consumer scanning apps: filled rooms,
solid walls with door swings and window glazing, every wall dimensioned, each room labelled with
its area and ceiling height. Intervals are printed with the numbers, because the interval is part
of the measurement.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Polygon as MplPolygon, Rectangle  # noqa: E402

from groundplan.contract import Measurement, Plan, Room  # noqa: E402

ROOM_COLORS = ["#E8F1FB", "#FDF1E3", "#EAF6EC", "#F6EAF5", "#FFF8DC", "#E9F4F4", "#F3EFE6", "#EEF0FA"]
WALL = "#2B2F36"
WALL_T = 0.10
DIM = "#3D5A80"
DAMAGE = {"water_stain": "#1F77B4", "mold": "#2CA02C", "crack": "#D62728", "hole": "#9467BD",
          "peeling_paint": "#FF7F0E", "other": "#7F7F7F"}


def fmt_len(m: Measurement, with_ci: bool = True) -> str:
    if not with_ci:
        return f"{m.value:.2f} m"
    half = (m.hi - m.lo) / 2
    if half < 0.1:
        return f"{m.value:.2f} m ±{half * 100:.1f} cm"
    return f"{m.value:.2f} m ({m.lo:.2f}–{m.hi:.2f})"


def fmt_area(m: Measurement) -> str:
    return f"{m.value:.1f} m² ({m.lo:.1f}–{m.hi:.1f})"


def _outward_strip(a: np.ndarray, b: np.ndarray, outward: np.ndarray, t: float) -> np.ndarray:
    return np.array([a, b, b + outward * t, a + outward * t])


def _draw_room(ax, room: Room, color: str, dims: bool, detail: bool, fontsize: float) -> None:
    poly = np.array(room.polygon)
    ax.add_patch(MplPolygon(poly, closed=True, facecolor=color, edgecolor="none", zorder=1))
    for w in room.walls:
        a, b = np.array(w.start), np.array(w.end)
        ang = math.radians(w.outward_normal_deg)
        outward = np.array([math.cos(ang), math.sin(ang)])
        strip = _outward_strip(a, b, outward, WALL_T)
        ax.add_patch(MplPolygon(strip, closed=True, facecolor=WALL, edgecolor=WALL, lw=0.3, zorder=3))
        if dims and w.length.value >= (0.15 if detail else 0.3):
            _dimension(ax, a, b, outward, w.length, fontsize, offset=WALL_T + (0.28 if detail else 0.22),
                       observed=w.observed_fraction)
    for op in room.openings:
        wall = next(w for w in room.walls if w.id == op.wall_id)
        a, b = np.array(wall.start), np.array(wall.end)
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        ang = math.radians(wall.outward_normal_deg)
        outward = np.array([math.cos(ang), math.sin(ang)])
        p0 = a + u * op.offset.value
        p1 = p0 + u * op.width.value
        gap = _outward_strip(p0, p1, outward, WALL_T)
        ax.add_patch(MplPolygon(gap, closed=True, facecolor="white", edgecolor="none", zorder=4))
        if op.kind == "window":
            for frac in (0.25, 0.5, 0.75):
                q0, q1 = p0 + outward * WALL_T * frac, p1 + outward * WALL_T * frac
                ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color="#5DA9E9", lw=1.0, zorder=5)
            ax.plot([p0[0], p0[0] + outward[0] * WALL_T], [p0[1], p0[1] + outward[1] * WALL_T], color=WALL, lw=1, zorder=5)
            ax.plot([p1[0], p1[0] + outward[0] * WALL_T], [p1[1], p1[1] + outward[1] * WALL_T], color=WALL, lw=1, zorder=5)
        elif op.kind == "door":
            inward = -outward
            leaf_end = p0 + inward * op.width.value
            ax.plot([p0[0], leaf_end[0]], [p0[1], leaf_end[1]], color=WALL, lw=1.0, zorder=5)
            start_deg = math.degrees(math.atan2(u[1], u[0]))
            end_deg = math.degrees(math.atan2(inward[1], inward[0]))
            t1, t2 = sorted((start_deg, end_deg))
            if t2 - t1 > 180:
                t1, t2 = t2, t1 + 360
            ax.add_patch(Arc(p0, 2 * op.width.value, 2 * op.width.value, theta1=t1, theta2=t2,
                             color=WALL, lw=0.6, ls="--", zorder=5))
        if detail:
            mid = (p0 + p1) / 2 - outward * 0.25
            ax.text(mid[0], mid[1], f"{op.kind.replace('_', ' ')} {fmt_len(op.width)}", fontsize=fontsize * 0.8,
                    ha="center", va="center", color="#7A4E00", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#E0C080", lw=0.5))


def _dimension(ax, a, b, outward, m: Measurement, fontsize, offset, observed) -> None:
    pa, pb = a + outward * offset, b + outward * offset
    ax.annotate("", xy=pa, xytext=pb, arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.6, shrinkA=0, shrinkB=0),
                zorder=6)
    for p, q in ((a, pa), (b, pb)):
        ax.plot([p[0] + outward[0] * WALL_T, q[0] + outward[0] * 0.04], [p[1] + outward[1] * WALL_T, q[1] + outward[1] * 0.04],
                color=DIM, lw=0.4, zorder=6)
    mid = (pa + pb) / 2 + outward * 0.09
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    if ang > 90 or ang <= -90:
        ang += 180
    style = "normal" if observed >= 0.3 else "italic"
    color = DIM if observed >= 0.3 else "#9AA5B1"
    ax.text(mid[0], mid[1], fmt_len(m), fontsize=fontsize, rotation=ang, rotation_mode="anchor", ha="center",
            va="center", color=color, style=style, zorder=7)


def _damage(ax, plan: Plan, room_ids: set[str]) -> None:
    walls = {w.id: w for r in plan.rooms for w in r.walls}
    for d in plan.damage_regions:
        if d.room_id not in room_ids:
            continue
        if d.surface_id in walls:
            w = walls[d.surface_id]
            a, b = np.array(w.start), np.array(w.end)
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            us = [p[0] for p in d.polygon_uv]
            p0, p1 = a + u * min(us), a + u * max(us)
            ang = math.radians(w.outward_normal_deg)
            inward = -np.array([math.cos(ang), math.sin(ang)])
            q0, q1 = p0 + inward * 0.06, p1 + inward * 0.06
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=DAMAGE[d.damage_class], lw=4, solid_capstyle="butt", zorder=8)
            mid = (q0 + q1) / 2 + inward * 0.18
            ax.text(mid[0], mid[1], d.id, fontsize=6, color=DAMAGE[d.damage_class], ha="center", zorder=8)
        else:
            pts = np.array(d.polygon_uv)
            ax.add_patch(MplPolygon(pts, closed=True, facecolor=DAMAGE[d.damage_class], alpha=0.35,
                                    edgecolor=DAMAGE[d.damage_class], hatch="//" if d.surface_id.endswith("CEILING") else None,
                                    zorder=2))
            c = pts.mean(axis=0)
            ax.text(c[0], c[1], d.id, fontsize=6, color=DAMAGE[d.damage_class], ha="center", zorder=8)


def render_plan(plan: Plan, out_base: Path, room_id: str | None = None, formats=("svg", "png")) -> list[Path]:
    rooms = [r for r in plan.rooms if room_id is None or r.id == room_id]
    if not rooms:
        return []
    detail = room_id is not None
    allpts = np.vstack([np.array(r.polygon) for r in rooms])
    lo, hi = allpts.min(axis=0) - 1.2, allpts.max(axis=0) + 1.2
    span = hi - lo
    scale = 1.6 if detail else 1.25  # inches per metre
    fig_w = float(np.clip(span[0] * scale, 6, 22))
    fig_h = float(np.clip(span[1] * scale, 5, 22)) + 0.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fontsize = 7.5 if detail else 6.5
    for k, r in enumerate(rooms):
        color = ROOM_COLORS[(plan.rooms.index(r)) % len(ROOM_COLORS)]
        _draw_room(ax, r, color, dims=True, detail=detail, fontsize=fontsize)
        c = _label_point(np.array(r.polygon))
        txt = f"{r.name}\n{fmt_area(r.floor_area)}\nceiling {r.ceiling_height.value:.2f} m"
        if not r.coverage.ceiling_observed:
            txt += " (not seen)"
        ax.text(c[0], c[1], txt, ha="center", va="center", fontsize=fontsize + 1, color="#1B263B", zorder=9,
                linespacing=1.3, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))
    _damage(ax, plan, {r.id for r in rooms})
    # scale bar
    sb0 = np.array([lo[0] + 0.3, lo[1] + 0.3])
    ax.add_patch(Rectangle(sb0, 1.0, 0.06, color=WALL, zorder=9))
    ax.text(sb0[0] + 0.5, sb0[1] + 0.14, "1 m", ha="center", fontsize=7, zorder=9)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    cal = "calibrated" if plan.calibration.fitted else "provisional (prior) intervals"
    title = (f"{plan.capture.id}  ·  {plan.capture.tier.upper()} tier  ·  "
             f"{'room ' + room_id if detail else 'stitched plan'}  ·  {int(plan.confidence_level * 100)}% intervals, {cal}")
    if not detail:
        title += f"\nfootprint {fmt_area(plan.stitched.footprint_area)}  ·  {len(plan.rooms)} rooms"
    ax.set_title(title, fontsize=9, color="#1B263B", loc="left")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_base.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=160 if fmt == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


def _label_point(poly: np.ndarray) -> np.ndarray:
    """A point well inside the polygon (pole of inaccessibility approximation)."""
    from shapely.geometry import Polygon

    P = Polygon(poly)
    if not P.is_valid or P.area <= 0:
        return poly.mean(axis=0)
    best, best_d = np.array(P.representative_point().coords[0]), -1.0
    minx, miny, maxx, maxy = P.bounds
    for x in np.linspace(minx, maxx, 15):
        for y in np.linspace(miny, maxy, 15):
            from shapely.geometry import Point

            pt = Point(x, y)
            if P.contains(pt):
                d = P.exterior.distance(pt)
                if d > best_d:
                    best, best_d = np.array([x, y]), d
    return best
