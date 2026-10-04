"""Scope of work: line items keyed to surfaces, with quantities carried as intervals.

Codes are generic trade categories (DRY drywall, PNT painting, WTR water mitigation, MLD mold,
INS insulation, FC finish carpentry), not a proprietary price-list format; quantities follow
common restoration practice and every item says how its quantity was derived. Quantity intervals
are propagated from the measured extents (Monte Carlo over the inputs' intervals), so a scope
built on a thin capture carries the uncertainty of that capture.
"""

from __future__ import annotations

import math

import numpy as np

from groundplan.contract import ConcealedFlag, DamageRegion, Measurement, Room, ScopeItem

SF_PER_M2 = 10.7639
LF_PER_M = 3.28084
FLOOD_CUT_STEPS = (0.6, 1.2)  # flood cuts at 2 ft or 4 ft
JOIST = 0.4  # ceiling board removal extends to framing at ~16 in centres
MC = 400


def _sample(m: Measurement, rng: np.random.Generator) -> np.ndarray:
    """Samples consistent with a reported interval (normal in the interval's half-width)."""
    half = (m.hi - m.lo) / 2
    sigma = half / 1.645 if half > 0 else 0.0
    centre = (m.hi + m.lo) / 2
    return np.maximum(rng.normal(centre, sigma, MC), 0.0)


def _quantity(samples: np.ndarray, value: float, unit: str, method: str) -> Measurement:
    lo, hi = np.percentile(samples, [5, 95])
    lo, hi = min(lo, value), max(hi, value)
    return Measurement(value=round(float(value), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), unit=unit,
                       confidence=0.9, sigma_abs=round(float(np.std(samples)), 4), sigma_log=0.0, method=method,
                       calibrated=False)


class _Builder:
    def __init__(self) -> None:
        self.items: list[ScopeItem] = []
        self.rng = np.random.default_rng(0)

    def add(self, code, desc, room, surface, samples, value, unit, derivation, regions=(), flags=()):
        label = {"m2": "SF", "m": "LF", "ea": "EA"}[unit]
        factor = {"m2": SF_PER_M2, "m": LF_PER_M, "ea": 1.0}[unit]
        q = _quantity(samples, value, unit, derivation)
        self.items.append(ScopeItem(
            id=f"S{len(self.items) + 1}", code=code, description=desc, room_id=room, surface_id=surface,
            region_ids=list(regions), flag_ids=list(flags), quantity=q, unit_label=label,
            quantity_imperial=round(value * factor, 2), derivation=derivation,
        ))


def build_scope(rooms: list[Room], regions: list[DamageRegion], flags: list[ConcealedFlag]) -> list[ScopeItem]:
    b = _Builder()
    rng = b.rng
    by_room = {r.id: r for r in rooms}
    flags_by_region: dict[str, list[ConcealedFlag]] = {}
    for f in flags:
        for rid in f.region_ids:
            flags_by_region.setdefault(rid, []).append(f)
    painted: set[str] = set()

    for d in regions:
        room = by_room[d.room_id]
        fl = flags_by_region.get(d.id, [])
        fids = [f.id for f in fl]
        rule_ids = {f.rule_id for f in fl}
        surf = d.surface_id
        wall = next((w for w in room.walls if w.id == surf), None)
        w_s, h_s, a_s = _sample(d.width, rng), _sample(d.height, rng), _sample(d.area, rng)

        if wall is not None:
            L, H = wall.length, wall.height
            Ls, Hs = _sample(L, rng), _sample(H, rng)
            if d.damage_class == "water_stain" and "CDR-02" in rule_ids:
                top = max(v for _, v in d.polygon_uv)
                cut = next((c for c in FLOOD_CUT_STEPS if top + 0.3 <= c), FLOOD_CUT_STEPS[-1])
                b.add("WTR-FLDCUT", f"Flood cut wall board at {cut:.1f} m and dispose", room.id, surf, Ls, L.value, "m",
                      f"full wall length; cut height = stain top {top:.2f} m + 0.3 m, rounded up to {cut} m",
                      [d.id], fids)
                b.add("DRY-INST", f"Hang, tape and finish wall board, lower {cut:.1f} m", room.id, surf, Ls * cut,
                      L.value * cut, "m2", f"wall length x {cut} m", [d.id], fids)
                b.add("FC-BASE", "Remove and replace baseboard", room.id, surf, Ls, L.value, "m", "full wall length",
                      [d.id], fids)
            elif d.damage_class in ("water_stain", "peeling_paint"):
                m = 0.15
                b.add("PNT-SEAL" if d.damage_class == "water_stain" else "PNT-PREP",
                      "Seal stain with stain-blocking primer" if d.damage_class == "water_stain"
                      else "Scrape, sand and prime peeling paint",
                      room.id, surf, (w_s + 2 * m) * (h_s + 2 * m), (d.width.value + 2 * m) * (d.height.value + 2 * m),
                      "m2", f"region {d.width.value:.2f} x {d.height.value:.2f} m plus {m} m margin", [d.id], fids)
            elif d.damage_class == "mold":
                m = 0.6
                b.add("MLD-REM", "Mold remediation of wall board incl. 0.6 m margin", room.id, surf,
                      (w_s + 2 * m) * np.minimum(h_s + 2 * m, Hs), (d.width.value + 2 * m) * min(d.height.value + 2 * m, H.value),
                      "m2", "visible extent plus 2 ft margin (IICRC S520 practice), capped at wall height", [d.id], fids)
            elif d.damage_class == "crack":
                length = math.hypot(d.width.value, d.height.value)
                b.add("DRY-CRACK", "Repair crack: cut out, tape and finish", room.id, surf,
                      np.hypot(w_s, h_s) + 0.3, length + 0.3, "m", "crack diagonal extent + 0.3 m", [d.id], fids)
            elif d.damage_class == "hole":
                size = max(d.width.value, d.height.value)
                if size <= 0.3:
                    b.add("DRY-PATCH", f"Patch wall board hole ({'small' if size < 0.1 else 'medium'})", room.id, surf,
                          np.ones(MC), 1.0, "ea", f"largest dimension {size:.2f} m", [d.id], fids)
                else:
                    b.add("DRY-SECT", "Replace wall board section", room.id, surf, (w_s + 0.4) * (h_s + 0.4),
                          (d.width.value + 0.4) * (d.height.value + 0.4), "m2", "extent + 0.2 m each side to framing",
                          [d.id], fids)
            if surf not in painted:
                painted.add(surf)
                b.add("PNT-WALL", "Paint wall, two coats (whole wall for a uniform finish)", room.id, surf,
                      Ls * Hs, L.value * H.value, "m2", "wall length x wall height", [d.id], fids)
        else:  # ceiling or floor
            area_m = next(s.area for s in room.surfaces if s.id == surf)
            As = _sample(area_m, rng)
            if surf.endswith("CEILING"):
                if d.damage_class in ("water_stain", "mold") and d.area.value > 0.1:
                    b.add("DRY-CEIL", "Remove and replace ceiling board to framing", room.id, surf,
                          (w_s + 2 * JOIST) * (h_s + 2 * JOIST), (d.width.value + 2 * JOIST) * (d.height.value + 2 * JOIST),
                          "m2", f"extent + {JOIST} m each side (joist spacing)", [d.id], fids)
                    b.add("INS-RR", "Remove and replace wet insulation above", room.id, surf,
                          (w_s + 2 * JOIST) * (h_s + 2 * JOIST), (d.width.value + 2 * JOIST) * (d.height.value + 2 * JOIST),
                          "m2", "same area as the ceiling board removed", [d.id], fids)
                elif d.damage_class == "crack":
                    b.add("DRY-CRACK", "Repair ceiling crack", room.id, surf, np.hypot(w_s, h_s) + 0.3,
                          math.hypot(d.width.value, d.height.value) + 0.3, "m", "crack extent + 0.3 m", [d.id], fids)
                if surf not in painted:
                    painted.add(surf)
                    b.add("PNT-CEIL", "Seal and paint ceiling", room.id, surf, As, area_m.value, "m2", "ceiling area",
                          [d.id], fids)
            else:
                b.add("WTR-FLOOR", "Moisture-map floor; lift and dry covering in the affected area", room.id, surf,
                      a_s * 4, d.area.value * 4, "m2", "4x visible area (wicking spreads under covering)", [d.id], fids)
    # one drying/verification line per room with water damage
    wet_rooms = sorted({d.room_id for d in regions if d.damage_class in ("water_stain", "mold")})
    for rid in wet_rooms:
        room = by_room[rid]
        fl = [f.id for f in flags if f.room_id == rid]
        b.add("WTR-DRY", "Structural drying with moisture verification (per room)", rid, f"{rid}-FLOOR",
              np.ones(MC), 1.0, "ea", "one per room with water-related damage", [], fl)
    return b.items
