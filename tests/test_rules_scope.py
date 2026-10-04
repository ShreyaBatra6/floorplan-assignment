from groundplan.contract import DamageRegion, Measurement, Opening, Room, RoomCoverage, RoomType, Surface, Wall
from groundplan.damage.rules import evaluate, load_rules
from groundplan.damage.scope import build_scope


def M(v, unit="m", half=0.01):
    return Measurement(value=v, lo=v - half, hi=v + half, unit=unit, sigma_abs=half / 1.645, sigma_log=0.0,
                       method="test", calibrated=False)


def room() -> Room:
    walls = []
    pts = [(0, 0), (4, 0), (4, 3), (0, 3)]
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        L = abs(b[0] - a[0]) + abs(b[1] - a[1])
        walls.append(Wall(id=f"R1-W{i + 1}", room_id="R1", index=i + 1, surface_id=f"R1-W{i + 1}", start=a, end=b,
                          length=M(L), height=M(2.5), outward_normal_deg=0.0, observed_fraction=1.0))
    surfaces = [Surface(id=w.id, room_id="R1", kind="wall", wall_id=w.id, area=M(w.length.value * 2.5, "m2"))
                for w in walls]
    surfaces += [Surface(id="R1-FLOOR", room_id="R1", kind="floor", area=M(12.0, "m2")),
                 Surface(id="R1-CEILING", room_id="R1", kind="ceiling", area=M(12.0, "m2"))]
    window = Opening(id="R1-O1", room_id="R1", wall_id="R1-W2", kind="window", offset=M(1.0), width=M(1.2),
                     height=M(1.2), sill_height=M(0.9), detection_confidence=0.9)
    return Room(id="R1", name="Room 1", type=RoomType(label="bathroom", confidence=0.8, source="test"),
                polygon=pts, floor_area=M(12.0, "m2"), perimeter=M(14.0), ceiling_height=M(2.5), walls=walls,
                openings=[window], surfaces=surfaces,
                coverage=RoomCoverage(floor_observed_fraction=1, walls_observed_fraction=1, ceiling_observed=True))


def region(rid, surface, cls, uv, area=0.06) -> DamageRegion:
    us = [p[0] for p in uv]
    vs = [p[1] for p in uv]
    return DamageRegion(id=rid, room_id="R1", surface_id=surface, damage_class=cls, confidence=0.8, polygon_uv=uv,
                        area=M(area, "m2", 0.005), width=M(max(us) - min(us)), height=M(max(vs) - min(vs)), views=5)


def test_rule_ids_unique():
    rules = load_rules()
    ids = [r["id"] for r in rules["rules"]] + [r["id"] for r in rules["pair_rules"]]
    assert len(ids) == len(set(ids))


def test_wall_base_stain_fires_wicking_rule_and_flood_cut():
    r = room()
    d = region("D1", "R1-W1", "water_stain", [(1.0, 0.05), (1.6, 0.05), (1.6, 0.4), (1.0, 0.4)], 0.2)
    flags = evaluate([r], [d], {"R1": {"bathroom"}})
    fired = {f.rule_id for f in flags}
    assert {"CDR-02", "CDR-03"} <= fired
    scope = build_scope([r], [d], flags)
    codes = [s.code for s in scope]
    assert "WTR-FLDCUT" in codes and "FC-BASE" in codes and "PNT-WALL" in codes and "WTR-DRY" in codes
    cut = next(s for s in scope if s.code == "WTR-FLDCUT")
    assert cut.surface_id == "R1-W1" and abs(cut.quantity.value - 4.0) < 1e-6
    assert cut.quantity.lo <= 4.0 <= cut.quantity.hi


def test_window_and_crack_rules():
    r = room()
    stain = region("D1", "R1-W2", "water_stain", [(0.8, 1.0), (1.1, 1.0), (1.1, 1.3), (0.8, 1.3)])
    crack = region("D2", "R1-W2", "crack", [(2.15, 2.2), (2.4, 2.2), (2.4, 2.45), (2.15, 2.45)], 0.01)
    flags = evaluate([r], [stain, crack], {})
    fired = {(f.rule_id, f.region_ids[0]) for f in flags}
    assert ("CDR-04", "D1") in fired
    assert ("CDR-07", "D2") in fired


def test_wall_ceiling_pair_rule():
    r = room()
    wall_stain = region("D1", "R1-W1", "water_stain", [(1.0, 2.0), (1.5, 2.0), (1.5, 2.45), (1.0, 2.45)])
    ceil_stain = region("D2", "R1-CEILING", "water_stain", [(1.0, 0.1), (1.5, 0.1), (1.5, 0.5), (1.0, 0.5)], 0.2)
    flags = evaluate([r], [wall_stain, ceil_stain], {})
    pair = [f for f in flags if f.rule_id == "CDR-12"]
    assert len(pair) == 1 and set(pair[0].region_ids) == {"D1", "D2"}
    assert any(f.rule_id == "CDR-01" for f in flags)
