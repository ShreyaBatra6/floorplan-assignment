"""Correspondence between a predicted Plan and the measured ground truth.

Rooms are matched by shape (Hungarian assignment on area and sorted wall-length signature); walls
within a matched room by the cyclic order of their lengths (walls are listed counter-clockwise in
both the plan and the ground truth, but the starting wall differs); openings by the wall they sit
on and their width. Nothing about a prediction is adjusted to fit: an unmatched ground-truth
opening is a miss and an unmatched predicted opening is a phantom, exactly as the brief scores it.

A manual room mapping (``{predicted_id: gt_name}``) can be supplied for ambiguous sites; it is
recorded in the score so a reader can see it was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from groundplan.bench.gt import GroundTruth, GTOpening, GTRoom
from groundplan.contract import Opening, Plan, Room, Wall


@dataclass
class WallPair:
    gt_id: str
    pred: Wall
    truth: float


@dataclass
class OpeningPair:
    gt: GTOpening | None
    pred: Opening | None


@dataclass
class RoomMatch:
    gt: GTRoom
    pred: Room
    walls: list[WallPair] = field(default_factory=list)
    wall_count_ok: bool = True
    openings: list[OpeningPair] = field(default_factory=list)


@dataclass
class PlanMatch:
    rooms: list[RoomMatch]
    unmatched_gt: list[str]
    unmatched_pred: list[str]
    manual: bool = False


def _signature(lengths: list[float], k: int = 4) -> np.ndarray:
    s = sorted(lengths, reverse=True)[:k]
    return np.array(s + [0.0] * (k - len(s)))


def match_rooms(plan: Plan, gt: GroundTruth, manual: dict[str, str] | None = None) -> PlanMatch:
    preds = plan.rooms
    if manual:
        pairs = [(next(p for p in preds if p.id == pid), gt.room(name)) for pid, name in manual.items()]
        used_p = {p.id for p, _ in pairs}
        used_g = {g.name for _, g in pairs}
    else:
        cost = np.zeros((len(preds), len(gt.rooms)))
        for i, p in enumerate(preds):
            for j, g in enumerate(gt.rooms):
                ga = g.area()
                if ga is None:  # partial truth (some walls not measurable): compare the walls it lists
                    pl = np.array([w.length.value for w in p.walls])
                    cost[i, j] = float(np.mean([np.min(np.abs(pl - w.length)) / w.length for w in g.walls])) * 4                         if g.walls and len(pl) else 9.0
                    continue
                cost[i, j] = abs(p.floor_area.value - ga) / ga + np.abs(
                    _signature([w.length.value for w in p.walls]) - _signature([w.length for w in g.walls])
                ).sum() / max(sum(w.length for w in g.walls[:4]), 1.0)
        rows, cols = linear_sum_assignment(cost) if cost.size else ([], [])
        pairs = [(preds[i], gt.rooms[j]) for i, j in zip(rows, cols) if cost[i, j] < 0.8]
        used_p = {p.id for p, _ in pairs}
        used_g = {g.name for _, g in pairs}
    rooms = [_match_room(p, g) for p, g in pairs]
    return PlanMatch(rooms, [g.name for g in gt.rooms if g.name not in used_g],
                     [p.id for p in preds if p.id not in used_p], manual=bool(manual))


def _match_room(pred: Room, gt: GTRoom) -> RoomMatch:
    m = RoomMatch(gt=gt, pred=pred)
    pl = np.array([w.length.value for w in pred.walls])
    gl = np.array([w.length for w in gt.walls])
    if len(pl) == len(gl) and len(gl) > 0:
        options = []
        for direction in (1, -1):
            seq = pl if direction == 1 else pl[::-1]
            for shift in range(len(pl)):
                order = list(range(len(pl))) if direction == 1 else list(range(len(pl)))[::-1]
                order = order[shift:] + order[:shift]
                options.append((float(np.abs(np.roll(seq, -shift) - gl).sum()), order))
        # a rectangle's walls (a, b, a, b) fit equally well turned by 180 degrees: among alignments
        # whose lengths fit about as well, take the one under which the most openings correspond
        best_err = min(o[0] for o in options)
        near = [o for o in options if o[0] <= best_err + 0.02 * len(gl)]

        def agreement(order):
            wall_map = {g.id: pred.walls[k].id for g, k in zip(gt.walls, order)}
            return sum(p.gt is not None and p.pred is not None for p in _match_openings(pred, gt, wall_map))

        _, order = max(near, key=lambda o: (agreement(o[1]), -o[0]))
        m.walls = [WallPair(g.id, pred.walls[k], g.length) for g, k in zip(gt.walls, order)]
    else:
        # different wall counts: each measured wall takes the closest unused predicted wall
        m.wall_count_ok = False
        free = set(range(len(pl)))
        for g in gt.walls:
            if not free:
                break
            k = min(free, key=lambda i: abs(pl[i] - g.length))
            free.discard(k)
            m.walls.append(WallPair(g.id, pred.walls[k], g.length))
    m.openings = _match_openings(pred, gt, {w.gt_id: w.pred.id for w in m.walls})
    return m


def _match_openings(pred: Room, gt: GTRoom, wall_map: dict[str, str]) -> list[OpeningPair]:
    pairs: list[OpeningPair] = []
    free = list(pred.openings)
    for g in gt.openings:
        cands = [o for o in free if (wall_map.get(g.wall) == o.wall_id or g.wall not in wall_map)
                 and _kind_compatible(o.kind, g.kind)]
        if not cands:
            pairs.append(OpeningPair(g, None))
            continue
        best = min(cands, key=lambda o: abs(o.width.value - g.width) + (0 if g.offset is None else
                                                                        0.5 * abs(o.offset.value - g.offset)))
        if abs(best.width.value - g.width) > max(0.25, 0.35 * g.width):
            pairs.append(OpeningPair(g, None))
            continue
        free.remove(best)
        pairs.append(OpeningPair(g, best))
    pairs += [OpeningPair(None, o) for o in free]
    return pairs


def _kind_compatible(pred: str, gt: str) -> bool:
    doorish = {"door", "open_passage"}
    return pred == gt or (pred in doorish and gt in doorish)
