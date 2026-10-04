"""Split-conformal calibration of the interval multipliers, scored leave-one-capture-out.

For every benchmark item (a measured quantity with a laser truth) the score is
``s = |error| / half-width`` of the interval that was reported (the half on the side of the truth).
The conformal factor ``f`` is the ``ceil((n+1)(1-alpha))``-th smallest score; scaling every interval
of that tier and quantity by ``f`` makes the nominal 90 % hold on exchangeable data. Coverage is
reported leave-one-capture-out (fit on the other captures, test on the held-out one), which is the
number that predicts the walk-in test; the in-sample number is shown next to it only for contrast.

Quantities with fewer than ``min_n`` items borrow the tier's pooled factor; tiers with too few
items keep their prior and stay marked ``calibrated: false``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from groundplan.calib.intervals import DEFAULT_PATH

ALPHA = 0.10


def _score(it: dict) -> float:
    half = (it["hi"] - it["value"]) if it["truth"] >= it["value"] else (it["value"] - it["lo"])
    return abs(it["truth"] - it["value"]) / max(half, 1e-9)


def conformal_factor(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float(np.max(scores)) * 1.25 if n else 1.0  # too few points for the quantile: be conservative
    return float(np.sort(scores)[k - 1])


def fit(metrics: dict, min_n: int = 8) -> dict:
    items = [(c["capture"], c["tier"], it) for c in metrics["captures"] for it in c["items"]]
    report = {"alpha": ALPHA, "factors": {}, "loo": {}}
    for tier in sorted({t for _, t, _ in items}):
        tier_items = [(cap, it) for cap, t, it in items if t == tier]
        pooled = np.array([_score(it) for _, it in tier_items])
        pooled_f = conformal_factor(pooled) if len(pooled) >= min_n else None
        kinds = sorted({it["kind"] for _, it in tier_items})
        report["factors"][tier] = {"default": pooled_f, "n": len(pooled)}
        for kind in kinds:
            sel = [(cap, it) for cap, it in tier_items if it["kind"] == kind]
            sc = np.array([_score(it) for _, it in sel])
            f = conformal_factor(sc) if len(sc) >= min_n else pooled_f
            report["factors"][tier][kind] = f
            # leave-one-capture-out coverage
            caps = sorted({cap for cap, _ in sel})
            hits, total = 0, 0
            for held in caps:
                train = np.array([_score(it) for cap, it in tier_items if cap != held and (it["kind"] == kind or len(sc) < min_n)])
                if len(train) < 3:
                    continue
                ff = conformal_factor(train)
                test = [it for cap, it in sel if cap == held]
                hits += sum(_score(it) <= ff for it in test)
                total += len(test)
            report["loo"][f"{tier}/{kind}"] = {"n": len(sc), "factor": f,
                                               "in_sample_coverage": float(np.mean(sc <= f)) if f else None,
                                               "loo_coverage": (hits / total) if total else None}
    return report


def apply(report: dict, source: str, path: Path = DEFAULT_PATH) -> Path:
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    history = cal.pop("history", [])
    history.append({k: cal[k] for k in ("version", "fitted", "source", "tiers") if k in cal})
    fitted_any = False
    for tier, factors in report["factors"].items():
        table = cal["tiers"].setdefault(tier, {})
        default = table.get("default", {"abs_mult": 1.0, "log_mult": 1.0, "abs_floor": 0.0, "log_floor": 0.0})
        for kind, f in factors.items():
            if kind == "n" or f is None:
                continue
            base = table.get(kind, default)
            table[kind] = {"abs_mult": round(base["abs_mult"] * f, 4), "log_mult": round(base["log_mult"] * f, 4),
                           "abs_floor": round(base["abs_floor"] * f, 5), "log_floor": round(base.get("log_floor", 0.0) * f, 5)}
            fitted_any = True
    cal["version"] = f"fit-{dt.date.today().isoformat()}"
    cal["fitted"] = fitted_any
    cal["source"] = source
    cal["fit_report"] = report
    cal["history"] = history
    Path(path).write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8", newline="\n")
    return Path(path)
