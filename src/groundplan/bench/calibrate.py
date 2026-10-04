"""Split-conformal calibration of the interval multipliers, scored leave-one-capture-out.

For every benchmark item (a measured quantity with a laser truth) the score is
``s = |error| / half-width`` of the interval that was reported (the half on the side of the truth).
The conformal factor ``f`` is the ``ceil((n+1)(1-alpha))``-th smallest score; scaling every interval
of that tier and quantity by ``f`` makes the nominal 90 % hold on exchangeable data. Coverage is
reported leave-one-capture-out (fit on the other captures, test on the held-out one), which is the
number that predicts the walk-in test; the in-sample number is shown next to it only for contrast.

Quantities with fewer than ``min_n`` items borrow the tier's pooled factor; tiers with too few
items keep their prior and stay marked ``calibrated: false``.

Before the interval fit, a systematic LiDAR depth-scale error is estimated (``fit_depth_scale``):
the robust median of truth / measured over LiDAR wall lengths and ceiling heights. It is adopted
only when correcting with a factor fitted on the other captures lowers the held-out capture's error,
summed over all captures; the interval multipliers are then fitted on the corrected values, so the
two stay consistent with what the next run will report.
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


SCALE_KINDS = ("wall_length", "ceiling_height")  # set by the depth scale, not by edge or segmentation errors
LENGTH_KINDS = ("wall_length", "ceiling_height", "opening_width", "opening_height", "damage_extent")
AREA_KINDS = ("floor_area", "footprint_area")


def _ratio(items: list[dict]) -> float:
    return float(np.exp(np.median([math.log(it["truth"] / it["value"]) for it in items])))


def fit_depth_scale(metrics: dict, min_n: int = 6, min_captures: int = 2) -> dict:
    """Systematic LiDAR depth-scale correction, validated leave-one-capture-out."""
    s0 = float(metrics.get("corrections_used", {}).get("lidar", {}).get("depth_scale", 1.0))
    sel = [(c["capture"], it) for c in metrics["captures"] if c["tier"] == "lidar" for it in c["items"]
           if it["kind"] in SCALE_KINDS and it["value"] > 0 and it["truth"] > 0]
    caps = sorted({cap for cap, _ in sel})
    out = {"previous": s0, "n": len(sel), "captures": len(caps), "adopted": False, "depth_scale": s0}
    if len(sel) < min_n or len(caps) < min_captures:
        out["reason"] = f"needs >= {min_n} LiDAR wall/ceiling items over >= {min_captures} captures"
        return out
    r = _ratio([it for _, it in sel])
    err_raw, err_cor = [], []
    for held in caps:
        r_train = _ratio([it for cap, it in sel if cap != held])
        for cap, it in sel:
            if cap == held:
                err_raw.append(abs(it["value"] - it["truth"]))
                err_cor.append(abs(it["value"] * r_train - it["truth"]))
    mae_raw, mae_cor = float(np.mean(err_raw)), float(np.mean(err_cor))
    out.update(ratio=round(r, 5), loo_mae_before_m=round(mae_raw, 4), loo_mae_after_m=round(mae_cor, 4))
    if mae_cor < 0.95 * mae_raw:
        out.update(adopted=True, depth_scale=round(s0 * r, 5))
    else:
        out["reason"] = "held-out error does not improve by 5 %: keeping the current correction"
    return out


def _rescaled(it: dict, r: float) -> dict:
    k = r if it["kind"] in LENGTH_KINDS else (r * r if it["kind"] in AREA_KINDS else 1.0)
    return {**it, "value": it["value"] * k, "lo": it["lo"] * k, "hi": it["hi"] * k}


def fit(metrics: dict, min_n: int = 8) -> dict:
    depth = fit_depth_scale(metrics)
    r = depth["depth_scale"] / depth["previous"] if depth["adopted"] else 1.0
    items = [(c["capture"], c["tier"], _rescaled(it, r) if c["tier"] == "lidar" else it)
             for c in metrics["captures"] for it in c["items"]]
    report = {"alpha": ALPHA, "factors": {}, "loo": {}, "depth_scale": depth}
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


def apply(report: dict, source: str, path: Path = DEFAULT_PATH, out: Path | None = None) -> Path:
    """Fold the fit into the calibration at ``path`` and write the result to ``out`` (default ``path``)."""
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
    depth = report.get("depth_scale", {})
    if depth.get("adopted"):
        cal.setdefault("corrections", {}).setdefault("lidar", {})["depth_scale"] = depth["depth_scale"]
    cal["version"] = f"fit-{dt.date.today().isoformat()}"
    cal["fitted"] = fitted_any
    cal["source"] = source
    cal["fit_report"] = report
    cal["history"] = history
    out = Path(out or path)
    out.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out
