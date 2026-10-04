"""Benchmark calibration: depth-scale correction and conformal multipliers on synthetic score sheets."""

import numpy as np

from groundplan.bench.calibrate import conformal_factor, fit, fit_depth_scale


def _metrics(bias: float, noise_m: float = 0.004, captures: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    caps = []
    for c in range(captures):
        items = []
        for k in range(10):
            truth = rng.uniform(2.0, 5.0) if k < 7 else rng.uniform(2.4, 2.7)
            value = truth * (1 + bias) + rng.normal(0, noise_m)
            items.append({"kind": "wall_length" if k < 7 else "ceiling_height", "room": "r", "ref": f"W{k}",
                          "truth": truth, "value": value, "lo": value - 0.02, "hi": value + 0.02})
        caps.append({"capture": f"cap{c}", "tier": "lidar", "items": items})
    return {"captures": caps, "corrections_used": {"lidar": {"depth_scale": 1.0}}}


def test_depth_scale_bias_is_found_and_adopted():
    d = fit_depth_scale(_metrics(bias=0.015))
    assert d["adopted"]
    assert abs(d["depth_scale"] - 1 / 1.015) < 0.002
    assert d["loo_mae_after_m"] < d["loo_mae_before_m"]


def test_no_bias_keeps_the_correction():
    d = fit_depth_scale(_metrics(bias=0.0))
    assert not d["adopted"] and d["depth_scale"] == 1.0


def test_single_capture_is_not_enough():
    d = fit_depth_scale(_metrics(bias=0.02, captures=1))
    assert not d["adopted"] and "captures" in d["reason"]


def test_intervals_are_fitted_on_corrected_values():
    rep = fit(_metrics(bias=0.015))
    # after removing the bias, +-2 cm intervals cover the 4 mm noise easily: factor well below 1
    assert rep["depth_scale"]["adopted"]
    assert rep["factors"]["lidar"]["wall_length"] < 0.6


def test_conformal_factor_quantile():
    s = np.arange(1, 20) / 10.0  # 19 scores -> k = ceil(20 * 0.9) = 18th smallest
    assert conformal_factor(s) == 1.8
