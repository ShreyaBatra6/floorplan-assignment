import math

import pytest

from groundplan.calib.intervals import Calibration, Multipliers
from groundplan.measure import Estimate, combine_sum, finalize, z_for


def _cal(**kw):
    return Calibration(
        version="test",
        fitted=False,
        source="test",
        confidence=0.9,
        tiers={"lidar": {"default": Multipliers(**kw)}},
    )


def test_z_values():
    assert z_for(0.9) == pytest.approx(1.6448536, rel=1e-6)
    assert z_for(0.5) == pytest.approx(0.6744898, rel=1e-5)


def test_additive_interval_is_symmetric():
    m = finalize(Estimate(3.0, sigma_abs=0.01), "wall_length", "lidar", _cal())
    assert m.hi - m.value == pytest.approx(m.value - m.lo)
    assert m.hi - m.value == pytest.approx(1.6448536 * 0.01, rel=1e-4)
    assert not m.calibrated


def test_scale_interval_is_log_symmetric():
    m = finalize(Estimate(4.0, sigma_log=0.05), "wall_length", "lidar", _cal())
    assert math.log(m.hi / m.value) == pytest.approx(math.log(m.value / m.lo), rel=1e-4)  # lo/hi stored to 6 dp


def test_multipliers_and_floor_widen():
    base = finalize(Estimate(2.5, sigma_abs=0.001), "x", "lidar", _cal())
    wide = finalize(Estimate(2.5, sigma_abs=0.001), "x", "lidar", _cal(abs_mult=2.0, abs_floor=0.004))
    assert wide.hi - wide.lo > base.hi - base.lo
    # the floor dominates a tiny analytic sigma
    assert wide.sigma_abs == pytest.approx(math.hypot(0.002, 0.004), rel=1e-4)


def test_nonnegative_clamp():
    m = finalize(Estimate(0.01, sigma_abs=1.0, unit="m2"), "x", "lidar", _cal())
    assert m.lo == 0.0


def test_combine_sum_shares_scale():
    total = combine_sum([Estimate(1.0, 0.003, 0.05), Estimate(2.0, 0.004, 0.05)])
    assert total.value == pytest.approx(3.0)
    assert total.sigma_abs == pytest.approx(0.005)
    assert total.sigma_log == pytest.approx(0.05)
