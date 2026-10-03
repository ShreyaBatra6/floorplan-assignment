"""Estimates with an explicit error budget, and their conversion to reported Measurements.

An ``Estimate`` carries two independent error components:

* ``sigma_abs``: additive 1-sigma error in the quantity's own unit (plane-fit noise, sensor depth
  noise, resolution, residual sensor bias, pose drift), and
* ``sigma_log``: multiplicative 1-sigma error in log space (global metric-scale uncertainty; it
  dominates the photo and video tiers, where scale comes from a learned depth model and priors).

``finalize`` turns an estimate into a contract ``Measurement``. It applies the per-tier,
per-quantity calibration (multipliers and an absolute floor, see ``calib/calibration.json``) and
builds an asymmetric interval: the scale part is symmetric in log space, the additive part
symmetric in linear space, and the two half-widths are combined in quadrature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from groundplan.calib.intervals import Calibration, load_calibration
from groundplan.contract import Measurement

_Z = {0.8: 1.2815515655446004, 0.9: 1.6448536269514722, 0.95: 1.959963984540054}

NONNEGATIVE_UNITS = {"m", "m2", "ea"}


def z_for(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    from scipy.stats import norm

    return float(norm.ppf(0.5 + confidence / 2.0))


@dataclass(frozen=True)
class Estimate:
    value: float
    sigma_abs: float = 0.0
    sigma_log: float = 0.0
    unit: str = "m"
    method: str = ""

    def with_method(self, method: str) -> Estimate:
        return replace(self, method=method)

    def scaled(self, k: float) -> Estimate:
        """Multiply by an exactly known constant (unit conversion, geometric factor)."""
        return replace(self, value=self.value * k, sigma_abs=self.sigma_abs * abs(k))

    def plus_abs(self, sigma: float) -> Estimate:
        """Add an independent additive error term (in quadrature)."""
        return replace(self, sigma_abs=math.hypot(self.sigma_abs, sigma))


def combine_sum(parts: list[Estimate], method: str = "sum") -> Estimate:
    """Sum of independent additive errors; scale errors are fully correlated (one global scale)."""
    value = sum(p.value for p in parts)
    sigma_abs = math.sqrt(sum(p.sigma_abs**2 for p in parts))
    # A shared global scale multiplies every part, so the sum inherits the largest log sigma.
    sigma_log = max((p.sigma_log for p in parts), default=0.0)
    unit = parts[0].unit if parts else "m"
    return Estimate(value, sigma_abs, sigma_log, unit, method)


def finalize(
    est: Estimate,
    kind: str,
    tier: str,
    calibration: Calibration | None = None,
    confidence: float | None = None,
) -> Measurement:
    cal = calibration or load_calibration()
    conf = confidence or cal.confidence
    m = cal.multipliers(tier, kind)
    z = z_for(conf)

    value = float(est.value)
    sigma_abs = math.hypot(est.sigma_abs * m.abs_mult, m.abs_floor)
    sigma_log = est.sigma_log * m.log_mult + m.log_floor

    h_abs = z * sigma_abs
    lo_scale = value * math.exp(-z * sigma_log)
    hi_scale = value * math.exp(z * sigma_log)
    lo = value - math.hypot(h_abs, abs(value - lo_scale))
    hi = value + math.hypot(h_abs, abs(hi_scale - value))
    if est.unit in NONNEGATIVE_UNITS:
        lo = max(lo, 0.0)

    return Measurement(
        value=round(value, 6),
        lo=round(lo, 6),
        hi=round(hi, 6),
        unit=est.unit,  # type: ignore[arg-type]
        confidence=conf,
        sigma_abs=round(sigma_abs, 6),
        sigma_log=round(sigma_log, 6),
        method=est.method or kind,
        calibrated=cal.fitted,
    )


def contains(measurement: Measurement, truth: float) -> bool:
    return measurement.lo <= truth <= measurement.hi
