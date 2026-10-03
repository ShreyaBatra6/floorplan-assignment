"""Per-tier interval calibration.

``calibration.json`` holds, for each tier and each measured quantity, the multipliers applied to
the analytic error budget before an interval is built. Until a benchmark fit exists they are
conservative priors and every Measurement is emitted with ``calibrated: false``.

``groundplan bench calibrate`` refits them with split-conformal leave-one-capture-out on the
benchmark and rewrites this file; the fit and its coverage are reported, never hand-edited.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("calibration.json")


@dataclass(frozen=True)
class Multipliers:
    abs_mult: float = 1.0
    log_mult: float = 1.0
    abs_floor: float = 0.0
    log_floor: float = 0.0


@dataclass
class Calibration:
    version: str
    fitted: bool
    source: str
    confidence: float
    tiers: dict[str, dict[str, Multipliers]] = field(default_factory=dict)
    corrections: dict[str, dict[str, float]] = field(default_factory=dict)

    def multipliers(self, tier: str, kind: str) -> Multipliers:
        table = self.tiers.get(tier, {})
        return table.get(kind) or table.get("default") or Multipliers()

    def correction(self, tier: str, name: str, default: float = 0.0) -> float:
        return float(self.corrections.get(tier, {}).get(name, default))


def _parse(raw: dict) -> Calibration:
    tiers: dict[str, dict[str, Multipliers]] = {}
    for tier, kinds in raw.get("tiers", {}).items():
        tiers[tier] = {kind: Multipliers(**vals) for kind, vals in kinds.items()}
    return Calibration(
        version=raw["version"],
        fitted=bool(raw.get("fitted", False)),
        source=raw.get("source", "prior"),
        confidence=float(raw.get("confidence", 0.9)),
        tiers=tiers,
        corrections=raw.get("corrections", {}),
    )


@lru_cache(maxsize=8)
def _load(path: str) -> Calibration:
    with open(path, encoding="utf-8") as fh:
        return _parse(json.load(fh))


def load_calibration(path: str | os.PathLike | None = None) -> Calibration:
    chosen = path or os.environ.get("GROUNDPLAN_CALIBRATION") or DEFAULT_PATH
    return _load(str(Path(chosen).resolve()))
