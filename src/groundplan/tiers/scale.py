"""Metric scale for the photo and video tiers: independent cues fused in log space.

A monocular depth model gives good *shape* (about 5 % relative error once scaled) but its metric
*scale* drifts from scene to scene: on the assessor captures Depth Anything V2 Metric-Indoor-Small
over-estimates depth by a median factor 1/0.684 with a per-frame log-sd of 0.15-0.20. So scale is
not taken from the model alone. Each cue below says what correction it implies, with an honest
log-sigma; cues are combined by inverse variance, and a cue that disagrees with the others by more
than 3 sigma is dropped (and reported). The fused sigma is what widens every photo/video interval.

Cues (sigmas are priors until the benchmark calibrates them):
* model:   calibrated model factor (default 0.684, measured on LiDAR frames)      sigma 0.12
* ceiling: room ceiling height vs residential prior 2.60 m                        sigma 0.08
* door:    detected door head heights vs 2.04 m (US 2.03, EU 2.0-2.05, IN 2.1)    sigma 0.035 per door
* camera:  hand-held camera height vs 1.45 m (photos) / 1.35 m (walking video)    sigma 0.10
* paper:   an A4/Letter sheet on the floor, when found                            sigma 0.015
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from groundplan.contract import ScaleCue, ScaleInfo

MODEL_FACTOR = 0.684
MODEL_SIGMA = 0.12
CEILING_PRIOR_M, CEILING_SIGMA = 2.60, 0.08
DOOR_HEAD_M, DOOR_SIGMA = 2.04, 0.035
CAMERA_PHOTO_M, CAMERA_VIDEO_M, CAMERA_SIGMA = 1.45, 1.35, 0.10
PAPER_SIGMA = 0.015


@dataclass
class Cue:
    name: str
    log_scale: float
    sigma: float
    detail: str


def model_cue(factor: float = MODEL_FACTOR, sigma: float = MODEL_SIGMA) -> Cue:
    return Cue("model", math.log(factor), sigma, f"depth model x{factor:.3f} (calibrated on LiDAR frames)")


def ceiling_cue(heights_raw: list[float]) -> Cue | None:
    h = [x for x in heights_raw if x and x > 0]
    if not h:
        return None
    m = float(np.median(h))
    return Cue("ceiling", math.log(CEILING_PRIOR_M / m), CEILING_SIGMA / math.sqrt(1 + 0.25 * (len(h) - 1)),
               f"ceiling {m:.2f} (raw) vs prior {CEILING_PRIOR_M} m over {len(h)} room(s)")


def door_cue(heads_raw: list[float]) -> Cue | None:
    h = [x for x in heads_raw if 1.0 < x * MODEL_FACTOR < 3.0] if heads_raw else []
    if not h:
        return None
    m = float(np.median(h))
    return Cue("door", math.log(DOOR_HEAD_M / m), DOOR_SIGMA / math.sqrt(len(h)) + 0.01,
               f"{len(h)} door head(s) median {m:.2f} (raw) vs {DOOR_HEAD_M} m")


def camera_cue(heights_raw: list[float], tier: str) -> Cue | None:
    h = [x for x in heights_raw if x and x > 0]
    if not h:
        return None
    m = float(np.median(h))
    prior = CAMERA_PHOTO_M if tier == "photo" else CAMERA_VIDEO_M
    return Cue("camera", math.log(prior / m), CAMERA_SIGMA, f"camera height {m:.2f} (raw) vs prior {prior} m")


def paper_cue(length_raw: float | None, paper: str = "A4") -> Cue | None:
    if not length_raw:
        return None
    long_side = 0.297 if paper == "A4" else 0.2794
    return Cue("paper", math.log(long_side / length_raw), PAPER_SIGMA, f"{paper} sheet long side {length_raw:.3f} (raw)")


def fuse(cues: list[Cue]) -> tuple[float, float, ScaleInfo]:
    """Inverse-variance fusion in log space with 3-sigma rejection. Returns (scale, sigma_log, info)."""
    active = [c for c in cues if c is not None]
    used = {c.name: True for c in active}
    while True:
        w = np.array([1 / c.sigma**2 for c in active if used[c.name]])
        x = np.array([c.log_scale for c in active if used[c.name]])
        if len(w) == 0:
            return 1.0, 0.5, ScaleInfo(source="none", sigma_log=0.5, cues=[])
        mu = float(np.sum(w * x) / np.sum(w))
        sig = float(1 / math.sqrt(np.sum(w)))
        worst, worst_z = None, 0.0
        if sum(used.values()) >= 3:
            for c in active:
                if not used[c.name]:
                    continue
                # compare with the fusion of the *other* cues
                wo = np.array([1 / o.sigma**2 for o in active if used[o.name] and o is not c])
                xo = np.array([o.log_scale for o in active if used[o.name] and o is not c])
                mo = float(np.sum(wo * xo) / np.sum(wo))
                so = float(1 / math.sqrt(np.sum(wo)))
                z = abs(c.log_scale - mo) / math.hypot(c.sigma, so)
                if z > worst_z:
                    worst, worst_z = c, z
        if worst is not None and worst_z > 3.0:
            used[worst.name] = False
            continue
        break
    info = ScaleInfo(
        source="fused: " + "+".join(c.name for c in active if used[c.name]),
        sigma_log=round(sig, 4),
        cues=[ScaleCue(name=c.name, log_scale=round(c.log_scale, 4), sigma_log=round(c.sigma, 4), used=used[c.name],
                       detail=c.detail) for c in active],
    )
    return math.exp(mu), sig, info
