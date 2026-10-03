"""Per-tier error budgets: the systematic terms added to every statistical fit.

A plane fitted to 50 000 LiDAR points has a statistical standard error far below a millimetre, and
reporting that would be confident garbage: no phone measures a room to a tenth of a millimetre.
These terms put the irreducible error sources back. They are first-principles priors (documented
in the technical report's error budget); the benchmark then corrects the whole budget per tier
through the calibration multipliers rather than by editing these numbers to fit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorBudget:
    # additive terms, metres (1 sigma)
    face_bias: float  # residual depth/scale bias of one surface after calibration
    face_drift: float  # residual pose drift of one surface after drift correction
    definition: float  # where exactly is "the wall" (skirting, plaster waviness, out-of-plumb)
    opening_jamb: float  # localisation of one jamb edge
    level: float  # floor or ceiling surface level (non-flatness, sensor bias)
    unobserved_wall: float  # wall face never seen: bounded by the free-space edge only
    # multiplicative term (log scale), 1 sigma; 0 for metric sensors
    scale: float = 0.0


LIDAR = ErrorBudget(face_bias=0.003, face_drift=0.002, definition=0.003, opening_jamb=0.015,
                    level=0.003, unobserved_wall=0.08)
VIDEO = ErrorBudget(face_bias=0.010, face_drift=0.010, definition=0.005, opening_jamb=0.03,
                    level=0.010, unobserved_wall=0.15)
PHOTO = ErrorBudget(face_bias=0.020, face_drift=0.015, definition=0.008, opening_jamb=0.04,
                    level=0.015, unobserved_wall=0.25)

BUDGETS = {"lidar": LIDAR, "video": VIDEO, "photo": PHOTO}

# Bounded prior for a ceiling the capture never saw: residential ceilings are 2.2-3.4 m in
# practice. Reported with that interval (and a warning), never as a confident number.
CEILING_PRIOR = (2.2, 3.4)
