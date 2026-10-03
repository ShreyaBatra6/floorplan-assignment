"""Frames and conventions.

* Camera frame: OpenCV (x right, y down, z forward). Stray Scanner's ``odometry.csv`` poses map
  OpenCV-convention camera points into the ARKit world frame: checked empirically on the sample
  captures, where this convention makes depth from different frames agree to 3-10 mm while the
  OpenGL convention disagrees by 20-40 cm.
* World frame: ARKit (gravity-aligned, y up, right-handed).
* Plan frame: 2D, x right / y up on the drawing, seen from above:
  ``plan = R(yaw) @ (world_x, -world_z)``, where ``yaw`` aligns the dominant wall direction with
  the plan axes (Manhattan alignment) so rasterised walls fall on grid rows and columns.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

UP = np.array([0.0, 1.0, 0.0])


def pose_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class PlanFrame:
    """Mapping between world (y-up) coordinates and the Manhattan-aligned 2D plan frame."""

    def __init__(self, yaw: float = 0.0, origin: np.ndarray | None = None, floor_y: float = 0.0):
        self.yaw = float(yaw)
        self.R = rot2(-self.yaw)
        self.origin = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
        self.floor_y = float(floor_y)

    def to_plan(self, world: np.ndarray) -> np.ndarray:
        world = np.asarray(world, dtype=float)
        xz = np.stack([world[..., 0], -world[..., 2]], axis=-1)
        return xz @ self.R.T - self.origin

    def dir_to_plan(self, world_dirs: np.ndarray) -> np.ndarray:
        world_dirs = np.asarray(world_dirs, dtype=float)
        xz = np.stack([world_dirs[..., 0], -world_dirs[..., 2]], axis=-1)
        return xz @ self.R.T

    def height(self, world: np.ndarray) -> np.ndarray:
        return np.asarray(world, dtype=float)[..., 1] - self.floor_y

    def to_world(self, plan_xy: np.ndarray, height: np.ndarray | float = 0.0) -> np.ndarray:
        plan_xy = np.asarray(plan_xy, dtype=float)
        xz = (plan_xy + self.origin) @ self.R  # inverse rotation
        h = np.broadcast_to(np.asarray(height, dtype=float), xz.shape[:-1])
        return np.stack([xz[..., 0], h + self.floor_y, -xz[..., 1]], axis=-1)


def yaw_from_quaternion_world(T_wc: np.ndarray) -> float:
    """Heading of the camera's optical axis projected on the floor (world frame, radians)."""
    fwd = T_wc[:3, 2]
    return float(np.arctan2(-fwd[2], fwd[0]))
