"""ARKitScenes stand-in benchmark: pose convention and the laser ground-truth procedure, checked on
synthetic inputs with known answers (the real data is fetched by scripts/arkitscenes.py)."""

import numpy as np
from scipy.spatial.transform import Rotation

from groundplan.bench import arkitscenes as A


def test_traj_poses_map_opencv_camera_into_a_y_up_world(tmp_path):
    # camera at (1, 2, 3) in the z-up world, looking along +x, level: OpenCV axes x=-y_w, y=-z_w, z=+x_w
    R_wc = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    C = np.eye(4)
    C[:3, :3], C[:3, 3] = R_wc, [1, 2, 3]
    E = np.linalg.inv(C)  # the file stores world-to-camera
    rv = Rotation.from_matrix(E[:3, :3]).as_rotvec()
    (tmp_path / "t.traj").write_text(f"10.0 {rv[0]} {rv[1]} {rv[2]} {E[0, 3]} {E[1, 3]} {E[2, 3]}\n")
    stamps, T = A.read_traj(tmp_path / "t.traj")
    assert stamps[0] == 10.0
    assert np.allclose(T[0][:3, 3], [1, 3, -2])  # (x, y, z)_z-up -> (x, z, -y)_y-up
    assert np.allclose(T[0][:3, 1], [0, -1, 0])  # image-down points down
    assert np.allclose(T[0][:3, 2], [1, 0, 0])  # looking along +x


def _plane(u0, u1, v0, v1, fn, step=0.02, hole=None, rng=None):
    us, vs = np.meshgrid(np.arange(u0, u1, step), np.arange(v0, v1, step))
    us, vs = us.ravel(), vs.ravel()
    if hole is not None:
        a, b, c, d = hole
        keep = ~((us > a) & (us < b) & (vs > c) & (vs < d))
        us, vs = us[keep], vs[keep]
    p = fn(us, vs)
    return p + (rng.normal(0, 0.001, p.shape) if rng is not None else 0)


def _synthetic_room(rng):
    """4.0 x 3.0 m room, ceiling 2.6 m; a 0.90 x 2.05 m door in wall S at 1.00 m from the SW corner,
    a hallway wall 1.5 m behind it; a 2.0 m tall wardrobe 0.6 m in front of wall E."""
    W, D, H = 4.0, 3.0, 2.6
    pts = [
        _plane(0, W, 0, D, lambda u, v: np.c_[u, v, np.zeros_like(u)], rng=rng),  # floor
        _plane(0, W, 0, D, lambda u, v: np.c_[u, v, np.full_like(u, H)], rng=rng),  # ceiling
        _plane(0, W, 0, H, lambda u, v: np.c_[u, np.zeros_like(u), v], hole=(1.0, 1.9, -1, 2.05), rng=rng),  # S
        _plane(0, W, 0, H, lambda u, v: np.c_[u, np.full_like(u, D), v], rng=rng),  # N
        _plane(0, D, 0, H, lambda u, v: np.c_[np.zeros_like(u), u, v], rng=rng),  # W
        _plane(0, D, 0, H, lambda u, v: np.c_[np.full_like(u, W), u, v], rng=rng),  # E
        _plane(0.2, 2.8, 0, H, lambda u, v: np.c_[u, np.full_like(u, -1.5), v], rng=rng),  # hallway, seen through the door
        _plane(0.8, 2.2, 0, 2.0, lambda u, v: np.c_[np.full_like(u, W - 0.6), u, v], rng=rng),  # wardrobe front
    ]
    return np.vstack(pts)


def test_laser_truth_measures_a_box_room_its_ceiling_and_its_door():
    rng = np.random.default_rng(0)
    xyz = _synthetic_room(rng)
    yaw = np.radians(23.0)  # the scan is not axis-aligned
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    xyz = xyz @ Rz.T + [5.0, -2.0, 290.0]  # site offset, like the real scans
    scanner = np.array([[1.5, 1.6, 1.4]]) @ Rz.T + [5.0, -2.0, 290.0]
    sc = A.prepare_scene(xyz)
    rooms = A.visit_truth(sc, scanner)
    assert len(rooms) == 1
    r = rooms[0]
    assert abs(r["walls"]["S"] - 4.0) < 0.005 and abs(r["walls"]["E"] - 3.0) < 0.005  # not stopped by the wardrobe
    assert abs(r["ceiling_height"] - 2.6) < 0.005
    doors = [o for w in r["openings"].values() for o in w]
    assert len(doors) == 1 and doors[0]["kind"] == "door"
    assert abs(doors[0]["width"] - 0.90) < 0.015
    assert abs(doors[0]["height"] - 2.05) < 0.03
