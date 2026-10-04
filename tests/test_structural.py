"""The texture-free photo registration recovers a rectangular room exactly from wall observations."""

import numpy as np

from groundplan.tiers.structural import WallObs, rotate_facing, solve_room

# room walls: x = 0 (faces +x), x = 4.2 (faces -x), y = 0 (faces +y), y = 3.3 (faces -y)
WALLS = {0: 0.0, 1: 4.2, 2: 0.0, 3: 3.3}


def observe(cam, k, scale, facings):
    """What a photo at ``cam`` rotated by k*90 deg (room <- photo) with depth scale 1/scale reports."""
    obs = []
    for f_room in facings:
        axis = 0 if f_room in (0, 1) else 1
        rel = WALLS[f_room] - cam[axis]  # room-frame offset of the wall from the camera
        # find the photo-frame facing that rotates into f_room
        f_photo = next(f for f in range(4) if rotate_facing(f, k) == f_room)
        # the offset along the photo axis: invert the rotation of the vector rel * e_axis
        v = np.array([rel, 0.0]) if axis == 0 else np.array([0.0, rel])
        for _ in range((4 - k) % 4):
            v = np.array([-v[1], v[0]])
        off = v[0] if f_photo in (0, 1) else v[1]
        obs.append(WallObs(f_photo, off / scale, 1.0))
    return obs


def test_four_middle_of_wall_photos():
    cams = [(2.1, 0.3), (3.9, 1.6), (2.0, 3.0), (0.3, 1.7)]
    ks = [0, 1, 2, 3]
    scales = [1.0, 1.08, 0.93, 1.03]
    # from the middle of a wall: front wall and both side walls
    views = [[3, 0, 1], [0, 2, 3], [2, 0, 1], [1, 2, 3]]
    obs = [observe(np.array(c), k, s, v) for c, k, s, v in zip(cams, ks, scales, views)]
    heights = [2.6 / s for s in scales]  # same physical room height, in each photo's own depth units
    res = solve_room(obs, ks, room_heights=heights)
    assert res.ok, res.rms
    width = res.walls[1] - res.walls[0]
    depth = res.walls[3] - res.walls[2]
    assert abs(width - 4.2) < 1e-4 and abs(depth - 3.3) < 1e-4
    assert np.allclose(res.scales, scales, atol=1e-4)


def test_back_wall_prior_alone_bounds_the_room():
    cams = [(2.1, 0.45), (3.75, 1.6), (2.0, 2.85), (0.45, 1.7)]
    ks = [0, 1, 2, 3]
    scales = [1.0, 1.05, 0.95, 1.02]
    views = [[3, 0, 1], [0, 2, 3], [2, 0, 1], [1, 2, 3]]
    backs = [2, 1, 3, 0]  # the wall behind each photographer
    obs = [observe(np.array(c), k, s, v) for c, k, s, v in zip(cams, ks, scales, views)]
    res = solve_room(obs, ks, back_walls=backs)
    assert res.ok
    assert abs((res.walls[1] - res.walls[0]) - 4.2) < 0.02
    assert abs((res.walls[3] - res.walls[2]) - 3.3) < 0.02
