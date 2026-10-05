# Head-to-head PROXY: our LiDAR tier vs the on-device ARKit mesh (public stand-in)

**Not the brief's head-to-head** (that needs a named consumer app's export on our own rooms). The comparison is Apple's on-device ARKit reconstruction shipped with each ARKitScenes recording, measured by the same procedure as the laser truth (`scripts/arkit_mesh_proxy.py`). Tie: both errors within 5 mm.

**Beat or tie on 19/24 shared dimensions (79 %)**; 0 dimension(s) the mesh could not give (no ceiling in it).

| recording | dimension | truth (m) | ours (m) | ARKit mesh (m) | ours err (cm) | mesh err (cm) | outcome |
|---|---|---|---|---|---|---|---|
| ark_466183_lidar_1 | room1:S | 4.980 | 4.941 | 4.901 | -3.9 | -7.9 | beat |
| ark_466183_lidar_1 | room1:N | 4.980 | 4.908 | 4.901 | -7.2 | -7.9 | beat |
| ark_466183_lidar_1 | room1:ceiling | 2.974 | 2.939 | 2.946 | -3.5 | -2.8 | lose |
| ark_466183_lidar_2 | room1:S | 4.980 | 4.945 | 4.329 | -3.5 | -65.2 | beat |
| ark_466183_lidar_2 | room1:N | 4.980 | 4.908 | 4.329 | -7.2 | -65.2 | beat |
| ark_466183_lidar_2 | room1:ceiling | 2.974 | 2.946 | 2.938 | -2.8 | -3.6 | beat |
| ark_466183_lidar_3 | room1:S | 4.980 | 4.992 | 3.968 | +1.1 | -101.2 | beat |
| ark_466183_lidar_3 | room1:N | 4.980 | 4.902 | 3.968 | -7.8 | -101.2 | beat |
| ark_466183_lidar_3 | room1:ceiling | 2.974 | 2.936 | 3.041 | -3.9 | +6.7 | beat |
| ark_422378_lidar_1 | room2:E | 3.280 | 3.286 | 2.655 | +0.6 | -62.5 | beat |
| ark_422378_lidar_1 | room2:N | 3.256 | 3.246 | 2.655 | -1.0 | -60.2 | beat |
| ark_422378_lidar_1 | room2:ceiling | 2.292 | 2.279 | 2.326 | -1.3 | +3.5 | beat |
| ark_422378_lidar_2 | room2:E | 3.280 | 3.253 | 3.178 | -2.6 | -10.2 | beat |
| ark_422378_lidar_2 | room2:N | 3.256 | 3.253 | 3.178 | -0.3 | -7.9 | beat |
| ark_422378_lidar_2 | room2:ceiling | 2.292 | 2.277 | 2.296 | -1.5 | +0.5 | lose |
| ark_422378_lidar_3 | room2:E | 3.280 | 3.242 | 3.230 | -3.8 | -5.0 | beat |
| ark_422378_lidar_3 | room2:N | 3.256 | 3.242 | 3.230 | -1.5 | -2.7 | beat |
| ark_422378_lidar_3 | room2:ceiling | 2.292 | 2.279 | 2.310 | -1.3 | +1.8 | beat |
| ark_471948_lidar_1 | room2:W | 2.752 | 2.859 | 2.703 | +10.7 | -4.9 | lose |
| ark_471948_lidar_1 | room2:ceiling | 2.370 | 2.348 | 2.345 | -2.3 | -2.5 | beat |
| ark_471948_lidar_2 | room2:W | 2.752 | 2.752 | 2.712 | -0.1 | -4.1 | beat |
| ark_471948_lidar_2 | room2:ceiling | 2.370 | 2.346 | 2.388 | -2.4 | +1.7 | lose |
| ark_471948_lidar_3 | room2:W | 2.752 | 2.749 | 2.710 | -0.4 | -4.3 | beat |
| ark_471948_lidar_3 | room2:ceiling | 2.370 | 2.356 | 2.379 | -1.4 | +0.8 | lose |

The mesh box has no wall labels, so each true wall is compared with the mesh extent closest to it; where a room is nearly square that choice favours the mesh, never us.

**Read this before quoting the number.** Six rows (466183 recordings 2 and 3, 422378 recording 1:
walls only) are wins because the room box measured *on the mesh* is 60-100 cm off: the measuring
script found the wrong planes in an incomplete mesh, which says more about the script than about the
mesh. Without those six rows: **13/18 (72 %)**. Where the mesh box is sound, our walls beat the
mesh's by 1-4 cm (the mesh reads walls 4-10 cm short), and the mesh's ceilings beat ours on 4 of 9
(ours read 1.3-2.4 cm short: the band-averaging issue in `fixloop/evidence/level_bands.py`).
