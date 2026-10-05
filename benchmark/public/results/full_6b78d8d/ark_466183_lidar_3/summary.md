# 466183_45260928: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 22.53 [21.76, 23.30] m²; bounding box 5.42 x 4.99 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring
* runtime: 41.3 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 20.34 [20.21, 20.47] m² | 2.936 [2.923, 2.949] m | 6 | 7 |
| R2 Room 2 | 2.19 [1.70, 2.67] m² | 2.800 [2.200, 3.400] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 0.432 [0.413, 0.450] m | 100% |
| R1-W2 | 0.089 [0.069, 0.110] m | 100% |
| R1-W3 | 3.651 [3.632, 3.670] m | 100% |
| R1-W4 | 4.992 [4.973, 5.011] m | 89% |
| R1-W5 | 4.083 [4.065, 4.101] m | 100% |
| R1-W6 | 4.902 [4.884, 4.921] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | door | R1-W3 | 1.200 [0.866, 1.534] m | 2.670 [2.586, 2.754] m | - |
| R1-O2 | door | R1-W3 | 1.230 [1.095, 1.365] m | 2.720 [2.636, 2.804] m | - |
| R1-O3 | door | R1-W4 | 1.180 [1.062, 1.298] m | 2.660 [2.610, 2.710] m | - |
| R1-O4 | window | R1-W4 | 1.145 [1.031, 1.259] m | 0.740 [0.680, 0.800] m | - |
| R1-O5 | window | R1-W4 | 1.370 [1.274, 1.466] m | 2.015 [1.923, 2.107] m | - |
| R1-O6 | door | R1-W6 | 0.970 [0.867, 1.073] m | 1.975 [1.933, 2.017] m | R2 |
| R1-O7 | window | R1-W6 | 0.510 [0.392, 0.628] m | 0.420 [0.308, 0.532] m | - |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.228 [1.207, 1.250] m | 0% |
| R2-W2 | 1.780 [1.385, 2.175] m | 97% |
| R2-W3 | 1.228 [1.207, 1.250] m | 0% |
| R2-W4 | 1.780 [1.385, 2.175] m | 84% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.770 [0.669, 0.871] m | 2.010 [1.926, 2.094] m | R1 |

## Adjacency

* R1 - R2: door R1-O6, R2-O1 (rays through a shared opening)

## Warnings

* 14 tall-furniture face(s) in front of walls ignored when outlining rooms
* R2: ceiling not observed; reported as a bounded prior interval
