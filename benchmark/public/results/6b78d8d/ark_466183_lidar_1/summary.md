# 466183_45260920: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 23.02 [22.28, 23.77] m²; bounding box 5.39 x 5.20 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring
* runtime: 62.2 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 20.73 [20.61, 20.85] m² | 2.939 [2.926, 2.952] m | 8 | 6 |
| R2 Room 2 | 2.30 [1.83, 2.76] m² | 2.800 [2.200, 3.400] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 0.467 [0.449, 0.485] m | 89% |
| R1-W2 | 0.294 [0.275, 0.314] m | 100% |
| R1-W3 | 3.232 [3.209, 3.255] m | 77% |
| R1-W4 | 0.261 [0.240, 0.281] m | 100% |
| R1-W5 | 0.329 [0.307, 0.351] m | 100% |
| R1-W6 | 4.941 [4.923, 4.960] m | 85% |
| R1-W7 | 4.028 [4.011, 4.044] m | 100% |
| R1-W8 | 4.908 [4.891, 4.925] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | open_passage | R1-W3 | 2.740 [2.622, 2.858] m | 0.930 [0.846, 1.014] m | - |
| R1-O2 | door | R1-W6 | 1.030 [0.912, 1.148] m | 2.685 [2.616, 2.754] m | - |
| R1-O3 | window | R1-W6 | 0.995 [0.894, 1.096] m | 0.735 [0.670, 0.800] m | - |
| R1-O4 | window | R1-W6 | 1.340 [1.248, 1.432] m | 2.380 [2.268, 2.492] m | - |
| R1-O5 | door | R1-W8 | 0.865 [0.800, 0.930] m | 2.040 [1.956, 2.124] m | R2 |
| R1-O6 | open_passage | R1-W8 | 1.550 [0.890, 2.210] m | 1.630 [1.546, 1.714] m | - |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.183 [1.163, 1.204] m | 0% |
| R2-W2 | 1.939 [1.544, 2.334] m | 100% |
| R2-W3 | 1.183 [1.163, 1.204] m | 0% |
| R2-W4 | 1.939 [1.544, 2.334] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.765 [0.667, 0.863] m | 2.220 [2.136, 2.304] m | R1 |

## Adjacency

* R1 - R2: door R1-O5, R2-O1 (rays through a shared opening)

## Warnings

* 12 tall-furniture face(s) in front of walls ignored when outlining rooms
* R2: ceiling not observed; reported as a bounded prior interval
