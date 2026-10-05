# 422378_42447310: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 15.90 [15.06, 16.74] m²; bounding box 5.19 x 4.94 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring (self-checked by wall-map entropy)
* runtime: 39.5 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 10.42 [10.32, 10.52] m² | 2.279 [2.266, 2.292] m | 4 | 2 |
| R2 Room 2 | 5.48 [4.94, 6.02] m² | 2.303 [2.290, 2.316] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 3.215 [3.198, 3.231] m | 100% |
| R1-W2 | 3.242 [3.225, 3.258] m | 100% |
| R1-W3 | 3.215 [3.198, 3.231] m | 100% |
| R1-W4 | 3.242 [3.225, 3.258] m | 86% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | open_passage | R1-W3 | 1.720 [1.658, 1.782] m | 1.990 [1.906, 2.074] m | - |
| R1-O2 | door | R1-W4 | 0.910 [0.792, 1.028] m | 2.190 [2.106, 2.274] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.760 [1.743, 1.777] m | 23% |
| R2-W2 | 3.114 [2.834, 3.394] m | 42% |
| R2-W3 | 1.760 [1.743, 1.777] m | 0% |
| R2-W4 | 3.114 [2.834, 3.394] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.440 [0.344, 0.536] m | 2.160 [2.110, 2.210] m | R1 |

## Adjacency

* R1 - R2: door R1-O2, R2-O1 (rays through a shared opening)

## Warnings

* 1 tall-furniture face(s) in front of walls ignored when outlining rooms
