# 471948_47204566: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 8.54 [7.52, 9.56] m²; bounding box 5.08 x 1.82 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring (self-checked by wall-map entropy)
* runtime: 25.1 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 5.01 [4.91, 5.11] m² | 2.356 [2.343, 2.369] m | 4 | 2 |
| R2 Room 2 | 3.53 [2.87, 4.19] m² | 1.972 [1.956, 1.988] m | 4 | 3 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 2.749 [2.730, 2.767] m | 83% |
| R1-W2 | 1.823 [1.804, 1.842] m | 100% |
| R1-W3 | 2.749 [2.730, 2.767] m | 100% |
| R1-W4 | 1.823 [1.804, 1.842] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | open_passage | R1-W3 | 1.590 [0.960, 2.220] m | 2.360 [2.276, 2.444] m | - |
| R1-O2 | door | R1-W4 | 0.850 [0.758, 0.942] m | 2.000 [1.950, 2.050] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 2.184 [2.164, 2.204] m | 52% |
| R2-W2 | 1.616 [1.336, 1.896] m | 95% |
| R2-W3 | 2.184 [2.164, 2.204] m | 0% |
| R2-W4 | 1.616 [1.336, 1.896] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | open_passage | R2-W2 | 0.750 [0.682, 0.818] m | 1.970 [1.886, 2.054] m | R1 |
| R2-O2 | door | R2-W3 | 0.820 [0.637, 1.003] m | 1.720 [1.590, 1.850] m | - |
| R2-O3 | door | R2-W3 | 0.400 [0.217, 0.583] m | 1.600 [1.470, 1.730] m | - |

## Adjacency

* R1 - R2: door R1-O2, R2-O1 (rays through a shared opening)

## Warnings

* 1 tall-furniture face(s) in front of walls ignored when outlining rooms
