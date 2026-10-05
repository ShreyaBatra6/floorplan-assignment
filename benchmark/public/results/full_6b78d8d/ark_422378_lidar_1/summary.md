# 422378_42447307: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 15.95 [15.21, 16.70] m²; bounding box 5.20 x 4.90 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring
* runtime: 69.7 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 10.44 [10.34, 10.54] m² | 2.279 [2.266, 2.291] m | 8 | 4 |
| R2 Room 2 | 5.51 [5.04, 5.98] m² | 2.302 [2.289, 2.316] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 3.286 [3.269, 3.303] m | 100% |
| R1-W2 | 3.246 [3.229, 3.263] m | 100% |
| R1-W3 | 3.000 [2.983, 3.016] m | 100% |
| R1-W4 | 0.263 [0.245, 0.280] m | 100% |
| R1-W5 | 0.192 [0.176, 0.209] m | 100% |
| R1-W6 | 1.574 [1.555, 1.592] m | 100% |
| R1-W7 | 0.094 [0.078, 0.111] m | 100% |
| R1-W8 | 1.409 [1.392, 1.427] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | open_passage | R1-W1 | 2.576 [2.458, 2.694] m | 1.150 [1.061, 1.239] m | - |
| R1-O2 | window | R1-W3 | 1.740 [1.618, 1.862] m | 0.790 [0.747, 0.833] m | - |
| R1-O3 | open_passage | R1-W6 | 1.540 [1.422, 1.658] m | 0.650 [0.561, 0.739] m | - |
| R1-O4 | window | R1-W8 | 1.330 [1.212, 1.448] m | 2.220 [2.108, 2.332] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.797 [1.780, 1.814] m | 10% |
| R2-W2 | 3.066 [2.785, 3.347] m | 43% |
| R2-W3 | 1.797 [1.780, 1.814] m | 0% |
| R2-W4 | 3.066 [2.785, 3.347] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.495 [0.388, 0.602] m | 2.160 [2.110, 2.210] m | R1 |

## Adjacency

* R1 - R2: door R1-O4, R2-O1 (rays through a shared opening)

## Warnings

* 2 tall-furniture face(s) in front of walls ignored when outlining rooms
