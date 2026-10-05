# 422378_42447308: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 16.24 [15.97, 16.52] m²; bounding box 5.21 x 4.92 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring (self-checked by wall-map entropy)
* runtime: 53.1 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 10.55 [10.44, 10.65] m² | 2.277 [2.264, 2.289] m | 4 | 2 |
| R2 Room 2 | 5.70 [5.59, 5.80] m² | 2.276 [2.261, 2.291] m | 4 | 2 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 3.242 [3.224, 3.260] m | 100% |
| R1-W2 | 3.253 [3.237, 3.270] m | 100% |
| R1-W3 | 3.242 [3.224, 3.260] m | 100% |
| R1-W4 | 3.253 [3.237, 3.270] m | 86% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | window | R1-W3 | 1.805 [1.720, 1.890] m | 0.870 [0.781, 0.959] m | - |
| R1-O2 | open_passage | R1-W4 | 0.950 [0.622, 1.278] m | 2.280 [2.230, 2.330] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.801 [1.783, 1.819] m | 29% |
| R2-W2 | 3.163 [3.134, 3.191] m | 62% |
| R2-W3 | 1.801 [1.783, 1.819] m | 23% |
| R2-W4 | 3.163 [3.134, 3.191] m | 90% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | window | R2-W1 | 0.720 [0.602, 0.838] m | 0.676 [0.526, 0.825] m | - |
| R2-O2 | door | R2-W2 | 0.530 [0.462, 0.598] m | 1.890 [1.840, 1.940] m | R1 |

## Adjacency

* R1 - R2: open_passage R1-O2, R2-O2 (rays through a shared opening)
