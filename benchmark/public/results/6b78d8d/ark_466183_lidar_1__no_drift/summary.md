# 466183_45260920: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 22.91 [22.19, 23.63] m²; bounding box 5.37 x 5.23 m
* drift: none (ARKit poses as-is)
* runtime: 48.6 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 20.65 [20.53, 20.77] m² | 2.946 [2.932, 2.959] m | 8 | 6 |
| R2 Room 2 | 2.26 [1.81, 2.71] m² | 2.800 [2.200, 3.400] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 0.443 [0.425, 0.462] m | 100% |
| R1-W2 | 0.294 [0.274, 0.313] m | 100% |
| R1-W3 | 3.259 [3.237, 3.281] m | 79% |
| R1-W4 | 0.262 [0.242, 0.283] m | 100% |
| R1-W5 | 0.308 [0.287, 0.328] m | 100% |
| R1-W6 | 4.939 [4.921, 4.958] m | 84% |
| R1-W7 | 4.010 [3.993, 4.027] m | 100% |
| R1-W8 | 4.908 [4.891, 4.925] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | open_passage | R1-W3 | 2.670 [2.330, 3.010] m | 0.930 [0.846, 1.014] m | - |
| R1-O2 | door | R1-W6 | 1.050 [0.932, 1.168] m | 2.680 [2.606, 2.754] m | - |
| R1-O3 | window | R1-W6 | 1.005 [0.940, 1.070] m | 0.675 [0.619, 0.731] m | - |
| R1-O4 | window | R1-W6 | 1.330 [1.212, 1.448] m | 2.440 [2.328, 2.552] m | - |
| R1-O5 | door | R1-W8 | 0.910 [0.814, 1.006] m | 2.020 [1.936, 2.104] m | R2 |
| R1-O6 | open_passage | R1-W8 | 1.510 [1.356, 1.664] m | 1.630 [1.546, 1.714] m | - |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.134 [1.112, 1.156] m | 0% |
| R2-W2 | 1.992 [1.598, 2.387] m | 100% |
| R2-W3 | 1.134 [1.112, 1.156] m | 0% |
| R2-W4 | 1.992 [1.598, 2.387] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.810 [0.667, 0.953] m | 2.210 [2.126, 2.294] m | R1 |

## Adjacency

* R1 - R2: door R1-O5, R2-O1 (rays through a shared opening)

## Warnings

* 9 tall-furniture face(s) in front of walls ignored when outlining rooms
* R2: ceiling not observed; reported as a bounded prior interval
