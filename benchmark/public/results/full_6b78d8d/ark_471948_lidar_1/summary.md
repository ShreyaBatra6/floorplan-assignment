# 471948_47204559: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 8.16 [7.08, 9.23] m²; bounding box 5.21 x 1.78 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring
* runtime: 19.3 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 5.08 [4.98, 5.17] m² | 2.348 [2.334, 2.361] m | 4 | 2 |
| R2 Room 2 | 3.08 [2.38, 3.78] m² | 2.336 [2.305, 2.368] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 2.859 [2.841, 2.878] m | 85% |
| R1-W2 | 1.777 [1.760, 1.793] m | 100% |
| R1-W3 | 2.859 [2.841, 2.878] m | 100% |
| R1-W4 | 1.777 [1.760, 1.793] m | 98% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | door | R1-W3 | 1.010 [0.942, 1.078] m | 1.800 [1.750, 1.850] m | - |
| R1-O2 | door | R1-W4 | 0.825 [0.733, 0.917] m | 1.995 [1.936, 2.054] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 2.306 [2.285, 2.327] m | 63% |
| R2-W2 | 1.334 [1.055, 1.614] m | 100% |
| R2-W3 | 2.306 [2.285, 2.327] m | 0% |
| R2-W4 | 1.334 [1.055, 1.614] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O2 | door | R2-W2 | 0.905 [0.843, 0.967] m | 2.190 [2.106, 2.274] m | R1 |

## Adjacency

* R1 - R2: door R1-O2, R2-O2 (rays through a shared opening)

## Warnings

* 3 tall-furniture face(s) in front of walls ignored when outlining rooms
