# 466183_45260925: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 22.80 [22.08, 23.52] m²; bounding box 5.39 x 5.22 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring
* runtime: 51.6 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 20.82 [20.69, 20.95] m² | 2.946 [2.933, 2.960] m | 8 | 4 |
| R2 Room 2 | 1.98 [1.53, 2.43] m² | 2.800 [2.200, 3.400] m | 4 | 2 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 0.453 [0.435, 0.471] m | 88% |
| R1-W2 | 0.309 [0.287, 0.330] m | 100% |
| R1-W3 | 3.256 [3.235, 3.278] m | 75% |
| R1-W4 | 0.272 [0.250, 0.294] m | 100% |
| R1-W5 | 0.325 [0.304, 0.346] m | 100% |
| R1-W6 | 4.945 [4.927, 4.964] m | 82% |
| R1-W7 | 4.035 [4.017, 4.052] m | 100% |
| R1-W8 | 4.908 [4.890, 4.926] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | door | R1-W6 | 1.110 [0.992, 1.228] m | 2.690 [2.626, 2.754] m | - |
| R1-O2 | window | R1-W6 | 0.985 [0.893, 1.077] m | 0.790 [0.701, 0.879] m | - |
| R1-O3 | window | R1-W6 | 1.370 [1.252, 1.488] m | 2.495 [2.408, 2.582] m | - |
| R1-O4 | door | R1-W8 | 1.215 [1.123, 1.307] m | 1.985 [1.939, 2.031] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.140 [1.120, 1.161] m | 0% |
| R2-W2 | 1.735 [1.341, 2.130] m | 90% |
| R2-W3 | 1.140 [1.120, 1.161] m | 0% |
| R2-W4 | 1.735 [1.341, 2.130] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.780 [0.712, 0.848] m | 2.240 [2.156, 2.324] m | R1 |
| R2-O2 | open_passage | R2-W4 | 1.450 [1.332, 1.568] m | 0.900 [0.836, 0.964] m | - |

## Adjacency

* R1 - R2: door R1-O4, R2-O1 (rays through a shared opening)

## Warnings

* 13 tall-furniture face(s) in front of walls ignored when outlining rooms
* R2: ceiling not observed; reported as a bounded prior interval
