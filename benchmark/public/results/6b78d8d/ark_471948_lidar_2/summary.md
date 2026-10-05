# 471948_47204563: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 8.44 [7.43, 9.45] m²; bounding box 5.09 x 1.77 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring
* runtime: 15.9 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 4.86 [4.77, 4.96] m² | 2.346 [2.332, 2.361] m | 4 | 2 |
| R2 Room 2 | 3.58 [2.92, 4.24] m² | 2.846 [2.292, 3.400] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 2.752 [2.734, 2.769] m | 100% |
| R1-W2 | 1.767 [1.750, 1.784] m | 100% |
| R1-W3 | 2.752 [2.734, 2.769] m | 100% |
| R1-W4 | 1.767 [1.750, 1.784] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | open_passage | R1-W3 | 1.430 [1.338, 1.522] m | 1.840 [1.756, 1.924] m | - |
| R1-O2 | door | R1-W4 | 0.865 [0.773, 0.957] m | 2.015 [1.969, 2.061] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 2.161 [2.142, 2.180] m | 42% |
| R2-W2 | 1.657 [1.377, 1.937] m | 100% |
| R2-W3 | 2.161 [2.142, 2.180] m | 0% |
| R2-W4 | 1.657 [1.377, 1.937] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.760 [0.664, 0.856] m | 2.070 [1.986, 2.154] m | R1 |

## Adjacency

* R1 - R2: door R1-O2, R2-O1 (rays through a shared opening)

## Warnings

* 1 tall-furniture face(s) in front of walls ignored when outlining rooms
* R2: ceiling not observed; reported as a bounded prior interval
