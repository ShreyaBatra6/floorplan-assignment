# 471948_47204559: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 8.03 [7.02, 9.05] m²; bounding box 5.14 x 1.78 m
* drift: none (ARKit poses as-is)
* runtime: 15.0 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 4.97 [4.87, 5.07] m² | 2.347 [2.334, 2.361] m | 4 | 3 |
| R2 Room 2 | 3.06 [2.41, 3.72] m² | 2.336 [2.305, 2.368] m | 6 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 2.793 [2.766, 2.819] m | 85% |
| R1-W2 | 1.780 [1.763, 1.798] m | 100% |
| R1-W3 | 2.793 [2.766, 2.819] m | 100% |
| R1-W4 | 1.780 [1.763, 1.798] m | 99% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | window | R1-W2 | 0.620 [0.542, 0.698] m | 1.685 [1.578, 1.792] m | - |
| R1-O2 | door | R1-W3 | 1.040 [0.944, 1.136] m | 1.840 [1.756, 1.924] m | - |
| R1-O3 | door | R1-W4 | 0.815 [0.723, 0.907] m | 2.020 [1.913, 2.127] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 2.308 [2.288, 2.328] m | 64% |
| R2-W2 | 1.661 [1.381, 1.940] m | 80% |
| R2-W3 | 0.046 [0.028, 0.065] m | 0% |
| R2-W4 | 0.327 [0.000, 0.722] m | 100% |
| R2-W5 | 2.354 [2.334, 2.373] m | 0% |
| R2-W6 | 1.334 [1.054, 1.613] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O2 | door | R2-W2 | 0.850 [0.754, 0.946] m | 2.190 [2.106, 2.274] m | R1 |

## Adjacency

* R1 - R2: door R1-O3, R2-O2 (rays through a shared opening)

## Warnings

* 2 tall-furniture face(s) in front of walls ignored when outlining rooms
