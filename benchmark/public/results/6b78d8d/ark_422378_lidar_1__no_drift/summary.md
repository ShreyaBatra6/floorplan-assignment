# 422378_42447307: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 16.40 [15.60, 17.20] m²; bounding box 5.18 x 4.95 m
* drift: none (ARKit poses as-is)
* runtime: 10467.0 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 10.43 [10.33, 10.53] m² | 2.275 [2.262, 2.287] m | 6 | 4 |
| R2 Room 2 | 5.97 [5.46, 6.48] m² | 2.303 [2.290, 2.316] m | 4 | 1 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 3.231 [3.215, 3.248] m | 100% |
| R1-W2 | 3.245 [3.228, 3.262] m | 100% |
| R1-W3 | 3.023 [3.006, 3.039] m | 100% |
| R1-W4 | 0.257 [0.239, 0.274] m | 100% |
| R1-W5 | 0.209 [0.192, 0.225] m | 100% |
| R1-W6 | 2.988 [2.971, 3.005] m | 94% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | open_passage | R1-W1 | 2.580 [2.462, 2.698] m | 1.150 [1.066, 1.234] m | - |
| R1-O2 | window | R1-W3 | 1.745 [1.643, 1.847] m | 0.790 [0.701, 0.879] m | - |
| R1-O3 | open_passage | R1-W6 | 1.500 [1.382, 1.618] m | 0.650 [0.566, 0.734] m | - |
| R1-O4 | open_passage | R1-W6 | 1.320 [0.304, 2.336] m | 2.230 [2.146, 2.314] m | R2 |

## R2 Room 2

| wall | length | observed |
|---|---|---|
| R2-W1 | 1.748 [1.731, 1.766] m | 10% |
| R2-W2 | 3.413 [3.134, 3.693] m | 45% |
| R2-W3 | 1.748 [1.731, 1.766] m | 0% |
| R2-W4 | 3.413 [3.134, 3.693] m | 97% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | door | R2-W2 | 0.470 [0.374, 0.566] m | 2.150 [2.100, 2.200] m | R1 |

## Adjacency

* R1 - R2: open_passage R1-O4, R2-O1 (rays through a shared opening)
