# 45260920: video tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: fused: model+ceiling+door+camera (log-sigma 0.034)
* footprint: 18.93 [15.59, 22.75] m²; bounding box 5.41 x 4.47 m
* drift: structure from motion with global bundle adjustment over the whole walk
* runtime: 2210.1 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 18.93 [15.85, 22.52] m² | 2.968 [2.724, 3.234] m | 8 | 6 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 1.545 [1.395, 1.703] m | 0% |
| R1-W2 | 3.290 [2.568, 4.020] m | 100% |
| R1-W3 | 0.211 [0.122, 0.300] m | 0% |
| R1-W4 | 0.102 [0.000, 0.579] m | 100% |
| R1-W5 | 3.659 [3.352, 3.990] m | 100% |
| R1-W6 | 4.468 [4.095, 4.871] m | 95% |
| R1-W7 | 5.414 [4.970, 5.896] m | 83% |
| R1-W8 | 1.077 [0.588, 1.567] m | 95% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | door | R1-W1 | 0.525 [0.394, 0.656] m | 2.230 [2.019, 2.455] m | - |
| R1-O2 | door | R1-W2 | 0.500 [0.354, 0.647] m | 1.940 [1.749, 2.143] m | - |
| R1-O3 | open_passage | R1-W5 | 3.145 [2.848, 3.461] m | 2.690 [2.447, 2.951] m | - |
| R1-O4 | open_passage | R1-W6 | 1.818 [1.603, 2.041] m | 2.910 [2.651, 3.188] m | - |
| R1-O5 | open_passage | R1-W7 | 2.710 [2.441, 2.995] m | 1.420 [1.261, 1.586] m | - |
| R1-O6 | open_passage | R1-W7 | 1.570 [1.364, 1.783] m | 2.750 [2.502, 3.015] m | - |

## Warnings

* 185 extra frame(s) kept where the view changed fast (turns)
* structure from motion: 280/280 frames in one model
* 26 tall-furniture face(s) in front of walls ignored when outlining rooms
