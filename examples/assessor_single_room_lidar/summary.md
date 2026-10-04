# c00a170fe1: lidar tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: lidar (log-sigma 0.000)
* footprint: 21.56 [19.20, 23.92] m²; bounding box 7.87 x 6.04 m
* drift: 4-DoF submap pose graph: odometry + ICP loop closures + Manhattan plane anchoring
* runtime: 164.5 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room 1 | 7.24 [6.00, 8.48] m² | 2.800 [2.200, 3.400] m | 8 | 1 |
| R2 Bathroom | 5.18 [5.09, 5.27] m² | 2.800 [2.200, 3.400] m | 4 | 2 |
| R3 Room 3 | 6.28 [5.46, 7.10] m² | 2.800 [2.200, 3.400] m | 4 | 0 |
| R4 Bathroom | 2.86 [2.37, 3.34] m² | 2.800 [2.200, 3.400] m | 6 | 0 |

## R1 Room 1

| wall | length | observed |
|---|---|---|
| R1-W1 | 1.405 [1.126, 1.685] m | 0% |
| R1-W2 | 0.416 [0.137, 0.695] m | 0% |
| R1-W3 | 2.999 [2.604, 3.394] m | 80% |
| R1-W4 | 2.144 [1.865, 2.423] m | 0% |
| R1-W5 | 3.005 [2.725, 3.284] m | 0% |
| R1-W6 | 1.991 [1.596, 2.386] m | 57% |
| R1-W7 | 1.400 [1.379, 1.420] m | 0% |
| R1-W8 | 0.569 [0.174, 0.964] m | 100% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | window | R1-W4 | 0.650 [0.554, 0.746] m | 0.720 [0.587, 0.853] m | R4 |

## R2 Bathroom

| wall | length | observed |
|---|---|---|
| R2-W1 | 2.556 [2.539, 2.572] m | 100% |
| R2-W2 | 2.027 [2.009, 2.044] m | 100% |
| R2-W3 | 2.556 [2.539, 2.572] m | 18% |
| R2-W4 | 2.027 [2.009, 2.044] m | 65% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R2-O1 | window | R2-W1 | 0.520 [0.198, 0.842] m | 1.500 [1.351, 1.649] m | - |
| R2-O2 | window | R2-W3 | 0.830 [0.762, 0.898] m | 0.766 [0.617, 0.916] m | R3 |

## R3 Room 3

| wall | length | observed |
|---|---|---|
| R3-W1 | 2.101 [1.821, 2.380] m | 100% |
| R3-W2 | 2.991 [2.972, 3.011] m | 100% |
| R3-W3 | 2.101 [1.821, 2.380] m | 90% |
| R3-W4 | 2.991 [2.972, 3.011] m | 0% |

## R4 Bathroom

| wall | length | observed |
|---|---|---|
| R4-W1 | 1.024 [1.007, 1.042] m | 100% |
| R4-W2 | 1.681 [1.664, 1.697] m | 62% |
| R4-W3 | 0.345 [0.065, 0.624] m | 100% |
| R4-W4 | 0.829 [0.549, 1.108] m | 0% |
| R4-W5 | 1.369 [1.089, 1.648] m | 0% |
| R4-W6 | 2.509 [2.230, 2.789] m | 42% |

## Warnings

* ceiling not observed in this capture; ceiling heights are bounded by priors
* floor surfaces are not assessed visually (rugs, mats and furniture make floor staining unreliable); floor-level water is flagged through wall-base staining (rule CDR-02)
* ceilings not observed: no ceiling damage could be assessed
* R1: ceiling not observed; reported as a bounded prior interval
* R2: ceiling not observed; reported as a bounded prior interval
* R3: ceiling not observed; reported as a bounded prior interval
* R4: ceiling not observed; reported as a bounded prior interval
