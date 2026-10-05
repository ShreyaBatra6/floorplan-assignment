# photos: photo tier

* intervals: 90% two-sided; calibration `prior-0` (prior, not yet fitted)
* scale source: per-room fusion of model, ceiling, door and camera-height cues (log-sigma 0.163)
* footprint: 15.42 [0.00, 42.52] m²; bounding box 4.09 x 4.35 m
* drift: per-room reconstructions (no accumulated drift); rooms joined by door pairing with compass and no-overlap constraints
* runtime: 17.3 s on Windows AMD64 · Intel64 Family 6 Model 140 Stepping 1, GenuineIntel · 8 threads

| room | floor area | ceiling height | walls | openings |
|---|---|---|---|---|
| R1 Room | 15.42 [5.31, 35.16] m² | 2.608 [1.736, 3.902] m | 6 | 1 |

## R1 Room

| wall | length | observed |
|---|---|---|
| R1-W1 | 4.088 [2.415, 6.333] m | 0% |
| R1-W2 | 4.350 [2.373, 6.889] m | 40% |
| R1-W3 | 0.066 [0.000, 1.054] m | 0% |
| R1-W4 | 0.587 [0.000, 1.617] m | 0% |
| R1-W5 | 4.022 [2.122, 6.426] m | 100% |
| R1-W6 | 3.763 [2.175, 5.865] m | 0% |

| opening | kind | wall | width | height | to |
|---|---|---|---|---|---|
| R1-O1 | window | R1-W2 | 0.780 [0.431, 1.231] m | 0.579 [0.271, 0.953] m | - |

## Warnings

* Room: structural registration skipped (needs a compass heading on every photo); feature-based registration used
* Room: 4 of 6 photos could not be registered (too little overlap or texture) and were not used
* Room: photo registration could not be verified (1 photo(s) disagree with the room outline); its dimensions carry a wider interval and should be re-captured
* Room: 4 of 6 walls appear in no photo; their positions are inferred and carry wide intervals (photograph every wall, protocol step C2)
