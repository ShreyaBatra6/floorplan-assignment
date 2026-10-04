# Device matrix

## Which tier runs on which hardware

| Device | Photo tier | Video tier | LiDAR tier | Notes |
|---|---|---|---|---|
| iPhone 15, 15 Plus | yes | yes | no (no LiDAR) | 1× camera; HEIC stills with EXIF focal + compass; 1080p30 video |
| iPhone 15 Pro, 15 Pro Max | yes | yes | yes | LiDAR via Stray Scanner |
| iPhone 16, 16 Plus, 16e | yes | yes | no | |
| iPhone 16 Pro, 16 Pro Max | yes | yes | yes | |
| iPhone 17 family (non-Pro / Pro) | yes | yes | Pro models | |
| iPhone 12 Pro – 14 Pro (Max) | yes* | yes* | yes | *older than the brief's iPhone 15 floor; works, not benchmarked |
| iPad Pro (2020 or later) | yes | yes | yes | LiDAR; wider field of view |

Requirements per tier: **Photo**: any iPhone camera, Location Services on for Camera (compass) is
recommended. **Video**: any iPhone, 1080p30, 1× lens. **LiDAR**: a device with a LiDAR scanner and
Stray Scanner (iOS 14+). Processing: any laptop (Windows, macOS, Linux), CPU only, 8 GB RAM.

## What each tier honestly delivers

The accuracy columns are filled from the benchmark (`groundplan bench run`, laser/tape ground
truth, see `docs/BENCHMARK_PROTOCOL.md`); each cell names the run it came from. Until the benchmark
has been run, only the evidence that exists is shown, and it is labelled for what it is.

| Tier | Wall length | Ceiling height | Opening width | Floor area | Evidence |
|---|---|---|---|---|---|
| LiDAR | median 1 mm, max < 10 mm (synthetic) | < 1 mm (synthetic) | 4/8 within 2 cm (synthetic) | < 0.1 % (synthetic) | simulated flat, exact ground truth; **real-room benchmark: pending** |
| Video | pending | pending | pending | pending | per-frame depth-model scale log-sd 0.15-0.20 before alignment (assessor frames vs LiDAR) |
| Photo | pending | pending | pending | pending | scale from fused cues; intervals carry the fused scale sigma |

Gates the brief sets per tier: LiDAR walls max(2 cm, 1 %) (Round 1, inferred), ceiling 1.5 cm,
openings 2 cm on 85 %; video walls ±3 %; photo walls ±8 % and stitched footprint ±8 %, all with
calibrated intervals.
