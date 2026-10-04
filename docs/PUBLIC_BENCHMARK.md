# Public stand-in benchmark (ARKitScenes)

**What this is.** The case study's benchmark needs our own captures of real rooms with an iPhone 15
and a LiDAR iPhone/iPad, measured on site with a laser. Until those exist, the benchmark harness,
the interval calibration and the fix loop run on a public dataset with survey-grade ground truth:
**ARKitScenes** (Apple; Baruch et al., *NeurIPS 2021 Datasets and Benchmarks*). Every number it
produces is labelled *public stand-in*; none of it is presented as the case study's own benchmark.

| | Brief's benchmark (pending) | This stand-in |
|---|---|---|
| Device | iPhone 15 (photo, video), LiDAR iPhone/iPad (Stray Scanner) | 2020 iPad Pro: LiDAR depth + ARKit poses, 1920x1440 wide-camera video |
| Capture | our protocol (ceiling/floor sweeps, doorways from both sides, photos from wall middles) | Apple's capture procedure for object detection: rooms walked once, ceilings rarely swept |
| Rooms | multi-room flat + connector, staged damage, repeats | single rooms in real homes, **3 independent recordings per room** (repeatability) |
| Truth | laser/tape on site | Faro Focus S70 laser scans of each home (mm-level), measured by our script and reviewed |
| Tiers | three captures per room | LiDAR: each recording converted to the Stray Scanner layout; video: the first recording's clip as a plain video; photo: 6 stills of the second recording |

## Pipeline from raw data to scores

```bash
export GROUNDPLAN_DATA=~/data            # the data lives outside the repository (~8 GB per home)
python scripts/arkitscenes.py fetch --visits 466183 422378 471948 437450
python scripts/arkitscenes.py convert    # recordings -> Stray Scanner folders (LiDAR tier input)
python scripts/arkitscenes.py truth      # laser scans -> benchmark/public/ground_truth/*.yaml + review renders
python scripts/arkitscenes.py tiers      # video clip + photo folder per home
python scripts/arkitscenes.py manifest   # benchmark/public/manifest.yaml
groundplan bench run --manifest benchmark/public/manifest.yaml --out benchmark/public/results/<sha>
```

**Conversion** (`src/groundplan/bench/arkitscenes.py`). `lowres_wide.traj` holds world-to-camera
poses at 10 Hz (axis-angle, metres) in a z-up world; inverted and turned y-up they are exactly the
Stray Scanner convention (OpenCV camera to ARKit world). Depth (uint16 mm, 256x192, 60 Hz) and
confidence frames are kept where a pose lies within 5 ms, as Apple's own loader does (about 10 Hz).
Colour comes from the 1920x1440 `.mov`, whose frame *j - 1* is colour frame *j* of `lowres_wide`
(checked by image differences: 0.2-0.7 grey levels on the matching frame, 3-16 on its neighbours).
Per-frame intrinsics are scaled from 256x192 to 1920x1440 with the half-pixel shift. Checked end to
end on visit 421063: the LiDAR tier's ceiling 2.299 m against 2.309 m on the laser scan.

**Video tier input.** The recording's `.mov` as it is: no QuickTime focal-length tag (it came from
ARKit, not the Camera app), so the reader falls back to its default focal length and structure from
motion refines it, which is the harder case.

**Photo tier input.** Six stills from the second recording, chosen like protocol photos (camera
level, views spread round the room); the iPad's real focal length is written as the
`FocalLengthIn35mmFilm` EXIF tag, rounded to a whole millimetre as phones write it. There is no
compass heading (the recordings have none), so the photo tier's structural registration cannot run
and feature registration is used: a hard case, reported as such.

## Ground truth from the laser scans

The scans of one home are already registered to each other in the raw files (median gap between
two scans 7.6 mm on visit 421063). `scripts/arkitscenes.py truth` fuses them at 2 cm and measures,
for every laser scanner position, the room it stood in. The procedure shares no code with the
pipeline it judges:

* **Floor and ceiling:** the lowest and highest dense horizontal layers over the room's footprint.
* **Walls:** the Manhattan directions come from the wall normals. In each of the four directions the
  wall is the best-supported vertical plane, within 50 cm of the nearest one, that reaches to within
  25 cm of the ceiling (a cornice may hide the last centimetres; furniture stops lower), runs for at
  least 80 cm and lies across the scanner's line of sight. A room with a second strong plane more
  than 15 cm behind the wall (alcoves, a recess) is flagged as not a box.
* **Openings:** in each wall plane (1 cm raster), regions where the laser saw beyond the plane,
  bridged across window bars; jambs are where the wall plane resumes, as a median over the middle
  half of the opening's height. Kept only if shaped like a door (floor to 1.8-2.6 m, 0.55-1.25 m
  wide), a passage (wider, at least 1.9 m tall) or a window (sill 0.3-1.6 m); regions running into a
  corner are recesses, not openings.
* **Review:** every room and wall is drawn (`benchmark/public/review/*.png`); what a person rejects
  on those renders is recorded with its reason in `benchmark/public/review/decisions.yaml`, and the
  ground-truth file says so. A wall a box cannot define (for example beside a chimney breast) is
  left out rather than guessed.

Unit tests (`tests/test_arkitscenes.py`) check the pose convention and the whole procedure on a
synthetic scan with a rotated 4.0 x 3.0 m room, a 2.6 m ceiling, a 0.90 m door with a corridor
behind it and a 2 m wardrobe in front of a wall: room within 5 mm, ceiling within 5 mm, door within
1.5 cm.

## Limits (read before quoting any number)

* An iPad Pro 2020 is not an iPhone 15, and these recordings did not follow our capture protocol
  (ceilings and doorways were rarely looked at), so the ceiling and opening gates are often *not
  measured* or harder than they would be.
* The rooms are single rooms; the multi-room, stitching and staged-damage parts of the brief's
  benchmark are not covered here.
* The truth is mm-level laser data, but the *measurement* of a room from it is our script's, with
  the limits above; the review renders are the check.
* ARKitScenes is licensed for non-commercial use under Apple's ARKitScenes license. The data is
  downloaded from Apple by the script and is not redistributed with this repository; only the
  derived measurements and review renders are committed.
