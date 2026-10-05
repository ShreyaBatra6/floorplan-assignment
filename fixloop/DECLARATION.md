# Fix declaration

*Filled in after the first full benchmark run, before the fix is written. The text of sections
1-3 is committed (and tagged `fixloop-declared`) before any change to the code, so the prediction
cannot be edited after the fact.*

## 1. Worst-performing gate

- Gate: **opening widths** (tier: lidar)
- Threshold (from the brief): <=2 cm on >=85%, misses+phantoms count
- Failing number: **0% (0/10)** on benchmark run `benchmark/public/results/6b78d8d` (commit `6b78d8d`, tag `fixloop-before`)
- Detail: missed 2, phantom 4

All failing gates of that run, worst first (share of the requirement met):

| gate | tier | value | met |
|---|---|---|---|
| opening widths | lidar | 0% (0/10) | 0% |
| ceiling height | lidar | 3/9 rooms; max 3.9 cm | 33% |
| repeatability (ark_471948_lidar_1 vs ark_471948_lidar_2) | lidar | 1/2 within tolerance | 50% |
| repeatability (ark_471948_lidar_1 vs ark_471948_lidar_3) | lidar | 1/2 within tolerance | 50% |
| wall lengths | lidar | 10/15 within; coverage 47% | 55% |
| repeatability (ark_466183_lidar_1 vs ark_466183_lidar_3) | lidar | 2/3 within tolerance | 67% |
| repeatability (ark_422378_lidar_1 vs ark_422378_lidar_2) | lidar | 2/3 within tolerance | 67% |
| repeatability (ark_422378_lidar_1 vs ark_422378_lidar_3) | lidar | 2/3 within tolerance | 67% |

## 2. Root-cause hypothesis and evidence

*Benchmark: the public stand-in (ARKitScenes homes with laser-scan truth, `docs/PUBLIC_BENCHMARK.md`),
LiDAR tier; our own captures do not exist yet. Openings are scored only in rooms whose openings the
laser established (home 422378: door D1 0.79 m, window N2 1.77 m; 3 recordings).*

- Hypothesis: two independent faults. (1) Openings are accepted without a physical shape check: a
  floor-level region of any height counts as a passage (only its width is tested), and a door whose
  threshold left a 2 cm strip unobserved counts as a window. (2) Jambs are located on a binned,
  thresholded profile of the sparse 2 cm voxel cloud, so widths come out several centimetres off
  even when the opening is found.
- Evidence (before run, per recording):
  1. Recording 1: two "open passages" 1.54 m and 2.58 m wide but only **0.65 m and 1.15 m tall**
     (nobody walks through them: regions under and behind furniture, where dark surfaces return no
     LiDAR depth and are read as see-through), and D1 reported as a 1.33 m "window" with a
     **0.02 m sill** and 2.22 m height. Together: 3 phantoms and 1 miss.
  2. Widths of the openings that were found: N2 -3.0 cm (rec. 1), +3.5 cm (rec. 2); D1 +16 cm
     (rec. 2), +12 cm (rec. 3). None within 2 cm.
  3. Reproduced in isolation: a synthetic wall with a 0.90 m door and a 1.8 m wide, 0.7 m tall
     floor-level gap, run through the before code's `detect_openings`, returns the door (0.89 m)
     **and** an "open_passage" 1.79 m wide and 0.68 m tall. (This scenario ships with the fix as
     `tests/test_opening_shapes.py`; tests may not change before the declaration.)
  4. The same binned-profile jamb rule measured synthetic doors with errors of -8 to +4 cm against
     exact truth (44 % within 2 cm, `tests` flats); re-locating jambs at the half-maximum of the
     wall-point density brought that to 69 % in a rehearsal on the synthetic benchmark.
- Alternatives considered and why the evidence rules them out: depth resolution alone (about 1 cm
  per pixel at the distances involved) cannot produce 12-16 cm width errors or 0.65 m tall
  "passages"; the matcher (rectangle ties now broken by openings) assigns D1 and N2 to the right
  walls in all three recordings.

## 3. The fix and the predicted number

- Fix: (1) shape rules in `geometry/openings.py`: a floor-level opening must reach at least 1.8 m
  to be a door or passage (lower floor-level regions are not openings); an opening whose bottom is
  within 15 cm of the floor is floor-level (a door or passage, not a window). (2) Each jamb is
  re-located at the half-maximum of the wall-point density across it (corrected by a quarter of the
  voxel size), kept only when consistent with the ray evidence.
- Predicted value of the gate after the fix: phantoms 4 -> 2 (recording 1's two low regions go;
  the misread door becomes a door but its width stays far off); openings within 2 cm 0 -> 1 or 2
  of 8 scored; the gate moves from **0 % to about 12-25 %** and **still fails** (85 % needed). The
  corner door D1 (its corner-side jamb is the adjoining wall, which the jamb search does not use)
  is expected to stay about 10 cm wide.
- What else could move, and the bound I expect on it: walls, ceilings and repeatability do not use
  this code (0 change expected); opening heights move with the reclassification only.

## 4. Outcome (after the fix ships; regenerate with `python fixloop/run_fixloop.py`)

- After-run value: **0 % (0/8)**, FAIL (commit `ea6c40c`, tag `fixloop-after`; `fixloop/RESULT.md`,
  `fixloop/fix.diff`). Walls, ceilings and repeatability unchanged, as predicted.
- Prediction vs outcome, part by part:
  1. Shape rules - **as predicted.** Phantoms 4 -> 2: recording 1's two floor-level "passages"
     (0.65 m and 1.15 m tall) are gone, and the door read as a 2.2 m "window" with a 2 cm sill is
     now a floor-level opening; the two phantoms left are the ones predicted to stay (that door,
     whose width is still far off at 1.33 m, and recording 3's 1.99 m tall passage).
  2. Jamb re-location - **wrong.** Predicted 1-2 of 8 openings within 2 cm; got 0, and the widths
     moved the wrong way: window N2 -3.0 -> +3.2 cm (rec. 1) and +3.5 -> +6.0 cm (rec. 2), door D1
     +16 -> +26 cm (rec. 2), unchanged in rec. 3.
- Why it fell short: the half-maximum rule assumes the wall plane ends cleanly at the jamb, which is
  true of the synthetic flats it was tuned on (44 -> 69 % there). Real door frames and window reveals
  carry casings and trim standing proud of the wall: wall-plane points thin out over several
  centimetres before the opening, so the half-maximum lands outside the true edge and every width
  grows. The evidence for this part came from synthetic data only, and the prediction inherited that
  blind spot. The gate's numerator stayed at 0 because no width moved inside 2 cm; its denominator
  fell from 10 to 8 because the shape rules removed two phantoms.
- What follows (on `main` after the loop): the jamb re-location is off by default (it widens real
  openings). The threshold rule (a sill under 15 cm means a door) stays on. Rejecting low floor-level
  gaps is opt-in: with it on, those regions behind furniture are searched for damage, and the
  assessor sample's staged-damage test showed a false positive there (passes before, fails after),
  which the stand-in benchmark could not show because it has no damage truth. Next: hand those
  regions to damage detection as unobserved areas, then turn the rule on; measure jambs at the
  change of depth where the reveal begins, validated on real recordings before predicting.
