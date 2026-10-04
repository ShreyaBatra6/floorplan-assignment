# groundplan: technical report

*Applied AI Engineer case study, Round 2. Six-page limit. Numbers marked **[bench]** are filled
from `benchmark/results/<sha>/benchmark_report.md` (laser/tape ground truth); numbers marked
*(synthetic)* come from ray-cast flats with exact ground truth; *(assessor data)* from the Round 1
sample captures, which have no ground truth.*

## 1. Approach in one paragraph

Every tier ends in the same place: a gravity-aligned, metric point cloud with the camera rays that
observed it. One geometric core turns that into rooms, walls, openings and adjacency; one damage
module works on per-surface orthomosaics; one assembly step attaches an error budget to every
number. The tiers differ only in how they produce the cloud and how well they know metric scale,
and that difference is carried explicitly (an additive sigma per quantity plus a log-scale sigma
per capture) all the way into the intervals. Capture is Route 2: Stray Scanner for LiDAR and the
built-in Camera app for photos and video, with a one-page protocol written so that its
instructions make the geometry observable (open doors, sweep to floor and ceiling lines, photos
from the middle of each wall).

## 2. Architecture

```
 photo folders --> per-photo metric depth -> gravity/Manhattan -> wall-based registration --+
 video clip -----> SfM (pycolmap) -> depth aligned per frame -> scale-cue fusion ------------+--> fused cloud + rays
 Stray export ---> ARKit depth + poses -> 4-DoF pose graph (drift) --------------------------+          |
                                                                                                     v
   geometric core: floor/ceiling levels | wall faces by vertical span & facing | furniture suppression
                   ray-carved interior -> watershed rooms (merge boundaries wider than a door)
                   rectilinear outlines from the wall-face arrangement, offsets refit to points
                   openings = holes in wall faces confirmed by rays through them (mirror test)
                                                                                                     |
   damage: orthomosaic per surface -> candidates (colour, ridges) -> geometric filters -> CLIP check
   rules (YAML, rule id + condition + facts) -> scope (surface-keyed, quantities with intervals)
   assembly: Estimate(value, sigma_abs, sigma_log) -> calibrated Measurement -> plan.json (schema)
```

The contract (`src/groundplan/contract.py`, published as `schema/groundplan.schema.json`) is the
same at every tier; each run validates its own output, including cross-references (every scope
item keyed to an existing surface, every flag citing existing regions).

## 3. Tiers and device matrix

| Tier | Device | Geometry | Metric scale | Stitching |
|---|---|---|---|---|
| LiDAR | Pro iPhone / iPad Pro | ARKit depth 256x192, 2 cm voxels | sensor | one walk, drift-corrected |
| Video | any iPhone 15+ | SfM poses + depth model aligned per frame | fused cues | one reconstruction |
| Photo | any iPhone 15+ | per-photo depth model, registered per room | fused cues per room | door pairing + compass |

Full matrix with per-device support and the accuracy each tier delivers: `docs/DEVICE_MATRIX.md`.

**Pose convention (assessor data).** Stray Scanner poses map OpenCV-convention camera points into
the ARKit world: back-projected depth from frames 30 apart agrees to 3-10 mm under that convention
and to 20-40 cm under the OpenGL one. World is gravity-aligned with y up (floor peak at a constant
height 1.5 m below the camera track).

## 4. LiDAR core decisions, and why

* **Wall evidence weighted by vertical span, per facing.** A wall is a face that spans floor to
  ceiling; a sofa back is not. Splitting by the direction a face looks into keeps the two sides of a
  12 cm partition apart, so room outlines use the face on their own side.
* **Furniture suppression.** A wardrobe front spans 2 m and looks like a wall in plan. It is
  rejected when rays pass over its top and land behind it, or when a higher face with the same
  orientation is seen behind it. *(synthetic)* living room 15.66 m2 vs truth 15.66 m2; without the
  rule, a 0.68 m2 notch and two wall lengths 0.2 m short.
* **Outlines from the wall-face arrangement.** The room is the union of arrangement cells mostly
  covered by its free space, then each edge is refitted to its face's points (robust location,
  effective sample size = number of frames). The same faces give the same outline capture after
  capture, which is what repeatability needs. *(synthetic)* walls median 1 mm, max < 10 mm.
* **Openings are holes in faces.** Through-rays (and no-return rays for glass) seed a candidate;
  jambs, head and sill are located between the last certain-open and first certain-solid evidence,
  and the width sigma reflects any unobserved band between them. *(synthetic)* 8/8 detected,
  0 phantoms, 4/8 within 2 cm: depth resolution (one 256x192 pixel is about 1 cm at 1.3 m)
  bounds this method; see section 9.

## 5. Drift accountability

Method: keyframes cut into submaps (~1 m / 40 deg); per-submap 4-DoF corrections (gravity is
observed, roll and pitch do not drift); odometry edges with a drift budget (1 %/m + 2 mm,
0.25 deg/m); loop closures from 4-DoF point-to-plane ICP between revisited submaps, **weighted by
the ICP information matrix** so a closure only constrains the directions its geometry constrains;
a plausibility gate against the drift expected over the walked distance; Manhattan plane anchoring
with the walk's heading as a free unknown; iteratively reweighted (Cauchy) closures starting from
the odometry-only solution. Two failure modes found and fixed during development are documented in
the git history: ICP sliding along corridors produced confident wrong closures (fixed by
information weighting), and the anchoring prior fought the gauge (fixed by the free heading).

Ablation *(synthetic, 1 cm/sqrt(m) and 0.4 deg/sqrt(m) incremental drift)*:

| | walls within max(2 cm, 1 %) | median wall error | interval coverage | footprint error |
|---|---|---|---|---|
| correction on | 16/16 | 0.6 cm | 100 % | -0.2 % |
| correction off (poses as-is) | 12/16 | 1.2 cm | 62 % | +0.1 % |

**Self-check.** A correction is kept only if it makes the map measurably more consistent: the
entropy of wall-point coordinates in the Manhattan frame (1 cm histograms per facing) must drop.
Without ground truth this is the standard consistency measure (doubled walls widen the
histograms). *(synthetic)* entropy 8.37 -> 7.51, correction kept. *(assessor data, multi-room walk)*
the computed correction would raise entropy 10.68 -> 11.30, so ARKit's poses are kept, consistent
with the low ARKit drift other analyses of this capture report; an earlier version without the
self-check applied a 0.55 m "correction" there, which this check now prevents. On the single-room
capture: 7.91 -> 7.67, kept (heading anchoring only). Real multi-room capture with laser truth, on
vs off: **[bench]** (`groundplan bench run` runs the ablation for every capture under
`drift_ablation`).

## 6. Metric scale without depth (photo and video)

A monocular metric model gives good shape but unreliable scale. Measured *(assessor data)* against
LiDAR depth on 24 frames per capture, Depth Anything V2 Metric-Indoor-Small: median scale ratio
(LiDAR/model) 0.684 on both captures, per-frame log-sd 0.15-0.20, 5-6 % relative error once
scaled (the Base model: 0.710, log-sd 0.19, no better). So scale is fused from independent cues in
log space with 3-sigma rejection: the calibrated model factor (sigma 0.12), the room's ceiling
height against a residential prior (0.08), door head heights (0.035 per door), camera height
(0.10), and an A4/Letter sheet on the floor when one is visible (0.015 if seen in two views,
0.025 in one). The sheet is found on each view's floor resampled top-down through the floor-plane
homography, by its side ratio (scale-free), measured with sub-pixel edges, and rejected if its two
sides disagree or the depth map puts it above the floor (a table top) *(synthetic renders: size
within 0.2 %)*. Video aligns every frame's depth map to the SfM reconstruction, which removes the
per-frame scale noise and leaves one global factor to fuse.
Video SfM breaks into pieces at fast turns past plain walls, so three measures keep a walk whole:
frames are sampled more densely where optical flow shows the view moving fast; frames that look
alike but are far apart in time (CLIP similarity in the clip's top 3 %) are matched besides the
sequential pairs; and partial models, normalised to depth-model units, are relocalised against the
largest by depth-aided PnP on their most similar frame pairs, accepted only when two pairs agree
(4 degrees, 0.2 m) and the poses are physically sane (points in front, camera near its partner).
On the assessor walkthrough (close range, fast turns, plain walls, 15 fps) the frames at each break
share fewer than 20 geometric inliers even with maximally sensitive features: no join can be
verified, the largest piece is kept (32 of 178 frames; 14 before) and the rest is reported as
dropped. A degenerate PnP pose (camera "at infinity" with 63 inliers) was caught there by the
sanity checks. The protocol (2 s per quarter turn, doorways from both sides) is the real remedy.
The photo tier registers a room's photos without texture: in a rectangular room each facing has
one wall, so wall observations are linear in camera positions, relative scales and wall positions;
the shared floor-to-ceiling height makes relative scales observable, and a rank check rejects
under-determined solves (unit test: exact recovery). Per facing, the wall is the layer with the
largest support x vertical extent (furniture fronts are low; the next room's wall, seen through a
doorway, is narrow). A room is measured only if its photos agree with it: each camera inside the
outline and each photo's floor inside it; a photo that disagrees is dropped and the room rebuilt,
otherwise the room's intervals widen and the warning names the photos.

## 7. Error budget

| Term (1 sigma) | LiDAR | Video | Photo | Source |
|---|---|---|---|---|
| face bias (depth/scale residual) | 3 mm | 10 mm | 20 mm | sensor literature; refit by calibration |
| face drift residual | 2 mm | 10 mm | 15 mm | ablation residuals |
| wall definition (skirting, waviness) | 3 mm | 5 mm | 8 mm | measurement practice |
| jamb localisation (per edge) | 15 mm | 30 mm | 40 mm | depth resolution |
| floor/ceiling level | 3 mm | 10 mm | 15 mm | plane fits |
| unobserved wall face | 80 mm | 150 mm | 250 mm | bounded by free space only |
| global scale (log) | 0 | fused, per capture | fused, per room | section 6 |

Statistical fit errors are combined with these in quadrature. A plane fitted to 50 000 points has a
sub-millimetre standard error, and reporting that would be confident garbage. Areas and footprints
propagate wall uncertainties by Monte Carlo. Unobserved ceilings are reported as a bounded prior
interval (2.2-3.4 m, lower bound raised to the highest wall point seen), never as a number.

## 8. Calibration

Each quantity's interval is scaled by a per-tier split-conformal factor fitted on the benchmark
(`groundplan bench calibrate`): score = |error| / reported half-width, factor = the
ceil((n+1)(1-alpha))-th score. Coverage is reported **leave-one-capture-out** (fit on the other
captures, test on the held-out one), which is the number that predicts the walk-in test. Until
fitted, multipliers are conservative priors and every Measurement says `calibrated: false`.
Before the interval fit, a systematic LiDAR depth-scale error is estimated (median truth/measured
over walls and ceilings) and written as a correction only if a factor fitted on the other captures
lowers the held-out error; the multipliers are then fitted on the corrected values.
Coverage per tier and quantity: **[bench]**.

## 9. Benchmark results

Composition as specified (multi-room + connector at all tiers, staged two-class damage, repeated
room per tier, laser truth, Polycam head-to-head): `docs/BENCHMARK_PROTOCOL.md`.
Gates, repeatability table, head-to-head table and timing: **[bench]**.

## 10. Damage, flags, scope

Orthomosaics (5 mm walls) take each pixel from the best-facing views with depth-tested occlusion
and a median over the top three views (removing glare and moving shadows). Candidates come from
colour distance to the surface's own background and from ridges; they are rejected if they stand
proud of the surface (measured per pixel from the depth maps: handles, racks, switches), sit in the
skirting or cornice band, hug unobserved edges, or (for cracks) are straight axis-aligned lines or
thicker than 1.2 cm; CLIP then verifies a natural crop against damage classes and 37 look-alikes.
On the undamaged assessor flat the first version reported 6 regions (all false); the final reports
0. Staged on the same footage *(assessor data, decals projected through the capture's poses)*:
water stain 0.345 x 0.245 m (truth 0.36 x 0.26), crack 0.41 x 0.125 m (truth 0.40 x 0.12), both
intervals containing the truth. Twelve concealed-damage rules (IICRC S500/S520, BRE 251) fire with
the facts that satisfied them; scope quantities carry propagated intervals.

## 11. Fix loop

**[after the benchmark]** Declaration (worst gate, failing number, root cause with evidence, fix,
predicted number) committed before the fix (`fixloop/DECLARATION.md`, tag `fixloop-declared`);
before/after regenerated from worktrees by `fixloop/run_fixloop.py`.

## 12. Mirrors, glass, wet-look surfaces, low light

* **Mirrors** return the reflected room "behind" the wall: an opening candidate whose behind-points,
  reflected across the plane, land on observed room surfaces is a mirror and stays wall.
* **Glass** returns little or no depth: no-return rays crossing a wall plane within 3.2 m count as
  open evidence (windows, glass doors); a glass shower screen is not a wall and is not reported.
* **Wet-look floors** create virtual points below the floor: the floor is the lowest *strong*
  horizontal level, sparse sub-floor points are ignored; mosaics use a multi-view median.
* **Low light** degrades the RGB-dependent parts (video SfM, damage), not LiDAR geometry; frames
  are chosen by sharpness, and failure is reported rather than guessed (see WALK_IN.md).
Benchmark hard-case captures (mirror, glass, glossy floor, dimmed room): **[bench]**.

## 13. Known failure modes

1. Opening widths at the LiDAR tier are bounded by depth resolution (about 1 cm per pixel at
   1.3 m); 2 cm on 85 % is not met by depth alone *(synthetic 4/8)*.
2. Photo tier on plain walls without compass headings: feature registration fails and rooms fall
   back to the observed footprint with unobserved walls (wide intervals, flagged).
3. Non-rectangular rooms in the photo tier: structural registration assumes one wall per facing and
   is rejected (rank/residual check), leaving feature registration.
4. Scale priors are residential and regional (door 2.04 m, ceiling 2.6 m); unusual buildings widen
   the fused interval or trigger cue rejection.
5. Tall furniture against a wall when the ceiling was never seen: the face cannot be told from a
   wall, so the floor area under it is lost (the outline follows the furniture front).
6. Memory: an 8 GB laptop is enough only without other heavy processes; structure from motion runs
   on two threads for that reason.
7. Objects lying flat on a wall (a towel, a poster) can pass the protrusion test and be reported as
   damage (one such region on the multi-room assessor capture, after the colour/texture guards).
8. Video clips with fast turns at close range past plain walls fragment; pieces that cannot be
   placed with two agreeing frame pairs are dropped and reported, never guessed (section 6).
