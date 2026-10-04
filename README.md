# groundplan

**Phone capture in, measured property plan out.** One command turns a capture at any of three
tiers (a folder of photos, a walkthrough video, or a LiDAR scan) into the same output: a
dimensioned per-room plan, one stitched whole-property plan, per-surface damage regions,
concealed-damage flags that name the rule that fired, a scope of work keyed to surfaces, and a
**calibrated 90 % interval on every number**.

| Tier | Input (see [`docs/CAPTURE_PROTOCOL.md`](docs/CAPTURE_PROTOCOL.md)) | Metric scale from | Expected accuracy |
|---|---|---|---|
| LiDAR | Stray Scanner export (Pro iPhone / iPad Pro) | the depth sensor | see [`docs/DEVICE_MATRIX.md`](docs/DEVICE_MATRIX.md) |
| Video | iPhone Camera clip, 1080p30 | structure from motion + fused scale cues | intervals widen with the scale uncertainty |
| Photo | 2-8 iPhone photos per room, one folder per room | fused scale cues | widest intervals; rooms stitched by door pairing |

## Quick start (clean machine, ~10 minutes)

```bash
# 1. get the code and an isolated environment (uv installs Python 3.11 itself)
git clone https://github.com/ShreyaBatra6/floorplan-assignment.git groundplan && cd groundplan
python -m pip install uv          # or see https://docs.astral.sh/uv/
uv sync --extra models --extra video
uv run python scripts/fetch_models.py      # depth + CLIP weights (~0.6 GB, Hugging Face)

# 2. one command per capture (tier detected automatically)
uv run groundplan run path/to/stray_export.zip       # LiDAR
uv run groundplan run path/to/IMG_0042.MOV            # video
uv run groundplan run path/to/photo_folders/          # photos: one sub-folder per room
```

Windows: `scripts\setup.ps1` does step 1. macOS/Linux: `scripts/setup.sh`.

Timed on a fresh clone (Windows 11, i5-1135G7, 8 GB, CPU only): clone 1 s, `uv sync` 25 s,
model fetch 8 s, first full run on the assessor LiDAR sample 177 s, total **3.5 min**, with the uv
and Hugging Face caches already warm. A machine with empty caches also downloads about 1.2 GB
(PyTorch CPU wheels plus ~0.7 GB of weights): roughly 3-8 more minutes on a typical connection,
which keeps the whole path under 15 minutes. Example outputs: [`examples/`](examples/).
If the repository lives in a synced folder (OneDrive, iCloud, Dropbox), put the environment
outside it: `set UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\groundplan` (Windows) or
`export UV_PROJECT_ENVIRONMENT=~/.venvs/groundplan`, otherwise the sync client re-uploads the
environment's thousands of files.

## What a run writes (`runs/<capture>/`)

| File | Contents |
|---|---|
| `plan.json` | the output contract, validated against [`schema/groundplan.schema.json`](schema/groundplan.schema.json) |
| `plan.png`, `plan.svg` | stitched whole-property plan: rooms, walls, doors, windows, every wall dimensioned with its interval |
| `rooms/R*.svg` | dimensioned per-room plans with opening widths |
| `summary.md` | the numbers as tables (read this out at a walk-in test) |
| `damage/D*.jpg` | evidence crop for each damage region |

Every measurement in `plan.json` is `{value, lo, hi, unit, confidence, sigma_abs, sigma_log, method,
calibrated}`. `calibrated` stays `false` until the interval multipliers have been fitted on the
benchmark (`groundplan bench calibrate`).

## Commands

```
groundplan run <capture> [--tier lidar|video|photo] [--no-drift] [--no-damage] [--no-cache] [-o dir]
groundplan inspect <capture>             what the pipeline will see, before running
groundplan validate runs/*/plan.json     schema + cross-reference checks
groundplan eval <plan.json> <truth.yaml> score one run against laser/tape ground truth
groundplan bench run|score|calibrate     the full benchmark (see docs/BENCHMARK_PROTOCOL.md)
```

`--no-drift` is the drift ablation (ARKit poses as-is). `--no-cache` forces live model inference;
by default model outputs are replayed from a content-hash cache, bit-identically.

## How it works (short version; details in [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md))

* **One geometric core for all tiers.** Every tier produces a fused, gravity-aligned point cloud
  with camera rays; the core finds floor and ceiling levels, wall faces (weighted by vertical
  extent, split by facing, tall furniture suppressed), rooms (ray-carved free space, watershed
  merged across boundaries wider than a door), rectilinear outlines from the wall-face arrangement
  refitted to the points, and openings as holes in wall faces confirmed by rays through them.
* **LiDAR:** ARKit depth + poses, with a 4-DoF submap pose graph (ICP loop closures weighted by
  their information matrix, Manhattan plane anchoring) correcting drift.
* **Video:** structure from motion (pycolmap), monocular metric depth aligned per frame to the
  reconstruction, metric scale fused from independent cues.
* **Photo:** per-room registration from the walls each photo sees (texture-free, linear least
  squares made observable by the shared floor-to-ceiling height), feature matching as fallback;
  rooms stitched by door pairing with compass headings and no-overlap constraints.
* **Damage:** per-surface orthomosaics at 5 mm (walls), candidates from colour and ridge
  detectors, rejected when they protrude from the surface or look like trim, verified by CLIP;
  extents measured on the metric raster. Concealed-damage rules in
  [`src/groundplan/damage/rules.yaml`](src/groundplan/damage/rules.yaml); scope quantities carry
  intervals.
* **Intervals:** an explicit error budget per quantity and tier, then split-conformal multipliers
  fitted leave-one-capture-out on the benchmark.

## Repository map

```
src/groundplan/   contract.py (output schema) · pipeline.py · tiers/ · geometry/ · damage/ · bench/ · render/
schema/           published JSON Schema
docs/             capture protocol, device matrix, benchmark protocol, technical report, compliance matrix
benchmark/        manifest, ground truth, app exports, results
fixloop/          fix declaration and before/after regeneration
tests/            unit + end-to-end tests (synthetic flats with exact ground truth; staged damage on real footage)
scripts/          setup, model/data fetch, development helpers
```

Public stand-in benchmark (ARKitScenes homes with laser-scan truth, until our own captures exist):
[`docs/PUBLIC_BENCHMARK.md`](docs/PUBLIC_BENCHMARK.md).
Requirement-by-requirement status: [`docs/COMPLIANCE_MATRIX.md`](docs/COMPLIANCE_MATRIX.md).

## Models, data, tools and APIs (disclosure)

Nothing calls a server we operate, and no cloud vision or language API is used. Model weights are
downloaded from the public Hugging Face hub on first use (or ahead of time with
`scripts/fetch_models.py`) and run locally on the CPU.

| Model | Used for | Source | License |
|---|---|---|---|
| Depth Anything V2 Metric-Indoor-Small | metric monocular depth (photo, video tiers) | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Apache-2.0 |
| CLIP ViT-B/32 | damage verification (zero-shot), room type | `openai/clip-vit-base-patch32` | MIT |

Evaluated and not used by default: Depth Anything V2 Metric-Indoor-Base (CC-BY-NC-4.0; no better
scale on the assessor frames) and MoGe-2 (MIT; its release builds CUDA extensions from source,
which breaks a 15-minute CPU install).

**Libraries:** pycolmap/COLMAP (BSD-3, structure from motion), OpenCV (Apache-2.0), NumPy, SciPy,
scikit-image, shapely, networkx (BSD/MIT-style), PyTorch, transformers, huggingface_hub
(BSD/Apache-2.0), PyAV/FFmpeg (BSD-3/LGPL), pillow, pillow-heif (MIT-CMU/BSD-3), pydantic,
jsonschema (MIT), matplotlib (PSF-style).

**Capture apps:** Stray Scanner (Stray Robots, free, open source MIT) for LiDAR; the built-in iPhone
Camera for photos and video; Polycam (free tier, Room mode) for the head-to-head only.

**Data:**
* Assessor sample captures (Round 1): development only (pose convention, depth-model scale against
  LiDAR, staged-damage test). No ground truth, so no accuracy claim rests on them.
* Our benchmark captures (laser/tape truth): protocol in `docs/BENCHMARK_PROTOCOL.md`, raw data
  fetched by `scripts/fetch_data.py` with SHA-256 checks (pending capture).
* Synthetic flats (`src/groundplan/sim/`): exact ground truth for the test suite and the drift ablation.
* [ARKitScenes](https://github.com/apple/ARKitScenes) (Apple; Baruch et al., NeurIPS 2021 Datasets and
  Benchmarks), Apple's ARKitScenes license, **non-commercial use**: iPad Pro 2020 recordings and Faro
  laser scans of real homes as the public stand-in benchmark (`docs/PUBLIC_BENCHMARK.md`). Downloaded
  from Apple by `scripts/arkitscenes.py`, never redistributed; only measurements derived from the laser
  scans and their review renders are committed. Evaluation and calibration only.
* No dataset was used to train or fine-tune anything.
