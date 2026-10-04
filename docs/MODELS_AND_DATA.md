# Models, data, tools and APIs used (disclosure)

The brief allows any pretrained model, dataset or API with disclosure, provided everything runs
without calling our own infrastructure. Nothing in groundplan calls a server we operate. Model
weights are downloaded from the public Hugging Face hub on first use (or ahead of time with
`scripts/fetch_models.py`) and then run locally on the CPU.

## Pretrained models

| Model | Used for | Source | License | Size |
|---|---|---|---|---|
| Depth Anything V2 Metric-Indoor-Small | metric monocular depth (photo, video tiers) | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Apache-2.0 | 25 M params, ~100 MB |
| CLIP ViT-B/32 | damage verification (zero-shot), room type | `openai/clip-vit-base-patch32` | MIT | ~600 MB |

Evaluated and not used by default: Depth Anything V2 Metric-Indoor-Base (CC-BY-NC-4.0; no better
scale on the assessor frames: median LiDAR/model ratio 0.710 vs 0.684, wider spread); MoGe-2 (MIT;
its current release builds CUDA extensions from source, which breaks a 15-minute clean install on a
CPU laptop).

## Libraries and tools

| Tool | Used for | License |
|---|---|---|
| pycolmap / COLMAP 4.x | structure from motion (video tier) | BSD-3-Clause |
| OpenCV | image processing, SIFT | Apache-2.0 |
| NumPy, SciPy, scikit-image, shapely, networkx | geometry, optimisation | BSD / MIT-style |
| PyTorch, transformers, huggingface_hub | model inference | BSD / Apache-2.0 |
| PyAV (FFmpeg) | video decoding with rotation and QuickTime metadata | BSD-3 / LGPL (FFmpeg) |
| pillow, pillow-heif | iPhone HEIC photos and EXIF | MIT-CMU / BSD-3 |
| pydantic, jsonschema | output contract and schema validation | MIT |
| matplotlib | rendered plans | PSF-style |

## Capture applications

| App | Tier | Notes |
|---|---|---|
| Stray Scanner (Stray Robots), App Store, free | LiDAR | open source (MIT); exports depth, confidence, ARKit poses, intrinsics, IMU, RGB video |
| iPhone Camera (built in) | photo, video | stills keep EXIF focal length and compass heading; videos keep QuickTime focal-length metadata |
| Polycam (free tier), Room mode | head-to-head only | the incumbent app the LiDAR tier is compared against |

## Data

* **Assessor sample captures** (Round 1, Stray Scanner format): used for development, for the pose
  convention check, for measuring the depth model's scale against LiDAR, and for the staged-damage
  test. They have no ground truth, so no accuracy claim rests on them.
* **Benchmark captures** (ours, with laser/tape ground truth): composition and measurement rules in
  `docs/BENCHMARK_PROTOCOL.md`; raw data fetched by `scripts/fetch_data.py` with SHA-256 checks.
* **Synthetic flats** (`src/groundplan/sim/`): ray-cast scenes with exact ground truth used in the
  test suite and for the drift ablation under controlled drift.
* No third-party dataset was used to train or fine-tune anything.

## External APIs

None. (No cloud vision or language API is called at any point.)

## Development assistance

An AI coding assistant (Claude Code) was used while writing this repository, as the brief allows;
commits it co-authored carry a `Co-Authored-By` trailer. Every design decision is documented in the
technical report and defended by the author.
