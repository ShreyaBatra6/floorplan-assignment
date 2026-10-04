# Example outputs

Produced by `groundplan run` on the Round 1 assessor sample capture `single_room/c00a170fe1`
(Stray Scanner, LiDAR tier), with the code at the commit recorded in each `plan.json`
(`runtime.git_commit`). They show the output contract and the rendered products; they are **not**
accuracy evidence, because the sample capture has no ground truth (accuracy is measured on the
benchmark, `docs/BENCHMARK_PROTOCOL.md`).

| File | What it is |
|---|---|
| `assessor_single_room_lidar/plan.json` | the full output contract (validates against `schema/groundplan.schema.json`) |
| `assessor_single_room_lidar/plan.png`, `plan.svg` | stitched plan: rooms, walls, openings, every wall dimensioned with its 90 % interval |
| `assessor_single_room_lidar/rooms/*.svg` | dimensioned per-room plans with opening widths |
| `assessor_single_room_lidar/summary.md` | measurement tables |
| `assessor_single_room_lidar/report.html` | self-contained report (open in a browser) |

What to notice: the ceiling was never in view in this capture, so every ceiling height is a bounded
prior interval (2.2-3.4 m) with a warning rather than a number; walls the scan never saw carry
much wider intervals than observed ones; no damage is reported on this undamaged flat.

Regenerate: `uv run groundplan run <SampleData>/single_room/c00a170fe1 -o examples/assessor_single_room_lidar`.
