# Fix loop bundle

The sequence the brief asks for, one command per step. `fixloop/loop.py` enforces the order with
git tags, so the record shows the prediction was made before the fix existed.

| Step | Command | What it does |
|---|---|---|
| 0 | `python fixloop/loop.py status` | where you are and what comes next |
| 1 | `python fixloop/loop.py before` | needs a clean tree and a filled `benchmark/manifest.yaml`; tags HEAD **`fixloop-before`**, runs the full benchmark into `benchmark/results/<sha>/`, ranks the failing gates by how much of their requirement they meet, writes section 1 of [`DECLARATION.md`](DECLARATION.md) (worst gate, threshold, failing number) and commits the run |
| 2 | *(by hand)* | write sections 2-3 of the declaration: root-cause hypothesis **with evidence from the before run**, alternatives ruled out, the fix, the **predicted number** |
| 3 | `python fixloop/loop.py declare` | refuses while sections 1-3 have blanks or if `src/`/`tests/` changed since `fixloop-before`; commits the declaration and tags **`fixloop-declared`** |
| 4 | *(ordinary commits)* | write the fix |
| 5 | `python fixloop/loop.py after` | refuses unless `src/` changed since the declaration; tags **`fixloop-after`**, regenerates both runs from their own git worktrees on the same raw data ([`run_fixloop.py`](run_fixloop.py)), writes [`RESULT.md`](RESULT.md) (every gate, before vs after) and [`fix.diff`](fix.diff), and fills the after-run value into section 4 |
| 6 | *(by hand)* | section 4: prediction vs outcome; if the gate still fails, why, with numbers. Commit `RESULT.md`, `fix.diff`, `runs/*/metrics.json` and the declaration |

`run_fixloop.py` builds one Python environment per ref with uv (found on PATH, in `$UV`, or in the
per-user install folder). Raw data comes from `$GROUNDPLAN_DATA` (default `./data`).

Candidate causes are not pre-judged here: the gate that fails worst on the real benchmark decides
what gets fixed.
