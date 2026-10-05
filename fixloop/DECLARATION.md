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

- Hypothesis: _____________________
- Evidence (numbers from the before run, plots, a diagnostic that isolates the cause):
  1. ...
  2. ...
- Alternatives considered and why the evidence rules them out: ...

## 3. The fix and the predicted number

- Fix: _____________________ (files: ______)
- Predicted value of the gate after the fix: ______ (gate passes / moves to ______)
- What else could move, and the bound I expect on it: ...

## 4. Outcome (after the fix ships; regenerate with `python fixloop/run_fixloop.py`)

- After-run value: ______ (commit `fixloop-after`)
- Prediction vs outcome: ______
- If the gate did not pass: why it fell short, with the numbers.
