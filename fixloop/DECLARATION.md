# Fix declaration

*Filled in after the first full benchmark run, before the fix is written. The text of sections
1-3 is committed (and tagged `fixloop-declared`) before any change to the code, so the prediction
cannot be edited after the fact.*

## 1. Worst-performing gate

- Gate: _____________________ (tier: ______)
- Threshold (from the brief): _____________________
- Failing number: _____________ on benchmark run `benchmark/results/<sha>` (commit `fixloop-before`)

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
