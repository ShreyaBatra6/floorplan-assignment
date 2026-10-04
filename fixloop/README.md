# Fix loop bundle

The sequence the brief asks for, and how each step is evidenced in git:

1. Run the full benchmark on the current code; tag it **`fixloop-before`**.
2. Fill in sections 1-3 of [`DECLARATION.md`](DECLARATION.md) (worst gate and its failing number,
   root-cause hypothesis with evidence, the fix and the predicted number); commit and tag
   **`fixloop-declared`**. The prediction is fixed before any fix is written.
3. Ship the fix in ordinary commits; tag the result **`fixloop-after`**.
4. `python fixloop/run_fixloop.py` regenerates both runs from their own worktrees on the same raw
   data and writes [`RESULT.md`](RESULT.md) (gates before vs after) and [`fix.diff`](fix.diff) (the
   readable source diff).
5. Fill in section 4 of the declaration (outcome vs prediction; if short of the gate, why).

Candidate causes are not pre-judged here: the gate that fails worst on the real benchmark decides
what gets fixed.
