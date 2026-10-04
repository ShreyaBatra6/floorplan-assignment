# Walk-in test runbook

The assessors capture a space we have never seen, with their own iPhone 15 or newer, following
`docs/CAPTURE_PROTOCOL.md`, at a tier they choose on the day. We run it cold, in front of them.

## Before the session (the day before)

```
uv sync --extra models --extra video
uv run python scripts/fetch_models.py          # no downloads needed during the session
uv run pytest -q -m "not slow"                 # sanity
uv run groundplan run <any previous capture>   # warm the caches, confirm timings on this laptop
```

Close other heavy applications (browsers, sync clients). Bring a USB-C/Lightning cable and a
spare way to receive files (AirDrop to a Mac, or USB to the laptop). Print the capture protocol.

## On the day

1. Hand over the printed protocol; answer questions only by pointing at the page.
2. Receive the files exactly as the protocol says (zip / video / photo folders).
3. `uv run groundplan inspect <capture>`: confirm the tier detected is the tier they chose.
4. `uv run groundplan run <capture>` and open `runs/<name>/plan.png`, `summary.md`.

| Tier | Typical runtime on an i5 laptop | What dominates |
|---|---|---|
| LiDAR | 1-3 min | depth fusion, drift graph, damage mosaics |
| Video | 4-8 min | structure from motion, depth model |
| Photo | 1-3 min | depth model per photo |

`--no-damage` gives the geometry in about half the time if they want measurements first.

## While they measure with their laser

Read numbers from `summary.md` as **value with its 90 % interval**. If an interval is wide, say
why (the plan states it: unobserved wall, scale uncertainty of the tier, ceiling not seen).

## If something goes wrong

| Symptom | Cause | Action |
|---|---|---|
| "no floor found" | the capture never pointed at the floor | ask for a re-capture following step 3 of the protocol |
| ceiling reported as a prior interval | the ceiling was never in view | report the interval; do not improvise a number |
| video: structure from motion failed | very fast motion or darkness | photo tier from the same visit, or re-record slowly |
| a room missing from the photo plan | its photos did not overlap | the plan lists it as unconnected; state it |
| out of memory | other applications | close them and re-run (results are deterministic) |

Never adjust a number by hand. The plan, its intervals and its warnings are the answer.
