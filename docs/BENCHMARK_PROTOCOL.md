# Building the benchmark set

The brief specifies the benchmark's composition so that it cannot be flattered. This page is the
checklist that satisfies it, the way each quantity is measured (so ground truth and pipeline mean
the same thing), and how the raw data is organised so `groundplan bench run` regenerates every
reported number.

## 1. What to capture (half a day)

| # | Capture | Tier(s) | Why (brief requirement) |
|---|---|---|---|
| 1 | The **whole flat in one walk**: ≥ 3 rooms **plus a hallway/connector** | LiDAR, video | multi-room capture; drift handling + ablation |
| 2 | The **same rooms as per-room photo folders** | photo | photo-tier whole-property stitch |
| 3 | **Capture 1 again** (second walk, same tier) | LiDAR | repeatability gate (two captures, same rooms, same tier) |
| 4 | **One room twice** | video and photo | repeatability table at every tier |
| 5 | **Furnished room with staged damage, two classes** (e.g. water stain + crack), all three tiers | all | damage regions, flags, scope |
| 6 | **Two rooms with Polycam** (free, Room mode) | app | head-to-head (Part 3) |
| 7 | Hard cases: a bathroom **mirror**, a **glass** door or large window, a **glossy/wet-look floor**, and **one room with the lights dimmed** (an extra LiDAR + photo capture) | LiDAR, photo | "real properties contain mirrors, glass, wet-look surfaces and low light" |

Follow `docs/CAPTURE_PROTOCOL.md` literally for every capture: the walk-in test will be captured
from that page by someone else, so the benchmark should be too.

## 2. Staging damage without damaging anything

* **Water stain:** brew very strong tea, paint an irregular blotch with a darker rim on off-white
  A3 paper, let it dry, trim loosely and fix it flat to the wall with removable tape on the back.
  Or print a photo of a real stain at roughly real size (30-50 cm). One stain should touch the
  bottom 15 cm of a wall (it exercises the wall-base wicking rule CDR-02).
* **Crack:** a thin wandering 2-3 mm dark grey line, 30-60 cm long, drawn on painter's tape that
  matches the wall colour (or printed), fixed flat.
* **Mould (optional third class):** a printed photo of a mould patch.
* Measure each staged region: **width, height, height of its bottom edge above the floor, and
  distance from the wall's start corner** (see wall numbering below).

## 3. Measuring ground truth

Use a laser distance meter (record make, model and stated accuracy) or a steel tape. Take **every
reading twice** and write both down; the benchmark uses the mean and keeps the spread.

* **Wall numbering.** Stand inside the room at its entrance. **W1 is the wall containing the
  entrance door.** Walk around the room keeping the wall at your **right shoulder**; number the
  walls W2, W3, ... in the order you reach them (this is counter-clockwise seen from above).
  A wall's *start corner* is the corner you reach it from.
* **Wall length:** interior face to interior face, corner to corner, at about **1 m height** (above
  skirting, below sills). If furniture blocks the line, measure higher (up to 1.8 m) and note it.
* **Ceiling height:** floor to ceiling, vertical, at **three spots** (room centre and two spots at
  least 0.5 m from the walls). Rooms with a bulkhead: measure the main ceiling, note the bulkhead.
* **Diagonals** (rectangular rooms): corner to corner both ways at 1 m height.
* **Doors and passages:** **width = clear distance between the two jamb faces (the inside of the
  door frame, not the trim), at mid-height**; height = floor to the underside of the head jamb;
  offset = from the wall's start corner to the nearer jamb. **List every door in both rooms it
  opens into** (each room sees it in one of its walls).
* **Windows:** width between the inside faces of the frame; sill height floor to sill; height sill
  to head.
* **Adjacency:** list the pairs of rooms joined by a door or open passage.

Write the numbers into `benchmark/ground_truth/<site>.yaml` (copy `TEMPLATE.yaml`). The
printable sheet `benchmark/templates/measurement_sheet.md` has the same layout for pencil-and-paper
on site.

## 4. Head-to-head app (Part 3)

1. Install **Polycam** (free tier) on the LiDAR device. Note its version (App Store → Polycam →
   version history, or the app's settings).
2. Scan the two benchmark rooms in **Room** mode (one scan per room), following the app's prompts.
3. Export what the free tier allows (GLTF) and **screenshot the floor plan and room details**
   showing the app's dimensions. Save both in `benchmark/app_exports/`.
4. Type the app's numbers into `benchmark/app_exports/polycam_dimensions.csv`
   (`room,dimension,app_value,source`, using the ground-truth wall and opening ids).

## 5. Organising the raw data

```
data/<site>/<capture_id>/         raw capture exactly as handed off (Stray zip/folder, .MOV, photo folders)
benchmark/ground_truth/<site>.yaml
benchmark/app_exports/            Polycam GLTF + screenshots + polycam_dimensions.csv
benchmark/manifest.yaml           one entry per capture (id, site, tier, path, repeat_of)
```

`data/` is not committed (it is large). `scripts/fetch_data.py` downloads it from the published
archive and verifies SHA-256 checksums; upload the archive and record its URL and checksums in
`data/MANIFEST.json`. Then:

```
groundplan bench run            # every capture, live; writes benchmark/results/<git sha>/
groundplan bench calibrate <results> --write   # fit interval multipliers (leave-one-capture-out)
groundplan bench run            # re-run with the fitted calibration
```
