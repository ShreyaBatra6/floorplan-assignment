# Capture protocol (follow exactly)

**Before any capture (2 min).** Switch on every light. Open curtains and blinds. **Open every interior door fully**: an open door can be measured and links the rooms; a closed one cannot. Move anything standing in a doorway. Phone charged ≥ 30 % with ≥ 3 GB free.

| Tier | Phone | App | Time |
|---|---|---|---|
| **A · LiDAR** | iPhone 12 Pro or newer *Pro*, or iPad Pro (has LiDAR) | **Stray Scanner** (free, App Store, by Stray Robots) | ~1 min per room |
| **B · Video** | any iPhone 15 or newer | built-in **Camera** app | ~1 min per room |
| **C · Photos** | any iPhone 15 or newer | built-in **Camera** app | 4–8 photos per room |

## A · LiDAR: Stray Scanner, one continuous recording
1. Install **Stray Scanner** from the App Store; open it and allow camera access. Do not change its settings.
2. Stand in the entrance. Tap **record**. Hold the phone **upright at chest height**, about **1 m from the walls**.
3. Walk **slowly (half your normal pace)** along the walls of the first room. Every 1–2 steps, **tilt up until the wall–ceiling line shows, then down until the wall–floor line shows**.
4. At each doorway: stop 1 m before it, point at the door frame for 2 seconds, walk through slowly pointing ahead.
5. Visit every room the same way. **Finish where you started**, pointing at the first wall for 2 seconds. Tap **stop**.
6. **Hand-off:** Files app → On My iPhone → Stray Scanner → press and hold the newest folder → **Compress** → AirDrop / USB / cloud the `.zip` to the computer.

## B · Video: Camera app, one continuous clip
1. Settings → Camera → Record Video: **1080p at 30 fps**. Turn **off** Action mode and Cinematic. Use the **1×** lens; **never zoom**.
2. Hold the phone **sideways (landscape)** at chest height. Start recording at the entrance and walk the **same path as A**: slow, along the walls, tilting up to the ceiling line and down to the floor line every 2 steps, 2 s pause at each doorway, every room, back to the start.
3. **Hand-off:** AirDrop to a Mac, or USB to Windows with Settings → Photos → Transfer to Mac or PC → **Keep Originals**. **Do not send through WhatsApp or other messengers** (they strip the data the pipeline needs).

## C · Photos: Camera app, one folder per room
1. Settings → Privacy & Security → Location Services → **Camera → While Using** (the compass stored in each photo helps join rooms). Photo mode, **1×** lens, **no zoom, no Portrait, flash off**. Hold the phone **sideways**.
2. **Walls:** stand with your **back against the middle of a wall** and photograph the **opposite wall**, so that **both of its corners, the floor line and the ceiling line** are in the picture. Repeat from the middle of every wall: a rectangular room gives 4 photos.
3. **Doorways:** for each doorway, stand 1 m inside the room facing it, so the **whole door frame and part of the next room** are in the picture: 1 photo per doorway.
4. Hallways: one photo from each end looking along it, plus one per doorway. **At most 8 photos per room.**
5. *Optional, improves accuracy:* lay **one sheet of A4 or Letter paper flat on the floor** where at least two photos of the room show it.
6. **Hand-off:** make **one folder per room**, named after the room (`01 Living`, `02 Hall`, ...), put that room's original photos in it (HEIC is fine), and put all room folders inside one folder. Same messenger warning as B.

## Avoid (all tiers)
Running or fast turns (take at least 2 s per quarter turn) · pointing at a mirror or window for more than a second · lights off · fingers over the camera · walking backwards · recordings over 8 minutes (split the property by floor).

## Run (one command per capture)
`groundplan run <the .zip, video file, or photos folder>`, then open `runs/<name>/plan.png` and `runs/<name>/report.html`.
