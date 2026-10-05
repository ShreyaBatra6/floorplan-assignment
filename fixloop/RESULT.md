# Fix loop: before vs after

`fixloop-before` vs `fixloop-after`, same raw data (`manifest_lidar.yaml`); the before run is `C:/Users/shrey/OneDrive/Pictures/Documents/Brynz/groundplan/benchmark/public/results/6b78d8d`, made at the `fixloop-before` commit; the after run is regenerated from its own worktree.

| gate | tier | before | after | before status | after status |
|---|---|---|---|---|---|
| wall lengths | lidar | 10/15 within; coverage 47% | 10/15 within; coverage 47% | FAIL | FAIL |
| ceiling height | lidar | 3/9 rooms; max 3.9 cm | 3/9 rooms; max 3.9 cm | FAIL | FAIL |
| opening widths | lidar | 0% (0/10) | 0% (0/8) | FAIL | FAIL |

Repeatability before (ark_466183_lidar_1 vs ark_466183_lidar_2): PASS, 3/3 within tolerance

Repeatability before (ark_466183_lidar_1 vs ark_466183_lidar_3): FAIL, 2/3 within tolerance

Repeatability before (ark_422378_lidar_1 vs ark_422378_lidar_2): FAIL, 2/3 within tolerance

Repeatability before (ark_422378_lidar_1 vs ark_422378_lidar_3): FAIL, 2/3 within tolerance

Repeatability before (ark_471948_lidar_1 vs ark_471948_lidar_2): FAIL, 1/2 within tolerance

Repeatability before (ark_471948_lidar_1 vs ark_471948_lidar_3): FAIL, 1/2 within tolerance

Repeatability after (ark_466183_lidar_1 vs ark_466183_lidar_2): PASS, 3/3 within tolerance

Repeatability after (ark_466183_lidar_1 vs ark_466183_lidar_3): FAIL, 2/3 within tolerance

Repeatability after (ark_422378_lidar_1 vs ark_422378_lidar_2): FAIL, 2/3 within tolerance

Repeatability after (ark_422378_lidar_1 vs ark_422378_lidar_3): FAIL, 2/3 within tolerance

Repeatability after (ark_471948_lidar_1 vs ark_471948_lidar_2): FAIL, 1/2 within tolerance

Repeatability after (ark_471948_lidar_1 vs ark_471948_lidar_3): FAIL, 1/2 within tolerance

Source diff: [`fix.diff`](fix.diff) (99 changed lines). Declaration and prediction: `fixloop/DECLARATION.md`.
