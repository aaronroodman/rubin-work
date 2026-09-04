---
name: aos-private-consdb-cache
description: "Planned lightweight private ConsDB/cache for AOS per-visit telemetry that needs EFD (M1M3 gradients, LUT values)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 25f21140-87f5-47d9-b0ac-9fc8c2e9c154
  modified: 2026-08-11T21:28:18.338Z
---

Deferred project (recorded 2026-08-11, not started): build a **lightweight private "shadow ConsDB"** — a small real database (sqlite/postgres) or parquet cache keyed by `(day_obs, seq_num)` — holding AOS per-visit quantities that require **EFD access** and are therefore slow/impossible to get from Slurm **batch compute nodes**.

**Why:** In the FAM/AOS pipeline, per-visit thermal telemetry splits by source: the ESS air temps/deltas and TMA truss temps come from the ConsDB transform (`efd_lsstcam`, fast, reachable everywhere), but the **4 M1M3 mirror gradients** (`x/y/z/radial_gradient`, computed from M1M3 thermocouples) come **only from the EFD**. The EFD is reachable from the RSP terminal / slaciana interactive nodes but NOT from milano/torino/roma batch compute nodes, so batch `mktable` silently drops all thermal and it has to be backfilled by hand afterward. The M1M3 gradients WILL eventually be added to the official ConsDB transform, but the plan is to NOT backfill pre-existing visits — so a private cache is the only way to avoid re-probing the EFD forever.

**How to apply:** Contents to hold — start with the M1M3 gradients; later add the **M1M3 LUT values as applied** (applied elevation/gravity/temperature forces / LUT terms) and any other AOS quantity that needs EFD. Sketch: probe the EFD once per visit from the RSP terminal → store in the cache → analysis/batch jobs read from the cache → keep current with a periodic RSP-terminal update job. Wire it in as a cache consulted before hitting the EFD in `aos/code/run_backfill_thermal.py` (the existing interactive visit-level backfill tool) and/or `get_thermal_data` / `get_m1m3_gradients` in the package `intrinsics_lib.py` (branch `tickets/RSO-809`). ~50 lines for the gradient-only version; a full DB only if the contents grow. Two related durable fixes on RSO-809: (1) split the all-or-nothing thermal `try/except` in `run_mktable` so an EFD outage still attaches the 9 ConsDB columns; (2) upstream, get M1M3 gradients into the ConsDB transform so nothing needs the EFD.
