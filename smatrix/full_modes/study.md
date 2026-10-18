---
study: full_modes
topic: smatrix
status: done
question: "How much wavefront leverage do the bending modes beyond the 20 per mirror that the AOS uses carry?"
answer: "M1M3 modes stay active well past 20: cumulative wavefront variance reaches 90% (dimensionless) only by mode about 99; M2 reaches it by about 36. B52 is 96% m=0 (dimensionless, power fraction) and drives Z4, Z11 and Z22, and is not in the AOS set."
uses: {}  # no rubin-work products; smatrix-internal bend_full and bend_zemax mode sets
started: 2026-07-31
---

# `full_modes`: the full M1M3 and M2 bending-mode sets

> **Code:** `code/` (analysis and plots; the builder `smatrix/code/build_full_modes.py` and `compute_smatrix.py` stay in the topic) · **Write-up:** [`../README.md`](../README.md), Full bending-mode set; [`../docs/plots.md`](../docs/plots.md) sections 3, 4, 6, 7 · **Output:** `smatrix/output/` (topic data)

## Question

How much wavefront leverage do the bending modes beyond the 20 per mirror that the AOS uses carry?

## Current answer

From the full FEA mode sets, 156 M1M3 and 72 M2 modes (`bend_full`, 238 DOF with the 10 rigid-body DOF):

- **M1M3 modes stay optically active well past the 20 the AOS uses:** cumulative wavefront variance reaches 90% (dimensionless) only by mode about 99. M2 falls off faster, 90% by about 36.
- Mode N drives progressively higher pupil Noll (the diagonal coupling in the pupil heatmap).
- **B52** (raw index 51) is 96% m=0 (dimensionless, power fraction), drives Z4, Z11 and Z22, and is not in the AOS-used 20.
- The M1M3 full modes equal the batoid/OFC modes (corr +1.000, dimensionless); M2 matches up to rotations within degenerate pairs.

## Log

### 2026-09-07 — backfilled from docs and history

Backfilled 2026-10-18 from `smatrix/README.md` and `docs/plots.md`. Done in the phase 5 triage (no work since July to September). Seven analysis and plot scripts moved here from `smatrix/code/`; the dataset builder `build_full_modes.py`, `zemax_build/` (tooling, the only recovery path for `bend_zemax/`) and `compute_smatrix.py` stay in `smatrix/code/`. Outputs stay in `smatrix/output/`.
