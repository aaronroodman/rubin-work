---
study: miw_recovery_constraint
topic: aos
status: active
question: "Is the optical state the MIW build subtracts physically reachable, and what does Range-Bounded Recovery do to the MIW when it enforces the allowed DOF ranges?"
answer: "No: unconstrained 50/34 puts 33 of 50 DOF out of range on at least 5 % of visits. RBR brings the worst range ratio from 60.25 to 1.713 (dimensionless) but grows the MIW by 0.2764 um RMS of wavefront."
uses: {}
started: 2026-09-28
---

# `miw_recovery_constraint`

> **Code:** `code/` (`check_dof_ranges.py`, `compare_rbr_arms.py`, `compare_schemes.py`, `compare_build_dof.py`, `test_rbr_against_prototype.py`) · **Write-up:** [`../docs/studies/miw.md`](../docs/studies/miw.md) sections on reachability and RBR · **Depends on:** unpushed `ts_ofc` and `ts_intrinsic_wavefront` branches

## Question

Is the optical state the MIW build subtracts physically reachable, and what does Range-Bounded Recovery do to the MIW when it enforces the allowed DOF ranges?

## Current answer

- The unconstrained 50/34 recovery puts 33 of 50 DOF out of range on at least 5 % of the in-family visits.
- RBR on one of nine bins brings the worst amplitude-over-range ratio from 60.25 to 1.713 (dimensionless) but grows the MIW by 0.2764 µm RMS of wavefront.
- `compare_schemes.py` compares 22/12, 50/34 without RBR and 50/50 with RBR behind one build.

## Log

### 2026-10-08 — backfilled from docs and history

Split from aos `miw` (triage row `miw_recovery_constraint`). The four scripts moved from `aos/code/miw/`; the MIW build itself is the `miw` product and `ts_intrinsic_wavefront`. Converted 2026-10-23 in phase 5.

### 2026-10-27 — Danish 1.3 test and v1000 MIW builds removed

The `miw` builds this study used, `d13t-A_50_34_i@old-tree-20260928`, `d13v1000-A_50_34_i@old-tree-20261006`,
`d13v1000-A_50_34_i_rbr@old-tree-20261006`, `d13v1000-A_22_12_i@old-tree-20261007` and
`d13v1000-A_50_50_i_rbr@old-tree-20261007`, were deleted on 2026-10-27 in the phase 7
cleanup, and dropped from `uses:`, with their variants dropped from `rubinwork/products/miw/variants.yaml`.
The originals are in `/sdf/data/rubin/user/roodman/LSST/backup/pre_reorg_outputs_20261024/aos_output.tar.zst`,
under `aos/output/miw/`. The results above stand; rerunning the scripts needs a new build.
The 50/50 RBR arm is redefined on the `danish_1_3` FAM tables as `d13-A_50_50_i_rbr` and
`d13-A_50_50_i_rbr_5rot`, same settings, not yet built.
