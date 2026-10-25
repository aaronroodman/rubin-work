---
study: cwfs
topic: aos
status: parked
question: "Does the optical state recovered from the corner wavefront sensors agree with the FAM full-focal-plane measurement, Zernike by Zernike?"
answer: "Not fully: per-Zernike CWFS-FAM offsets in Z11, Z14 and Z7 persist under a radius-matched comparison. The Z11/Z14 intra/extra split is unexplained and is now part of aos/intra_extra."
uses: {}  # old aos/output tree
started: 2026-09-06
---

# `cwfs`

> **Code:** `code/` (corner and DOF comparisons, the parked refit ensemble) · **Notebooks:** `notebooks/` · **Write-up:** [`cwfs.md`](cwfs.md) · **Output:** `output/` (the study `output@`: `wfs_dof_compare/`, dated `wfs_corner_compare` runs); tables in the `wfs_corner_compare` and `cwfs_tables` products

## Question

Does the optical state recovered from the corner wavefront sensors agree with the FAM full-focal-plane measurement, Zernike by Zernike?

## Current answer

- Per-Zernike CWFS–FAM offsets (Z11 spherical, Z14 tetrafoil, Z7 coma) persist under a radius-matched comparison: a real disagreement, not a field-mean artifact.
- The Z11/Z14 intra- against extra-focal split is unexplained; it belongs to [`../intra_extra`](../intra_extra/study.md).
- The refit-ensemble pair is parked pending Danish 1.2 FAM reprocessing.

## Log

### 2026-10-08 — backfilled from docs and history

Moved from `aos/code/cwfs/` (less `run_wfs_mimic.py`, now its own study, and the `run_wfs_mktable.py` shim, which stays) and `aos/notebooks/cwfs/`. `robust_fit` moved to `rubinwork.common.utils`. Parked in the triage. Converted 2026-10-23 in phase 5.
