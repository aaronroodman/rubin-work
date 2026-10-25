---
study: wfs_mimic
topic: aos
status: parked
question: "What corner-to-corner covariance of the measured-intrinsic deviation would the four corner sensors see, mimicked from FAM donuts at the corner radius?"
answer: "A covariance product only (84 by 84 cross-corner, 21 by 21 single-corner); no conclusion of its own."
uses: {}  # old aos/output tree
started: 2026-06-24
---

# `wfs_mimic`

> **Code:** `code/run_wfs_mimic.py` (the `wfs_mimic` rule); the wedge geometry is the library `rubinwork.wfs_mimic` · **Notebook:** `notebooks/wfs_mimic_covariance.ipynb` · **Output:** `output/<P>_<M>/` (the study `output@`)

## Question

What corner-to-corner covariance of the measured-intrinsic deviation would the four corner sensors see, mimicked from FAM donuts at the corner radius?

## Current answer

A covariance product: the 84 by 84 cross-corner and 21 by 21 single-corner covariance of the deviation, sampled over images. `psf_fp_maps` and `closed_loop` use the same wedges through `rubinwork.wfs_mimic`. Triage: parked, or a product.

## Log

### 2026-09-20 — backfilled from docs and history

Moved from `aos/code/cwfs/run_wfs_mimic.py`. Its wedge geometry (`DEFAULT`, `CORNER_OFFSETS`, `wedge_medians`) became `rubinwork.wfs_mimic`, which removes the `psf` and `closed_loop` imports of a study module. Converted 2026-10-23 in phase 5.
