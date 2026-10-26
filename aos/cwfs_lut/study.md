---
study: cwfs_lut
topic: aos
status: active
question: "How does the open-loop optical state from the corner sensors depend on elevation and rotator angle over the science survey, and does it agree with the bounce tests?"
answer: "One term dominates: M2 hexapod dx against rotator angle, +20.39 um/deg (Pearson r = +0.718, Spearman rho = +0.759, n = 96,278 visits). Agreement with the bounce tests is cosine +0.829 (dimensionless) on the six hexapod translations."
uses:  # bounce_compare.py defaults to the _july bounce run (--bounce-run)
  value_added: v1
  bounce_tables: d12-A_50_34_i_5rot_july@old-tree-20261001
started: 2026-10-06
---

# `cwfs_lut`

> **Code:** `code/` (`cwfs_lut_lib.py`, `run_cwfs_lut.py`, `bounce_compare.py`, figures, tests) · **Write-up:** [`cwfs_lut.md`](cwfs_lut.md) · **Output:** `output/` (the study `output@`)

## Question

How does the open-loop optical state from the corner sensors depend on elevation and rotator angle over the science survey, and does it agree with the bounce tests?

## Current answer

Over 96,278 visits and 214 nights: M2 hexapod dx against rotator angle, +20.39 µm/deg of rotator (Pearson r = +0.718, Spearman rho = +0.759, both dimensionless) is the only rigid-body trend with a correlation above 0.6. Against the bounce tests, cosine similarity +0.829 (dimensionless) on the six hexapod translations.

## Log

### 2026-10-12 — backfilled from docs and history

Moved from `aos/code/cwfs_lut/` and the study doc. It reads the `bounce_tables` product through `bounce_compare._bounce_stats_path`. Converted 2026-10-23 in phase 5.
