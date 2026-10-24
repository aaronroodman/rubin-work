---
study: wavefront_outliers
topic: aos
status: parked
question: "Which individual donut fits are catastrophically bad, and what per-donut cut flags them?"
answer: "A robust z-score above 20 (dimensionless) against the per-CCD median flags 1.535e-02 of donuts on danish_1_3_test; which threshold to adopt, and whether it belongs in the fam_tables build, is a product decision."
uses: {}  # old aos/output tree
started: 2026-09-22
---

# `wavefront_outliers`

> **Notebook:** `notebooks/wavefront_outliers.ipynb` · **Write-up:** [`../../rubinwork/products/fam_tables/qa/fam_processing.md`](../../rubinwork/products/fam_tables/qa/fam_processing.md)

## Question

Which individual donut fits are catastrophically bad, and what per-donut cut flags them?

## Current answer

A robust z-score above 20 (dimensionless) against the per-CCD median flags 1.535e-02 of donuts on `danish_1_3_test`. The tail reaches 7704 µm of wavefront on Z5 against an nMAD of 0.32 µm. Parked as a `fam_tables` decision: which threshold, and whether the cut belongs in the build.

## Log

### 2026-09-22 — backfilled from docs and history

Moved from `aos/notebooks/fam_processing/`. Converted 2026-10-23 in phase 5.
