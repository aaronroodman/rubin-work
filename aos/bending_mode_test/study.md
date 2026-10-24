---
study: bending_mode_test
topic: aos
status: parked
question: "In the bending-mode test blocks T377-T380, which hexapod LUT and Trim values were commanded at each corner-sensor exposure?"
answer: ""
uses: {}  # value_added through the reader default
started: 2026-09-18
---

# `bending_mode_test`

> **Notebook:** `notebooks/bending_mode_test_lut_trim.ipynb` (reads the value-added database) · **Write-up:** the bending-mode section of [`../bounce/bounce.md`](../bounce/bounce.md)

## Question

In the bending-mode test blocks T377-T380, which hexapod LUT and Trim values were commanded at each corner-sensor exposure?

## Current answer

None recorded: the notebook plots the commanded hexapod LUT (10 axes) and Trim (all 50 DOF) against `seq_num`; no conclusion.

## Log

### 2026-09-18 — backfilled from docs and history

Moved from `aos/notebooks/bounce/`. Separate from `bounce` (triage). It imports `rubinwork.products.value_added.reader` in place of the `efd_db` shim. Converted 2026-10-23 in phase 5.
