---
study: nightly_aos_tables
topic: nightlyiq
status: dropped
question: "What did the AOS state (v-modes, per-corner Zernikes, DOF) look like night by night?"
answer: "Dropped: superseded by the value-added database."
uses: {}  # none
started: 2026-03-07
---

# `nightly_aos_tables`: nightly AOS tables and plots

> **Notebooks:** `notebooks/nightly_tablemaker.ipynb`, `notebooks/aos_nightly_plots.ipynb`

## Question

What did the AOS state (v-modes, per-corner Zernikes, DOF) look like night by night?

## Current answer

Dropped: superseded by the value-added database.

## Log

### 2026-07-13 — backfilled from docs and history

Backfilled 2026-10-18 from `nightlyiq/README.md` (commit `9de682e`). `nightly_tablemaker` wrote one parquet per night from EFD, ConsDB and Butler; `aos_nightly_plots` plotted FWHM, v-mode and DOF time series from them. Dropped in the phase 5 triage: the value-added database holds this per exposure.
