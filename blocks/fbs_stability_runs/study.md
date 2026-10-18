---
study: fbs_stability_runs
topic: blocks
status: parked
question: "Which visits make up each FBS-driven AOS stability run, and what was the PSF FWHM of each run?"
answer: ''
uses: {}  # reads the visits_summary parquet that image_quality_trending writes
started: 2026-08-12
---

# `fbs_stability_runs`: the FBS-driven AOS stability runs

> **Code:** `code/fbs_stability_runs.py`

## Question

Which visits make up each FBS-driven AOS stability run, and what was the PSF FWHM of each run?

## Current answer

None written: the script lists the runs (one night of one block each) with their mean and median per-visit PSF FWHM, as CSV only.

## Log

### 2026-08-12 — backfilled from docs and history

Backfilled 2026-10-18 from the script docstring and commit `9d4e29e`. Parked in the phase 5 triage: CSVs only, no write-up (old tree `blocks/output/fbs_stability_runs_zenith.csv`). The script reads `output/visits_summary_*.parquet` relative to the working directory; that file is `image_quality_trending`'s output, so this is a study reading another study's output (organization plan section 2, rule 3), not caught by the checker because the path is a glob. Fix with the other section 2 violations.
