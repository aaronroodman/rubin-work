---
study: m1m3_zgradient_survey
topic: blocks
status: dropped
question: "How does the M1M3 vertical thermal gradient (z_gradient) vary night by night over the science and acq exposures?"
answer: "Dropped: a plot only, superseded by the per-visit telemetry in the value-added database."
uses: {}  # none
started: 2026-07-08
---

# `m1m3_zgradient_survey`: M1M3 vertical thermal gradient over the survey

> **Notebook:** `notebooks/m1m3_zgradient_survey.ipynb`

## Question

How does the M1M3 vertical thermal gradient (z_gradient) vary night by night over the science and acq exposures?

## Current answer

Dropped: a plot only, superseded by the per-visit telemetry in the value-added database.

## Log

### 2026-07-08 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook header and commit `03fcbd5`. Per-night violins of `z_gradient` from the EFD thermocouples, cached to parquet. Dropped in the phase 5 triage: `rubinwork.common.visit_telemetry` and the value-added database carry the M1M3 gradients per visit.
