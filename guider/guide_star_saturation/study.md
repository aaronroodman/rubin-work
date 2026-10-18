---
study: guide_star_saturation
topic: guider
status: parked
question: "At what peak level do guide stars saturate, so the moments can flag them?"
answer: "The working saturation level is about 40000 ADU (peak after the blank-stamp column bias). A per-CCD saturation flag is not implemented."
uses: {}  # none; raw guider stamps
started: 2026-09-01
---

# `guide_star_saturation`: guide-star peak levels and saturation

> **Code:** `code/run_guider_saturation.py` (writes `output/guider_saturation_<dayObs>.parquet`; run from this study directory)

## Question

At what peak level do guide stars saturate, so the moments can flag them?

## Current answer

The working saturation level is about 40000 ADU (peak after the blank-stamp column bias). A per-CCD saturation flag is not implemented.

## Log

### 2026-09-01 — backfilled from docs and history

Backfilled 2026-10-18 from `guider/docs/pipeline_runbook.md` and commit `bfa8de4`. Parked in the phase 5 triage: the flag would be a `guider_moments` product change. The script writes one row per (expId, detector) with `peak_top3` and `peak_max` (ADU, after the stamp-0,1 column-median bias), ROI position and star row and column. Run for day_obs 20260705, 20260706 and 20260708 to 20260712 (old tree `guider/output/`). It moved here from `guider/code/`.
