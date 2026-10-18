---
study: roi_quality
topic: guider
status: active
question: "Was each guider ROI placed on the expected guide star, judged by matching found stars to the guide-star catalog?"
answer: ''
uses: {}  # none; guide-star catalog and guider ROI metadata
started: 2026-02-06
---

# `roi_quality`: placement of the guider regions of interest

> **Notebook:** `notebooks/guider_roi_quality.ipynb` · **Output:** `output/`

## Question

Was each guider ROI placed on the expected guide star, judged by matching found stars to the guide-star catalog?

## Current answer

None yet: the notebook (status In Progress) matches measured star positions to the catalog, after pointing offsets, and separates matched from volunteer stars; no conclusion recorded.

## Log

### 2026-03-24 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook header and commit `3f6c885`. Converted as active in the phase 5 triage (the recommendation was dropped or parked: untouched since 2026-03-24, and the summit_utils tracker may supersede it). The notebook moved here; it still imports `guider/code/guider_utils.py` by `sys.path`, and writes to this study's `output/`.
