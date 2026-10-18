---
study: cwfs_sky_foreground
topic: wfs
status: done
question: "How does the sky foreground on the corner wavefront sensors vary with focal-plane radius?"
answer: "Closed with no written result: the notebooks make per-visit PDFs of the sky level against focal-plane radius on the eight corner-sensor half-chips, but no conclusion was recorded."
uses: {}  # no rubin-work products; post_isr_image from the AOS cwfs collection on /repo/main
started: 2026-06-12
---

# `cwfs_sky_foreground`: sky foreground against focal-plane radius on the corner sensors

> **Notebooks:** `notebooks/wfs_corner_postisr_visualize.ipynb` (post-ISR images), `notebooks/wfs_sky_foreground_radius.ipynb` (sky against radius) · **Output:** `output/`

## Question

How does the sky foreground on the eight corner wavefront sensor (CWFS) half-chips vary with focal-plane radius, out to the 1.75 deg (310.5 mm) donut-acceptance radius? Measured on post-instrument-signature-removal (post-ISR) images of high-galactic-latitude visits.

## Current answer

Closed with no written result: the notebooks make per-visit PDFs of the sky level against focal-plane radius on the eight corner-sensor half-chips, but no conclusion was recorded.

## Log

### 2026-06-12 — backfilled from docs and history

Backfilled 2026-10-18 from the two notebook headers and `wfs/README.md`. Built on 2026-06-12: per visit, the eight half-chips with a sky stretch, a donut mask (threshold on the residual after a radial sky model), clipped-mean sky against radius, and a 0.25 mm-binned profile normalised per CCD over 295 to 310 mm. No conclusion was written. Marked done in the phase 5 triage (Aaron, 2026-10-18).

The notebooks moved here from `wfs/notebooks/` and now write to this study's `output/`. Both read the retired `danish_1_0/wep_17_3_0` CWFS collection.
