---
study: image_quality_trending
topic: blocks
status: active
question: "How does delivered image quality (PSF FWHM, ellipticity, AOS contributions) trend night by night across LSSTCam survey visits?"
answer: ''
uses: {}  # no rubin-work products; ConsDB
started: 2026-05-04
---

# `image_quality_trending`: delivered image quality across the survey

> **Notebook:** `notebooks/image_quality_trending.ipynb` · **Code:** `code/consdb_fetch.py` (chunked, cached ConsDB fetch) · **Output:** `output/`

## Question

How does delivered image quality (PSF FWHM, ellipticity, AOS contributions) trend night by night across LSSTCam survey visits?

## Current answer

None written: recurring monitoring plots (old tree `blocks/output/iq_plots_*`, `iqscore_plots_*`, `aos_cam_fwhm_violin_*`, last 20251026 to 20260730), with no question answered.

## Log

### 2026-07-28 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook change log and `blocks/README.md`. Based on Keith Bechtol's notebook; from 2026-05-04 a chunked ConsDB fetch with per-chunk caching, a combined per-visit IQ score (percentile average of `aos_cam_fwhm`, `psf_fwhm_95_05`, `psf_e_50`, `psf_e_95_05` and the shapelets score; 99 = best), and per-night violins. Last run for day_obs 20251026 to 20260730. Converted in the phase 5 triage as active (Aaron, 2026-10-18; the recommendation was parked, as monitoring with no question). The notebook and `consdb_fetch.py` moved here; the notebook writes to this study's `output/`.
