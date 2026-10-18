---
study: guider_iq
topic: nightlyiq
status: parked
question: "How much of the science-image PSF width do the guider seeing and the pointing jitter explain, image by image?"
answer: ''
uses: {}  # no rubin-work products; ConsDB visit1_quicklook
started: 2026-07-13
---

# `guider_iq`: per-image PSF quality against guider metrics

> **Notebook:** `notebooks/guider_iq.ipynb`

## Question

How much of the science-image PSF width do the guider seeing and the pointing jitter explain, image by image?

## Current answer

None written. The notebook produced per-band PDFs for day_obs 20260420 to 20260712 (`nightlyiq/output/guider_iq_*byband_20260420_20260712.pdf`, old tree) with no written conclusion.

## Log

### 2026-07-14 — backfilled from docs and history

Backfilled 2026-10-18 from `nightlyiq/README.md` and the commits `9de682e` to `78a59c8`. Per band: robust Theil-Sen fits of donut blur (or FWHM) against `guider_total_seeing` and the detrended pointing RMS (arcsec); quadrature residuals FWHM² − equivalent² (arcsec²) against `aos_fwhm²` (arcsec²) and against the corner Z4 sum of squares (µm² of wavefront). PDFs only, no conclusion. Parked in the phase 5 triage; it overlaps the guider topic, but stays here (Aaron, 2026-10-18).
