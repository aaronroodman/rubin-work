---
study: astrometry_residual_field
topic: astrometry
status: parked
question: Does the per-visit astrometric residual field (single-frame WCS minus Gaia, before the turbulence correction) carry a coherent atmospheric pattern, and how does it compare with the PSF ellipticity field of the same visit?
answer: ''
uses: {}  # no rubin-work products; visit images and star catalogs from the /repo/main Butler
started: 2026-08-13
---

# `astrometry_residual_field`: per-visit astrometric residuals against PSF moments

> **Notebook:** `notebooks/astrometry_residual_field_v1.ipynb` · **Output:** `output/`

## Question

Does the per-visit astrometric residual field carry a coherent atmospheric pattern, and
how does it compare with the PSF ellipticity field of the same visit? The residual is the
single-frame World Coordinate System (WCS) solution, built on the fixed
`astrometry_camera` distortion model plus a per-visit affine, minus the Gaia reference
position, before the Gaussian-process turbulence correction (`fit_turbulence`).

## Current answer

None yet. The tooling is built: residual vectors (mas) and PSF ellipticity, median-binned
on a per-CCD grid, for one visit and stacked over a night's in-focus visits.

## Log

### 2026-08-14 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook's own description and its commits (`db09ac5` to
`f88b510`); parked in the phase 5 triage. The notebook makes per-visit residual quivers
against PSF ellipticity whiskers, linear residual histograms over ±50 mas, and a
multi-page PDF with a night's stack (each visit demeaned, 16 × 16 bins per CCD).

Two caveats found: `astrometry_camera` covers only part of the 189 science CCDs, and the
rest fall back to nominal geometry, so their residuals keep uncorrected distortion; and a
single detector-centre offset against nominal is absorbed by the per-visit affine, so only
the mean-subtracted distortion means anything.

Next, if resumed:

- 3rd and 4th moments from `rubinwork.hsm_moments`, joined on the same grid;
- the wind test: rotate (dRA, dDec) into alt-az and look for a preferred axis and the
  E/B two-point correlation;
- each visit minus the night's persistent field, correlated with the PSF-moment field.
