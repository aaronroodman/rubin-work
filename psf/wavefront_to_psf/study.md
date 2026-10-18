---
study: wavefront_to_psf
topic: psf
status: dropped
question: "What does the focal-plane PSF look like when drawn from the median per-detector Zernikes of each visit?"
answer: "Dropped: superseded by the aos psf study, which draws the forward PSF from the wavefront."
uses: {}  # read an old aos/output FAM Zernike parquet, not a product
started: 2026-03-19
---

# `wavefront_to_psf`: focal-plane PSF mosaics from per-donut Zernikes

> **Notebook:** `notebooks/wavefront_to_psf.ipynb`

## Question

What does the focal-plane PSF look like when drawn from the median per-detector Zernikes of each visit?

## Current answer

Dropped: superseded by the aos psf study, which draws the forward PSF from the wavefront.

## Log

### 2026-03-19 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook header and `psf/README.md` (commits `a11c1ff`, `254a120`). Drew one PNG per visit, PSFs at raft positions with FWHM colour and ellipticity whiskers, from an old `aos/output` FAM Zernike table. Marked In Progress, never finished. Dropped in the phase 5 triage; the aos `psf` study does this forward model.
