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

### 2026-10-24 — input repointed to fam_tables

The notebook read `aos/output/fam_zernikes_wep_v16_8_0_dviz_v3_5_0_*`, which no longer
exists. It now reads the `fam_tables` `danish_1_2` donut and visit tables through the
catalog, for the same nights (20251020 to 20251231): 1,947,115 donuts and 733 visits (count).
Run on one visit (20251105 seq 27, 178 detectors); no full rerun. The study stays dropped.

### 2026-03-19 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook header and `psf/README.md` (commits `a11c1ff`, `254a120`). Drew one PNG per visit, PSFs at raft positions with FWHM colour and ellipticity whiskers, from an old `aos/output` FAM Zernike table. Marked In Progress, never finished. Dropped in the phase 5 triage; the aos `psf` study does this forward model.
