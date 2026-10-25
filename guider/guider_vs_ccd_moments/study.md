---
study: guider_vs_ccd_moments
topic: guider
status: active
question: "Do guide-star second moments, measured with the same galsim-HSM estimator, match the PSF second moments on the adjacent science CCD?"
answer: ''
uses: {}  # guider_moments v4 nights (incremental, no single build to pin); guider_psfmoments in this study's output/night_<d>/
started: 2026-08-20
---

# `guider_vs_ccd_moments`: guide-star moments against the adjacent science CCD

> **Definition:** [`guider_vs_ccd_moments.md`](guider_vs_ccd_moments.md) · **Notebook:** section "Guider vs adjacent-CCD PSF" of `../guider_ellipticity_ensemble/notebooks/guider_moments_ensemble.ipynb` · **Code:** `guider/code/extract_adjacent_psf_moments.py` (Snakefile rule `psfmoments`)

## Question

Do guide-star second moments, measured with the same galsim-HSM estimator, match the PSF second moments on the adjacent science CCD?

## Current answer

No headline number yet. With matched HSM on both sides, the trace T compares 1:1; Q1 and Q2 need each side rotated to OCS first, and that panel does not exist yet. Definitions in [`guider_vs_ccd_moments.md`](guider_vs_ccd_moments.md).

## Log

### 2026-10-11 — backfilled from docs and history

Backfilled 2026-10-18 from [`guider_vs_ccd_moments.md`](guider_vs_ccd_moments.md) (formerly `guider/docs/`), and converted as active in the phase 5 triage. Both sides use `rubinwork.hsm_moments`; the doc's reference to `optatmo/code/moments_hsm.py` is corrected to that. The adjacent-CCD side reads `single_visit_star` and cuts stamps from `preliminary_visit_image` on `/repo/main`, not ConsDB.

Next:

- split the guider-vs-CCD section out of the ensemble notebook into this study's `notebooks/`;
- add the OCS-rotated Q1, Q2 panel (rotator angle plus the cameraGeom detector yaw);
- `guider_psfmoments` parquet is read by the ensemble notebook too, so it may become a small product (triage).
