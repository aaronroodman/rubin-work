---
study: zernike_psf_moments
topic: psf
status: done
question: "What FWHM and ellipticity does a Rubin PSF have for given annular Zernike coefficients Z4 to Z22?"
answer: "A finished interactive tool: GalSim draws the optical plus atmospheric PSF from Z4 to Z22 and HSM adaptive moments give FWHM, e1, e2 and centroid."
uses: {}  # no rubin-work products; simulation only
started: 2026-02-24
---

# `zernike_psf_moments`: PSF moments from annular Zernike coefficients

> **Notebook:** `notebooks/zernike_psf_moments.ipynb`

## Question

What FWHM and ellipticity does a Rubin PSF have for given annular Zernike coefficients Z4 to Z22?

## Current answer

A finished interactive tool: GalSim draws the optical plus atmospheric PSF from Z4 to Z22 and HSM adaptive moments give FWHM, e1, e2 and centroid.

## Log

### 2026-03-19 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook header and `psf/README.md` (commits `a11c1ff`, `b7decb9`). The notebook is marked Complete: a widget explores the Zernike coefficient space and reports the HSM moments. Done in the phase 5 triage.
