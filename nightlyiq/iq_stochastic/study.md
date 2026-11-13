---
study: iq_stochastic
topic: nightlyiq
status: active
question: How much of the per-visit FWHM and median ellipticity does AOS_FWHM predict, band by band, and how large is the scatter it leaves?
answer: 'AOS_FWHM predicts little of either. Per band, Pearson r is 0.26 to 0.44 for FWHM and 0.16 to 0.30 for median ellipticity. The Huber fit cuts the nMAD of FWHM from 0.22-0.30 arcsec to 0.20-0.28 arcsec, and leaves the ellipticity nMAD almost unchanged (0.03-0.04, dimensionless).'
uses:
  value_added: v1@20261105
started: 2026-11-13
---

# `iq_stochastic`

## Question

How much of the per-visit FWHM and median ellipticity does AOS_FWHM predict, band by band, and how large is the scatter it leaves?

## Current answer

AOS_FWHM predicts little of either. Per band, Pearson r is 0.26 to 0.44 for FWHM and 0.16 to 0.30 for median ellipticity. The Huber fit cuts the nMAD of FWHM from 0.22-0.30 arcsec to 0.20-0.28 arcsec, and leaves the ellipticity nMAD almost unchanged (0.03-0.04, dimensionless).

## Log

### 2026-11-13 — started

Study created.

### 2026-11-13 — FWHM and median ellipticity vs AOS_FWHM

`code/iq_vs_aos.py`, run `output/20261113_iq_vs_aos/` (`iq_vs_aos.pdf`, `huber_fits.csv`).
Sample: 89,389 LSSTCam science visits over 250 nights, day_obs 20250417 to 20261112, from
ConsDB with AOS_FWHM set and engineering observation reasons excluded. FWHM is
`psf_sigma_median` x 2.355 x 0.2 arcsec/pixel. ConsDB `psf_ellipticity_median` is empty
for every visit, so median ellipticity is value_added `psf_e_azel_p50` (median |e| over
CCDs, needs at least 150 CCDs; 83,425 visits). Per band: 2D histogram plus a Huber RLM line.

| band | FWHM slope (arcsec/arcsec) | FWHM r / rho | FWHM nMAD resid / nMAD y (arcsec) | e slope (1/arcsec) | e r / rho | e nMAD resid / nMAD y (dimensionless) |
|---|---|---|---|---|---|---|
| u | 0.87 | 0.40 / 0.40 | 0.241 / 0.283 | 0.038 | 0.17 / 0.15 | 0.029 / 0.030 |
| g | 0.98 | 0.39 / 0.40 | 0.206 / 0.244 | 0.055 | 0.16 / 0.17 | 0.040 / 0.040 |
| r | 0.71 | 0.26 / 0.22 | 0.276 / 0.295 | 0.079 | 0.29 / 0.27 | 0.030 / 0.032 |
| i | 0.89 | 0.41 / 0.30 | 0.226 / 0.250 | 0.091 | 0.30 / 0.24 | 0.033 / 0.034 |
| z | 1.04 | 0.44 / 0.33 | 0.207 / 0.216 | 0.101 | 0.30 / 0.30 | 0.035 / 0.036 |
| y | 1.11 | 0.44 / 0.34 | 0.204 / 0.218 | 0.121 | 0.29 / 0.28 | 0.037 / 0.038 |

r is Pearson and rho is Spearman, both over the visits in that band. The FWHM slopes are
near 1 arcsec/arcsec, but at a fixed AOS_FWHM the FWHM still spans about 1 arcsec
(`figures/fwhm_vs_aos_fwhm.png`).
