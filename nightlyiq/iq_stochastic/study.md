---
study: iq_stochastic
topic: nightlyiq
status: active
question: How much of the per-visit FWHM and median ellipticity does AOS_FWHM predict, band by band, and how large is the scatter it leaves?
answer: 'AOS_FWHM sets a trend but not the visit. Across AOS_FWHM 0.25 to 0.75 arcsec the per-bin median FWHM rises by 0.23 to 0.38 arcsec, depending on band, while the nMAD of FWHM within one bin is 0.16 to 0.30 arcsec. For median ellipticity the per-bin median rises by 0.013 to 0.047 against a within-bin nMAD of 0.021 to 0.054 (both dimensionless).'
uses:
  value_added: v1@20261105
started: 2026-11-13
---

# `iq_stochastic`

## Question

How much of the per-visit FWHM and median ellipticity does AOS_FWHM predict, band by band, and how large is the scatter it leaves?

## Current answer

AOS_FWHM sets a trend but not the visit. Across AOS_FWHM 0.25 to 0.75 arcsec the per-bin median FWHM rises by 0.23 to 0.38 arcsec, depending on band, while the nMAD of FWHM within one bin is 0.16 to 0.30 arcsec. For median ellipticity the per-bin median rises by 0.013 to 0.047 against a within-bin nMAD of 0.021 to 0.054 (both dimensionless).

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

### 2026-11-13 — binned medians replace the Huber lines

The Huber lines missed most of the behavior: the residuals were nearly as wide as y
itself. Run `output/20261113_binned_median/` (`iq_vs_aos.pdf`, `band_stats.csv`,
`binned_medians.csv`) drops them and the y = x line. Each panel now shows the median of y
in 0.05 arcsec AOS_FWHM bins, with an nMAD error bar, for bins of at least 250 visits.
AOS_FWHM is cut to 0 to 1.0 arcsec.

Across the populated bins (about 0.25 to 0.70 arcsec AOS_FWHM), the median FWHM rises by
0.23 arcsec (u band) to 0.38 arcsec (i band). The within-bin FWHM nMAD is 0.16 to 0.30 arcsec.
Below about 0.45 arcsec AOS_FWHM the median FWHM is nearly flat. The median ellipticity rises
by 0.013 (u band) to 0.047 (z band), while the within-bin nMAD is 0.021 to 0.054 (both
dimensionless). `figures/fwhm_vs_aos_fwhm.png` is now the FWHM page of this run.
