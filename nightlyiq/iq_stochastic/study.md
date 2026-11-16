---
study: iq_stochastic
topic: nightlyiq
status: active
question: How much of the per-visit FWHM and median ellipticity does AOS_FWHM predict, band by band, and how large is the scatter it leaves?
answer: 'Seeing, not the AOS, sets most of the per-visit FWHM. The donut blur tracks FWHM with Spearman rho 0.78 to 0.90, per band. In u through z, FWHM is close to the blur and AOS_FWHM added in quadrature: what is left has median -0.015 to +0.014 arcsec and nMAD 0.08 to 0.11 arcsec, against an FWHM nMAD of 0.21 to 0.29 arcsec. At fixed seeing, FWHM and median ellipticity both still rise with AOS_FWHM. Two open issues: FWHM minus blur drifts by about 0.2 arcsec between epochs, and the y band sits about 0.2 arcsec below the other bands.'
uses:
  value_added: v1@snapshots/20261105
started: 2026-11-13
---

# `iq_stochastic`

## Question

How much of the per-visit FWHM and median ellipticity does AOS_FWHM predict, band by band, and how large is the scatter it leaves?

## Current answer

Seeing, not the AOS, sets most of the per-visit FWHM. The donut blur tracks FWHM with Spearman rho 0.78 to 0.90, per band. In u through z, FWHM is close to the blur and AOS_FWHM added in quadrature: what is left has median -0.015 to +0.014 arcsec and nMAD 0.08 to 0.11 arcsec, against an FWHM nMAD of 0.21 to 0.29 arcsec. At fixed seeing, FWHM and median ellipticity both still rise with AOS_FWHM. Two open issues: FWHM minus blur drifts by about 0.2 arcsec between epochs, and the y band sits about 0.2 arcsec below the other bands.

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
Most of the rise is above about 0.45 arcsec AOS_FWHM; see the 2026-11-16 entry for the numbers. The median ellipticity rises
by 0.013 (u band) to 0.047 (z band), while the within-bin nMAD is 0.021 to 0.054 (both
dimensionless). `figures/fwhm_vs_aos_fwhm.png` is now the FWHM page of this run.

### 2026-11-16 — stability visits kept, common rise range

`fbs_driven_aos_stability_test` visits are now kept: all 10,808 are `img_type` science,
taken by the survey scheduler. The other engineering reasons are still excluded. The
ConsDB pull is refetched to a dated cache (`output/cache/consdb_iq_20261116.parquet` and
a `.json` holding the fetch time and query). Sample: 101,167 visits over 268 nights, day_obs
20250417 to 20261114, before cuts. The i-band FWHM sample grows from 24,244 to 35,029 visits.
Ellipticity comes from value_added `snapshots/20261105`, so visits after that build have
none (93,852 visits have it). Run `output/20261116_binned_median/`.

The rise per band is now measured over one range for every band: AOS_FWHM 0.30 to
0.60 arcsec, where each 0.05 arcsec bin has at least 250 visits. It is the median of the
last bin minus the median of the first.

| band | FWHM rise (arcsec) | FWHM bin nMAD (arcsec) | e rise (dimensionless) | e bin nMAD (dimensionless) |
|---|---|---|---|---|
| u | 0.196 | 0.223-0.256 | 0.013 | 0.021-0.032 |
| g | 0.222 | 0.189-0.221 | 0.009 | 0.035-0.043 |
| r | 0.162 | 0.259-0.289 | 0.014 | 0.026-0.035 |
| i | 0.286 | 0.166-0.235 | 0.020 | 0.028-0.044 |
| z | 0.231 | 0.172-0.237 | 0.021 | 0.027-0.043 |
| y | 0.292 | 0.170-0.268 | 0.032 | 0.028-0.046 |

Correction to the 2026-11-13 entry: the median FWHM is not flat below 0.45 arcsec AOS_FWHM
in every band. From the 0.30 bin to the 0.45 bin it rises by 0.03 arcsec (r band) to
0.11 arcsec (g band). From the 0.45 bin to the 0.60 bin it rises by 0.17 arcsec (u band) to
0.32 arcsec (z band). The y band has no 0.60 bin, and rises 0.21 arcsec from 0.45 to 0.55.

### 2026-11-16 — donut blur as the per-visit seeing

`donut_blur_fwhm` (ConsDB) is the atmospheric blur fitted to the corner-sensor donuts of
the same exposure. It is set for 92% of visits. Run `output/20261116_aos_and_blur/`
(`iq_stochastic.pdf`, `aos_stats.csv`, `blur_stats.csv`, `binned_medians.csv`) adds pages A
to E. The sample has AOS_FWHM, FWHM and blur (0.1 to 3.0 arcsec) all inside their cuts.

| band | n (visits) | rho (blur, FWHM) | rho (AOS_FWHM, blur) | rho (AOS_FWHM, FWHM² - blur²) | FWHM - blur median / nMAD (arcsec) | leftover median / nMAD (arcsec) | FWHM nMAD (arcsec) |
|---|---|---|---|---|---|---|---|
| u | 4044 | 0.88 | 0.24 | 0.17 | 0.070 / 0.086 | -0.015 / 0.091 | 0.272 |
| g | 8073 | 0.83 | 0.18 | 0.42 | 0.098 / 0.126 | +0.011 / 0.109 | 0.231 |
| r | 15999 | 0.90 | 0.04 | 0.39 | 0.095 / 0.105 | +0.014 / 0.102 | 0.294 |
| i | 33551 | 0.86 | 0.07 | 0.47 | 0.100 / 0.097 | +0.008 / 0.084 | 0.215 |
| z | 19286 | 0.81 | 0.08 | 0.49 | 0.098 / 0.124 | +0.004 / 0.108 | 0.215 |
| y | 12307 | 0.78 | 0.06 | 0.46 | -0.073 / 0.139 | -0.164 / 0.117 | 0.214 |

rho is Spearman. Leftover is FWHM - sqrt(blur² + AOS_FWHM²).

- A: FWHM against blur lies about 0.1 arcsec above y = x in u through z.
- B: blur hardly depends on AOS_FWHM, so bad seeing does not inflate AOS_FWHM.
- C: in g, r, i and z the binned medians of FWHM² - blur² follow y = AOS_FWHM²
  (`figures/excess_vs_aos_fwhm.png`). C was first drawn as a signed root,
  ±sqrt|FWHM² - blur²|. That root's infinite slope at zero empties the bins near zero and
  splits the plot into two false branches, so it was replaced by the square.
- E: with blur fixed to one tercile, FWHM still rises with AOS_FWHM by about 0.2 arcsec over
  0.30 to 0.55 arcsec (i and z bands). The median ellipticity is highest in the low-blur
  tercile: seeing dilutes the optical ellipticity. In u band only a few bins per tercile
  reach 250 visits.

Open issues, not yet run down:

- **Epoch drift.** The median of FWHM - blur changes with time in every band. It is about
  0.15 to 0.30 arcsec from 202508 to 202603, and 0.00 to 0.07 arcsec from 202605 on (monthly
  medians). This looks like a change in how `donut_blur_fwhm` or the PSF FWHM is computed,
  not in the sky. The pooled leftover nMAD therefore includes an epoch spread.
- **y band.** FWHM - blur is about 0.2 arcsec lower in y than in the other bands, in every
  month. The donut blur may be overestimated in y.
- The guider PSF FWHM, a second seeing measure, waits for more processed guider stamps.
