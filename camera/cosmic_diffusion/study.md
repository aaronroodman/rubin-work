---
study: cosmic_diffusion
topic: camera
status: active
question: What diffusion coefficient, sigma_diff over sqrt(delta z) in um per sqrt(um), do long cosmic-ray muon tracks in 20261102 darks give for R21 REB0 (about 10 V back bias) compared with R21 REB1 (nominal 50 V)?
answer: 'Full night 20261102 (406 darks), sigma0 prior 2.89 +- 0.5 um: median per-track k = 0.0692 and 0.0686 on R21_S01 and S02 at 8.8 V, 0.0415 to 0.0420 on REB1 at 50 V (dimensionless); R21_S00 is the outlier at 0.0885. All exceed the uniform-field sqrt(2kT/eV), by 1.2x at 8.8 V and 1.7x at 50 V.'
uses: {}
started: 2026-11-03
---

# `cosmic_diffusion`: Charge diffusion from cosmic-ray muon tracks in dark visits

## Question

What diffusion coefficient, sigma_diff over sqrt(delta z) in um per sqrt(um), do long cosmic-ray muon tracks in 20261102 darks give for R21 REB0 (about 10 V back bias) compared with R21 REB1 (nominal 50 V)?

> **Code:** `code/find_cosmics.py` (ISR, track finding, per-visit FITS),
> `code/analyze_diffusion.py` (widths, fits, PDF), `code/cosmic_tools.py` ·
> **Method:** Fisher-Levine & Nomerotski 2015, JINST 10 C08006 (`notes/`)

## Current answer

All 406 darks of 20261102, R21 only, 7,942 ok tracks. Per-track Huber fit
sigma^2 = sigma0^2 + k^2 D dz, D = 100 um, with sigma0 under a Gaussian prior
2.89 +- 0.5 um (10 um / sqrt(12)); section widths from a pixel-integrated Gaussian.
Table gives medians over tracks; error on k 1.2533 nMAD / sqrt(n) is <= 0.0002
(dimensionless) in every row. k is dimensionless.

| CCD | BSSVBS (V) | ok tracks | k, sigma0 prior | k, sigma0 free | k, sigma0 fixed | k theory | k prior / k theory | sigma0 prior fit (um) |
|---|---|---|---|---|---|---|---|---|
| R21_S00 | 8.825 | 1339 | 0.0885 | 0.0973 | 0.0865 | 0.0582 | 1.52 | 2.14 |
| R21_S01 | 8.825 | 1333 | 0.0692 | 0.0705 | 0.0659 | 0.0582 | 1.19 | 2.34 |
| R21_S02 | 8.825 | 1374 | 0.0686 | 0.0700 | 0.0654 | 0.0582 | 1.18 | 2.34 |
| R21_S10 | 50.0 | 1313 | 0.0419 | 0.0420 | 0.0411 | 0.0244 | 1.72 | 2.82 |
| R21_S11 | 49.975 | 1279 | 0.0420 | 0.0421 | 0.0410 | 0.0244 | 1.72 | 2.79 |
| R21_S12 | 49.975 | 1304 | 0.0415 | 0.0417 | 0.0407 | 0.0244 | 1.70 | 2.80 |

k theory is sqrt(2kT/(eV)) at T = 173.15 K (Fisher-Levine eq. 3.3). Figure:
`figures/sigma_vs_dz_per_ccd.png`.

- On REB1 the free fit gives sigma0 = 2.78-2.81 um, at the 2.89 um pixel term, so
  the prior barely matters there. At 8.8 V the free sigma0 is low (2.07 um on
  S01/S02) and the constraint lowers k by 2% (prior) to 7% (fixed).
- S01 and S02 agree; S00 is 1.28 times higher (dimensionless, S00 over the S01/S02
  mean, prior fit). Its width grows faster than sqrt(dz) beyond 60 um and its free
  sigma0 is negative on 88% of tracks. Its `CCDTEMP` reads -86.7 C against about
  -101 C elsewhere.
- S01/S02 over REB1 is 1.65 (dimensionless, prior fit), against sqrt(50/8.825) =
  2.38 for k proportional to 1/sqrt(V): part of the width does not scale with back
  bias.

## Log

### 2026-11-03 — started; first visit, seq 533

Study created; finder and analysis scripts written and run on seq 533 (six R21
sensors, 47 tracks pass the finder). Output: `output/20261103_find_20261102/visits/`
(per-visit FITS and parquet; since archived) and `output/20261103_diffusion/`
(archived).

Decisions:

- ISR: overscan, bias, defects and PTC gains only (images in electrons); no dark,
  flat, crosstalk, CTI, brighter-fatter or linearity.
- Back bias from the raw header `BSSVBS`: 8.775 V on REB0, 50.0 V on REB1.
  `CCDTEMP` reads -86.7 C on R21_S00 and -101 to -102 C elsewhere; Aaron says S00's
  sensor has an offset, so kT/e uses a fixed -100 C.
- Finder cuts: length >= 10 px, weighted rms across the track <= 0.8 px, peak
  < 50,000 ADU, no hot column (per-amp column median > 5 nMAD), not a 1- or 2-column
  vertical source. The analysis keeps tracks >= 15 px.
- Delta-ray cut: more than 5% of the above-threshold charge beyond 2.5 px from the
  line. Counting all pixels in that cut, noise included, rejected nearly every track.
- Section fit errors include a 10% model term; with Poisson errors alone the
  bright sections fail chi2/dof < 5.
- Track direction: the wide end is the back surface (longest drift); tracks whose
  width slope is smaller than its error are dropped as ambiguous.
- All six R21 sensors are e2v, so REB0 and REB1 compare the same CCD type.
- dE/dx validation: Landau-Gauss MPV 69.8 +- 5.0 electrons per um of path, n = 47,
  close to the roughly 75 e/um expected for 100 um of silicon.

**Next:** run all 406 darks of 20261102; check whether the excess over uniform-field
theory comes from the front-side clock voltages adding to the field.

### 2026-11-03 — full night, revised cuts

Commits `ce6bdb5` (half-charge ends, centre straightness, width-scaled delta-ray
cut, validation pages), `37e9305` (gallery). Finder rerun on all 406 darks as Slurm
array 42969223; output `output/20261103_find_20261102_v2/visits/`. Analysis
run `20261103_diffusion`, since superseded and archived.

Decisions:

- Finder perp_rms cut loosened from 0.8 to 2.0 px: perp_rms includes the diffusion
  width, and at 0.8 px it rejected 17% of long REB0 tracks against 13% on REB1.
  Straightness is now the rms of 2 px section centres about the line, <= 0.25 px.
- Track ends at half the median charge per unit length, not at the last 3-sigma
  pixel; ends must lie >= 3 px from a detector edge or defect.
- Delta-ray cut: charge beyond 1 px + 3 x the track's 90th-percentile width,
  <= 2% of the track charge. The fixed 2.5 px version rejected 0 of 7,357 tracks.
- Bootstrap dropped for the median over per-track fits; it took most of a 31 min
  run. Measurements are cached, so `--reuse <run>` reruns cuts and plots alone.
- The first full-night finder run (`20261103_find_20261102_all`) had 5 of 15 array
  tasks fail writing the shared run.json; only task 0 writes it now.
- Status counts, REB0 / REB1: ok 4294 / 4169, short 2369 / 2033, crooked
  929 / 887, few_sections 484 / 547, near_edge 108 / 76, delta_ray 30 / 72,
  ambiguous 16 / 22.
- The k values moved by at most 0.0007 from the partial-night run with the old
  cuts.
- The depth-mapping check (charge per px against dz) is flat, but it is weak: the
  half-charge ends make the interior sections full by construction.

**Next:** explain the negative-sigma0 population in the sigma0 against k scatter;
check S00 (field profile, real temperature); model the part of the width that
does not scale with back bias.

### 2026-11-03 — pixel-integrated widths, sigma0 prior, cleanup

Commit `9107d93`. Remeasured all tracks; output
`output/20261103_diffusion_pixint/cosmic_diffusion.pdf`. Results as in Current
answer.

- Section profile is now a Gaussian averaged over each pixel's area (4 x 4 points),
  with per-pixel errors from the model, refit 3 times. Reported sigma adds the
  pixel term 1/12 px^2 back in quadrature. On synthetic tracks of 0.2-0.7 px width
  it recovers the total width within 0.005 px, chi2/dof 0.9-1.1. The old fit
  sampled the Gaussian at pixel centres and took errors from the data.
- Per-track and pooled fits run three ways: sigma0 free, fixed at 2.89 um, and with
  a Gaussian prior 2.89 +- 0.5 um (`--sig0-prior`). The prior is a fixed-weight
  row in a Huber IRLS fit, so the robust weights cannot discount it.
- Negative free sigma0^2 per track: S00 88%, S01/S02 5% (was 13-14%), REB1 0%
  (was 2-3%). The negative-sigma0 population was mostly the old fit.
- chi2/dof cut kept at 5 with the 10% model term. Failure rate on ok-track
  sections rises with amplitude: 2% below 300 e-, 12% at 300-1000 e-, 29% at
  1000-3000 e-, 46% above. Taken as Landau fluctuations with knock-on electrons
  (Aaron); the cut may be too tight. Median chi2/dof without the model term 8.0.
- New pages: amplitude against sigma heat maps per CCD; chi2/dof against amplitude.
- Old and test runs moved to `output/_archive_delete_me/` for deletion by hand.

**Next:** decide the chi2/dof cut; check S00 (field profile, real temperature);
model the width that does not scale with back bias.

### 2026-11-04 — night 20261103, REB0 at 11.3 V

104 darks of 20261103 (seq 1-104), REB0 `BSSVBS` 11.325 V, REB1 50.0 V. Finder run
`output/20261104_find_20261103/`, analysis `output/20261104_diffusion_20261103/cosmic_diffusion.pdf`,
same code and cuts as the 20261102 pass. 1,584 ok tracks (REB0 806, REB1 778). k is
dimensionless; sigma0 prior 2.89 +- 0.5 um; errors on k <= 0.0003.

| CCD | k prior, 11.3 V night | k prior, 8.8 V night | k theory at 11.3 V | sigma0 prior fit (um) |
|---|---|---|---|---|
| R21_S00 | 0.0889 | 0.0885 | 0.0513 | 2.15 |
| R21_S01 | 0.0664 | 0.0692 | 0.0513 | 2.39 |
| R21_S02 | 0.0657 | 0.0686 | 0.0513 | 2.44 |
| R21_S10 | 0.0421 | 0.0419 | 0.0244 | 2.81 |
| R21_S11 | 0.0418 | 0.0420 | 0.0244 | 2.83 |
| R21_S12 | 0.0416 | 0.0415 | 0.0244 | 2.81 |

- REB1 repeats the 20261102 night within 0.0002, so night-to-night drift is small.
- S01/S02 k drops 4.2% from 8.8 V to 11.3 V (k at 8.8 V over k at 11.3 V = 1.043,
  dimensionless), against 1.133 for k proportional to 1/sqrt(V).
- S00 does not change with back bias (0.0885 to 0.0889); its free sigma0 is again
  negative on 89% of tracks.
- `code/batch_find.sh` now takes `DAY_OBS` and `RUN_DATE` from the environment.

**Next:** same as above; S00 not responding to the back bias points at a different
field or a defect on that sensor.
