---
study: cosmic_diffusion
topic: camera
status: active
question: What diffusion coefficient, sigma_diff over sqrt(delta z) in um per sqrt(um), do long cosmic-ray muon tracks in 20261102 darks give for R21 REB0 (about 10 V back bias) compared with R21 REB1 (nominal 50 V)?
answer: 'Full night 20261102 (406 darks): median per-track k = 0.0708 and 0.0706 on R21_S01 and S02 at 8.8 V, 0.0437 to 0.0441 on REB1 at 50 V (dimensionless); R21_S00 is the outlier at 0.0968 with a near-linear width growth. All exceed the uniform-field sqrt(2kT/eV), by 1.2x at 8.8 V and 1.8x at 50 V.'
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

All 406 darks of 20261102, R21 only, 8,463 ok tracks. Per-track Huber fit
sigma^2 = sigma0^2 + k^2 D dz, D = 100 um; table gives the median over tracks with
error 1.2533 nMAD / sqrt(n). k is dimensionless.

| CCD | BSSVBS (V) | ok tracks | median k | k theory | k / k theory | median sigma0 (um) |
|---|---|---|---|---|---|---|
| R21_S00 | 8.825 | 1405 | 0.0968 +- 0.0002 | 0.0582 | 1.66 | -2.24 +- 0.03 |
| R21_S01 | 8.825 | 1429 | 0.0708 +- 0.0001 | 0.0582 | 1.22 | 1.73 +- 0.02 |
| R21_S02 | 8.825 | 1460 | 0.0706 +- 0.0001 | 0.0582 | 1.21 | 1.70 +- 0.02 |
| R21_S10 | 50.0 | 1410 | 0.0439 +- 0.0001 | 0.0244 | 1.80 | 2.48 +- 0.01 |
| R21_S11 | 49.975 | 1371 | 0.0441 +- 0.0001 | 0.0244 | 1.81 | 2.45 +- 0.01 |
| R21_S12 | 50.0 | 1388 | 0.0437 +- 0.0001 | 0.0244 | 1.79 | 2.45 +- 0.01 |

k theory is sqrt(2kT/(eV)) at T = 173.15 K (Fisher-Levine eq. 3.3); k / k theory is
dimensionless. Figure: `figures/sigma_vs_dz_per_ccd.png`.

- S01 and S02 agree; S00 is 1.37 times higher (dimensionless, S00 over the S01/S02
  mean) and its width grows near-linearly with dz, so its sigma0 fit is negative.
  It is also the CCD whose `CCDTEMP` reads -86.7 C against about -101 C elsewhere.
- REB1's three CCDs agree to 0.0004 in k.
- S01/S02 over REB1 is 1.61 (dimensionless), against sqrt(50/8.825) = 2.38 for
  k proportional to 1/sqrt(V): part of the width does not scale with back bias.
- The per-track sigma0 against k scatter shows a second population at negative
  sigma0 in both REBs, not yet explained.

## Log

### 2026-11-03 — started; first visit, seq 533

Study created; finder and analysis scripts written and run on seq 533 (six R21
sensors, 47 tracks pass the finder). Output: `output/20261103_find_20261102/visits/`
(per-visit FITS and parquet) and `output/20261103_diffusion/cosmic_diffusion.pdf`.

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
array 42969223; output `output/20261103_find_20261102_v2/visits/`. Analysis:
`output/20261103_diffusion/cosmic_diffusion.pdf`, results as in Current answer.

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
