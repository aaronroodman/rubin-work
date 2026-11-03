---
study: cosmic_diffusion
topic: camera
status: active
question: What diffusion coefficient, sigma_diff over sqrt(delta z) in um per sqrt(um), do long cosmic-ray muon tracks in 20261102 darks give for R21 REB0 (about 10 V back bias) compared with R21 REB1 (nominal 50 V)?
answer: 'First visit only (seq 533): k = 0.0932 +- 0.0017 at 8.8 V (REB0) and 0.0445 +- 0.0007 at 50 V (REB1), dimensionless, both 1.6 to 1.8 times the uniform-field sqrt(2kT/eV).'
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

From one 38 s dark (20261102 seq 533), R21 only. The fit is
sigma^2 = sigma0^2 + k^2 D dz, with D = 100 um and k dimensionless:

| group | BSSVBS (V) | tracks | sections | k | k theory | sigma0 (um) |
|---|---|---|---|---|---|---|
| REB0 (S00, S01, S02) | 8.775 | 12 | 68 | 0.0932 +- 0.0017 | 0.0583 | 0 (fit < 0) |
| REB1 (S10, S11, S12) | 50.0 | 18 | 103 | 0.0445 +- 0.0007 | 0.0244 | 2.41 +- 0.07 |

k theory is sqrt(2kT/(eV)) at T = 173.15 K, Fisher-Levine eq. 3.3. Measured over
theory is 1.60 at 8.8 V and 1.82 at 50 V (dimensionless). The ratio of measured k,
REB0 over REB1, is 2.09; sqrt(50/8.775) = 2.39. At 50 V the width grows from
2.4 um at the gates to 4.9 um at the back. The REB0 widths grow more linearly than
sqrt(dz), so their sigma0 fit goes to zero. Errors are formal Huber errors; with
one visit, R21_S00 carries 8 of the 12 REB0 tracks.

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
