---
study: guider_atmo_sim
topic: guider
status: active
question: "Does a phase-screen atmosphere simulation reproduce the guide-star size and centroid motion measured on the guiders?"
answer: "Partly: over 6 visits of 20260706 the guider centroid-motion size matches on average (median T_motion sim over data 1.31, dimensionless) but per visit ranges 0.33 to 2.41; the motion direction does not match (median angle difference 48 deg over 48 sensor-visits, against 45 deg for no relation), with either ERA5 or weather-station ground wind."
uses:
  guider_moments: v4@nights/20260706
  atmo_sim: v2@visits/20260706{00195,00372,00603,00688,00726,00908}
  atmo_sim_moments: v2@visits/20260706{00195,00372,00603,00688,00726,00908}
started: 2026-08-06
---

# `guider_atmo_sim`: simulated atmospheric PSF on the guide sensors

> **Code:** `code/guider_atmo_*.py`, `code/fetch_cp_weather.py` · **Notebook:** `notebooks/galsim_guider_movie.ipynb` · **Output:** `output/` (the scripts default to a relative `output/`; run from this directory)

## Question

Does a GalSim phase-screen atmosphere (imSim Ellerbroek profile, or psf-weather-station winds), drawn as 50 ms stamps at 5 Hz on the 8 guide sensors, reproduce the guide-star size and centroid motion measured by the guider moments?

## Current answer

Partly, over 6 visits of day_obs 20260706 (seq 195, 372, 603, 688, 726, 908), with the
real geometry, ERA5 winds aloft, the site weather-station ground wind (`atmo_sim` v2) and
a seeing match to the guider FWHM:

- **Motion size matches on average, not per visit.** Median over 48 sensor-visits of
  `T_motion` sim over data is 1.31 (dimensionless); per visit 0.33 to 2.41. The per-visit
  ratio rises with the ground-layer weight psfws draws at random (0.30 to 0.93, fraction
  of Cn²): Pearson r 0.54, Spearman rho 0.37, n 6 visits.
- **Motion direction does not match.** Median |PA sim − PA data| per visit is 35 to 59
  deg, 48 deg over all 48 sensor-visits; 45 deg is what no relation gives. Neither the
  station ground wind nor the corrected sky rotation of the winds changed this (v1: 49
  deg). Motion ellipticity |Q|/T: median 0.11 to 0.26 (sim) against 0.23 to 0.43 (data),
  dimensionless.
- **Static size** `T_stamp`: median sim over data 0.91 (dimensionless).
- **Science CCDs**: median FWHM 0.895 arcsec (sim, atmosphere only) against 0.880 arcsec
  (ConsDB); median ellipticity e 0.018 (sim) against 0.060 (ConsDB), dimensionless.
- **Astrometric shift** (sim only, one star per science CCD, 30 s): rms over CCDs 5.8 to
  28.3 mas per visit; 5.6 to 20.3 mas after a 6-parameter affine fit across the focal
  plane is removed.

## Method

[`method.md`](method.md): what the simulation does (turbulence profile, seeing match,
phase screens, rendering, guide and science-CCD stars, geometry, what is left out).

## Log

### 2026-11-20 — `atmo_sim` v2: station ground wind, wind sky rotation, astrometric shifts

`atmo_sim` v2 and `atmo_sim_moments` v2 for the same 6 visits (Slurm array 46210471,
45 to 62 min per visit on roma). Changes against v1:

- the ground-layer wind is the site weather station's (header `WINDSPD`, `WINDDIR`):
  4.8 to 7.8 m/s from 5 to 32 deg, against ERA5's 3.6 m/s from 319 deg;
- the local winds are rotated to the sky at latitude −30.2446 deg. v1 used psfws's
  default, +30.2446 deg, which turned every sky wind direction by 14 to 29 deg;
- the ERA5 step stays the nearest one, and the ground-layer weight stays psfws's draw
  (same seeds, same weights as v1);
- per-layer winds (local speed and direction from, sky speed and direction toward) are
  in `atmosphere.json`, the simulator log and the PDF summary page;
- the astrometric shift of each science-CCD star (unweighted centroid less true
  position) has its own PDF page.

| seq | T_motion sim/data | e_motion sim | e_motion data | median \|ΔPA\| (deg) | CCD FWHM sim (arcsec) | shift rms (mas) | shift rms, affine removed (mas) |
|---|---|---|---|---|---|---|---|
| 195 | 0.43 | 0.18 | 0.26 | 59 | 0.85 | 6.8 | 6.3 |
| 372 | 1.14 | 0.16 | 0.23 | 35 | 1.15 | 18.7 | 11.3 |
| 603 | 1.72 | 0.13 | 0.43 | 39 | 0.92 | 28.0 | 20.0 |
| 688 | 2.41 | 0.26 | 0.28 | 54 | 0.71 | 25.3 | 6.5 |
| 726 | 0.33 | 0.11 | 0.27 | 42 | 0.88 | 5.8 | 5.6 |
| 908 | 1.65 | 0.22 | 0.35 | 55 | 1.37 | 28.3 | 20.3 |

Ratios and ellipticities are dimensionless medians over the 8 guiders. Over all 48
sensor-visits: T_motion sim/data 1.31 (v1 1.28), median |ΔPA| 48 deg (v1 49 deg). The
direction mismatch is not from the ground wind or the wind rotation. Run:
`output/20261120_v2_visits_20260706_six/` (`v2_visits_20260706_six.pdf`).

### 2026-10-28 — six visits; guider motion ellipticity

Simulated seq 195, 372, 603, 726 and 908 (spread through the night, all r band, 30 s, all
8 guiders), about 11 min each (`code/run_visits.sh`). Added the guider motion ellipticity
|Q|/T and its position angle to `compare_visit.py`: DVCS +x lies along FOCAL_PLANE +x for
all 8 guiders, so the angle is in the FOCAL_PLANE frame. Run
`output/20261028_visits_20260706_six/`. Results in Current answer.

The ERA5 ground wind (3.5 m/s from 319 deg for 06:00) disagrees with the summit weather
station (seq 688 header: 6.9 m/s from 18.9 deg). With the ground layer at 0.3 to 0.93 of
Cn², it sets most of the motion and its direction.

Candidate changes for Aaron to choose: fix the ground-layer weight rather than draw it;
take the ground-layer wind from the site weather; interpolate ERA5 in time.

### 2026-10-28 — products `atmo_sim` and `atmo_sim_moments`; seq 688 rerun

Built the two products (`rubinwork/products/atmo_sim/`, `rubinwork/products/atmo_sim_moments/`):
- `atmo_sim` v1 draws the guide stars in the real ROIs at the tracked positions and
  fluxes, with the real stamp times, pedestals and read noise. The rotator angle enters
  through the initial `SkyWcs`. The stamps are written in the real `guider_raw` file
  layout and ingested in place into `u/roodman/atmo_sim/v1/20260706`, chained with
  `LSSTCam/raw/all` as `u/roodman/atmo_sim/v1`. One star per science CCD goes to
  `science_stars.fits`. Method changes are in [`method.md`](method.md).
- `atmo_sim_moments` v1 runs the `guider_moments` builder unchanged on that chain, and
  measures the science stars in the ConsDB `ccdvisit1_quicklook` columns.
- Tests: `rubinwork/products/atmo_sim/tests` checks the simulated files against the
  real headers and the star position after the amplifier inverse; 54 passed with the
  `guider_roi` and catalog tests.

Seq 688 (rotator 32.03 deg, ERA5 step 06:00 UTC, r0_500 0.109 m, L0 48.1 m, screens
3276.8 m built against 2144 m needed, 11 min and 26 GB on one core). Run
`output/20261027_visits_20260706_688/` (`compare_visit.py`): numbers in Current answer.
The rotator angle is `boresightRotAngle` 32.03 deg; the −13.09 deg in the August log is
the summit_utils `camRotAngle`, a different angle.

Next: rerun 688 over a few psfws seeds (ground-layer weight 0.05 to 0.93) to see how much
of the motion excess is the turbulence split; then 5 visits.

### 2026-10-27 — review and restructure plan

### 2026-10-27 — review and restructure plan

Reviewed the August state. Problems with the seq 688 result:
- the run used `camRotAngle` 0 deg against the real −13.09 deg;
- `--data-summary output/night_<D>/moments/...` no longer exists (data is in the
  `guider_moments` product, `v4/nights/20260706`);
- the stamps are `.npy`;
- the three output directories carry no `run.json`.

On s3df, GalSim 2.8.5, psfws, imSim 2.0, xarray, cfgrib, cdsapi and summit_utils
import; `lsst.summit.extras` does not. Aaron has the CDS token and the July 2026 ECMWF
file and will copy them here.

Agreed layout:
- **product `atmo_sim`**: the simulated pixels.
  - Guide stamps are written as `guider_raw` `Stamps` to a RUN collection
    `u/roodman/atmo_sim/...` in `/repo/main`, read through a chain with
    `LSSTCam/raw/all`. They copy the real visit's headers (ROI, `STMPTMJD`, pointing,
    `ROTPA`), so `GuiderReader.get` and the `guider_moments` builder run unchanged.
  - One star per science CCD (189), at the CCD centre in its pixel frame, goes to one
    multi-HDU FITS per visit.
  - It uses the same atmosphere as the guiders.
- **product `atmo_sim_moments`**: the `guider_moments` builder run on the sim collection,
  and a per-CCD PSF table in the ConsDB `ccdvisit1_quicklook` columns (`psf_sigma`,
  `psf_ixx`, `psf_iyy`, `psf_ixy`, CCD pixel frame).
- **this study**: sim against data on (1) FWHM and ellipticity amount and pattern across
  the science CCDs, against ConsDB, and (2) guider centroid-motion second moments,
  against `guider_moments`. The ellipticity frame rotation is done here.

The inverse of the `summit_utils` amplifier-to-DVCS chain is the library
`rubinwork.guider_roi`: `raw = ampToCcdView(rot90(dvcs, +nQuarter))`. Its tests
(`rubinwork/tests/test_guider_roi.py`, 3 passed) run on the real 20260706/688
`guider_raw` for all 8 guiders:
- `dvcs_to_amp` inverts `roiImageToDvcs` bit for bit;
- an L-shaped marker written through `ccd_to_amp` and read back by `summit_utils`
  `convertRawStampsToView` lands at the CCD pixel it was drawn at, with DVCS axes along
  the FOCAL_PLANE images of CCD +x and +y (no sign flip);
- the local Jacobian gives 0.2 arcsec/pix within 2% (relative).

Next:
1. Write the `atmo_sim` product skeleton (variants, writer to the Butler and FITS,
   manifest).
2. Rerun 688 with the real `ROTPA` and time.
3. Then 5 visits, then the night as a batch job (needs Aaron's OK).

The old `atmo_sim/` and `atmo_sim_20260705_688/` output directories are to be deleted
once Aaron agrees.

### 2026-08-18 — backfilled from docs and history

Backfilled 2026-10-18 from the archived note `notes/claude-memory/guider-atmo-sim.md` (the docs lived only there) and the commits `509325f` to `4b4f617`. Converted as active in the phase 5 triage.

The simulator has two modes: `guiders` (one star per guide CCD, 150 or 30 s visits) and `fov` (one star per science CCD, for focal-plane maps). Render `phot` (geometric plus second kick) or `fft` as a cross-check. Astrometric shifts from the FoV mode are about 10 mas RMS. `guider_atmo_compare.py` reuses `measureStampMoments` from the `guider_moments` product, now imported from `rubinwork.products.guider_moments.guiderMoments` rather than through the `guider/code` shim.

Next: an RSP run with the real geometry and exact UTC for the definitive motion ratio; the night batch.
