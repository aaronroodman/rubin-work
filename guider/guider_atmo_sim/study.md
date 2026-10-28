---
study: guider_atmo_sim
topic: guider
status: active
question: "Does a phase-screen atmosphere simulation reproduce the guide-star size and centroid motion measured on the guiders?"
answer: "Partly: over 6 visits of 20260706 the guider centroid-motion size matches on average (median T_motion sim over data 1.28, dimensionless) but per visit ranges 0.36 to 2.29 and follows psfws's random ground-layer weight; the motion direction does not match (median angle difference 42 to 69 deg, against 45 deg for no relation)."
uses:
  guider_moments: v4@nights/20260706
  atmo_sim: v1@visits/20260706{00195,00372,00603,00688,00726,00908}
  atmo_sim_moments: v1@visits/20260706{00195,00372,00603,00688,00726,00908}
started: 2026-08-06
---

# `guider_atmo_sim`: simulated atmospheric PSF on the guide sensors

> **Code:** `code/guider_atmo_*.py`, `code/fetch_cp_weather.py` · **Notebook:** `notebooks/galsim_guider_movie.ipynb` · **Output:** `output/` (the scripts default to a relative `output/`; run from this directory)

## Question

Does a GalSim phase-screen atmosphere (imSim Ellerbroek profile, or psf-weather-station winds), drawn as 50 ms stamps at 5 Hz on the 8 guide sensors, reproduce the guide-star size and centroid motion measured by the guider moments?

## Current answer

Partly, over 6 visits of day_obs 20260706 (seq 195, 372, 603, 688, 726, 908), with the
real geometry, ERA5 winds and a seeing match to the guider FWHM:

- **Motion size matches on average, not per visit.** Median over 48 sensor-visits of
  `T_motion` sim over data is 1.28 (dimensionless); per visit 0.36 to 2.29. The per-visit
  ratio follows the ground-layer weight that psfws draws at random (0.30 to 0.93,
  fraction of Cn²): Pearson r 0.63, Spearman rho 0.54, n 6 visits.
- **Motion direction does not match.** Median |PA sim − PA data| per visit is 42 to 69
  deg; 45 deg is what no relation gives. Motion ellipticity |Q|/T: median 0.09 to 0.40
  (sim) against 0.23 to 0.43 (data), dimensionless.
- **Static size** `T_stamp`: median sim over data 0.97.
- **Science CCDs**: median FWHM 0.915 arcsec (sim, atmosphere only) against 0.880 arcsec
  (ConsDB); median ellipticity e 0.024 (sim) against 0.060 (ConsDB). The atmosphere gives
  a third to half of the measured ellipticity amplitude.
- 5 of the 6 visits use one ERA5 step (06:00 UTC), and every ERA5 ground wind is about
  3.5 m/s, so the sim winds barely change through the night.

## Method

[`method.md`](method.md): what the simulation does (turbulence profile, seeing match,
phase screens, rendering, guide and science-CCD stars, geometry, what is left out).

## Log

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
