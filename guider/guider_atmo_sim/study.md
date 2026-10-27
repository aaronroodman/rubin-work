---
study: guider_atmo_sim
topic: guider
status: active
question: "Does a phase-screen atmosphere simulation reproduce the guide-star size and centroid motion measured on the guiders?"
answer: "Partly: static T_stamp matches the data to 0.5% (relative), but the centroid-motion second moments are over-predicted 1.5 to 1.6 times (dimensionless, sim over data; T_motion 0.0078 against 0.0050 arcsec^2) on day_obs 20260706 seq 688. The night batch was never run."
uses:  # imports guiderMoments only; its data side still reads guider moments from the old guider/output tree
  guider_moments: v4
started: 2026-08-06
---

# `guider_atmo_sim`: simulated atmospheric PSF on the guide sensors

> **Code:** `code/guider_atmo_*.py`, `code/fetch_cp_weather.py` · **Notebook:** `notebooks/galsim_guider_movie.ipynb` · **Output:** `output/` (the scripts default to a relative `output/`; run from this directory)

## Question

Does a GalSim phase-screen atmosphere (imSim Ellerbroek profile, or psf-weather-station winds), drawn as 50 ms stamps at 5 Hz on the 8 guide sensors, reproduce the guide-star size and centroid motion measured by the guider moments?

## Current answer

Partly, on one visit (day_obs 20260706, seq 688):

- **Static size matches:** `T_stamp` agrees with the data to 0.5% (relative), so the FWHM calibration works. Static ellipticity is optics-dominated and the sim has no optics, so it is for reference only.
- **Centroid motion is over-predicted** 1.5 to 1.6 times (dimensionless, sim over data): `T_motion` 0.0078 against 0.0050 arcsec², `Ixx` worst at 1.6 times. Centre jitter per axis 0.09 arcsec (sim) against 0.07 arcsec (data). Likely too much ground-layer or wind motion in psfws, or the placeholder date.

The night batch (`guider_atmo_night.py`) was never run.

## Method

[`method.md`](method.md): what the simulation does (turbulence profile, seeing match,
phase screens, rendering, guide and science-CCD stars, geometry, what is left out).

## Log

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
