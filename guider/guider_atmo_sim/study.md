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

## Log

### 2026-08-18 — backfilled from docs and history

Backfilled 2026-10-18 from the archived note `notes/claude-memory/guider-atmo-sim.md` (the docs lived only there) and the commits `509325f` to `4b4f617`. Converted as active in the phase 5 triage.

The simulator has two modes: `guiders` (one star per guide CCD, 150 or 30 s visits) and `fov` (one star per science CCD, for focal-plane maps). Render `phot` (geometric plus second kick) or `fft` as a cross-check. Astrometric shifts from the FoV mode are about 10 mas RMS. `guider_atmo_compare.py` reuses `measureStampMoments` from the `guider_moments` product, now imported from `rubinwork.products.guider_moments.guiderMoments` rather than through the `guider/code` shim.

Next: an RSP run with the real geometry and exact UTC for the definitive motion ratio; the night batch.
