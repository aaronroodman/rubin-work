---
study: iq_environment
topic: nightlyiq
status: active
question: How do science-visit FWHM, AOS_FWHM, ellipticity and donut blur correlate with airflow, thermal telemetry and the intra - extra donut blur difference?
answer: ''
uses:
  cwfs_science: d13-blitz_rso1071@20261026
  value_added: v1@snapshots/20261105
started: 2026-11-11
---

# `iq_environment`

## Question

How do science-visit FWHM, AOS_FWHM, ellipticity and donut blur correlate with airflow, thermal telemetry and the intra - extra donut blur difference?

## Current answer

None yet.

## Log

### 2026-11-11 — started

Study created.

### 2026-11-11 — ellipticity validation

`code/ellipticity_validation.py` writes
`output/20261111_ellipticity_validation/ellipticity_validation.pdf` (10 pages) from the
`value_added` `psf_shape` columns, Az/El frame. Sample: `psf_n_ccd` >= 150 of 189 science
CCDs, 20260424 seq < 398 dropped (LUT compensation off); 128,835 visits, 287 nights. No
science-program cut yet: that needs ConsDB `observation_reason`, down on 2026-11-11.

- Internally consistent: no visit has percentiles out of order or |mean e| > mean |e|.
  xi_e1e1 + xi_e2e2 is negative on 1 visit at 0.35–0.59 deg and on 3,664 (fraction
  0.0284 of visits) at 1.29–1.52 deg.
- Medians over visits (dimensionless): |e| p10 0.0378, p50 0.0777, p90 0.1319;
  |mean e| 0.0612. Most of the ellipticity is field-coherent: |mean e| / mean |e| has
  median 0.782 ([ellipticity_coherence](figures/ellipticity_coherence.png)).
- The two-point functions add little: xi_e1e1 + xi_e2e2 at 1.29–1.52 deg tracks
  |mean e|² (Pearson r +0.996, Spearman rho +0.983, n 128,835), and at 0.35–0.59 deg
  tracks |e| p50 (Spearman rho +0.99). For correlations, |e| p50, |mean e| and their
  ratio carry the information.
- Mean e1 (Az/El) sits at about −0.015 to −0.02 (dimensionless), flat against rotator
  angle, so the coherent part is telescope-fixed, not camera-fixed.
- Coverage: no observing 20260715 to 20261025 (snow storm on Cerro Pachón). The build
  predates the late-October nights, so `psf_shape` is empty on 20261026, 20261028 and
  20261031 to 20261103; a later `value_added` build will fill them.

### 2026-11-11 — first correlation pass

`code/fetch_consdb.py` caches ConsDB metadata, temperatures and visit IQ
(`output/cache/consdb_per_visit.parquet`); `seeing_zenith_500nm_median` is NULL on every
visit, so `fwhm_z500` is computed here (FWHM x airmass^-0.6 x (lambda_eff / 500 nm)^0.2).
`code/iq_correlations.py` writes `output/20261111_iq_correlations/iq_correlations.pdf`
(6 pages) and the Spearman matrices as CSV. Sample: img_type science, the nightly IQ
report's engineering `observation_reason` pattern excluded, 20260424 seq < 398 dropped;
88,348 visits, 245 nights. Within night: night medians subtracted, nights with >= 20 visits.

- Medians: PSF FWHM 1.0738 arcsec, AOS FWHM 0.4192 arcsec, donut blur 0.9825 arcsec,
  |e| p50 0.0728 (dimensionless).
- Within night every correlation is weak: no |Spearman rho| above 0.19 except elevation.
  The airmass^-0.6 scaling overcorrects: within night, elevation against raw FWHM
  gives rho -0.36 and against fwhm_z500 +0.13.
- Top-ring sonic axis 2, within night, against fwhm_z500 (rho): 123 -0.18, 124 -0.04,
  125 +0.13, 126 -0.04 (n about 49,000). More flow down the tube goes with smaller FWHM
  at 123 ([iq_vs_sonic_axis2](figures/iq_vs_sonic_axis2.png)); the sign differs by
  sensor, as the wind pattern in `airflow/airflow_correlations` does. The axis-2 RMS and
  cross-axis speed at 124 give -0.17. Wind and seeing are not yet separated.
- Intra - extra blur: within night, rho -0.12 against fwhm_z500, -0.15 against donut
  blur and +0.07 against AOS FWHM. Over all visits it is larger (-0.42 against donut
  blur, +0.39 against |e| p90), so most of it is between-night.
- Ellipticity has no input above |rho| 0.13 within night; over all visits it follows
  the temperatures (|e| p90 against outside temperature +0.37), a seasonal pattern.

### 2026-11-11 — wind and elevation controlled; split by pointing relative to the wind

`code/iq_controls.py` writes `output/20261111_iq_controls/iq_controls.pdf` (7 pages),
`controls.csv` and `wind_direction_split.csv`. Two controls: partial Spearman rho (ranks
regressed on wind-speed and elevation ranks) and Huber `RLM(target ~ input + wind speed +
airmass, HuberT)`. Each over all visits and within night. Huber errors are formal and
ignore the correlation of visits within a night, so they are too small.

- Sonic 123 axis 2 survives the controls within night: against fwhm_z500, rho -0.176,
  partial -0.171, Huber -0.0473 +/- 0.0014 arcsec per m/s (n 47,390); |axis 2| gives
  partial -0.182, Huber -0.0670 arcsec per m/s. It is negative in every pointing bin and
  every wind-speed tercile, including winds below 3.5 m/s (rho about -0.16), so it is
  not a wind-speed proxy ([sonic_axis2_wind_split_within](figures/sonic_axis2_wind_split_within.png)).
  Sonics 124 to 126 give no consistent sign. Over all visits the 123 signal is masked
  (raw rho -0.003, partial -0.127): nights differ in seeing more than in airflow.
- Median axis-2 flow at 123 rises with wind speed when pointing into or across the wind
  (0.73 and 0.87 m/s in the top tercile, 6.6 to 20.6 m/s) and is near zero down wind.
- Intra - extra blur against donut blur: within night rho -0.151, partial -0.187, Huber
  -0.2045 +/- 0.0052 arcsec per arcsec (n 55,580). The control raises it within night.
- AOS FWHM over all visits falls with sonic 124 axis-2 RMS (partial -0.142) and 123
  axis 2 (partial -0.137); within night these are below |rho| 0.1.

### 2026-11-11 — note: ellipticity is not primarily optical

From `optatmo/psf_vs_cwfs` (Aaron): an S-matrix optical model fitted to the focal-plane
moments reproduces the ellipticity, third-moment and FWHM patterns, but its wavefront
agrees with the donut (CWFS) wavefront only in Z4. So the ellipticity likely comes from
elsewhere. The corner Z5 and Z6 inputs in `ellipticity_drivers.py` are therefore a null
check, not a candidate driver. The coherent ellipticity is fixed in Az/El, not in the
camera (mean e1_AzEl about -0.015 at every rotator angle, elongation along elevation).
That was read as pointing to image motion; it does not, since Az/El-fixed is OCS-fixed
and so includes the telescope optics (corrected in the ellipticity-drivers entry below).

### 2026-11-11 — guider shape and motion; elevation dependence of e1

`code/fetch_guider.py` caches per-visit guider quantities from `guider_moments` v4
`stars` (`output/cache/guider_per_visit.parquet`): image motion in Az/El (per stamp,
median over >= 3 stars of `daz` x cos(alt) and `dalt`; drift kept) and the median guider
stamp shape `e1_altaz`, `e2_altaz`. Only 7 nights have star tables, 20260705 to 20260712
(20260512 and 20260513 are empty). `code/guider_ellipticity.py` writes
`output/20261111_guider_ellipticity/guider_ellipticity.pdf` (6 pages); 4,441 science
visits.

- The guider stamps see the science ellipticity, visit by visit: science mean e1_AzEl
  against guider e1_altaz Spearman rho -0.745, e2 against e2 -0.550, the same within
  night (-0.758, -0.550); slope about -1 ([science_vs_guider_e](figures/science_vs_guider_e.png)).
  The sign flip in both components means summit_utils `altaz` is rotated 90 deg from
  the psf_shape Az/El frame.
- Guided image motion does not cause it: motion RMS 0.0883 arcsec, predicted |e| 0.0055
  against science |mean e| 0.0510 (medians, dimensionless). The ellipticity is in each
  short guider stamp, so its source acts faster than the 5 Hz guider rate, or is not
  motion at all.
- Mean e1_AzEl falls with tan(zenith angle) in every band (88,348-visit sample):
  median from +0.019 to +0.049 at tan z <= 0.2 down to -0.092 (g), -0.065 (u), -0.034
  to -0.039 (r, i, z), -0.023 (y) at tan z 1.0 to 1.5. Elongation along elevation away
  from zenith. g steepest fits differential chromatic refraction, but r, i and z
  sharing one slope does not: a second, achromatic elevation-dependent term is needed.
  The positive e1 near zenith (elongation along azimuth) may be azimuth tracking error.
- Consequence for the driver search: band and zenith angle must be controlled before
  thermal or airflow inputs are tested against signed ellipticity.

### 2026-11-11 — ellipticity drivers, band and zenith angle fitted out

`code/ellipticity_drivers.py` writes `output/20261111_ellipticity_drivers/ellipticity_drivers.pdf`
(7 pages) with `pointing_model.csv`, `variance_budget.csv`, the Spearman matrices,
`forward_selection.csv` and `tree_importance.csv`. Pointing model: per band, Huber
target ~ 1 + tan z + tan^2 z; every fit uses its residuals and elevation is not an input.
Huber forward selection (5 steps, 5 night-grouped folds, 12,000-visit subsample) and
boosted trees on the same 40,406 visits that have every input. Inputs: the 23 thermal,
17 airflow, intra - extra blur, louvers, time after twilight, |pointing - wind|, corner
Z4-Z6 deviation (OCS) and AOS FWHM.

- Corner focus drives the signed ellipticity. e1_AzEl residual against corner Z4
  deviation: Spearman rho -0.568 over all visits, -0.537 within night (n 40,406); Huber
  slope -0.1311 +/- 0.0006 per um of wavefront over all visits (n 59,209), -0.1260
  within night, the same in every band (-0.114 u to -0.156 z, y). Corner Z4 nMAD is
  0.179 um, so focus moves e1 by about 0.023, half its residual nMAD of 0.046. It is
  the first input chosen for e1 and cuts the held-out nMAD from 0.0459 to 0.0393 (all
  visits) and 0.0423 to 0.0366 (within night). e2 follows weakly (+0.032 per um).
- Az/El-fixed is OCS-fixed, so this is consistent with static OCS astigmatism made
  visible by a varying focus: e1 changes sign through focus, is achromatic and is in
  every guider stamp. It also matches `optatmo/psf_vs_cwfs`, where only Z4 agrees.
  This supersedes the image-motion reading in the 2026-11-11 note above.
- Thermal, airflow and the blur difference add little once focus is in: no other input
  improves any held-out nMAD by more than about 2 %, and within night every thermal or
  airflow |Spearman rho| against an ellipticity residual is <= 0.12. The blur
  difference is the first pick for |e| p50 and p90 over all visits (p90 0.0493 to
  0.0449) but drops to rho 0.07 within night, so it is a between-night term.
- AOS FWHM tracks the ellipticity magnitude within night (rho +0.21 against |e| p50,
  +0.26 against |e| p90 residuals). Corner Z5, Z6 against e1, e2 stay below |rho| 0.17.
- Variance budget: 18 to 45 % of each target's variance is between nights (|e| p90
  highest), so most of the change is visit to visit.
- Next: does focus error explain the achromatic part of the e1 trend with zenith angle
  (gravity-driven focus the LUT misses)? Fit e1 on corner Z4 and tan z together.

### 2026-11-11 — focus against zenith angle

`code/focus_zenith.py` writes `output/20261111_focus_zenith/focus_zenith.pdf` and
`focus_zenith.csv`: per band, Huber e1_AzEl ~ 1 + tan z + tan^2 z with and without the
corner Z4 deviation, on the same 59,209 visits.

- Focus does not carry the zenith-angle trend. Corner Z4 changes by -0.064 to +0.012 um
  of wavefront from zenith to tan z = 1 in g to y (+0.215 um in u), and adding Z4 leaves
  the modelled e1 change at tan z = 1 in place: r -0.075 to -0.078, i -0.071 to -0.078,
  z -0.074 to -0.073 (dimensionless). Focus and zenith angle are separate contributors.
- The achromatic part is about -0.075 in e1 at tan z = 1 (r to y); g (-0.132) and u
  (-0.149) are steeper, as DCR would add. Its source is open.

### 2026-11-11 — pointing into the wind

`code/wind_pointing.py` writes `output/20261111_wind_pointing/wind_pointing.pdf`
(6 pages) and `into_wind_fits.csv`. into = |pointing azimuth - wind direction| < 45 deg,
cross 45-135 deg, down wind > 135 deg the baseline. Huber q ~ into + cross + wind speed +
airmass, per weather-tower wind-speed tercile (0-3.5, 3.5-6.6, 6.6-20.6 m/s), then with
the four sonic axis-2 and |axis-2| flows added, on the 47,385 visits with every top-ring
sonic. Over all visits and within night.

- Reproduced: pointing into the wind gives a smaller FWHM, and only in wind. Into minus
  down wind, fwhm_z500: -0.1451 +/- 0.0047 arcsec in the top tercile and -0.0942 +/-
  0.0025 over all speeds (n 47,385); within night -0.0914 and -0.0661 arcsec. Below
  3.5 m/s it vanishes. In the top tercile the median is 1.00 arcsec into the wind and
  about 1.15 arcsec from 80 deg onward ([iq_vs_pointing_into_wind](figures/iq_vs_pointing_into_wind.png)).
- It is not optics: donut blur moves the same way (-0.1090 arcsec, all visits) and AOS
  FWHM does not (-0.0009 +/- 0.0009 arcsec). So it is seeing in or near the dome.
- The axis-2 flow carries only part of it. Within night, adding the sonic flows cuts
  the into-wind FWHM term by 24 to 31 % (fwhm_z500 -0.0661 to -0.0456 arcsec); over all
  visits the change is mixed (+16 % overall, -16 % in the top tercile). No sonic's
  axis-2 flow peaks when pointing into the wind: sonic 123 is largest across the wind
  (about 0.9 m/s at 100 deg in the top tercile) and sonic 125 is negative into the wind.
- Ellipticity barely moves: into minus down wind, e2_AzEl residual -0.0102 (all visits,
  -0.0140 in the top tercile), |e| p90 residual +0.0096, e1 residual +0.0001
  (dimensionless). Corner Z4 shifts by +0.024 um into the wind, worth about -0.003 in e1
  at -0.13 per um.
