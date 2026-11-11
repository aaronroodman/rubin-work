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
