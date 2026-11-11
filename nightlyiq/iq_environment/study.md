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
