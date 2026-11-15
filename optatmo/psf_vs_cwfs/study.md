---
study: psf_vs_cwfs
topic: optatmo
status: parked
question: "Does a Double-Zernike plus atmosphere fit to stellar PSF moments recover the same wavefront as the corner wavefront sensors?"
answer: "Only focus agrees: the PSF-fit v1 (Z4) against CWFS has a robust slope 0.81 (dimensionless, PSF fit per CWFS), intercept +0.074, over about 924 FAM in-focus visits; v-modes above focus sit at noise level."
uses: {}  # reads the danish_1_2 FAM visits and the MIW from the old aos/output tree, not through the catalog
started: 2026-07-09
---

# `psf_vs_cwfs`: the PSF-moment wavefront fit against the corner sensors

> **Code:** `optatmo/code/` (`fit_optatmo.py`, `ensemble_corners.py`, `pipelines/`) · **Output:** old tree `optatmo/output/runs/famcwfs_prod/`

## Question

Does a Double-Zernike (DZ) wavefront on top of the measured intrinsic wavefront (MIW), plus an atmospheric kernel and a constant moment offset, fitted to stellar 2nd to 4th-order HSM moments, recover the same wavefront as the corner wavefront sensors (CWFS)?

## Current answer

Only focus agrees. Over campaign `famcwfs_prod` — about 924 FAM in-focus visits fitted, 17 nights, day_obs 20260315 to 20260513, plan `focus_then_full` — the PSF-fit v1 (Z4) against the CWFS v1 has a robust Huber slope of 0.81 (dimensionless, PSF fit per CWFS) and intercept +0.074 µm of wavefront. V-modes above focus sit near noise. The working hypothesis is that the higher-order PSF wavefront is dominated by atmospheric turbulence, which varies over the field and so is not at the pupil.

## Log

### 2026-11-14 — seeing-dependence of spin-2 moments filed

Filed an ad-hoc GalSim study from 2026-06-29 that bears on this question: how far e1·T and
e2·T (T = Ixx + Iyy, arcsec²) move when only the seeing changes. PSF = Kolmogorov(FWHM
0.6 to 1.2 arcsec) convolved with an 8.36 m, 0.61-obscuration optic, 750 nm, Z4 to Z11
random (0.3 µm rms of wavefront), HSM adaptive moments at 0.2 arcsec/pixel. Scripts in
`code/seeing_moments/` (they write to the current directory; not converted to
`study_run`); figures and `seeing_moments_delta.npz` (1000 realizations) in
`output/20260629_seeing_moments/`.

- `seeing_moments_study.py`: Z4 fixed at 0.3 µm, 12 realizations.
- `seeing_moments_varyz4.py`: Z4 to Z11 all random, 16 realizations, colored by Z4.
- `seeing_moments_noise.py`: shot noise for a 17th-mag star with a 250 e-/pixel sky.
- `seeing_moments_delta.py`: change in e1·T, e2·T across the FWHM range, 1000 realizations.

No result was written up at the time. The seeing-driven drift is a systematic error for any
fit of wavefront to stellar moments that does not model the atmosphere's size exactly.

### 2026-10-24 — inputs moved to the products

`optatmo/code/` now reads the FAM visits and fits through the catalog, `fam_tables`
variant `danish_1_2`, instead of `aos/output/fam_processing/danish_1_2/`. The product's
`rotator_angle` differs from the old table by up to 0.858 deg (3017 of 3385 visits, count;
at most 0.110 deg on 20260513). Recorded results were not rerun: for most purposes the camera rotator angle only needs to be known to about 1 deg (giant donuts excepted), so Aaron accepted the change (2026-10-24). `extract_cwfs_sigma.py` still reads the `aos/cwfs`
study's `wfs_corner_compare.parquet`, allow-listed; there is no product for it.

### 2026-08-21 — backfilled from docs and history

Backfilled 2026-10-18 from the archived note `notes/claude-memory/optatmo-moments-project.md` (the answer lived only there), `optatmo/README.md` phase 3, and the commits to `f16d1ae`. Parked in the phase 5 triage.

- First pass, day_obs 20260513: only Z4 correlates, slope about 0.66 (dimensionless); Tikhonov regularization dropped (raised the moment cost, no gain in correlation); MIGRAD equals scipy L-BFGS-B, so use L-BFGS-B.
- A frame bug was fixed on the way: star positions and moments were rotated from DVCS to OCS without the DVCS to CCS swap, which reflected the wavefront.
- The CWFS corners come from ConsDB (`consdb_cwfs.py`); on 20260513 seq 25 they agree with the Danish table to ≲ 0.03 µm Z4 on R00, R04 and R44.
- `ensemble_corners.py` writes one row per visit, `ensemble_corners_alldays_stage_full.parquet` (old tree).

Not done: an M3-footprint stitching test of the turbulence hypothesis (designed, not built).

The fit code stays in `optatmo/code/` with the forward-model library; moving it waits for the library split (triage: the JAX model becomes `rubinwork.optatmo` or a topic library). Its reads of `../aos/output` are allow-listed section 2 violations.
