---
study: ofc_reproduction
topic: smatrix
status: active
question: "Does a batoid_rubin build of the double-Zernike sensitivity matrix reproduce the matrix shipped in ts_ofc, DOF by DOF?"
answer: "Yes: 50 of 50 DOF agree with |corr| > 0.999 (dimensionless), RMS ratio 1.000 and the same sign; max residual 0.11 µm of wavefront, relative residual < 0.2% for every DOF. It needs ZCS, both bending-mode flips, degree angle units and a sign patch on dy and Ry."
uses: {}  # no rubin-work products; batoid_rubin data and the ts_ofc matrix
started: 2026-07-31
---

# `ofc_reproduction`: batoid_rubin reproduces the ts_ofc sensitivity matrix

> **Code:** `code/compare_ofc.py`, `code/detailed_comparison.py`, `code/plot_comparison.py` (library `rubinwork.smatrix.compute_smatrix`) · **Conventions:** [`../docs/conventions.md`](../docs/conventions.md) · **Write-up:** [`../README.md`](../README.md), Results · **Output:** `smatrix/output/` (topic data)

## Question

Does a batoid_rubin build of the double-Zernike sensitivity matrix reproduce the matrix shipped in ts_ofc, DOF by DOF?

## Current answer

**All 50 DOF agree with the ts_ofc matrix** (band r, ts_ofc `develop`): |corr| > 0.999 (dimensionless), RMS ratio 1.000 (dimensionless, ours over OFC), same sign. The max residual is 0.11 µm of wavefront, on the high-leverage M2 and camera Rx, Ry; the relative residual is < 0.2% for every DOF.

Matching needed: the ZCS frame, `flip_m1m3_bending_modes` and `flip_m2_bending_modes`, `dof_angle_units="degree"`, the `bend.yaml` mode set (M1M3 [0..18, 26], M2 [0..16, 25, 26, 27]), a 1e-3 deg angle step, and a sign flip on M2 and camera dy and Ry (`OFC_Y_SIGN_FLIP`). Why batoid_rubin's ZCS differs from OFC by that sign is an open question for the batoid_rubin maintainers ([`../docs/conventions.md`](../docs/conventions.md)).

## Log

### 2026-09-07 — backfilled from docs and history

Backfilled 2026-10-18 from `smatrix/README.md` (Results, gotchas) and `docs/conventions.md`. Done in the phase 5 triage; the sign-flip question stays open in `conventions.md`. The three scripts moved here from `smatrix/code/`; they still write and read the matrices in `smatrix/output/`, which stays the topic's data directory (triage decision 3: the mode datasets and matrices stay smatrix-internal). They now import `CONFIG_TAG` from `rubinwork.smatrix.compute_smatrix` instead of the shim.

### 2026-11-06 — tracking the small residuals

Reopened to reproduce the dense matrix again as the baseline for `sparse_fit` and to explain the residuals. New code: `code/probe_field_points.py` (per-field-point batoid against the OFC DZ evaluated at the same points), `code/build_dense.py` and `build_dense.sbatch` (full rebuild into a study run, bands `ref` and `r`). The old matrices in `smatrix/output/` no longer exist.

- **The ts_ofc matrix was built at band `ref`** (`LSST_g_500.yaml`, 0.5 µm), not `LSST_r.yaml`. With `ref`, the rigid-body scale (ours over OFC, dimensionless) is 0.99998 to 1.00003; with `r` it is 0.99945 (M2 dx, Rx), 1.0003 (camera Rx) and 1.0012 (camera dx). The `r` intrinsic file in ts_ofc does match `LSST_r` (6.8e-5 µm RMS wavefront).
- **Its values date from 2024-03-27** (ts_ofc 1707e32). The 2024-12-12 commit 2e79e94 negated columns of the file by hand (M2 and camera dy and Ry, the M1M3 bending modes) and multiplied the angle columns by 3600; the 2025-10-03 update left the values unchanged (all three agree to < 1e-12 relative). So the `OFC_Y_SIGN_FLIP` patch reproduces a hand edit in ts_ofc, not a batoid_rubin convention.
- Ruled out as causes: finite-difference step (central, 1 arcsec, 1e-2 deg), ray grid (nx 511), `batoid.zernike` projection and reference, the 2024-11 batoid_rubin builder, batoid 0.6.3, the v3.12, v3.14 and v1000 optics, and the bare `LSSTBuilder` defaults.
- The per-point probe cannot resolve the bending modes: its own DZ-truncation floor is 6e-3 relative for M2 B14 against 7e-5 for M2 Rx. The bending-mode comparison waits for the full DZ build.

