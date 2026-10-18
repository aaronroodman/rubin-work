---
study: ofc_reproduction
topic: smatrix
status: done
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
