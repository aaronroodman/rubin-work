---
study: normalization
topic: smatrix
status: done
question: "Can the 50-DOF normalization weights w_j = r_j^0.5 f_j^-0.5 of the OFC v13 configuration be rebuilt from the range and PSF-width sensitivity of each DOF?"
answer: "Yes: rebuilt weights match v13 to < 0.1% (relative), but only with f_j computed over Noll Z4 to Z22. The canonical weights are the ones stored in the v13 OFC config; read them, do not recompute."
uses: {}  # no rubin-work products; ts_config_mttcs v13
started: 2026-08-01
---

# `normalization`: the per-DOF SVD normalization weights

> **Code:** `code/make_normalization.py`, `code/full_normalization.py`, `code/validate_normalization.py` (library `rubinwork.smatrix.normalization_weights`) · **Reference:** [`../docs/vmode_normalization.md`](../docs/vmode_normalization.md) · **Output:** `smatrix/output/` (topic data)

## Question

Can the 50-DOF normalization weights w_j = r_j^0.5 f_j^-0.5 of the OFC v13 configuration be rebuilt from the range and PSF-width sensitivity of each DOF?

## Current answer

- The rebuilt 50-DOF weights `w_j = r_j^0.5 f_j^-0.5` (`r_j` allowed range, `f_j` field-averaged PSF FWHM response in arcsec per DOF unit) **match v13 to < 0.1%** (relative), and only when `f_j` uses Noll Z4 to Z22.
- The weight choice changes which mode is v1: with the v13 weights v1 is focus; with `OFCData()` built without `config_dir` it is M2 tilt (the old OLR and nightly-report bug).
- **Read the stored v13 `normalization_weights`; do not recompute them.** Details, the 50 values and the degenerate v-modes are in [`../docs/vmode_normalization.md`](../docs/vmode_normalization.md); open items go to `../docs/status/future_issues.md`.

## Log

### 2026-09-07 — backfilled from docs and history

Backfilled 2026-10-18 from `docs/vmode_normalization.md` and `docs/plots.md` section 5. Done in the phase 5 triage; the reference doc stays in `smatrix/docs/`. The three scripts moved here from `smatrix/code/` and now import `rubinwork.smatrix.normalization_weights` instead of the shim.
