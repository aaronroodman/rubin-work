# Handoff: RBR against a conventional penalty matrix — continue on S3DF

> **Status:** open · **Last updated:** 2026-10-19 · **Kind:** handoff (study `regularized_inversion`)

The laptop run gave a preliminary answer (write-up section "RBR against a conventional
penalty matrix" in [`regularized_inversion.md`](regularized_inversion.md)). It was
numpy-only and used substitute inputs. Redo it on S3DF with the smatrix machinery and
Aaron's chosen normalization before quoting any number.

## Question

Is 50/50 RBR the only way to keep every DOF within range and still correct the optical
state well, or does a conventional penalty matrix (RSO-217 section 3.5, eq. 8) do as well?
Aaron found 50/34 RBR held the hexapods back while it constrained the small mirror modes,
and moved to 50/50 RBR.

## Preliminary answer (laptop, to be confirmed)

- The 34-mode truncation itself is the problem. Its implied `H` couples hexapod and
  bending. A pure hexapod move is not in span(V_34): the principal angles reach 31.8 deg.
  RBR inside that span shrinks the hexapod to remove the bending that comes with it. A pure
  M2 dz at 0.2 of its range is recovered at 0.656 of the true amplitude by RBR-34. RBR-50
  and the full-rank quadratic penalty recover it at 1.000 (dimensionless, recovered over
  true amplitude).
- At max `|d_j|/r_j` = 1, RBR-50 had the lowest residual. The full-rank quadratic
  `H = diag(1/r_j²)` was 2% to 10% worse, is linear, does not depend on the normalization,
  and has 12% to 31% lower Cam dx noise.

## What the laptop run substituted, and what to use on S3DF

| input | laptop (substitute) | S3DF (use this) |
|---|---|---|
| normalization `w` | `ts_config_mttcs/.../range0.5_fwhm-0.15.yaml`, read directly | Aaron's sqrt normalization, `w_j = r_j^0.5 f_j^-0.5`, through `build_ofc_svd` (its default `DEFAULT_NORM_YAML`). Confirm this is the file Aaron means |
| ranges `r_j` | `smatrix/output/normalization_50dof.npz` `r` (rb_stroke and force-limited bending), **not** matched to the weights above | the ranges that go with the sqrt normalization: `ri.dof_range_vector(svd)` (`r = w² f_quadrature`), cross-checked against `OFCData.dof_ranges` on ts_ofc `tickets/RSO-1007` |
| data | two cached legs (`Elev=70→40`, `Rot=0→60`), median `delta` per (k, j), noise from `delta_err` | all six bounce legs, per pair, via `rubinwork.bounce` exactly as `code/run_oic_compare.py` does |
| metric | L2 residual norm, µm of wavefront | also the achieved-residual FWHM, arcsec, through `aos/code/aos_fwhm.py` and ts_wep |
| RBR solver | `rubinwork.smatrix.regularized_inversion.invert_range_penalty` | the same, and cross-check against ts_ofc `range_bounded_recovery` (`tickets/RSO-1007`) |

The two local yaml files disagree. The `range0.5_fwhm-0.15.yaml` header says alpha 0.5,
beta −0.5, but its weights are not `r^0.5 f^-0.5` with the `normalization_50dof.npz` `f`.
Its ratio to `range-fwhm.yaml` runs from 0.006 to 6206 (dimensionless) per DOF. Settle
which file and which `f` define "the sqrt one" before running. The 34-mode results depend
on it; the full-rank ones do not.

One cross-check passed: with `range0.5_fwhm-0.15.yaml`, truncated 50/34 reproduced the
elevation 40 deg DOF in the write-up (M2_dx 112.5 µm, Cam_dx −436.9 µm). So that is the
file the bounce study used.

## Code

- Runner: [`code/run_penalty_matrix_compare.py`](code/run_penalty_matrix_compare.py).
  Keep the `Solvers` class, `report_structure`, `synthetic_hexapod`,
  `hexapod_amplitude_scan`, `real_legs` and `rho_scan`. Replace `load_inputs` and
  `load_legs`:
  - Build the SVD with `build_ofc_svd(pupil_j, 1, 6, 50, n_dof=50)`, with
    `pupil_j = list(range(4, 20)) + list(range(22, 27))`. Use `n_keep = 50` so `U_eff`
    holds all 50 modes. `Solvers` then takes `A = U Σ Vᵀ / w` from the SVD rather than the
    yaml, and the 34-mode solvers slice to rank 34.
  - Take `ranges = ri.dof_range_vector(svd)`.
  - Take the legs and per-pair `W` from the `main()` of `code/run_oic_compare.py`: the
    `analysis_section('bounce', ...)` call, the `quality_visit_mask` cut and
    `project_dz_table`. Score per pair and take the median, as the OIC table does.
- Write output through `rubinwork.common.study_run("smatrix/regularized_inversion",
  "penalty_matrix")`, not to `smatrix/output/...` as the laptop run did.
- Branches the RBR path needs: ts_ofc `tickets/RSO-1007` (`979ec73`, unpushed) and
  ts_intrinsic_wavefront `tickets/RSO-809` (`957a54f`, unpushed); see
  `notes/status/item5_miw_pupil_model_handoff.md`. Check those commits exist in the S3DF
  checkouts before running.

Template invocation, from `run_oic_compare.py` (same FAM fit table and param_set):

```bash
cd /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/smatrix/regularized_inversion/code && python run_penalty_matrix_compare.py --fits /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x --mi-name pathA_50_34_i_5rot --min-detectors 160
```

These arguments do not exist in the runner yet. Add them when replacing `load_legs`.

## Next steps

1. Settle the normalization file and its ranges (above). Rerun `report_structure` and the
   synthetic hexapod tests under it. These need no data.
2. Run all six bounce legs per pair: residual norm, FWHM, max `|d_j|/r_j`, and hexapod
   retention against the truncated 50/34 recovery, for T34, R34, R50, Q50 and Q34x.
3. Replace the `delta_err` Monte Carlo with the empirical pair-to-pair scatter of each
   solver's DOF.
4. Pick one global `ρ` for Q50 across legs and report its spread in max `|d_j|/r_j`.
5. Test the hybrid: RBR plus a small quadratic floor in range units
   (`lam0`-like, on `d_j / r_j` rather than `x`). It may keep R50's residual with Q50's
   hexapod noise.
6. Repeat on full states rather than differences: science visits (`olr/scheme_comparison`
   sample) and the MIW route. Hexapod amplitudes there are larger, and the R34 hexapod
   loss grows with amplitude (0.925 at 0.02 r to 0.554 at 0.4 r for M2 dz,
   dimensionless retention).
7. Replace the preliminary section in `regularized_inversion.md` with the S3DF numbers,
   then update `study.md`.
