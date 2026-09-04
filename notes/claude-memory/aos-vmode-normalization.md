---
name: aos-vmode-normalization
description: "How AOS v-modes are normalized/computed across rubin-work (geom weights, StateEstimator vs build_ofc_svd, 4-CWFS vs double-Zernike)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9325d2f9-106f-4f9c-9a8a-9af33e47c0f5
---

AOS v-mode computation in rubin-work (established July 2026, documented in `rubin-work/olr/docs/vmode_normalization.md`):

- **Canonical normalization = geom_mean** `n_j = r_j^0.5 * f_j^-0.5`. It is stored in the OFC config's `normalization_weights` (v13: `ts_config_mttcs/MTAOS/v13/ofc`). The yaml is misleadingly named `range0.5_fwhm-0.15.yaml` — a typo for `-0.5`; its header says `alpha=0.5, beta=-0.5`. Get it by `OFCData("lsst", config_dir=<v13>)` — **never** bare `OFCData()` (loads an old NON-geom default → v1 becomes the M2-tilt mode ~1e-4 at convergence instead of focus). Do NOT recompute `sqrt(r/f)` from `compute_normalization_components` (that uses corner-point FWHM, ~sqrt(2) off from the config's field-averaged FWHM).
- **Two v-mode engines, both geom-normalized, used for different cases (both needed):**
  - `StateEstimator.get_vmodes_from_dofs` (lsst.ts.ofc) — the **4-CWFS** case (sensitivity evaluated at the 4 corner sample_points). Used by nightly_report, olr nightly_table, nightly_tablemaker, aos_openloop, t539, and `aos/code/aos_state.py`.
  - `build_ofc_svd` (lsst.ts.intrinsic.wavefront.ofc_svd) — the **double-Zernike** case (field-mode k range × pupil-Zernike). Used by all `aos/code/run_*` scripts (bounce, dz_correlations, build_lut, thermal_correlations, plot_vmode_dof_matrix, psf_fp_maps, wfs_dof_compare). CWFS DOF recovery from 4 corners = `pinv(B @ U_eff) @ z_dev` (least-squares onto the low-rank controllable subspace; 4 corners over-determine n_keep modes, they do NOT determine the full field).
- `aos/code/aos_state.py` is the shared helper: `make_state_estimator` + `vmodes_from_dofs` (StateEstimator), plus `build_geom_svd` (hand-rolled 4-corner SVD, used only for olr `zk_constrained` reconstruction).
- **Degeneracy caveat:** several singular values are exactly degenerate (x/y symmetry doublets), so individual v-modes within a pair are basis-dependent (arbitrary SVD rotation). Only v1 (focus) and per-pair magnitudes are unique; don't over-interpret individual v2/v3/v4/v5/v8/v9 or cross-compare individual modes between the two engines.
- **Zernike set (resolved July 2026): use 21 = Z4–Z26 excl Z20,Z21.** The OFC config default of 19 (Z4–Z22) is a never-used constructor default — the operational loop reconfigures to 21 on EVERY visit. In ts_mtaos `Model._calculate_corrections` (`model.py:1422`) `self.ofc.ofc_data.zn_selected = zk_indices`, where `zk_indices` is the Noll list from the WEP donut-fit output (WavefrontCollection), currently the 21. (Also a static path `set_ofc_data_values` model.py:1778 from the `zn_selected` startClosedLoop config key set by the EnableAOSClosedLoop standard script, but the per-visit WEP override is operative.) So operations AND analysis are both 21 — no mismatch. `aos_state.make_state_estimator` sets `ofc.zn_selected = ZK_NOLL` (21); confirmed sensitivity becomes (84,22). Both aos_state engines + recover_optical_state are on the 21-term basis.

Related: [[aos-dof-terminology]].
