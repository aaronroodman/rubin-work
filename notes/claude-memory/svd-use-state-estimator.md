---
name: svd-use-state-estimator
description: v-modes come only from aos_state.make_state_estimator; the S-matrix is fixed at rotator zero and wavefronts must be OCS
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0db870d9-43c0-4dd3-9cb2-116d0e3e9f39
  modified: 2026-09-17T05:43:35.475Z
---

Never build the OFC sensitivity-matrix SVD outside OFC code. The helpers in
`aos/code/aos_state.py` are the only sanctioned path, and there are exactly two pieces:

- **`make_state_estimator`** → the v-mode basis. Wraps ts_ofc `StateEstimator`, asserts the
  resolved controller normalization is `range0.5_fwhm-0.15.yaml`, and raises rather than
  silently correcting. Every v-mode — measured, LUT, Trim — is reported in this basis.
- **`corner_recovery_basis`** → the *inversion* basis for a measured 84-row corner wavefront
  (4 corners x 21 Zernikes). `StateEstimator.Vh` comes from the unselected 899-row Double
  Zernike slab and does not span that problem, leaving a 2.3e-02 µm wavefront-residual floor
  even on noiseless data; the corner-evaluated SVD closes to 2.1e-14 µm.

`recover_optical_state` is the hybrid: inverts in the corner basis, reports v-modes through
`get_vmodes_from_dofs`. `build_geom_svd` and `project_dofs_to_vmodes` are retired.

Two constraints that are easy to get backwards:

- **The sensitivity matrix is always evaluated at camera rotator angle 0.0 deg**
  (`SMATRIX_ROTATION_ANGLE_DEG`) — an AOS group decision, because re-evaluating per rotator
  angle redefines the v-modes visit by visit and would confuse the control loop. There is
  deliberately no rotation-angle argument. Do not add one.
- **Wavefronts must therefore be in the telescope frame (OCS, `ZK_FRAME`)**, which is what all
  of Aaron's work to date assumes. `recover_optical_state` takes an explicit `zk_frame` and
  raises on anything else, because CCS input is undetectable from the numbers — it returns a
  rotator-dependent answer that inverts sign over 90 deg. Note ts_ofc's own `dof_state` wants
  the opposite pairing (CCS plus the angle); the two are consistent pairs, and mixing them is
  a silent error.

`truncate_index` sets the mode count, not the DOF set — the controller yaml default of 12
silently caps a 34-mode scheme, so pass `n_modes`. `zn_selected` does **not** affect
`StateEstimator.Vh` (its SVD skips `zn_idx`); it enters only `get_sensitivity_matrix`.

**Why:** four separate SVD sites had drifted into two genuinely different bases, and
near-degenerate singular-value pairs make individual v-mode vectors basis-dependent — only v1
and the pair magnitudes are unique — so v-modes computed two ways cannot be compared at all
unless they share one `Vh`.

**How to apply:** call the `aos_state` helpers; never write `np.linalg.svd` on a sensitivity
matrix. If a v1 number disagrees with a stored one by a sign, suspect the basis before
suspecting the physics.

Related: [[aos-vmode-normalization]], [[frame-conventions-ccs-ocs]], [[aos-dof-terminology]],
[[aos-22dof-reduced-set]].
