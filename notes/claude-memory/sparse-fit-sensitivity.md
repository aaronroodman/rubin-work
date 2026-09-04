---
name: sparse-fit-sensitivity
description: "Sparse donut-fit idea for the MIW: fix secondary/tertiary Zernikes, revise the sensitivity; Piece-1 sensitivity-matrix findings"
metadata: 
  node_type: memory
  type: project
  originSessionId: 25f21140-87f5-47d9-b0ac-9fc8c2e9c154
  modified: 2026-09-02T19:30:49.136Z
---

Motivation: the MIW shows residual cross-correlations between radial orders of the
same azimuthal aberration (astig Z5/6<->Z12/13, coma Z7/8<->Z16/17, tetrafoil,
etc.; see radial-correlation plots optatmo/output/miw_radial_correlations.pdf).
The MIW is also NOT stable (a fraction of FAM coadds differ from the 5rot MIW), so
using it as-is won't give consistent results. Aaron's SPARSE idea: in the donut
wavefront fit, fit ONLY the primary aberrations and FIX secondary/tertiary to the
batoid intrinsic. That pushes real secondary/tertiary into the primary; to use it
consistently needs (a) a revised sparse sensitivity matrix and (b) a model of how
secondary/tertiary leak into the primary. See [[miw-products-and-m3-backprojection]].

ts_ofc sensitivity: `packages/ts_config_mttcs/MTAOS/ofc/sensitivity_matrix/*.yaml`,
a galsim DoubleZernike S[k,j,d] shape (31 field-Noll k, 29 pupil-Noll j, 50 DOF).
k=1 = field-constant; pupil index = Noll j. DOF order (state0_in_dof.yaml): M2 hex
dZ,dX,dY,rX,rY (0-4), Cam hex (5-9), M1M3 bend mode1-20 (10-29), M2 bend (30-49).
Because the field basis is orthonormal, a DOF's primary<->secondary field-MAP
correlation = corr(S[1:,j_prim,d], S[1:,j_sec,d]) -- no sim needed.

CORRECT MATRIX (verified 2026-09-01): the OFC loads its sensitivity via OFCData,
as in `rubin-work/aos/vmode_dof_ts_ofc.ipynb`: `OFCData("lsst",
config_dir=$TS_CONFIG_MTTCS_DIR/MTAOS/v13/ofc)`, `configure_controller()`,
`await configure_instrument("lsst")`, then `ofc.sensitivity_matrix`. OFCData's glob
`f"{camera}_sensitivity*.yaml"` resolves to **lsst_sensitivity_dz_31_29_50.yaml**,
which has ALL radial orders populated (coma2/tref2/astig3/tet2 present, 0 NaN).
The `new_measured_lsst_sensitivity_*` file (which zeroes those orders) does NOT
match that glob and is NOT loaded. RETRACTED my earlier wrong claim that
"production zeroes coma2/tref2/...". analyze_sensitivity_sparse.py loads via
OFCData (fallback: read the lsst_sensitivity yaml).

**Piece 1 done (2026-09-01), `aos/code/analyze_sensitivity_sparse.py` ->
output/sensitivity_sparse_analysis.pdf (loads the CORRECT full matrix):**
1. All radial orders ARE present in the OFC's matrix (page 0). [Earlier
   "production zeroes higher orders" was WRONG -- that was the non-loaded
   new_measured file; retracted.]
2. Nearly EVERY DOF (esp. bending modes + hexapod tilts)
   drives primary AND secondary of the same family with field correlation ~+/-1:
   per-DOF the secondary is a fixed scalar multiple of the SAME field pattern as the
   primary. => primary/secondary are near-degenerate per DOF -> their separation in
   the donut fit is ill-conditioned -> small biases leak between orders -> the MIW
   correlation. Strong motivation for sparse mode. Astig 1st-2nd from tilts
   (corr cos -0.44..-0.95, sin -0.40..-0.87) brackets the MIW astig 1st-2nd
   (cos -0.28, sin -0.62).

**Zernike PACKING (critical, resolved 2026-09-01):** the OFC measures the 21 PACKED
Zernikes `zn = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26]` (Z4-Z26
minus Z20,Z21), set via `ofc.zn_selected = zn` (see `aos/smatrix_vmode_info.ipynb`;
also `ZK_NOLL=[z for z in range(4,27) if z not in (20,21)]` in aos/code/aos_state.py,
camera_gravity_maps.py; ts_intrinsic_wavefront). `ofc.zn_idx = zn-4` after that
(default, if you DON'T set zn_selected, is a useless [0..18]). The raw
`ofc.sensitivity_matrix` (31,29,50) pupil axis = Noll (M2 tilt -> Noll7 coma), so
select the 21 by Noll index; `SensitivityMatrix.evaluate(...)[:, zn_idx, :]` output
pupil axis is Noll-4. StateEstimator's se.S/se.Vh use the RAW matrix incl piston/tilt
(Noll0-3) -> for observability recompute the SVD on the 21-packed subset.

**Sparse OBSERVABILITY (Piece 1b, `aos/code/analyze_sparse_observability.py` ->
output/sparse_observability.pdf, 4 pages = 2 schemes x 2 samplings):** if the donut
fit measures ONLY the primary (drop 2nd/3rd = [12,13,16,17,18,19,22,23,24,25,26]) do
the controlled v-modes stay observable? v-modes recomputed on the 21-packed set. Two
samplings via `_obs_matrices(ofc,se,sens,sampling)`: `dz`=field-complete (raw DZ, 31
field, pupil=Noll); `wfs`=the 4 corner WFS R00/R04/R40/R44_SW0 (`sens.evaluate` pupil
axis = Noll-4, so index by j-4). **ROBUST to sampling** -- same conclusion both ways:
- 22/12 (operational): ALL 12 controlled v-modes FULLY observable (median 1.00,
  min 0.99) at BOTH WFS and field-complete -- sparse loses NOTHING (22-DOF/12-mode
  set is primary-dominated).
- 50/34: field-complete median 0.97 / min 0.36; WFS median 0.96 / min 0.44. Same
  degraded DOF both samplings = high-order M2 bending (b10,11,14,15,19,20) -> 0.37-0.63
  (signal is in the dropped 2nd/3rd); a few controlled v-modes ~0.4. WFS is if anything
  slightly LESS degraded (the 4-corner sampling doesn't resolve the high-order field
  structure that distinguishes those modes anyway).
BOTTOM LINE: for the operational 22/12 scheme sparse mode is FREE (no DOF sensitivity
lost); only the extended 50/34 pays, in the highest M2 bending modes, and that's the
no-leakage UPPER BOUND (real Piece-2 leakage returns some). Committed 1564be9.
(Earlier version was WRONG: used default zn_idx + raw pupil incl piston/tilt ->
spurious top-mode degradation. Fixed with the ZN packing.)

NEXT: Piece 2 = the leakage matrix L[j_prim,j_sec] = how a fixed secondary in the
donut image projects onto the fitted primary (needs the WEP/danish donut forward
model, a Jacobian from injected-secondary sims). Piece 3 = A_sparse = A_prim +
L @ A_sec; rebuild a sparse MIW.
