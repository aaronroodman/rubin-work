---
name: miw-coadd-retrieval-bias
description: "Coadd-vs-MIW variation: the retrieval-bias hypothesis, the L estimator, and the equations doc that supersedes my earlier unbiased model"
metadata: 
  node_type: memory
  type: project
  originSessionId: 25f21140-87f5-47d9-b0ac-9fc8c2e9c154
  modified: 2026-09-04T18:17:23.962Z
---

**Primary reference: `rubin-work/aos/MIW_COADD_EQUATIONS.md`** — equation-level
nomenclature + derivation for the coadd-vs-MIW study. The *iterative MIW method*
itself was already written up in `rubin-work/notes/aos-measured-intrinsics/note.md`
§"MIW method" (steps 1-6) — don't duplicate it. ts_intrinsic_wavefront has no
equations. See [[miw-products-and-m3-backprojection]], [[sparse-fit-sensitivity]].

**Aaron's leading hypothesis (his, not mine):** a small bias in the Danish donut fit
redistributes power among primary/secondary/tertiary radial orders; the 50/34
SOFTWARE correction computed from that biased wavefront then subtracts a pattern
corresponding to no real optical state — *exacerbating* rather than removing it —
and block-to-block variation in the bias drives the coadd scatter.

**Key algebra.** With bias `B_+ = 1+B` acting on the PUPIL index j (same at every
field order k), `B_+ R[W] = R[B_+ W]`, so the measurement is the unbiased data model
with `(I_true, W) -> (B_+ I_true, B_+ W)`. Then
`L = (1-P) B_+ S_hat (U_eff^T B_+ S_hat)^+` and **L == 0 IDENTICALLY when B=0**
(proof collapses to `V_35:50^T V_34 = 0`). So L is a direct estimator of the
retrieval bias with a null of exactly zero.

**MISTAKES I MADE — do not repeat:**
1. Claimed a `dS` sensitivity-matrix "gain error in the removal". WRONG: the removal
   is pure software (subtracts exactly the `R[P w]` it just fitted), so there is no
   gain to be in error, and the §4 result holds for whatever P is. Aaron caught it.
2. Concluded "the u-modes cancel at first order, use (1-P)w instead". That holds ONLY
   for an unbiased retrieval — it assumed the error is zero-mean and independent of
   the wavefront measured. A content-dependent fit bias violates exactly that, and
   makes Delta_a FIRST order. Aaron caught it.
3. Quoted the k>6 leakage as the hi/lo *ratio* (up to 0.47) rather than fraction of
   total power (up to 0.18) — overstated.
4. Asserted the 3 iterations make the MIW start-independent — it lands on a fixed-point
   MANIFOLD in one step and retains `G = P F[I_true - I_batoid]`. iter2≈iter3 shows it
   LANDED, not which point. (Moot for differences: both builds share I_batoid.)

**k<=6 truncation (real, sub-dominant).** OFC's `SensitivityMatrix.evaluate` sums all
31 field orders, but the build fits/reconstructs only k=1..6, so the k>6 response of
the estimated state is never subtracted. (a) k<=6 IS sufficient to specify the state
(full rank 50, top-50 v-mode subspace identical); (b) but the k>6 part is not removed
— fix by propagating recovered DOF through the full S (note `S_{k<=6} d == P w`
exactly, so it's purely additive). Build-state leakage 0.022um; 11.5/10.1% of MIW
Z5/Z6, 1.5/2.9% of Z7/Z8. Shape matches MIW COMA (+0.80/+0.64) not astig (+0.19/-0.02),
because coma peaks at k=7,8 (n=3, just above the cut) and astig at k=15,22,23,25 (n=4-6).
MIW power above k=6: Z5 0.96, Z6 0.73, Z7 0.93, Z8 0.88, Z11 0.06 (partly by construction).

**Measurement so far** (`aos/code/analyze_miw_bias_regression.py` ->
`coadd_50_34/miw_bias_regression.pdf`): regress residual MAPS on Delta_a, ridge,
leave-night-out. Delta_a CV R2 **+0.322**, permutation null mean -0.144 -> **p=0.000**.
BUT thermal-only +0.344 and thermal+Delta_a +0.371, so Delta_a adds only **+0.028**
over thermal — largely redundant, so this alone does NOT separate bias from thermal
proxying. Per-mode drop-one OVER THERMAL: u31 +0.014, u25 +0.011, u9 +0.010, u30/u29/
u27/u33 ~+0.009. u9/u13/u14 carry signal with ~zero k>6 leakage — favours bias.

**THE DISCRIMINATOR (pending Aaron's RSP rerun, pushed 48ad283).** Unbiased: `(1-P)W`
lies entirely in the 16 reachable-but-discarded dims u_35..50. Bias: `B_+W` leaves
col(S_hat), so the residual also gets null(S_hat^T) components ∝ Delta_a — impossible
under the unbiased model. `run_coadd_blocks_miw.py` now saves `raw_dz`, `resid_dz`,
`U_eff/U_all/U_disc` (+kj_grid, iZs, sigma, V, norm_weights) in block_grids.npz, and
`block_grids_ext.npz` with the secondary/tertiary maps via `--ext-terms` (Z5-Z8 are all
PRIMARY so the main npz can't show which orders B mixes). Legacy npz keys unchanged.
analyze_miw_bias_regression.py auto-detects and prints a skip message until the rerun.

**PER-VISIT DZ CONSISTENCY (2026-09-03).** `aos/code/analyze_dz_goodness_of_fit.py`
and `analyze_umode_null_coupling.py`. 126 coefficients constrain 34 v-modes, so the
fit is hugely redundant and testable. Split w: a=U_eff^T w (34, um), d=U_disc^T w
(16, um), n_null=U_null^T w (76, um, unreachable by any DOF).
- The `_err` columns in fits.parquet are Huber RLM formal errors ~ scale/sqrt(N_donuts)
  — they assume ~2800 donuts are INDEPENDENT, which turbulence violates. Using them
  gives chi2_50/dof ~ 500 (artifact, RETRACTED).
- Fix = EMPIRICAL covariance Sigma_emp [um^2, 126x126] from within-block
  visit-to-visit scatter, **stable T614 blocks only** (Aaron: T377/T378/T379/T380/T381
  are S-matrix tests that STEP the state within a block; they weren't in this build's
  visit set anyway). 833 residual dof, 82 blocks.
- sigma_emp/sigma_RLM = **8.5** (dimensionless) -> ~39 of 2838 effective independent
  donuts. Adjacent-consecutive-visit-difference covariance gives 7.9 and a drift
  ratio 0.93, so within-block scatter IS stationary noise.
- CALIBRATED chi2_50/dof = **4.2** (block-mean) / 5.1 (adjacent-pair), dof=76;
  chi2_34/dof 5.2 / 7.0, dof=92. Real inconsistency, ~100x smaller than the artifact.
- Unweighted [um]: ||(1-P)w|| 0.111, null part 0.081, discarded 0.075, ||a|| 0.734
  (that 0.734 is the MEAN over visits of the per-visit 34-vector NORM, not an RMS;
  true RMS 0.801). Mode 1 (focus) carries 41.4% of a's variance (a_1 std 0.491 um) —
  open-loop focus drift, so never correlate against total ||a||.
- n_null: static mean vector 0.0395 um (20.5% of its power), varying part 0.0777 um;
  Sigma_emp predicts noise-only 0.0425 um, so ~70% of the varying part is real.
- **KEY RESULT:** ridge (lambda/n ~ 0.11) of n_null [um] on the 34 SIGNED a_m [um],
  leave-night-out CV over 19 nights, n=1126: **CV R^2 = +0.408**; simulated noise null
  (keep each visit's real reachable part, redraw the rest from Sigma_emp) mean +0.014,
  95th +0.021, 0 of 100 sims exceeded -> p < 0.01. Per-mode drop-one CV dR^2: u13
  +0.040, u31 +0.025, u2 +0.020, u21 +0.014, u20 +0.012.
- **dS IS a valid mechanism for THIS test** (unlike for the coadd-MIW residual, where
  the software subtraction is exact): the null space is defined by the TABULATED
  S_hat, so if the real response leaves col(S_hat), genuine states leak in prop to a.
  Discriminator (NOT YET BUILT): a pupil-index bias B is k-INDEPENDENT, so L must
  factor as a 21x21 pupil mixing replicated over the focal orders; dS has no such
  constraint.
- **BUT** see [[miw-focal-order-truncation]]: all of this lives inside k<=6, which is
  only 16.9% of MIW power. The n_null coupling is real but confined to that corner;
  redo on a k=1..30 refit (630 coefficients, 580-dim null) before over-interpreting.

**More mistakes to avoid:** Ledoit-Wolf shrinks toward a SCALED IDENTITY — wrong for
the 126 DZ coefficients (they span ~30x in sigma, 0.00038-0.186 um), it leaked Z5's
variance into Z26's and produced bogus inflation (41-49x), "~2 effective donuts", and
a fake drift ratio 0.23-0.35. Use `_shrink_cov`: keep empirical variances exactly,
shrink only the CORRELATION matrix (lambda ~0.16). Also: I mislabeled a mean-of-norms
as an "RMS", and quoted "29x the null" (a bare R^2 ratio) as if it were a statistic.
See [[reporting-units-standard]].

**Gotchas:** every `d[key]` on an NpzFile re-decompresses the whole array — pull out
once before looping. `run_coadd_blocks_miw.py` needs `--param-set` (required) and
OOMs an RSP pod -> use `aos/run_coadd_blocks_miw.sbatch` (roma, 128G, 12h) from an
s3df node. macOS has no `timeout`. Requiring map cells finite in ALL blocks empties the mask
(per-block coverage differs); use coverage >= 0.97 + column-mean impute.
`recompute_coadd_metrics.py` is now the SINGLE timeseries renderer (`--rebin 1 3`,
one page per factor, `coadd_timeseries.pdf`); native median r 0.660 vs rebin-3 0.739.
Env-only GBT R2 is 0.619 now (not the older 0.68 — camera-temp merge dropped n to 123).
