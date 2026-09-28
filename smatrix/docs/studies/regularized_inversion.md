# Study: `regularized_inversion` — bounding the recovered optical state to the allowed range

> **Status:** current · **Last updated:** 2026-09-28 · **Kind:** reference (study)

> **Code:** `code/regularized_inversion/` · **Output:** `output/regularized_inversion/<data>/`

The Optical Feedback Control (OFC) open-loop recovery inverts a measured Double Zernike
(DZ) wavefront onto degrees of freedom (DOF) by a truncated singular value decomposition
(SVD): the leading 34 of 50 singular modes are kept and the rest discarded. Truncation is
the only regularization in the scheme, and it is a blunt one — a mode is either fully
trusted or fully ignored. The per-DOF normalization weights `w_j = r_j^0.5 f_j^-0.5`, with
`r_j` the allowed range and `f_j` the full width at half maximum (FWHM) response in
arcsec per DOF unit, put the range into the *metric* of the fit but never into the
*feasible set*. Nothing in the inversion prevents a recovered mirror bending-mode
amplitude from exceeding the actuator-force-limited stroke the mirror can physically
reach.

It does exceed it. Inverting the measured median paired-difference wavefront from each
full-array-mode (FAM) bounce leg with the current truncated recovery gives:

| leg | n pairs | DOF over range (of 50) | largest amplitude over range | bending-mode L2 amplitude (µm) |
|---|---|---|---|---|
| elev 40 deg | 22 | 10 | 8.59 | 0.169 |
| elev 60 deg | 4 | 13 | 2.92 | 0.126 |
| elev 50 deg | 6 | 12 | 8.75 | 0.158 |
| elev 30 deg | 5 | 14 | 11.27 | 0.275 |
| elev 75 deg | 6 | 1 | 1.36 | 0.067 |
| rotator 60 deg | 31 | 11 | 2.94 | 0.173 |

Ratios are dimensionless, recovered amplitude over allowed range. The elevation 75 deg leg
is the upward near-null control and is the only one close to feasible. This study asks what
image quality (IQ) a *feasible* solution costs — one in which every recovered DOF sits
inside its range — and compares two ways of getting one.

## The two methods

Both solve for the normalized DOF `x = d / w`, the variable the SVD is taken in, so the
v-mode machinery and `aos_state` remain the sole owner of the decomposition. Method B
penalizes a function of the physical `d = w x` while still fitting in `x`, which is a
change of variables rather than a new basis.

**Method A — damped SVD (Tikhonov).** Minimizes `||dW - S x||^2 + lambda^2 ||x||^2`,
replacing the hard cut at 34 modes with a smooth roll-off of the mode gain,
`s_i / (s_i^2 + lambda^2)`. One scalar knob, `lambda`, in the units of a singular value
(µm of wavefront per unit normalized DOF), so it is read directly against the spectrum.
Because `w_j` already carries `sqrt(r_j / f_j)`, the penalty `||x||^2` is weakly
range-aware: a DOF with a small range is penalized harder per physical unit. It is still
not a constraint.

**Method B — superlinear range penalty.** Minimizes

```
||dW - S x||^2  +  sum_j ( |d_j| / (kappa * r_j) ) ^ (2 p)
```

The penalty is far below unity while `|d_j| / r_j` stays under `kappa` and rises steeply
as the ratio approaches and passes it — small at half the range, large at the range and
beyond. It is solved by iteratively reweighted least squares (IRLS) with the penalty
curvature `q_j = (kappa r_j)^(-2p) |d_j|^(2p-2)` recomputed each iteration.

Two implementation points that are load-bearing:

- **The solve is restricted to the retained mode coefficients**, `x = V_r b`, not carried
  out in `x` directly. The rank-34 operator leaves 16 null directions over 50 DOF, and a
  direct solve fills them with noise: with the penalty switched off, the in-`x` version
  returned a largest `|d_j| / r_j` of 30.2 against the truncated solution's 16.0, when the
  two should agree exactly. Restricting to `V_r` reproduces the truncated solution to
  machine precision in that limit, which is the test the code carries.
- **Each IRLS step is backtracked on the objective.** The reweighting is a local model, and
  for `p >= 3` a fixed-fraction step enters a limit cycle rather than converging: measured
  at `p = 3`, a half-step stalls at a relative change of 1.6e-2 indefinitely and reports a
  largest `|d_j| / r_j` of 0.585 against the converged 0.537. Halving the step until the
  objective decreases makes the iteration monotone; all 180 range-penalty solves reported
  here converge, in 2 to 43 iterations.

## The image-quality metric measures the achieved correction, not the subspace

`run_bounce.py` reports `fwhm_after_50_34` from the subspace projection
`(I - U_eff U_eff^T) dW` — what an *ideal* correction in the kept subspace would leave. It
does not depend on the recovered amplitudes at all, so it is structurally incapable of
seeing a regularizer trade wavefront for amplitude, and it is why the over-range
amplitudes never showed up in the bounce FWHM numbers.

This study scores the **achieved** residual `dW - S (d / w)` instead, which does depend on
`d`. For the unregularized truncated solution the two are identical, and the comparison
script asserts it: the largest difference across the six legs is 1.7e-16 arcsec FWHM. The
`fwhm_before` and truncated FWHM values therefore reproduce the committed
`bounce_fwhm_metric.parquet` exactly — elevation 40 deg at 0.2986 to 0.0381 arcsec,
rotator 60 deg at 0.2082 to 0.0191 arcsec — so the new numbers stay traceable to the
existing bounce products.

## Result: a feasible solution is nearly free, and method B is uniformly better

Per leg, the cheapest setting of each family for which **all 50 DOF sit inside range**,
against the current truncated recovery. FWHM values are the achieved correctable FWHM in
arcsec; the cost is the excess over the truncated FWHM; ratios are dimensionless,
recovered amplitude over allowed range, maximized over all 50 DOF.

| leg | n pairs | FWHM no correction | FWHM truncated | truncated max ratio | FWHM method A | cost A | FWHM method B | cost B | B setting | B max ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| elev 40 deg | 22 | 0.2986 | 0.0381 | 8.59 | 0.0685 | 0.0304 | **0.0414** | **0.0033** | `p=3, kappa=4` | 0.93 |
| elev 60 deg | 4 | 0.1409 | 0.0326 | 2.92 | 0.0386 | 0.0061 | **0.0342** | **0.0017** | `p=3, kappa=5` | 0.92 |
| elev 50 deg | 6 | 0.2516 | 0.0664 | 8.75 | 0.0879 | 0.0215 | **0.0715** | **0.0052** | `p=3, kappa=4` | 0.93 |
| elev 30 deg | 5 | 0.3987 | 0.0598 | 11.27 | 0.0982 | 0.0384 | **0.0783** | **0.0185** | `p=2, kappa=6` | 1.00 |
| elev 75 deg | 6 | 0.0908 | 0.0153 | 1.36 | 0.0151 | −0.0002 | **0.0151** | **−0.0002** | `p=3, kappa=4` | 0.55 |
| rotator 60 deg | 31 | 0.2082 | 0.0191 | 2.94 | 0.0231 | 0.0040 | **0.0203** | **0.0012** | `p=3, kappa=4` | 0.92 |

Three things follow.

**A physically reachable correction costs almost no image quality.** On five of the six
legs method B buys full feasibility for 0.0012 to 0.0052 arcsec of FWHM — 3% to 9% of the
truncated residual, and well under 2% of the uncorrected FWHM the bounce actually
produces. The elevation 30 deg leg is the exception at 0.0185 arcsec, and it is also the
largest throw and the leg with the most extreme over-range excursion. On the near-null
upward elevation 75 deg leg the feasible solution is very slightly *better* than the
truncated one (−0.0002 arcsec), which is what one expects when the discarded amplitude was
noise.

**Method B beats method A on every leg**, by a factor of 2 to 9 in cost. The reason is
structural: method A shrinks every DOF by a function of its singular value alone, so
pulling one badly-behaved bending mode inside range drags all 49 others down with it,
whereas method B's penalty is per-DOF and only engages where the ratio is actually large.
Method A is still worth keeping as the one-knob diagnostic — it needs no range vector and
its `lambda` can replace the arbitrary truncation at 34 — but it is not the method to
adopt for the recovery.

**The knee belongs well past the range, not at it.** `kappa` near 1 over-shrinks: on the
elevation 40 deg leg `kappa = 1` lands at a largest ratio of 0.21, deep inside the
feasible set and at needless FWHM cost, while `kappa = 4` lands at 0.93. The penalty only
has to bound the *largest* amplitude, and a knee at the range itself taxes the other 49
DOF for no gain. Across the legs `p = 3, kappa = 4` is the best or near-best setting on
five of six.

This confirms the interpretation reached from the bounce data alone: the over-range
amplitudes are an ill-conditioning artifact rather than signal. They carry a large share of
the recovered amplitude while producing almost none of the wavefront, so removing them
costs almost nothing.

## Caveat: the rigid-body DOF split is not preserved, though its wavefront is

Regularizing does not leave the recovered DOF vector alone, and the change is not confined
to the bending modes. On the elevation 40 deg leg `M2_dz` goes from −40.95 µm to +41.61 µm
— a sign flip — and `M2_dx` falls from 112.5 µm to 28.2 µm, while `Cam_dx` is stable to 1%
(−436.9 to −432.6 µm). A naive M2-plus-camera sum does not account for it either.

Two measurements bound what this means:

- **The entire DOF change is nearly invisible in the wavefront.** The wavefront produced by
  the difference between the two DOF solutions is 2.0% to 11.2% of the measured wavefront
  (µm of wavefront RMS over µm of wavefront RMS, dimensionless), across the six legs. The
  reshuffling happens almost entirely along near-null directions.
- **The rigid-body wavefront is preserved.** Restricting the forward model to DOF 0–9 and
  comparing the wavefront each solution's rigid-body part produces gives ratios of 1.027
  (elev 40 deg), 1.054 (elev 50 deg) and 1.029 (rotator 60 deg), with elevation 30 deg the
  outlier at 0.861.

So the defensible claim is about the wavefront, not the DOF: the correction a regularized
solution applies is optically equivalent to the truncated one to within a few percent, and
it is physically reachable. **A per-DOF rigid-body amplitude from a regularized fit is a
constrained estimate and should not be quoted as a measurement of hexapod motion** —
which is the same reason the range penalty biases toward zero by construction. For a
look-up-table (LUT) fit that needs the rigid-body amplitudes themselves, the truncated fit
remains the estimator; the regularized fit answers whether a reachable correction exists
and what it would achieve.

## Definition note: which over-range count

The `truncated max ratio` column above maximizes `|d_j| / r_j` over **all 50 DOF**. The
table in [`../../../aos/docs/studies/bounce.md`](../../../aos/docs/studies/bounce.md)
counts only DOF significant at over 3σ, which is a different and smaller set — hence 7.3
there against 8.59 here on the elevation 40 deg leg. Both are correct under their own
definition; neither is a correction of the other.

## Relation to the OIC controller's motion penalty in ts_ofc

`ts_ofc` already carries a per-DOF penalty, in the Optimal Integral Controller
(`ts_ofc/python/lsst/ts/ofc/controllers/oic_controller.py`, `uk` and `authority`). It is
**not** functionally equivalent to the range penalty here, and the difference is not a
matter of tuning. The two agree exactly on the rigid-body range and differ in both the
functional form of the penalty and the bending-mode range.

### What the OIC minimizes

The controller states its cost function as `J = x^T Q x + rho u^T H u`, and builds

```
mat_h = np.diag(authority[dof_idx] ** 2)
mat_f = np.linalg.inv(motion_penalty**2 * mat_h + q_mat)
```

so the penalty term is `rho^2 * sum_j a_j^2 d_j^2` with `rho = ofc_data.motion_penalty`
(dimensionless) and `a_j` the **authority**, which `authority()` assembles as

| block | authority `a_j` |
|---|---|
| rigid body (10 DOF) | `rb_stroke[0] / rb_stroke_j` — a reciprocal stroke, normalized to `M2_dz` |
| M1M3 bending (20) | `m1m3_actuator_penalty * std(rot_mat_j)`, `m1m3_actuator_penalty = 13.2584` |
| M2 bending (20) | `m2_actuator_penalty * std(rot_mat_j)`, `m2_actuator_penalty = 134.0` |

`rot_mat` is force per µm of bending amplitude, with 156 M1M3 and 72 M2 actuators, so
`std(rot_mat_j)` is an actuator-force spread in N per µm and `1/a_j` is a range-like
quantity in the DOF's own unit.

**The penalty is off by default:** `base_ofc_data.py` ships `motion_penalty: float = 0.0`,
and nothing under `policy/` overrides it — the only non-zero values in the package are in
the tests, `1e-4` in `test_ofc.py` and `1e-5` in `test_oic_controller.py`. At `rho = 0` the
`mat_h` term vanishes and `mat_f` is the plain inverse of `q_mat` — the unregularized solve.
Whatever is deployed, the shipped default applies no motion penalty at all. The test values
bracket the low end of the `rho` scan below, where the OIC is still 4 to 11 times over range.

### Where the two agree exactly

The rigid-body block is the *same numbers*. `rb_stroke` is
`[5900, 6700, 6700, 0.12, 0.12, 8700, 7600, 7600, 0.24, 0.24]` (µm for translations,
arcsec for rotations), and `dof_range_vector` back-derives exactly these ten values as
`r_j`. So `1 / a_j`, rescaled so that `M2_dz` matches, reproduces `r_j` to `ratio = 1.000`
for all ten rigid-body DOF. The OIC authority *is* a reciprocal allowed range there.

### Where they differ — the bending-mode range

Both blocks disagree, and the disagreement factorizes exactly:

```
r_j(OIC-implied) / r_j(ours) = PREFACTOR_block * (max_j / std_j)
```

verified to floating-point equality on both blocks. Two independent causes:

1. **Force statistic.** We use `(force_range / 20) / max|force-per-µm|`; the OIC uses
   `std(force-per-µm)`. The ratio `max_j / std_j` is not constant across modes — median
   2.337 (range 1.99 to 4.214) for M1M3, median 1.874 (range 1.44 to 2.26) for M2 — so no
   single constant reconciles them. A peak-force limit and a force-spread limit are
   different physical statements about what an actuator set can do.
2. **Block prefactor.** `rb_stroke[0] * 20 / (actuator_penalty * force_range)` is 66.42 for
   M1M3 and 19.57 for M2 (dimensionless). These encode how much bending the scheme buys per
   unit of hexapod motion, and the two schemes choose differently.

Net effect: the OIC's implied bending range is **132 to 280 times** ours for M1M3 (median
155) and **28 to 44 times** ours for M2 (median 37). Relative to the rigid-body block, the
OIC penalizes bending far more weakly than our `r_j` does — which is consistent with
`m1m3_actuator_penalty` and `m2_actuator_penalty` being tuning knobs rather than derived
stroke limits.

### Where they differ — the functional form, which is the substantive difference

The OIC penalty is **quadratic with fixed curvature**; ours is degree `2p = 6` and
amplitude-dependent. Writing `t_j = |d_j| / r_j`, the effective quadratic curvature that
RBR's IRLS reweighting applies is `q_j = (kappa r_j)^(-2p) |d_j|^(2p-2)`, so in units of
`(kappa r_j)^-2`:

| `t = |d_j| / r_j` | RBR effective weight `(t/kappa)^(2p-2)` | OIC effective weight |
|---|---|---|
| 0.10 | 3.906e-07 | 1 (by construction) |
| 0.25 | 1.526e-05 | 1 |
| 0.50 | 2.441e-04 | 1 |
| 1.00 | 3.906e-03 | 1 |
| 2.00 | 6.250e-02 | 1 |
| 4.00 | 1.000e+00 | 1 |

RBR's weight spans a factor of **2.56e+06** (dimensionless) between `t = 0.1` and
`t = 4.0`; a quadratic penalty is flat by construction. This is the property the study was
built around: a DOF well inside its range should be left nearly free, and only one
approaching the range should be pulled back. A quadratic penalty cannot express that — it
taxes a DOF at 10% of its range in the same proportion as one at 400%.

### Measured consequence on the bounce data

Both were run on the same per-pair Δ wavefronts from every bounce leg, in the same 34-mode
`V_r` subspace, so only the penalty differs. `rho = 1e-3` is the value that brings the OIC
closest to feasibility; residuals are the achieved `||dW − S d||` in µm of wavefront,
median over pairs:

| leg | n pairs | trunc max `|d_j|/r_j` | trunc resid | RBR resid | OIC resid | RBR/trunc | OIC/trunc | OIC/RBR |
|---|---|---|---|---|---|---|---|---|
| `Elev=40` | 22 | 9.160 | 0.0613 | 0.0795 | 0.2089 | 1.30 | 3.41 | 2.63 |
| `Elev=60` | 6 | 6.262 | 0.0598 | 0.0796 | 0.1331 | 1.33 | 2.23 | 1.67 |
| `Elev=50` | 6 | 10.919 | 0.0829 | 0.1093 | 0.1963 | 1.32 | 2.37 | 1.80 |
| `Elev=30` | 8 | 11.100 | 0.0717 | 0.1069 | 0.2734 | 1.49 | 3.81 | 2.56 |
| `Elev=75` | 6 | 2.928 | 0.0403 | 0.0440 | 0.0847 | 1.09 | 2.10 | 1.92 |
| `Rot=60` | 31 | 5.120 | 0.0683 | 0.0760 | 0.2169 | 1.11 | 3.18 | 2.85 |

Residuals are µm of wavefront; `RBR/trunc`, `OIC/trunc` and `OIC/RBR` are dimensionless
residual ratios, regularized over truncated and OIC over RBR. Median inflation over the
truncated residual is **1.31×** for RBR against **2.77×** for the OIC, so the OIC gives up
about **2.24×** more wavefront than RBR for the same nominal range compliance.

Feasibility is also less controllable. RBR lands at max `|d_j|/r_j` of 0.791 to 1.213 across
the six legs at a single `(kappa = 4, p = 3)`. The OIC at a single `rho = 1e-3` scatters over
0.405 to 1.602 — it overshoots the near-null `Elev=75` leg and still leaves `Elev=30` over
range, because a fixed quadratic weight cannot adapt to how far over range a given leg
actually is. Reaching feasibility on the worst leg requires a `rho` that heavily over-damps
the others: at `rho = 1e-2` every leg is at max ratio 0.025 to 0.099, having given up 0.1364
to 0.4182 µm of wavefront — three to six times the truncated residual, to bound amplitudes
that only needed bounding by a factor of a few.

**Conclusion: not functionally equivalent.** A translation table is therefore not
meaningful for the penalty as a whole. The one exact correspondence worth recording is the
rigid-body range, where `a_j = rb_stroke[0] / r_j` and the two schemes coincide.

### An RBR that would fit in ts_ofc

The OIC's structure accommodates RBR with no change to its interface, because the IRLS
reweighting produces exactly the diagonal `mat_h` the controller already builds. The single
change is that `mat_h` becomes iteration-dependent rather than fixed:

```python
def range_authority(self, dof_state, kappa=4.0, power=3):
    """Amplitude-dependent authority reproducing the RBR penalty curvature.

    Returns the diagonal of H for the current state, so that
    ``motion_penalty**2 * H`` equals the IRLS curvature
    ``q_j = (kappa * r_j)**(-2p) * |d_j|**(2p - 2)``.

    Parameters
    ----------
    dof_state : `numpy.ndarray`
        Current DOF state, ``(n_dof,)``, µm and arcsec per DOF unit.
    kappa : `float`, optional
        Dimensionless knee: the penalty reaches unit weight at
        ``|d_j| = kappa * r_j``.
    power : `int`, optional
        Penalty exponent ``p``; penalty goes as ``|d_j|**(2p)``.

    Returns
    -------
    a : `numpy.ndarray`
        Authority ``(n_dof,)`` such that ``H = diag(a**2)``, in inverse DOF
        units, for use in place of `authority`.
    """
    r_j = self.dof_range()          # rb_stroke, then force-limited bending stroke
    d = np.abs(np.asarray(dof_state, float)[self.ofc_data.dof_idx])
    d = np.maximum(d, 1e-12)        # floor, DOF units, avoids 0**negative
    q = (kappa * r_j) ** (-2 * power) * d ** (2 * power - 2)
    return np.sqrt(q) / max(self.ofc_data.motion_penalty, 1e-30)
```

Three further changes are needed for this to be correct rather than merely type-compatible:

1. **`dof_range()` must return a stroke, not an authority.** The rigid-body entries are
   `rb_stroke` directly. The bending entries need the force-limited stroke
   `(force_range / per_mode_budget) / max|force-per-µm|`, which is a different statistic
   from the `std(rot_mat)` the current `authority()` uses — see above. `m1m3_force_range`
   134 N and `m2_force_range` 45.0 N with `per_mode_budget = 20` are the values used here.
2. **Iterate.** One `mat_f` solve is no longer enough: `q` depends on `d`, which depends on
   the solve. The study's solver needs a backtracking line search on the true objective to
   converge at `p = 3` — a fixed half-step enters a limit cycle. Since the OIC is already an
   *integral* controller taking small steps toward a target, an alternative is to recompute
   `mat_h` once per control step from the current `dof_state` and let the closed loop supply
   the iteration. That is cheaper and fits the existing control flow, but it is not the same
   fixed point and would need its own convergence check on sky.
3. **`motion_penalty` must become non-zero.** It ships at 0.0, which disables the term. With
   the RBR curvature folded into `H` the natural choice is `rho = 1`, leaving `kappa` as the
   single knob.

The cheaper alternative, if an amplitude-dependent `H` is unwelcome in the controller, is to
keep the quadratic form and simply **fix the bending-mode range** — replace
`std(rot_mat)`-based authority with `1 / r_j` from the force-limited stroke, making the
rigid-body and bending blocks consistent. That does not buy RBR's selectivity, and on this
data it costs about 1.93× more wavefront than RBR at matched compliance, but it is a
one-line change to `authority()` and removes the 28–280× inconsistency between blocks.

## Code

| file | role |
|---|---|
| `regularized_inversion.py` | library: the rank-limited forward operator, the allowed-range vector, and the three inversions (truncated, damped, range penalty), plus the achieved residual |
| `run_regularized_compare.py` | sweeps `lambda`, `kappa` and `p` over every bounce leg, scores IQ and feasibility, and writes the tables and the comparison PDF |
| `run_oic_compare.py` | compares the ts_ofc OIC controller's quadratic motion penalty against the range penalty — the authority-versus-`r_j` decomposition, the penalty-curvature table, and both inversions on the same per-pair bounce Δ. Prints only; writes no output files |

The comparison script reuses the bounce leg definitions and pairing from
`aos/code/bounce/`, so the median paired-difference Δ it scores is the same quantity the
committed bounce products report rather than a restatement of it.

## Outputs

`output/regularized_inversion/danish_1_2_A_50_34_i_5rot_july/`, from the
measured-intrinsic-wavefront-referenced fit over all six bounce legs:

| product | content |
|---|---|
| `regularized_inversion_metrics.parquet` | one row per (leg, method setting), 246 rows: `fwhm_before_arcsec`, `fwhm_projection_arcsec` and `fwhm_achieved_arcsec` in arcsec FWHM, `residual_rms_um` in µm of wavefront, `max_dof_over_range` and `n_dof_over_range` (dimensionless, amplitude over range), `bending_l2_um` in µm of mode amplitude, the setting knobs `lam` / `kappa` / `power`, and `irls_iter` / `irls_converged` |
| `regularized_inversion_dof.parquet` | per-DOF recovered amplitude, unit, range and ratio for every (leg, method setting) |
| `regularized_inversion_best_feasible.parquet` | the cheapest feasible setting per (leg, family) with its FWHM cost in arcsec |
| `regularized_inversion_compare.pdf` | the IQ-against-feasibility trade-off curve, the per-leg bar comparison, and one per-DOF-against-range page per leg |

## Running

```bash
cd ~/notebooks/rubin-work/smatrix/code/regularized_inversion
python run_regularized_compare.py \
  --fits /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet \
  --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
  --mi-name pathA_50_34_i_5rot \
  --out-dir /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/smatrix/output/regularized_inversion/danish_1_2_A_50_34_i_5rot_july \
  --min-detectors 160 \
  --kappas 1 1.5 2 3 4 5 6 8 11 16 --powers 2 3 4 \
  --lambdas 0.01 0.015 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3
```

Needs `lsst.ts.ofc` and `lsst.ts.wep`, so it is RSP/USDF-only.

The OIC comparison takes the same fit table and leg definitions:

```bash
cd ~/notebooks/rubin-work/smatrix/code/regularized_inversion
python run_oic_compare.py \
  --fits /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet \
  --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
  --mi-name pathA_50_34_i_5rot \
  --min-detectors 160
```

## Outstanding

- The knee setting is chosen per leg here. A single `(p, kappa)` for operational use has
  not been fixed; `p = 3, kappa = 4` is the candidate on this evidence.
- Only the median paired Δ per leg is inverted. Propagating the per-pair scatter through
  the regularized fit would give an error on the constrained amplitudes, which the
  penalty's bias makes non-trivial to interpret.
- A hard-bounded solve (`scipy.optimize.lsq_linear` with `bounds=(-r, r)`) would give the
  best achievable feasible wavefront as a reference floor, against which the penalty's
  smoothness costs something. Not run.
- The OIC comparison scores an open-loop single-shot inversion on both sides. The OIC is an
  *integral* controller taking a gain-limited step per iteration, so its penalty acts over a
  loop rather than in one solve. Whether the closed loop recovers some of the 2.24×
  wavefront gap is not measured here, and it is the question to settle before proposing a
  ts_ofc change.
- Which bending-mode stroke is right — a peak-force limit `max|force-per-µm|` as used here,
  or the force spread `std(force-per-µm)` the OIC uses — is an engineering question about the
  actuator sets, not a modelling preference. It sets an 28 to 280× factor on the allowed
  bending range and should be confirmed with the M1M3 and M2 groups.

## See also

- [`vmode.md`](vmode.md) — the SVD whose truncation this study softens, and the
  conditioning ranking that predicts which DOF go over range
- [`../vmode_normalization.md`](../vmode_normalization.md) — where `r_j` and `f_j` come from
- [`../../../aos/docs/studies/bounce.md`](../../../aos/docs/studies/bounce.md) — the bounce
  data inverted here, and the over-range finding that motivated the study
