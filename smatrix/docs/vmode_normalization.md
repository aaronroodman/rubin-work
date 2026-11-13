# V-mode normalization: default vs geom_mean (why v1 differs)

> **Status:** current · **Last updated:** 2026-11-13 · **Kind:** reference (normalization conventions)

**Date:** 2026-07-08
**Author:** Aaron Roodman (analysis with Claude)
**Context:** computing v-modes from the AOS DOF trim in
`blocks/t539_closedloop_aos.ipynb`, cross-checked against `olr` and
`aos/smatrix_vmode_info.ipynb`.

## Summary (corrected)

The v-modes are the right singular vectors of the **normalized** sensitivity
matrix `Ã = A · diag(n)`, where `n_j` is the per-DOF normalization weight. The
choice of `n_j` changes the SVD, so it changes **which physical mode is v1** and
the numerical scale of the v-mode amplitudes.

**The canonical normalization is geom_mean `n_j = r_j^0.5 · f_j^-0.5`, and it is
already stored in the OFC config's `normalization_weights`** (v13). Read it
directly — do NOT recompute it. The confusing part is *how you obtain it*:

| How you get `n_j` | Result | v1 is… | Notes |
|-------------------|--------|--------|-------|
| `OFCData('lsst', config_dir=v13).normalization_weights` | **official geom** (`r^0.5 f^-0.5`, field-averaged FWHM) | **focus** | ✅ correct — same weights `build_ofc_svd`/bounce use via the ts yaml |
| `OFCData()` **without** `config_dir` | old non-geom default | **M2 tilt** | ❌ wrong — this is the OLR / nightly_report bug (v1 ~1e-4 at convergence) |
| `sqrt(r/f)` with the **corner-point** f_j of `compute_normalization_components` | geom with the wrong f_j | focus | ⚠️ corner-point f_j is about 2× the field-averaged f_j, so the weights are ~√2 too small (dimensionless ratio) |

So there is really only **one** intended normalization (geom_mean); the
apparent "conventions" are just three ways of computing it, two of which are
wrong. The fix everywhere is: pass `config_dir` and use the config's
`normalization_weights` as-is.

### The `range0.5_fwhm-0.15.yaml` filename typo

The ts_config_mttcs normalization file `build_ofc_svd` defaults to is named
`range0.5_fwhm-0.15.yaml`, but its **header declares `alpha=0.5, beta=-0.5`** and
its values are exactly the v13 `normalization_weights`. The `-0.15` in the
filename is a typo for `-0.5`. So `build_ofc_svd` (bounce and all the
`run_*` scripts) has been using the correct geom all along.

### Why my manual `sqrt(r/f)` was ~√2 off

`compute_normalization_components` returned the **corner-point** FWHM `f`, while
the config yaml was generated with the **field-averaged (quadrature)** FWHM ("a
circular quadrature grid over the field of view", per the yaml header). The two
differ by ≈ 2× in `f` (so ≈ √2 in `n = r^0.5 f^-0.5`), and non-uniformly for the
field-dependent bending modes. Same `r`, different `f`. Lesson: use the config's
stored weights, not a hand-rolled `sqrt(r/f)`.

### What was actually wrong

`StateEstimator.get_vmodes_from_dofs` and the olr `vmodes` column
(`nightly_table.py`), `nightly_tablemaker`, `aos_openloop`, `olr_quicklook`, and
the GitHub `nightly_report` all build `OFCData()` **without `config_dir`**, so
they pick up the old non-geom default → v1 = M2-tilt → ~1e-4 at convergence.
That is the bug to fix (pass `config_dir`), not the exponent.

## Evidence

Singular-value spectra for the standard_22 / 12-mode scheme, from `build_ofc_svd` on the
DZ slab (focal k = 1..6, pupil Z4..Z26 without Z20, Z21), dimensionless:

```
default : s[0:12] = [121.4, 121.4, 46.68, 46.68, 3.401, 3.400, 3.332, 3.331, 2.057, 1.712, 1.700, 0.202]
geom    : s[0:12] = [17.43, 6.377, 6.377, 3.637, 3.637, 3.346, 3.339, 1.264, 1.262, 1.044, 1.027, 0.167]
```

`default` is `default.yaml`; `geom` is `range0.5_fwhm-0.15.yaml`, the shipped geom_mean.

- **default:** s[0] = s[1] = 121.4 is a *degenerate pair* → the top mode is the
  M2-tilt (rx/ry) doublet, not focus. V[:,0] is dominated by M2_rx (−0.959) and
  M2_ry (−0.231); M2_dz and Cam_dz are ~0.
- **geom:** s[0] = 17.43 is *non-degenerate* (focus is a single mode), followed
  by the degenerate pair s[1] = s[2] = 6.377. V[:,0] is
  **M2_dz +0.628, Cam_dz +0.777** — i.e. focus.

Why the default v1 is ~0 at convergence: a high-gain mode produces a large
wavefront per unit amplitude, so the closed loop nulls its residual almost
perfectly. The tilt pair is the highest-gain mode under default weights, so its
residual (default v1) is ~1e-4. Under geom weights the gain spread is smaller
(s[0]/s[11] = 104 against 601, dimensionless), and v1 = focus carries a real ~0.1
residual.

## Worked example — geom v1 for 20260420 seq 13

`v1 = Σ_i V[i,0] · d_i / w_i`, with the shipped geom weights `w_j = r_j^0.5 f_j^-0.5`
(`build_ofc_svd`, standard_22 / 12 modes; Trim `d_i` from the value-added database):

| DOF | trim d_i [µm] | w_i [µm / arcsec^0.5] | d/w [arcsec^0.5] | V[i,0] (dimensionless) | term |
|-----|---------------|-----------------------|------------------|------------------------|------|
| M2_dz   | +429.333 | 713.9  | +0.6014 | +0.6284 | +0.3779 |
| Cam_dz  | −323.603 | 851.0  | −0.3802 | +0.7772 | −0.2955 |
| M1M3_b3 | −0.0948  | 0.2831 | −0.3349 | −0.0231 | +0.0077 |
| M2_b5   | −0.0712  | 0.2572 | −0.2768 | −0.0216 | +0.0060 |

`v1 = 0.0959` (dimensionless). The focus part is the near-degenerate M2_dz / Cam_dz
pair: `+0.378 (M2_dz) − 0.296 (Cam_dz) = +0.082`, i.e. the loop applies large,
opposite-sign dz on M2 (+429 µm) and camera (−324 µm) — the two produce nearly
the same focus signal — leaving a modest net focus residual. Under the default
normalization v1 is the tilt mode, ~0 at convergence.

## r_j and f_j — definitions and storage

`n_j` is built from two per-DOF components:

- **r_j — range weight.** The physical range (stroke) of each DOF: hexapod
  stroke, or bending-mode force range, divided by the max element of the DOF's
  rotation matrix. Sourced from the OFC configuration
  (`ts_config_mttcs/MTAOS/v13/ofc`, referenced via `ofc_config_dir`). Unit-bearing
  (µm for hexapod position, deg for hexapod tilt, µm for bending-mode amplitude;
  the tilt unit is settled in `aos/docs/telemetry.md` §7).
- **f_j — FWHM sensitivity weight.** The PSF FWHM (arcsec) a unit of each DOF
  produces, averaged over the field: `convertZernikesToPsfWidth` applied to the
  DOF's Zernike response (pupil Z4..Z22) on a 5-ring, 6-spoke quadrature grid out to
  1.75 deg field radius, combined in quadrature. Larger f_j = more leverage on image
  quality. This field-averaged f_j is the one the shipped weights were made from;
  `rubinwork.smatrix.normalization_weights.compute_f_quadrature` computes it.

Normalization modes (from `smatrix_vmode_info`, `norm_mode`):

```
default   : stored ofc.normalization_weights          (NOT unit-invariant)
r_only    : n_j = r_j
inv_f     : n_j = 1 / f_j
rf        : n_j = r_j * f_j
geom_mean : n_j = sqrt(r_j / f_j)                      (unit-invariant; recommended)
tunable   : n_j = r_j**a * f_j**(a-1)                  (a=0.5 -> geom_mean)
```

`geom_mean` is unit-invariant (the r_j units cancel against f_j under the square
root in the ratio that enters `Ã`), which is why it is the recommended scheme
(see `aos/normalization_study.ipynb`).

### Extract the values

```python
import numpy as np
from lsst.ts.intrinsic.wavefront.ofc_svd import load_normalization_weights
from lsst.ts.ofc import OFCData
from rubinwork.smatrix.normalization_weights import compute_f_quadrature

w = load_normalization_weights(None, 'range0.5_fwhm-0.15.yaml')     # shipped geom weights
sens = np.asarray(OFCData('lsst').sensitivity_matrix)
f = compute_f_quadrature(sens, rings=5, spokes=6, znmin=4, znmax=22)  # arcsec / unit
r = w**2 * f                                                          # unit
```

`smatrix/vmode/code/plot_vmode_dof_matrix.py` writes the same table as
`dof_normalization_<scheme>.pdf`.

### Values (all 50 DOF)

The recovered `r_j` reproduce the stroke limits (M2_dz 5900 µm, M2_rx 0.12 deg), which
confirms the field-averaged `f_j`.

| index | DOF | unit | r_j [unit] | f_j [arcsec / unit] | w_j [unit / arcsec^0.5] |
|------:|-----|------|-----------:|--------------------:|------------------------:|
| 0 | M2_dz | µm | 5900 | 0.01158 | 713.9 |
| 1 | M2_dx | µm | 6700 | 0.000596 | 3353 |
| 2 | M2_dy | µm | 6700 | 0.000596 | 3353 |
| 3 | M2_rx | deg | 0.12 | 68.03 | 0.042 |
| 4 | M2_ry | deg | 0.12 | 68.03 | 0.042 |
| 5 | Cam_dz | µm | 8700 | 0.01201 | 851 |
| 6 | Cam_dx | µm | 7600 | 0.0001611 | 6869 |
| 7 | Cam_dy | µm | 7600 | 0.0001611 | 6869 |
| 8 | Cam_rx | deg | 0.24 | 37.64 | 0.07985 |
| 9 | Cam_ry | deg | 0.24 | 37.64 | 0.07985 |
| 10 | M1M3_b1 | µm | 0.4541 | 0.7504 | 0.7779 |
| 11 | M1M3_b2 | µm | 0.4524 | 0.7497 | 0.7768 |
| 12 | M1M3_b3 | µm | 0.08771 | 1.094 | 0.2831 |
| 13 | M1M3_b4 | µm | 0.06697 | 1.138 | 0.2425 |
| 14 | M1M3_b5 | µm | 0.06603 | 1.154 | 0.2392 |
| 15 | M1M3_b6 | µm | 0.02352 | 1.386 | 0.1303 |
| 16 | M1M3_b7 | µm | 0.02127 | 1.382 | 0.1241 |
| 17 | M1M3_b8 | µm | 0.02206 | 1.216 | 0.1347 |
| 18 | M1M3_b9 | µm | 0.01947 | 1.219 | 0.1264 |
| 19 | M1M3_b10 | µm | 0.01376 | 1.706 | 0.08982 |
| 20 | M1M3_b11 | µm | 0.0132 | 1.7 | 0.08812 |
| 21 | M1M3_b12 | µm | 0.009088 | 0.89 | 0.1011 |
| 22 | M1M3_b13 | µm | 0.009297 | 1.444 | 0.08025 |
| 23 | M1M3_b14 | µm | 0.0093 | 1.423 | 0.08084 |
| 24 | M1M3_b15 | µm | 0.004087 | 0.8109 | 0.07099 |
| 25 | M1M3_b16 | µm | 0.004447 | 0.8085 | 0.07416 |
| 26 | M1M3_b17 | µm | 0.005905 | 2.051 | 0.05366 |
| 27 | M1M3_b18 | µm | 0.00599 | 1.984 | 0.05494 |
| 28 | M1M3_b19 | µm | 0.004873 | 0.2182 | 0.1494 |
| 29 | M1M3_b20 | µm | 0.002206 | 1.405 | 0.03962 |
| 30 | M2_b1 | µm | 0.9637 | 0.5252 | 1.355 |
| 31 | M2_b2 | µm | 0.968 | 0.5252 | 1.358 |
| 32 | M2_b3 | µm | 0.1369 | 0.7633 | 0.4235 |
| 33 | M2_b4 | µm | 0.1252 | 0.787 | 0.3988 |
| 34 | M2_b5 | µm | 0.0743 | 1.124 | 0.2572 |
| 35 | M2_b6 | µm | 0.03063 | 1.301 | 0.1534 |
| 36 | M2_b7 | µm | 0.03103 | 1.301 | 0.1544 |
| 37 | M2_b8 | µm | 0.03596 | 0.9475 | 0.1948 |
| 38 | M2_b9 | µm | 0.03584 | 0.9475 | 0.1945 |
| 39 | M2_b10 | µm | 0.01699 | 1.404 | 0.11 |
| 40 | M2_b11 | µm | 0.01689 | 1.404 | 0.1097 |
| 41 | M2_b12 | µm | 0.01447 | 1.133 | 0.113 |
| 42 | M2_b13 | µm | 0.01467 | 1.133 | 0.1138 |
| 43 | M2_b14 | µm | 0.008744 | 1.574 | 0.07454 |
| 44 | M2_b15 | µm | 0.007361 | 1.593 | 0.06797 |
| 45 | M2_b16 | µm | 0.00681 | 0.5165 | 0.1148 |
| 46 | M2_b17 | µm | 0.007098 | 0.5216 | 0.1167 |
| 47 | M2_b18 | µm | 0.002466 | 2.405 | 0.03202 |
| 48 | M2_b19 | µm | 0.001699 | 2.282 | 0.02728 |
| 49 | M2_b20 | µm | 0.001665 | 2.282 | 0.02701 |

**Reading the table — why geom reorders the modes.** The rigid-body DOFs split
sharply:
- **Focus (M2_dz, Cam_dz):** large stroke `r_j` (5900, 8700 µm) but small FWHM
  leverage `f_j` (0.0116, 0.0120 arcsec/µm) → large weight (714, 851 µm/arcsec^0.5).
- **Decenter (M2_dx/dy, Cam_dx/dy):** large stroke (6700, 7600 µm), even smaller
  `f_j` (6.0e-4, 1.6e-4 arcsec/µm) → the largest weights (3353, 6869 µm/arcsec^0.5).
- **Tilt (M2_rx/ry, Cam_rx/ry):** small stroke `r_j` (0.12, 0.24 deg) but large
  `f_j` (68.0, 37.6 arcsec/deg) → small weight (0.042, 0.080 deg/arcsec^0.5).

So geom up-weights the large-stroke / low-leverage DOFs (focus, decenter) and
down-weights tilt. Combined with the sensitivity matrix `A`, this puts focus at
the top of the SVD (v1) — whereas the default weights leave tilt dominant.

> **Note — default weights are config-dependent.** The `w_j` column above is the
> shipped geom set. A bare `OFCData()` (no `config_dir`, as
> `StateEstimator.get_vmodes_from_dofs` uses in the olr `vmodes` path) loads a
> different set (e.g. M2_dz ≈ 68 vs 714 µm / arcsec^0.5 here) — another reason the default
> normalization is not a stable basis for analysis. The geom weights are derived
> from `r_j`/`f_j` and so are reproducible given the config.

## Degenerate v-modes and the canonical engine (StateEstimator)

Even with identical geom weights, two independent SVDs (numpy vs the OFC
`StateEstimator`) do **not** give the same individual v-modes, because several
singular values are **exactly degenerate** (x/y symmetry: astigmatism, decenter,
tilt doublets). Within a degenerate 2-D subspace the singular vectors are
arbitrary, so `v2` vs `v3` (etc.) depend on which rotation the SVD happens to
pick. Verified: modes with equal `s` gave `|cos(V_se, V_manual)| ≈ 0.43`, while
non-degenerate modes matched at `≈ 0.9995`.

Consequences:
- **Only `v1` (focus, non-degenerate) and the per-pair magnitudes**
  (e.g. `sqrt(v2² + v3²)`) are physically unique. Individual degenerate v-modes
  are not — do not over-interpret `v2`, `v3`, `v4`, `v5`, `v8`, `v9` singly.
- For the *code* to agree, every path must use the **same `Vh`**. We standardize
  on the OFC `StateEstimator` (`get_vmodes_from_dofs`), whose math is exactly
  `inv(N) @ dof[idx] @ Vh[:trunc].T` = `V[:,:k].T @ (dof/w)` with `N = diag(w)`
  the geom weights. `aos/code/aos_state.py` provides `make_state_estimator` +
  `vmodes_from_dofs`; OLR `nightly_table.py`, `t539`, and the nightly-report PR
  all call it, so their v-modes are bit-identical.

The corner-evaluated decomposition, now `aos_state.corner_recovery_basis`, is used
only to **invert** a measured corner wavefront — it is not a v-mode basis.
`StateEstimator.Vh` cannot do that job: built from the unselected 899-row Double
Zernike slab, it does not span the 84-row corner problem and leaves an irreducible
2.3e-02 µm wavefront-residual floor even on noiseless data, where the
corner-evaluated SVD closes to 2.1e-14 µm. `aos_state.recover_optical_state` is
therefore a hybrid: it inverts in the corner basis and reports v-modes through
`get_vmodes_from_dofs`, so no v-mode leaves the sanctioned basis.

## Which to use

**Always geom_mean, and get it from the OFC config's `normalization_weights`
(v13) via `config_dir`.** There is no reason to use the old `OFCData()`-default
weights or a hand-rolled `sqrt(r/f)`. In code, use the shared helpers in
`rubinwork.aos_state` (`make_state_estimator` + `vmodes_from_dofs`), which both
`olr/code/nightly_table.py` and `blocks/t539_closedloop_aos.ipynb` now call, so
the normalization stays in one place — `make_state_estimator` asserts the resolved
yaml and raises rather than silently correcting it. The `build_ofc_svd` run-scripts
(bounce, etc.) already get the same geom via the ts_config_mttcs yaml.

`build_geom_svd` and `project_dofs_to_vmodes` were removed in September 2026; so was
`nightly_table.py`'s local `build_sensitivity_svd`, which duplicated the first.

**Do not** use `OFCData()` without `config_dir` (old non-geom default → v1=tilt)
or recompute `sqrt(compute_normalization_components r/f)` (corner-FWHM, ~√2 off).
