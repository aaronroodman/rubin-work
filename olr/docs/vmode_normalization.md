# V-mode normalization: default vs geom_mean (why v1 differs)

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
| `sqrt(compute_normalization_components r/f)` | geom but with **corner-point FWHM** | focus | ⚠️ ~√2 off from the official field-averaged geom (my earlier t539 mistake) |

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

Singular-value spectra for the standard_22 / 12-mode scheme:

```
default : s[0:12] = [2161, 2161, 63.4, 63.4, 3.95, 2.67, 2.66, 0.58, 0.57, 0.11, 0.10, 0.06]
geom    : s[0:12] = [24.66, 10.69, 10.69, 6.00, 6.00, 4.72, 4.70, 1.94, 1.94, 1.48, 1.46, 0.27]
```

- **default:** s[0] = s[1] = 2161 is a *degenerate pair* → the top mode is the
  M2-tilt (rx/ry) doublet, not focus. V[:,0] is dominated by M2_rx (0.66),
  M2_ry (0.75); M2_dz and Cam_dz are ~0.
- **geom:** s[0] = 24.66 is *non-degenerate* (focus is a single mode), followed
  by the degenerate tilt pair s[1] = s[2] = 10.69. V[:,0] is
  **M2_dz +0.628, Cam_dz +0.777** — i.e. focus.

Why the default v1 is ~0 at convergence: a high-gain mode (s = 2161) produces a
huge wavefront per unit amplitude, so the closed loop nulls its residual almost
perfectly. The tilt pair is the highest-gain mode under default weights, so its
residual (default v1) is ~1e-4. Under geom weights the gain spread is far
smaller (24.66 → 0.27, ~90:1 vs ~36000:1), and v1 = focus carries a real ~0.1
residual.

## Worked example — geom v1 for 20260420 seq 13

`v1 = Σ_i V[i,0] · d_i / w_i`, with `w_j = sqrt(r_j/f_j)` (geom):

| DOF | trim d_i (µm) | w_i | d/w | V[i,0] | term |
|-----|---------------|-----|-----|--------|------|
| M2_dz  | +429.333 | 504.522 | +0.8510 | +0.6284 | +0.5348 |
| Cam_dz | −323.603 | 601.464 | −0.5380 | +0.7772 | −0.4182 |
| M1M3_b3 | −0.095 | 0.202 | −0.4702 | −0.0229 | +0.0108 |
| M2_b5   | −0.071 | 0.182 | −0.3909 | −0.0215 | +0.0084 |

`v1 = 0.1355`. The focus part is the near-degenerate M2_dz / Cam_dz pair:
`+0.535 (M2_dz) − 0.418 (Cam_dz) = +0.117`, i.e. the loop applies large,
opposite-sign dz on M2 (+429 µm) and camera (−324 µm) — the two produce nearly
the same focus signal — leaving a modest net focus residual. Under the default
normalization the same trim gives v1 ≈ 0.0002 (because default v1 = tilt, ~0).

## r_j and f_j — definitions and storage

`n_j` is built from two per-DOF components returned by
`ofc_svd.compute_normalization_components(ofc, sens, field_angles)`:

- **r_j — range weight.** The physical range (stroke) of each DOF: hexapod
  stroke, or bending-mode force range, divided by the max element of the DOF's
  rotation matrix. Sourced from the OFC configuration
  (`ts_config_mttcs/MTAOS/v13/ofc`, referenced via `ofc_config_dir`). Unit-bearing
  (µm for hexapod position, arcsec for tilt, N or dimensionless for bending).
- **f_j — FWHM sensitivity weight.** How much PSF FWHM (arcsec) a unit of each
  DOF produces, computed from the sensitivity matrix
  (`convertZernikesToPsfWidth` applied to the DOF's Zernike response). Larger f_j
  = the DOF has more leverage on image quality.

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
from lsst.ts.intrinsic.wavefront.ofc_svd import compute_normalization_components
from lsst.ts.ofc import OFCData, SensitivityMatrix

ofc_config_dir = '/home/r/roodman/u/LSST/packages/ts_config_mttcs/MTAOS/v13/ofc'
zn = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26])
sensors = ['R00_SW0','R04_SW0','R40_SW0','R44_SW0']
labels = ['M2_dz','M2_dx','M2_dy','M2_rx','M2_ry','Cam_dz','Cam_dx','Cam_dy',
          'Cam_rx','Cam_ry'] + [f'M1M3_b{i}' for i in range(1,21)] + \
         [f'M2_b{i}' for i in range(1,21)]

ofc = OFCData('lsst', config_dir=ofc_config_dir); ofc.zn_selected = zn
fa = [ofc.sample_points[s] for s in sensors]
r, f = compute_normalization_components(ofc, SensitivityMatrix(ofc), fa)
default = ofc.normalization_weights
geom = np.sqrt(r / f)
print(f"{'DOF':9s} {'r_j':>12s} {'f_j':>12s} {'geom sqrt(r/f)':>15s} {'default':>12s}")
for lab, ri, fi, gi, di in zip(labels, r, f, geom, default):
    print(f"{lab:9s} {ri:12.4g} {fi:12.4g} {gi:15.4g} {di:12.4g}")
```

### Values (all 50 DOF, v13 config)

From the extraction snippet against `ts_config_mttcs/MTAOS/v13/ofc`. Hexapod
positions in µm, tilts in arcsec, bending modes dimensionless; f_j in arcsec
FWHM per DOF unit.

**Column meaning (see corrected Summary):** the `default` column is the v13
stored `normalization_weights` — this **is the official geom_mean** (`r^0.5
f^-0.5`, field-averaged FWHM), the one to use. The `geom sqrt(r/f)` column is the
hand-computed `sqrt(r_j/f_j)` using `compute_normalization_components`'
corner-point `f_j`; it is ~√2 smaller than the official geom and should **not**
be used. (The `f_j` column shown is therefore the corner-point FWHM; the
field-averaged FWHM implied by the config is ≈ `f_j / 2`.)

| DOF | r_j | f_j | geom sqrt(r/f) | default |
|-----|-----|-----|----------------|---------|
| M2_dz  | 5900 | 0.02318 | 504.5 | 713.9 |
| M2_dx  | 6700 | 0.001294 | 2275 | 3353 |
| M2_dy  | 6700 | 0.001294 | 2275 | 3353 |
| M2_rx  | 0.12 | 148.6 | 0.02842 | 0.042 |
| M2_ry  | 0.12 | 148.6 | 0.02842 | 0.042 |
| Cam_dz | 8700 | 0.02405 | 601.5 | 851 |
| Cam_dx | 7600 | 0.0003458 | 4688 | 6869 |
| Cam_dy | 7600 | 0.0003459 | 4688 | 6869 |
| Cam_rx | 0.24 | 102.4 | 0.04842 | 0.07985 |
| Cam_ry | 0.24 | 102.4 | 0.04842 | 0.07985 |
| M1M3_b1  | 0.4541  | 1.491  | 0.5519  | 0.7779 |
| M1M3_b2  | 0.4524  | 1.492  | 0.5507  | 0.7768 |
| M1M3_b3  | 0.08771 | 2.157  | 0.2016  | 0.2831 |
| M1M3_b4  | 0.06697 | 2.254  | 0.1724  | 0.2425 |
| M1M3_b5  | 0.06603 | 2.286  | 0.17    | 0.2392 |
| M1M3_b6  | 0.02352 | 2.691  | 0.09349 | 0.1303 |
| M1M3_b7  | 0.02127 | 2.686  | 0.089   | 0.1241 |
| M1M3_b8  | 0.02206 | 2.734  | 0.08983 | 0.1347 |
| M1M3_b9  | 0.01947 | 2.772  | 0.08382 | 0.1264 |
| M1M3_b10 | 0.01376 | 3.185  | 0.06574 | 0.08982 |
| M1M3_b11 | 0.0132  | 3.824  | 0.05875 | 0.08812 |
| M1M3_b12 | 0.009088 | 2.2   | 0.06426 | 0.1011 |
| M1M3_b13 | 0.009297 | 0.6449 | 0.1201 | 0.08025 |
| M1M3_b14 | 0.0093  | 0.6272 | 0.1218  | 0.08084 |
| M1M3_b15 | 0.004087 | 2.272 | 0.04241 | 0.07099 |
| M1M3_b16 | 0.004447 | 2.285 | 0.04411 | 0.07416 |
| M1M3_b17 | 0.005905 | 3.967 | 0.03858 | 0.05366 |
| M1M3_b18 | 0.00599 | 3.926  | 0.03906 | 0.05494 |
| M1M3_b19 | 0.004873 | 0.5305 | 0.09584 | 0.1494 |
| M1M3_b20 | 0.002206 | 3.461 | 0.02524 | 0.03962 |
| M2_b1  | 0.9637  | 1.048 | 0.9591  | 1.355 |
| M2_b2  | 0.968   | 1.046 | 0.9621  | 1.358 |
| M2_b3  | 0.1369  | 1.532 | 0.2989  | 0.4235 |
| M2_b4  | 0.1252  | 1.58  | 0.2815  | 0.3988 |
| M2_b5  | 0.0743  | 2.241 | 0.1821  | 0.2572 |
| M2_b6  | 0.03063 | 2.611 | 0.1083  | 0.1534 |
| M2_b7  | 0.03103 | 2.611 | 0.109   | 0.1544 |
| M2_b8  | 0.03596 | 2.156 | 0.1291  | 0.1948 |
| M2_b9  | 0.03584 | 2.154 | 0.129   | 0.1945 |
| M2_b10 | 0.01699 | 3.431 | 0.07038 | 0.11 |
| M2_b11 | 0.01689 | 2.854 | 0.07694 | 0.1097 |
| M2_b12 | 0.01447 | 1.199 | 0.1098  | 0.113 |
| M2_b13 | 0.01467 | 1.201 | 0.1106  | 0.1138 |
| M2_b14 | 0.008744 | 3.55 | 0.04963 | 0.07454 |
| M2_b15 | 0.007361 | 3.662 | 0.04484 | 0.06797 |
| M2_b16 | 0.00681 | 0.5553 | 0.1107 | 0.1148 |
| M2_b17 | 0.007098 | 0.4869 | 0.1207 | 0.1167 |
| M2_b18 | 0.002466 | 4.952 | 0.02232 | 0.03202 |
| M2_b19 | 0.001699 | 4.923 | 0.01858 | 0.02728 |
| M2_b20 | 0.001665 | 4.928 | 0.01838 | 0.02701 |

**Reading the table — why geom reorders the modes.** The rigid-body DOFs split
sharply:
- **Focus (M2_dz, Cam_dz):** huge stroke `r_j` (5900, 8700 µm) but tiny FWHM
  leverage `f_j` (~0.023 arcsec/µm) → large geom weight (504, 601).
- **Decenter (M2_dx/dy, Cam_dx/dy):** huge stroke, even tinier `f_j`
  (~3e-4–1.3e-3) → the largest geom weights (2275, 4688).
- **Tilt (M2_rx/ry, Cam_rx/ry):** tiny stroke `r_j` (0.12, 0.24) but very high
  `f_j` (148, 102 arcsec per unit) → tiny geom weight (0.028, 0.048).

So geom up-weights the large-stroke / low-leverage DOFs (focus, decenter) and
down-weights tilt. Combined with the sensitivity matrix `A`, this puts focus at
the top of the SVD (v1) — whereas the default weights leave tilt dominant.

> **Note — default weights are config-dependent.** The `default` column above is
> the v13 config. A bare `OFCData()` (no `config_dir`, as
> `StateEstimator.get_vmodes_from_dofs` uses in the olr `vmodes` path) loads a
> different set (e.g. M2_dz ≈ 68 vs 714 here) — another reason the default
> normalization is not a stable basis for analysis. The geom weights are derived
> from `r_j`/`f_j` and so are reproducible given the config.

## Which to use

- **Analysis / interpreting the optical state** (this notebook, bounce,
  smatrix_vmode_info): **geom_mean** — physically meaningful, unit-invariant,
  v1 = focus.
- **Reproducing the operational loop / olr `vmodes` / RubinTV**: **default** —
  that is what the telescope OFC and `get_vmodes_from_dofs` use.
