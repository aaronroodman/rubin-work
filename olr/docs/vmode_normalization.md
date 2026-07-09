# V-mode normalization: default vs geom_mean (why v1 differs)

**Date:** 2026-07-08
**Author:** Aaron Roodman (analysis with Claude)
**Context:** computing v-modes from the AOS DOF trim in
`blocks/t539_closedloop_aos.ipynb`, cross-checked against `olr` and
`aos/smatrix_vmode_info.ipynb`.

## Summary

The v-modes are the right singular vectors of the **normalized** sensitivity
matrix `Ã = A · diag(n)`, where `n_j` is the per-DOF normalization weight. The
choice of `n_j` changes the SVD, so it changes **which physical mode is v1** and
the numerical scale of the v-mode amplitudes. Two conventions are in use:

| Convention | `n_j` | v1 is… | v1 at closed-loop convergence |
|------------|-------|--------|-------------------------------|
| **default** (OFC stored `normalization_weights`) | stored weights, not unit-invariant | **M2 tilt** (rx/ry) | ~1e-4 (tightly controlled → ~0) |
| **geom_mean** (recommended; `smatrix_vmode_info`, bounce) | `sqrt(r_j / f_j)` | **focus** (M2_dz + Cam_dz) | ~0.1 (real residual focus) |

Both are "correct" — they are just different bases. But they are **not
interchangeable**, and comparing a v1 from one against a v1 from the other is
meaningless. `blocks/t539_closedloop_aos.ipynb` uses **geom_mean** so that v1 =
focus and the amplitudes match the bounce test / `smatrix_vmode_info`.

The operational OFC (and `StateEstimator.get_vmodes_from_dofs`, and the olr
`vmodes` column from `nightly_table.py`) use the **default** normalization —
so those v-modes differ from this notebook's geom v-modes by construction.

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
import sys, numpy as np
sys.path.insert(0, '<repo>/aos/code')
from ofc_svd import compute_normalization_components
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

### Values (standard_22 DOF)

Default weights (from the notebook run) and geom weights where computed. **r_j /
f_j columns to be filled from the extraction snippet above.**

| DOF | r_j | f_j | geom sqrt(r/f) | default |
|-----|-----|-----|----------------|---------|
| M2_dz  | — | — | 504.52 | 68.302 |
| M2_dx  | — | — | — | 3.534 |
| M2_dy  | — | — | — | 3.534 |
| M2_rx  | — | — | — | 7.252 |
| M2_ry  | — | — | — | 7.252 |
| Cam_dz | — | — | 601.46 | 104.487 |
| Cam_dx | — | — | — | 1.107 |
| Cam_dy | — | — | — | 1.107 |
| Cam_rx | — | — | — | 0.496 |
| Cam_ry | — | — | — | 0.496 |
| M1M3_b1 | — | — | — | 0.341 |
| M1M3_b2 | — | — | 0.551 | 0.339 |
| M1M3_b3 | — | — | 0.202 | 0.096 |
| M1M3_b4 | — | — | — | 0.076 |
| M1M3_b5 | — | — | — | 0.076 |
| M1M3_b6 | — | — | — | 0.031 |
| M1M3_b7 | — | — | — | 0.028 |
| M2_b1 | — | — | — | 0.506 |
| M2_b2 | — | — | — | 0.508 |
| M2_b3 | — | — | — | 0.103 |
| M2_b4 | — | — | — | 0.097 |
| M2_b5 | — | — | 0.182 | 0.083 |

## Which to use

- **Analysis / interpreting the optical state** (this notebook, bounce,
  smatrix_vmode_info): **geom_mean** — physically meaningful, unit-invariant,
  v1 = focus.
- **Reproducing the operational loop / olr `vmodes` / RubinTV**: **default** —
  that is what the telescope OFC and `get_vmodes_from_dofs` use.
