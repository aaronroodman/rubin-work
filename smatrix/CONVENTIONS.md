# batoid_rubin DZ sensitivity matrix — convention choices (for review)

Prepared for input from Josh Meyers and Guillem Megias-Homar.

## What we are computing
The double-Zernike (DZ) sensitivity matrix **∂Z(pupil) / ∂(DOF)**, sampled as a
function of field angle, exactly as in `ts_ofc`'s
`scripts/generate_sensitivity_matrix.py` (which itself uses `batoid` +
`batoid_rubin.LSSTBuilder` — no separate optical model).

- Shape **(31 field-Noll k, 29 pupil-Noll j, 50 DOF)**; field radius 1.75°,
  pupil ε = 0.61, `jmax=28`, `kmax=30`, `batoid.zernike(nx=255)`, Gauss-Legendre
  field sampling (11 rings × 35 spokes).
- Value units: **µm of wavefront per (µm or degree) of DOF**.
- DOF order: M2 {dz,dx,dy,Rx,Ry}, Cam {dz,dx,dy,Rx,Ry}, 20 M1M3 bending modes,
  20 M2 bending modes. **Rx/Ry are tip/tilt** (rotations about the x/y axes); we
  reserve "rotation" for the camera rotator, which is not a DOF here.

We reproduce the OFC-shipped matrix (`lsst_sensitivity_dz_31_29_50`) to **< 0.2 %
per DOF** once the conventions below are matched; the only non-trivial residuals
(~0.1 µm) are on the very-high-leverage hexapod tilts (~300 µm/deg).

## The convention choices

| Choice | OFC-shipped matrix | Our target (to match OFC) | Effect if changed |
|---|---|---|---|
| **Coordinate system** (`dof_coord_system`) | **ZCS** (Zemax), from bare `LSSTBuilder` defaults | **ZCS** | Flips the **sign of all hexapod rigid-body DOF** (M2 & camera dz, dx, dy, and Rx/Ry tip/tilt). OCS ↔ ZCS per [SITCOMTN-003](https://sitcomtn-003.lsst.io/). Our other AOS code (`ts_intrinsic_wavefront`) uses **OCS**. |
| **Bending-mode sign** (`flip_m1m3_bending_modes`, `flip_m2_bending_modes`) | flips **on** (needed to match OFC) | `flip_m1m3=True`, `flip_m2=True` | Flips sign of the M1M3 / M2 bending-mode columns. Independent of coordinate system. SVD modes have an intrinsic sign ambiguity, so this is a pure convention. |
| **Angle-DOF units** (`dof_angle_units`) | effectively **degree** (columns are ~3600× the per-arcsec values, despite the file header saying "arcsec") | **degree** | Scales the 4 hexapod-tilt columns by 3600. |
| **Mode selection** (`use_m1m3_modes`, `use_m2_modes`) | `bend.yaml` default: **M1M3 = [0…18, 26]**, **M2 = [0…16, 25, 26, 27]** (20 of the 30 stored raw SVD modes) | same (`=None`) | Choosing `range(20)` instead puts the wrong physical modes in the top slots. |
| **Angle perturbation step** | (n/a — linear) | 1e-3° (divided out) | A full 1° tilt vignettes the pupil → rank-0 Zernike fit; a small step avoids this and is exact by linearity. |

## Agreement status
- **Magnitudes: all 50 DOF match OFC to < 0.2 %** (min |corr| = 0.9998).
- **Signs:** with **ZCS + `flip_m1m3=True` + `flip_m2=True` + degree**, 46 of 50
  DOF reproduce OFC's sign exactly. The **only** sign mismatches are four DOF:

  | DOF | index |
  |---|---|
  | M2 dy   | 2 |
  | M2 Ry   | 4 |
  | Cam dy  | 7 |
  | Cam Ry  | 9 |

  These are precisely the **y-translation (dy) and y-tip/tilt (Ry)** of *both*
  M2 and the camera; the x and z counterparts (dz, dx, Rx) all match. So
  batoid_rubin's ZCS and the OFC matrix differ by a **sign on the y-axis
  hexapod quantities** — a left/right-handedness convention on y (and hence on
  the tip/tilt Ry about y).

  We reproduce OFC exactly by additionally flipping the sign of those 4 DOF
  columns (`OFC_Y_SIGN_FLIP` in `compute_smatrix.py`).

  **[QUESTION for Josh / Guillem]** Is this y-axis sign the intended convention
  in the shipped OFC matrix (i.e. does OFC define +y opposite to batoid_rubin's
  ZCS +y), or is one side simply inconsistent? We'd like to pin down the
  authoritative y / tip-tilt-Ry handedness before extending the matrix.

## Notes / open items
1. The OFC matrix's provenance: generated with bare `LSSTBuilder()` defaults at
   some batoid_rubin version. Its angle columns are per-degree while the current
   `LSSTBuilder` default angle unit is arcsec — worth confirming which
   batoid_rubin version / units the shipped matrix intends.
2. We ultimately want to **extend beyond the 20 AOS modes** to the full
   actuator-space bending basis (M1M3 156 actuators, M2 ~72, minus rigid-body /
   force-balance constraints). That mode basis must be generated from the FEA
   influence functions (as the `bend` dataset's 30 modes were) — a separate step
   from this sensitivity-matrix reproduction.
