# Corner-wavefront recovery: `recover_optical_state` vs OFC `dof_state`

> **Status:** current · **Last updated:** 2026-09-16 · **Kind:** status (decision recorded)

Measurements comparing the three candidate routes from measured corner-wavefront Zernikes
to physical degrees of freedom (DOF), made to decide which one `optical_state` should use
before the old Singular Value Decomposition (SVD) code is retired.

**Decision: the hybrid.** `aos_state.recover_optical_state` **inverts** in the
corner-evaluated basis (`aos_state.corner_recovery_basis`), because that is the matrix
actually being inverted and it closes to machine precision, but **reports v-modes** in the
`make_state_estimator` basis via `get_vmodes_from_dofs`, so every v-mode in the repository —
measured, look-up-table (LUT) and Trim — is in the basis the Main Telescope AOS runs on the
summit. The sections below are the measurements that led there.

The three routes:

| route | how it inverts |
|---|---|
| `aos_state.recover_optical_state` (current) | own SVD of the corner-evaluated, 21-Zernike-selected matrix; truncates `U`/`s` directly |
| `StateEstimator.dof_state` (what MTAOS runs) | truncates via `StateEstimator.Vh`, then noise-weighted `lstsq` in v-mode space |
| `get_sensitivity_matrix` + explicit `lstsq` | same matrix as `dof_state`, solve written out |

## Which Zernike frame `dof_state` expects: CCS

The two routes want **different frames**, which is the easiest thing here to get backwards.
`dof_state` re-evaluates the matrix at the rotator angle you hand it, so it wants the
camera-fixed wavefront (CCS) plus that angle. The hybrid fixes the matrix at rotator zero, so
it wants the telescope-fixed wavefront (OCS) and no angle. Neither is more correct in the
abstract; they are consistent pairs, and mixing them is a silent error.

`dof_state` takes the wavefront in the **camera-fixed CCS frame** — `zk_ccs`, as measured on
the detector — together with the camera rotator angle in degrees. It applies the two
rotations itself, in opposite senses:

- **field angles** (`get_sensitivity_matrix`, `state_estimator.py:296`) are rotated by the
  matrix built from `-rotation_angle`; because row vectors are right-multiplied
  (`field_angles @ rot_mat`), the net azimuth change is **+`rotation_angle`**. Verified:
  the four corner azimuths at rotator +30 deg move from (-135, +135, -45, +45) to
  (-105, +165, -15, +75) deg, a change of exactly +30 deg each.
- **wavefront Zernikes** (`dof_state`) are rotated by **+`rotation_angle`** via
  `galsim.zernike.Zernike.rotate`.

Verified end to end with a telescope-fixed (Optical Coordinate System, OCS) misalignment of
M2-hexapod dx = +100.0 µm, recovering DOF at four rotator angles:

| rotator angle [deg] | feed OCS Zernikes → recovered dx [µm] | feed CCS Zernikes → recovered dx [µm] |
|---|---|---|
| 0 | 81.55 | 81.55 |
| 30 | 66.14 | 78.65 |
| 60 | 30.79 | 78.88 |
| 90 | -12.11 | 82.21 |

Feeding CCS Zernikes gives a rotation-independent answer; feeding OCS Zernikes degrades
monotonically and inverts sign by 90 deg, the signature of applying the rotation twice.

The recovery is ~80 µm rather than 100 µm because of mode truncation, not the frame:
projecting M2-hexapod dx = 100.0 µm onto the 34 retained v-modes and back returns 79.69 µm,
so the retained subspace holds 79.7% of that DOF (dimensionless, amplitude not power). With
a truth vector drawn *inside* the retained subspace the round trip closes to a relative
`max|ΔDOF|` of 7.3e-03 to 1.5e-02 (dimensionless, over a truth scale of
max|DOF| = 645.7 µm or arcsec) at every rotator angle tested.

`ofc_data.rotation_offset` is 0.0 deg for LSSTCam, so `OFC.calculate_corrections` passes the
camera rotator angle through unmodified (`ofc.py:211`).

## The forward models are identical

`build_geom_svd`'s reconstruction `U diag(s) V^T diag(1/w)` agrees with OFC's
`get_sensitivity_matrix(field_angles, 0.0, normalize=False, truncate=False)` to
`max|ΔA| = 1.9e-12` (DZ sensitivity units, µm of wavefront per DOF unit), in both the
22-DOF/12-v-mode and 50-DOF/34-v-mode schemes. The old code was decomposing the right
matrix; the disagreement is entirely in the truncation basis.

## The inverse solves differ, and the retained subspaces differ

Noiseless synthetic corner data, truth drawn inside the retained subspace, rotator 0 deg:

| scheme | max principal angle between retained DOF subspaces [deg] | median [deg] |
|---|---|---|
| `standard_22` / 12 | 4.768 | 0.031 |
| `all_50` / 34 | 89.951 | 0.000 |

Recovered DOF from the two routes on identical input agree only to
`cos(dof_old, dof_new)` = 0.9869 to 0.9998 (dimensionless), with relative `max|ΔDOF|` of
3.6e-02 to 2.65e-01; v1 agrees to roughly 0.4-1%.

## Each route is self-consistent only in its own basis

50 random truth vectors per scheme, each drawn inside the route's *own* retained subspace,
noiseless corner data:

| scheme | route | wavefront residual RMS [µm] | relative DOF error (dimensionless) |
|---|---|---|---|
| `standard_22` / 12 | `recover_optical_state` | 2.1298e-14 | 1.736e-15 |
| `standard_22` / 12 | `dof_state` | 2.3205e-02 | 9.540e-04 |
| `all_50` / 34 | `recover_optical_state` | 1.9471e-14 | 5.524e-15 |
| `all_50` / 34 | `dof_state` | 2.3675e-02 | 8.458e-04 |

`recover_optical_state` closes to machine precision, as an SVD of the matrix actually being
inverted must. `dof_state` does **not**: it has an irreducible floor of
2.3e-02 µm wavefront residual RMS even on noiseless data drawn from its own subspace.

**Cause.** `StateEstimator.Vh` is built from the full, *unselected* DZ slab flattened to 899
rows (31 focal x 29 pupil), with no field-point evaluation, while the problem being inverted
is the 84-row (4 corners x 21 Zernikes) field-evaluated, Zernike-selected matrix. The
truncation basis therefore does not span the data space. This is the same
`zn_selected`/`Vh` inconsistency recorded as an open question for the OFC maintainers in
[`../studies/smatrix_vmode.md`](../studies/smatrix_vmode.md) — here it has a measurable
consequence on recovered DOF rather than being an internal inconsistency only.

On the same synthetic input at rotator 0 deg, wavefront residuals were 1.1884e-02 µm RMS
(old) vs 2.3205e-02 µm RMS (`dof_state`) for 22/12, and 7.3109e-03 vs 2.3675e-02 µm RMS for
50/34 — the old route fits the corner wavefront roughly 2-3x better, because its basis is
matched to the problem.

## Constraints on whichever route is chosen

- `olr/code/nightly_table.py` consumes `zk_constrained` from `recover_optical_state`; OFC's
  `dof_state` returns DOF only, so the constrained per-corner wavefront must be
  reconstructed separately.
- The Measured Intrinsic Wavefront (MIW) variant supplies its own intrinsic, so it needs
  `subtract_intrinsics=False`.
- `dof_state` expects `wfe` shaped `(n_sensors, znmax - znmin + 1)` = `(4, 25)` covering
  Z4-Z28, and applies `zn_idx` internally. Passing the 21 selected columns raises
  `IndexError`.
- `get_dofs_from_vmodes` returns a vector of length `len(dof_idx)` for `standard_22`, not
  the length 50 its docstring states; callers must handle both.
- OFC's noise covariance for LSSTCam is sized for **8** sensors (both halves of each
  corner). Passing the 4 SW0 names triggers its "not supported yet" fallback to an identity
  matrix, so `dof_state`'s noise weighting is a no-op here. Verified identity: diagonal
  1.0 to 1.0, max off-diagonal 0.0.

## The decision, and what it changed

A straight swap to `dof_state` is not supported by the measurements: it is what MTAOS runs,
but on this problem it carries a 2.3e-02 µm wavefront-residual floor that the corner-evaluated
route does not, and at 50/34 its retained subspace sits up to 89.95 deg away. Adopting the
corner route wholesale is also wrong, because then the measured v-modes are in a basis nothing
else in the repository uses. The hybrid keeps each piece where it is correct.

### The sensitivity matrix is fixed at camera rotator angle zero

`aos_state.SMATRIX_ROTATION_ANGLE_DEG` = 0.0 deg, always, and `corner_recovery_basis` takes
**no** rotation-angle argument. This is an AOS group decision, not an implementation
shortcut: evaluating the matrix at the observed rotator angle is formally more correct, but it
redefines the v-modes visit by visit, making the control loop's basis a moving target and
putting a rotator-angle confound into every v-mode time series. The fidelity given up is small
(7.3e-03 to 1.5e-02 relative in recovered DOF, dimensionless, over a truth scale of
max|DOF| = 645.7 µm or arcsec), and the near-degenerate singular-value pairs — consecutive
fractional gaps of 0.78% and 1.75% — mean a small rotation can swap or arbitrarily mix two
modes rather than rotating the basis smoothly.

The consequence is that the **wavefront** carries the rotation: `recover_optical_state`
requires its input in the telescope frame (Optical Coordinate System, OCS —
`aos_state.ZK_FRAME`), which is what all work in this repository has assumed. It takes an
explicit `zk_frame` argument and raises `ValueError` on anything else, because camera-frame
(Camera Coordinate System, CCS) input is not detectable from the numbers: it returns a
rotator-dependent answer that inverts sign over 90 deg of rotation.

### Verified equivalence to the retired code

The hybrid's inversion is **bit-identical** to the retired `build_geom_svd` path, for both
schemes, on identical synthetic corner input:

| quantity | agreement |
|---|---|
| normalization weights | `max\|Δw\| = 0.000e+00` (per-DOF weight units) |
| singular values | `max\|Δs\| = 0.000e+00` (DZ sensitivity units) |
| recovered DOF | `max\|ΔDOF\| = 0.000e+00` (µm or arcsec) |
| `zk_constrained` | `max\|Δ\| = 0.000e+00` µm of wavefront |

So `optical_state`'s recovered DOF and constrained wavefront do **not** change. What changes
is the reported v-modes, which move from the corner basis to the estimator basis.

### v-mode numbers that change, and the sign flip

v1 per µm of hexapod dz, old (corner) basis against new (estimator) basis:

| scheme | DOF | old [1/µm] | new [1/µm] | ratio (dimensionless) |
|---|---|---|---|---|
| `standard_22`/12 | camera-hexapod dz | +9.1327060e-04 | -8.9153336e-04 | -0.9762 |
| `standard_22`/12 | M2-hexapod dz | +8.8029438e-04 | -9.1035126e-04 | -1.0341 |
| `standard_22`/12 | `v1_per_um_dz` | 8.9678249e-04 | 9.0094231e-04 | 1.0046 |
| `all_50`/34 | `v1_per_um_dz` | 8.9677770e-04 | 9.0085143e-04 | 1.0045 |

Two things follow. The **magnitude** scale factor `v1_per_um_dz` moves by only 0.46%
(dimensionless, new over old), so results that use it as a calibration scale are essentially
unchanged, and it still agrees between the two schemes to five decimal places. The **sign**
flips, so every stored `v1_lut`, `v1_trim` and `v1` reverses. On synthetic LUT-like DOF
(n = 3,000) the correlation between old and new v1 is Spearman
rho = -0.9992 (dimensionless), and Spearman rho between v1 and camera-hexapod dz [µm] goes
from +0.7002 to -0.6801. The previously recorded real-data value of +0.9513 will therefore
come back near -0.95: **a basis convention change, not a regression.**

### Retired

`aos_state.project_dofs_to_vmodes` and `aos_state.build_geom_svd` have no remaining Python
callers; `olr/code/nightly_table.py`'s local `build_sensitivity_svd`, the fourth SVD site and
a verbatim duplicate of `build_geom_svd`, is now dead code there too. Commanded LUT and Trim
projections all go through `aos_state.vmodes_from_dofs`. The 12-mode cap that once argued for
the separate geom path was `StateEstimator.truncate_index`, a controller-yaml default rather
than a limit — `make_state_estimator` now sets it from `n_modes`, so the 50/34 scheme returns
34 modes.
