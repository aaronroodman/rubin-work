# smatrix — AOS double-Zernike sensitivity matrix

Compute the Rubin AOS **double-Zernike (DZ) sensitivity matrix** with
`batoid_rubin`, and compare it to the matrix shipped in `ts_ofc`.

## Matrix definition
Shape `(31, 29, 50)` = (field-Zernike k, pupil-Zernike j, DOF), units
**µm wavefront per µm/arcsec of DOF**. Field radius 1.75°, pupil ε=0.61,
`jmax=28`, `kmax=30`. This exactly reproduces the ts_ofc generator
`lsst.ts.ofc.scripts.generate_sensitivity_matrix` (same `double_zernike`
helper, sampling, and unit handling).

## DOF ordering (length 50)
| idx | DOF | unit |
|----|----|----|
| 0-2 | M2 dz, dx, dy | µm |
| 3-4 | M2 Rx, Ry | arcsec |
| 5-7 | Cam dz, dx, dy | µm |
| 8-9 | Cam Rx, Ry | arcsec |
| 10-29 | M1M3 bending modes 0..19 | µm |
| 30-49 | M2 bending modes 0..19 | µm |

## Convention
`LSSTBuilder` is configured to match our other AOS work
(`ts_intrinsic_wavefront`): **OCS**, `flip_m1m3_bending_modes=False`,
`flip_m2_bending_modes=False`, `dof_angle_units="arcsec"`.

The OFC-shipped matrix was generated with bare `LSSTBuilder(fiducial)`
defaults (**ZCS**, `flip_m2=True`). The comparison is expected to reveal
per-DOF sign flips on the columns where these conventions differ (hexapod
tilts, M2 bending modes); magnitudes should still agree.

## Data
Uses the pre-downloaded batoid_rubin data at
`/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data/`
(subdirs `fea_legacy/` and `bend/`). Override with `--data-dir` or
`$BATOID_RUBIN_DATA_DIR`. The `bend` dataset ships 20 M1M3 + 20 M2 modes,
so this is already the "full mirror" set at the standard 50 DOF; expanding
to more modes is just widening `use_m1m3_modes` / `use_m2_modes` and `NDOF`.

## Run
```bash
cd code
python compute_smatrix.py --band r      # -> ../output/smatrix_dz_31_29_50_r_ocs_flipfalse.{npy,fits}
python compare_ofc.py     --band r      # compares against ts_ofc matrix
```
`compare_ofc.py` finds the OFC matrix from an installed `lsst.ts.ofc`, or
falls back to fetching it from the ts_ofc GitHub repo.
`plot_comparison.py` writes `output/smatrix_ofc_comparison_<band>.png`.

## Results (band r, vs ts_ofc `develop`)
Report `output/compare_ofc_r.txt`; plots:
- `smatrix_ofc_comparison_r.png` — per-DOF corr + rms ratio
- `smatrix_ofc_fieldconst_r.png` — field-constant pupil-Zernike × DOF heatmaps
- `smatrix_ofc_examples_r.png`   — full 31×29 DZ maps ours/OFC/diff, sample DOF
- `smatrix_ofc_residual_r.png`   — per-DOF residual after global sign

**All 50/50 DOF agree with OFC to |corr| > 0.999 and rms ratio 1.000**, sharing
a single global sign **ours = −OFC**. After that one scalar sign, the max
residual is **0.11 µm**, confined to the very-high-leverage tilt DOF (M2/Cam
Rx,Ry, sensitivity ~300 µm/deg → ~3×10⁻⁴ relative); **relative residual is
< 0.2% for every DOF.** The global sign flips even the coordinate-independent
axial terms (M2 dz, Cam dz) and the bending modes, so it is a wavefront-sign
convention, *not* the OCS-vs-ZCS hexapod choice.

### Three gotchas that had to be matched
1. **Mode selection.** The `bend` dataset stores **30** raw SVD modes per mirror;
   the AOS uses a curated 20 given by `bend.yaml`: M1M3 = [0..18, **26**], M2 =
   [0..16, **25, 26, 27**]. Passing `use_*_modes=None` reads this (what OFC does).
   Forcing `range(20)` instead put the *wrong* physical modes at the top and was
   the sole cause of an initial 4-DOF mismatch (M1M3 B20, M2 B18/B19/B20).
2. **Angle-DOF units.** The OFC matrix's Rx,Ry columns are effectively
   **per-degree** (~3600× the per-arcsec values) despite the header saying
   "arcsec". We use `dof_angle_units="degree"`.
3. **Angle-DOF step.** A unit (=1°) tilt vignettes the telescope → rank-0 Zernike
   fit. We perturb angle DOF by 1e-3° and divide it out (see `STEP`).

## Mode galleries
`code/mode_gallery.py` → `output/mode_gallery_M1M3.png`, `output/mode_gallery_M2.png`:
the 20 AOS bending-mode surfaces per mirror (unit 1 µm coefficient; µm surface),
reconstructed as `Zernike(mode.zk) + grid_residual` on the true mirror footprint
(M1M3 = M1 outer annulus + M3 inner annulus), labeled B1..B20 with the raw SVD
index annotated — for comparison with Megias-Homar et al. 2024 (arXiv:2406.04656).
