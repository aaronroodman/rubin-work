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
`LSSTBuilder` is configured to reproduce the OFC-shipped matrix **exactly**
(see `docs/conventions.md`): **ZCS**, `flip_m1m3_bending_modes=True`,
`flip_m2_bending_modes=True`, `dof_angle_units="degree"`, `use_*_modes=None`
(bend.yaml AOS mode set). One extra convention detail: batoid_rubin's ZCS
differs from the OFC matrix by a sign on the **y-translation (dy) and y-tip/tilt
(Ry)** of M2 and camera, so those 4 DOF columns are sign-flipped
(`OFC_Y_SIGN_FLIP`) — an open question for Josh/Guillem (see docs/conventions.md).

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

**All 50/50 DOF agree with OFC to |corr| > 0.999, rms ratio 1.000, and the same
sign (ours = +OFC).** Max residual **0.11 µm**, confined to the very-high-leverage
tip/tilt DOF (M2/Cam Rx,Ry, ~300 µm/deg → ~3×10⁻⁴ relative); **relative residual
is < 0.2% for every DOF.** Getting the sign right required ZCS (hexapod),
bending-mode flips, and the y-axis (dy, Ry) sign patch — see `docs/conventions.md`.

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

## Full bending-mode set (all actuator modes)
The `bend` dataset ships only 30 modes/mirror, but the FEA influence data in
`fea_legacy` (`M1M3_1um_156_grid`, `M2_1um_grid`) provides the complete
orthonormal mode sets: **156 M1M3 + 72 M2** modes (SVD of each surface matrix
gives all singular values = 1). The M1M3 modes are verified identical to the
batoid/OFC modes (corr +1.000); M2 matches up to rotations within degenerate
pairs.

```bash
python build_full_modes.py         # -> $BATOID_RUBIN_DATA_DIR/bend_full/ (156+72 modes)
python compute_smatrix.py --band r --bend-dir bend_full --jobs 8
# -> output/smatrix_dz_31_29_238_r_bend_full.npy   (31, 29, 238)  [10 rigid + 156 M1M3 + 72 M2]
python full_mode_analysis.py --band r
```
`compute_smatrix.py` is mode-count-aware (reads the target bend dir's bend.yaml),
so the same OFC convention (ZCS, flips, degree, y-sign) applies to all 238 DOF.
`build_full_modes.py` replicates the batoid_rubin `decompose_sag` pipeline
(Zernike Noll≤28 + gridded residual) from the FEA node data; M2 is zero-padded
to 156 stored modes so `load_bend` can index all mirrors uniformly
(`use_m2_modes` stays 72).

**Result (`full_mode_analysis.py`):** the controllability spectrum
(`fullmode_controllability_r.png`) shows wavefront leverage vs mode number.
M1M3 modes stay optically active well past the AOS-used 20 (variance reaches
90% only by mode ~99); M2 falls off faster (90% by ~36). The pupil heatmap
(`fullmode_pupil_heatmap_r.png`) shows the diagonal coupling — mode N drives
progressively higher pupil Noll.

## Docs

Reference docs in `docs/`; transient working state in `docs/status/`. Each carries a
status + last-updated line under its title.

| doc | what it holds |
|---|---|
| [`docs/conventions.md`](docs/conventions.md) | the convention choices (ZCS, bending-mode flips, degree angle units, the y-sign patch) — prepared for review by Josh Meyers / Guillem Megias-Homar |
| [`docs/miw_astig_coma_investigation.md`](docs/miw_astig_coma_investigation.md) | what can produce the observed high-field-order Z5–Z8 in the MIW that the batoid design model does not predict |
| [`docs/plots.md`](docs/plots.md) | index of the plots/outputs in `output/` and how to regenerate each |
| [`docs/status/future_issues.md`](docs/status/future_issues.md) | running list of open issues and follow-ups |
