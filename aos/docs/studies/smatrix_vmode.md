# Study: `smatrix_vmode` — sensitivity matrix and mode structure

> **Status:** current · **Last updated:** 2026-09-07 · **Kind:** reference (study)

Analysis of the Optical Feedback Control (OFC) sensitivity matrix and its mode
structure: the singular value decomposition, v-mode composition, and which degrees of
freedom (DOF) are observable.

The sensitivity matrix maps 50 degrees of freedom to a double-Zernike wavefront. Its SVD
gives **v-modes** (DOF-space directions) and **u-modes** (wavefront-space directions).
Which modes are observable — and how badly they mix — sets the ceiling on everything the
AOS can do. This study is the linear algebra behind the other studies' interpretation.

The matrix *construction* lives in the sibling [`smatrix/`](../../../smatrix/) topic;
this study is about using and diagnosing it.

## Code

| file | role |
|---|---|
| `run_build_lut.py` | pipeline `build_lut` — averaged-DOF look-up table: project per-visit DZ fits onto the OFC SVD, recover DOF, collapse over elevation and rotator |
| `plot_vmode_dof_matrix.py` | five-page SVD diagnostic for one DOF/v-mode scheme (default 22 DOF / 12 v-modes) → `output/smatrix_vmode/vmode_dof_matrix_<scheme>.pdf` |
| `analyze_sparse_fit.py` | whether a **sparse** donut fit — primaries only, secondary and tertiary terms fixed at nominal — can still constrain the optical state → `output/smatrix_vmode/sparse_fit_<part>.pdf` |

### `plot_vmode_dof_matrix.py` — the five pages

1. **V matrix**, the dimensionless per-DOF composition of each v-mode, drawn with square
   cells. All v-modes are shown with a line marking the `n_keep` truncation, so the
   discarded modes are visible rather than cropped away.
2. **Singular values**, the spectrum with the truncation marked.
3. **Double Zernike per unit v-mode** — µm of wavefront that a unit-amplitude v-mode
   produces (`sigma_m * u_m = S v_m`), rows being the (focal `k`, pupil `Zj`) terms.
   Retained v-modes only.
4. **Reachability and residual per DZ term** — the fraction of each elementary DZ term
   the retained v-modes can produce, and the irreducible remainder.
5. **Normalization weights** — the per-DOF weight `w_i` applied, decomposed into its
   range factor `r_i` (DOF-units of stroke) and FWHM factor `f_i` (arcsec of PSF width
   per DOF-unit), since `w_i = r_i^0.5 * f_i^-0.5`.

`--check` runs a regression test instead of plotting: it asserts `build_ofc_svd`
reproduces ts_ofc's `StateEstimator.get_dofs_from_vmodes` (DOF-per-v-mode = `N.V`) on
identical inputs, and exits 0=PASS / 1=FAIL.

### `analyze_sparse_fit.py` — the two parts

`--part sensitivity` asks which DOF drive the secondary and tertiary aberrations of an
azimuthal family, and whether those same DOF also drive a field-*correlated* primary of
that family. Those are the DOF whose mis-separation would push real secondary/tertiary
content into the primary, manufacturing exactly the radial-order correlations the MIW
shows — astig Z5/6 ↔ Z12/13 ↔ Z23/24, coma Z7/8 ↔ Z16/17.

`--part observability` re-forms the sensitivity matrix using only the primary pupil-Noll
rows and compares the singular-value spectrum, per-v-mode observability and per-DOF
observability ratio against the full matrix, for both the 50-DOF/34-v-mode and
22-DOF/12-v-mode schemes.

`--part both` (the default) does both in one PDF. Neither part needs FAM data: both work
directly on the ts_ofc DoubleZernike sensitivity matrix `S[k, j, d]` (31 field-Zernike
`k`, 29 pupil-Noll `j`, 50 DOF `d`). Because the field basis is orthonormal, the
field-map correlation of two pupil Zernikes' responses to a DOF is just the correlation
of their field-coefficient vectors, so no simulation is required.

## Notebooks

| notebook | content |
|---|---|
| `notebooks/smatrix_vmode/vmode_dof_ts_ofc.ipynb` | v-mode/DOF normalization through the `ts_ofc` `StateEstimator` |
| `notebooks/smatrix_vmode/smatrix_vmode_info.ipynb` | early exploratory treatment: SVD with `StateEstimator` plus custom-SVD validation, v-mode composition, wavefront signatures, control equations, noise/gain, a normalization-scheme unit-invariance study, and DZ field patterns |

`smatrix_vmode_info.ipynb` predates the decision to use `StateEstimator` everywhere and
is the one place that still carries the alternative normalizations, kept deliberately as
the record of that comparison.

## Output sits outside any `param_set`

Both scripts write to **`output/smatrix_vmode/`**, at the top level, not under a
`param_set`. The v-mode/DOF structure is a property of the OFC sensitivity matrix and the
DOF scheme alone — no FAM data enters it. The one data-derived input is the pupil-Zernike
set, which is identical in every `param_set` built to date (Z4–Z26 omitting Z20 and Z21,
21 terms), so it defaults in code as `ZK_NOLL_DEFAULT`. Passing `--param-set` to
`plot_vmode_dof_matrix.py` reads it from that `param_set`'s `visits.parquet` instead and
warns if it differs from the default.

## Normalization — the trap

There are **two** ways to get v-modes and they do not agree unless you are careful:
`StateEstimator` (4-CWFS) versus `build_ofc_svd` (double-Zernike), with geometric
weights in play. There is a genuine degeneracy caveat. Read
`../../../notes/claude-memory/aos-vmode-normalization.md` and
[`../../../olr/docs/vmode_normalization.md`](../../../olr/docs/vmode_normalization.md)
before comparing v-modes computed two ways.

The scripts here take the normalization weights from a ts_config_mttcs yaml rather than
defining their own. `plot_vmode_dof_matrix.py` reads the name from the OFC controller
config at runtime (`ofc.controller['normalization_weights_filename']`), so it uses
whatever the configured OFC uses; `ofc_svd.DEFAULT_NORM_YAML` is
`range0.5_fwhm-0.15.yaml`, matching the `w_i = r_i^0.5 * f_i^-0.5` decomposition on page
5. `--check` verifies the result against `StateEstimator`.

Sign and unit conventions (ZCS, bending-mode flips, degree angle units, the y-sign patch)
are settled in [`../../../smatrix/docs/conventions.md`](../../../smatrix/docs/conventions.md).

## The 22-DOF reduced set is not the first 22 indices

`22_12` means 10 rigid-body + 7 M1M3 bending + 5 M2 bending DOF, which is indices 0–16
plus 30–34, **not** 0–21. Pass the explicit index list `aos_state.DOF22` to
`build_ofc_svd(..., n_dof=...)`; passing the scalar `22` silently selects the wrong set.
`analyze_sparse_fit.py` takes the equivalent route through ts_ofc's own
`comp_dof_idx` with `(7, 5)` bending modes per mirror.

## State and open questions

- The sparse-fit study found that **all DOF couple primary↔secondary at ±1**, and that
  production zeroes coma2/tref2 — see `../../../notes/claude-memory/sparse-fit-sensitivity.md`.
- `build_lut` currently projects the **Phase-1** `fits.parquet`, not the MI-refit one.
- `svd._keep()` is used here and in `cwfs`. Its leading underscore is mislabelling
  rather than a stability boundary — it is `ts_intrinsic_wavefront`'s own function, used
  as public API in four call sites across two repositories. Left as-is deliberately; see
  [`../status/code_review_backlog.md`](../status/code_review_backlog.md).

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until build_lut
python code/smatrix_vmode/plot_vmode_dof_matrix.py --scheme 22_12
python code/smatrix_vmode/plot_vmode_dof_matrix.py --scheme 50_34
python code/smatrix_vmode/plot_vmode_dof_matrix.py --check
python code/smatrix_vmode/analyze_sparse_fit.py --part both
```

All of these need `lsst.ts.ofc` and `$TS_CONFIG_MTTCS_DIR`; note `ts_ofc` is **not** in
`lsst_distrib`, so they need the AOS/CWFS environment.

## See also

- [`../../../smatrix/README.md`](../../../smatrix/README.md) — matrix construction
- [`correlations.md`](correlations.md) — the v-mode correlation consumer
- [`cwfs.md`](cwfs.md) — the corner OFC inverse uses the same SVD
