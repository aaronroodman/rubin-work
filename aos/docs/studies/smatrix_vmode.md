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
| `plot_vmode_dof_matrix.py` | five-page SVD diagnostic for one DOF/v-mode scheme (default 22 DOF / 12 v-modes) → `output/smatrix_vmode/vmode_dof_matrix_<scheme>.pdf` |
| `analyze_sparse_fit.py` | whether a **sparse** donut fit — primaries only, secondary and tertiary terms fixed at nominal — can still constrain the optical state → `output/smatrix_vmode/sparse_fit_study.pdf` |

This study is diagnostics of the matrix itself. The consumer that *builds a product* from
the same SVD is [`lut`](lut.md), which projects the FAM Double Zernike fits onto it to
recover degrees of freedom.

### `plot_vmode_dof_matrix.py` — the five pages

1. **V matrix**, the dimensionless per-DOF composition of each v-mode, drawn with square
   cells. All v-modes are shown with a line marking the `n_keep` truncation, so the
   discarded modes are visible rather than cropped away.
2. **Singular values**, the spectrum with the truncation marked.
3. **Double Zernike per unit v-mode** — µm of wavefront that a unit-amplitude v-mode
   produces (`sigma_m * u_m = S v_m`), rows being the (focal `k`, pupil `Zj`) terms.
   Retained v-modes only.
4. **Reachability and residual per DZ term** — the fraction of each elementary DZ term
   the retained v-modes can produce, and the irreducible remainder. The derivation is in
   `notebooks/smatrix_vmode/jk_coverage_plots.ipynb`; the short version is that the
   $(k,j)$ axes are an arbitrary coordinate choice, whereas the columns of $U$ are the
   intrinsic orthonormal basis of $\mathrm{col}(S)$, so the meaningful quantity is
   $f_{k,j} = \|U^\top \mathbf{e}_{k,j}\|^2$ — the squared row-sum of $U$.
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

`--part both` (the default) does both, and is the study — it writes
`sparse_fit_study.pdf`. A single `--part` is for a quick look and must be given its own
`--out`, so a partial run cannot overwrite the full PDF. Neither part needs FAM data: both work
directly on the ts_ofc DoubleZernike sensitivity matrix `S[k, j, d]` (31 field-Zernike
`k`, 29 pupil-Noll `j`, 50 DOF `d`). Because the field basis is orthonormal, the
field-map correlation of two pupil Zernikes' responses to a DOF is just the correlation
of their field-coefficient vectors, so no simulation is required.

## Notebooks

| notebook | content |
|---|---|
| `notebooks/smatrix_vmode/jk_coverage_plots.ipynb` | **derivation behind page 4**: why reachability is the right quantity, and the algebra for $f_{k,j}$ and the u-mode residual |
| `notebooks/smatrix_vmode/vmode_dof_ts_ofc.ipynb` | v-mode/DOF normalization through the `ts_ofc` `StateEstimator` |
| `notebooks/smatrix_vmode/smatrix_vmode_info.ipynb` | early exploratory treatment: SVD with `StateEstimator` plus custom-SVD validation, v-mode composition, wavefront signatures, control equations, noise/gain, a normalization-scheme unit-invariance study, and DZ field patterns |

`smatrix_vmode_info.ipynb` predates the decision to use `StateEstimator` everywhere and
is the one place that still carries the alternative normalizations, kept deliberately as
the record of that comparison.

The reachability derivation stays a notebook rather than becoming PDF text pages because
its equations use `\underbrace` to label the reachable and residual parts of a
decomposition, which matplotlib's mathtext cannot render, and no system LaTeX is
available here for `usetex`. The script owns the figures; the notebook owns the algebra.
Its plotting sections 1 and 2 were dropped when it moved, since pages 1 and 3 of the PDF
supersede them.

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
whatever the configured OFC uses; `ofc_svd.DEFAULT_NORM_YAML` names
`range0.5_fwhm-0.15.yaml`. `--check` verifies the result against `StateEstimator`.

**The `-0.15` and `-0.5` filenames hold the same weights.** An older
`range0.5_fwhm-0.5.yaml` exists on the `ts_ofc` branch `tickets/DM-54762` (commit
`76764fe`) and is not in any checked-out `normalization_weights/` directory here. Its 50
weights are numerically identical to `range0.5_fwhm-0.15.yaml` — verified with
`np.allclose`, ratio exactly 1.0, both spanning 0.02701 to 6869 in per-DOF weight units.
So the exponent in the filename does not describe the file's contents, and seeing the
other name in an old notebook does not mean a different normalization was used.

Sign and unit conventions (ZCS, bending-mode flips, degree angle units, the y-sign patch)
are settled in [`../../../smatrix/docs/conventions.md`](../../../smatrix/docs/conventions.md).

## The 22-DOF reduced set is not the first 22 indices

`22_12` means 10 rigid-body + 7 M1M3 bending + 5 M2 bending DOF, which is indices 0–16
plus 30–34, **not** 0–21. Pass the explicit index list `aos_state.DOF22` to
`build_ofc_svd(..., n_dof=...)`; passing the scalar `22` silently selects the wrong set.
`analyze_sparse_fit.py` takes the equivalent route through ts_ofc's own
`comp_dof_idx` with `(7, 5)` bending modes per mirror.

## Two `k` regimes — the `k=1..6` convention and where it does not apply

The focal (field) Noll index `k` can be truncated in one regime and not the other, and
conflating them is the easy mistake:

| regime | how the matrix is formed | `k` truncation |
|---|---|---|
| **DZ-space** — `build_ofc_svd(iZs, k_min, k_max, ...)` | slices the slab, `S_full[k_min:k_max+1, iZs, :]` | **`k=1..6` throughout `aos/`** — `bounce/`, `coadd/`, `cwfs/`, `static_optics/`, `lut/`, `correlations/`, `psf_maps_lib.py` |
| **field-evaluated** — `StateEstimator` | `Vh` from the whole slab flattened to 899 rows (31 focal × 29 pupil); `get_sensitivity_matrix` evaluates the DZ polynomial at the corner field angles | **none possible** — evaluating at a field point sums every `k` from 0 to 30, leaving no `k` axis |

Both are legitimate bases for different questions. `optical_state` v-modes use the
**full-`k` `StateEstimator` basis**, because that is what MTAOS runs on the summit; the
`k=1..6` basis is what the DZ correlation work uses, and the two are not interchangeable.

## Open question for the OFC maintainers: `Vh` ignores `zn_selected`

`StateEstimator._update_from_ofc_data` (`state_estimator.py:93-96`) builds the v-mode basis
from the **full** slab, applying no pupil-Zernike selection:

```python
dz_sens_matrix = ofc_data.sensitivity_matrix.reshape(-1, ofc_data.ndofs)[:, ofc_data.dof_idx]
self.U, self.S, self.Vh = np.linalg.svd(dz_sens_matrix @ self.normalization_matrix, ...)
```

`get_sensitivity_matrix` **does** apply it (`[:, self.ofc_data.zn_idx, :]`, line 312) and
then projects the result onto that same `Vh` (`sensitivity_matrix @ self.Vh.T`, line 316).
So the Zernike-selected corner matrix is truncated in a basis built from the unselected
slab. Setting `ofc_data.zn_selected` therefore leaves `Vh` bit-identical; setting it before
`configure_controller` is additionally reverted, since that re-reads the key from the
controller yaml (`ofc_data.py:714`).

Measured, for the 21-term set (Z4–Z26 omitting Z20, Z21), comparing the current `Vh`
against one rebuilt from the `zn_idx`-selected slab (651 rows rather than 899):

| DOF set | `n_keep` | `\|cos\|` per mode (dimensionless) | `\|ΔS\|/S` (dimensionless) |
|---|---|---|---|
| `all_50` | 34 | min 1.04e-05 at v27; median 0.99808 | max 0.2838 at v31; median 0.00287 |
| `standard_22` | 12 | min 0.99999392 at v11; median 1.00000000 | max 0.00127 at v12; median 2.10e-05 |

Principal angles between the top-`n_keep` DOF subspaces, and the largest fraction of a
retained mode's power (dimensionless, power not amplitude) falling outside the other
convention's retained block:

| `n_keep` | max principal angle [deg] | max power leaking outside |
|---|---|---|
| 6 | 0.0042 | 5.5e-09 |
| 12 | 0.3795 | 4.4e-05 |
| 18 | 8.5154 | 2.19e-02 |
| 24 | 55.19 | 0.672 |
| 34 | 89.68 | 0.626 |
| 50 | 0.0000 | 2.3e-15 |

Three qualifications keep this short of "the basis is wrong":

- **It is not mode reordering.** All-slab mode `v_m` inside the top-`m` selected subspace
  gives 0.0036 at v23, 0.374 at v29, 0.627 at v30 (fractions of power) — modes genuinely
  rotate and re-rank rather than swapping.
- **It is a truncation-boundary effect in near-degenerate singular values.** The affected
  modes sit where consecutive singular values are close (v29→v30 fractional gap 0.78%,
  v33→v34 1.75%), where individual singular vectors are ill-defined. At `n_keep=50` the
  subspaces agree exactly (max angle 0.0000 deg), so both decompositions span the same
  space and differ only in ordering and mixing of the weak modes.
- **The solve does not degrade.** On the corner-evaluated matrices (84×50 and 84×22) over
  200 unit-scale random DOF probes, median DOF recovery RMS is 8.8635e+02 (current) vs
  9.1488e+02 (Zernike-consistent) for 50/34 — ratio 0.969 dimensionless — and 4.2908e+01 vs
  4.7976e+01 for 22/12, ratio 0.894. Conditioning is comparable: `cond(A_v)` = 3.4846e+05
  vs 2.9376e+05 for 50/34, identical to five figures for 22/12. Units of the RMS are mixed
  (µm for translations and bending, arcsec for tilts), so only the ratio is meaningful.

Using the full slab makes `Vh` independent of controller configuration, which may well be
deliberate. What is not defensible is applying `zn_idx` on one side of the projection and
not the other: at `n_keep` between 18 and 34 that makes "the retained v-modes" ambiguous at
the level of a 89.7 deg subspace rotation. **For the 22-DOF/12-v-mode scheme the effect is
negligible** and no result here depends on the resolution.

## State and open questions

- The sparse-fit study found that **all DOF couple primary↔secondary at ±1**, and that
  production zeroes coma2/tref2 — see `../../../notes/claude-memory/sparse-fit-sensitivity.md`.
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
python code/smatrix_vmode/analyze_sparse_fit.py
```

All of these need `lsst.ts.ofc` and `$TS_CONFIG_MTTCS_DIR`; note `ts_ofc` is **not** in
`lsst_distrib`, so they need the AOS/CWFS environment.

## See also

- [`../../../smatrix/README.md`](../../../smatrix/README.md) — matrix construction
- [`correlations.md`](correlations.md) — the v-mode correlation consumer
- [`cwfs.md`](cwfs.md) — the corner OFC inverse uses the same SVD
