# AOS code review — open backlog

> **Status:** current · **Last updated:** 2026-09-06 · **Kind:** working state (review backlog)

Items deferred during the study reorganization, each because fixing it would change
results or behaviour rather than structure. Kept separate from
[`code_review_findings.md`](code_review_findings.md), which is the older
severity-ordered review whose line anchors are stale.

Aaron is reviewing `aos/code/` file by file, assessing (a) whether code belongs in
`common/` and (b) readability, including bringing docstrings up to the Rubin DM
standard. These are the known items to fold into that pass.

## Duplicated helpers that are not equivalent

Both were left alone in the shared-helper extraction because the copies genuinely
differ; unifying either changes results.

### `quality_cut` — four copies, two signatures
| file | signature |
|---|---|
| `run_dz_correlations.py` | `quality_cut(df, prefix, max_coeff_um)` |
| `run_thermal_correlations.py` | `quality_cut(df, prefix, max_coeff_um)` |
| `run_dz_explained.py` | `quality_cut(df, prefix, maxc)` |
| `run_vmode_correlations.py` | `quality_cut(df, prefix, maxc)` |

The bodies also differ, not just the parameter name. Decide the one correct cut —
including what it does about non-finite coefficients — then share it. Candidate home:
`aos/code/dz_columns.py`, alongside `dz_coeff_columns`, or `common/` if it turns out to
be generic over any coefficient table.

### `load_miw` — four copies, four behaviours
| file | signature |
|---|---|
| `camera_gravity_maps.py` | `load_miw(path)` |
| `run_m3_backprojection_miw.py` | `load_miw(path, stride)` → `(pts[N,2] deg, zk[N,27] µm OCS, df)` |
| `run_miw_backprojection_surfaces.py` | `load_miw(path, stride)` |
| `run_miw_joint_fit.py` | `load_miw(path, stride)` |

Spans the `static_optics` and `coadd` studies. The three `stride` versions are not
identical either. This is the natural place to also settle *which* MIW product is
canonical — the `_5rot` `intrinsic_split_maps` with OCS columns — so that every caller
reads the same thing. A shared MIW reader is a strong `common/` or
`aos/code/miw_io.py` candidate.

## `combine_parquets.py` — two versions, `olr/`'s is ahead

`aos/code/combine_parquets.py` is 148 lines; `olr/code/combine_parquets.py` is 168.
Not a copy: the `olr/` version **drops 0-row sentinel inputs before schema
unification**, and handles the all-inputs-empty case by writing a valid 0-row output.

Why it matters: a night with no AOS products writes an empty marker file so the DAG
completes. That marker's minimal schema would shrink the common-column intersection to
nothing, so the `aos/` copy would fail or silently produce a column-starved table.

Fix is to promote one implementation — the `olr/` logic is the right one — ideally into
`common/`, since concatenating parquet chunks is not topic-specific. It changes the
pipeline's combine step, so it needs a test over a chunk set that includes an empty
night.

## Live defects confirmed during the reorganization

### `run_psf_fp_maps.py:141` — NaN is truthy
```python
v = np.nanpercentile(np.abs(m[key]), 98) or 0.01
```
If the percentile is NaN (all-NaN input for that key), `NaN or 0.01` evaluates to
**NaN**, because NaN is truthy in Python — verified. The `or 0.01` guard only catches an
exact `0.0`. The NaN then propagates into `vmin`/`vmax` and the colour scale breaks
silently. Fix with an explicit `np.isfinite` check. Lines 135 and 159 use
`nanpercentile` without the `or` idiom and are unaffected.

### `analyze_miw_field_order.py` — stale products, unhelpful failure
In `output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/coadd_50_34/`,
`block_grids.npz` has umodes shape **(130, 34)** while
`coadd_metrics_rebin3.parquet` has **221 rows** (`build_used` sum 16). The two are from
different runs, so the boolean mask raises `IndexError`. Either regenerate both from one
run, or assert `len(mt) == len(Um)` with a message naming both files and both counts.

This is the row-count/row-order invariant defect from the original review, occurring for
real. The general lesson for the review pass: **a count mismatch is caught, an order
mismatch is not** — sidecar tables row-aligned to `donuts.parquet` have an undocumented
ordering contract.

### Laptop paths that cannot resolve on S3DF
`analyze_sensitivity_sparse.py` and `analyze_sparse_observability.py` still set
`_PKG = "/Users/roodman/..."`. Outside `aos/`, **11 more tracked files** do the same for
data directories: 3 in `filters/code/design_*.py` (throughput dir) and 8 in
`smatrix/code/` (`batoid_rubin_data`, `ts_config_mttcs`). These are data-directory
constants rather than import bootstrapping, so they were out of scope for the import
fix. Each wants an env-var plus fallback (`$BATOID_RUBIN_DATA_DIR`,
`$TS_CONFIG_MTTCS_DIR`).

### `svd._keep()` — private API
Two call sites use the private `_keep()` from `ts_ofc`:
`plot_vmode_dof_matrix.py:147` and `run_wfs_dof_compare.py:520`, both to index
`svd.Sigma` down to the kept singular values. Flagged in the original review; a
package-internal dependency that can break on any `ts_ofc` update. Worth asking whether
`ts_ofc` exposes a public equivalent.

## Notes for the file-by-file pass

- **`common/` candidates.** `common/` already holds `nmad`, `alt_to_deg`, `repo_root`,
  `FocalPlaneInterpolator`, and the text-histogram helpers. Further candidates, in
  rough order of how clearly they are generic:
  - **`psf_render.py`** — GalSim `OpticalPSF` + Kolmogorov atmosphere + HSM
    measurement. Nothing in it is AOS-specific: it takes Zernikes and a wavelength and
    returns rendered/measured moments. Already used by two studies (`psf`, `cwfs`) and
    would plausibly serve `optatmo/` and `guider/` too, both of which do their own PSF
    moment work. Strongest candidate.
  - **the parquet combiner** — see above; concatenating chunked parquet with schema
    unification is not topic-specific, and `olr/`'s implementation is the better one.
  - **a MIW reader** — via the four `load_miw` copies; settles which product is
    canonical at the same time. Probably `aos/code/miw_io.py` rather than `common/`,
    since the MIW is an AOS concept.
  - **focal-plane binning/gridding helpers** that recur across `static_optics` and
    `coadd`; worth a closer look during the file-by-file pass.
- **Docstrings:** numpydoc with backticked types, per the root `CLAUDE.md`. Only a
  handful of files currently comply. Units and frame (OCS/CCS) belong in the parameter
  and return descriptions.
- **`aos/code/` has no test suite.** The one `test_*.py` file, `test_m1m3.py`, is a
  manual command-line probe of M1M3 thermal-gradient retrieval, not a unit test, and it
  needs live EFD access. Each behaviour decision above (the quality cut, the MIW reader,
  the parquet combiner) is a natural place for the first real tests, since each is a pure
  function over a table.
