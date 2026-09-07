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

### `quality_cut` — RESOLVED 2026-09-07

Was four copies under one name implementing **two different cuts**:
`run_dz_correlations.py` and `run_thermal_correlations.py` dropped visits flagged
`bad_fit` before the coefficient cut; `run_dz_explained.py` and
`run_vmode_correlations.py` did not, so they analysed ~24 known-bad fits that the other
two excluded. `bad_fit` marks visits with too few donuts to constrain the k=1..6
focal-plane terms, and it is nearly independent of the large-coefficient cut, so those
visits survived it.

Replaced by `fam_quality_selection` in `aos/code/fam_selection.py` (renamed from
`dz_columns.py`, which now also holds `dz_coeff_columns`). It is the single place that
decides which FAM visits are usable:

- **always** drops any visit with a true value in *any* `*bad_fit` column, combined with
  a logical OR rather than trusting one flag;
- **optionally** cuts on the maximum absolute DZ coefficient (`max_coeff_um`, µm of
  wavefront) and on the maximum per-visit median donut blur (`max_blur_arcsec`, arcsec).
  Both default to `None`, so adding a cut is an explicit choice visible at the call site;
- reports a count per cut applied.

All four scripts now call it, reading both thresholds from `analysis_config.yaml`, where
`max_blur_arcsec` is present but commented out in each of the four sections.

Verified on `pathA_50_34_i_5rot/fits.parquet` (1126 visits) at the configured
`max_coeff_um = 2.0` µm: `run_dz_correlations` and `run_thermal_correlations` are
**unchanged**, selecting the same 1101 visits with an identical index;
`run_dz_explained` and `run_vmode_correlations` go from 1125 to 1101, correctly losing
the 24 bad-fit visits. **Their existing output predates this fix and should be
regenerated.**

Blur, for reference on that param_set: range 0.62–1.41 arcsec, median 0.86. A
`max_blur_arcsec` of 1.0 would drop 185 of 1126 visits; 1.1 would drop 71.

### `load_miw` — RESOLVED 2026-09-07

Was four copies across `static_optics`. Three (`run_m3_backprojection_miw`,
`run_miw_backprojection_surfaces`, `run_miw_joint_fit`) were identical apart from
returning a 2- or 3-tuple; `camera_gravity_maps` used a laxer row cut.

Replaced by `load_miw` in `aos/code/miw_io.py`, which returns `(pts, zk, df)` — field
positions in degrees, Noll-indexed Zernikes in µm of wavefront, and the surviving rows.
`require=` controls which Zernikes must be finite for a row to be kept, defaulting to all
of `js`.

That parameter exists because the row cut genuinely differed, and for a good reason:
`camera_gravity_maps` plots only Z5–Z8, and requiring all 21 Zernikes finite would
discard **108 field points** where `Z4_OCS` (defocus, CCD-height sensitive) is non-finite
but Z5–Z8 are fine. It now passes `require=(5, 6, 7, 8)` and keeps the same 3969 rows as
before. The three back-projection scripts get a bit-identical grid — verified same rows,
same `pts`, same `zk` at stride 1 and 8 — so no output needs regenerating.

**Deliberately parquet-only.** The MIW also lives in a Butler as an
`lsst.ip.isr.IntrinsicZernikes` calibration (dataset type `intrinsicZernikes`, one per
detector). That is *not* wrapped, because its only consumer,
`aos_miw_cwfs_intrinsic_check.ipynb`, exists to verify that the ingested calibration
reproduces what ts_wep computes — so it must call `getIntrinsicZernikes` directly, per
detector, with the real `rotTelPos`, rather than through a wrapper that could mask the
behaviour under test. A future Butler-backed reader should be a separate
`load_miw_calib()` returning the calibration object, not a mode of this function.

## `combine_parquets.py` — RESOLVED 2026-09-07

The backlog said two versions; there were **three**, and the one the AOS pipeline
actually runs was neither of the ones being compared.

- `aos/code/combine_parquets.py` — **dead code**. Nothing referenced it: `aos/Snakefile`
  calls `{WF_BIN}/combine_parquets.py`, i.e. the external package's copy. Deleted.
- `ts_intrinsic_wavefront/bin.src/combine_parquets.py` — what the AOS pipeline runs, and
  it lacked the sentinel logic. **Patched** (branch `tickets/RSO-809`, commit `f4f23b6`),
  now byte-identical to the `olr/` version. `bin/` is a scons-generated copy of
  `bin.src/`, so `bin.src/` is the file to edit.
- `olr/code/combine_parquets.py` — already correct, independently invoked by
  `olr/Snakefile`. Unchanged.

The bug, demonstrated on two chunks with columns (a, b) plus a 0-row sentinel with only
(a): the unified schema is the **intersection** of all input schemas, so the sentinel
silently shrank the output from 2 columns to 1 — reported as a `note:`, not an error.
Real column loss.

```
before:  note: dropped column 'b' — absent from 1/3 inputs
         combined 3 files: 5 rows (1 columns, 1 dropped)
after:   note: skipping empty input empty.parquet (0 rows)
         combined 2 files: 5 rows (2 columns, 0 dropped)
```

Verified: with no sentinels the output is bit-identical to before (pyarrow
`Table.equals`); with every input empty, a valid readable 0-row file is written so the
DAG target exists.

**Not promoted to `common/`.** The packaged copy runs via a `bin/` shim and cannot import
from this repo, so it has to stay standalone; putting a third copy in `common/` would add
a file to keep in sync rather than remove one. The remaining duplication is
`olr/code/combine_parquets.py` against the package, and those two are now identical.

**Package commit is not pushed** — `ts_intrinsic_wavefront` is a separate repo.

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

## Review the `psf` study — what is it for?

Deferred from the Phase 7 output reorganization (2026-09-06). Aaron's note: *"I now
don't even remember what this study did."* Its output was moved to `<ps>/psf/` as a
holding location, not a considered placement.

What is there: `code/psf/run_psf_fp_maps.py` (634 lines) plus the shared
`code/psf_render.py`, and 14 PDFs in
`output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/psf/`, one per case:

| case | what it renders |
|---|---|
| `miw_i` | PSF from the MIW wavefront |
| `fam50_i`, `fam22_i` | PSF from the FAM-recovered state, 50-DOF and 22-DOF schemes |
| `mimic50_i`, `mimic22_i` | PSF from the WFS-mimic corner recovery |
| `loop{22,50}_i_{miw,tabulated}_{ordered,random}_nplustwo_g0.3` | 8 closed-loop simulations: gain 0.3, N+2 latency, MIW vs tabulated intrinsic, ordered vs random visit order |
| `validate_i` | validation of the FWHM formula against rendered PSFs |

Questions to settle:

- **Is the closed-loop simulation still wanted?** 8 of the 14 outputs are loop cases.
  That is a control-simulation study, arguably distinct from "render a PSF from a
  wavefront", and might deserve its own study name (`aosloop`?) or retirement.
- **Where should output live?** It is at `<ps>/psf/` now. The script reads **two**
  different `<mi>` builds in one run — `--split-mi` (default `pathA_50_34_i_5rot`,
  supplying `intrinsic_split_maps.parquet`) and `--fam-mi` (default `pathA_50_34_i`,
  supplying `zk_intrinsic.parquet`) — and several cases use no MIW at all, so it does
  not belong under a single `<mi>`. If the loop cases are split out, the remainder may
  be simple enough to key properly.
- **Is `psf_render.py` a `common/` candidate?** It is GalSim `OpticalPSF` + Kolmogorov
  + HSM with nothing AOS-specific, already used by `psf` and `cwfs`, and `optatmo/` and
  `guider/` do similar moment work. Decision deferred at Aaron's request.
- **One live bug**: the NaN-truthy colour-scale guard at
  `code/psf/run_psf_fp_maps.py:141` (see above).
