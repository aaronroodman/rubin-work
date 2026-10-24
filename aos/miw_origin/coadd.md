# Study: `coadd` — FAM coadd vs the MIW, and retrieval bias

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (study)

> **Code:** `code/` of the `miw_origin` study (formerly `aos/code/coadd/`); the builder is `rubinwork.products.coadds`
> **Output:** the `coadds` **product** — `<products>/coadds/<variant>/<build>/`, variants `d12-A_50_34_i_5rot` and `d12-A_50_34_i_5rot_v2`; the old tree is `output/coadd/<P>/{50_34,50_34_v2}/`

Comparison of per-block FAM wavefront coadds against the MIW, and the retrieval-bias
model developed to explain their disagreement.

The standard MIW build pools **all** FAM visits in a camera-rotator bin, across many
nights. If instead you coadd Z5–Z8 maps per FAM *block* (one contiguous run), the result
does not match the pooled MIW. The live explanation is **retrieval bias**: a small bias
`B` in the Danish donut fit redistributes power among primary/secondary/tertiary radial
orders, so what the build recovers depends on the visit mix.

The derivations are in [`../docs/miw_coadd_equations.md`](../docs/miw_coadd_equations.md), which is
the reference for the notation and the algebra. The handoff document lists claims from
earlier stages of this investigation that were subsequently retracted.

## What a block is, and which blocks are used

A **block** is one contiguous Full Array Mode (FAM) run at a single pointing — for the
standard programs, visits sharing a science program with (altitude, azimuth, rotator)
within 5 deg and a `seq_num` span under 36, so a 12-triplet run stays whole and missing
triplets inside it are tolerated. The T710 family, whose image plan drifts over a long
session, instead uses a 20 deg azimuth and rotator tolerance with no `seq_num` break. The
bounce programs T720 and T724 alternate between two pointings across a wide `seq_num`
span, so they are split by rounded (altitude, rotator) state: an elevation bounce between
70 deg and 40 deg becomes one block per state.

No elevation or rotator cut is applied — every block is detected and summarised. Three
selections then decide what is built:

| selection | default | effect |
|---|---|---|
| band (`--bands`) | inherits `mi_config.yaml` `defaults: filter: [i]` when omitted | which bands enter at all |
| `--min-visits` | 6 | blocks below this are summarised but not built |
| `--max-blur` | 1.4 arcsec | drops visits whose `median_blur_arcsec` exceeds it |

**Band selection dominates the 2025 sample.** FAM data from 2025 is mostly r-band (1229
visits) rather than i-band (689 visits), so an i-band-only run builds 130 blocks (40 from
2025, 90 from 2026) while an all-band run builds 216 (117 from 2025, 99 from 2026). 2026
is nearly insensitive to the choice, being 1103 i-band visits of 1204. An all-band run
therefore compares r, g, z, u and y blocks against an i-band Measured Intrinsic Wavefront
(MIW) reference, since the MIW build itself is i-band with `day_obs_max: 20260513`.

Separately from what is built, `build_used` flags the 16 blocks on 4 nights
(2026-03-15 to 2026-03-24, elevation about 70 deg) that went into the 5rot MIW: i-band,
program `BLOCK-T614_triplets`, elevation 65–75 deg, one of the 5 in-family rotator
windows, and `day_obs <= 20260513`. These are the self-comparison set, shaded on the time
series. It does not change with the band selection.

## Code

| file | role |
|---|---|
| `run_coadd_blocks_miw.py` | **moved** to `rubinwork/products/coadds/builders/` in phase 3a: the per-block grids and summaries it writes are the `coadds` product. A shim remains at the old path until phase 5. Build with the product Snakefile, or `aos/run_coadd_blocks_miw.sbatch` for a hand run against the old tree |
| `recompute_coadd_metrics.py` | recompute coadd-vs-MIW metrics on rebinned grids **without overwriting** existing output; the native fine-grid Pearson r / residual RMS are noise-limited |
| `run_fam_coadd_miw.py` | pipeline `fam_coadd_miw` rule — the coadd-vs-MIW map comparison |
| `analyze_miw_bias_regression.py` | estimate the retrieval-bias operator **L** from the coadd-vs-MIW residual maps (§4.2) |
| `analyze_umode_null_coupling.py` | the sharp form of the retrieval-bias test: split each visit's 126-coefficient DZ vector into three orthogonal pieces (§4.2) |
| `analyze_miw_dz_full_k.py` | fit the MIW to the full focal basis k=1..K; how much lives above the k≤6 the build fits |
| `analyze_miw_field_order.py` | is the MIW's high-field-order astigmatism the un-subtracted k>6 wavefront? |
| `analyze_dz_goodness_of_fit.py` | per-visit goodness-of-fit of the DZ fit vs the sensitivity matrix; 126 coefficients constrain only 34 v-modes, so it is massively overdetermined |
| `check_k_truncation.py` | (a) is k=1..6 enough to *specify* 50 DOF / 34 v-modes? (b) does the R subtraction remove higher-k content? |

## Inputs and outputs

The builder reads the per-chunk `chunks/*/visits.parquet` of its `fam_tables` build
(per chunk rather than the combined table, so a chunk carrying thermal telemetry keeps
it), `donuts.parquet`, and the `miw` build's per-donut sidecar `zk_intrinsic.parquet`,
which must exist. **Two upstreams, both recorded in the manifest** — the FAM build sets
which visits exist and the MIW sets what is subtracted, so neither alone makes the
numbers reproducible. The block-selection settings live in the product's `variants.yaml`.
It writes into `--out-dir`, which the product Snakefile sets to the build directory:

| product | content |
|---|---|
| `blocks_summary.parquet` | every detected block: program, `day_obs`, `seq_num` range, visit count, altitude/azimuth/rotator means, blur full width at half maximum (FWHM) in arcsec, band, in-family flag, 13 thermal means, and whether it was built |
| `coadd_metrics.parquet` | one row per built block: metadata, coadd ordinal, band, 13 thermal block-means, per-Zernike residual root mean square (RMS) in µm of wavefront and spatial Pearson r over both the full focal plane and the corner-wavefront-sensor annulus, and `umode_1..34` in µm of wavefront |
| `block_grids.npz` | stacked coadd and MIW Z5–Z8 grids, u-modes, metadata, and the Double Zernike (DZ) primitives — block-median raw wavefront `raw_dz`, residual `resid_dz`, and the subspace bases `U_eff` / `U_all` / `U_disc` |
| `block_grids_ext.npz` | coadd and MIW grids for the extra pupil Zernikes selected by `--ext-terms`, row-aligned by block |
| `coadd_blocks_miw_perblock.pdf` | one block per page: coadd, MIW and difference rows across Z5–Z8 |

Saving the DZ primitives is what makes the u-mode amplitudes, the removed part of the
wavefront, and the reachable-but-discarded against null-space split of the residual all
derivable without the LSST stack.

`recompute_coadd_metrics.py` reads **only** `block_grids.npz` plus the sibling
`blocks_summary.parquet` (for `program`, which the npz does not carry), so it runs on the
laptop from synced output with nothing but numpy, pandas, matplotlib and scipy. It writes
`coadd_metrics_rebin<f>.parquet` per `--rebin` factor and one combined
`coadd_timeseries.pdf` — **study** outputs, not part of the build, so `--variant` requires
`--out-dir` and will not write into the product directory it read. Rebinning matters because the native fine-grid Pearson r and
residual RMS are noise-limited by the few donuts per cell and the large single-donut
wavefront scatter, understating the real structural agreement — spatial Pearson r rises
from 0.54 to 0.74 at 3×3 rebinning on a test block. `--analysis-rebin` (default: the
coarsest factor) selects which metric set feeds the correlation, u-mode-displacement and
machine-learning pages.

The other scripts read the `miw` build's `fits.parquet` (the MI-refit DZ) and the coadd
products, and write per-analysis PDFs. Three of them —
`recompute_coadd_metrics.py`, `analyze_miw_bias_regression.py` and
`analyze_miw_field_order.py` — built `output/<long param_set>/coadd_50_34/` before phase
3a, a layout the tree lost when `dir_name` arrived, so they resolved nothing on disk; each
now takes `--variant` (or `COADD_VARIANT`, in the one script with no argument parser) and
resolves the build through the catalog.

## State and open questions

- **The retrieval-bias hypothesis is the live explanation**; `L ≡ 0` iff the retrieval is
  unbiased, which is what `analyze_miw_bias_regression.py` tests.
- **`block_grids.npz` and `coadd_metrics_rebin<f>.parquet` must come from one run.**
  `analyze_miw_field_order.py` masks the metrics table with a boolean array sized from the
  npz, so pairing products built under different band selections raises `IndexError`
  rather than a clear message. Superseded output is kept under
  `output/coadd/<P>/50_34/archive/`, each archive directory carrying a note of its band
  selection and block count.
- **Never correlate against total `‖a‖`** — open-loop focus drift dominates it; use
  per-mode signed `a`. See the handoff.

### Findings (2026-09-03)

The headline results (focal-order truncation, the per-visit `L` test, the empirical
covariance) are in the handoff §4.1–§4.7. These are the supporting results not
recorded there.

**Focal-order content of the MIW** (`analyze_miw_dz_full_k.py` →
`output/miw_dz_full_k.pdf` and `miw_dz_full_k_coeffs.parquet`, the fitted coefficients
`w_kj`, 21 pupil × 45 focal, µm of wavefront). Each OCS MIW map is fitted to focal Noll
k=1..45 over the 3985 finite cells, focal radius 1.8 deg.
- The k ≤ 30 boundary that ts_ofc imposes cuts focal radial order n=7 (k=29..36) in
  half, so the boundary is slightly asymmetric.
- The 0.079 of MIW power (dimensionless, power-weighted over the 21 pupil Zernikes)
  left unfit even at k ≤ 45 is probably not smooth field structure, since 45 terms
  over 3985 cells would capture that. Per-sensor or CCD-height-scale structure is the
  likely carrier; untested.

**Block-level regression of residual maps on Δa** (`analyze_miw_bias_regression.py`).
Δa adds +0.028 in cross-validated (CV) R² over thermal alone (handoff §4.5). Per-mode
drop-one ΔR² over the thermal model (dimensionless): u31 +0.014, u25 +0.011, u9 +0.010,
u30/u29/u27/u33 ≈ +0.009 each. u9, u13 and u14 carry signal with near-zero k>6
leakage. Under the unbiased model a low-leakage mode cannot drive the residual, so
signal there beyond thermal favours retrieval bias.

**Per-visit goodness of fit** (`analyze_dz_goodness_of_fit.py`). With the empirical
covariance from the adjacent-consecutive-visit difference instead of the block mean,
chi²_50/dof = 5.1 (dof = 76) and chi²_34/dof = 7.0 (dof = 92), against 4.2 and 5.2 from
the block-mean covariance. Both are a real inconsistency.

**Coadd-vs-MIW agreement.** Median spatial Pearson r between block coadd and MIW over
blocks (dimensionless): 0.660 on the native grid, 0.739 at 3×3 rebinning. Environmental
telemetry predicts per-block r with gradient-boosted-tree (GBT) CV R² = +0.619
(leave-night-out GroupKFold, n = 123 non-build blocks after merging the camera-body
temperatures); an earlier +0.68 on 127 blocks predates that merge. Signal-to-noise-only predictors
(FWHM, donut count, visit count) give a negative CV R², and `z_gradient` (M1M3 vertical
thermal gradient) dominates permutation importance, with r rising as `z_gradient` → 0,
the build state.

**Retracted.** `build_used` once flagged 81 blocks (in-family rotator and
`day_obs >= 20260101`), ignoring band, program and elevation. The correct selection is
the 16 blocks described above.

**Practical points.**
- No map cell is finite in every block, since per-block coverage differs. Keep cells with
  coverage ≥ 0.97 (fraction of blocks) and impute the rest with the column mean
  (`analyze_miw_bias_regression.py --min-cov`).
- Each `d[key]` on a numpy `NpzFile` decompresses the whole array again; extract arrays
  once before looping.

### Camera temperature and MIW agreement (2026-08-25)

Camera-body temperatures come from the raw Engineering Facility Database (EFD) topic
`lsst.MTCamera.utiltrunk_body` ([`../docs/telemetry.md`](../docs/telemetry.md#where-camera-temperatures-come-from)).
`recompute_coadd_metrics.py` averages them per block from the merged visits table.

- **Absolute camera temperatures barely correlate with the coadd-vs-MIW Pearson r.** The
  default feature is therefore the gradient against the camera ambient sensor,
  `cam_<field>_dAmb = mean(cam_<field>) − mean(cam_AmbAirtemp)` per block (deg C), and only
  `cam_AverageTemp_dAmb` is kept (`--camera-ref`, `--camera-raw`, `--camera-all` change this).
- **`cam_AverageTemp_dAmb` mostly co-moves with `z_gradient`.** Over n = 176 non-build
  blocks: Spearman rho(r, `cam_AverageTemp_dAmb`) = −0.49 (camera warmer than ambient,
  worse agreement), but rho(`cam_AverageTemp_dAmb`, `z_gradient`) = −0.53. Partial
  correlations: r with `cam_AverageTemp_dAmb` given `z_gradient` = −0.26 (p = 5e-4); r with
  `z_gradient` given `cam_AverageTemp_dAmb` = +0.46 (p = 2e-10). Linear R² (dimensionless)
  rises only from 0.372 (`z_gradient` alone) to 0.389 with the camera term added.
- In the environmental GBT, `cam_AverageTemp_dAmb` has zero or negative permutation
  importance and `z_gradient` about 0.70 (dimensionless, importance). Using all 23 camera
  `_dAmb` sensors lowers the CV R² from 0.68 to 0.575 (collinear dilution); keeping only
  `cam_AverageTemp_dAmb` gives 0.62.

`z_gradient` remains the single dominant driver of coadd-vs-MIW agreement; the camera
temperature is not an independent one.

## Running

The block coadd is a batch job: it holds every block's donut table plus the stacked
coadd/MIW arrays, the DZ primitives and the extra-Zernike maps, which is enough to exhaust
an RSP notebook pod. `run_coadd_blocks_miw.sbatch` (in the `aos/` root) bakes in the
current collection, `--mi-name pathA_50_34_i_5rot` and `--ext-terms secondary`; any option
on the sbatch line overrides, since argparse takes the last occurrence. Submission is
**MUST-ASK** — hand over the submit plus `tail -f` pair rather than submitting.

```bash
cd ~/notebooks/rubin-work/aos
sbatch run_coadd_blocks_miw.sbatch --bands g r i z u y
```

The metrics recompute and the analyses need no batch job and no LSST stack:

```bash
cd ~/notebooks/rubin-work/aos
python miw_origin/code/recompute_coadd_metrics.py --rebin 1 3
python miw_origin/code/analyze_dz_goodness_of_fit.py --help
```

## Notebooks

None. This study is entirely scripted.

## See also

- [`../docs/miw_coadd_equations.md`](../docs/miw_coadd_equations.md) — **the** reference for the equations
- [`miw_investigation_handoff.md`](miw_investigation_handoff.md) — retracted claims
- [`../docs/studies/miw.md`](../docs/studies/miw.md), [`static_optics.md`](static_optics.md)
