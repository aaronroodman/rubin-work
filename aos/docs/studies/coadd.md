# Study: `coadd` — FAM coadd vs the MIW, and retrieval bias

> **Status:** current · **Last updated:** 2026-09-14 · **Kind:** reference (study)

Comparison of per-block FAM wavefront coadds against the MIW, and the retrieval-bias
model developed to explain their disagreement.

The standard MIW build pools **all** FAM visits in a camera-rotator bin, across many
nights. If instead you coadd Z5–Z8 maps per FAM *block* (one contiguous run), the result
does not match the pooled MIW. The live explanation is **retrieval bias**: a small bias
`B` in the Danish donut fit redistributes power among primary/secondary/tertiary radial
orders, so what the build recovers depends on the visit mix.

The derivations are in [`../miw_coadd_equations.md`](../miw_coadd_equations.md), which is
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
| `run_coadd_blocks_miw.py` | per-FAM-block MIW-style Z5–Z8 maps (Path A, 50/34, 3 iterations); has an `.sbatch` |
| `recompute_coadd_metrics.py` | recompute coadd-vs-MIW metrics on rebinned grids **without overwriting** existing output; the native fine-grid Pearson r / residual RMS are noise-limited |
| `run_fam_coadd_miw.py` | pipeline `fam_coadd_miw` rule — the coadd-vs-MIW map comparison |
| `analyze_miw_bias_regression.py` | estimate the retrieval-bias operator **L** from the coadd-vs-MIW residual maps (§4.2) |
| `analyze_umode_null_coupling.py` | the sharp form of the retrieval-bias test: split each visit's 126-coefficient DZ vector into three orthogonal pieces (§4.2) |
| `analyze_miw_dz_full_k.py` | fit the MIW to the full focal basis k=1..K; how much lives above the k≤6 the build fits |
| `analyze_miw_field_order.py` | is the MIW's high-field-order astigmatism the un-subtracted k>6 wavefront? |
| `analyze_dz_goodness_of_fit.py` | per-visit goodness-of-fit of the DZ fit vs the sensitivity matrix; 126 coefficients constrain only 34 v-modes, so it is massively overdetermined |
| `check_k_truncation.py` | (a) is k=1..6 enough to *specify* 50 DOF / 34 v-modes? (b) does the R subtraction remove higher-k content? |

## Inputs and outputs

`run_coadd_blocks_miw.py` reads the per-chunk `output/<ps>/chunks/*/visits.parquet` (per
chunk rather than the combined table, so a chunk carrying thermal telemetry keeps it),
`donuts.parquet`, and the MIW sidecar `output/<ps>/<mi>/zk_intrinsic.parquet`, which must
exist. It writes into `output/<ps>/<out-name>/`:

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
`coadd_timeseries.pdf`. Rebinning matters because the native fine-grid Pearson r and
residual RMS are noise-limited by the few donuts per cell and the large single-donut
wavefront scatter, understating the real structural agreement — spatial Pearson r rises
from 0.54 to 0.74 at 3×3 rebinning on a test block. `--analysis-rebin` (default: the
coarsest factor) selects which metric set feeds the correlation, u-mode-displacement and
machine-learning pages.

The other scripts read `output/<ps>/<mi>/fits.parquet` (the MI-refit DZ) and the
`coadd_50_34/` products, and write per-analysis PDFs.

## State and open questions

- **The retrieval-bias hypothesis is the live explanation**; `L ≡ 0` iff the retrieval is
  unbiased, which is what `analyze_miw_bias_regression.py` tests.
- **`block_grids.npz` and `coadd_metrics_rebin<f>.parquet` must come from one run.**
  `analyze_miw_field_order.py` masks the metrics table with a boolean array sized from the
  npz, so pairing products built under different band selections raises `IndexError`
  rather than a clear message. Superseded output is kept under
  `output/<ps>/coadd_50_34/archive/`, each archive directory carrying a note of its band
  selection and block count.
- **Never correlate against total `‖a‖`** — open-loop focus drift dominates it; use
  per-mode signed `a`. See the handoff.

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
python code/coadd/recompute_coadd_metrics.py --rebin 1 3
python code/coadd/analyze_dz_goodness_of_fit.py --help
```

## Notebooks

None. This study is entirely scripted.

## See also

- [`../miw_coadd_equations.md`](../miw_coadd_equations.md) — **the** reference for the equations
- [`../status/miw_investigation_handoff.md`](../status/miw_investigation_handoff.md) — retracted claims
- [`miw.md`](miw.md), [`static_optics.md`](static_optics.md)
