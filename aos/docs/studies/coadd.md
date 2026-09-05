# Study: `coadd` — FAM coadd vs the MIW, and retrieval bias

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** Why does the per-FAM-block coadd disagree with the MIW?

The standard MIW build pools **all** FAM visits in a camera-rotator bin, across many
nights. If instead you coadd Z5–Z8 maps per FAM *block* (one contiguous run), the result
does not match the pooled MIW. The live explanation is **retrieval bias**: a small bias
`B` in the Danish donut fit redistributes power among primary/secondary/tertiary radial
orders, so what the build recovers depends on the visit mix.

This is the most mathematically developed study in the topic. **The derivations are in
[`../miw_coadd_equations.md`](../miw_coadd_equations.md)** — do not re-derive them, and
read the retracted-claims section of the handoff before repeating any earlier conclusion.

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

Reads `output/<ps>/<mi>/fits.parquet` (MI-refit DZ) and the `coadd_50_34/` products
(`block_grids.npz`, `coadd_metrics_rebin3.parquet`). Writes
`plots/fam_coadd_miw_maps.pdf`, `coadd_metrics_rebin*.parquet`, and per-analysis PDFs.

## State and open questions

- **The retrieval-bias hypothesis is the live explanation**; `L ≡ 0` iff the retrieval is
  unbiased, which is what `analyze_miw_bias_regression.py` tests.
- **`analyze_miw_field_order.py` currently fails on stale data.** In
  `output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/coadd_50_34/`, `block_grids.npz` has
  **umodes shape (130, 34)** while `coadd_metrics_rebin3.parquet` has **221 rows**
  (`build_used` sum 16) — the two products are from different runs, so a boolean mask
  raises `IndexError`. Either regenerate both from one run, or make the script assert
  `len(mt) == len(Um)` with a clear message. Recorded in
  [`../status/code_review_findings.md`](../status/code_review_findings.md).
- Its imports were fixed on 2026-09-05 (it previously loaded `ofc_svd` from a hardcoded
  `/Users/roodman` macOS path and could never run on S3DF).
- **Never correlate against total `‖a‖`** — open-loop focus drift dominates it; use
  per-mode signed `a`. See the handoff.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/recompute_coadd_metrics.py --help
python code/analyze_dz_goodness_of_fit.py --help
```

Batch form for the block coadd is `run_coadd_blocks_miw.sbatch` — **MUST-ASK**, hand over
the submit plus `tail -f` pair rather than submitting.

## See also

- [`../miw_coadd_equations.md`](../miw_coadd_equations.md) — **the** reference for the equations
- [`../status/miw_investigation_handoff.md`](../status/miw_investigation_handoff.md) — retracted claims
- [`miw.md`](miw.md), [`static_optics.md`](static_optics.md)
