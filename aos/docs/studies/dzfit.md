# Study: `dzfit` — Double-Zernike fitting and data quality

> **Status:** current · **Last updated:** 2026-09-08 · **Kind:** reference (study)

Validation of the per-visit Double Zernike (DZ) fit of the Full Array Mode (FAM) donut
wavefront. This is the stage before any Measured Intrinsic Wavefront (MIW) exists: the
fit is made against the **batoid design intrinsic**, and these products show whether that
fit is trustworthy.

Two focal-plane expansions are fitted for every visit and every pupil Zernike, and both
are checked here:

| prefix | focal terms | columns in `fits.parquet` |
|---|---|---|
| `z1toz3` | Noll Z1..Z3 — piston and two tilts | `z1toz3_z<j>_c<k>` for k=1..3 |
| `z1toz6` | Noll Z1..Z6 — adds defocus and two astigmatisms | `z1toz6_z<j>_c<k>` for k=1..6 |

Each carries a formal error `_err` per coefficient and one robust residual scale
`z1toz<n>_z<j>_scale` per pupil Zernike, in µm of wavefront. The fit is a Huber
M-estimator with a least-squares fallback, on a focal-plane unit disk of radius
1.75 deg.

Everything here depends only on the `param_set` — the Butler collection paired with a
processing variant — not on any MIW build.

## Code

| file | role |
|---|---|
| `run_dz_fit_check.py` | pipeline `dz_fit_check` rule — residual metrics and plots for the k<=3 and k<=6 fits |
| `run_dz_plots.py` | pipeline `plots` and `residual_movie` rules — data / model / residual trio comparisons, fit-parameter pages, and the per-visit residual movie |

The plotting library `dz_plotting.py` sits at `aos/code/` because the
[`correlations`](correlations.md) study uses it too.

The DZ fit *itself* is `dz_fitting.py` in the external `ts_intrinsic_wavefront` package,
called by the pipeline's `fit` rule. This study covers the validation of that fit, not
its implementation.

### `run_dz_fit_check.py` — the residual the fit does not store

`fits.parquet` records coefficients, their formal errors and a robust scale, but no
per-donut residual. This script recomputes it, reproducing the fit's own definition

    resid = (zk_<coord> - zk_intrinsic_<coord>) - A(thx, thy) @ c

with `A` the focal-plane Noll basis and `c` the stored coefficients, then reduces it to
robust metrics per (visit, prefix, pupil Zernike). Reconstruction fidelity is confirmed
against the `_scale` column the Huber fit writes: Pearson r = +0.999665 (dimensionless)
with a median ratio of recomputed nMAD to stored scale of 0.9974 (dimensionless) over
1050 visit-Zernike pairs.

It streams `donuts.parquet` by row group and buffers per visit, so peak memory is set by
the largest row group rather than the 9.08-million-donut table.

Pages, in order: residual scatter per pupil Zernike with the k<=6 to k<=3 ratio; the
cross-check against the stored scale; coefficient errors against the residual scale;
mean residual focal-plane maps per prefix; k<=3 and k<=6 maps side by side with their
difference; coefficient distributions; coefficient time histories.

### `run_dz_plots.py` — trio maps and the residual movie

Reconstructs per-donut fit values, then produces measured / model / residual map trios
per pupil Zernike, fit-parameter pages grouped by pointing, and one residual-map frame
per visit rendered into `single_image_residuals.mp4` by ffmpeg. The `plots` rule skips
the movie; the `residual_movie` rule produces only the movie.

## Inputs and outputs

Reads the combined `output/<ps>/{donuts,fits,visits}.parquet`. Writes to
`output/<ps>/dzfit/`:

| product | from |
|---|---|
| `dz_fit_check.pdf`, `dz_fit_check.parquet` | `dz_fit_check` |
| `trio_comparison_all.pdf`, `trio_comparison_k1to6_all.pdf` | `plots` |
| `fit_params_resid_z1toz6_all.pdf` | `plots` |
| `single_image_residuals.mp4` | `residual_movie` |

`dz_fit_check.parquet` has one row per (visit, prefix, pupil Zernike), carrying donut
count, residual nMAD, RMS and median in µm of wavefront, and the deviation nMAD for
context.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until dz_fit_check
python code/dzfit/run_dz_fit_check.py --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
python code/dzfit/run_dz_fit_check.py --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x --max-visits 25
./run_snake.sh --until residual_movie
```

`residual_movie` renders one frame per visit and is not in `rule all`; ask for it
explicitly. `--skip-metrics` replots from an existing `dz_fit_check.parquet`, without the
focal-plane maps, which need the donut stream.

The `plots` rule loads the full donut table and is memory-heavy; the Snakefile's `mem_mb`
declaration serializes it deliberately. `dz_fit_check` streams instead and is declared
smaller. Everything in this study runs wherever the parquet tables exist — no Butler,
Consolidated Database (ConsDB) or Engineering Facilities Database (EFD) access needed.

## State and open questions

- `residual_movie` and `plots` both call `run_dz_plots.py`, which materializes the donut
  table through astropy; a streaming rewrite would remove the 12 GiB declaration.

## See also

- [`miw.md`](miw.md) — what happens after this stage, once the measured intrinsic exists
- [`correlations.md`](correlations.md) — the same DZ coefficients after MIW subtraction, and the per-donut aberration-pair analysis
- [`fam_processing.md`](fam_processing.md) — the chunk build and telemetry this study's inputs come from
- [`../double_zernike_convention_validation.md`](../double_zernike_convention_validation.md) — DZ index and normalization conventions
- [`../miw_pipeline.md`](../miw_pipeline.md) — the `plots`, `dz_fit_check` and `residual_movie` rules in context
