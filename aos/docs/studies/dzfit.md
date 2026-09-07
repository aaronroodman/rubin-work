# Study: `dzfit` — Double-Zernike fitting and data quality

> **Status:** current · **Last updated:** 2026-09-06 · **Kind:** reference (study)

Validation of the per-visit Double Zernike (DZ) fit, and quality checks on the Full
Array Mode (FAM) donut data it is fitted to. This is the stage before any Measured
Intrinsic Wavefront (MIW) exists: the DZ fit is made against the *batoid design*
intrinsic, and these products show whether that fit is trustworthy and whether the
input data is sound.

Everything here depends only on the `param_set` — the Butler collection paired with a
processing variant — not on any MIW build.

## Code

| file | role |
|---|---|
| `run_dz_plots.py` | pipeline `plots` rule — data / DZ model / residual trio comparisons across the focal plane, plus fit-parameter and residual pages |
| `run_aberration_pairs.py` | pipeline `aberration_pairs` rule — per-donut primary→secondary aberration-pair correlations (defocus→spherical, astigmatism→2nd astigmatism, and so on) |
| `plot_visits_summary.py` | per-`param_set` visit coverage: elevation versus rotator angle, one panel per band |
| `check_chunk.py` | pre-flight check on a date chunk before `mktable` runs — visits present in the Consolidated Database (ConsDB) but missing from the Butler collection |
| `inspect_visit_provenance.py` | Butler provenance consistency across a `param_set`'s date chunks |
| `compare_to_archive.py` | combined pipeline outputs versus an archived copy, after a from-scratch run |

The plotting library `dz_plotting.py` sits at `aos/code/` rather than here, because the
[`correlations`](correlations.md) study uses it too.

The DZ fit *itself* is not in this repository — it is `dz_fitting.py` in the external
`ts_intrinsic_wavefront` package, called by the pipeline's `fit` rule. This study covers
the validation of that fit, not its implementation.

## Inputs and outputs

Reads the combined `output/<ps>/{donuts,fits,visits}.parquet`. Writes to
`output/<ps>/dzfit/`:

| product | from |
|---|---|
| `trio_comparison_all.pdf`, `trio_comparison_k1to6_all.pdf` | `plots` |
| `fit_params_resid_z1toz6_all.pdf` | `plots` |
| `aberration_pairs.pdf`, `aberration_pairs_summary.parquet` | `aberration_pairs` |
| `visits_check.pdf` | `plot_visits_summary.py` |

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until plots
python code/dzfit/check_chunk.py --help          # before mktable, on a new chunk
python code/dzfit/plot_visits_summary.py --help
```

The `plots` rule loads the full donut table and is memory-heavy; the Snakefile's
`mem_mb` declaration serializes it deliberately. Everything in this study runs wherever
the parquet tables exist — no LSST stack needed — except `check_chunk.py` and
`inspect_visit_provenance.py`, which query the Butler and ConsDB.

## Notes

`run_aberration_pairs.py` uses a quartile-of-primary ordinary least-squares slope. That
predates the standing preference for robust fits; see the root `CLAUDE.md`.

## See also

- [`miw.md`](miw.md) — what happens after this stage, once the measured intrinsic exists
- [`correlations.md`](correlations.md) — the same DZ coefficients after MIW subtraction
- [`../double_zernike_convention_validation.md`](../double_zernike_convention_validation.md) — DZ index and normalization conventions
- [`../miw_pipeline.md`](../miw_pipeline.md) — the `plots` and `aberration_pairs` rules in context
