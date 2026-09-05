# Study: `smatrix_vmode` — sensitivity matrix and mode structure

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** What does the OFC sensitivity matrix's mode structure actually let us
observe and control?

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
| `plot_vmode_dof_matrix.py` | render the OFC SVD **V** matrix (DOF composition of each v-mode) + singular-value spectrum for a scheme (default 22-DoF/12-v-mode) |
| `analyze_sensitivity_sparse.py` | which DOF drive the **secondary/tertiary** aberrations of each azimuthal family |
| `analyze_sparse_observability.py` | if the donut fit measures only **primary** aberrations (drop secondary/tertiary rows), which DOF stay observable? |

## Notebooks

`smatrix_vmode_info.ipynb` is the comprehensive treatment — SVD with `StateEstimator` +
custom-SVD validation, v-mode composition, wavefront signatures, control equations,
noise/gain, a normalization-scheme deep-dive (unit invariance), and DZ field patterns.
It consolidates the former `normalization_study` and `smatrix_doublez`.
`vmode_dof_ts_ofc.ipynb` covers v-mode/DOF normalization via the `ts_ofc`
`StateEstimator`; `jk_coverage_plots.ipynb` has 50-DOF SVD visualizations.

## Normalization — the trap

There are **two** ways to get v-modes and they do not agree unless you are careful:
`StateEstimator` (4-CWFS) versus `build_ofc_svd` (double-Zernike), with geometric
weights in play. There is a genuine degeneracy caveat. Read
`../../../notes/claude-memory/aos-vmode-normalization.md` and
[`../../../olr/docs/vmode_normalization.md`](../../../olr/docs/vmode_normalization.md)
before comparing v-modes computed two ways.

Sign and unit conventions (ZCS, bending-mode flips, degree angle units, the y-sign patch)
are settled in [`../../../smatrix/docs/conventions.md`](../../../smatrix/docs/conventions.md).

## State and open questions

- **`analyze_sensitivity_sparse.py` and `analyze_sparse_observability.py` hardcode a
  `/Users/roodman` macOS data path** and will fail on S3DF. Not yet fixed — see
  [`../status/code_review_findings.md`](../status/code_review_findings.md).
- The sparse-fit study found that **all DOF couple primary↔secondary at ±1**, and that
  production zeroes coma2/tref2 — see `../../../notes/claude-memory/sparse-fit-sensitivity.md`.
- `build_lut` currently projects the **Phase-1** `fits.parquet`, not the MI-refit one.
- Uses the private `svd._keep()` from ts_ofc in at least one place (flagged in the review
  backlog) — a fragile dependency on package internals.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until build_lut
python code/plot_vmode_dof_matrix.py --help
```

All of these need `lsst.ts.ofc` and `$TS_CONFIG_MTTCS_DIR` — RSP only, and note `ts_ofc`
is **not** in `lsst_distrib`.

## See also

- [`../../../smatrix/README.md`](../../../smatrix/README.md) — matrix construction
- [`correlations.md`](correlations.md) — the v-mode correlation consumer
- [`cwfs.md`](cwfs.md) — the corner OFC inverse uses the same SVD
