# Study: `correlations` — what does the residual DZ correlate with?

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** After the measured intrinsic is subtracted, what structure is left in the
per-visit Double-Zernikes — internal DZ↔DZ coupling, v-mode structure, or telemetry?

All four scripts run on the **MI-refit** residual (`output/<ps>/<mi>/fits.parquet`), not
the raw DZ. All four are pipeline rules. Knobs live in `analysis_config.yaml`, kept
separate from `mi_config.yaml` so editing an analysis knob never re-triggers a slow
intrinsic build.

## Code

| file | role |
|---|---|
| `run_dz_correlations.py` | DZ_kj ↔ DZ_k'j' Pearson heatmap, top-\|r\| scatters, astigmatism-symmetry pairs, conjugate-orbit grids, Fisher-z significance. Also a `_optcorr` variant on the post-OFC-correction residual |
| `run_vmode_correlations.py` | project the MI-subtracted DZ onto the OFC SVD and correlate v-modes, for both 50/34 and 22/12 schemes |
| `run_thermal_correlations.py` | DZ_kj × EFD temperature-variable Pearson heatmap plus per-term scatter pages |
| `run_dz_explained.py` | per-visit fraction of the measured DZ explained by the OFC sensitivity subspace, 22/12 and 50/34 |

## Outputs

`<mi>/plots/dz_correlations{,_optcorr}.{pdf,_pairs.parquet}`,
`vmode_correlations_{50_34,22_12}.pdf` + summary parquets,
`thermal_correlations.pdf` + `_summary.parquet`, `dz_explained.{pdf,parquet}`.

**Phase 7 will move these** out of the flat `<mi>/plots/` dump into
`<mi>/correlations/`.

## Statistical cautions

These are correlation studies, so the reporting rules matter more than usual:

- **Robust methods, and ask which** before implementing. Report **both** Pearson r and
  Spearman rho (`robust-fits-aos`).
- Every number needs its quantity name and units, or an explicit "dimensionless" with
  numerator and denominator named. `chi2` always as `chi2/dof` with dof stated.
  Correlations need the statistic, both variables with units, and `n`.
- The review backlog flags real issues here that may still be live: significance computed
  with a global complete-case `n` rather than per-pair `n`; an `r` clamp near ±1 that
  manufactures huge significances on the `_optcorr` run; and no detrending in the thermal
  correlations, where both temperature and AOS state drift through the night. See
  [`../status/code_review_findings.md`](../status/code_review_findings.md) — verify
  against current source, the line anchors are stale.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until dz_correlations
./run_snake.sh -n            # check what is stale first
```

`run_vmode_correlations.py` and `run_dz_explained.py` build the OFC SVD, so they need
`lsst.ts.ofc` — RSP only.

## See also

- [`smatrix_vmode.md`](smatrix_vmode.md) — where the v-modes come from
- [`telemetry.md`](telemetry.md) — where the temperature columns come from
- [`../miw_pipeline.md`](../miw_pipeline.md#phase-3--analyses-on-the-mi-refit-fits-per-param_set--mi_name)
