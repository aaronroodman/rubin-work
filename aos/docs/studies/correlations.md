# Study: `correlations` — what does the residual DZ correlate with?

> **Status:** current · **Last updated:** 2026-09-23 · **Kind:** reference (study)

> **Code:** `code/correlations/` · **Notebooks:** `notebooks/correlations/`
> **Output:** `output/correlations/<P>_<M>/`, `output/correlations/<P>/aberration_pairs.*`


Correlation analysis of the per-visit Double Zernike (DZ) coefficients remaining after
the measured intrinsic is subtracted: against each other, against Optical Feedback
Control (OFC) v-modes, and against telescope telemetry. Also the per-donut
primary→secondary aberration-pair correlations, on the single-Zernike values rather than
the DZ fits.

The first four scripts run on the **MI-refit** residual (`output/miw/<P>_<M>/fits.parquet`),
not the raw DZ. All five are pipeline rules. Knobs live in `analysis_config.yaml`, kept
separate from `mi_config.yaml` so editing an analysis knob never re-triggers a slow
intrinsic build.

## Code

| file | role |
|---|---|
| `run_dz_correlations.py` | DZ_kj ↔ DZ_k'j' Pearson heatmap, top-\|r\| scatters, astigmatism-symmetry pairs, conjugate-orbit grids, Fisher-z significance. Also a `_optcorr` variant on the post-OFC-correction residual |
| `run_vmode_correlations.py` | project the MI-subtracted DZ onto the OFC SVD and correlate v-modes, for both 50/34 and 22/12 schemes |
| `run_thermal_correlations.py` | DZ_kj × EFD temperature-variable Pearson heatmap plus per-term scatter pages |
| `run_dz_explained.py` | per-visit fraction of the measured DZ explained by the OFC sensitivity subspace, 22/12 and 50/34 |
| `run_aberration_pairs.py` | per-donut primary→secondary aberration pairs (defocus→spherical, astigmatism→2nd astigmatism, and so on), split into quartiles of the primary |

`run_aberration_pairs.py` works on the **Phase-1** per-donut `zk_<coord>` values in
`donuts.parquet`, so it needs no `mi_name` and writes to `output/correlations/<P>/`.
It streams the donut table by row group.

The focal-plane-uniform defocus DZ(k=1, j=4) against Telescope Mount Assembly (TMA) truss
temperature is **not** part of this study: it lives in the top-level `thermal_focus/` topic
(`../../../thermal_focus/docs/thermal_focus.md`), which reads the value-added database rather
than `aos/` output. The LUT ↔ Trim anti-correlation it established — a Huber slope of
−1.089 ± 0.014 (dimensionless, v1 from the hexapod look-up table per unit v1 from the Trim)
on the camera-hexapod dz axis — is recorded there. See
[`../telemetry.md`](../telemetry.md) for the LUT / Trim / Tweak column mapping.

## Outputs

`<mi>/plots/dz_correlations{,_optcorr}.{pdf,_pairs.parquet}`,
`vmode_correlations_{50_34,22_12}.pdf` + summary parquets,
`thermal_correlations.pdf` + `_summary.parquet`, `dz_explained.{pdf,parquet}`, and
`<ps>/correlations/aberration_pairs.{pdf,_summary.parquet}`.

The first four currently share `<mi>/plots/` with the bounce and coadd output; splitting
them per study is outstanding work.

## Statistical cautions

These are correlation studies, so the reporting rules matter more than usual:

- **Robust methods, and ask which** before implementing. Report **both** Pearson r and
  Spearman rho (`robust-fits-aos`). `run_aberration_pairs.py` uses a quartile-of-primary
  ordinary least-squares slope, which predates that preference.
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

## Notebooks

The five analyses above are scripts, all five Snakemake rules. One notebook sits alongside
them in `notebooks/correlations/`.

### `consdb_vs_efd_aos_dof_20260513.ipynb`

The Engineering Facility Database (EFD) as the source of record for the AOS degree-of-freedom
Trim and hexapod look-up table (LUT), and where the Consolidated Database (ConsDB) copy of
those quantities is filled. Written for colleagues outside this project, so it imports **only
released LSST code** — `lsst.summit.utils`, `lsst_efd_client` — and nothing from `rubin-work`;
the client construction is inlined rather than taken from `common/telemetry_clients.py`.
Keeping that constraint is the point of the notebook, so any edit that adds a repo import
defeats it.

One night, `day_obs = 20260513`, chosen for its mix of `acq`, `cwfs` and `science` exposures.

**The EFD side.** Three checks establish that the terms are correctly identified before any
ConsDB comparison is made.

`MTHexapod.logevent_compensationMode` is checked first, because `compensationOffset` publishes
a computed LUT value whether or not the hexapod acts on it. Of the 822 exposures, 737 ran with
compensation enabled and 85 with it disabled — and the 85 are entirely `bias` and `dark`
frames, so every `science`, `acq`, `cwfs` and `flat` exposure of the night had the LUT applied.
It is a state-change event (nine events for the camera hexapod over a seven-day window), so it
needs an as-of lookup with a lookback of days, not a per-exposure join.

The identity `compensatedPosition = uncompensatedPosition + compensationOffset` holds
**exactly**: maximum residual 0 µm on x, y, z and 0 deg on u, v, w, on both hexapods, over 261
(M2) and 427 (camera) matched event triples, unchanged when restricted to compensation-enabled
events. So `uncompensatedPosition` is the accumulated Offset (Trim) alone, `compensationOffset`
is the LUT alone, and the LUT has two independent EFD routes. The matching tolerance is the one
trap: a single-stage match with a loose window pairs values across a hexapod move during a slew
and shows spurious residuals of hundreds of µm, which is a property of the pairing rather than
of the identity. The notebook matches in two nearest-in-time stages and reports both a 1 s and a
5 s tolerance.

**The ConsDB gaps.** The Trim (`mt_logevent_aggregated_dof`) reaches 24.1% of 510 science
exposures and **0.0% of all 114 `acq` and all 110 `cwfs` exposures**, confirming the
`img_type='science'` gate. The hexapod LUT (`compensation_offset`) is a few percent on all
three on-sky types for the camera hexapod and under a percent for M2 — sparse everywhere and
*not* img_type-gated, a different failure mode. There is **no ConsDB column for
`compensatedPosition`**, so a physical position can be rebuilt from the ConsDB only where both
term families happen to be populated.

**What the ConsDB columns actually hold**, measured by comparing each pivoted family against
all three `MTHexapod` topics rather than inferred from its name:

- `{camera,m2}_hexapod_aos_corrections_*` is the **accumulated Offset**, closest to
  `uncompensatedPosition` on 6 of 6 translation axes with the runner-up topic wrong by a factor
  of 11 to 264. The name suggests a per-iteration correction; it is not one, and the magnitudes
  agree — thousands of µm on x and y is a standing alignment.
- `{camera,m2}_hexapod_compensation_offset_*` is the **LUT**, closest to `compensationOffset`
  on 6 of 6 axes.

**Where populated, the ConsDB values are correct.** The LUT difference from the EFD as-of value
at `obs_start` (about 10 µm median on the camera hexapod x axis) collapses to 1.2 µm or less
against the best-matching `compensationOffset` event inside the exposure, and to exactly 0 µm
for M2 — a fraction of a percent of the 1959 µm (x), 1730 µm (y) and 4000 µm (z) range the LUT
covers in the night. The ConsDB value is a sample of the same stream at a different instant;
the LUT genuinely moves during an exposure.

The transform rule itself is settled by the Trim, which `telemetry.md` previously only
inferred. Of the 123 exposures carrying both values, 113 agree to within 1 × 10⁻⁶ µm and 10
disagree; on **all 10**, exactly one MTAOS event fell inside the exposure window and the ConsDB
value equals that event, while the EFD as-of value is the state at exposure start. The ConsDB
takes a **within-exposure** event, not the most-recent-before one, which explains both the
sparse coverage and the residual disagreement. For a step-function quantity like the Trim that
is the wrong instant.

Scatter plots of ConsDB against EFD, one panel per axis and coloured by `img_type`, are given
for all three families, so the coverage question and the value-agreement question are answered
separately.

## See also

- [`../../../thermal_focus/docs/thermal_focus.md`](../../../thermal_focus/docs/thermal_focus.md) — prediction of the uniform-defocus error from thermal telemetry
- [`../../../smatrix/docs/studies/vmode.md`](../../../smatrix/docs/studies/vmode.md) — where the v-modes come from
- [`telemetry.md`](../telemetry.md) — where the temperature columns come from
- [`../miw_pipeline.md`](../miw_pipeline.md#phase-3--analyses-on-the-mi-refit-fits-per-param_set--mi_name)
