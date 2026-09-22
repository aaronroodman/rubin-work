# Study: `bounce` — elevation and rotator bounce tests

> **Status:** current · **Last updated:** 2026-09-22 · **Kind:** reference (study)

> **Code:** `code/bounce/` · **Notebooks:** `notebooks/bounce/`
> **Output:** `output/bounce/<P>_<M>/bounce_*.pdf`, `output/bounce/<P>_<M>/bounce_kj_stats.parquet`, `output/bounce/bending_mode_test_meta.parquet`

Analysis of elevation and rotator bounce test data, for Look-Up-Table (LUT)
development. A bounce test moves the telescope to a position and back; a repeatable
difference between the two visits measures hysteresis or gravity-driven flexure rather
than noise.

Two BLOCKs supply the data: **BLOCK-T720** (elevation) and **BLOCK-T724** (rotator
0↔60 deg). The analysis is a *paired* Δ — time-ordered comp−ref pairs **within a night**,
which cancels slowly-varying terms.

## The elevation bounce is a sweep, not a single throw

BLOCK-T720 carries **one reference leg at elevation 70 deg and five comparison legs**, each a
±3 deg window about a measured elevation. Different nights throw to different elevations, so a
night contributes only to the leg(s) it actually exercises:

| comparison leg | throw from 70 deg (deg) | nights |
|---|---|---|
| elev 40 deg | −30 | 20260418, 20260419, 20260513 |
| elev 60 deg | −10 | 20260709 |
| elev 50 deg | −20 | 20260711 |
| elev 30 deg | −40 | 20260713 |
| elev 75 deg | +5 (upward) | 20260713 |

The 75 deg leg uses `alt_range: [73.0, 78.0]` deg so it does not overlap the 67–73 deg
reference window; being a small *upward* throw it serves as a near-null control rather than a
flexure measurement. BLOCK-T724 has **no July nights**, so the rotator bounce carries its
single `Rot=60` leg unchanged, with `camera_hexapod_only: true`.

Because a multi-leg bounce need not exercise every leg on every night, `bounce_lib.bounce_nights`
qualifies a night on the legs it actually has — at least one populated leg, and every populated
leg clearing `night_min_visits`. Requiring *all* configured legs would leave the per-night
breakdown empty for any multi-leg bounce.

The legs live in `analysis_config.yaml` under `overrides:
fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x: bounce:`, at the **param_set** level rather than
under one `mi_name`, so the measured-intrinsic refit table and the Phase-1 batoid-intrinsic
table resolve identical bounce definitions and their results are directly comparable. That
block also sets `pupil_j_range` explicitly, because the Phase-1 table carries no
`nollIndices` column for the script to infer the grid from.

Results, including the throw-vs-FWHM trend and the measured bound on the intrinsic choice, are
written up in [`../../../notes/aos-bounce-test-summary/note.md`](../../../notes/aos-bounce-test-summary/note.md).

## Code

| file | role |
|---|---|
| `bounce_lib.py` | library: pairing, per-(k,j) statistics, significance, plotting (894 lines) |
| `run_bounce.py` | pipeline `bounce` rule — paired Δ for DZ coefficients, OFC v-modes, and physical DOF, with significance/pass heatmaps, vs-ordinal pages, night cross-scatter |

## Notebooks

| file | role |
|---|---|
| `notebooks/bounce/bending_mode_test_lut_trim.ipynb` | commanded hexapod LUT (10 axes) and Trim (all 50 DOF) against `seq_num` for the CWFS exposures of the bending-mode test BLOCKs T377–T380, read from the value-added telemetry DuckDB |

### Bending-mode test BLOCKs

BLOCK-T378 (M1M3 bending modes) and BLOCK-T379 (M2 bending modes) command an individual
mirror bending mode and take Corner Wavefront Sensor (CWFS) pairs, with the mode named in
the `science_program` suffix (`BLOCK-T378_M1M3B12`, `BLOCK-T378_M1M3B20`,
`BLOCK-T379_M2B18`). The notebook shows what the hexapod LUT had loaded and what the Trim
had accumulated while each mode was exercised.

The LUT and the Trim are **different index spaces** that agree only over their first ten
entries. Both order the hexapods M2 first (indices 0–4, `M2_dz/dx/dy/rx/ry`) then camera
(5–9, `Cam_dz/dx/dy/rx/ry`); the Trim continues with M1M3 bending (10–29) and M2 bending
(30–49), which the hexapod LUT has no counterpart for. Labels, units and index groups come
from `lsst.ts.intrinsic.wavefront.ofc_svd` (`LABELS_50DOF`, `DOF_UNITS_50`, `DOF_GROUPS`),
and the LUT axis order is documented at `common/dof_telemetry.py:fetch_hexapod_lut_for_visits`.

One unit trap: the LUT angular axes are **deg**, as the hexapod reports them, while the
Trim rotations are **arcsec**, the OFC convention. Translations are µm in both.

## Why the O/C split matters here

Any intrinsic that is **fixed in the fitting frame cancels in a Δ**. So the
telescope-fixed **O** term drops out, and it is the **rotating camera term C** that
changes a rotator-bounce result. This is precisely why the MIW's O + C decomposition
matters — a bounce analysis run against a frame-fixed intrinsic would show an artifact.

That cancellation is now measured, not just argued. Running the same bounce against the
batoid design intrinsic (`output/fam_processing/danish_1_2/fits.parquet`) and against the MIW
refit (`output/miw/danish_1_2_A_50_34_i_5rot/fits.parquet`) on 20260418 `Elev=40`, where both
tables select an identical visit set, the two Δ sets differ by median −0.0001 µm of wavefront
with `nmad` 0.0003 µm of wavefront, against a Δ signal of 0.0370 µm RMS over the 126 (k, j)
coefficients — Pearson r = 1.000, Spearman rho = 0.983, n = 126. On the BLOCK-T724 rotator
bounce, where C *does* rotate within a pair, the difference is larger but still small:
`nmad` 0.0010–0.0015 µm of wavefront against signals of 0.0238–0.0264 µm RMS, Pearson
r ≈ 0.981. So at present precision the intrinsic choice is not a limiting systematic for
either bounce, which is what makes the batoid-intrinsic July elevation legs trustworthy while
the MIW build stays frozen at `day_obs` 20260513.

## Outputs

`<mi>/plots/bounce_*.pdf`, `<mi>/bounce_kj_stats.parquet` (per-(k,j) Δ, its error and
significance, per night and pooled) and `<mi>/bounce_fwhm_metric.parquet` (`fwhm_before`,
`fwhm_after_50_34`, `fwhm_after_5_5`, all arcsec FWHM, per leg). The DOF and v-mode Δ are
computed in memory and rendered into the PDFs only — they are **not** persisted to a parquet.
With `add_dof_trim` enabled it additionally queries the EFD live for the MTAOS Trim overlay,
so that mode needs RSP/EFD access.

Two output directories hold the two intrinsic choices, so neither shadows the other:
`output/bounce/danish_1_2_A_50_34_i_5rot/` from the MIW refit (April/May only, the lead result
for the 40 deg leg and the rotator bounce) and `output/bounce/danish_1_2_batoid/` from the
Phase-1 batoid-intrinsic fit (the only table covering the July elevation legs). Both are
hand-run with `--out-dir` and `--min-detectors 160`, matching the `bounce` rule.

These PDFs currently share `<mi>/plots/` with three other studies' output; splitting
them per study is outstanding work.

## Statistics note — SEM of a median

A paired Δ reports `median(diffs)`, whose standard error is `1.2533 * sigma_mad /
sqrt(n)`, **not** `sigma_mad / sqrt(n)`. The review backlog flagged
`bounce_lib.py:652` for using the mean form; checked 2026-09-05 and **it now has the
1.2533 factor**, consistent with `stats_per_kj` at line 136. That item is fixed —
do not "re-fix" it.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until bounce
```

Knobs in `analysis_config.yaml` under `bounce`.

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md) — the `bounce` rule in context
- [`../../../smatrix/docs/studies/vmode.md`](../../../smatrix/docs/studies/vmode.md) — where the v-modes and DOF come from
- [`../../../notes/aos-bounce-test-summary/note.md`](../../../notes/aos-bounce-test-summary/note.md) — results summary: elevation sweep 30–75 deg and rotator 0→60 deg
- `../../../notes/claude-memory/apr-2026-50dof-lut.md` — the fixed 50-DOF LUT on sky Apr 24–28 2026, which explains the 20260424/28 anomaly
