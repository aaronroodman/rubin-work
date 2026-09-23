# Study: `bounce` — elevation and rotator bounce tests

> **Status:** current · **Last updated:** 2026-09-22 · **Kind:** reference (study)

> **Code:** `code/bounce/` · **Notebooks:** `notebooks/bounce/`
> **Output:** `output/bounce/<P>_<M>/bounce_*.pdf`, `output/bounce/<P>_<M>/bounce_kj_stats.parquet`, `output/bounce/<P>_<M>/bounce_dof_stats.parquet`, `output/bounce/<P>_<M>/bounce_fwhm_metric.parquet`, `output/bounce/bending_mode_test_meta.parquet`

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

| comparison leg | throw from 70 deg (deg) | nights with pairs | n pairs |
|---|---|---|---|
| elev 40 deg | −30 | 20260418, 20260419, 20260513 | 22 |
| elev 60 deg | −10 | 20260709 | 4 |
| elev 50 deg | −20 | 20260711 | 6 |
| elev 30 deg | −40 | 20260713 | 5 |
| elev 75 deg | +5 (upward) | 20260713 | 6 |

Only the 40 deg leg has more than one night, so it is the only leg carrying a night-to-night
repeatability measurement; the night-pair products are drawn only where 2 or more nights
qualify.

The 75 deg leg uses `alt_range: [73.0, 78.0]` deg so it does not overlap the 67–73 deg
reference window; being a small *upward* throw it serves as a near-null control rather than a
flexure measurement. BLOCK-T724 has **no July nights**, so the rotator bounce carries its
single `Rot=60` leg unchanged, with `camera_hexapod_only: true`, over nights 20260420 and
20260513 with 31 pairs.

**The per-visit quality cut has two active parts**, and the blur one is what removes visits
here. `quality_visit_mask` applies a donut-count floor (`--min-detectors 160`) *and*
`median_blur_arcsec <= 1.2` arcsec, the latter at its library default
`DEFAULT_MAX_MEDIAN_BLUR_ARCSEC`. Every BLOCK-T720 visit on all six nights clears the CCD
floor at 176–180 CCDs, so relaxing `--min-detectors` changes nothing; three visits fail the
blur cut, both of 20260711's elevation 30 deg visits at 1.525 and 1.934 arcsec and one of
20260713's at 1.799 arcsec. That is why the 30 deg leg is a single-night 5-pair leg rather
than a two-night 8-pair one, and it is a seeing limitation, not a processing one.

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
| `bounce_lib.py` | library: pairing, per-(k,j) statistics, significance, leg-night coverage, plotting |
| `run_bounce.py` | pipeline `bounce` rule — paired Δ for DZ coefficients, OFC v-modes, and physical DOF, with vs-ordinal pages, night cross-scatter and the per-DOF-vs-B-set panels |

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

## The vs-ordinal marker scheme

The three vs-ordinal PDFs encode three visit properties in one marker, defined in
`lsst.ts.intrinsic.wavefront.intrinsics_lib` (the external package, not `aos/code/`) so that
every study drawing these pages shares one scheme:

| property | encoding |
|---|---|
| elevation | marker **colour**, one per grid centre at 30, 40, 50, 60, 70, 75 deg |
| camera rotator angle | marker **shape** — an arrow whose direction gives the rotator angle |
| filter band | a small **dot overlaid at the centre of the arrow shaft**, in the band's colour |

A visit is assigned to the nearest grid centre within a half-width, `ELEV_HALFWIDTH_DEG = 2.0`
deg in elevation, tightened from 5 deg so that the 70 and 75 deg legs separate. The half-width
is a parameter (`elev_halfwidth_deg` / `ab_elev_halfwidth_deg`, and the rotator equivalent)
rather than a constant.

The band is a dot rather than the marker edge because the edge colour competes with the
elevation fill: at plotted marker sizes an edge-encoded band makes the elevation unreadable.
Band colours come live from `lsst.utils.plotting.get_band_dicts()['colors']`, the canonical
Rubin palette (DM-51122, revised DM-51690), with the current hex values kept as a fallback for
environments where `lsst.utils.plotting` is unavailable.

## Why the O/C split matters here

Any intrinsic that is **fixed in the fitting frame cancels in a Δ**. So the
telescope-fixed **O** term drops out, and it is the **rotating camera term C** that
changes a rotator-bounce result. This is precisely why the MIW's O + C decomposition
matters — a bounce analysis run against a frame-fixed intrinsic would show an artifact.

That cancellation is now measured on **every leg**, not just argued. The MIW-referenced fit was
run over the July nights as well (`output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet`),
which needed no MIW rebuild — the build stays frozen at `day_obs` 20260513 and the fit merely
references it. Comparing its Δ against the batoid design intrinsic
(`output/fam_processing/danish_1_2/fits.parquet`) per (k, j), with n = 126 DZ coefficients on
each leg, the `nmad` of the difference over the leg's own RMS(Δ) is 0.9% to 1.6% on the four
single-night elevation legs (Pearson r ≥ 0.999), 3.1% on the 40 deg leg — where the two tables
select different visit sets, 22 pairs against 10, so that row mixes intrinsic choice with a
genuine sample change — and 5.2% on the BLOCK-T724 rotator bounce, where C *does* rotate
within a pair and the difference is expected to be largest. So at present precision the
intrinsic choice is not a limiting systematic for either bounce. The full table is in the
results note.

## Outputs

Three parquet tables and eight PDFs per run.

| product | content |
|---|---|
| `bounce_kj_stats.parquet` | per-(k, j) Δ in µm of wavefront, its error and significance, per night and pooled |
| `bounce_dof_stats.parquet` | per-DOF and per-v-mode Δ, per night and pooled, one row per (bounce, leg, night, quantity) with a `kind` of `dof`, `vmode`, `dof5` or `vmode5` and a `unit` column — µm for translations and bending-mode amplitudes, arcsec for hexapod rotations, dimensionless for v-modes |
| `bounce_fwhm_metric.parquet` | `fwhm_before`, `fwhm_after_50_34`, `fwhm_after_5_5`, all arcsec FWHM, per leg |

`bounce_summary.pdf` opens with a leg-night coverage table — which nights back each leg and how
many night-pair scatter pages follow — then the night-vs-night Δ cross-scatter, wide and
zoomed, for every leg with 2 or more qualifying nights. `bounce_dof_night_scatter.pdf` carries
the same coverage table and the per-DOF night-pair scatter. `bounce_dof_night_values.pdf` shows
the paired Δ DOF as small per-DOF panels against the B-set position — elevation in deg for
BLOCK-T720, camera rotator angle in deg for BLOCK-T724 — with one point per (night, leg); for a
`camera_hexapod_only` bounce only the 5-DOF / 5-v-mode camera-hexapod scheme is drawn, since
that is the scheme the result is used in. `bounce_dz_vs_ordinal.pdf` opens with the marker
legend and an A/B bounce-position table per bounce.

With `add_dof_trim` enabled the run additionally queries the EFD live for the MTAOS Trim
overlay, so that mode needs RSP/EFD access.

Three output directories, so none shadows another:
`output/bounce/danish_1_2_A_50_34_i_5rot_july/` from the MIW-referenced fit over all six nights
(the lead result, every number in the note), `output/bounce/danish_1_2_A_50_34_i_5rot/` from the
same MIW build over April/May only (superseded), and `output/bounce/danish_1_2_batoid/` from the
Phase-1 batoid-intrinsic fit (the intrinsic-choice comparison). All are hand-run with
`--out-dir`, `--fits` and `--min-detectors 160`, matching the `bounce` rule.

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
