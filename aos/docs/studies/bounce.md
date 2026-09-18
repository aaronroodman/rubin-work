# Study: `bounce` — elevation and rotator bounce tests

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

Analysis of elevation and rotator bounce test data, for Look-Up-Table (LUT)
development. A bounce test moves the telescope to a position and back; a repeatable
difference between the two visits measures hysteresis or gravity-driven flexure rather
than noise.

Two BLOCKs supply the data: **BLOCK-T720** (elevation 40↔70 deg) and **BLOCK-T724**
(rotator 0↔60 deg). The analysis is a *paired* Δ — time-ordered comp−ref pairs **within a
night**, which cancels slowly-varying terms.

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
and the LUT axis order is documented at `aos_trim.fetch_hexapod_lut_for_visits`.

One unit trap: the LUT angular axes are **deg**, as the hexapod reports them, while the
Trim rotations are **arcsec**, the OFC convention. Translations are µm in both.

## Why the O/C split matters here

Any intrinsic that is **fixed in the fitting frame cancels in a Δ**. So the
telescope-fixed **O** term drops out, and it is the **rotating camera term C** that
changes a rotator-bounce result. This is precisely why the MIW's O + C decomposition
matters — a bounce analysis run against a frame-fixed intrinsic would show an artifact.

## Outputs

`<mi>/plots/bounce_*.pdf` and `<mi>/bounce_kj_stats.parquet`. With `add_dof_trim`
enabled it additionally queries the EFD live for the MTAOS Trim overlay, so that mode
needs RSP/EFD access.

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
- [`smatrix_vmode.md`](smatrix_vmode.md) — where the v-modes and DOF come from
- `../../../notes/claude-memory/apr-2026-50dof-lut.md` — the fixed 50-DOF LUT on sky Apr 24–28 2026, which explains the 20260424/28 anomaly
