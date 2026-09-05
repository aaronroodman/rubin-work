# Study: `bounce` — elevation and rotator bounce tests

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** When the telescope is moved to a position and back, does the optical state
return? A repeatable Δ between two elevations (or two rotator angles) is hysteresis or
gravity-driven flexure, not noise.

Two BLOCKs supply the data: **BLOCK-T720** (elevation 40↔70 deg) and **BLOCK-T724**
(rotator 0↔60 deg). The analysis is a *paired* Δ — time-ordered comp−ref pairs **within a
night**, which cancels slowly-varying terms.

## Code

| file | role |
|---|---|
| `bounce_lib.py` | library: pairing, per-(k,j) statistics, significance, plotting (894 lines) |
| `run_bounce.py` | pipeline `bounce` rule — paired Δ for DZ coefficients, OFC v-modes, and physical DOF, with significance/pass heatmaps, vs-ordinal pages, night cross-scatter |

## Why the O/C split matters here

Any intrinsic that is **fixed in the fitting frame cancels in a Δ**. So the
telescope-fixed **O** term drops out, and it is the **rotating camera term C** that
changes a rotator-bounce result. This is precisely why the MIW's O + C decomposition
matters — a bounce analysis run against a frame-fixed intrinsic would show an artifact.

## Outputs

`<mi>/plots/bounce_*.pdf` and `<mi>/bounce_kj_stats.parquet`. With `add_dof_trim`
enabled it additionally queries the EFD live for the MTAOS Trim overlay, so that mode
needs RSP/EFD access.

**Phase 7 will move the PDFs** out of the flat `<mi>/plots/` dump into `<mi>/bounce/`.

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
