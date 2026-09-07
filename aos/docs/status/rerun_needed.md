# Outputs that need regenerating

> **Status:** current · **Last updated:** 2026-09-07 · **Kind:** working state (rerun list)

Products on disk that predate a code change and no longer match what the current code
would produce. Kept here so a stale plot is not mistaken for a current result.

The current `param_set` is `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`. Superseded
`param_set` outputs are in `output/archive/` and are not worth regenerating.

## Pending

### Everything, once the 2025 FAM data is folded in
Aaron: most of the analyses below were last run *before* the 2025 FAM data was added to
the current `param_set` as five extra date chunks, so they want a rerun on the fuller
sample regardless of the code changes. That makes a single pass over the whole DAG the
efficient move rather than targeted reruns.

```bash
cd ~/notebooks/rubin-work/aos && ./run_snake.sh -n     # confirm scope first
```

Batch form is a MUST-ASK — see the root `CLAUDE.md` "Batch jobs" for the submit plus
`tail -f` pair.

### `dz_explained` and `vmode_correlations` — quality cut changed
**Why:** until 2026-09-07 these two used a `quality_cut` that did **not** drop visits
flagged `bad_fit`, while `dz_correlations` and `thermal_correlations` did. They therefore
included ~24 visits whose DZ fit had failed, usually for too few donuts to constrain the
k=1..6 focal-plane terms. Both now call the shared `fam_quality_selection`.

**Measured effect** on `pathA_50_34_i_5rot/fits.parquet` at the configured
`max_coeff_um = 2.0` µm: the selected sample drops from 1125 to 1101 of 1126 visits.
`dz_correlations` and `thermal_correlations` are unaffected — same 1101 visits, identical
index — so their existing output is still valid.

**Affected files:**
`output/<ps>/<mi>/correlations/dz_explained.{pdf,parquet}`,
`vmode_correlations_{50_34,22_12}.pdf`, `vmode_correlations_summary_{50_34,22_12}.parquet`

```bash
./run_snake.sh --until dz_explained
./run_snake.sh --until vmode_correlations
```

## Not affected, for the record

- **`static_optics`** — the four scripts now share `miw_io.load_miw`, but it reproduces
  each previous implementation exactly. The three back-projection scripts get a
  bit-identical grid (verified: same rows, same `pts`, same `zk` at stride 1 and 8), and
  `camera_gravity_maps` keeps its laxer row cut via `require=(5, 6, 7, 8)`, giving the
  same 3969 rows as before. No rerun needed.
- **`dz_correlations`, `thermal_correlations`** — see above.
- **`smatrix_vmode`** — `vmode_dof_matrix_{22_12,50_34}.pdf` were regenerated on
  2026-09-06 into `output/smatrix_vmode/` when the output moved out of `<ps>`.

## Notes

Two products are stale for a data reason rather than a code reason, and a rerun of the
owning step is the fix:

- `output/<ps>/coadd_50_34/` — `block_grids.npz` (130 umode rows) and
  `coadd_metrics_rebin3.parquet` (221 rows) are from **different runs**, which makes
  `analyze_miw_field_order.py` fail with an `IndexError`. See
  [`code_review_backlog.md`](code_review_backlog.md).
