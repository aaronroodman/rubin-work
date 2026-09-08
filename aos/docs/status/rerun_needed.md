# Outputs that need regenerating

> **Status:** current · **Last updated:** 2026-09-07 · **Kind:** working state (rerun list)

Products on disk that predate a code change and no longer match what the current code
would produce. Kept here so a stale plot is not mistaken for a current result.

The current `param_set` is `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`. Superseded
`param_set` outputs are in `output/archive/` and are not worth regenerating.

## The `mktable` re-run trigger, and how to bypass it

A plain `./run_snake.sh -n` plans **77 jobs**, five of them `mktable` — the expensive
Butler step — with:

```
reason: Params have changed since last execution:
        before: '--no-thermal' now: ''
```

The five are exactly the **2025 chunks** (`20250415_20250531`, `20250601_20250930`,
`20251001_20251115`, `20251116_20251130`, `20251201_20251231`). They were built with
`mktable: no_thermal: true` to get donuts quickly; `snake_config.yaml` now says `false`,
so Snakemake correctly sees the recorded params differ.

**But the data is not stale.** The thermal columns in those chunks *are* populated —
checked `20250415_20250531`: 623 of 637 rows have `cam_air_temp`, `m2_air_temp`,
`m1m3_air_temp`, `outside_temp` and the gradients. `donuts.parquet` is dated Aug 20 and
`visits.parquet` Aug 24, four days later: the thermal columns were attached after the
chunk was built, which `code/fam_processing/run_attach_telemetry.py` now does.

So the trigger is **stale provenance metadata, not stale data**. Bypass it with:

```bash
cd ~/notebooks/rubin-work/aos
snakemake -n --rerun-triggers mtime -j 4 --resources mem_mb=14000    # 67 jobs, no mktable
```

That drops the 5 `mktable` and 5 dependent `fit` jobs, leaving 67 — everything else
unchanged. `--rerun-triggers mtime` tells Snakemake to decide staleness from file times
only, ignoring the params and code hashes.

The alternative, if you would rather keep the default triggers, is to wipe just those
outputs' metadata:
`snakemake --cleanup-metadata output/<ps>/chunks/<chunk>/visits.parquet` per chunk.

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

### `psf` and `closed_loop` — single MI build, study split, new output paths
**Why:** three changes on 2026-09-07. The `psf` study previously mixed **two**
measured-intrinsic builds — `--split-mi` (`pathA_50_34_i_5rot`) for the MIW split maps and
`--fam-mi` (`pathA_50_34_i`) for the per-visit FAM fits. `pathA_50_34_i` is a superseded
first-pass build, so every existing PDF was made partly from stale input. Both studies now
take a single `--mi`, defaulting to `pathA_50_34_i_5rot`, which also carries the larger
sample (1126 versus 960 visits in `fits.parquet`).

The closed-loop cases also moved into their own
[`closed_loop`](../studies/closed_loop.md) study, and output moved from `<ps>/psf/` to
`<ps>/<mi>/{psf,closed_loop}/`.

**Affected files:** the 6 PDFs now in `output/<ps>/<mi>/psf/` and the 8 in
`output/<ps>/<mi>/closed_loop/`. The latter still carry their old `psf_fp_maps_loop*`
names; a rerun writes `closed_loop_*` instead.

```bash
python code/psf/run_psf_fp_maps.py --case all
python code/psf/run_psf_fp_maps.py --case mimic
python code/psf/run_psf_fp_maps.py --case validate
python code/closed_loop/run_closed_loop.py --case loop
```

### `static_optics` camera-gravity — bending basis may have changed
**Why:** `camera_gravity.py:95` picks a bend directory in the order `bend_zemax` →
`bend_full` → `bend`, first match wins. Until 2026-09-07 only `bend` existed on S3DF;
`bend_full` was then regenerated there, so the same code now selects a **different basis**
(156 M1M3 + 72 M2 modes instead of 20 per mirror) with no change to the code or arguments.

The script's docstring states that gravity does not use the bending-mode basis, so output
may be identical — but that is unverified. Any camera-gravity output produced after
2026-09-07 should be checked against the earlier PDFs in `output/camera_gravity/`, or the
basis pinned explicitly via `bend_dir`.

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
