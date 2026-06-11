# AOS FAM Pipeline — Reorganization Design

**Status:** Draft for discussion · **Author:** Aaron Roodman (with Claude) · **Date:** 2026-06-10

Goal: make the full FAM analysis suite reproducible and re-runnable with one
command on new FAM processing, retire the duplicate orchestration, and make the
**measured intrinsic** (telescope O + camera C) a first-class, switchable input
to all downstream analysis.

---

## 1. Current state

### Orchestration (two systems today)
- **`run_pipeline.py`** — a make-like DAG engine driven by `runs.yaml` (run
  instances) + `param_sets.yaml` (dataset defs). Steps
  `mktable → fit → {plots, movie}` (and `_ccs` variants), with `STEP_DEPS`,
  per-step status, file-locking, `logs/`, and git-sync. Solid; keep it.
- **`build_measured_intrinsic_batch.ipynb`** — a *separate* papermill driver
  with its own `PARAM_SETS`, scanning elevation/rotator/band/n_keep. Parallel
  universe to `run_pipeline`.
- **Manual notebooks** — `intrinsic_split` and every `study_*` are run by hand
  with hard-coded input paths.

So the DAG covers ~the first third of the real pipeline; everything downstream
is hand-run and not reproducible end-to-end.

### Module layer (healthy)
`intrinsics_lib`, `dz_fitting`, `dz_plotting`, `measured_intrinsic`, `ofc_svd`,
`intrinsic_split`, `ccd_height`, `aos_trim`.

### Data flow (today)
```
mktable  (intrinsics_lib): Butler -> donut parquet, bakes batoid zk_intrinsic_{OCS,CCS}
  -> fit (dz_fitting): DZ of (zk_data - zk_intrinsic_{coord})  -> *_fits.parquet
       -> plots / movie (validation)
       -> build_measured_intrinsic (papermill scan): measured-intrinsic grids
            -> intrinsic_split: O (OCS) + C (CCS) per Zernike  [all Noll 4-19,22-26]
       -> study_bounce / study_doublezernike / study_aberrationpairs /
          intrinsics_thermal_correlations / study_wfs_mimic
            (all read *_fits.parquet -> inherit the TABULATED batoid intrinsic)
```

### Where the intrinsic enters (crux)
- `mktable` welds the **batoid** `zk_intrinsic_{coord}` into the donut parquet.
- `fit` subtracts that column: `residual = zk_data - zk_intrinsic_{coord}`
  (`dz_fitting._fit_one_image`, `include_intrinsic=True`).
- There is **no seam** to choose a different intrinsic; downstream is locked to
  batoid.

---

## 2. Pain points

1. **Two orchestrators** (run_pipeline DAG vs papermill batch) → double
   maintenance, drift, two config formats.
2. **DAG stops at fit/plots/movie**; measured_intrinsic, intrinsic_split, and
   all study_* are outside it → manual, error-prone, not reproducible.
3. **Intrinsic hard-wired** at mktable; no switch for measured / O+C.
4. **study_\* hard-code input parquet paths** → reprocessing means editing N
   notebooks by hand.
5. **Duplicated notebooks** (papermill copies) drift from the source notebook.

---

## 3. Design A — switchable intrinsic source

### 3.1 Principle (and the bounce subtlety)
Re-fit rather than patch: DZ fitting is linear, so the cleanest way to change
the intrinsic is to recompute the per-donut `zk_intrinsic` and re-run `fit`.

For **difference** analyses (study_bounce ΔDZ between settings), any intrinsic
that is *fixed in the fitting frame cancels in the difference*. So:
- A single **measured (OCS) map** behaves like batoid for bounce Δ (it cancels);
  it mainly helps the non-difference analyses (thermal correlations, doublez,
  wfs_mimic) and the absolute-state plots.
- The **O + C split** is what changes the bounce: **C is camera-fixed**, so
  subtracting C rotated to each visit's rotator removes the camera term that
  otherwise contaminates the rotator bounce (T724); **O** matters for the
  elevation bounce only if elevation-dependent.

→ Support **both, switchable**; default **measured_OC** for study_bounce.

### 3.2 `code/intrinsic_provider.py` (new)
```python
def per_donut_intrinsic(donut_df, source, *, noll_list, maps=None, coord_sys='OCS'):
    """Per-donut intrinsic zk (n_donut, n_zk) to subtract before DZ fitting.

    source:
      'batoid'      -> use the existing zk_intrinsic_{coord} column (default/now)
      'measured'    -> sample a combined measured-intrinsic grid at the donut's
                       OCS field position (build_measured_intrinsic grid parquet)
      'measured_OC' -> O_Zj(thx_OCS, thy_OCS) + C_Zj(thx_CCS, thy_CCS) per Zernike,
                       from the intrinsic_split all-Zernike O/C parquet
    `maps` points at the grid / O-C parquet for the measured sources.
    Uses thx_OCS/thy_OCS and thx_CCS/thy_CCS already in the donut parquet.
    """
```
Notes:
- `measured_OC` needs no rotator angle explicitly — evaluating C at the donut's
  **CCS** field position already places the camera-fixed pattern correctly,
  because the donut's CCS coords encode the rotation.
- Interpolators reuse `intrinsic_split.polar_field_to_points` / a 2-D
  scattered interpolation of the O/C grids.

### 3.3 Re-fit seam
`run_dz_fit.py` gains `--intrinsic-source {batoid|measured|measured_OC}` and
`--intrinsic-maps <parquet>`. `dz_fitting.fit_dz_modes_*` calls
`per_donut_intrinsic(...)` to obtain `zk_intrinsic` instead of reading the
column when source != 'batoid'. Output parquet name carries the source tag,
e.g. `..._fits__measured_OC.parquet`, so batoid and measured fits coexist.

### 3.4 The measured-intrinsic is a shared artifact
The measured intrinsic (and O/C) is built from the **rotator-scan** runs, then
**applied** to other runs (e.g. the bounce data). That's a cross-run
dependency. Model it explicitly: a small `intrinsics.yaml` (or a section in
`param_sets.yaml`) names each measured-intrinsic build (which runs feed it,
n_keep, etc.) and yields an artifact id; downstream runs reference
`intrinsic_source: measured_OC` + `intrinsic_id: <id>`.

---

## 4. Design B — unified orchestration

### 4.1 Generic `notebook` step type
`run_pipeline` already shells commands per step. Add a step kind that
papermill-executes a parameterized notebook with params drawn from the run:
```
build_command(... ) -> ['python','-m','papermill', <nb>, <out_nb>,
                        '-p','fits_parquet', <path>, '-p', ...]
```
This single mechanism replaces both `build_measured_intrinsic_batch` (papermill)
and the hand-run study_* notebooks. The notebooks keep a papermill-tagged
`parameters` cell (build_measured_intrinsic already has one).

### 4.2 Extended DAG
```
STEP_ORDER += ['measured_intrinsic', 'intrinsic_split', 'apply_intrinsic',
               'fit_measured', 'study_bounce', 'study_doublez',
               'study_aberrationpairs', 'thermal_corr', 'wfs_mimic', 'report']
STEP_DEPS:
  measured_intrinsic: [fit]            # (from the rotator-scan run set)
  intrinsic_split:    [measured_intrinsic]
  fit_measured:       [mktable, intrinsic_split]   # re-fit w/ measured_OC
  study_bounce:       [fit_measured]               # default measured_OC
  study_doublez:      [fit_measured]
  study_aberrationpairs:[fit_measured]
  thermal_corr:       [fit_measured]
  wfs_mimic:          [fit_measured]
  report:             [plots, study_bounce, ...]   # assemble PDFs
```
Retire `build_measured_intrinsic_batch.ipynb`; its `PARAM_SETS` collapse into
`runs.yaml` (one run per scan point) + `param_sets.yaml`.

### 4.3 One config, one command
`python code/run_pipeline.py run` walks the whole DAG; status/logging/git-sync
already exist. Every validation PDF becomes a tracked per-run step output.

---

## 5. Design C — parameterize `study_*`

Replace hard-coded input paths in each `study_*` with a papermill `parameters`
cell: `fits_parquet_paths`, `intrinsic_source`, `intrinsic_maps`, `output_dir`.
Then they slot into the DAG and rerun consistently. (study_bounce already has a
parameters cell — extend it; the others need one added.)

## 6. Design D — validation report

A `report` step assembles each run's PDFs (fit-params, trio, measured-intrinsic
validation, intrinsic_split, study_* summaries) into an index / contact-sheet
(one HTML or a combined PDF), so a reprocessing yields one browsable bundle.

---

## 7. Phased migration

- **P0 (this doc).**
- **P1 — intrinsic provider + re-fit (delivers the immediate study_bounce goal).**
  `intrinsic_provider.py`; `run_dz_fit.py --intrinsic-source`; validate
  `measured_OC` vs `batoid` on one bounce chunk (expect the difference to be the
  rotating camera term); point study_bounce at the measured_OC fits.
- **P2 — generic `notebook` step in run_pipeline**; fold
  build_measured_intrinsic + intrinsic_split into the DAG; retire the papermill
  batch.
- **P3 — fold study_\* into the DAG** (parameterize inputs; add `fit_measured`,
  cross-run `intrinsic_id`).
- **P4 — `report` step.**

---

## 7b. Snakemake evaluation prototype (added 2026-06-10)

Files: `Snakefile` + `snake_runs.yaml` (reuses the existing `param_sets.yaml`).
Covers the **script** steps only (`mktable → fit → {plots, movie}`, OCS) for two
runs — enough to judge ergonomics. **`run_pipeline.py` stays the production
orchestrator**; this changes nothing in it.

Try it (install once: `pip install snakemake`):
```
cd aos
snakemake -n                 # dry-run: shows exactly what is stale / would run
snakemake -j4                # build, up to 4 jobs in parallel  (mktable needs RSP/Butler)
snakemake --dag | dot -Tpng > dag.png
```

What to look at:
- **Incremental rebuild** — with the existing donut/fits/trio outputs present,
  `-n` reports only the two `movie` jobs pending; it will *not* re-run the
  expensive Butler `mktable` or the `fit`. Touch a donut parquet and watch the
  downstream go stale.
- **Parallelism** — `-j4` runs independent runs/steps concurrently (vs
  run_pipeline's sequential loop).
- **Wildcards** — the scan/runs live in `snake_runs.yaml`; file paths
  (`output/{stem}…`) drive the DAG. `stem` = `<collection_phrase>_<dmin>_<dmax>`,
  matching the existing filenames exactly.

Caveats surfaced by the prototype (the real ergonomic costs):
- **`wildcard_constraints` are required** — `stem` must be constrained to end in
  `_<8 digits>_<8 digits>`, else `output/{stem}.parquet` is ambiguous with
  `output/{stem}_fits.parquet`. This is the wildcard pitfall, made concrete.
- **Multi-output steps need a representative output or a sentinel** — `plots`
  declares the `trio_comparison_all.pdf`; `movie` uses a `touch`ed
  `.movie.done` marker (it emits many JPEGs + an optional mp4). Production use
  would enumerate outputs or keep sentinels.
- **No git-sync / status-board** — run_pipeline's per-step commit + human-readable
  `runs.yaml` board would become `onsuccess:`/`onerror:` hooks + `--summary`.
- mktable needs Butler, so a full local build isn't possible; `-n` and the
  downstream (fit/plots/movie) are what you can exercise off-RSP.

Decision after trying it: keep extending `run_pipeline`, or commit to migrating
the whole suite to Snakemake (then the cross-run intrinsic dep + scan become
`lambda`/`expand`, per §4).

## 8. Open questions

1. **Cross-run intrinsic dependency** — `intrinsics.yaml` vs a `param_sets`
   section vs a convention. Which feels most natural in your `runs.yaml` model?
2. **Re-fit vs sidecar** — recompute `zk_intrinsic` inside `fit` (clean, one
   artifact) vs an `apply_intrinsic` step that writes a new donut parquet
   (re-uses fit unchanged, more disk). Leaning: compute inside `fit`.
3. **Keep batoid fits too?** Default to producing both batoid and measured_OC
   fits per run, or only the selected source?
4. **Scan representation** — one `runs.yaml` entry per (elev,rot,band,n_keep)
   scan point (verbose but uniform) vs a compact `scan:` expansion in a run.
5. **study_* that are inherently difference-based** (bounce) vs absolute
   (thermal, doublez): confirm which default to `measured_OC` vs `measured`.

---

## 9. Phase-1 implementation — chunk/combine pipeline (committed 2026-06-11)

Committed to Snakemake. Files: `aos/Snakefile`, `aos/snake_config.yaml`,
`aos/code/combine_parquets.py`. (`snake_runs.yaml` retired.)

**New output layout** (param_set grouped; no shared top-level `output/`):
```
output/<param_set>/
  chunks/<dmin>_<dmax>/  donuts.parquet  fits.parquet  visits.parquet   # per chunk
  donuts.parquet  fits.parquet  visits.parquet                          # COMBINED
  plots/                                                                # on combined
  measured_intrinsic/  intrinsic_split/  study_*/                       # downstream (Phase 2)
```

**Chunk/combine model** (per Aaron's design):
- Chunks belong to a param_set (`snake_config.yaml: param_sets.<ps>.chunks: [[dmin,dmax],…]`);
  Butler defs stay in `param_sets.yaml`. No global chunk1/chunk2 list.
- `mktable` + `fit` run **per chunk** → short-named `donuts/fits/visits.parquet`
  in `chunks/<dmin>_<dmax>/` (mktable writes its derived name, the rule renames).
- `combine_{donuts,fits,visits}` concatenate the chunks → one param_set-level
  table each. **All downstream uses the combined tables.**
- Add data → add/edit a chunk → Snakemake re-runs `combine` + everything
  downstream automatically (the incremental DAG, the whole point).

**The Snakefile owns all paths** and passes them explicitly to the scripts
(`--output`/`--output-dir`), so the convention lives in one place.

Phase-1 scope: `mktable → fit → combine → plots`. Validated: path logic +
wildcard disambiguation (combined `output/<ps>/X.parquet` vs chunk
`output/<ps>/chunks/<d>_<d>/X.parquet`; `ps` constrained to no-slash).
Not yet run end-to-end (snakemake/pyarrow unavailable locally — RSP test pending).

**Phase 2 (next):** `movie` (per chunk); papermill notebook rules for
`build_measured_intrinsic`, `intrinsic_split`, `study_*`; the parameter scan
(nkeep/alt/rot) and cross-run intrinsic dependency; `resources: mem_mb` for
parallel-safe memory; CCS fits.
