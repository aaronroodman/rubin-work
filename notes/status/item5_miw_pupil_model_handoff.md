# Item 5 step A — MIW on the v1000 pupil model

> **Status:** current · **Last updated:** 2026-10-05 · **Kind:** status (handoff)

Step A of [todo item 5](../todos/todo-ideas.md#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint):
rebuild the 50 degree-of-freedom / 34 v-mode (50/34) Measured Intrinsic Wavefront (MIW)
on Josh Meyers' v1000 pupil-model processing and compare it against the existing build on
the legacy pupil model. Scheme held fixed at 50/34; only the pupil model varies.

**Where it stands:** the input tables are built and the configs are in. The MIW build
itself is a batch job and has **not** been submitted — that is a hard MUST-ASK and is the
next action, below. Nothing is blocked otherwise.

## Done and committed

Commit SHA: _see below, committed in this session_

- **`aos/param_sets.yaml`** — new param set `danish_1_3_v1000` on
  `u/jmeyers3/t614_fam_unpaired_v1000`, matching `danish_1_3_test` in every field except
  the collection. `day_obs_min/max` 20260315/20260619 verified against the collection
  rather than copied blindly.
- **`aos/mi_config.yaml`** — `measured_intrinsics: danish_1_3_v1000` with
  `pathA_50_34_i` and `pathA_50_34_i_5rot`, knob-for-knob identical to `danish_1_3_test`
  (checked programmatically, zero differing keys). A **fresh build, not a `build_from`**:
  the wavefronts differ, so the per-rotator-bin grids must be rebuilt.
- **`aos/code/miw/compare_pupil_models.py`** (new, 338 lines) — the image-quality and
  localization half of the comparison, which `compare_miw_versions.py` does not do:
  inferred full width at half maximum (FWHM) in arcsec via ts_wep
  `convertZernikesToPsfWidth`, the difference by field annulus, and each pupil Zernike
  term's share of the difference **power**.
- **`aos/code/miw/compare_build_dof.py`** (new, 251 lines) — the two builds' recovered
  degrees of freedom (DOF) and v-modes, differenced **per visit on the common
  `(day_obs, seq_num)` set**, with each DOF difference scaled against its allowed range
  `r_j`.
- **`aos/docs/studies/miw.md`** — new section "The pupil model: legacy against v1000"
  carrying the provenance table and both runnable commands; the builds table and Code
  table updated.
- **`aos/docs/studies.md`** — `miw` inventory row updated to 5 files / 1475 lines.
- **`notes/todos/todo-ideas.md`** — item 5's Q7 open part answered (see below); item 7
  given the `maskModel` read-back recipe.

## In progress / on disk but not committed

Data products, gitignored by design:

- `aos/output/fam_processing/danish_1_3_v1000/` — the recast tables, **complete**:
  `donuts.parquet` 13,296,016,114 bytes, `visits.parquet` 966 visit rows,
  `fits.parquet` 873 fit rows x 448 columns, plus `provenance.yaml`.
  966 visits recast, 873 passing the quality cuts (36 dropped on
  `n_detectors_with_min_donuts >= 170`, 57 on `median_blur_arcsec <= 1.2 arcsec`),
  2,898,743 donuts fitted, 1 visit flagged `bad_fit`
  (`day_obs` 20260327 `seq_num` 228). Log:
  `aos/logs/blitz_mktable_v1000_20261005_000032.log`.

The recast took about 33 min wall clock on an s3df interactive node.

## Next concrete action

**Submit the MIW build — batch, so Aaron starts it.** The dry run is clean: 10 jobs, 9
`build_intrinsic` grids plus 1 `intrinsic_split`, no `combine_*` (the hand-built tables
are terminal inputs, as the Snakefile documents).

```bash
cd ~/notebooks/rubin-work/aos && ./run_snake.sh --mode batch -- \
  output/miw/danish_1_3_v1000_A_50_34_i_5rot/intrinsic_split_maps.parquet
```

Monitor:

```bash
tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/batch_*.out | head -1)"
```

Then, once it finishes, the three comparisons — all three commands are in
`aos/docs/studies/miw.md`, and `compare_miw_versions.py` is the third:

```bash
cd ~/notebooks/rubin-work/aos
python code/miw/compare_miw_versions.py \
  --miw-a output/miw/danish_1_3_test_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --miw-b output/miw/danish_1_3_v1000_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --label-a "Danish 1.3 legacy pupil" \
  --label-b "Danish 1.3 v1000 pupil" \
  --out-dir output/miw/danish_1_3_legacy_vs_v1000 \
  --out-name miw_legacy_vs_v1000_OCS
```

Report, per the item's scope: the common visit count, the term-by-term and field-map
comparison, both MIW as inferred FWHM in arcsec, whether the difference concentrates at
the field edge and the pupil edge, and each build's recovered DOF and v-modes.

## Tried and rejected, and why

- **Using `_legacy` as the baseline instead of the existing build.** The item's Q7 left
  open whether the two output RUNs differ *only* in the pupil model. They do **not**: the
  unsuffixed `t614_fam_unpaired` the existing MIW used is an older code version too —
  `danish` `5037d9f3` against `ca41ae8c`, `ts_wep` `9651cd23` against `639a89d9`, and the
  task is `donutBlitzFamTask` with no `maskModel` config field at all, against
  `donutBlitzFam`. `u/jmeyers3/t614_fam_unpaired_legacy` *is* the airtight pair — same
  `danish` commit, same task, config differing in **exactly one line**, `maskModel`. I
  proposed building it as a third arm; **Aaron declined 2026-10-04**: "I know that the
  existing danish 1.3 used an older code version, but there were only small changes since
  then. So v1000 against the existing build is fine." So the pupil model is the leading
  term and the version change is a stated caveat. Do not re-propose the third build
  without a result that actually needs it.
- **Reading the task config through `butler.get`.** Fails with
  `ModuleNotFoundError: No module named 'lsst.ts.wep.blitz'` in `w_2026_39` — the pex
  formatter imports the config's own module. Read the `.py` straight off disk at
  `<run>/donutBlitzFam*_config/*.py` instead.
- **`butler.collections.query_info(..., include_summary=True)` to list RUN children.**
  Returned `RUN 0` for chains that plainly have one. The CLI
  `butler query-collections /repo/main <chain> --chains=flatten --collection-type=RUN`
  works and is what the original Q7 measurement used.
- **Letting the MIW build pull the tables through `rule all` / `combine_*`.** It cannot:
  `danish_1_3_v1000` declares no chunks in `snake_config.yaml`, exactly like
  `danish_1_3_test`, so `combine_parquets.py` would be invoked with an empty input list.
  The blitz recast writes `{donuts,visits,fits}.parquet` directly and Snakemake then
  treats them as terminal inputs. Run `run_blitz_mktable.py` first; that is not optional
  and not a `build_from`.
- **Running the recast in batch.** It needs the Consolidated Database (ConsDB), so it is
  interactive-only — its own docstring says so.

## Non-obvious constraints found

- **The recast cannot run in batch** (ConsDB), but it is also the expensive step here
  (~33 min, 13 GB), not `mktable`. `mktable` never runs for this param_set at all.
- **All three T614 FAM collections cover identical visits** — 966
  `donutBlitzFamResults` over the same 15 nights — so the common-visit set is not limited
  by coverage, and `day_obs_min/max` copy across unchanged.
- **The two fits tables agree on 873 of 873 visits**, so the comparison runs on the full
  common set. The DZ shifts are small and led by the spherical term: median
  `z1toz6_z11_c1` moves +0.0028 µm of wavefront against +0.0001 µm on `z4` — the
  signature a pupil-rim change should have, and a first indication the build will show a
  real but small difference.
- **`blitz` still has no `donut_blur` column** (it names it `group_fwhm`), so the blur DZ
  fit is skipped on this param_set as on `danish_1_3_test`. Pre-existing, unrelated to
  the pupil model, and already a standing item.
- **The convex-hull edge defect is present in both builds** and is carried deliberately.
  `compare_pupil_models.py` reports the outermost ring as its own annulus so it cannot
  contaminate an inner one — validated on the Danish 1.2-vs-1.3 pair, where the inner
  annuli rise monotonically 0.0126 to 0.0202 µm of wavefront and the hull ring jumps to
  0.1081 µm.
- **Baseline for reading the spherical share:** on the Danish 1.2-vs-1.3 pair (a
  *retrieval* change, not a pupil change) Noll 11 and 22 together carry 0.0609 of the
  difference power (dimensionless). A pupil-rim change should carry a larger share than
  that; it is the number the v1000 result gets compared against.

## Step B is not started

Step B — Range-Bounded Recovery (RBR) on the MIW — waits on step A's comparison, per the
item. Q8 (which build or builds RBR applies to) is to be answered from the step A result.
`aos/code/miw/check_dof_ranges.py` already measures how far the existing build's states
fall outside `r_j` and changes nothing.
