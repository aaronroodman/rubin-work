# Phase 3 handoff: the other products — move, fix, rebuild

> **Status:** 3a and 3b complete; **3c IN PROGRESS**. All three `fam_tables` variants are
> built, compared and **promoted** as build `20261013`: `danish_1_2` (3,385 visits over
> 106 nights, the same as the old tree), `danish_1_3_test` and `danish_1_3_v1000` (966
> visits each). The **M1M3 gradient defect is FIXED** (commit `bef17ba`): the sidecar now
> reads the four gradients from the value-added database, verified exact on one night, and
> `danish_1_2@20261014` is staged to rebuild only the steps below `mktable`.
> `cwfs_tables` is **DONE**: all four variants compared clean and **promoted** to
> `current@20261013`, each confirmed through `catalog.load()`. The only differences are
> `rotator_angle` (max abs 0.4446 deg of camera rotator angle on all four) and, on
> `d12-refitWcs`, the 2025 fold (+2,168 visit and +62,554 donut rows, every old row kept,
> shared 2026 rows unchanged but for `rotator_angle`). **Session stopped on the plan's
> spend limit** at the end of that step, not on a problem; no process is left running and
> the snakemake lock is free. Next: run the staged `20261014`, then `miw` (item 4) —
> resume steps in "Next concrete action".
> · **Last updated:** 2026-10-13 · **Kind:** working state (handoff)

Phase 3 of `notes/status/organization_plan.md`: move every remaining product's code (3a),
apply the intended fixes (3b), then rebuild every product fresh into the new tree (3c).
Read the Phase 3 part of section 8 there for the step list; this file holds the state
between sessions.

## Decisions for Aaron

Per section 12 of the plan: questions needing your judgment that do not block the rest of
the step. Answered items move to "Decisions taken".

5. **`visits.parquet` row order is not deterministic, and two stored columns depend on
   it** (open, found 2026-10-12 by item 1's after-check; **nothing is waiting on it**).
   `visits.parquet` is not sorted by (`day_obs`, `seq_num`) in either the phase 2
   reference or a fresh rebuild, because the chunks are read in parallel. Two columns are
   computed from the row order as stored:
   - `tweak_dof0..49`, from `common.dof_telemetry.derive_tweak`, which differences
     **consecutive rows**. Measured: Trim (`dof0..49`) and `dof_event_id` are identical
     on 53 of 53 visits of the reference chunk, yet 22 of 50 `tweak_dof*` columns differ
     by up to **313.9 (OFC DOF units)** on 29 of 53 visits between the two builds.
     Recomputed in a common sorted order the two agree at **max abs difference 0**.
   - `image_idx`, a positional index, permuted between builds (same value set 0..n).

   So the **stored Tweak is wrong** whenever the table is not time-ordered, which it is
   not — this is a science column, not a bookkeeping one. The options:
   - **sort `visits.parquet` by (`day_obs`, `seq_num`) before deriving Tweak**
     (recommended): one change in `run_attach_telemetry`, makes both columns
     deterministic and Tweak correct. It changes `tweak_dof*` values in any rebuild, so
     it belongs **before** 3c rather than after.
   - sort the whole table at write time, which also makes every build byte-comparable but
     touches more of the builder.
   - leave it and have readers sort, which does not fix the stored values.

   Not folded into item 1: it is outside that item's scope and moves a science column.

   **The blast radius is small, checked 2026-10-12.** The value-added database computes
   its own `tweak_dof*` with the same `derive_tweak`, but on a spine built with
   `ORDER BY day_obs` / `ORDER BY seq_num` (`build_efd_db.py:131,159`), so **the
   database's Tweak is correctly ordered**. Every science use of Tweak goes through that
   table (the item 2 survey found `visits.parquet`'s `tweak` group is read only by
   `run_chunk_status.py` and `fam_telemetry_history.ipynb`, both by prefix for an audit
   count), so the defect appears not to have reached a result. The fix is still worth
   making before 3c so the rebuilt column is right.

1. ~~**The two `value_added` backfill scripts mutate product tables in place.**~~
   **Answered 2026-10-12: route 1**, both become build steps in 3c. Moved to "Decisions
   taken" below; nothing is waiting on it.

2. **`dof_lut`'s input, before it is rebuilt in 3c** (open, raised 2026-10-12). The LUT
   fits the **plain FAM DZ**, so its DOF values are absolute — the reference shows
   `M2_dy = -2364 µm` of hexapod decentre, far outside any physical stroke. Choose before
   3c rebuilds it:
   - **plain FAM DZ** (today's behaviour): upstream stays `fam_tables`, nothing changes.
   - **post-MIW (`refit_mi`) DZ**: the upstream becomes `miw`, which is a
     variant-definition change, not just a builder change.
   Nothing is waiting on it now; no code changed. See `aos/docs/studies/lut.md`.

3. ~~**The five audit-only telemetry groups stay in `visits.parquet`**~~ **Answered
   2026-10-12: keep them, no longer provisional**, and repoint the two readers at the
   value-added database in **phase 5**, after which they are dropped. Moved to
   "Decisions taken"; kept here for the survey's reasoning. The
   survey in "Item 2" below found `camera`, `thermal` and `gradients` are read as real
   input — one of them by a **product builder** and one by a prefix-discovered ML feature
   set — so they must stay. The other five (`wind`, `lut`, `hexlut`, `trim`, `tweak`,
   228 + 8 + 10 + 51 + 50 columns) are read only by the `run_chunk_status.py` audit page
   and the `fam_telemetry_history.ipynb` history plots, both **by prefix**. Options:
   - **keep them** (what 3b did): `visits.parquet` stays ~405 columns wide, nothing
     downstream changes, and `attach_telemetry` keeps needing the EFD. Reversible.
   - **drop them and repoint the two readers at the DuckDB**: ~347 columns leave
     `visits.parquet`, but both readers discover columns by prefix with hard-coded
     expected counts, so each needs editing, and `fam_telemetry_history.ipynb` is a
     notebook in a sibling study.
   Nothing is waiting on it: `mktable --no-thermal` is what removes the EFD from the
   expensive stage, and it landed either way.

4. ~~**`rotTelPos` moves every visit, not 19 — confirm before item 1 commits**~~
   **Answered 2026-10-12: CONFIRMED**, item 1 unblocked. Moved to "Decisions taken";
   kept here for the measurement. The plan predicted that only the ~19
   ConsDB-NULL visits of the reference chunk would move, by up to 0.188 deg of camera
   rotator angle, because visits with ConsDB "agree to about 1e-5 deg". Measured on all
   53 visits of chunk `20251116_20251130`: **0 of 53 agree to 1e-5 deg**, the median
   change is **0.129 deg** and the max **0.456 deg**, and `rotTelPos` is present on all
   53 so ConsDB is never used. The 1.57e-05 deg agreement in the phase 2 A1 table was
   one visit of a different night and does not generalize.

   The change still looks right: the identity
   `rotTelPos = parallacticAngle − rotAngle − 90 deg` holds to **5.7e-14 deg** on 53 of
   53, so `rotTelPos` is the angle the rotation used. Also found: the stored
   `rotTelPos` is **not wrapped** — 32 of 53 visits read near ±360 deg — and the written
   code wraps it to (−180, 180].

   What I need: confirmation that a median 0.129 deg shift on **every** FAM visit is the
   intended consequence, since it reaches every rotator-binned MIW product rather than a
   handful of edge visits. If yes, I commit item 1 and measure the bin changes on the
   `miw` rebuild. Nothing else in 3b is blocked by it — items 2, 3, 4 and 6 are done and
   pushed.

## Decisions taken

Answered 2026-10-13 on the two 3c defects:

- **Defect A is NOT a data loss** (Aaron, 2026-10-13). The `< 5 visit_pairs` cut in
  `run_mktable.py` is per `day_obs`, so it applied inside the old chunk-level builds too:
  the phase 2 reference chunk `20251116_20251130` had 74 Butler visits over 15 nights but
  built **53 visits over 6 nights** (20251116, 17, 26, 27, 28, 29), and that chunk's 11
  "lost" nights are exactly its missing nights. A per-night job with no surviving pairs
  exits non-zero instead of silently producing nothing. The 31 `FAILED` jobs with
  `No visit pairs remaining after filtering` are **expected empty nights**, not a
  difference to explain in the comparison. **Verified against the old tree — see
  "Defect A verified" below.** None of the three options in the defect A writeup is taken;
  the filter is left alone and the external package is untouched.
- **Defect B: resubmit at 64G** (Aaron, 2026-10-13), 8 workers unchanged — then
  **superseded the same day**. The 64G set failed 70 of 87, so on Aaron's "measure first"
  instruction the worker sweep ran and showed the OOM was memory *pressure*, not demand
  (see "Defect B: the OOM was memory PRESSURE" below). The 128G fallback in the original
  rule was **not used**: the remaining 70 nights went at **96G**, which the sweep measured
  as ~8x clear of the real 12.25 GB working set, on Aaron's OK of 2026-10-13.
  `MEM` is a variable in `submit_mktable_nights.sh` and an explicit night list can be
  passed (`--nights`, `--nights-file`), committed as `09eb0b8` before the resubmission.
  Submissions: 87 nights at 64G as jobs **40708212–40708298**, then the 70 nights still
  without tables at 96G as jobs **40712911–40712982**
  (`aos/logs/mktable_submit_20261013_96g.txt`,
  night list `aos/logs/mktable_todo_nights_20261013.txt`).

### Defect B: the `Snakefile`'s 3 GB/worker figure is 3x too low — measured 2026-10-13

**The cause is mundane: `mktable` needs about 9 to 11 GB of resident set size per donut
worker, not 3 GB.** At 8 workers that is ~74 GB, so a 32G request bought roughly 4
GB/worker and killed 87 of 137 nights; the 50 that survived were the small ones. There is
no allocator effect, no per-night fixed cost, and nothing peculiar about the per-night
split.

The worker-count scan that settles it — night 20251114, `danish_1_2`, 96G request, whole-job
`MaxRSS` from `sacct`:

| workers | `MaxRSS` | per worker | wall clock |
|---|---|---|---|
| 2 | 21.66 GB | 10.83 GB | 1:01:08 |
| 4 | 40.69 GB | 10.17 GB | 0:38:04 |
| 8 | 73.59 GB | 9.20 GB | 0:22:30 |

Memory scales close to linearly in worker count and wall clock close to inversely, so
`mem_mb = 3000 * MKT_WORKERS` has the **right form** and a **3x-too-low constant**. The
"conservative bin-x2 donut estimate" in the comment at `Snakefile:265` is simply stale.

**Read `MaxRSS` from `sacct`, never `/usr/bin/time -v`, for a multi-process job.** This
section previously recorded a wrong conclusion — "footprint is flat in worker count at
12.2 GB, so memory is held per night and the `Snakefile` form is wrong" — committed in
`3c7258d` and corrected here. `/usr/bin/time -v`'s "Maximum resident set size" reports the
**largest single process** in the tree, not the sum over it, and `sweep.sh` carried a
comment asserting the opposite. The flat 12.25 / 12.27 / 12.44 GB across w8 / w4 / w2 was
flat *per-process* memory — exactly what 8 workers each holding a similar slice looks
like. Slurm's figure for the same sweep job (40711796) is **73.59 GB**, agreeing to within
2.5% with the production run of the same night (40712965, **75.41 GB**), so there was never
a path difference between `sweep.sh` and the production jobs either.

**Memory tracks night size only weakly: the footprint is mostly fixed per worker.** All
**70 of 70 `COMPLETED`** nights of the 96G set, 8 workers throughout (the table was first
written at 69, before the largest night finished; regenerated at 70). Per-night rows are in
`aos/logs/mktable_memory_20261013.csv` (night, FAM pairs, donuts, `MaxRSS` GB, elapsed,
Slurm job ID):

| quantity | min | 25% | median | 75% | max |
|---|---|---|---|---|---|
| `MaxRSS` (GB of resident set size) | 63.48 | 69.74 | 75.24 | 77.05 | 83.79 |
| FAM pairs per night (dimensionless count) | 8 | 15 | 25 | 34 | 145 |
| donuts per night (dimensionless count) | 21,712 | 40,062 | 68,108 | 87,190 | 379,458 |

`MaxRSS` (GB of resident set size) against night size, in quintiles of donut count,
n = 70 nights at 8 workers and a 96G request:

| donuts per night (count) | nights | FAM pairs (count) | `MaxRSS` (GB) range | `MaxRSS` (GB) median |
|---|---|---|---|---|
| 21,712 – 37,101 | 14 | 8 – 13 | 63.48 – 70.79 | 66.58 |
| 37,101 – 54,897 | 14 | 14 – 28 | 66.70 – 78.52 | 73.59 |
| 54,897 – 77,304 | 14 | 20 – 30 | 72.36 – 83.79 | 76.72 |
| 77,304 – 100,569 | 14 | 28 – 40 | 69.77 – 77.70 | 75.47 |
| 100,569 – 379,458 | 14 | 48 – 145 | 75.60 – 79.35 | 77.73 |

Correlation of `MaxRSS` (GB of resident set size) against night size, n = 70 nights:

| against | Pearson r | Spearman rho |
|---|---|---|
| donuts per night (count) | +0.556 | +0.748 |
| FAM pairs per night (count) | +0.542 | +0.724 |

The rank correlation is the stronger of the two, so the relation is monotonic but not
linear. OLS gives
**`MaxRSS` (GB) = 70.4 + 0.0410 × (donuts / 1000)**, residual nMAD **4.04 GB**, max
|residual| **10.60 GB**. So a **17.5x** range in donut count (21,712 to 379,458) buys only
about **15 GB**, while the intercept is **70.4 GB — 8.80 GB per worker of fixed cost**. The
largest night of the whole set, 379,458 donuts over 145 FAM pairs, peaked at only
**79.35 GB** — below three smaller nights — which is the clearest statement that night size
is not what sets the request. The per-worker constant is; this is why the worker-count scan
above is the measurement that matters and why `mem_mb` scales on workers.

**The `Snakefile`'s `mem_mb = 3000 * MKT_WORKERS` is CORRECTED to 11000.** The form is
right and the constant was 3x low. The measured per-worker figures, all at 8 workers unless
stated: the worker-count scan gives **10.83 / 10.17 / 9.20 GB per worker** at 2 / 4 / 8
workers, and the 70-night set's OLS intercept gives **8.80 GB per worker** of fixed cost
with the largest single peak at **83.79 GB (10.47 GB per worker)**. **11 GB per worker**
covers every one of those, including the 2-worker case where the per-worker figure is
highest. At 8 workers it declares 88 GB, which is 4.2% above the largest measured peak —
intentionally tight, because `mem_mb` is a Snakemake scheduling figure, not the batch
request.

**Three nights depart from the trend**, all explainable and none blocking:

| night | donuts | `MaxRSS` (GB) | residual (GB) | note |
|---|---|---|---|---|
| 20260713 | 68,954 | 83.79 | **+10.60** | the **`/repo/embargo`** night — a different repository and S3 backend, so a larger read buffer is plausible. Highest `MaxRSS` of the whole set, and still 12 GB inside the 96G request |
| 20250814 | 33,284 | 63.48 | −7.85 | lowest `MaxRSS` of the set, a small night |
| 20251128 | 21,874 | 63.88 | −6.86 | smallest donut count of the set |

The two negative residuals are the floor of the fixed cost showing through on small nights,
which the linear form over-predicts. Only 20260713 is a genuine outlier, and it is on the
safe side.

**Ruled out as a cause: `/tmp` is not tmpfs and the Butler cache is not in memory.**
Probed on a milano node 2026-10-13 (`sdfmilan255`, a 1-CPU 1 GB `srun`, record-only):

| quantity | value |
|---|---|
| `$TMPDIR` | `/lscratch/roodman` |
| filesystem of `$TMPDIR` | **xfs** on `/dev/nvme2n1p1`, 6.0 TB total, 47 GB used |
| filesystem of `/tmp` | **xfs** on `/dev/mapper/vg_raid-lv_tmp`, 20 GB total, 177 MB used |
| `/dev/shm` (the only tmpfs) | 252 GB, 25 MB used — nothing writes here |
| `$DAF_BUTLER_CACHE_DIRECTORY` | **unset**, and `submit_mktable_nights.sh` sets neither it nor `TMPDIR` |

With `DAF_BUTLER_CACHE_DIRECTORY` and `DAF_BUTLER_CACHE_DIRECTORY_IF_UNSET` both unset,
`daf_butler`'s cache manager falls back to `tempfile.mkdtemp(prefix="butler-cache-dir-")`
(`cache_manager.py:553,598`), and `tempfile` honours `$TMPDIR` — verified, it returns
`/lscratch/roodman`. So the datastore cache lands on **node-local NVMe disk**, not in
memory.

**So this cannot explain memory that tracks the cgroup limit.** A tmpfs or `/dev/shm`
cache would be charged to the job's cgroup and would grow to fill whatever limit it was
given, which is the signature that was worth excluding. Neither `/tmp` nor `$TMPDIR` is
tmpfs here, so the cache is never charged to the memory cgroup, and the apparent
limit-tracking was the per-process `MaxRSS` reading error above, not a cache effect.

**Aaron's load-bearing fact, which held throughout: the `danish_1_2` tables have not
changed** (2026-10-13) — the `danish_1_3` blitz tables are larger, `danish_1_2`'s are not.
That correctly ruled out a change in the input, and left the request size as the cause.

**Raising `mem_mb` trips Snakemake's provenance trigger, so the interactive tail is pinned
to `mtime`.** `resources` and `params` are part of Snakemake's default `--rerun-triggers`
set, so changing `rule mktable`'s `mem_mb` marks it for re-run on every chunk. Measured on
the 106-night build: the default triggers plan **106 `mktable` jobs (218 total)**, which
would redo the whole ~45 h Butler stage; `--rerun-triggers mtime` plans **1 (113 total)**,
that one being the night still running when it was measured. `finish_mktable_rebuild.sh`
now passes the flag (commit `a0dd3d0`, Aaron's call 2026-10-13). The donuts on disk are
current: only the memory declaration changed, not `mktable`'s code path. **Anyone editing
a `resources` or `params` value on an expensive rule in this Snakefile hits the same
trap** — the Snakefile's own comment at `rule combine_donuts` covers only the *script*
trigger, which is a different mechanism.

**What was spent learning this:** 87 jobs resubmitted at 64G before any measurement, which
the previous session's "next step on B" had already prescribed; then two wrong intermediate
conclusions drawn from the per-process `MaxRSS`, one of them committed. 17 of the 87
completed, so the batch spend was not total, but the measurement should have come first and
should have used `sacct`.

### Defect A verified against the old tree, 2026-10-13

The check Aaron asked for: for each of the 31 `FAILED` nights, the old-tree
`danish_1_2` `visits.parquet` (`aos/output/fam_processing/danish_1_2/visits.parquet`,
**3,385 visits over 106 distinct nights**) must hold **0 visits**.

**All 31 nights hold 0 visits in the old tree.** Nothing to report back, so nothing stops.
Rather than 31 rows of zeros, the three set identities that make the point, each exact:

| check | result |
|---|---|
| of the 31 `FAILED` nights, how many appear in the old-tree `visits.parquet` | **0 of 31** (dimensionless count) |
| old-tree nights not in the `COMPLETED` ∪ `OUT_OF_MEMORY` set | **0 of 106** — every night the old build has, the per-night set also produced or will produce |
| `COMPLETED` or `OUT_OF_MEMORY` nights absent from the old tree | **0 of 106** — the per-night set adds no night the old build lacks |
| reference chunk `20251116_20251130` | **53 visits over 6 nights** (20251116, 17, 26, 27, 28, 29), exactly as Aaron stated |

So the per-night split costs **0 FAM pairs (dimensionless count)**, not the 63 the earlier
writeup feared: those 63 pairs were never in the old build either. The 19 `COMPLETED` plus
87 `OUT_OF_MEMORY` nights partition the old build's 106 nights exactly, which is the
strongest form of the check — it rules out both loss and gain, not just loss.

Answered 2026-10-13, all five acted on in this session:

- **Batch partition: `milano`, not `roma`** (Aaron, 2026-10-13), for throughput. Account
  `rubin:developers@milano`, the form `rubinwork/common/scripts/check_batch_import.sl`
  and `optatmo/pipelines/*.sbatch` already use. Measured cause: the
  `rubin:developers@roma` association carries `GrpTRES cpu=960,mem=3840G` and the milano
  association carries **no `GrpTRES` cap at all**, which is why the probe sat `PENDING`
  on `AssocGrpCpuLimit`. Commit `8703074` makes `PARTITION`/`ACCOUNT` environment
  variables in `submit_mktable_nights.sh` with the milano values as defaults, roma still
  selectable. Aaron's earlier OKs on the probe and on the per-night set, including
  transient resubmits, **carry over unchanged to milano**.
- **`dof_lut`: the input switch stays** (Aaron, 2026-10-13). The implausible rigid-body
  magnitudes are a **regularization** question, not a reorganization item, and are not in
  3c's scope. Recorded in `aos/docs/studies/lut.md` as a hypothesis to check: the largest
  values pair up as M2 against camera decentres in the same axis (`M2_dy` −2407 µm /
  `Cam_dy` −1867 µm, `M2_dx` +727 µm / `Cam_dx` −286 µm of hexapod decentre), consistent
  with a near-null M2-plus-camera lateral mode the inversion does not suppress; the check
  is that mode's singular value and whether `n_keep` or the normalization weights admit
  it.
- **`value_added/docs/schema.md` gains the Tweak warning** (Aaron, 2026-10-13), beside
  the existing `rotator_angle` note: `fam_tables` `visits.parquet` `tweak_dof*` is the
  change since the **previous FAM visit**, not the per-exposure Tweak this database
  stores, and a study must not join them as one column.

- **Decision 5, `visits.parquet` row order: sort by (`day_obs`, `seq_num`) in
  `run_attach_telemetry` before deriving Tweak** (Aaron, 2026-10-13), done **before** any
  3c rebuild. Commit `83df1b4`. The sort is in `_visit_keys`, so every fetch and the Trim
  differencing run in time order; the sidecar joins back on (`day_obs`, `seq_num`), so no
  stored table's row order changes. Measured against the phase 2 `fam_tables` reference
  (chunk `20251116_20251130`, 53 visits): **550 of 550 other columns identical at zero
  tolerance**, all **50 of 50 `tweak_dof*` changed**, and the new stored Tweak equals the
  time-ordered Trim difference on **0 of 2,650 cells disagreeing**, against **584 of
  2,650 cells by up to 410.6 (OFC DOF units)** for the reference.
  **`image_idx` is NOT fixed by this** — see "What the row-order fix does not cover".
  **The Tweak does not equal the value-added database's Tweak, and cannot** — see the
  same section.
- **Decision 2, `dof_lut`'s input: the post-MIW (`refit_mi`) DZ** (Aaron, 2026-10-13).
  Commit `733c01d`. `dof_lut`'s upstream is now `miw` for the fits and `fam_tables` for
  the visits, both under the manifest's `inputs`; the builder takes `--fits-dir`. **Not
  rebuilt.** The old plain-FAM LUT stays only in the old tree.
  **The switch does not deliver the physically plausible DOF it was meant to** — see
  "`dof_lut`: the switch landed, the motivation did not".
- **The embargo readability probe: submitted** as shown, job **40681922**.
- **The per-night `mktable` set: OK'd**, covering every job `submit_mktable_nights.sh`
  submits and any resubmit of a night that fails for a transient (node or filesystem)
  reason. Not yet submitted — gated on the probe.
- **`ts_intrinsic_wavefront` branch `tickets/rubinwork-rottelpos`: PUSHED** to
  `origin` (`git@github.com:lsst-ts/ts_intrinsic_wavefront.git`) as a branch of the same
  name, no pull request. **Pushed commit `a6394e709e624483e865c2b1358768b8e8dbecd0`
  (`a6394e7`)** — this is the hash every 3c manifest records.

- **`rotator_angle` = `rotTelPos` on every visit: CONFIRMED** (Aaron, 2026-10-12),
  unblocking item 1. His reasoning, which settles the size question: `rotator_angle` is
  used **only to select FAM visits within a degree or two of the nominal rotator angles**
  (0, ±15, ±30 deg of camera rotator angle and similar), so a shift of a fraction of a
  degree does not matter. Using the value the donut code itself used is right, and
  fetching the angle from anywhere else is wasted effort. **Why the sources differ:** the
  camera rotator turns during the exposure to keep the sky stationary on the focal plane,
  so each source samples the angle at a different moment of the exposure — expected
  tracking motion, **not a pipeline error**. The plan's 3b item and
  `aos/docs/frame_conventions.md` both wrongly implied visits with ConsDB agree to about
  1e-5 deg (that was one visit, day_obs 20260420 seq_num 89); both are corrected, and
  `frame_conventions.md` now carries the rotator-tracking explanation.
- **The five audit-only telemetry groups stay in `visits.parquet`** (Aaron, 2026-10-12 —
  confirmed, no longer provisional). `wind`, `lut`, `hexlut`, `trim` and `tweak` (228 + 8
  + 10 + 51 + 50 columns) keep being written by `attach_telemetry`. **In phase 5**,
  `aos/code/fam_processing/run_chunk_status.py` and
  `aos/notebooks/fam_processing/fam_telemetry_history.ipynb` are repointed at the
  value-added database, and **then** the five groups are dropped from `visits.parquet`.
  Nothing to do in 3b or 3c.
- **Item 5's scope: keep `danish_1_0` as `registered: false`** (Aaron, 2026-10-12), and
  retire **only** the CWFS half `d10-wep17_3_0`. So the FAM variant and its four live
  references stay as they are, and item 5 is the `d12-refitWcs_2025` fold plus the one
  CWFS entry.
- **The two `value_added` backfill scripts: route 1** (Aaron, 2026-10-12). Both become
  **build steps in 3c**: the pointing fetch folds into `build_optical_state`, and the
  commanded-v-mode projection becomes a post-merge step of `run_build.sh`. So a merged
  database is complete when the build finishes and nothing edits a finished build, which
  is rule 5 of the plan's section 2. **Nothing changes now** — the scripts stayed as they
  were in the 3a move and still work through their shims. Item 1 of "Decisions for Aaron"
  is answered by this and is kept there only for the reasoning.
- **`value_added/code/run_build_PREMOVE_tmp.sh` deleted** with Aaron's OK, commit
  `bc97ae1`. Nothing to do.
- **`bounce_tables` split: route (a)**, the shared analysis as a library function both
  halves call (Aaron, 2026-10-12 — confirmed, no longer provisional). The builder calls
  it and writes the tables; the study rule calls it and renders, so all nine PDFs keep
  working and the per-ordinal DZ pages keep their per-visit arrays. It costs a second run
  of the analysis (167.6 s of wall clock) when both halves are built. Route (b), the
  study reading the four product tables back, was rejected because three of the nine
  pages need per-visit arrays no table carries.
  **Later option, not now:** the builder could also write the per-visit arrays as a fifth
  product table, after which the study would read only the product and the second
  analysis run would go away.
- **`dof_lut` fits the plain FAM DZ and that is recorded, not changed** (Aaron,
  2026-10-11; restated 2026-10-12 with the absolute-DOF consequence). The choice of input
  for the 3c rebuild is open above.
- **The value-added database stays on ConsDB `physical_rotator_angle`** (Aaron,
  2026-10-11) for `optical_state.rotator_angle_deg`. `value_added/docs/schema.md` records
  that for FAM visits the FAM tables' `rotator_angle` is the donut metadata `rotTelPos`
  (after 3b) and can differ by up to about 0.30 deg of camera rotator angle (blitz path),
  or be present where the database's is NULL.

**Resume protocol.** Each step writes its state here and then stops, so the session can be
cleared. A fresh session re-reads this file and the plan's sections 2–5, not a summary.
Phase 2's own state is in `notes/status/phase2_fam_tables_handoff.md`, which is complete
and is the worked example for every later product: reference build, move, zero-tolerance
comparison, manifest.

## Done and committed

| commit | what |
|---|---|
| `f7985db` | `rubinwork/products/fam_tables/refbuild/check_collections.py`, the 3c pre-flight |
| `9392f9b` | the 3c rebuild-feasibility results below |
| `71a345f` | 3a step 1: the `cwfs_tables` reference build with the unmoved code |
| `b880f2d` | 3a step 2: the move — builder, shim, `variants.yaml`, product Snakefile, tests |
| `160f8b7` | 3a steps 3 and 4 for `cwfs_tables`: verification and the zero-tolerance comparison |
| `34b9980` | `aos/Snakefile` reports the CWFS variants `wfs_variants()` skips |
| `d81cc3d` | 3a step 1: the `miw` reference build with the unmoved code |
| `b9756aa` | 3a step 2: the `miw` move — five rules, `variants.yaml` + generated `mi_config.yaml`, both readers, external registrations, tests |
| `48258b2` | 3a steps 3 and 4 for `miw`: the verification and the zero-tolerance comparison |
| `fec3741` | the resolver bridging the `aos/` study rules to the catalog |
| `3f607ab` | 3a step 1: the `coadds` reference build with the unmoved code |
| `d0e4182` | 3a step 2: the `coadds` move — builder, shim, `variants.yaml`, product Snakefile, two-upstream manifest, npz comparison, tests |
| `3d4f993` | 3a steps 3 and 4 for `coadds`: the verification and the zero-tolerance comparison |
| `581aa29` | the coadd study scripts resolve the old tree before the catalog, as `aos/Snakefile` does |
| `7d23afd` | `dof_lut`'s plain-FAM-DZ projection recorded as under review (Aaron's call) |
| `d54e5bd` | 3a step 1 for `dof_lut` and `bounce_tables`: both reference builds, scope and variants |
| `ed53b39` | 3a steps 2-4 for `dof_lut`: the move, verification and the zero-tolerance comparison |
| `26cbbfd` | the `run_bounce.py` split mapped section by section before cutting it |
| `884fb18` | 3a step 2a for `bounce_tables`: `rubinwork.bounce`, the shared library, and the four `smatrix` imports |
| `3adce0f` | 3a steps 2-4 for `bounce_tables`: the split, verification and the zero-tolerance comparison |
| `fb14ab5` | `value_added` 3a step 1 plan: the exact commands and the expected-difference list, fixed before the build ran |
| `102cef7` | `value_added` 3a step 2: the move, the product reader, the shims, the manifest rule and the doc fixes |
| `21e8bf0` | `value_added` 3a step 3: the verification |
| `6d6553a` | the MIW-route regression fixed: `decomp_path` takes the joined build name |
| `e11e731` | `guider_moments` 3a step 1 plan: scope, readers, layout and the reference-build commands |
| `b4ca14d` | `guider_moments` 3a step 1: the reference build with the unmoved code |
| `8b488c9` | `guider_moments` 3a step 2-4: the move, the two libraries, the shims, verification and the zero-tolerance comparison |
| `83df1b4` | **3c step 1**: `visits.parquet` sorted by (`day_obs`, `seq_num`) before Tweak is derived, with a shuffled-input test |
| `2f5da5c` | **3c step 2**: the per-night chunk split prefers the narrowest matching chunk, in both the submit script and the Snakefile |
| `733c01d` | **3c step 6**: `dof_lut` reads the post-MIW `refit_mi` fits; two manifest inputs, tests and `lut.md` updated. Not rebuilt |
| `a6394e7` | **pushed to `lsst-ts/ts_intrinsic_wavefront`** as branch `tickets/rubinwork-rottelpos` (3b item 1). External repo, no pull request |

**Phase 3a, `cwfs_tables`, is DONE.** The moved code reproduces the reference exactly.

**Phase 3a, `miw`, is DONE.** The moved code reproduces the reference exactly — 0 differing
columns across 12 tables at zero tolerance.

**Phase 3a, `coadds`, is DONE.** The moved code reproduces the reference exactly — 0
differing columns across 2 tables (26 and 81 columns) and 0 differing arrays across 2 npz
files (42 and 6 arrays), at zero tolerance.

**The study rules now read the catalog too** (`fec3741`), so a product rebuilt into the new
tree is visible to them before phase 5.

**Phase 3a, `dof_lut`, is DONE** (`ed53b39`). The moved code reproduces the reference
exactly — 0 differing columns across 3 tables (7, 6 and 7 columns), 0 differing fields
across 16 parquet schema-metadata fields and 11 provenance keys, at zero tolerance.
`aos/` `snakemake -n` went 37 → 34 jobs, −3 for `build_lut` as predicted.

**Phase 3a, `bounce_tables`, is DONE** (`884fb18` the library, `3adce0f` the split,
verification and comparison). The split code reproduces the reference exactly — 0
differing columns across 4 tables (17, 24, 9 and 11 columns) at zero tolerance on
verified-unique keys, and all 9 study PDFs identical in size. `aos/` `snakemake -n` is
**unchanged at 34 jobs**, not 31: the handoff's predicted −3 assumed `rule bounce` would
go away, and it survives as the plotting half. Only the file `rule all` names changed,
from `bounce_kj_stats.parquet` to `bounce_summary.pdf`; the rule still plans its 3 jobs.

What the split settled, in case a later product has the same shape:

- **Route (a), confirmed by Aaron 2026-10-12.** The reduction became
  `rubinwork.bounce` — `legs` (the 24 pure functions, moved verbatim, plus `DEFAULT`,
  `DEFAULT_BOUNCES`, `DOF22`, `CAM_HEX_DOF` and a new `resolve_config`) and
  `analysis.run_analysis`, which returns a `BounceAnalysis` dataclass. Both halves call it.
  `bounce_lib.py` keeps the ~20 plotters and **re-exports every moved name**, so its
  existing callers are unaffected and no caller outside had to change.
- **The manifest's one upstream is `miw`**, not `fam_tables` — the build reads the MIW
  build's fit table. The opposite assignment from `dof_lut`, which reads the plain FAM
  tables; `fam_variant` goes under `config` here. A test asserts `fam_tables` is not in
  `inputs`.
- **Two of the nine PDFs are declared in `aos/Snakefile`, not all nine.** Seven are
  conditional on what the run finds — five on the OFC SVD building, one on a bounce having
  a `camera_hexapod_only` leg, two on `lsst.ts.wep` — so declaring them would fail a
  legitimate run. That is the opposite call from `dof_lut`'s `lut_by_rotbin.parquet`,
  which was declared because every variant has the column it needs. The product Snakefile
  declares all four tables.
- **`bounce_kj_stats` needs `night` in its key.** Without it only 756 of the 1,890 rows
  are distinct, so the plan-shaped guess would have compared the wrong rows. A test
  asserts both halves of that.
- **`key_is_unique` moved** from `dof_lut`'s `compare_builds` into `fam_tables`', the
  shared comparison module both import. `bounce_tables` needs no mapping comparison: this
  builder stamps no schema metadata and writes no frozen provenance file.
- **`aos_fwhm` is reached lazily by path insert** from `rubinwork/bounce/analysis.py`,
  inside a function, the same idiom `rubinwork.open_loop` already uses. It is a library by
  the section 2 rules and should move into `rubinwork/` when the flat `aos/code/` modules
  do; a copy would be a second definition of the one correctable-FWHM metric in the repo.
- The one cross-study reader, `aos/code/cwfs_lut/bounce_compare.py`, resolves old tree then
  catalog, keyed on the run directory name through `_bounce_variant`.

268 product tests pass (222 + 46); import smoke test unchanged at 63/63 modules.

## `decomp_path` check for `value_added`: no variant went through the broken default

Read-only, 2026-10-12, asked for before the `value_added` move because
`miw_corner.decomp_path` was broken until the `miw` 3a move (it built
`output/<long param_set>/<mi_name>/`, a layout the tree lost when `dir_name` arrived), so
every **default** MIW lookup missed. The question was whether any `optical_state` variant
with `intrinsic_route=miw` was built through that default, and what the builder did on a
miss.

**Answer: no variant did, and a miss is a loud error, not a silent fallback.** Nothing for
"Decisions for Aaron"; no data changed.

| variant | `intrinsic_route` | `intrinsic_ref` | rows built | how the MIW was resolved |
|---|---|---|---|---|
| `v50_34__miw__consdb_v1` | `miw` | `danish_1_2_A_50_34_i_5rot` | **98,831** | **explicit `--intrinsic-ref`**, all 12 shards |
| `v50_34__batoid__consdb_v1` | `batoid` | `ofc_v13` | 102,463 | no MIW |
| `v22_12__batoid__consdb_v1` | `batoid` | `ofc_v13` | 102,463 | no MIW |
| `v50_34_rbr__batoid__consdb_v1` | `batoid` | `ofc_v13` | 102,463 | no MIW |

Evidence, in the order it settles the question:

- `state_variant` has exactly **one** `intrinsic_route=miw` row, and its `intrinsic_ref`
  is a `dir_name`-form build name, not empty.
- `build_optical_state.py:727`-`:729`: `--intrinsic miw` **without** `--intrinsic-ref` is
  an **argparse error**, so the broken default could never be reached from the command
  line for the MIW route at all.
- `rubinwork/miw_corner.py:363`-`:365` (`MiwCornerLookup.__init__`): a non-existent
  decomposition path **raises `FileNotFoundError`** naming the path. So even a path built
  by the old broken `decomp_path` would have failed loudly rather than falling back.
- The build logs, `value_added/logs/batch_state_20261006_001235_{01..12}.out`, all twelve
  shards of the one MIW build (`state_variant.created_at` 2026-10-06 00:12:45), print one
  line each: `intrinsic miw [danish_1_2_A_50_34_i_5rot]` and
  `MIW intrinsic from .../aos/output/miw/danish_1_2_A_50_34_i_5rot/intrinsic_split_decomp.parquet`
  — the `dir_name` layout, which exists.

**One stale doc found in passing, not yet fixed.**
`value_added/docs/status/build_progress.md` says the MIW variant has **0 rows** and is
"registered, never built", and warns that reading it "returns an empty DataFrame rather
than raising". That was true when the doc was written for the 2026-10-03 batoid build; the
MIW variant was built three days later and holds 98,831 rows over `day_obs` 20250728 to
20260713. The doc also puts `optical_state` at 307,389 rows over three variants; it is
**406,220 rows over four** (counts dimensionless). Fix it in the `value_added` 3a commit,
since the move touches that doc anyway.

Also noted: **`fetch_log` records only the telemetry builder's EFD groups** (nine group
names, `camera` through `wind_derived`), not the state or FAM-DZ builds. So a state
variant's provenance exists only as `state_variant.created_at` plus the batch log, which
is what made the log the deciding evidence here. The plan's section 6 already lists adding
the code git commit and build time to `fetch_log`; this is the concrete reason it matters.

**Phase 3a, `guider_moments`, is DONE** (`b4ca14d` the reference build, `8b488c9` the
move, verification and comparison) — and with it **phase 3a is complete**. The moved code
reproduces the reference exactly: **0 differing columns across 3 tables** (169, 52 and 50
columns) at zero tolerance on verified-unique keys, both text sidecars and `exposures.txt`
byte-identical, and all 9 per-exposure files identical. 350 product tests pass; import
smoke test unchanged at 59/59 modules. `snakemake -n` in `guider/` is unchanged at 1 job
for `rule all`, and every per-night job-table change is accounted for rule by rule.

**The MIW route is fixed** (`6d6553a`), not deferred to 3b: it was a regression of the
reorganization rather than a pre-existing defect. See "The MIW route regression" below.

**Phase 3a, `value_added`, is DONE** (`102cef7` the move, `21e8bf0` the verification,
and the step-4 comparison below). The moved code reproduces the reference exactly — **0
unexpected differing columns across 8 tables and registries** (196, 13, 27, 14, 7, 10, 13
and 4 columns compared) at zero tolerance on verified-unique keys, with only the seven
build-time timestamp columns differing, every one of them on a list committed **before**
the reference build ran. The merge path and both individual shards match too. `aos/`
`snakemake -n` is unchanged at 34 jobs, and `run_build.sh --dry-run` gives an identical
shard plan in both modes.

## RESOLVED: the first per-night submission, 118 of 137 jobs failed

Kept for the two defects it found, both now settled — see "The per-night `mktable` set is
COMPLETE" above for the final state. **Two defects, both caused by splitting the rebuild by
night.** Neither was transient, so neither was covered by Aaron's resubmit OK; both have
since been answered (defect A: not a data loss; defect B: the `mem_mb` constant was 3x
low).

Submitted as instructed: 137 jobs, IDs **40705595–40705733**, `FAM_BUILD=20261013`,
partition `milano`, account `rubin:developers@milano`. Every job started **RUNNING
immediately** — the milano switch worked exactly as intended, no queueing at all against
roma's 6.6 h hold on a 1-CPU probe. Submit record:
`aos/logs/mktable_submit_20261013.txt`; per-job Slurm logs
`aos/logs/mktable_<night>_<jobid>.log`; mktable's own logs
`<build>/_work/logs/<night>_<night>.mktable.log`.

| outcome | jobs | cause |
|---|---|---|
| `COMPLETED` | **19** | wrote `donuts.parquet` and `visits.parquet`; 19 of 23 chunk directories have tables |
| `FAILED` | **31** | mktable's own `< 5 visit_pairs` per-night filter — see defect A |
| `OUT_OF_MEMORY` | **87** | 8 donut workers exceed the 32 GB request — see defect B |

Tally the job IDs from `aos/logs/mktable_submit_20261013.txt`, not from a
`seq` range: the range 40705595–40705733 contains other users' jobs (two
`4FGL_J*_daily_bin2*` entries appeared in a first pass and are not ours). Filter on
`JobName` matching `mkt_*`.

### Defect A — the `< 5 visit_pairs` filter discards sparse nights when run per night

31 nights die on `ERROR: No visit pairs remaining after filtering!`, from
`Removed 1 day_obs with < 5 visit_pairs`. Each of these nights has **1 to 4 FAM pairs**
(10 nights with 1, 6 with 2, 7 with 3, 5 with 4, 3 with the count not parsed), totalling
**63 FAM pairs (dimensionless count)** that the per-night run throws away.

**This is an artifact of the per-night split, not of the data.** The filter is a
per-`day_obs` cut inside the external package's `run_mktable.py`. In a ten-night chunk a
sparse night's pairs are read alongside its neighbours' and the chunk still produces a
table; driven one night per job, the same cut is applied to each night alone and the job
exits non-zero with no output. So the phase 2 `danish_1_2` build contains these nights'
visits and a per-night rebuild would **silently lose 63 FAM pairs** — exactly the kind of
unexplained row-count difference the comparison plan says must stop the step.

The nights: 20250425 20250427 20250507 20251022 20251030 20251031 20251115 20251118
20251119 20251120 20251121 20251122 20251123 20251124 20251125 20251130 20251201 20251202
20251203 20251204 20251206 20251208 20251209 20251210 20251217 20251220 20251229 20251231
20260319 20260521 20260522.

Note that chunk `20251116_20251130` — the phase 2 reference chunk, and the one every
3b column-level check was measured on — loses **11 of its nights** this way.

**Options for Aaron, none taken:**

1. **Group the sparse nights back into their parent chunk** and submit those as
   chunk-level jobs, keeping one-job-per-night only for nights with >= 5 pairs. Keeps the
   external package untouched; the wall clock of a sparse-night chunk is small because it
   has few pairs. **Recommended.**
2. **Expose the threshold** as a `run_mktable.py` option and pass a lower value for
   per-night runs. Touches `ts_intrinsic_wavefront`, so it needs its own OK beyond the
   pre-approved 3b items, and it changes what the filter means for every caller.
3. Accept the loss. **Not recommended** — it changes the product's row count for a
   reason unrelated to any 3b fix, and would have to be added to the comparison plan's
   expected list.

### Defect B — 8 donut workers exceed the 32 GB request

87 jobs are killed by the cgroup. The failure surfaces in mktable's log as
`concurrent.futures.process.BrokenProcessPool: A process in the process pool was
terminated abruptly`, and in Slurm as `OUT_OF_MEMORY`, exit `0:125`.

| quantity | value |
|---|---|
| `MaxRSS` over the 87 OOM jobs | min **6.1**, median **17.3**, max **33.9** GB |
| `MaxRSS` over the 19 COMPLETED jobs | max **10.3** GB |
| requested per job | **32 GB** (`MEM=32G`) |
| Snakefile budget | `mem_mb=3000 * 8 workers` = **24 GB** |

So the 3 GB/worker estimate at `Snakefile:267` is too low, and 32 GB is not enough
headroom either. **The scale is the puzzle, and it is why this is recorded rather than
retried with a bigger `--mem`:** night 20250415 OOMs at **33.9 GB peak on only 6 FAM
pairs**, while 20260315 completes on **74 FAM pairs under 10.3 GB**. Memory is not
tracking the number of pairs, so simply raising `MEM` may only move the ceiling. What has
not been checked: whether the OOM nights are the **bin-x1** collections (the Snakefile's
3 GB figure is explicitly the "bin-x2 estimate") or whether something per-night in the
Butler query holds the full night's data in one worker.

**Next step on B: measure before resubmitting.** Run one OOM night interactively on an
sdfiana node with `--workers 1` and watch RSS, to separate per-worker footprint from
worker count. That is interactive work inside the approved scope, not a batch job.

### Where this leaves 3c

Both defects are settled and **the `mktable` set is complete** — see "The per-night
`mktable` set is COMPLETE" below. Defect A is not a data loss (verified against the old
tree) and defect B was the 3x-low `mem_mb` constant, now corrected.

## The per-night `mktable` set is COMPLETE, 2026-10-13

**All 137 nights are accounted for, and the set reproduces the old tree's night and visit
counts exactly.** The 70 nights resubmitted at 96G (jobs **40712911–40712982**, submit
record `aos/logs/mktable_submit_20261013_96g.txt`, night list
`aos/logs/mktable_todo_nights_20261013.txt`) all finished:

| outcome | jobs | `MaxRSS` (GB of resident set size) |
|---|---|---|
| `COMPLETED` | **70 of 70** | min **63.48**, median **75.24**, max **83.79** |
| `FAILED` | 0 | — |
| `OUT_OF_MEMORY` | **0** | — |

So **no 128G retry was needed** and Aaron's 128G fallback OK went unused. Tally by
`JobName` matching `mkt_*`, never by ID range.

The whole set, `FAM_BUILD=20261013`, `danish_1_2`:

| check | result |
|---|---|
| distinct nights submitted across all attempts (32G, 64G, 96G) | **137** |
| nights with both `donuts.parquet` and `visits.parquet` in `_work/chunks/<night>_<night>/` | **106** — and 106 of 106 have *both*, none has only one |
| nights with no table, all `FAILED` with `No visit pairs remaining after filtering` | **31 of 31**, the verified-empty nights of defect A |
| 106 + 31 against the 137 submitted | **exact partition** — 0 nights submitted without an outcome, 0 nights on disk that were never submitted |
| combined `visits.parquet` | **3,385 visits over 106 distinct nights**, keys (`day_obs`, `seq_num`) unique on 3,385 of 3,385 |
| the same, old tree `aos/output/fam_processing/danish_1_2/visits.parquet` | **3,385 visits over 106 distinct nights** — identical counts |

The combine already ran (top-level `donuts.parquet` 12.36 GB and `visits.parquet` 735 kB,
written 2026-10-13 12:58) during the `--rerun-triggers mtime` measurement.

## The three `fam_tables` builds are BUILT and COMPARED, 2026-10-13

All three rebuilt into the products tree, build `20261013`, and compared with their
old-tree builds using `compare_builds.py`, which keys on verified-unique
(`day_obs`, `seq_num`) for the per-visit tables and
(`day_obs`, `seq_num`, `detector`, `extra_donut_id`) for `donuts.parquet` — **never
positionally**. **No `set-current` has run.**

Row counts in and out, every table:

| variant | `donuts.parquet` | `visits.parquet` | `fits.parquet` |
|---|---|---|---|
| `danish_1_2` | **9,083,526** rows, old = new | **3,385** rows over 106 nights, old = new | **2,528** rows, old = new |
| `danish_1_3_test` | **3,155,734** rows, old = new | **966** rows, old = new | **873** rows, old = new |
| `danish_1_3_v1000` | **3,155,836** rows, old = new | **966** rows, old = new | **873** rows, old = new |

**Every row count is identical to the old tree on all nine tables**, so there is nothing
to explain on rows in or out. In particular `danish_1_2` has the 3,385 visits over 106
nights the old tree has, and `fits.parquet`'s 2,528 rows show the 8 sentinel nights of the
fit-quality-cut fix cost nothing (they contribute 0 fit rows to the old tree too).

`donuts.parquet` is **identical at zero tolerance** for all three variants — 48 of 48
columns for `danish_1_2` and 66 of 66 for both blitz variants, 0 columns only on one side.

### `danish_1_3_v1000` — fully explained, difference 1 only

| table | columns only in old | columns only in new | differing columns |
|---|---|---|---|
| `donuts` | 0 | 0 | **0 of 66, identical** |
| `visits` | `skyAngle` | `rotTelPos`, `rotator_angle_source` | **1**: `rotator_angle`, max **0.4446 deg** of camera rotator angle |
| `fits` | `skyAngle` | `rotTelPos`, `rotator_angle_source` | **1**: `rotator_angle`, max **0.4446 deg** |

Exactly expected difference **1**, including the `skyAngle` -> `rotTelPos` rename and the
new `rotator_angle_source` column, and the 0.4446 deg max is the "about 0.30 deg blitz
offset" the plan predicted, over 966 visits rather than one night. Nothing else differs.

### `danish_1_3_test` — fully explained, differences 1 and 3

Same as `v1000` on `donuts` (identical) and `visits` (`rotator_angle` only, max
**0.4446 deg**). Its `fits.parquet` additionally gains **4 columns** —
`intra_visit`, `extra_visit`, `median_blur_intra_arcsec`, `median_blur_extra_arcsec` —
which is expected difference **3**, the restored sidecar columns, in this variant's
smaller form: **both** trees' `visits.parquet` carry all four, and only the **new**
`fits.parquet` does, which is precisely the `combine_fits` defect of 3b item 6.
Its other 420 differing `fits` columns are the DZ coefficients at
max **5.9e-14** (µm of wavefront and its fit units) — float round-off, so **the fits
themselves are unchanged**, as item 1's prediction requires.

### `danish_1_2` — differences 1, 3, 4 and 5 all confirmed, plus ONE class not on the list

Confirmed and on the expected list:

| # | difference | measured |
|---|---|---|
| 1 | `rotator_angle` (deg of camera rotator angle) | **3,016 of 3,385** visits differ, median **0.1124 deg**, max **0.8579 deg**. Plus `skyAngle` -> `rotTelPos` and the new `rotator_angle_source`. The plan predicted median 0.129 deg / max 0.456 deg from a **53-visit** chunk; a larger max over 3,385 visits is the same effect sampled 64x more |
| 3 | the restored `cam_*` sidecar columns in `fits.parquet` | **375** columns present in the new `fits.parquet`, absent from the old |
| 4 | the dropped thermal columns | `cam_DomeXMinusTemp` is the one column only in the old tree, and it was **all-NaN on 3,385 of 3,385 visits** — a deliberate removal recorded at `rubinwork/common/visit_telemetry.py:58` |
| 5 | all 50 `tweak_dof*` (OFC DOF units) | **50 of 50** columns differ, max abs difference **4,740.9**, from the row-order fix. `image_idx` also differs and is known not to be fixed |

The DZ coefficients in `fits.parquet` differ by at most **1.46e-13**, i.e. **all 420 DZ
columns are float round-off**, so `danish_1_2`'s per-visit fits are unchanged too.

**Not on the expected list: the four M1M3 bulk thermal gradients.** This is the one thing
that needs Aaron's judgment before `set-current`.

| column (degC per metre) | visits differing | max abs difference | RMS of the column | RMS(diff)/RMS(col) |
|---|---|---|---|---|
| `x_gradient` | **1,456 of 3,006** | 0.0261 | 0.00696 | 0.383 |
| `y_gradient` | **1,456 of 3,006** | 0.6684 | 0.0194 | 0.672 |
| `z_gradient` | **1,456 of 3,006** | 0.1123 | 0.1792 | 0.038 |
| `radial_gradient` | **1,456 of 3,006** | 0.2642 | 0.0378 | 0.168 |

Also off-list, and the same mechanism: **`lut_dof5..9` on 2 visits of one night**
(20250903 seq_num 365 and 408), the largest being `lut_dof6` at **32.9** of a value of
about **4,500** (0.7%, hexapod LUT axial-force units).

**What it is NOT.** Four candidate causes were tested and excluded:

- **Not the per-night split.** `get_thermal_data` already loops **per `day_obs`**
  (`intrinsics_lib.py`), so a per-night build passes the same span a chunk build did.
- **Not a changed visit set.** The (`day_obs`, `seq_num`) key sets of the two
  `visits.parquet` are **identical**, 0 only-old and 0 only-new over 3,385.
- **Not a span-dependent bin grid.** `ThermocoupleAnalysis.load(start, end, time_bin=30)`
  returns samples on **absolute** 30-second boundaries (01:00:30, 01:01:00, ...), not
  offsets from `start`, so the grid does not move with the span.
- **Not non-determinism in the fetch.** Refetching night 20251128 **twice in one process**
  gives **max|A-B| = 0.0** and both reproduce the **new** build exactly, not the old.

**The noise explanation above was WRONG, and the three-way check against the value-added
database overturned it on 2026-10-13 — see "The three-way gradient check" below. The NEW
build is the outlier.** What survives of this paragraph is only the observation that every
telemetry column that comes from the **ConsDB transform** rather than live EFD is
**bit-identical**:
`cam_air_temp`, `m2_air_temp`, `m1m3_air_temp`, `outside_temp`, `m2_delta_t`,
`cam_m1m3_delta_t`, `dome_delta_t`, `tma_truss_temp_pxpy`, `tma_truss_temp_mxmy` — **0 of
3,385 visits differ on all nine**. That isolates the change to the one live-EFD path.

**A finding worth having: the OLD tree is internally inconsistent and the new one is not.**
In the old `danish_1_2`, `fits.parquet` and `visits.parquet` disagree with **each other**
on `cam_air_temp` for **867 of 2,291** shared visits, max **0.0283 degC** — the two tables
got their telemetry from separate fetches at different times. In the new build, `fits` and
`visits` agree at **0 differing of 2,291** on `cam_air_temp` and on every other column
checked (`m2_air_temp`, `outside_temp`, `radial_gradient`, `x_gradient`, `rotator_angle`:
0 differing each). This is why the `fits` comparison shows temperature differences the
`visits` comparison does not — those are the **old** tree's defect, not the new build's.

## The three-way gradient check: the NEW build is the outlier, 2026-10-13

Aaron's condition of 2026-10-13: `set-current` for `danish_1_2@20261013` was OK **only if**
the new M1M3 gradients agree with the value-added database. **They do not, so no
`set-current` has run for `danish_1_2`.** He also rejected the noise explanation above on
its face — `y_gradient` differs by up to 0.668 degC/m against a column RMS of 0.0194
degC/m, which no 30-second-bin noise produces — and he was right.

The check joined three sources on (`day_obs`, `seq_num`) for the **3,006** visits that have
gradients in all three: the old tree
(`aos/output/fam_processing/danish_1_2/visits.parquet`), the new build
(`fam_tables/danish_1_2/20261013`), and `visit_telemetry` in the live value-added database
(read-only connection, `m1m3_{x,y,z,radial}_gradient_c_per_m`).

| column (degC/m) | RMS(new − DB) | RMS(old − DB) | RMS of the column | new is the outlier | old is the outlier |
|---|---|---|---|---|---|
| `x_gradient` | 2.663e-03 | **8.307e-05** | 6.962e-03 | 308 | 0 |
| `y_gradient` | 1.304e-02 | **9.573e-05** | 1.941e-02 | 333 | 0 |
| `z_gradient` | 6.766e-03 | **3.029e-04** | 1.792e-01 | 725 | 1 |
| `radial_gradient` | 6.333e-03 | **5.048e-05** | 3.776e-02 | 513 | 0 |

"Outlier" means that build differs from **both** of the other two by more than 5x the
30-second-bin swing (5 x 6.7e-4 = 3.35e-03 degC/m) while those two agree within it.
**RMS(old − DB) is 30x to 140x smaller than RMS(new − DB) on every column.** Per visit,
with a 1e-9 degC/m tolerance (the median |new − old| is ~1e-15 degC/m, so almost every
visit's "difference" in the earlier table is float round-off):

| classification | visits of 3,006 |
|---|---|
| old agrees with the DB, new does not — **new is the outlier** | **1,373** |
| new agrees with the DB, old does not | **0** |
| all three agree | 1,467 |
| no two of the three agree | 83 |

**Zero visits** have the new build agreeing with the database where the old one does not.

**The mechanism, and it is deterministic, not noise: the thermocouple count.** Joining
`m1m3_thermal_r2.m1m3_n_sensors` (count of thermocouples, dimensionless) shows a clean
split — **all 1,467 visits where the three builds agree have `m1m3_n_sensors = 146`**, the
full complement, and **every one of the 1,539 visits where the new build disagrees with the
database has fewer than 146** (median 125 of 146).
Spearman rho(`m1m3_n_sensors` [count], max|new − DB| [degC/m]) = **−0.859**, p < 1e-300,
n = 3,006. So the two builds disagree exactly when a thermocouple is missing, which is a
difference in how the reduced sensor set is handled or in which sensors the EFD returned
for the span — not a 30-second bin landing differently.

**Refetching the gradients today reproduces the OLD tree and the database, not the new
build.** `ess_telemetry.get_m1m3_gradients` rerun on five nights, max abs difference in
degC/m:

| night | max&#124;refetch − new&#124; | max&#124;refetch − old&#124; | max&#124;refetch − DB&#124; | max&#124;old − DB&#124; |
|---|---|---|---|---|
| 20250706 | 6.684e-01 | 5.523e-02 | 5.523e-02 | **3.219e-03** |
| 20250813 | 1.046e-01 | 1.624e-02 | 1.624e-02 | **6.086e-04** |
| 20250826 | 2.115e-02 | 8.388e-02 | 8.388e-02 | **1.146e-03** |
| 20260402 | 2.494e-02 | 4.322e-02 | 4.322e-02 | **2.359e-03** |
| 20251117 | 5.583e-02 | 2.919e-02 | 2.919e-02 | **5.830e-04** |

`max|old − DB|` is the smallest entry on every night (6e-04 to 3.2e-03 degC/m). The
refetch tracks old/DB on four of the five. The single clearest case is night **20250706**
seq_num **662**: the new build has `y_gradient = +0.6198 degC/m` where today's refetch, the
old tree and the database all give about **−0.0486 degC/m** — a sign flip and 13x the
magnitude, which is a bad fit, not a bin shift.

One earlier claim in "What it is NOT" above is now suspect: refetching night 20251128
"reproduces the **new** build exactly, not the old". That night is one of the weakest cases
(3 of 8 visits deviating) and the claim was not reproduced in this check.

### Conclusion, one line

**The new build is wrong and the old build is right: the old build agrees with the
value-added database (RMS of old − DB, 5.0e-05 to 3.0e-04 degC/m, at the noise level) and
the new build does not (2.7e-03 to 1.3e-02 degC/m), on exactly the visits that have fewer
than 146 thermocouples.**

### Why: reduced-sensor 30-second bins give badly constrained plane fits

`fit_plane_gradients` (`ts_m1m3_utils/.../thermal_gradients.py`) drops non-finite sensors
per time row and fits anyway as soon as more than 4 survive, so a bin with a reduced sensor
set is fitted but poorly constrained. Night 20250706 refetched on its **true** span —
08:12:21 to 08:37:29 UTC, 25.1 min, the span the build actually used, 49 bins — shows it
directly:

| bins with | y_gradient, degC/m |
|---|---|
| 146 finite thermocouples | about **−0.050**, steady |
| 109–110 finite thermocouples | **−0.125** or **+0.050** — sign flips and 2.5x excursions |

The reduced-sensor bins interleave with good bins every few samples, so whether a visit's
interpolated value is sane depends on which bins bracket it. That is the whole 1,539-visit
effect, and it is why the deviation tracks `m1m3_n_sensors` so tightly.

### What is NOT explained, and is being left alone (Aaron, 2026-10-13)

The single worst value, night 20250706 seq_num 662 at `y_gradient = +0.6198 degC/m`, is
**5x larger than any bin on its own span** (max |y_gradient| over the 49 bins is 0.1252
degC/m), so no refetch at any span reproduces it. Its `radial_gradient` is +0.6183 against
an `x_gradient` of −0.0102 — radial tracking y almost exactly, the signature of a fit
dominated by a few sensors along one axis. Consistent with a transient sparse EFD return
during the build, which by its nature will not reproduce. It cannot be recomputed from the
build's own stored temperatures because the 3b sidecar shrink dropped the 146
per-thermocouple columns from `visits.parquet`. **Aaron's call 2026-10-13: stop here, do
not chase this outlier.**

### Hypotheses tested and EXCLUDED

- **Fetch span.** Narrow (one night) and wide (two nights) agree to 2.1e-03 degC/m and
  neither reproduces the new build — see the table below.
- **Non-monotonic `np.interp` input.** The gradient index is strictly 30 s with 0 negative
  steps and 0 duplicates, so `np.interp` gets a sorted x.
- **`sort_values()[0]` / `[-1]` picking the wrong span bounds.** `data_times` is a
  `DatetimeIndex`, not a Series, so positional indexing is correct; checked on three
  nights, `s[0]` and `s[-1]` are the true min and max.
- **Per-night vs per-chunk looping.** `attach_telemetry` logs "106 chunk(s)" and fetches
  once per night, so the new build already fetched per night.
- **A failed fetch silently writing bad values.** Only **14** nights logged a warning in
  `attach_telemetry.log` (13 `'NoneType' object has no attribute 'index'`, 1
  `"['coldJunction114'] not in index"`), and those yield NaN, not wrong numbers — they are
  not the 92 deviating nights. Night 20250706 logged no warning at all.

### What this changed, and what it did not

**`set-current` for `danish_1_2@20261013` RAN anyway, on Aaron's decision of 2026-10-13**,
taken after this result: the gradient columns are being replaced by the value-added
database, so they will be dead rather than wrong, and holding the promotion buys nothing.
Difference 6 is therefore recorded as a **known defect in the new build's gradient
columns**, not as an expected difference — the distinction matters for later 3c
comparisons, which must not treat a gradient difference as explained.

**The real fix, and it is now a required follow-up rather than an option:** the FAM
telemetry sidecar should read the four M1M3 gradients from the value-added database instead
of recomputing them from live EFD at build time. Recorded in the plan's section 10.

The span-width test, for the record — night 20250706 refetched over the night alone
(6 visits) and the night plus its neighbour (18 visits), max abs difference in degC/m:
narrow vs wide **2.1e-03**, narrow vs old tree 4.0e-03, wide vs old tree 4.0e-03, wide vs
database 4.0e-03, narrow vs new build **6.68e-01**, wide vs new build **6.71e-01**.

## `cwfs_tables` rebuilt and compared, 2026-10-13

Item 3 of the comparison plan. Four registered variants, all keyed to `fam_tables`
`danish_1_2@20261013`; builds named `20261013`. Compared against
`aos/output/wfs_ingest/danish_1_2/<upstream_name>/` with
`rubinwork.products.cwfs_tables.compare_builds` at zero tolerance, aligned on the
**verified-unique** key (`day_obs`, `seq_num`, `detector`, `thx_OCS`, `thy_OCS`) for
`donuts` and (`day_obs`, `seq_num`) for `visits` — uniqueness asserted on both sides by the
tool and printed in each run below.

**All four are compared, clean, and PROMOTED to `current@20261013`.** Each confirmed
resolving through `catalog.load('cwfs_tables', <variant>, file=...)` — note `load()`
requires `file=` here, because a `cwfs_tables` build holds three files. `cwfs_tables` is
**DONE**; item 3 of the comparison plan is closed.

| variant | table | rows old | rows new | cols | differing columns | the difference |
|---|---|---|---|---|---|---|
| `d12-tarts` | `donuts` | 7,813 | 7,813 | 12 | 1 | `rotator_angle`, max abs **0.4446 deg** of camera rotator angle |
| `d12-tarts` | `visits` | 977 | 977 | 8 | 1 | same, max abs **0.4446 deg** |
| `d12-paired_3mm` | `donuts` | 1,223 | 1,223 | 20 | 1 | same, max abs **0.4446 deg** |
| `d12-paired_3mm` | `visits` | 134 | 134 | 8 | 1 | same, max abs **0.4446 deg** |
| `d12-refitWcs` | `donuts` | 34,631 | **97,185** | 20 | 1 | +62,554 rows, all 2025 (the fold); on the shared 2026 rows, `rotator_angle` only, max abs **0.4446 deg** |
| `d12-refitWcs` | `visits` | 1,122 | **3,290** | 8 | 1 | +2,168 rows, all 2025; same on the shared rows |
| `d12-ai_donut` | `donuts` | 37,198 | 37,198 | 20 | 1 | `rotator_angle`, max abs **0.4446 deg** |
| `d12-ai_donut` | `visits` | 1,122 | 1,122 | 8 | 1 | same, max abs **0.4446 deg** |

On `d12-tarts` and `d12-paired_3mm` the row counts are **equal on every table**. The one
differing column is `rotator_angle`, which is **expected difference 1**, and its 0.4446 deg
maximum sits just under the 0.456 deg maximum that difference predicts from the FAM
reference chunk. No other column moved, so the 3b `rotTelPos` change reaches these tables
through the FAM `visits.parquet` they read and nothing else. Column sets are identical on
every variant — 0 columns only in the old build, 0 only in the new.

**`d12-refitWcs` gains rows, and that is expected difference 5** (the `d12-refitWcs_2025`
fold of 3b item 5: one date-keyed variant now covers 2025 and 2026, where the old tree's
`refitWcs` directory held 2026 only). Checked explicitly rather than assumed, because
`compare_builds` compares **no columns at all** when the row counts differ — its
"values: identical" line on a row-count mismatch is vacuous and must not be read as a pass:

- **Every old row survives**: 0 old-only keys on both tables, on the verified-unique key.
- **Every added row is 2025**: 2,168 visit rows and 62,554 donut rows, all with `day_obs`
  in 2025, against 0 added 2026 rows. Old by year: 2026 only (1,122 / 34,631). New by
  year: 2025 (2,168 / 62,554) plus 2026 (1,122 / 34,631).
- **The shared 2026 rows are unchanged** apart from difference 1: restricting the new build
  to `day_obs // 10000 == 2026` and comparing column by column on the sorted key gives
  **1 differing column of 8** (`visits`) and **1 of 20** (`donuts`), `rotator_angle` in
  both at max abs **0.4446 deg** — the identical value the other two variants show.

Its `wfs_mktable_validation.pdf` grew from 1,620,637 to 3,605,112 bytes (+122.45%), which
is the extra year of data in the plot, not a defect.

Two things these tables do **not** show, both expected: no `rotator_angle_source` column
(the CWFS builder does not carry it, which is why the column sets match exactly), and no
gradient columns, so the gradient defect does not reach this product.

`wfs_mktable_validation.pdf` is size-compared rather than byte-compared — matplotlib embeds
a creation timestamp. `d12-tarts` 1,429,180 vs 1,429,182 bytes and `d12-paired_3mm`
314,322 vs 314,320 bytes, both 0.00% of the old size.

**Not yet done:** `set-current` for either, pending the other two builds so the product is
promoted as a set.

## The gradient fix and the `20261014` rebuild, 2026-10-13

The fix for the defect above, required by the plan's section 10 before any product that
reads gradients is rebuilt — the coadd ML feature set discovers them by prefix.

**The code (commit `bef17ba`).** `run_attach_telemetry` no longer recomputes the four M1M3
bulk thermal gradients from live EFD. `fetch_gradients` joins
`visit_telemetry.m1m3_{x,y,z,radial}_gradient_c_per_m` on (`day_obs`, `seq_num`) through
`rubinwork.products.value_added.reader`, read-only, and a visit the database has no row for
gets NaN and is counted. `_attach_thermal` now takes only `CORE_THERMAL` from the external
package's `get_thermal_data`, and `gradients` left the set of groups that need an EFD
client — one fewer live-EFD dependency in the FAM build, which serves the batch goal in
section 10.

**Coverage does not change.** The database has gradients for **exactly** the same visits
the `20261013` build does: 3,006 of 3,385 `danish_1_2` visits, with the NaN pattern
identical on all four columns (0 visits NaN in one source and finite in the other). So the
fix moves values only.

**Verified on night 20250706**, the clearest case of the defect (6 visits):

| check | result |
|---|---|
| new gradients vs the database, all four columns | **max abs difference 0.0 degC/m** — exactly equal |
| every other sidecar column vs `20261013` | **0 differing of 385** |
| old gradients vs the database, `y_gradient` | **0.668 degC/m** — the defect |

**The build (`danish_1_2@20261014`), staged and not yet run.** Builds are immutable, so
this is a new build that reuses `20261013`'s expensive stages:

- `_work/chunks/` was **hard-linked** from `20261013` (`cp -al`, 106 night directories,
  12 GB apparent and no extra disk). The per-chunk `visits.parquet`, `fits.parquet` and
  `telemetry.parquet` — the 318 files the merge and attach write — were then **unlinked**
  (copied in place) so the run cannot touch `20261013`. Only `donuts.parquet` stays
  shared, and nothing writes it.
- The manifest records the reuse: `inputs: {"fam_tables": "danish_1_2@20261013"}`, through
  a new `write_manifest --input` flag, plus `reuse_mktable_from=20261013` under
  `config.build_options`.
- `--config telemetry_groups=gradients` fetches **only** that group and carries the other
  381 sidecar columns forward from the copied sidecars. Without it `attach_telemetry` would
  re-query the EFD for `camera`, `hexlut` and `trim`, which is live telemetry and need not
  come back bit-identical — it would manufacture differences the rebuild is meant not to
  have.
- `snakemake -n` plans exactly **6 jobs**: the three combines, `attach_telemetry`,
  `manifest`, `all`. **No `mktable` and no `fit`**, which is the point.

The command, which needs the repository-root snakemake lock:

```bash
cd /sdf/home/r/roodman/notebooks/rubin-work && \
snakemake -s rubinwork/products/fam_tables/Snakefile --cores 8 \
  --resources mem_mb=32000 --rerun-triggers mtime \
  --config variant=danish_1_2 build=20261014 \
  nights="$(find /sdf/group/rubin/u/roodman/LSST/rubin-work/products/fam_tables/danish_1_2/20261014/_work/chunks -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sed 's/_.*//' | sort -u | paste -sd,)" \
  telemetry_groups=gradients reuse_mktable_from=20261013
```

`nights=` must list the 106 nights on disk; without it the Snakefile plans the 9 declared
date-range chunks instead and would re-run the whole Butler stage. `--rerun-triggers mtime`
for the reason `finish_mktable_rebuild.sh` documents.

**The acceptance check, before any `set-current`:** the four gradient columns equal the
database at zero tolerance on every visit the database has, with the NaN count reported,
and **every other column of `visits`, `fits` and `donuts` identical to `20261013` at zero
tolerance**. Only then `catalog set-current fam_tables danish_1_2 20261014`.

The two blitz variants (`danish_1_3_test`, `danish_1_3_v1000`) are **unaffected**, checked
2026-10-13 on each `current` build: `attach_telemetry` never ran on them, so their
`visits.parquet` carries **24 columns** against `danish_1_2`'s 405, and **none of the four
gradient columns is present** in either. Nothing to rebuild there.

## Next concrete action

**STOPPED ON SPEND, 2026-10-14 05:46 UTC, at the end of a completed step.** Plan section
12: the month's total is past the pace that stays under budget — **$1,208.91 of $2,000.00
spent (60.4%)** over the 30-day period that resets 2026-11-01, a pace of **$98.76/day**
projecting to **148% of budget**. Landing under needs **$44.44/day** ($791.09 remaining
over 17.8 days). Nothing is broken, nothing is half-written, and no process is left
running; `cwfs_tables` finished and was promoted before the stop.

The previous session's `/tmp/aos3c/build_cwfs.sh` has **exited**, so the repository-root
snakemake lock is **free** and the `20261014` rebuild can start immediately. For the record
of what that build cost: `d12-refitWcs` took **3 h 15 min** and `d12-ai_donut` about as
long, against 17 min each for the two small variants — the two big ones have ~37,000 and
97,000 donut rows, and refitWcs opens **two Butlers** for the 2025+2026 fold.

**Resume here, in this order. Step 1 is the first real work:**

1. Run the `20261014` command in "The gradient fix and the `20261014` rebuild", then its
   acceptance check:
   ```bash
   cd /sdf/home/r/roodman/notebooks/rubin-work && \
   python -m rubinwork.products.fam_tables.refbuild.check_gradient_rebuild \
     --variant danish_1_2 --reference 20261013 --build 20261014
   ```
   It must print `RESULT: gradients OK, other columns OK`. Run against `20261013` itself it
   prints `gradients FAILED` — that is the defect, and is the check working.
2. Then `set-current fam_tables danish_1_2 20261014` (pre-approved if the check passes),
   and confirm `catalog.load()` resolves to it.
3. Then **`miw`**, item 4 of the comparison plan: the two `d12` variants
   (`d12-A_50_34_i` and its `build_from` child `d12-A_50_34_i_5rot`), compared against
   `aos/output/miw/danish_1_2_A_50_34_i[_5rot]/` — both old-tree directories exist with all
   four product tables — and the 3a reference in `_phase2_refbuild/miw/ref/`, expecting
   difference 2 (the theta rotation of the CCS maps) and 1 carried through the visits it
   reads. **Run it only after `20261014` is `current`**, or it carries the gradient defect
   into the coadd ML feature set.
4. Then items 5 to 8 in order: `coadds`, `dof_lut` (expected to differ **by design** —
   post-MIW vs plain-FAM input, so a shape check rather than a zero-tolerance comparison),
   `bounce_tables`, and **stop at `value_added`** (item 8), which writes the live database
   and needs its own plan and Aaron's OK.

**State of the two products 3c has finished:**

`fam_tables` `20261013` is built, compared and **promoted** to `current` for all three
variants (`danish_1_2`, `danish_1_3_test`, `danish_1_3_v1000`), each confirmed resolving
through `catalog.load()`. `danish_1_2` will be superseded by `20261014`.

`cwfs_tables` is **DONE**: all four variants built, compared clean and promoted to
`current@20261013`. **A correction worth keeping**, because it is the kind of error that
costs a session: an earlier version of this section claimed the four builds were compared
and "recorded below". **They were not** — two were built, none compared, nothing recorded,
and the build script from the cleared session was still running. Do not trust a "done and
recorded" claim in a handoff without the numbers beside it; this file now carries them.

**The `fam_tables` `danish_1_2@20261014` rebuild is staged and ready to run**: 106 night
chunks hard-linked from `20261013`, the writable per-chunk tables unlinked, the Snakefile
planning exactly the 6 downstream jobs, and the fix verified on one night. The lock is now
free. Command, the reasoning behind each flag, and the acceptance check are in "The gradient
fix and the `20261014` rebuild".

### Tried and rejected this session, so the next one does not redo it

- **Launching a second `snakemake` for `d12-refitWcs`.** Done once, on the belief the
  build had been interrupted by the `/clear`. It failed on the repository-root
  `LockException` and **changed nothing**, which is the lock doing its job. The previous
  session's shell script survives a `/clear` — check `pgrep -af snakemake` before
  concluding a build is dead. A build log that stops 22 s in looks identical to an
  interrupted one and is not evidence either way; the live evidence is the process state
  and `/proc/<pid>/io`'s `rchar` climbing.
- **Running `attach_telemetry` with all eight groups on the `20261014` rebuild.** Rejected:
  `camera`, `hexlut` and `trim` would be re-fetched from the **live** EFD, which need not
  return bit-identical values, so the rebuild would manufacture differences it is supposed
  not to have and the "every other column identical at zero tolerance" check could not
  pass. Hence `telemetry_groups=gradients` and the sidecar carry-forward.
- **Letting the Snakefile plan the `20261014` chunks itself.** It plans the **9 declared
  date-range chunks**, does not see the 106 per-night directories, and schedules 9
  `mktable` plus 9 `fit` jobs — the whole expensive Butler stage. `nights=` is not
  optional for any build made by the per-night route.
- **Hard-linking the whole chunk tree and running.** The merge writes the per-chunk
  `visits.parquet`, `fits.parquet` and `telemetry.parquet`, so those 318 files had to be
  **unlinked** (copied in place) first or the run would have modified the immutable
  `20261013`. Only `donuts.parquet` can stay shared, and it is 99% of the 12 GB.
- **Night 20250417 as the gradient test case.** It has **0 of 55 visits with gradients** in
  either source — an early 2025 night with no M1M3 thermocouple coverage — so it proves
  nothing. Use **20250706**, the clearest case of the defect.
- **Reading `cwfs_tables`/`fam_tables` `compare_builds` output on a row-count mismatch.**
  It prints `columns compared: 0` and then **`values: identical`**, which is vacuous: with
  unequal row counts it aligns nothing and compares no columns. `d12-refitWcs` hit this.
  When counts differ, restrict to the shared subset and compare column by column yourself —
  the 2025/2026 split and the per-column check in the comparison section are the pattern.
  Worth fixing in the tool so the line cannot be misread.

**`set-current` is pre-approved** for a 3c build whose comparison is clean (plan section
12, Aaron 2026-10-13): every difference on the product's expected-difference list, written
before the comparison ran, and row counts in and out explained. This **supersedes** the
earlier "no `set-current` without Aaron's OK" line in this section. A known defect is not
an expected difference, so it still blocks promotion.

**Carry forward into every later 3c comparison:** in a build whose gradients were
recomputed at build time — that is, `20261013` and anything derived from it — the four
M1M3 gradient columns are a **known defect**, not an expected difference. From
`danish_1_2@20261014` on they are read from the value-added database and are correct;
expected difference 6 becomes "gradients equal the value-added database". See "The
three-way gradient check" and "The gradient fix and the `20261014` rebuild".

The embargo probe does **not** need redoing. Its result, for the record:

**The probe is GREEN, 2026-10-13.** Job **40681922** on `roma` never started — `PENDING`
for 6.6 h on `AssocGrpCpuLimit`, the account's 960-CPU cap saturated by other group
members — so it was cancelled (`CANCELLED`, exit 0:0) and resubmitted on `milano` as job
**40705197**, identical but for `--partition=milano --account=rubin:developers@milano`.
It **COMPLETED in 11 s** on `sdfmilan261`:

| check | result |
|---|---|
| Butler opens at `/repo/embargo` | OK |
| `aggregateAOSVisitTableRaw` datasets, `day_obs` 20260713 | **24 of 24** (dimensionless count), matching the pre-flight |
| first dataset URI resolves | `s3://embargo@rubin-summit-users/u/scichris/...` |

So a Slurm node does carry the embargo S3 credentials, and the `danish_1_2` rebuild needs
no interactive chunk. Log:
`/sdf/home/r/roodman/notebooks/rubin-work/aos/logs/embargo_check_40705197.log`.

The submit command, for when the two defects are settled (Aaron's OK of 2026-10-13 covers
every job it submits, and any resubmit of a night that fails transiently; it does **not**
cover a resubmission of the 118 failed nights, because neither failure was transient):

```bash
cd /sdf/home/r/roodman/notebooks/rubin-work/rubinwork/products/fam_tables && \
FAM_BUILD=20261013 ./submit_mktable_nights.sh
```

Keep `FAM_BUILD=20261013` to reuse the 19 nights already on disk, or set a new date for a
clean build. Set it explicitly either way: the conda environment exports `BUILD=x86_64-conda-linux-gnu`
(see the two traps under "3c groundwork"). Then monitor:

```bash
squeue -u roodman
tail -f "$(ls -t /sdf/home/r/roodman/notebooks/rubin-work/aos/logs/mktable_*.log | head -1)"
```

After the jobs land, `finish_mktable_rebuild.sh` runs the interactive tail and stops
before `set-current`. **Do not run it, or anything downstream of it, until Aaron says
so** (his instruction, 2026-10-13).

### What the row-order fix does not cover

Two things the decision-5 wording expected that the code cannot deliver, both measured
2026-10-13. Neither blocks 3c.

- **`image_idx` is still non-deterministic.** It is written by the **external** package's
  fit (`dz_fitting.py`, `for img_idx, v in enumerate(visit_info)`), from the per-chunk
  `visits.parquet` that `mktable` wrote — and the `fit` rule consumes that table **before**
  `attach_telemetry` runs. A sort inside `run_attach_telemetry` therefore cannot reach it.
  On the phase 2 reference `fits.parquet` the 47 `image_idx` values are a permutation of
  0..46, not monotonic in (`day_obs`, `seq_num`). Fixing it needs either a sort in
  `mktable` (external package) or a new repo-side step between `mktable` and `fit`;
  neither was authorized. `image_idx` is a positional bookkeeping index, not a science
  column, so this is recorded rather than fixed.
- **The FAM tables' Tweak will never equal the value-added database's Tweak**, so the
  "confirm it now matches" check cannot pass — and should not. They are **different
  quantities**: the database differences Trim over **every exposure of the night** (535
  rows for `day_obs` 20251116), while `visits.parquet` differences over only the **FAM-paired
  visits** (5 that night). Measured: 72 of 50 comparisons differ for **both** the new and
  the old build (50 value columns plus 22 NaN-pattern columns), and on this chunk **no
  FAM visit has another FAM visit as its immediate database predecessor**, so there is not
  even a subset where the two definitions coincide. The database's is the true
  per-iteration correction; the FAM table's is the change since the previous FAM visit.
  The fix makes the latter well-defined and time-ordered, which is all it can do.
  **A study must not join the two as if they were the same column** — worth a line in
  `value_added/docs/schema.md` beside the existing `rotator_angle` note.

### `dof_lut`: the switch landed, the motivation did not

Decision 2's reason was the production LUT's `M2_dy = -2364 µm` of M2 hexapod decentre,
read as evidence that the plain-FAM LUT recovered absolute DOF including the as-built
intrinsic. **That reading was wrong**, measured 2026-10-13 on the `miw` 3a reference
(nights 20260315/16/17, 169 visits, variant `d12-A_50_34_i_5rot`, `n_dof=50`,
`n_keep=34`, median over visits):

| DOF | unit | plain FAM | post-MIW |
|---|---|---|---|
| `M2_dz` | µm of M2 hexapod piston | 545.28 | 551.32 |
| `M2_dx` | µm of M2 hexapod decentre | 727.23 | 719.56 |
| `M2_dy` | µm of M2 hexapod decentre | **−2407.11** | **−2407.34** |
| `M2_rx` | arcsec of M2 hexapod tilt | −0.02 | −0.02 |
| `M2_ry` | arcsec of M2 hexapod tilt | −0.01 | −0.01 |
| `Cam_dz` | µm of camera hexapod piston | −508.84 | −522.93 |
| `Cam_dx` | µm of camera hexapod decentre | −285.97 | −270.17 |
| `Cam_dy` | µm of camera hexapod decentre | −1867.15 | −1878.14 |
| `Cam_rx` | arcsec of camera hexapod tilt | −0.00 | −0.00 |
| `Cam_ry` | arcsec of camera hexapod tilt | 0.00 | 0.00 |

**Why it barely moves: the plain FAM fit already subtracts an intrinsic** — the tabulated
batoid design `zk_intrinsic` — and `refit_mi` only **substitutes the measured one for
it** (`dz_fitting.py`, `fit_focal_zernikes_streaming`, the `intrinsic_lookup` branch:
with a sidecar it replaces `zk_intrinsic` rather than adding a subtraction). The two
intrinsics differ by at most **0.022 µm of wavefront** on this slice, so the recovered
DOF differ by at most about 16 µm of hexapod motion.

What the switch does buy is **fit quality**: the averaged DZ residual over the 126
(k, j) pairs falls from **0.01028 to 0.00172 µm of wavefront RMS**, a factor of 6, and
one more visit survives (169 against 168).

**The implausible magnitudes are an ill-conditioned inversion, and stay open.** The LUT
inverts a DZ signal of only **0.037 µm of wavefront RMS** into thousands of µm of hexapod
motion, and the robust scatter is as large as the value: `|median|/scatter_mad` is
**2.40** for `M2_dy` and between **0.37 and 2.06** for the other nine rigid-body DOF
(dimensionless, both in the row's `dof_unit`). Recorded in `aos/docs/studies/lut.md`.
**For Aaron:** the input change is committed as instructed, but if the goal was physically
meaningful hexapod numbers, the next thing to look at is the regularization (`n_keep`, or
the normalization weights), not the input table.

## The 3c comparison plan

Written 2026-10-13 as instructed, before any product is rebuilt. Dependency order; each
product is compared with its old-tree build using that product's `compare_builds.py`, and
**every difference must be on this list before the comparison runs** (plan section 12: an
unexplained difference stops the step).

**Differences expected across every FAM-derived product**, from 3b:

1. **`rotator_angle` on every FAM visit**, not just the ConsDB-NULL ones: median
   **0.129 deg** and max **0.456 deg** of camera rotator angle (0 of 53 agree to 1e-5 deg
   on the reference chunk), about **0.30 deg** on the blitz path. Plus the new
   `rotator_angle_source` column, and `skyAngle` renamed to `rotTelPos` (now in rad).
2. **The CCS-map rotation**, downstream of 1: no visit changes MIW rotator bin (0 of 169
   and 0 of 53, minimum margin to a bin edge 2.631 deg), but the per-bin split angle
   theta moves by a median **0.3387 deg**, rotating the camera-fixed maps — CCS RMS change
   **7.8e-5 to 9.2e-4 µm of wavefront** (1.9 to 6.1% of each term's own RMS), OCS at most
   **7e-6 µm** (0.02%). O is theta-independent; C lives in the rotating frame.
3. **The restored 362 `cam_*` columns** in `fits.parquet` (3b item 6): present in the
   rebuild, absent from any old-tree `fits.parquet` that was re-combined.
4. **The dropped thermal columns**: `mktable --no-thermal` omits the thermal columns
   `mktable` itself used to write. The eight `attach_telemetry` groups all **stay**
   (decision 3, 2026-10-12), so `visits.parquet` keeps its ~405 telemetry columns.
5. **All 50 `tweak_dof*` columns**, from the row-order fix committed today — and
   `image_idx` still permuted, which is not fixed (above).
6. **The four M1M3 bulk thermal gradients** (`x_`, `y_`, `z_`, `radial_gradient`, degC per
   metre) and `lut_dof5..9` — now **"gradients equal the value-added database"**, an
   expected difference, as of the fix of 2026-10-13 (commit `bef17ba`, build
   `danish_1_2@20261014`). From that build on, the gradient columns are joined from
   `visit_telemetry` rather than recomputed, so a comparison against the old tree should
   show them differing from the **old** values by up to about **0.03 degC/m**
   (`x_gradient`) to **0.67 degC/m** (`y_gradient`) and equal to the database at **zero
   tolerance**. Check them against the database, not against the old tree.

   **In `20261013` and anything derived from it, this is still a KNOWN DEFECT, not an
   expected difference.** The three-way check of 2026-10-13 showed that build is the wrong
   one (RMS of new − DB 2.7e-03 to 1.3e-02 degC/m against old − DB 5.0e-05 to 3.0e-04
   degC/m; 1,373 visits where old agrees with the database and new does not, 0 the other
   way), because reduced-thermocouple 30-second bins get badly constrained plane fits, on
   exactly the visits with fewer than 146 sensors. See "The three-way gradient check".
   `danish_1_2@20261013` was promoted anyway on Aaron's 2026-10-13 decision. Every
   ConsDB-transform telemetry column is bit-identical, so only this one path carries it.

| # | product / variant | compared against | expected differences | what blocks it |
|---|---|---|---|---|
| 1 | **DONE 2026-10-13** `fam_tables` **`danish_1_2`** | `aos/output/fam_processing/danish_1_2/`, and the phase 2 reference chunk `20251116_20251130` for the column-level check | 1, 3, 4, 5. Row **order** differs too (chunks are now per night, so `combine_*` concatenates 137 inputs rather than 10) — compare keyed on (`day_obs`, `seq_num`), never positionally | the embargo probe, then the 137-job set |
| 2 | **DONE 2026-10-13** `fam_tables` **`danish_1_3_v1000`**, **`danish_1_3_test`** (blitz, `run_blitz_mktable`) | `aos/output/fam_processing/danish_1_3_v1000/`, and the phase 2 A4 blitz reference `day_obs=20260428` | 1 (about **0.30 deg** here, the larger blitz offset), 3, 5. About an hour each, so interactive — no batch ask | item 1 landing; independent of the embargo night |
| 3 | **DONE 2026-10-14** `cwfs_tables` (4 variants, all promoted) | `aos/output/wfs_ingest/danish_1_2/<upstream_name>/` per variant | 1 only, plus the **`d12-refitWcs_2025` fold** (3b item 5): one date-keyed variant now covers 2025 and 2026, so the 2025 rows appear under `d12-refitWcs` and the retired `d10-wep17_3_0` entry is gone | nothing — can run alongside item 1 |
| 4 | `miw` (the 2 `d12` variants first; the 10 others have no complete old-tree build) | `aos/output/miw/<dir_name>/`, and the 3a reference in `_phase2_refbuild/miw/ref/` | 2 (the theta rotation, the headline number), and 1 carried through the visits it reads | items 1 and 3 |
| 5 | `coadds` (`d12-A_50_34_i_5rot`; `_v2` has only an `archive/`) | `aos/output/coadd/<dir_name>/`, 2 tables + 2 npz | 2, through the MIW maps it coadds | item 4 |
| 6 | **`dof_lut`** (the 2 `d12` variants) | `aos/output/lut/<dir_name>/` | **differs from the old tree by design: absolute vs post-MIW DOF.** Not a zero-tolerance comparison at all — the input table changed (commit `733c01d`). Compare the *shape* instead: 50 DOF rows, 126 (k, j) rows, 5 populated rotator bins, the 8 schema-metadata fields and the 11 `lut_config.yaml` keys all present and self-consistent. Measured on the 3a reference, the hexapod DOF move by at most about 16 µm of hexapod motion and the DZ residual improves 6x — see the section above | item 4 (it now reads a `miw` build) |
| 7 | `bounce_tables` (4 variants) | `aos/output/bounce/<dir_name>/`, 4 tables | 2, through the `miw` fit table it reads (its upstream is `miw`, unlike `dof_lut`'s old assignment). The nine PDFs are the study half and are not part of the product comparison | item 4 |
| 8 | `value_added` rows: `fam_dz`, and the MIW-route `optical_state` variants | the live database's existing rows, read-only | 1 and 2 reach `fam_dz` through the FAM fits. **`optical_state.rotator_angle_deg` stays on ConsDB** (decision, 2026-10-11), so it does **not** pick up difference 1 — a FAM visit's two angles legitimately differ by up to about 0.30 deg. Also in 3c: the two backfill scripts become build steps (route 1, 2026-10-12), and the registry rows move to catalog names (plan section 6) | items 1 and 4. **Writes to `v1/live` only, never to the current live database in place** |

**Not in 3c:** `guider_moments` and `hsm_moments` (deferred), and the nine incomplete
`miw` variants, which have no old-tree build to compare against and so are built fresh
without a comparison.

**Phases 3a and 3b are COMPLETE**: every one of the eight products has moved, each
verified by a zero-tolerance reference comparison, `hsm_moments` stays deferred as the
plan says, and all six 3b items are committed with the branch pushed.

### The 3b items, and what each is checked against

Pre-approved by section 12 of the plan: the `rotTelPos` fix in `ts_intrinsic_wavefront`
and the sidecar shrink. Show Aaron a diff only if the work departs from that description.

| # | item | where it was found | the reference build it is checked against |
|---|---|---|---|
| 1 | **`rotTelPos`** in `ts_intrinsic_wavefront`: the donut-metadata rotator angle the FAM tables carry, differing from ConsDB `physical_rotator_angle` by up to about 0.30 deg of camera rotator angle on the blitz path | plan section 8, pre-approved | **both** phase 2 references: the `fam_tables` `danish_1_2` chunk `20251116_20251130` (rerun `mktable`, since `rotator_angle` is written there) and the blitz night `danish_1_3_v1000` `day_obs=20260428`. Then the `miw` 3a reference rebuilt on the new FAM slice. See the expected-difference list below |
| 2 | **the FAM telemetry sidecar shrunk**, and `mktable --no-thermal` | plan section 10, pre-approved | the **`fam_tables`** `danish_1_2` reference (chunk `20251116_20251130`), not the `miw` one: exactly the removed columns disappear from `visits.parquet`, every other column identical at zero tolerance, and a `miw` rebuild on the shrunk slice is identical at zero tolerance |
| 3 | **`build_fam_dz.resolve_paths`** builds `<aos_output>/<param_set>/fits.parquet` from the **long** param_set key, and `fam_variant.fits_path` records a directory that does not exist | `value_added` 3a step 1 | the `value_added` 3a `fam_dz` reference: 2,528 FAM pairs, 14 columns. Today it is reachable only with `--aos-output ../aos/output/fam_processing --param-set danish_1_2`; after the fix the **registered variant's own `fits_path`** must resolve and give the same table |
| 4 | **`fam_variant.fits_path`** itself, the registry row behind item 3 | the same | the same reference; zero tolerance apart from `computed_at`. **The LIVE database registry is not changed in 3b** — a scratch database only; 3c rebuilds the rows |
| 5 | **`d12-refitWcs_2025` folded into `d12-refitWcs`**, and **`danish_1_0` retired whole** | `cwfs_tables` 3a | the `cwfs_tables` 3a reference (`71a345f`): `aos/param_sets.yaml` must stay byte-identical except for the entries deliberately removed, and `snakemake -n` in `aos/` must lose exactly the retired variant's jobs |
| 6 | **`combine_fits` drops the 362 `cam_*` sidecar columns**, so a rebuilt `fits.parquet` loses them — the combined file was the only place they lived | phase 2 handoff, "Step 4 requirements" | a rebuilt reference `fits.parquet` carries the `cam_*` columns and is otherwise identical at zero tolerance |

**Item 2 is the FAM TELEMETRY sidecar, not the MIW `zk_intrinsic` sidecar.** What shrinks
is `visits.parquet` in a `fam_tables` build: the eight `run_attach_telemetry` groups
(`thermal`, `gradients`, `wind`, `camera`, `lut`, `hexlut`, `trim`, `tweak`,
`GROUPS` at `run_attach_telemetry.py:90`), plus the thermal columns `mktable` itself
writes unless `--no-thermal`. The `miw` product's `intrinsic_sidecar` /
`wfs_intrinsic_sidecar` rules are a different thing and are untouched by item 2.

**Item 1's expected-difference list.** The per-visit **DZ fits do not change**: the fit
runs on OCS coordinates the pipeline had already rotated with `rotTelPos`, so changing
which number the `rotator_angle` *column* reports does not move the fit. The expected
differences are exactly:

1. `visits.rotator_angle` (deg of camera rotator angle), on visits where the old value
   came from the EFD window-mean fallback (ConsDB NULL) or where ConsDB differs from
   `rotTelPos` — the 19 ConsDB-NULL visits of chunk `20251116_20251130`, and the blitz
   night, where it moves by about 0.30 deg.
2. `fits.rotator_angle` (deg), on the same visits, if `fits.parquet` carries the column.
3. The new `rotator_angle_source` column (one of `rotTelPos`, `consdb`, missing).

Downstream, a `miw` rebuild changes **only** through visits that switch rotator bin.
Those visits and their old/new bins are listed explicitly below before the item is
committed. Anything outside this list stops the item, per section 12.

Two items are **decisions, not fixes**, and are in "Decisions for Aaron" above:
`dof_lut`'s input before 3c rebuilds it (nothing in 3b depends on it), and the two
`value_added` backfill scripts (answered: route 1, both become build steps in **3c** —
nothing changes in 3b).

### Where to start

Items in order: **1** (`rotTelPos`, in the external package), **2** (the sidecar shrink),
**6** (`cam_*` in `combine_fits`), **3 and 4** (one builder plus its registry row),
**5** (config only). Each is its own commit with its own before/after check, recorded
here.

Item 1 is in `ts_intrinsic_wavefront`, which lives at
`/sdf/data/rubin/user/roodman/LSST/packages/ts_intrinsic_wavefront` (reached as
`$TS_INTRINSIC_WAVEFRONT_DIR`). **Work on a new local branch there, commit, and do not
push or open a PR** — the branch name, the diff summary and the push commands go to Aaron
instead. Note its pre-existing state, which this work must not disturb: branch
`tickets/RSO-809`, **7 commits ahead of `origin/tickets/RSO-809`**, with
`python/lsst/ts/intrinsic/wavefront/intrinsic_build_plots.py` modified and uncommitted
(18 insertions, 5 deletions).

**Read before starting 3b**: the plan's section 8 Phase 3 part for the pre-approved
description and section 10 for the sidecar item, this file's `value_added` and
`cwfs_tables` sections for the references items 3 to 5 are measured against,
`notes/status/phase2_fam_tables_handoff.md` sections "A1 — `rotator_angle`" and "Step 4
requirements" for the rotator and `cam_*` background, and
`aos/docs/frame_conventions.md`.

### Item 2: which telemetry group any study still reads from `visits.parquet`

Surveyed 2026-10-12 over `aos/`, `thermal_focus/`, `olr/`, `blocks/`, `value_added/`,
`rubinwork/`, `smatrix/`, `optatmo/`, `guider/` and the notebooks. Reads of the
value-added DuckDB, of the per-chunk `telemetry.parquet` sidecar, and of the EFD/ConsDB
fetchers in `rubinwork/common/` do **not** count — those are other sources.

| group | columns | still read from `visits.parquet`? | reader (file:line) | decision |
|---|---|---|---|---|
| `camera` | ~24 `cam_*Temp` + `cam_AmbAirtemp`, `cam_n_samp` | **yes, as science input** | `aos/code/coadd/recompute_coadd_metrics.py:511-513` discovers them by the `cam_` prefix and turns each into a per-block ML feature (`:545`, `:1198-1204`); also `aos/code/fam_processing/run_chunk_status.py:62,85-87` | **KEEP** |
| `thermal` | `cam_air_temp`, `m2_air_temp`, `m1m3_air_temp`, `outside_temp`, `m2_delta_t`, `cam_m1m3_delta_t`, `dome_delta_t`, `tma_truss_temp_pxpy`, `tma_truss_temp_mxmy` | **yes** | `rubinwork/products/coadds/builders/run_coadd_blocks_miw.py:108-113,760-765` (`THERMAL_VARS`, read per chunk); `run_chunk_status.py:54-57`; `recompute_coadd_metrics.py:68-71` uses the list to *exclude* these from `camcols`, so removing them would silently widen that ML feature set | **KEEP** |
| `gradients` | `x_gradient`, `y_gradient`, `z_gradient`, `radial_gradient` | **yes** | `run_coadd_blocks_miw.py:111,117,585,770-772` (`z_gradient` is a `SCATTER_VARS` plot axis and NaN-checked); `run_chunk_status.py:56` | **KEEP** |
| `wind` | 8 `wind_*` / `sonic_*` | audit only | `run_chunk_status.py:83-85` (prefix, schema-level presence count); `fam_telemetry_history.ipynb` | **KEEP provisionally** — see below |
| `lut` | `m1m3elev_0..155`, `m2grav_0..71` (228 cols, N) | audit only | `run_chunk_status.py:76-82` (prefix, schema-level); `fam_telemetry_history.ipynb` | **KEEP provisionally** |
| `trim` | `dof0..49`, `dof_event_id` | audit only | `run_chunk_status.py:70-72`; `fam_telemetry_history.ipynb`. Every science use goes through the DuckDB `visit_telemetry` table (`build_optical_state.py:459-470`) | **KEEP provisionally** |
| `tweak` | `tweak_dof0..49` | audit only | `run_chunk_status.py:73-75`; `fam_telemetry_history.ipynb` | **KEEP provisionally** |
| `hexlut` | `lut_dof0..9` | audit only | `fam_telemetry_history.ipynb` only. Science use is the DuckDB table, or a direct EFD fetch in `thermal_focus/notebooks/corner_z4_vs_temperature_science.ipynb` | **KEEP provisionally** |

**Three things this survey changes about item 2 as the plan describes it.**

1. **`camera`, `thermal` and `gradients` cannot be dropped.** The plan's premise was that
   the eight groups serve only downstream studies that "could read ConsDB and the
   value-added DuckDB instead". For these three the reader is not a study that could be
   repointed cheaply: `recompute_coadd_metrics.py` discovers `cam_*` **by prefix** and
   feeds it to a per-block ML fit, so dropping the group changes a science result, and
   `run_coadd_blocks_miw.py` is a **product builder**, not a study.
2. **`mktable --no-thermal` is still right, and is the part that buys the batch node.**
   The 13 `thermal` and `gradients` columns are written by `run_attach_telemetry` from
   **ConsDB and the EFD via the sidecar** as well — verified against the reference
   chunk's `telemetry.parquet`, which carries all 13 — so `--no-thermal` does not lose
   them. It is the loop keeping the EFD inside `mktable`, so it is what makes the
   expensive chunk batch-runnable.

   **What `--no-thermal` does drop, measured on the reference chunk:** the
   **146 `m1m3_tc_*`** and **50 `m1m3_dt_*`** per-thermocouple and per-cell temperature
   columns (deg C, and deg C per m for the `dt` family), 196 columns that only
   `mktable`'s own loop writes — the sidecar does not supply them. No study reads them:
   the only references are two **prefix-based audit views**
   (`aos/notebooks/fam_processing/fam_telemetry_history.ipynb` code-line 133 and
   `aos/snippets.ipynb` code-lines 265-266) and a note in
   `aos/docs/status/dof_telemetry_availability.md` recording them as temperatures, not
   DOF. So this is the one real shrink item 2 delivers: **196 columns of 600**, not a
   group removal.
3. **No `attach_telemetry` group is removed in 3b.** Five groups are
   read only by an audit page (`run_chunk_status.py`) and a history notebook, both by
   prefix with hard-coded expected counts — dropping them turns those pages into
   mismatch reports rather than breaking a result, and repointing them at the DuckDB is a
   real change to a sibling topic's code. Marked **provisional** per section 12 and
   listed in "Decisions for Aaron": the cheap, reversible default is to keep them.

So item 2 as executed is: **`mktable --no-thermal`** (`no_thermal: true` in
`aos/snake_config.yaml`), `attach_telemetry` groups unchanged. It drops the 196
`m1m3_tc_*` / `m1m3_dt_*` columns and removes the EFD from `mktable`.
`attach_telemetry` still needs the EFD for `gradients`, `camera`, `hexlut` and `trim`, so
**that stage stays interactive in 3c** — which is what the Snakefile already assumes
(`attach_telemetry=0` under `--mode batch`).

## 3b item 6 — `combine_fits` dropped the sidecar columns: DONE

`merge_sidecars` now merges the telemetry sidecar into **`fits.parquet`** as well as
`visits.parquet`, per chunk and combined, through one `_merge_into` helper
(`MERGE_TARGETS`). `fits.parquet` became an **input of the `attach_telemetry` rule**, so
the merge runs after `combine_fits`, which is the step that drops the columns. The
sidecar was already the single source of truth; it simply had one target too few.

**The check has to be read the other way round from the earlier items: the reference
`fits.parquet` is the *broken* state.** It carries only 13 of the 385 sidecar columns and
2 `cam_*`, which is the defect. So the check is (a) no pre-existing column changes, (b)
the added columns equal the sidecar, (c) `visits.parquet` does not move.

Measured on the `danish_1_2` chunk `20251116_20251130`, per-chunk and combined alike:

| quantity | value |
|---|---|
| `fits.parquet` columns | 653 → **1025** (dimensionless counts) |
| `cam_*` columns in `fits.parquet` | 2 → **27** |
| columns lost | **0** |
| pre-existing columns differing | **0 of 653**, zero tolerance |
| added columns matching the sidecar on (`day_obs`, `seq_num`) | **372 of 372** |
| `visits.parquet` columns / differing | 600 / **0**, zero tolerance |

**372 added, not the 362 the phase 2 handoff predicted**, and that is expected: 362 was
the `cam_*` count of the August build, while the sidecar now carries all eight groups, so
`fits.parquet` gains every telemetry column rather than only the camera ones. Nothing is
lost either way, which is what the item was about. The 196 `m1m3_tc_*` / `m1m3_dt_*`
columns item 2 drops are **not** in this count — they are `mktable` output, not sidecar
output, so they never reached `fits.parquet` through this path.

Checker committed as
`rubinwork/products/fam_tables/refbuild/check_sidecar_in_fits.py`, for re-use on the 3c
rebuild; its scratch tree is `_phase2_refbuild/_scratch_3b/item6/`. The full `rubinwork`
suite passes, **350 tests**.

## 3c groundwork — the per-night `mktable` submit script, WRITTEN AND NOT SUBMITTED

`rubinwork/products/fam_tables/submit_mktable_nights.sh`, with a `--dry-run` that prints
every `sbatch` line and submits nothing. Per section 12, Aaron's OK on this script covers
every job it submits, including resubmitting a night that failed for a transient reason.
**Not submitted; he has not seen it yet.**

It needed one Snakefile change: a **`nights=A,B,...`** config override that splits the
chunks into one chunk per night, each **inheriting its parent chunk's collection
override** — `danish_1_2` is a composite variant and five of its ten chunks carry one, so
a night driven without it would query the wrong collection. Verified on night 20251126:
the planned command carries `--collections .../wep_17_7_0/.../refitWcs`,
`--collection-phrase aos_fam_danish_1_2_0_wep17_7_0_bin2x_2025`,
`--programs BLOCK CWFS AOSSEQUENCE` and `--no-thermal`, into its own
`chunks/20251126_20251126/` directory.

| quantity | value |
|---|---|
| nights with `cwfs` visits in ConsDB, over the ten chunks | **138** (dimensionless count) |
| jobs the script would submit | **138**, one per night |
| default plan with no `nights=` | **unchanged**: 10 `mktable` + 10 `fit` + 3 `combine_*` + `attach_telemetry` + `manifest` + `all` = 26 jobs |
| per job | partition `roma`, 8 cpus, 32 GB, 8 h limit |

Two traps found while writing it, both worth keeping:

- **The conda environment exports `BUILD=x86_64-conda-linux-gnu`**, so a
  `BUILD="${BUILD:-$(date -u +%Y%m%d)}"` silently used it as the build name — the dry run
  printed `build=x86_64-conda-linux-gnu` and a matching build directory. The variable is
  `FAM_BUILD` now. Any script in this repo taking a `${BUILD:-...}` default has this bug.
- **`$TS_INTRINSIC_WAVEFRONT_DIR` is set to the RSP-only `/sdf/home/r/roodman/u/...`
  form**, which the root `CLAUDE.md` warns fails in a Slurm job. It is a symlink to
  `/sdf/group/rubin/u/roodman`, so the script `readlink -f`s it and exports the resolved
  `/sdf/data/rubin/user/roodman/...` path into each job rather than trusting the symlink
  on a compute node.

What the script deliberately does **not** do: `fit`, the three `combine_*` rules,
`attach_telemetry` and `manifest` run afterwards and interactively, because
`attach_telemetry` still needs the EFD (item 2). No manifest until it has run, so an
interrupted rebuild stays invisible to the catalog.

### Dry run re-measured 2026-10-12, and one defect to fix before submitting

`./submit_mktable_nights.sh --dry-run` exits 0 and prints **138 `sbatch` lines**, 145
lines of output in total — the 138 jobs the earlier session recorded, confirmed.

| quantity | value |
|---|---|
| `sbatch` lines the script would submit | **138** (dimensionless count) |
| **distinct** nights | **137** — see the defect below |
| ConsDB `cwfs` visits over those nights | **7,232** (dimensionless count) |
| visits per night | median **34**, max **291** (day_obs 20250913), min 2 |
| longest job wall clock, at 258 s + 39.1 s/visit | **3.23 h** |
| median job wall clock | **26.5 min** |
| nights exceeding the 8 h `--time` limit | **0** |

**The wall-clock figures are an upper bound, roughly 2x.** The 7,232 count is every
ConsDB `cwfs` exposure in the chunk ranges, while `mktable` reads only the paired FAM
visits that have donut datasets — the 3c pre-flight measured **3,523 Butler visits** for
this variant. So the longest night is nearer 1.5 h than 3.2 h, and the whole set is about
45 h of CPU spread over 138 jobs rather than 88 h. Either way every night fits the 8 h
limit with room, and the wall clock is set by the longest night, not the sum.

**FIXED 2026-10-13, commit `2f5da5c`.** The dry run now prints **137 `sbatch` lines over
137 distinct nights** (was 138 over 137), and night 20260713 resolves to `/repo/embargo`
with phrase `aos_fam_danish_1_2_0_wep17_7_0_embargo0713`. The fix had to go in **two**
places, not one: `night_plan` in the submit script **and** `_chunk_entries` in
`rubinwork/products/fam_tables/Snakefile`, which had the same defect — with
`nights=20260713` it emitted two entries collapsing to the same `20260713_20260713`
chunk name, and the job resolved to the broad `/repo/main` phrase
`aos_fam_danish_1_2_0_wep17_7_0_bin2x`, which holds no data for that night. Snakemake
deduped the identical output paths to one job, so the script-only fix would have looked
correct and still queried the wrong repository. Verified: a normal night (20251126) is
unchanged at `..._bin2x_2025`, and the default plan with no `nights=` is still **26 jobs**.

The original finding, kept for the reasoning:

**Defect, found by this dry run: night 20260713 is planned twice.** It
falls in two chunk ranges — the broad `[20260514, 20260731]` (`/repo/main`) and its own
single-night `[20260713, 20260713]`, which is the **`/repo/embargo`** chunk with phrase
`aos_fam_danish_1_2_0_wep17_7_0_embargo0713`. `night_plan` emits one line per
(night, chunk) pair, so both are submitted, and **both write into the same
`_work/chunks/20260713_20260713/` directory** — two jobs racing on one output, with the
`/repo/main` one finding no data there. Fix before submitting: have `night_plan` prefer
the **most specific** chunk for a night (narrowest `day_obs` range wins), which also
keeps the embargo repository and phrase correct for that night. Expected count after the
fix: **137 jobs**.

## 3c prep — the embargo probe and the post-batch script, both WRITTEN AND NOT RUN

Commit `6dc9480`. Neither is a batch submission; both wait on Aaron.

**`rubinwork/products/fam_tables/refbuild/check_embargo_from_batch.py`** — the read-only
probe for the one chunk that lives only in `/repo/embargo` (`day_obs` 20260713). It opens
the Butler, counts `aggregateAOSVisitTableRaw` datasets for that night, and resolves one
dataset's URI, which is the step a credentialed-but-unreadable repository fails on. It
writes nothing. **Interactive run passes** on `sdfiana029`: Butler opens, **24 of 24
datasets** (matching the 3c pre-flight's count), URI resolving to
`s3://embargo@rubin-summit-users/u/scichris/...`. What is unknown is only whether a Slurm
node carries the same embargo S3 credentials, which is what the one-job `sbatch` in "Next
concrete action" answers.

**`rubinwork/products/fam_tables/finish_mktable_rebuild.sh`** — the interactive tail of
the 3c `danish_1_2` rebuild, as one `snakemake` call on `rule all`. Since the per-night
donuts and visits tables are already on disk, the only pending jobs are `fit`, the three
`combine_*`, `attach_telemetry` and `manifest`, in dependency order. It runs interactively
because `attach_telemetry` still needs the EFD (item 2), and the manifest is last so an
interrupted run stays invisible to the catalog. Two guards worth keeping:

- It **refuses to run** if any per-night chunk directory has no `donuts.parquet` — a
  failed or still-running batch job — because the three `combine_*` rules would otherwise
  silently combine an incomplete set.
- It passes `nights=` listing exactly the nights found on disk, so Snakemake plans over
  the per-night chunks the batch jobs produced rather than the ten parent chunks.

It stops **before `catalog set-current`** and prints the compare and set-current commands
instead: moving `current@` is what makes a build the default every study reads, and 3c
does that only after the rebuilt product is compared with its old-tree build.

## 3b item 1 — `rotTelPos`: COMMITTED AND VERIFIED (2026-10-12)

Aaron confirmed decision 4 on 2026-10-12, so the item committed: `a6394e7` on the
`ts_intrinsic_wavefront` branch `tickets/rubinwork-rottelpos` (**pushed 2026-10-13**, no PR) and
`5bdd5f8` in this repo. The after-check is in "Item 1 after-check" below; the
before-check that stopped the item is kept here for the measurement.

The before-check contradicted the expected-difference list in three ways. Two were new
facts about the data; the third was the question Aaron answered.

### Measured on the `danish_1_2` reference chunk `20251116_20251130`, 53 visits

`rotTelPos` read from each visit's `aggregateAOSVisitTableRaw` metadata, converted from
rad to deg, against the `rotator_angle` the reference `visits.parquet` holds:

| quantity | value |
|---|---|
| visits | 53 (dimensionless count) |
| visits with `rotTelPos` present | **53 of 53** — never missing, so the ConsDB fallback never fires here |
| visits needing the (−180, 180] wrap (\|raw\| > 180 deg) | **32 of 53** |
| visits with \|difference\| ≤ 1e-5 deg of camera rotator angle | **0 of 53** |
| visits with \|difference\| in (1e-5, 0.05] deg | 10 |
| visits with \|difference\| > 0.05 deg | **43** |
| median \|difference\| | **0.129 deg** of camera rotator angle |
| max \|difference\| | **0.456 deg** (day_obs 20251126, seq_num 30) |
| identity `rotTelPos = parallacticAngle − rotAngle − 90 deg`, wrapped | holds to **5.7e-14 deg**, 53 of 53 |

### Three ways this departs from the prediction

1. **`rotTelPos` is not wrapped in the metadata.** On day_obs 20251116 seq_num 455 it
   reads **−359.824 deg** where the rotator sits at +0.18 deg; 32 of 53 visits are like
   this. Left unwrapped it would trip `rotator_flagged` (threshold 90 deg) on nearly
   every visit and corrupt any rotator binning. **Fixed in the written code** — wrapped
   to (−180, 180] as `wcsutils.calc_rotator_from_visitinfo` does — and the identity above
   confirms the wrapped value is the right quantity. This is a finding, not a blocker.
2. **All 53 visits change, not 19.** The plan predicted that visits with ConsDB agree to
   about 1e-5 deg, so only the 19 ConsDB-NULL visits would move. Measured: **0 of 53
   agree to 1e-5 deg**, the median move is 0.129 deg and the max is 0.456 deg. So the
   phase 2 A1 finding — "both builds match ConsDB to 0.000000000 deg on the 34 visits
   that have it" — says ConsDB and the *old column* agree, which is true; it does not
   say ConsDB agrees with `rotTelPos`. On this chunk they differ by 0.1 to 0.5 deg.
   The 1.57e-05 deg agreement in A1's table was measured on **one** visit of a different
   night (day_obs 20260420, seq_num 89) and does not generalize to this chunk.
3. **So the "19 ConsDB-NULL visits" framing does not apply to this chunk at all.**
   `rotTelPos` is present on all 53, so the new code uses it on all 53 and ConsDB is
   never consulted. The old column used ConsDB on 34 and the EFD window-mean on 19; the
   new one uses `rotTelPos` everywhere. Every visit's `rotator_angle` moves.

The change is still **the right one** by Aaron's 2026-10-11 reasoning — the column must
hold the angle the CCS → OCS rotation actually used, and the identity check confirms
`rotTelPos` is that angle to 5.7e-14 deg. What changed is the **size and reach** of the
effect: not 19 visits by up to 0.188 deg, but every visit by a median 0.129 deg.

### What is waiting on Aaron

**Decision 4 below.** The downstream question the plan framed as "only visits near a
rotator-bin edge can change MIW bin" now has a much larger input: with every visit
moving by a median 0.129 deg, how many change bin depends on the bin edges and is no
longer a handful by inspection. I have not run the MIW rebuild, because committing the
FAM-side change first is what the item order requires and that is the step now stopped.

### Where the code sits

Nothing committed. In `ts_intrinsic_wavefront`, local branch
**`tickets/rubinwork-rottelpos`** (**pushed to `origin` 2026-10-13**, commit `a6394e709e624483e865c2b1358768b8e8dbecd0`, no PR):

- `visit_meta['skyAngle']` (which held `meta['rotAngle']`, the boresight sky angle) →
  `rotTelPos` from `meta['rotTelPos']`, rad, in both table builders.
- `get_rotator_data` is **ConsDB-only**; `get_efd_rotator_angles` and
  `get_visitinfo_rotator_angles` stay as documented standalone cross-checks, out of the
  chain. Exactly **one** `makeEfdClient()` remains in `intrinsics_lib.py`, inside
  `if include_thermal:` — which is what item 2's `--no-thermal` turns off, so the mktable
  path then creates no EFD client at all.
- new `resolve_rotator_angle(table, rotator_df)`: `rotTelPos` wrapped to (−180, 180]
  first, ConsDB `physical_rotator_angle` as the fallback, NaN plus a per-visit report
  when both are missing, and a `rotator_angle_source` column
  (`rotTelPos` / `consdb` / `missing`). Unit-tested, 3 of 3 source cases.
- `scons` re-run: `version.py` and `bin/` regenerate; flake8 findings in
  `intrinsics_lib.py` went **49 → 47** (the change removed two, added none). The one
  failing linter test is pre-existing across the package.

In this repo, commit `5bdd5f8`: the blitz half — `blitz_reader` record `skyAngle` →
`rotTelPos` (rad from `rot_tel_pos`), `run_blitz_mktable.resolve_rotator_angle` with the
same precedence, wrap and source column, and `VISITS_COLUMNS` updated. Unit-tested, 3 of
3 cases; the full `rubinwork` suite passes, 350 tests. Docs in the same commit:
`aos/CLAUDE.md`, `aos/docs/frame_conventions.md`, `aos/docs/telemetry.md` (whose pointing
row also had `alt`/`az`/`ra`/`dec` mislabelled deg when they are rad — verified against
the reference, max 6.28).

### Item 1 after-check: the two references, 2026-10-12

**The Snakemake path**, `danish_1_2` chunk `20251116_20251130` rebuilt with thermal on,
to match the phase 2 reference's configuration (so this isolates item 1 and item 6, not
item 2). Scratch tree
`_phase2_refbuild/_scratch_3b/item1/products/fam_tables/danish_1_2/ref_after/`.

| table | rows | result |
|---|---|---|
| `donuts.parquet` | 145,264 | **identical**, 48 of 48 columns, zero tolerance |
| `visits.parquet` | 53 | `skyAngle` → `rotTelPos`, `rotator_angle_source` added; `rotator_angle` and 50 `tweak_dof*` differ |
| `fits.parquet` | 47 | 653 → **1026** columns (item 6's 372 sidecar columns + the 2 new); `rotator_angle` and `image_idx` differ |

`rotator_angle` reproduces the before-check exactly, so the measurement is confirmed on a
real build:

| quantity | value |
|---|---|
| `rotator_angle_source` | **`rotTelPos` on 53 of 53** visits, ConsDB never used |
| median \|change\| | **0.1287 deg** of camera rotator angle |
| max \|change\| | **0.4558 deg** |
| visits agreeing to 1e-5 deg | **0 of 53** |
| `rotator_flagged` changed | **0 of 53** visits; 0 flagged before and after |
| wrap works | raw `rotTelPos` reaches **−6.303 rad** (−361.1 deg); output stays in −1.132 to +14.779 deg |

**Two differences are NOT item 1 and are a pre-existing defect, found here**
(`tweak_dof*` on `visits`, `image_idx` on `fits`). Both are **row-order** artifacts:

- Neither build's `visits.parquet` is sorted by (`day_obs`, `seq_num`) — the row order is
  not stable between two runs of the same chunk, because the chunks are read in parallel.
- `common.dof_telemetry.derive_tweak` differences **consecutive rows as stored**, and
  `image_idx` is a positional index assigned in table order. So both depend on an order
  that is not deterministic.
- Proof it is only the order: **Trim** (`dof0..49`, read straight from the EFD) is
  identical on 53 of 53 visits and `dof_event_id` on 53 of 53, yet 22 of 50 `tweak_dof*`
  columns differ by up to 313.9 (OFC DOF units) on 29 of 53 visits. Recomputing
  `derive_tweak` for **both** builds from their own identical Trim and event ids in a
  common sorted order gives **max abs difference 0** across all 50 columns with matching
  NaN patterns. `image_idx` likewise holds the same value set 0..n, permuted.

So the stored Tweak is wrong whenever `visits.parquet` is not time-ordered, which it is
not. **Not fixed here** — it is outside item 1 and changing it moves a science column, so
it is logged as a new item below rather than folded into this commit.

**The blitz path**, `danish_1_3_v1000` `day_obs=20260428`, 12 visits, rerun through the
moved builder. Scratch tree `_phase2_refbuild/_scratch_3b/item1/blitz_after/`.

| table | rows | result |
|---|---|---|
| `donuts.parquet` | 40,548 | **identical**, 66 of 66 columns, zero tolerance |
| `visits.parquet` | 12 | `skyAngle` → `rotTelPos`, `rotator_angle_source` added, `rotator_angle` changed |
| `fits.parquet` | 12 | same three changes, 447 other columns identical |

**Better than the plan predicted, and the prediction's framing does not fit this night:**
the old `rotator_angle` was **NaN on 12 of 12** visits (ConsDB has no
`physical_rotator_angle` for them), and the new one is populated on 12 of 12 from
`rot_tel_pos` (14.083 to 59.117 deg of camera rotator angle). So on this night the change
**fills** the column rather than shifting it by about 0.30 deg — there was no old value
to shift. 12 of 12 visits still pass every quality cut and 0 of 12 are flagged `bad_fit`,
as in the reference.

### Item 1 downstream: NO visit changes rotator bin, but the split angle moves

The plan's remaining question was which visits change MIW rotator bin. **None do**, on
either slice, and the margin is comfortable rather than lucky.

Measured on the **`miw` 3a reference's own FAM slice** — nights 20260315/16/17, 169
visits — by reading `rotTelPos` from each visit's metadata with the cheap header reader
(107 s for 169 visits) and binning both the old and the new angle under the build's
**half-open `[lo, hi)`** convention (`measured_intrinsic.py:1054-1056`; the closed
interval is wrong):

| quantity | value |
|---|---|
| visits read, `rotTelPos` missing | 169, **0 missing**; 60 of 169 needed the wrap |
| median / max \|change\| | **0.4029 / 0.4446 deg** of camera rotator angle |
| **visits changing rotator bin** | **0 of 169** |
| bin occupancy, before and after | **identical**: bin 0 `[-65,-55)` 36, bin 3 `[-20,-10)` 24, bin 4 `[-3,3)` 37, bin 5 `[10,20)` 36, bin 8 `[55,65)` 36 (visit counts) |
| min margin to a bin edge | **2.631 deg**, against a max change of 0.4446 deg — a factor of about 6 |

Same on the `danish_1_2` reference chunk: **0 of 53** visits change bin, occupancy 52 in
bin 4 and 1 in bin 5 before and after, min margin 0.990 deg against a max change of
0.4558 deg. This is Aaron's reasoning confirmed numerically: the bins sit a few degrees
clear of the nominal angles, so a few tenths of a degree cannot move a visit.

**But the per-bin split angle theta does move, and that was not in the prediction.**
`run_intrinsic_split` sets each bin's theta to the **mean `rotator_angle`** of that bin's
visits (`intrinsic_split.mean_rotator`, called at `run_intrinsic_split.py:142`), and theta
enters the decomposition as `C(r, phi - s*theta)`. So the visit sets are unchanged while
the OCS/CCS split rotates:

| bin | theta before | theta after | change |
|---|---|---|---|
| `[-65,-55)` | −60.0075 | −59.6435 | +0.3640 |
| `[-20,-10)` | −15.0620 | −14.6620 | +0.4000 |
| `[-3,3)` | −0.0063 | +0.3323 | +0.3387 |
| `[10,20)` | +14.6683 | +14.9842 | +0.3158 |
| `[55,65)` | +59.9963 | +60.3313 | +0.3350 |

(deg of camera rotator angle; median \|change\| 0.3387 deg, max 0.4000 deg.)

**Measured effect on the MIW maps**, by rerunning `run_intrinsic_split` twice against the
3a reference's already-built per-bin grids — once with the reference `fits.parquet` and
once with a copy carrying the new `rotator_angle` — so only theta differs. The before run
reproduced the reference's thetas `[-0.01, 60.0]` exactly; the after run gave
`[0.33, 60.33]`. The 3,985-point field grid is identical, so the maps compare point by
point:

| family | RMS change (µm of wavefront) | as a fraction of that term's own RMS |
|---|---|---|
| **OCS** (telescope-fixed), 21 terms | 1e-6 to **7e-6** | **0.00 to 0.02%** |
| **CCS** (camera-fixed), 21 terms | 7.8e-5 to **9.2e-4** | **1.9 to 6.1%** |

Largest absolute change **0.000919 µm of wavefront** (`Z6_CCS`, against its own RMS of
0.033129 µm); largest fractional change `Z25_CCS` at 6.1%. The `intrinsic_split_rms`
table agrees: `C_cam` moves 4.4e-5 µm of wavefront, `residRMS` 2.3e-5 µm, `O_tel` only
7.8e-7 µm, and `intrinsic_split_decomp` is bit-identical in every structural column
(`s`, `part`, `n_spin`, `n_r` all change by exactly 0).

**This is the expected behaviour, not a defect.** O is theta-independent, so it does not
move; C is defined in the rotating frame, so a common theta offset rotates it by that
angle. The change is small in absolute terms — under 1e-3 µm of wavefront — and it is the
*more correct* value, since theta is now the angle the pipeline actually rotated with.
Nothing outside `rotator_angle`, `rotator_angle_source`, `rotTelPos` and the theta-driven
CCS maps changed.

Scratch tree for the two splits: `_phase2_refbuild/_scratch_3b/item1/split/{before,after}/`.

Before-check script committed as
`rubinwork/products/fam_tables/refbuild/read_rottelpos_meta.py` — **and note how to read
this metadata cheaply**, which cost most of the session's wall clock to learn:
`aggregateAOSVisitTableRaw` is a **1.1 GB ECSV text file**, so `butler.get` costs
**393 s per visit** (measured). The scalar metadata is a `meta: !!omap` block in the first
~60 `#` lines, as one-key maps; `yaml.safe_load` refuses it (`!!omap` and
`!numpy.ndarray` tags), so match the scalars with a regex and stop once found. That reads
all 53 visits in **5 s** instead of 5.8 h.

## 3b item 5 — the `refitWcs` fold and the `danish_1_0` CWFS entry: DONE

Commit `662f91a`. Two changes, exactly as Aaron scoped them on 2026-10-12.

**The fold.** `d12-refitWcs` now carries a date-keyed `collections` list instead of one
`collection`, so one variant covers both seasons as the FAM `danish_1_2` already does per
chunk:

| `day_obs` window | collection |
|---|---|
| [20260101, 20261231] | `.../danish_1_2_0_alpha0/wep_17_6_1/dv_4_5_0/bin_x2/refitWcs` |
| [20250101, 20251231] | `.../danish_1_2_0/wep_17_7_0/dv_4_7_0/bin_x2/paired/refitWcs/2025` |

Verified against the Butler: the 17.6.1 collection holds nights **20260315..20260513**
and the 17.7.0 reprocessing **20250415..20251231** — both inside their windows and
disjoint, so a night resolves unambiguously.

What made this a builder change rather than a config change: `run_wfs_mktable` opened
**one** `Butler(repo, collections=...)` per `--wfs-name`. It now resolves the collection
per visit through two new helpers, `parse_collection_ranges` and
`collection_for_day_obs`, and **caches a Butler per collection** (opening one costs
seconds and the walk is per visit). A visit on a night no collection covers is **skipped
and counted** (`n_no_coll`, reported in the summary line) rather than queried against the
wrong collection. A bare-string `collection` still works through the same resolver, so
the four unfolded variants are untouched.

`cwfs_tables.reader` gained `collections` in `BUILDER_FIELDS`, and `_builder_entry`
renders a folded entry as a dict (a bare-collection entry stays a bare string, which is
what the other entries' byte-identity depends on).

**The retirement.** `d10-wep17_3_0` is gone — never built, and nothing reads a CWFS table
for `danish_1_0`. **The FAM variant `danish_1_0` stays** as `registered: false`, per
Aaron: the four live references to its long param_set key all still resolve
(`aos/calibration/miw/intrinsic_split_maps_v1.provenance.yaml:2,4` and
`aos/calibration/README.md:42,45`; `notes/aos-measured-intrinsics/make_figures.py:23` and
`provenance.md:5-6`; the `overrides:` block at `aos/analysis_config.yaml:148`; and a
`fam_tables` test).

**Checks.** `aos/param_sets.yaml` changes in **exactly the two intended places and
nowhere else** (10 insertions, 4 deletions: the `danish_1_0` `wfs_collections` block
removed, `refitWcs` folded, `refitWcs_2025` gone). Before state regenerated rather than
trusted: md5 **`0f72da651cf9319cc283754c614051fd`** and **34 jobs**, both matching the
recorded values. After the change `snakemake -n` in `aos/` is **unchanged at 34 jobs**,
with a byte-identical job table — expected, since `refitWcs_2025` was never built and
`d10-wep17_3_0` was unregistered, so neither had a study job to lose. The registered
variant count goes 5 → 4. **352 tests pass** (350 before, plus two for the resolver).

## 3b items 3 and 4 — the dead long-`param_set` path in `fam_dz`: DONE

One change fixes both. `build_fam_dz.resolve_paths` built
`<aos_output>/<param_set>/fits.parquet` from the **long** param_set key, a layout the
tree lost when `dir_name` arrived; it now maps the key to its `fam_tables` variant with
`fam_tables.variant_for_param_set` — for `fam_tables` the variant name **is** the
`dir_name` — and resolves **old tree, then the catalog's `current` build**, the same
precedence as the `aos/Snakefile` resolver. Either name may be passed in, and the error
names both places looked. Item 4 needed no separate change: `fam_variant.fits_path` is
written from the resolved path (`build_fam_dz.py:373`), so it now records a path that
resolves.

A second defect fixed in passing: `DEFAULT_AOS_OUTPUT` was `aos/output`, missing the
`fam_processing` level — `aos/output` is the parent of every study's output, and
`--aos-output`'s own help text already said "the `aos/output/fam_processing`
directory".

**Check: the builder was run with no `--aos-output` override**, so it had to resolve the
registered variant's own path — which is the thing that could not be done before.
Against the `value_added` 3a reference, into a **scratch** database copy
(`_phase2_refbuild/_scratch_3b/item34/`; the live database is untouched, 3c rebuilds it):

| quantity | value |
|---|---|
| `fam_dz` rows | 2,528 reference, **2,528** rebuilt (dimensionless counts) |
| FAM pairs read / with a NaN DZ coefficient | 2,528 / **0** |
| SVD | `U_eff = (126, 34)`, `n_dof = 50`, `n_keep_eff = 34`, `n_kj = 126` |
| `fam_dz` columns identical | **15 of 16**, zero tolerance |
| columns differing | `computed_at` only — on the expected list |
| `fam_variant` rows / columns differing beyond the expected | 1 / **none** |
| `fam_variant.fits_path` | was `../aos/output/fam_processing/danish_1_2/fits.parquet` (relative, only valid from `value_added/`), now the absolute path, **resolves on disk** |

The 3a table above records "14 columns" for `fam_dz`; that is the count of **value**
columns, excluding the two key columns `visit_id` and `fam_variant_id`. The table has 16
columns in total, so 15 identical plus `computed_at` is the same statement.

Checker at `rubinwork/products/value_added/refbuild/check_fam_dz_paths.py`.

## Constraints found the hard way in 3a, which 3b and 3c will meet too

Carried forward from the `bounce_tables`, `value_added` and `guider_moments` moves.

- **A declared conditional output fails a legitimate run.** `dof_lut` taught "declare
  every output"; `bounce_tables` showed the limit — seven of its nine PDFs depend on what
  the run finds, so declaring them would have turned a healthy build into a failure. Check
  whether an output is unconditional before declaring it. `guider_moments`'
  `detector_counts_<night>.txt` *is* unconditional, so it is declared.
- **Do not predict a job-count change from the product's name.** The handoff predicted
  `aos/` 34 → 31 because every earlier product's rule left; `bounce`'s rule survived and
  the count did not move. `guider_moments` moved a **checkpoint**, which changed *when*
  the study fan-out is planned rather than how much work there is. Run `snakemake -n`
  before and after and diff the job table, do not reason about it.
- **Moving a checkpoint moves a fan-out.** Any study rule that fanned out from a product's
  checkpoint loses its source. Have it read the product's own listing through the resolver
  rather than re-deriving it — a second Butler query would be a second definition of the
  same set and could disagree.
- **A relative hop between topics breaks silently on a move.** `makeHsmFunc`'s
  `sys.path.insert(__file__/../../optatmo/code)` would have resolved to a non-existent
  path and returned `None`, dropping every `hsm_*` column with no error. The import-side
  twin of `value_added`'s default-path trap. Grep a moving file for `..` in a path before
  moving it.
- **A shim must give the same module object**, `sys.modules[__name__] = _real`, not a
  namespace copy, and the identity test needs **one fresh interpreter per shim**: in a
  single process a sibling's `sys.path` bootstrap can legitimately load the same file
  twice, and an in-process check then compares the wrong pair.
- **Check an "unused import" against the pre-move state before fixing it.**
  `bounce_lib.py` had five genuinely unused imports before the split and
  `guider/code/extract_adjacent_psf_moments.py` has four; attributing them to the move
  would be wrong. `git stash` + `pyflakes` is the cheap way to tell. **Use `git stash` with
  care when the move has left untracked files** — `git stash pop` refuses to overwrite an
  untracked file of the same name, and the stash then has to be inspected rather than
  trusted.
- **A test that matches docstring prose breaks on correct code.** Assert on the import
  lines, the declared rule outputs or the non-comment lines, not on whole-file text, where
  the old name legitimately appears in a comment explaining the move.
- **A wildcard must appear in every output and log of a rule.** A `{night}` wildcard beside
  a fixed `skipped_visits.txt` output raises `RuleException`. A one-partition-per-invocation
  Snakefile takes the partition from `--config` instead.
- **`--config` and `--resources` are greedy.** A target after either is swallowed; put the
  target last behind `--`, or before the greedy flag.

**Not attempted, deliberately:** `aos_fwhm.py` was left as a flat `aos/code/` module. It
is a library by the section 2 rules — 88 lines, pure computation, writes nothing, already
reached from `rubinwork.open_loop` — and `rubinwork/bounce/analysis.py` reaches it by a
lazy path insert. Promoting it is a separate change to a sibling topic's flat modules,
which the root `CLAUDE.md` pins. A candidate for whenever the flat `aos/code/` modules
move. The same applies to `guider/code/guider_utils.py`, which only one notebook imports
and which therefore is not yet a library by the section 2 test.

## 3a, `guider_moments` — the move

The last product of 3a, and the **second incremental** one, so `value_added` is the worked
example rather than the five snapshot products.

### Scope: what is the product and what stays, agreed 2026-10-12

**The product** is the per-night guider moment tables and the rules that build them:

| file | role |
|---|---|
| `guider/code/run_guider_moments.py` | the builder: one exposure to `<seq>_{moments,stars,metrics}.parquet` |
| `guider/code/combine_moments.py` | per-seq to per-night `guider_{moments,stars,metrics}_<night>.parquet` + `skipped_visits.txt` |
| `guider/code/list_guider_exposures.py` | the Butler discovery that keys the night's partition |
| `guider/code/guiderMoments.py` | the moment decomposition and **`SCHEMA_VERSION = 6`**, which *is* the variant |
| rules `discover_night`, `moments`, `combine` | the three rules that write those files |

**Stays in `guider/`** — study or tooling, every one of them: the RubinTV plots
(`run_guider_rubintv_plots.py`, rules `rtvplot`/`rtvplots`), the movies
(`run_guider_movie.py`, rules `movie`/`movies`), the per-day QA PDF
(`run_guider_dayplots.py`, rule `plots`), the adjacent-CCD science side
(`extract_adjacent_psf_moments.py`, rule `psfmoments`), the saturation study
(`run_guider_saturation.py`), the atmosphere tracker and simulator (the eleven
`guider_atmo_*.py`), `run_guider_whiskers.py`, `run_guider_validation.py`,
`check_rotator_field.py`, `inspect_guider_outputs.py`, `guider_adjacent_ccds.py`,
`guider_science_collections.py`, `fetch_cp_weather.py`, `guider_utils.py`,
`test_galsim_centroid_convention.py` and the seven notebooks.

**`guider_psfmoments_<night>.parquet` is NOT in this product.** It is the science-CCD side
of the guider-vs-CCD comparison (`docs/guider_vs_ccd_moments.md`), read from
`preliminary_visit_image` stamps rather than from guider data, and the plan's product table
names only the three guider tables. Making it a product is a separate decision, not a 3a
move.

**`guiderEdgeRecovery.py` is a library by section 2**: `makeTrackerConfig` is imported by
the builder *and* by two study runners (`run_guider_movie.py`,
`run_guider_rubintv_plots.py`), and it writes nothing. It moves to `rubinwork/` with a
shim, the same call as `rubinwork.bounce`.

**`guiderMoments.py` goes into the product, not into `rubinwork/`**, even though one study
(`guider_atmo_compare.py`) imports `measureStampMoments`, `MomentConfig` and
`_FWHM_PER_SIGMA` from it. It carries `SCHEMA_VERSION`, which the plan's product table
names as the variant (`v6` is "the current `guiderMoments` schema"), so the schema and the
code that writes it stay together. The shim at the old path keeps the study working
unchanged — the `value_added` precedent, where eight shimmed builders left five bare-name
importers untouched.

### Readers of the product files, found with `rg` 2026-10-12

Few, and none outside `guider/`:

| reader | reads | kind |
|---|---|---|
| `guider/code/combine_moments.py` | the per-seq `*_{moments,stars,metrics}.parquet` | **product** (the combine step) |
| `guider/code/run_guider_dayplots.py` | `guider_moments_<night>.parquet`, as an argv path | study (rule `plots`) |
| `guider/guider_moments_ensemble.ipynb` | `output/night_*/guider_moments_*.parquet` by glob, plus the psfmoments glob | study |
| `guider/run_snake.sh` | names `guider_moments_<night>.parquet` as a target | pipeline driver |

Nothing in another topic reads them. The `guider` hits in `nightlyiq/guider_iq.ipynb`,
`blocks/camera_rotator_consdb_survey.ipynb` and `aos/snippets.ipynb` are ConsDB
`visit1_quicklook` `guider_*` columns, a different thing entirely. The eleven
`guider_atmo_*.py` scripts read `guider_atmo_moments.parquet` — **simulation** output, not
this product. `astrometry/astrometry_residual_field_v1.ipynb` names `moments_hsm.py` only
in a markdown provenance note; it does not import it.

### The HSM estimator becomes `rubinwork.hsm_moments`

`optatmo/code/moments_hsm.py` (157 lines, pure `numpy`/`galsim`, writes nothing) is
imported by three scripts in two topics, so it is a **library** by section 2:

| importer | how it reaches it today |
|---|---|
| `guider/code/run_guider_moments.py` | `sys.path.insert(__file__/../../optatmo/code)` inside `makeHsmFunc` |
| `guider/code/extract_adjacent_psf_moments.py` | the same relative insert |
| `optatmo/code/extract_psf_moments.py` | plain `sys.path` insert of its own directory |

It moves to `rubinwork/hsm_moments.py` with a shim at `optatmo/code/moments_hsm.py` giving
the same module object. This also removes the **relative-hop trap**, the import-side
version of the default-path trap `value_added` hit: `makeHsmFunc`'s
`__file__/../../optatmo/code` is exactly the kind of hop that breaks silently when a file
moves. `guider/CLAUDE.md`'s cross-topic coupling table is updated to say the coupling is
now a library import. The **`hsm_moments` data product stays deferred** — this moves the
library only.

### Variant and layout: `v6`, with nights as partitions

Per plan sections 3 and 5: an incremental product carries a schema version rather than
dated builds, so the variant is `v6` — `guiderMoments.SCHEMA_VERSION`, asserted by a test
so the two can never drift — and nights are partitions under
`products/guider_moments/v6/nights/<YYYYMMDD>/`, each with its own `manifest.json`
recording that night's provenance. There is no single `current` build.

Provenance per night, which this product has more of than `value_added` did: the builder
already stamps `summit_utils_version`, `schema_version` and the seven tracker/moment
settings into every row's `provenance`, so the night manifest records the same plus the
**`summit_utils` git commit** — `e57e0f4` on branch `u/aaronroodman/guider-fixes` as of
2026-10-12. `summit_utils` is **external and is not changed**.

### Step 1 plan: the reference build, and what may differ between two builds

Written **before** the reference build ran, so step 4 has a fixed list to check against.

The smallest build that exercises all three product rules is **one night limited to a few
exposures**, which is what `--limit` is for. `20260706` is a night already built in the old
tree, so the discovery query is known to resolve:

```bash
cd ~/notebooks/rubin-work/guider
REF=/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/guider_moments/ref
mkdir -p $REF/night_20260706/seq
python code/list_guider_exposures.py --day-obs 20260706 \
    --repo main --collections LSSTCam/raw/guider LSSTCam/raw/all \
    --image-types science --limit 3 --output $REF/night_20260706/exposures.txt
# then, for each discovered seqNum S:
python code/run_guider_moments.py --day-obs 20260706 --seq-num S \
    --outdir $REF/night_20260706/seq \
    --repo main --collections LSSTCam/raw/guider LSSTCam/raw/all \
    --view dvcs --guider-hz 5.0 --min-snr 10.0 --max-ellipticity 0.7 \
    --edge-margin 3 --min-finite-fraction 0.5 \
    --ap-nsigma 4.0 --cen-niter 3 --residual-scale 10.0
python code/combine_moments.py --seqdir $REF/night_20260706/seq \
    --outdir $REF/night_20260706 --dayobs 20260706
```

Every flag value is `snake_config.yaml`'s, so the reference is the pipeline's own
configuration rather than argparse defaults.

**Expected differences between two builds — the fixed list.** Unlike `value_added`, this
product stamps **no wall-clock timestamp at all**: the per-seq writers record only the
provenance dict and the tables, and `combine_moments.py` concatenates. So the expectation
is **zero differing columns in every table**, with two caveats that are about the
comparison rather than about build-time variation:

| table | caveat |
|---|---|
| `*_stars` | `timestamp` is converted from astropy `Time` to MJD floats by `parquetSafe`; it is exposure time, not build time, so it must match exactly |
| all three | `provenance` carries `summit_utils_version`, which is a package version rather than a clock, so it must match between two runs in one environment |

Any difference at all is therefore unexpected and stops the step.

Comparison keys, to verify unique on the reference before comparing anything:

| table | key |
|---|---|
| `<seq>_moments` / `guider_moments_<night>` | `(expId, detector)` |
| `<seq>_stars` / `guider_stars_<night>` | to verify — likely `(expId, detector, starId, stamp)` |
| `<seq>_metrics` / `guider_metrics_<night>` | to verify — likely `(expId, detector)` |

The `stars` and `metrics` keys are **summit_utils**' tables, not this repository's, so they
are read off the reference rather than assumed — the `bounce_kj_stats` lesson, where the
plan-shaped guess at a key left only 756 of 1,890 rows distinct.

### 3a step 1 — the reference build: DONE

Built at commit `e11e731` (clean tree), interactive on `sdfiana029` so the Butler
resolved. `--limit 3` on `day_obs` 20260706 discovered `seq_num` 44, 45 and 46.

| build | wall clock | result |
|---|---|---|
| `list_guider_exposures.py`, one night, `--limit 3` | **5 s** | 3 guider science exposures |
| `run_guider_moments.py`, 3 exposures | **~95 s** (about 32 s per exposure) | moments 5/4/5 rows, stars 664/529/642 rows, metrics 1/1/1 rows per exposure |
| `combine_moments.py` | **2 s** | moments 14, stars 1,835, metrics 3 rows; 0 skipped visits |

The night's 3 visits give a mean of 4.67 guiders per visit (dimensionless), all 3 partial
(fewer than 8 guiders), which the combine step reports — the same shape the full night
shows, so the small build exercises the partial-night path rather than only the happy one.

**Table shapes:** `guider_moments` 14 rows x 169 columns, `guider_stars` 1,835 rows x 52
columns, `guider_metrics` 3 rows x 50 columns (counts, dimensionless).

**Keys verified unique on the reference**, which is what the comparison needs:

| table | key | duplicate rows |
|---|---|---|
| `guider_moments` | `(expId, detector)` | 0 of 14 |
| `guider_stars` | `(expid, detector, trackid)` | 0 of 1,835 |
| `guider_metrics` | `(expId)` | 0 of 3 |

Note the **case split**: this repository's moments table uses `expId` and summit_utils'
stars table uses `expid`. Both are real and neither is renamed by the move.

**Two things found in step 1, recorded:**

1. **`combine_moments.py` writes an undeclared output.** It writes
   `detector_counts_<night>.txt` beside the three parquet files and
   `skipped_visits.txt`, and the `combine` rule declares only the latter four. Same class
   as `dof_lut`'s `lut_by_rotbin.parquet`. It is unconditional — the combine always writes
   it — so the product rule declares all five.
2. **`guider_metrics` columns are a union across exposures.** The per-detector columns
   (`R00_SG0`, ..., `R44_SG0`) exist only for the detectors a given exposure tracked, so
   `R44_SG0` appears last in the combined table because only `seq_num` 46 has it. The
   comparison matches columns **by name**, so column order carries no meaning and is not
   compared.

### Baseline `snakemake -n` job counts, before the move (2026-10-12)

| target | jobs | breakdown |
|---|---|---|
| no target (`rule all`, `nights: []`) | **1** | `all` only — the config lists no nights |
| `output/night_20260706/guider_moments_20260706.parquet` | **899** | 898 `moments` + 1 `combine` |

The 898 is a code-change provenance trigger over the night's 899 existing per-seq files,
not new work. The number to hold constant across the move is the **job table**, rule by
rule, not the total: the three product rules leave `guider/Snakefile`, so a per-night
*moments* target will no longer plan in `guider/` at all, and that change is the move
rather than a defect. Both counts are re-measured after the move and every change
explained rule by rule.

### 3a step 2 — the move: DONE (commit `8b488c9`)

What moved, and what each thing became:

| from | to | shim at the old path |
|---|---|---|
| `guider/code/run_guider_moments.py` | `rubinwork/products/guider_moments/builders/` | yes |
| `guider/code/combine_moments.py` | the same | yes |
| `guider/code/list_guider_exposures.py` | the same | yes |
| `guider/code/guiderMoments.py` | `rubinwork/products/guider_moments/` | yes |
| `guider/code/guiderEdgeRecovery.py` | `rubinwork/guider_edge_recovery.py` (**library**) | yes |
| `optatmo/code/moments_hsm.py` | `rubinwork/hsm_moments.py` (**library**) | yes |

New in the product package: `__init__.py`, `reader.py`, `variants.yaml`,
`write_manifest.py`, `compare_builds.py`, `Snakefile`, `builders/__init__.py` and
`tests/test_guider_moments.py`.

**Every shim registers the real module**, `sys.modules[__name__] = _real`, and the three
builder shims also keep `python guider/code/<name>.py ...` working through
`_real.main()`. Module identity is asserted in a **fresh interpreter per shim** — six of
them — for the same reason `value_added` needed it.

**`guider/Snakefile` keeps the study rules and resolves the product.** `momentsDir(night)`
returns the old-tree `output/night_<night>/` when `guider_moments_<night>.parquet` is
there, else the catalog's `v6/nights/<night>/` partition, else the old-tree path so a
rule's own gate reports the missing file as before. An `onstart` handler prints which
source each night resolved to, as `aos/Snakefile` does.

Three things the move settled that the earlier products did not meet:

1. **The study rules lost their fan-out source.** `movie`/`movies` and
   `rtvplot`/`rtvplots` fanned out from the `discover_night` **checkpoint**, which is a
   product rule and left with it. They now read the product's own `exposures.txt` through
   the resolver (`nightSeqNums`) rather than re-querying the Butler — a second Butler
   query would have been a second definition of "the night's guider exposures", and could
   disagree with the one the product built from.
2. **`run_snake.sh` had to become two runs, not one.** A whole night is the product build
   (from the **repository root**) followed by the study rules (from `guider/`), so
   `--day-obs N` now runs the product Snakefile for each night and then the study targets,
   in order, inside the same local run or the same sbatch job. The per-night target list
   lost `guider_moments_<night>.parquet`, which has no producer in `guider/` any more.
3. **The product Snakefile takes the night from `--config`, not from a wildcard.** A
   `{night}` wildcard made `rule combine` raise `RuleException`: `skipped_visits.txt` and
   the rule's log carry no wildcard, and snakemake requires every output and log of a rule
   to carry the same ones. `seqNum` is the only wildcard; a test asserts that.

**The relative-hop trap, found and removed.** `makeHsmFunc` did
`sys.path.insert(__file__/../../optatmo/code)` — the import-side version of the
default-path trap `value_added` hit, and it would have resolved to
`rubinwork/products/optatmo/code` after the move, silently returning `None` and dropping
every `hsm_*` column with no error. The library import removes the hop; a test asserts the
string `optatmo` no longer appears in the builder.

**Two `optatmo/code/extract_psf_moments.py` imports became unused** (`os`, `sys`) and were
removed. Checked against the pre-move state with `git stash` + `pyflakes` first, as the
`bounce_tables` split taught: the four pyflakes warnings in
`guider/code/extract_adjacent_psf_moments.py` are **pre-existing** and were left alone.

**One shared-code change outside the product**: `catalog.list_products` descends one level
into a partition layer, so a directory under a variant that carries no manifest but whose
children do — the `nights/` layer — contributes one entry per child with ``build`` as
`"nights/<night>"`, which is what `catalog.path` takes. Without it an incremental
product's nights were invisible to `list()` and so to the generated `PRODUCTS.md`.

### 3a step 3 — verification: DONE

| check | result |
|---|---|
| import smoke test | **59 of 59 modules import**, and the inventory is **unchanged** — `--compare` reports "all 59 module import outcomes identical" |
| product tests | **350 pass** (315 before this product + 35 new); no pre-existing test needed changing |
| `pyflakes` on every new and moved file | clean |
| `snakemake -n` in `guider/`, no target | **1 job** (`all`), unchanged — `nights: []` in the config |
| `snakemake -n` in `guider/`, the moments target | `MissingRuleException`, **as intended**: that file is the product's now |
| product Snakefile dry run | 4 jobs — `discover_night`, `combine`, `manifest`, `all` — plus the per-exposure fan-out the checkpoint adds |
| `run_snake.sh --day-obs 20260706 --limit 3 -n` | runs both: 4 product jobs then 20 study jobs |
| `snakemake -n` in `aos/` | **unchanged at 34 jobs** — nothing in `aos/` was touched, and the one shared-code change (`catalog.list_products`) is not on that Snakefile's path |

**The `guider/` job-count change, rule by rule.** For the five per-night targets a
pre-move run planned **6 jobs**: `discover_night` 1, `combine` 1, `psfmoments` 1,
`movies` 1, `rtvplots` 1, `plots` 1 (the `moments` fan-out appears only after the
checkpoint resolves). After the move the same work is **4 product jobs + 20 study jobs**:

* `discover_night` and `combine` moved to the product, as did the `moments` fan-out: −2
  planned jobs in `guider/`, and the product's own DAG gains them plus `manifest` and
  `all`.
* `plots` and `movies` are **absent** from the post-move study count because both outputs
  already exist for this night and are up to date — not because they were lost. `plots`
  now takes its input through the resolver.
* `rtvplot` went from **0 planned to 18**, with `rtvplots` still 1. Same total work: the
  18 were previously hidden behind the checkpoint and appeared only after it resolved,
  whereas `nightSeqNums` reads `exposures.txt` directly and so plans them at DAG time.
  18 = 899 discovered exposures strided by `movie_stride = 50`.

### 3a step 4 — the comparison: DONE, exact match

The reference built with the unmoved code, reproduced by running the **product Snakefile**
end to end into a second scratch root, at zero tolerance on the keys verified unique on
the reference:

| table | rows | columns compared | result |
|---|---|---|---|
| `guider_moments` | 14 | **169** | **identical** |
| `guider_stars` | 1,835 | **52** | **identical** |
| `guider_metrics` | 3 | **50** | **identical** |

271 columns compared in total (count, dimensionless), **0 differing**, and the expected
difference list was **empty before the build ran** — this product writes no build-time
timestamp anywhere, so there was nothing a second run could legitimately change.

Also identical: both text sidecars **byte for byte** (`skipped_visits.txt` and
`detector_counts_20260706.txt`), `exposures.txt` byte for byte — so the moved discovery
query returns the same three exposures — and **all 9 per-exposure files** (3 exposures x 3
tables), compared individually on the same keys.

**No manifest was written anywhere under `_phase2_refbuild/`**: a `find` over the whole
scratch tree returns nothing. `write_manifest.is_catalog_build` decides by path, so the
`manifest` rule fails by design on a scratch `night_root`, and the reference build was run
up to `combine`.

## 3a, `value_added` — the move

### Step 1 plan: the exact commands, and what may differ between two builds

Written **before** the reference build ran, so step 4 has a fixed list to check against.
Nothing outside the expected-difference list below may differ.

**Flags taken from `run_build.sh`'s own invocations**, not its usage comment, which is
what the handoff's previous entry warned about. Five things the comment does not tell you:

1. `run_build.sh` passes **`--db <path>`**, not `AOS_EFD_DB`. Both work; `--db` is what a
   shard gets. This session sets `AOS_EFD_DB` as well, as a second guard against ever
   touching the live database.
2. **A state shard needs `--telemetry-db`.** Its own database's `visit_telemetry` is
   empty, so without it the commanded v-modes and the whole open-loop arm come back NaN,
   and the builder refuses an empty telemetry source. `run_build.sh` points it at the
   **main** database; for a scratch build it must point at the **scratch telemetry
   database**, or the reference reads live data.
3. **The MIW route needs `--miw-param-set`**, because `state_variant` has no column for
   it and the module default does not exist on disk. `run_build.sh` errors out without it.
4. `run_build.sh` resolves a variant's defining flags (`--scheme`, `--intrinsic`,
   `--opd-version`, `--intrinsic-ref`, `--ofc-version`) **out of the main database's
   `state_variant`** and passes those instead of `--variant`, because a shard's own
   registry starts empty. A scratch build must do the same, or the builder exits with
   "variant is not registered".
5. **`build_fam_dz.py` uses `--db-path`**, not `--db`.

```bash
REF=/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/value_added/ref
cd ~/notebooks/rubin-work/value_added

# 1. telemetry, both nights, all groups
AOS_EFD_DB=$REF/aos_efd.duckdb python -u code/build_efd_db.py \
    --day-obs 20251116,20251117 --groups all --db $REF/aos_efd.duckdb

# 2. the M1M3 bulk thermal gradients, same nights
AOS_EFD_DB=$REF/aos_efd.duckdb python -u code/build_m1m3_thermal_r2.py \
    --day-obs 20251116,20251117 --db $REF/aos_efd.duckdb

# 3. optical_state, the MIW route -- explicit --intrinsic-ref AND --miw-param-set
AOS_EFD_DB=$REF/aos_efd.duckdb python -u code/build_optical_state.py \
    --day-obs 20251116,20251117 --scheme 50_34 --intrinsic miw \
    --opd-version consdb_v1 --intrinsic-ref danish_1_2_A_50_34_i_5rot \
    --ofc-version v13 --miw-param-set danish_1_2_A_50_34_i_5rot \
    --db $REF/aos_efd.duckdb --telemetry-db $REF/aos_efd.duckdb

# 4. optical_state, the batoid route
AOS_EFD_DB=$REF/aos_efd.duckdb python -u code/build_optical_state.py \
    --day-obs 20251116,20251117 --scheme 50_34 --intrinsic batoid \
    --opd-version consdb_v1 --intrinsic-ref ofc_v13 --ofc-version v13 \
    --db $REF/aos_efd.duckdb --telemetry-db $REF/aos_efd.duckdb

# 5. fam_dz, the one registered fam_variant
AOS_EFD_DB=$REF/aos_efd.duckdb python -u code/build_fam_dz.py \
    --param-set danish_1_2 --aos-output ../aos/output/fam_processing \
    --prefix z1toz6 --scheme 50_34 --db-path $REF/aos_efd.duckdb
```

**`build_fam_dz.py` cannot run on its registered variant today** — a pre-existing defect
of the same class as `miw_corner.decomp_path` and the three coadd scripts.
`resolve_paths` (`build_fam_dz.py:104`-`:106`) builds
`<aos_output>/<param_set>/fits.parquet` from the **long** param_set key, and
`fam_variant.fits_path` records
`aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/fits.parquet`. No such directory
exists: the tables are at `aos/output/fam_processing/danish_1_2/`, the `dir_name` layout.
So the reference build passes `--aos-output ../aos/output/fam_processing --param-set
danish_1_2`, which resolves. **Recorded, not fixed** — a path fix is a 3b item, and 3a is
mechanical. It does mean the `fam_dz` reference exercises the builder but not the
registered variant's own `fits_path`.

**The merge path, exercised as two shards merged into a second scratch database:**

```bash
SHARDS=/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/value_added/ref_shards
MERGED=/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/value_added/ref_merged
cd ~/notebooks/rubin-work/value_added
for N in 20251116 20251117; do
  AOS_EFD_DB=$SHARDS/telemetry_$N.duckdb python -u code/build_efd_db.py \
      --day-obs $N --groups all --db $SHARDS/telemetry_$N.duckdb
done
AOS_EFD_DB=$MERGED/aos_efd.duckdb python -u code/merge_db_shards.py \
    --shards $SHARDS/telemetry_20251116.duckdb $SHARDS/telemetry_20251117.duckdb \
    --db $MERGED/aos_efd.duckdb
```

### Expected differences between two builds — the fixed list for step 4

**Only these may differ.** Every one is a wall-clock timestamp written at build time, so
two runs of identical code necessarily disagree; there is no other source of
run-to-run variation in this product (no row order dependence — every table has a
verified-unique key — and no randomness in any builder).

| table | column | why it differs |
|---|---|---|
| `fetch_log` | `attempted_at` | `datetime.now(timezone.utc)` at each group fetch (`efd_db.py:538`) |
| `column_coverage` | `updated_at` | `datetime.now(timezone.utc)` per coverage refresh (`efd_db.py:573`) |
| `state_variant` | `created_at` | `datetime.now(timezone.utc)` at registration (`efd_db.py:638`) |
| `fam_variant` | `created_at` | `datetime.now(timezone.utc)` at registration (`efd_db.py:859`) |
| `optical_state` | `computed_at` | `datetime.now(timezone.utc)` per row batch (`efd_db.py:766`) |
| `fam_dz` | `computed_at` | `datetime.now(timezone.utc)` per row batch (`efd_db.py:932`) |
| `m1m3_thermal_r2` | `computed_at` | `pd.Timestamp.utcnow()` per row batch (`efd_db.py:1042`) |

`visit_telemetry` carries **no** timestamp column, so it must match exactly, column for
column. The two `optical_state` variants and `fam_dz` must match on every value column;
only `computed_at` may move.

Comparison keys, to verify unique on the reference before comparing anything:

| table | key |
|---|---|
| `visit_telemetry` | `(visit_id)` |
| `optical_state` | `(visit_id, variant_id)` |
| `m1m3_thermal_r2` | `(visit_id)` |
| `fam_dz` | `(visit_id, fam_variant_id)` |
| `column_coverage` | `(column_name)` |
| `state_variant` | `(variant_id)` |
| `fam_variant` | `(fam_variant_id)` |
| `fetch_log` | `(day_obs, group_name)` |

### 3a step 1 — the reference build: DONE, and two builders cannot run as registered

Built at commit `fb14ab5` (clean tree), interactive on `sdfiana029` so the EFD resolved.
Wall clock per builder:

| build | wall clock | result |
|---|---|---|
| telemetry, both nights, all 9 groups | **171 s** | 1,017 exposures, 1,017 `visit_telemetry` rows, 196 `column_coverage` columns |
| `m1m3_thermal_r2`, both nights | **73 s** | 1,017 rows, 1,017 with a fit; 1,573 and 1,558 time bins at 30 s |
| `optical_state`, MIW route | **50 s** | 1,017 rows, 839 recovered, 1,012 with commanded v-modes |
| `optical_state`, batoid route | **45 s** | 1,017 rows, 839 recovered, same visits |
| `fam_dz`, one variant | **38 s** | 2,528 FAM pairs, 0 with a NaN DZ coefficient, `U_eff = (126, 34)` |
| two shards + merge | **190 s** | 535 + 482 rows merged to 1,017; 18 `fetch_log` rows |

**The reference reproduces the production numbers**, which is what shows it is the real
measurement and not a differently-configured run: on 20251116 the MIW route gives a median
CWFS FWHM contribution of **0.2546 arcsec** over 460 visits, exactly the figure in
`logs/batch_state_20261006_001235_04.out` from the production build.

**Two builders could not be run as their registry rows describe them.** Both are
pre-existing defects of the same class — a path built from the **long** `param_set` key,
a layout the tree lost when `dir_name` arrived — and both are **recorded, not fixed**,
because 3a is mechanical and a path fix is a 3b item:

1. **`build_fam_dz.py`**: `resolve_paths` (`:104`-`:106`) builds
   `<aos_output>/<param_set>/fits.parquet`, and `fam_variant.fits_path` records
   `aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/fits.parquet`. **No such
   directory exists** — the tables are at `aos/output/fam_processing/danish_1_2/`. Worked
   round with `--aos-output ../aos/output/fam_processing --param-set danish_1_2`, which
   resolves; the consequence is that the reference registers
   `fam__danish_1_2__batoid__z1toz6__50_34` rather than the live
   `fam__fam_danish_1_2_0_..._bin2x__batoid__z1toz6__50_34`. Both sides of the comparison
   do the same, so the comparison is unaffected.
2. **`build_optical_state.py --intrinsic miw`** — **not pre-existing: a regression of the
   reorganization, and FIXED 2026-10-12** (see "The MIW route regression" below). It
   passes `intrinsic_ref` as `MiwCornerLookup`'s `mi_name` (`:770`-`:772`), and the
   **`miw` 3a move (`b9756aa`) made `miw_corner.decomp_path`** take the long
   `(param_set, mi_name)` keys and join their `dir_name`s, so the joined directory name
   `state_variant.intrinsic_ref` holds — `danish_1_2_A_50_34_i_5rot` — raised
   `KeyError: MI config 'danish_1_2_A_50_34_i_5rot' not found`. The reference build worked
   round it by passing the long keys (`--intrinsic-ref pathA_50_34_i_5rot --miw-param-set
   fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`), which resolve to the same decomposition
   file. The earlier text here and in `value_added/docs/status/build_progress.md` called
   it pre-existing and deferred it to 3b; both are corrected.

A third, smaller one: **`build_m1m3_thermal_r2.py` has its own `parse_day_obs`** that
takes a single night or a `A-B` range but **not** a comma list, unlike `build_efd_db`'s.
`--day-obs 20251116,20251117` fails with `ValueError: invalid literal for int()`. Not a
defect exactly — just an inconsistency between two builders in the same product worth
knowing before writing a command.

### 3a step 3 — verification: DONE

**Import smoke test: 59 of 59 import, and the 63 → 59 inventory change is accounted for
file by file.** The inventory is discovered dynamically — "every `.py` importing a moved
library" — so it shrinks when a file stops importing one:

| direction | files | why |
|---|---|---|
| left (7) | the `value_added/code/` originals | their shims import `rubinwork.products.value_added...`, which is not a `MOVED_NAMES` library, so they no longer match the pattern |
| entered (3) | `builders/{build_fam_dz,build_optical_state,test_build_optical_state}.py` | the moved copies still import `aos_state` / `regularized_inversion` |

The other four moved files (`build_efd_db`, `build_m1m3_thermal_r2`, the reader,
`backfill_optical_state_pointing`) import `rubinwork.common` rather than bare `common`
now, so they correctly fall out of a "imports a moved library" inventory. **No module
fails to import, before or after.**

**Tests: 312 passed** across `rubinwork/products/` — the 44 new `value_added` tests plus
all 268 from before.

**Every bare-name importer and all three notebooks resolve the shim to the SAME module
object**, checked from a fresh interpreter each:

| caller | result |
|---|---|
| `thermal_focus/code/run_thermal_focus.py` | same module as `reader` |
| `wfs/code/giant_donuts/select_exposures.py` | same module |
| `aos/code/cwfs_lut/run_cwfs_lut.py` | same module |
| `thermal_focus/code/thermal_focus_lib.py` | imports cleanly; it only *prepares the path* for its sibling and never imports `efd_db` itself |
| `aos/code/cwfs_lut/cwfs_lut_lib.py` | same, path-only |
| the three notebooks' idiom | same module, `visits` callable, default path = the live database |

The two "path-only" files were worth checking rather than assuming: both carry a
`sys.path.insert` of `value_added/code` with a `-> efd_db` comment, but neither imports it
— the inserts exist for `run_thermal_focus.py` and `run_cwfs_lut.py`, which do.

**`snakemake -n` in `aos/`: 34 jobs, unchanged, rule for rule.** `value_added` has no rule
there, which is why this is the first move with no job-count change to explain — and it
was measured rather than assumed, per the constraint the `bounce_tables` split added.

**`run_build.sh --dry-run`: the same shard plan and the same arguments, only the script
path differs.** Diffed against the pre-move script (`fb14ab5`) run from the same working
directory, with the shard timestamp normalised:

```diff
-  cmd: python -u code/build_efd_db.py --day-obs 20251116 --db output/shards/telemetry_<TS>_01.duckdb --groups all
+  cmd: python -u -m rubinwork.products.value_added.builders.build_efd_db --day-obs 20251116 --db output/shards/telemetry_<TS>_01.duckdb --groups all
```

Night expansion, shard count, shard database paths and log paths are byte-identical. The
`--what state` path was diffed too, and its variant-flag resolution out of the main
database's `state_variant` still produces exactly
`--scheme 50_34 --intrinsic batoid --opd-version consdb_v1 --intrinsic-ref ofc_v13
--ofc-version v13 --telemetry-db <main db>` — so the moved script still reads the registry
correctly, which is the part of that script most likely to have broken.

### 3a step 4 — the comparison: DONE, exact match

Every table and registry, at zero tolerance, on the keys verified unique on both sides:

| table | rows | columns compared | result |
|---|---|---|---|
| `visit_telemetry` | 1,017 | **196** | **identical** — no timestamp column, so nothing may differ |
| `optical_state` | 2,034 | 13 | identical; `computed_at` differs as expected |
| `m1m3_thermal_r2` | 1,017 | 27 | identical; `computed_at` differs as expected |
| `fam_dz` | 2,528 | 14 | identical; `computed_at` differs as expected |
| `column_coverage` | 196 | 7 | identical; `updated_at` differs as expected |
| `state_variant` | 2 | 10 | identical; `created_at` differs as expected |
| `fam_variant` | 1 | 13 | identical; `created_at` differs as expected |
| `fetch_log` | 18 | 4 | identical; `attempted_at` differs as expected |

**Nothing outside the pre-registered expected-difference list differs**, and that list was
committed in `fb14ab5` before the reference build ran, so it could not be widened after
seeing a result. `RESULT: builds match`, exit 0.

The **merge path matches too**: two shards built separately and merged give 1,017
`visit_telemetry` rows and 18 `fetch_log` rows, identical on both sides, and each
individual shard compares clean as well.

**No manifest was written anywhere under `_phase2_refbuild/`**, confirming the
scratch-versus-catalog rule: `is_catalog_build` decides by path, so a build whose database
comes from `--db` or `AOS_EFD_DB` outside the catalog build directory cannot be
registered. A `find` for `manifest.json` over the whole scratch tree returns nothing.

The comparison needed two things the parquet products' did not:

* **List-valued columns.** `optical_state` and `fam_dz` store the 50 DOF and the 34
  v-modes as DuckDB `LIST`, which pandas holds as object arrays that `!=` cannot compare
  elementwise, so `_values_equal` compares them shape-first then elementwise.
* **NaN equal to NaN.** An unrecovered visit stores NaN rather than NULL (178 of the 1,017
  visits here), so a plain equality test would have reported every one of them as a
  difference between two identical builds.

## The MIW route regression: fixed 2026-10-12

**What it was.** `value_added`'s `optical_state` MIW route could not run. The builder
passes `state_variant.intrinsic_ref` — a **joined build directory name** such as
`danish_1_2_A_50_34_i_5rot` — as `MiwCornerLookup`'s `mi_name`, and the `miw` 3a move
(`b9756aa`) had redefined `decomp_path` to take the long `(param_set, mi_name)` keys and
join their `dir_name`s. A joined name is not an `mi_config.yaml` entry, so the lookup
raised `KeyError: MI config 'danish_1_2_A_50_34_i_5rot' not found`. Broken 2026-10-10 to
2026-10-12.

**Why it was fixed now rather than in 3b.** The reorganization caused it, so restoring it
is part of 3a's "no change in results" contract, not a 3b improvement. The earlier entry
called it pre-existing, which was wrong.

**The fix.** `decomp_path` (`rubinwork/miw_corner.py`) now resolves a `mi_name` that is
itself a build directory under `aos/output/miw/` directly to that directory's
`intrinsic_split_decomp.parquet`, and falls through to the long-key route otherwise.
A **directory** of that name settles it rather than the file existing: the long-key route
would raise `KeyError` for a joined name, which would hide an incomplete build behind the
wrong error, where this way the missing-file `FileNotFoundError` names the path. The
builder was not changed. Three tests: the joined form on a `tmp_path` tree with a
deliberately invalid `param_set`, both forms giving the same file on the real tree, and an
unknown name still raising `KeyError` on the long keys.

**Verification.** The step-1 MIW-route reference command rerun exactly as `run_build.sh`
issues it — `--intrinsic-ref danish_1_2_A_50_34_i_5rot`, with `--miw-param-set danish_1_2`,
which is deliberately **not** a valid long `param_set` key, so the run only succeeds if
the joined name resolves on its own — into a new scratch database
(`_phase2_refbuild/value_added/miwfix/`). It read
`aos/output/miw/danish_1_2_A_50_34_i_5rot/intrinsic_split_decomp.parquet`, the same file
the reference read, and reproduced the reference's numbers: 1,017 visits, 839 recovered,
1,012 with commanded v-modes, and a median CWFS FWHM contribution of **0.2546 arcsec over
460 visits** on `day_obs` 20251116.

`optical_state` for `variant_id = v50_34__miw__consdb_v1` compares at zero tolerance on
the verified-unique key `(visit_id, variant_id)`: 1,017 rows on both sides, **13 value
columns compared, 0 unexpected differences**, only `computed_at` differing, which is on the
list committed in `fb14ab5`. `state_variant.intrinsic_ref` differs by construction —
`pathA_50_34_i_5rot` in the reference's long-key workaround against the joined name here —
which is the point of the fix rather than a difference in results.

315 product tests pass (312 + the 3 new); import smoke test unchanged at 59/59 modules.

## 3a, `cwfs_tables` — the move

### Scope, agreed 2026-10-10

Moves: `aos/code/cwfs/run_wfs_mktable.py` only, to
`rubinwork/products/cwfs_tables/builders/`, with a shim at the old path. Snakefile rule
`wfs_mktable` only. The other five `aos/code/cwfs/` scripts (`corner_compare`,
`dof_compare`, `mimic`, `refit_ensemble`, `fam_compare`) are the CWFS-vs-FAM **study** and
stay; `wfs_intrinsic_sidecar` belongs to the `miw` product and is left alone.

Everything that reads the CWFS tables keeps reading the **old tree**
(`aos/output/wfs_ingest/<P>/<cwfs>/`) until phase 5. Each consumer reaches them through a
`--wfs-dir` argument or a literal old-tree path, and none of them imports
`run_wfs_mktable`, so the move cannot break them: `rule all`, `wfs_corner_compare`,
`wfs_dof_compare` and `wfs_intrinsic_sidecar` in `aos/Snakefile`; `run_wfs_fam_compare.py`;
`run_wfs_corner_compare.py` and `run_wfs_dof_compare.py` (both via `--wfs-dir`);
`aos/code/infra/migrate_output_layout.py`; `aos/snippets.ipynb`. As in phase 2, the three
table targets become inputs `aos/Snakefile` no longer builds.

### Variants: six entries, `aos/param_sets.yaml` stays byte-identical

`cwfs_tables/variants.yaml` owns them, each naming its `fam_variant` and carrying the
`collection`, `dataset_type`, `reader` and `seq_offset` fields that were in
`fam_tables/variants.yaml` `wfs_collections`. `fam_tables/variants.yaml` drops
`wfs_collections`; `gen_param_sets.py` merges both files back into `aos/param_sets.yaml`.

| variant | fam_variant | registered | fields beyond `collection` |
|---|---|---|---|
| `d12-refitWcs` | `danish_1_2` | true | — |
| `d12-refitWcs_2025` | `danish_1_2` | true | — |
| `d12-paired_3mm` | `danish_1_2` | true | `seq_offset: 0` |
| `d12-ai_donut` | `danish_1_2` | true | `dataset_type: aggregateZernikesRaw` |
| `d12-tarts` | `danish_1_2` | true | `reader: unpaired`, `dataset_type: aggregateAOSVisitTableAvg` |
| `d10-wep17_3_0` | `danish_1_0` | **false** | provenance only; never built |

`aos/param_sets.yaml` is md5 `0f72da651cf9319cc283754c614051fd` before the move and must
stay so — that is the acceptance test, checked with `--check` and a byte `diff`.

**Two decisions deferred to 3b**, both Aaron's calls on 2026-10-10:

1. **`d12-refitWcs_2025` folds into `d12-refitWcs`.** They are the same variant; the 2025
   visits just need a different collection, exactly as the FAM `danish_1_2` carries a
   per-chunk collection override for its 2025 chunks. The two CWFS collections are
   disjoint in `day_obs` (`refitWcs` 20260315..20260513, `refitWcs_2025`
   20250415..20251231), so a date-keyed collection list is the shape. Deferred because
   `run_wfs_mktable` opens **one** `Butler(repo, collections=...)` per `--wfs-name`, so
   merging them is a builder change, not an imports-and-paths move; 3a stays mechanical.
2. **`danish_1_0` is retired whole, both halves.** Dropping only the CWFS half would make
   the generated `aos/param_sets.yaml` lose two lines, and dropping the FAM half in 3a
   would break four live things: the staged-MIW provenance chain
   (`aos/calibration/miw/intrinsic_split_maps_v1.provenance.yaml`, `stage_miw.py`,
   `calibration/README.md`), the `aos-measured-intrinsics` tech note's
   `make_figures.py:23` and `provenance.md:5`, a **live** `overrides:` block at
   `aos/analysis_config.yaml:148` keyed on the long param_set, and
   `test_fam_tables.py:263`. That is a `fam_tables`/`miw` change with its own before/after
   check. So `d10-wep17_3_0` is carried as `registered: false` for now.

### 3a step 1 — reference build with the unmoved code: DONE

Built at commit `9392f9b` (clean tree), interactive on `sdfiana`, **65.3 s wall clock**.

Variant `refitWcs_2025` under param_set `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`, one
night, `day_obs = 20251126`. That variant has **no old-tree build**, so the reference
cannot touch one.

Input: one night of the phase-2 FAM reference, staged by
`rubinwork/products/cwfs_tables/refbuild/stage_fam_night.py` because `run_wfs_mktable` has
no `--day-obs` filter and would otherwise walk all 53 visits of the chunk:

```bash
python -m rubinwork.products.cwfs_tables.refbuild.stage_fam_night \
    --reference /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_2 \
    --day-obs 20251126 \
    --out /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/cwfs_tables/_fam_20251126
```

Staged FAM slice: `visits.parquet` 6 rows x 600 columns, `fits.parquet` 5 rows x 653
columns, `donuts.parquet` 17,019 rows x 48 columns (all counts dimensionless).

**Night choice matters.** 20251116, the chunk's first night, has 6 FAM visits but **0**
`fits.parquet` rows — the DZ fit's `median_blur_arcsec <= 1.2 arcsec` quality cut drops all
of them — which makes the validation plot's FAM k=1 overlay all-NaN and the reference
degenerate. 20251126 is the smallest night that does not. The staging script warns on an
empty `fits.parquet`.

The build command, run from `aos/` (the unmoved script's own path):

```bash
cd ~/notebooks/rubin-work/aos && python code/cwfs/run_wfs_mktable.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --wfs-name refitWcs_2025 \
    --coord-sys OCS \
    --tables-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/cwfs_tables/_fam_20251126 \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/cwfs_tables/refitWcs_2025_20251126
```

Output, in `_phase2_refbuild/cwfs_tables/refitWcs_2025_20251126/`:

| file | rows | columns | bytes |
|---|---|---|---|
| `donuts.parquet` | 171 | 20 | 95,409 |
| `visits.parquet` | 6 | 8 | 6,000 |
| `wfs_mktable_validation.pdf` | — | — | 106,605 |

All 6 FAM visits found an in-focus CWFS table (0 missing). Per-visit donut counts
(count, dimensionless): 28, 28, 26, 36, 21, 32, band `i` throughout.
`donuts.meta['nollIndices']` is the 21-term Noll set
`[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26]`.

**Comparison key, verified unique.** `run_wfs_mktable` writes **no donut-id column**, so
the `fam_tables` key `(day_obs, seq_num, detector, extra_donut_id)` does not exist here.
The CWFS key is **`(day_obs, seq_num, detector, thx_OCS, thy_OCS)`**, where `thx/thy_OCS`
are field angles in radians (OCS). Uniqueness, checked on the reference and on all three
built old-tree variants (unique rows of total rows, counts dimensionless):

| table | `day_obs, seq_num, detector` | plus `thx_OCS, thy_OCS` |
|---|---|---|
| reference `refitWcs_2025` (171 rows) | — | **171 / 171 unique** |
| old-tree `refitWcs` (34,631 rows) | 4,485 — not unique | **34,631 / 34,631 unique** |
| old-tree `paired_3mm` (1,223 rows) | 515 — not unique | **1,223 / 1,223 unique** |
| old-tree `tarts` (7,813 rows) | 7,813 — unique | **7,813 / 7,813 unique** |

The detector-only key fails on the paired variants (several donuts per corner sensor per
exposure). Adding the centroids also works, but `tarts` has no centroid columns, so the
field-angle key is the only one that covers every variant. `visits.parquet` keys on
`(day_obs, seq_num)`. The comparison asserts uniqueness so a future variant that breaks it
fails loudly instead of mis-aligning.

The validation PDF is **not** byte-comparable (matplotlib embeds a creation timestamp), so
step 4 compares the two parquet tables at zero tolerance and checks only that the PDF is
produced at a comparable size.

`rotator_angle` and `alt` in the CWFS tables are **copied from the FAM `visits.parquet`**,
not re-derived, so the A1 `rotTelPos` non-reproducibility cannot affect this comparison.

### 3a step 2 — the move: DONE (commit `b880f2d`)

`run_wfs_mktable.py` moved by `git mv` into
`rubinwork/products/cwfs_tables/builders/`, with a shim at
`aos/code/cwfs/run_wfs_mktable.py` that binds the real module's namespace and sets
`sys.modules[__name__] = _real`. Verified: a bare-name `import run_wfs_mktable` after a
`sys.path.insert` of `aos/code/cwfs` gives the **same module object**, and all seven
checked attributes (`main`, `get_aidonut_zernikes`, `get_unpaired_zernikes`, `_fam_noll`,
`_validation_plot`, `_wfs_shell`, `AIDONUT_SW0`) are the same objects, not copies.

New in the product: `__init__.py` (with `load`), `reader.py`, `variants.yaml`,
`Snakefile`, `write_manifest.py`, `compare_builds.py`, `tests/test_cwfs_tables.py` (26
tests) and `refbuild/stage_fam_night.py`.

**The one builder change, and why it is not a results change.** The builder gained a
`--variant` flag that resolves a `cwfs_tables` variant to its `(param_set, wfs_name)`
pair; the original `--param-set` plus `--wfs-name` route is untouched and still reads the
configuration through `intrinsics_lib.load_param_sets()`. Both routes end at the same
`pset['wfs_collections'][name]` lookup, so they cannot disagree — and step 4's comparison
is the unmoved `--param-set` route against the moved `--variant` route, which is the
stronger check.

Four things found or decided during the move:

1. **`wfs_mktable` must run with `aos/` as its cwd**, for the same reason `fam_tables`'
   `mktable` does: the external `intrinsics_lib.load_param_sets()` opens
   `Path('param_sets.yaml')` relative to the working directory. The product Snakefile does
   `cd <repo>/aos && python -m ...`.
2. **`reader` must be written before `dataset_type`** in a generated `wfs_collections`
   entry. `BUILDER_FIELDS` in `cwfs_tables/reader.py` fixes that order; with
   `dataset_type` first, `aos/param_sets.yaml` differed by the two swapped lines of the
   `tarts` entry and byte-identity failed.
3. **An entry with only a collection must be emitted as a bare string**, not a
   one-key dict, which is the shape `aos/param_sets.yaml` has today.
   `_builder_entry()` does that.
4. **`aos/Snakefile`'s `wfs_variants()` had to start filtering on what is on disk.** See
   step 3 — this is the one non-obvious consequence of the move.

The product Snakefile builds one variant and one build at a time, from the repository
root:

```bash
snakemake -s rubinwork/products/cwfs_tables/Snakefile -n \
    --config variant=d12-refitWcs build=20261010 fam_build=20260920
```

It takes `fam_build` (default `current`) and resolves it through the catalog, recording
the real build name under the manifest's `inputs` as
`{"fam_tables": "danish_1_2@20260920"}`. `manifest` depends on both tables **and** the
validation PDF, so an interrupted build leaves no manifest and the catalog does not see
it. A `registered: false` variant raises before any path resolves, naming the reason.

### 3a step 3 — verification: DONE

**`aos/param_sets.yaml` is byte-identical.** md5 `0f72da651cf9319cc283754c614051fd` before
and after; `gen_param_sets --check` passes, a byte `diff` against the pre-move copy is
empty, and `git diff 9392f9b -- aos/param_sets.yaml` is empty.

**`snakemake -n` in `aos/`: 199 -> 189 jobs, a drop of 10 — not 5.** The moved rule
accounts for 5; the other 5 are a real consequence that the next product's move will meet
again:

| rule | before | after | change | why |
|---|---|---|---|---|
| `wfs_mktable` | 5 | 0 | **-5** | moved to the product |
| `wfs_corner_compare` | 5 | 4 | -1 | `refitWcs_2025` lost its producer |
| `wfs_intrinsic_sidecar` | 10 | 8 | -2 | same |
| `wfs_dof_compare` | 10 | 8 | -2 | same |
| every other rule | | | **0** | unchanged |
| **total** | **199** | **189** | **-10** | |

All counts are jobs (dimensionless). **`refitWcs_2025` is the one CWFS variant with no
old-tree build** — `aos/output/wfs_ingest/danish_1_2/` holds `refitWcs`, `paired_3mm`,
`ai_donut` and `tarts`, but not it. Before the move, `wfs_mktable` was its producer, so
its four downstream study jobs were plannable. After the move there is no producer in
`aos/`, and a plain `snakemake -n` aborted the whole DAG with `MissingInputException`
rather than listing jobs.

So `wfs_variants()` now enumerates only the variants whose `donuts.parquet` exists. The
four built variants are completely unaffected — post-move they still plan 16, 16, 16 and
20 `wfs_corner_compare`/`wfs_dof_compare` output paths respectively — and zero
`refitWcs_2025` jobs remain. Building that variant with the product Snakefile makes its
study jobs reappear. **This is the step-3 criterion failing as literally written** ("the
same job list apart from the moved rule's jobs"), and it is reported rather than hidden:
the extra -5 is one never-built variant, enumerated exactly, not a side effect on anything
that was working.

**The product Snakefile plans the expected jobs**: 3 for one variant — `wfs_mktable` 1,
`manifest` 1, `all` 1 — verified against a staged `fam_tables` build in a scratch data
root (`_phase2_refbuild/_scratch_3a/`), never the real products tree. The `mktable`
command it issues is the moved builder with `--variant`, `--coord-sys OCS`,
`--dz-prefix z1toz6`, `--tables-dir <fam build>` and `--out-dir <cwfs build>`.

**Import smoke test: no regressions.** 63 of 63 modules import before and after
(`rubinwork/common/scripts/import_smoke_test.py --compare`), "all 63 module import
outcomes identical".

**Tests: 90 passed** across `rubinwork/products/` — the 26 new `cwfs_tables` tests plus
all 64 pre-existing `fam_tables` and `catalog` tests, none of which needed changing.
`test_generated_param_sets_carry_the_wfs_half` in `test_fam_tables.py` still passes
untouched, because it asserts on the generated output rather than on where the data lives.

### 3a step 4 — the comparison: DONE, exact match

The reference rerun with the moved code at commit `b880f2d`, **51.0 s wall clock**
(reference 65.3 s), same 6 in-focus exposures, same 171 donut rows, same 0 FAM visits
missing a CWFS table, and the same fitted validation offsets to the printed precision.
Output in `_phase2_refbuild/cwfs_tables/refitWcs_2025_20251126_moved/`.

`python -m rubinwork.products.cwfs_tables.compare_builds` at **zero tolerance**
(`rtol=0.0, atol=0.0`):

| table | rows | columns compared | columns only in one side | differing columns |
|---|---|---|---|---|
| `donuts.parquet` | 171 | 20 | 0 | **0** |
| `visits.parquet` | 6 | 8 | 0 | **0** |

The key `(day_obs, seq_num, detector, thx_OCS, thy_OCS)` is unique on both sides (171 of
171 rows), so the alignment is sound rather than accidentally agreeing. The validation PDF
came out at 106,605 bytes on both sides (0.00% difference), though it is deliberately not
byte-compared.

**Nothing differs — not one column in either table.** `RESULT: builds match`, exit 0.

## 3a, `miw` — the move

### Scope, agreed 2026-10-10

Moves **five Snakefile rules**, the measured-intrinsic configuration, and the two readers.
No builder code moves: the build runners are external, in `ts_intrinsic_wavefront/bin/`,
and were not touched.

| rule | jobs before | runner |
|---|---|---|
| `build_intrinsic` | 36 | `$WF_BIN/run_build_intrinsic.py` |
| `intrinsic_split` | 9 | `$WF_BIN/run_intrinsic_split.py` |
| `intrinsic_sidecar` | 12 | `$WF_BIN/run_make_intrinsic_sidecar.py` |
| `wfs_intrinsic_sidecar` | 8 | same, `--wfs-corner-height` |
| `refit_mi` | 12 | `$WF_BIN/run_dz_fit.py` |

All counts are jobs (dimensionless). **`refit_mi` belongs to `miw`**, Aaron's call
2026-10-10: its `fits.parquet` is a function of the MIW variant and nothing downstream
re-derives it, `fam_tables` already owns the plain `fit` rule calling the same
`run_dz_fit.py` without a sidecar, and nine `aos/` study rules read
`output/miw/<d>/fits.parquet` as product data. Owning it makes a `miw` build
self-contained: grids -> split -> sidecar -> refit, manifest last.

Everything in `aos/code/miw/` is **study** code and stayed: `compare_build_dof`,
`compare_miw_versions`, `compare_pupil_models`, `compare_rbr_arms`, `compare_schemes`,
`run_study_radialbins`, `check_dof_ranges`, `test_rbr_against_prototype`.

### `aos/mi_config.yaml` is now GENERATED, and its body is byte-identical

The same situation as `param_sets.yaml`:
`lsst.ts.intrinsic.wavefront.mi_config.default_config_path()` returns
`Path('mi_config.yaml')` relative to the working directory with no other candidate, and
**ten** things read it through that default — the three build runners plus
`aos/code/{coadd/run_coadd_blocks_miw, coadd/recompute_coadd_metrics, bounce/run_bounce,
lut/run_build_lut, miw/run_study_radialbins, correlations/run_dz_correlations,
output_paths}.py`. So it cannot be retired and must not change.

**The configuration is held in `variants.yaml` as literal text blocks, and the generator
concatenates them.** This is the one real design decision of the move, and it was reached
after both parsing routes failed:

- `yaml.safe_dump`, which `fam_tables.gen_param_sets` uses, **discards comments**.
  `mi_config.yaml` is hand-written and carries comments *inside* entries — the
  `day_obs_max` freeze note, the Range-Bounded Recovery rationale blocks, the per-param_set
  preambles — so this loses load-bearing prose.
- a **`ruamel.yaml` round-trip preserves comments but moves them** once the structure is
  reshaped. Measured: a straight round-trip of the file reproduces it byte-identically with
  `indent(mapping=2, sequence=4, offset=2)`, `width=70`, an explicit `null` representer and
  two fixups. But regrouping the entries from `param_set -> [entries]` into
  `variant -> entry` scatters them, because the tokens attach to **key names and sequence
  indices**: the commented-out `danish_1_0` block hangs off the `measured_intrinsics` key,
  the per-param_set preambles off the previous group's last entry, and the 22/12 note off
  sequence index 5. Reshaping drops or relocates all of them.

Holding each block as a `text:` scalar makes byte-identity true **by construction** and
keeps every comment where its author put it. What the generator still *checks*, rather than
merely copies: each entry's `variant`, `mi_name` and `dir_name` index fields must agree
with the `name:` and `dir_name:` in its own text block, and `variant` must be the group's
`short_code` plus the `dir_name`. A block edited without its index fields fails loudly.

Verified three ways:

- the generated **body** (everything from `defaults:` on) is byte-identical to the
  pre-move file, committed as `rubinwork/products/miw/tests/mi_config_premove_body.yaml`
  (13,197 bytes, md5 `ee466552be72a16cc312388fcb044f76`).
- the **resolved** config `load_mi_config` hands each runner is identical for **all 12**
  `(param_set, mi_name)` pairs, compared as canonical JSON. That is what the rules hash as
  their rerun trigger, so no cached output is invalidated.
- the whole-file md5 is `e0369b212b64057dec9aaa5997a679bb`, against the hand-written
  `580fcb974be53e3d43dafd401fde2637`. **The two differ only in the header**: the generated
  file replaces the hand-written preamble with the DO-NOT-EDIT notice, and that preamble's
  content moved into `variants.yaml`. The external package parses the body and ignores
  comments.

`aos/param_sets.yaml` is untouched: md5 `0f72da651cf9319cc283754c614051fd` before and
after. `fam_tables/variants.yaml` gained a `short_code` field (`d10`, `d12`, `d13t`,
`d13v1000`), which `gen_param_sets`'s `PASSTHROUGH` whitelist drops.

### Variants: twelve, named `<short_code>-<dir_name>`

| variant | param_set | mi_name | note |
|---|---|---|---|
| `d12-A_50_34_i` | `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` | `pathA_50_34_i` | `day_obs_max: 20260513` |
| `d12-A_50_34_i_5rot` | same | `pathA_50_34_i_5rot` | `build_from` |
| `d13t-A_50_34_i` | `danish_1_3_test` | `pathA_50_34_i` | |
| `d13t-A_50_34_i_5rot` | same | `pathA_50_34_i_5rot` | `build_from` |
| `d13v1000-A_50_34_i` | `danish_1_3_v1000` | `pathA_50_34_i` | |
| `d13v1000-A_50_34_i_rbr` | same | `pathA_50_34_i_rbr` | superseded, kept |
| `d13v1000-A_50_50_i_rbr` | same | `pathA_50_50_i_rbr` | |
| `d13v1000-A_50_34_i_5rot` | same | `pathA_50_34_i_5rot` | `build_from` |
| `d13v1000-A_50_50_i_rbr_5rot` | same | `pathA_50_50_i_rbr_5rot` | `build_from` |
| `d13v1000-A_22_12_i` | same | `pathA_22_12_i` | explicit 22-DOF list |
| `d13v1000-A_22_12_i_5rot` | same | `pathA_22_12_i_5rot` | `build_from` |
| `d13v1000-A_50_34_i_rbr_5rot` | same | `pathA_50_34_i_rbr_5rot` | `build_from` |

`build_from` names a **product variant** in `variants.yaml`; the generator writes the
parent's `mi_name`, which is what `--mi-name` expects. `reader.build_source()` resolves it
back and the tests assert it never crosses a `fam_variant`.

**Two names exceed the plan's 24-character variant limit** —
`d13v1000-A_50_34_i_rbr_5rot` and `d13v1000-A_50_50_i_rbr_5rot` are both 27. The limit in
section 5 of the plan is raised to 32 rather than mangling names that have to round-trip to
`dir_name`.

### The official MIW: three external variants, manifests only

Registered with `catalog.register(..., location=...)`, no files, build name `external`
(an external variant has one entry, not dated builds; a new release is a new variant).
Probed 2026-10-10 in `/repo/main`:

| variant | location | datasets |
|---|---|---|
| `official-20260528a` | `LSSTCam/calib/DM-55048/intrinsicZernikes.v1.0/intrinsicsGen.20260528a` | 1,230 `intrinsicZernikes` over bands u g r i z y |
| `official-gmegias-v3` | `u/gmegias/calib/DM-55048/intrinsicZernikes.v3` | 1,230, the same set |
| `staged-v1` | `aos/calibration/miw/intrinsic_split_maps_v1.parquet` | — |

All counts are datasets (dimensionless), 205 detectors x 6 bands. `official-gmegias-v3` is
the collection five `optatmo/code/` scripts read by default. `staged-v1` carries its
`provenance.yaml`'s `param_set`, `mi_name` and `git_sha` in the manifest `config`, and a
test asserts the three still agree with the YAML. **`aos/calibration/` is not touched** —
the manifest points at it. Nothing the official MIW needs comes from this repository.

### Readers: one product reader, one library

| old path | now | kind |
|---|---|---|
| `aos/code/miw_io.py` | `rubinwork/products/miw/reader.py` | **product reader** — takes a path, returns `(pts, zk, df)` from an `intrinsic_split_maps` parquet. `load_miw` kept as an alias of `load_maps`. |
| `aos/code/miw_corner_intrinsic.py` | `rubinwork/miw_corner.py` | **library** — evaluates a decomposition it is handed at the `ts_ofc` corner sample points, writes nothing, needs `lsst.ts.ofc` and `batoid_rubin`. A library may not import a product, so `MiwCornerLookup` takes `path=`. |

Shims at both old paths bind the real module's namespace and set
`sys.modules[__name__] = _real`. Verified: a bare-name import after a `sys.path.insert`
gives the **same module object**, and every checked attribute is the same object, not a
copy — `load_miw`/`JS_DEFAULT` for `miw_io`, and `MiwCornerLookup`, `decomp_path`,
`load_decomposition`, `corner_field_points`, `corner_z4_height_um`, `DEFAULT_PARAM_SET`,
`DEFAULT_MI_NAME` for `miw_corner_intrinsic`. The three real consumer routes were exercised
directly: `static_optics` (`from miw_io import load_miw`), `code/miw`
(`from miw_io import JS_DEFAULT`) and `value_added/code/build_optical_state.py`
(`from miw_corner_intrinsic import DEFAULT_PARAM_SET, MiwCornerLookup`).

**`decomp_path` was broken and is fixed — the one behaviour change in the move.** It built
`output/<param_set>/<mi_name>/`, joining the two **long** keys, a layout the tree has not
had since `dir_name` was introduced; the real layout is
`output/miw/<ps dir_name>_<mi dir_name>/`. So every default lookup missed.
`notes/status/vmode_thermal_and_lut_handoff.md` records `DEFAULT_PARAM_SET` and
`DEFAULT_MI_NAME` as "both still stale and name nothing on disk" — **that reading was
wrong**: the keys were always right and the path construction was not. With the path fixed
the default pair resolves to a real file
(`aos/output/miw/danish_1_2_A_50_34_i_5rot/intrinsic_split_decomp.parquet`, confirmed on
disk). Still pass `--intrinsic-ref` and `--miw-param-set` explicitly: which MIW is
subtracted defines the optical state, so it belongs in the variant, not a module default.

### 3a step 1 — reference build with the unmoved code: DONE (commit `d81cc3d`)

Built at commit `34b9980` (clean tree), interactive on `sdfiana`, **454.2 s wall clock**
total across the six steps.

A throwaway `pathA_50_34_i_ref2bin` entry on `danish_1_2`, `n_dof 50 / n_keep 34`, in its
own `mi_config_ref.yaml` passed via `--config`, so **`aos/mi_config.yaml` was never
touched**. Two rotator bins, `[-3, 3]` and `[55, 65]`.

**Two bins, not one.** `run_intrinsic_split` forces OCS-only when it sees fewer than two
distinct rotator angles (`run_intrinsic_split.py:223`), so a one-bin reference would never
exercise the CCS half. With two the split saw thetas `[-0.01, 60.0] deg` (2 distinct) and
produced `|O|` telescope-fixed RMS 0.0542 µm and `|C|` camera-fixed RMS 0.0130 µm of
wavefront.

Input: a **three-night FAM slice**, 20260315/16/17, staged by
`rubinwork/products/miw/refbuild/stage_fam_nights.py`:

```bash
python -m rubinwork.products.miw.refbuild.stage_fam_nights \
    --reference aos/output/fam_processing/danish_1_2 \
    --day-obs 20260315 20260316 20260317 \
    --out /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/miw/_fam3n
```

Staged slice: `visits.parquet` 169 rows x 405 columns, `fits.parquet` 168 rows x 456
columns, `donuts.parquet` 475,128 rows x 48 columns in 169 row groups (507 MB, 4.1% of the
full 12.36 GB). 8.3 s.

**Why a slice.** `build_intrinsic` selects its own visits and is already small, but
`intrinsic_sidecar` and `refit_mi` read the **whole** FAM `donuts.parquet` — 9,083,526 rows
and 12.36 GB for `danish_1_2` — which would dominate the reference.

**Two things the staging had to get right, both found by failing first:**

1. **The donut slice must be copied ROW GROUP BY ROW GROUP.**
   `run_build_intrinsic.load_kept_donuts` selects donuts by row group, keying each group on
   its `day_obs`/`seq_num` column **statistics**, so a FAM `donuts.parquet` holds exactly
   one row group per visit. The first attempt streamed by row batch and rewrote the slice as
   one large group: every lookup missed and the build died on `RuntimeError: No donut row
   groups matched the kept visits`. Copying groups is also 16x faster — 8.3 s against 133 s.
2. **`visits.parquet` carries `alt` in RADIANS**, while `mi_config.yaml` states the
   elevation window in degrees (`alt_min_deg: 65.0`, `alt_max_deg: 75.0`). A first bin count
   filtering on `alt` directly returned **0 visits** of 3,385. `bin_visit_counts` converts.

Bin sizes under the build's own cuts (band `i`, `BLOCK-T614_triplets`, elevation 65-75 deg,
`day_obs <= 20260513`): `[-3, 3]` 37 visits and `[55, 65]` 36, both drawing only from those
three nights, so the slice does not change what the build sees in them. Counts are visits
(dimensionless).

The commands, run from `aos/` (the unmoved rules' own cwd), and the measured cost:

| step | wall clock | result |
|---|---|---|
| `build_intrinsic` rot_-3_3 | 145.1 s | 37 visits, 37 good, 104,194 donuts on a 73x73 grid |
| `build_intrinsic` rot_55_65 | 133.4 s | 36 visits, 36 good, 99,696 donuts |
| `intrinsic_split` | 58.4 s | 3,985 map rows, 21 (j, part) records, 64-page PDF |
| `intrinsic_sidecar` | 14.7 s | 475,128 of 475,128 donuts with a finite intrinsic, Z4_height mean -0.0091 µm of wavefront |
| `wfs_intrinsic_sidecar` (`refitWcs`) | 13.4 s | 34,631 of 34,631 corner donuts, Z4_height mean -0.0277 µm of wavefront |
| `refit_mi` | 88.9 s | 169 rows x 831 columns, 0 of 169 visits flagged bad_fit |

Output in `_phase2_refbuild/miw/ref/`, 21 files:

| file | rows | columns | bytes |
|---|---|---|---|
| `build/rot_-3_3/intrinsic_grid.parquet` | 3,831 | 9 | 2,488,564 |
| `build/rot_-3_3/dz_fits.parquet` | 37 | 683 | 494,718 |
| `build/rot_-3_3/intrinsic_cov_edge.parquet` | 21 | 4 | 7,823 |
| `build/rot_-3_3/mi_config.yaml` | — | — | 1,031 |
| `build/rot_55_65/intrinsic_grid.parquet` | 3,843 | 9 | 2,493,349 |
| `build/rot_55_65/dz_fits.parquet` | 36 | 683 | 489,296 |
| `build/rot_55_65/intrinsic_cov_edge.parquet` | 21 | 4 | 7,815 |
| `build/rot_55_65/mi_config.yaml` | — | — | 1,032 |
| `intrinsic_split_maps.parquet` | 3,985 | 44 | 1,610,291 |
| `intrinsic_split_decomp.parquet` | 21 | 10 | 9,602,937 |
| `intrinsic_split_rms.parquet` | 21 | 12 | 5,886 |
| `zk_intrinsic.parquet` | 475,128 | 8 | 96,589,099 |
| `wfs/refitWcs/zk_intrinsic.parquet` | 34,631 | 8 | 7,513,964 |
| `fits.parquet` | 169 | 831 | 1,414,885 |
| `intrinsic_split.pdf` | — | — | 4,458,423 |
| 6 per-bin PDFs | — | — | 810,920 to 2,421,301 each |

**Comparison keys, each verified unique on the reference** (unique rows of total, counts
dimensionless):

| table | key | unique |
|---|---|---|
| `intrinsic_split_maps` | `thx_deg, thy_deg` | 3,985 / 3,985 |
| `intrinsic_split_decomp` | `j, part` | 21 / 21 |
| `intrinsic_split_rms` | `j` | 21 / 21 |
| `fits` | `day_obs, seq_num` | 169 / 169 |
| `zk_intrinsic` | `day_obs, seq_num, detector, centroid_x_extra, centroid_y_extra` | 475,128 / 475,128 |
| `wfs/refitWcs/zk_intrinsic` | same | 34,631 / 34,631 |
| `build/rot_*/intrinsic_grid` | `thx_deg, thy_deg` | 3,831 / 3,831 and 3,843 / 3,843 |
| `build/rot_*/dz_fits` | `day_obs, seq_num` | 37 / 37 and 36 / 36 |
| `build/rot_*/intrinsic_cov_edge` | `j` | 21 / 21 |

`day_obs, seq_num, detector` alone is **not** unique on either sidecar — 30,352 of 475,128
on the FAM one and 4,485 of 34,631 on the corner one, since many donuts share a detector in
one exposure — so the extra-focal centroid is what separates the rows. The comparison
asserts uniqueness on both sides, so a build that breaks it fails loudly instead of
mis-aligning.

The PDFs are **not** byte-comparable (matplotlib embeds a creation timestamp), so step 4
compares the parquet tables at zero tolerance, byte-compares the per-bin `mi_config.yaml`,
and checks only that each PDF is produced at a comparable size.

### 3a step 2 — the move: DONE (commit `b9756aa`)

New in the product: `__init__.py` (with `load`), `reader.py`, `variants.yaml`,
`gen_mi_config.py`, `Snakefile`, `write_manifest.py`, `compare_builds.py`,
`refbuild/stage_fam_nights.py`, and `tests/test_miw.py` (56 tests) with
`tests/mi_config_premove_body.yaml`.

The product Snakefile builds one variant and one build at a time, from the repository root:

```bash
snakemake -s rubinwork/products/miw/Snakefile -n \
    --config variant=d12-A_50_34_i_5rot build=20261010 fam_build=20260920 \
             src_build=20261010
```

Four things it does that the `aos/` rules did not have to:

1. **`cd aos/` for every rule**, for the same reason `fam_tables` and `cwfs_tables` do: the
   external `mi_config.load_mi_config` and `intrinsics_lib.load_param_sets` open their
   config files relative to the working directory.
2. **`build_intrinsic` is not defined at all for a `build_from` variant.** Such an entry
   reuses its parent's grids, so they are inputs this Snakefile does not build, resolved
   from the parent's build through the catalog (`--config src_build=`, default `current`).
   Verified: `d12-A_50_34_i_5rot` plans **7** jobs with **zero** `build_intrinsic`, and its
   `intrinsic_split` reads exactly the five `rotator_select` bins.
3. **`wfs_intrinsic_sidecar` fans out over the `cwfs_tables` variants of the same FAM
   variant that have a build to read**, and reports every one it skips with the reason.
4. **An external variant raises before any path resolves**, naming the
   `--register-external` command instead.

Planned job counts, verified against staged `fam_tables`, `cwfs_tables` and parent-`miw`
builds in a scratch data root (`_phase2_refbuild/_scratch_3a/`), never the real products
tree:

| variant | jobs | breakdown |
|---|---|---|
| `d12-A_50_34_i` (self-building) | 16 | `build_intrinsic` 9, `intrinsic_split` 1, `intrinsic_sidecar` 1, `wfs_intrinsic_sidecar` 2, `refit_mi` 1, `manifest` 1, `all` 1 |
| `d12-A_50_34_i_5rot` (`build_from`) | 7 | the same without the 9 `build_intrinsic` |

`manifest` depends on every table, the split PDF and every CWFS sidecar, so an interrupted
build leaves no manifest and the catalog does not see it. It records
`{"fam_tables": "danish_1_2@20260920"}` and, for a `build_from` variant,
`{"miw": "d12-A_50_34_i"}` — without which the provenance of the maps is unrecoverable from
the manifest, since such a build runs only the split.

### 3a step 3 — verification: DONE

**`aos/mi_config.yaml`: body byte-identical**, `aos/param_sets.yaml` unchanged at md5
`0f72da651cf9319cc283754c614051fd`. Both generators pass `--check`. The resolved config is
identical for all 12 `(param_set, mi_name)` pairs.

**`snakemake -n` in `aos/`: 189 -> 37 jobs, a drop of 152.** 77 are the moved rules; the
other 75 are nine incomplete MIW builds losing their study jobs. Every rule accounted for:

| rule | before | after | change | why |
|---|---|---|---|---|
| `build_intrinsic` | 36 | 0 | **-36** | moved |
| `intrinsic_sidecar` | 12 | 0 | **-12** | moved |
| `refit_mi` | 12 | 0 | **-12** | moved |
| `intrinsic_split` | 9 | 0 | **-9** | moved |
| `wfs_intrinsic_sidecar` | 8 | 0 | **-8** | moved |
| `build_lut` | 12 | 3 | -9 | 9 incomplete MIW builds |
| `wfs_mimic` | 12 | 3 | -9 | same |
| `bounce` | 12 | 3 | -9 | same |
| `dz_correlations` | 12 | 3 | -9 | same |
| `dz_explained` | 12 | 3 | -9 | same |
| `thermal_correlations` | 12 | 3 | -9 | same |
| `vmode_correlations` | 12 | 3 | -9 | same |
| `study_radialbins` | 12 | 2 | -10 | 9 incomplete, plus 1 now up to date |
| `wfs_dof_compare` | 8 | 6 | -2 | `tarts` has no corner sidecar |
| `wfs_corner_compare` | 4 | 4 | 0 | unchanged |
| `aberration_pairs`, `dz_fit_check`, `plots`, `all` | 1 each | 1 each | 0 | unchanged |
| **total** | **189** | **37** | **-152** | |

All counts are jobs (dimensionless). **This is the step-3 criterion failing as literally
written**, by far the most of any product so far, and it is reported rather than hidden.
Three distinct causes:

1. **Only 3 of 12 MIW builds are complete on disk.** `danish_1_2_A_50_34_i`,
   `danish_1_2_A_50_34_i_5rot` and `danish_1_3_test_A_50_34_i_5rot` have both
   `fits.parquet` and `zk_intrinsic.parquet`; six more have the split only, and three have
   nothing. Before the move those nine were plannable because `build_intrinsic` through
   `refit_mi` were in this Snakefile and would have produced them. After it they have no
   producer here, and a plain `snakemake -n` aborted the whole DAG with
   `MissingInputException`. So `MI_PAIRS` now filters on `fits.parquet` **and**
   `zk_intrinsic.parquet` — every study rule reads one or the other — and prints each skip
   with what is missing. The 3 complete builds keep **every** study job.
2. **`tarts` has CWFS donuts but no MIW corner sidecar**, so its 2 `wfs_dof_compare` jobs
   lost their producer when `wfs_intrinsic_sidecar` moved. Filtered and reported the same
   way, on `output/wfs_dof_compare/<d>/<cwfs>/zk_intrinsic.parquet`.
3. **`study_radialbins` drops 10, not 9**, and the extra one is correct:
   `danish_1_3_test_A_50_34_i_5rot`'s PDF (2026-09-28T11:03:10) is **newer than all five of
   its source grids** (11:02:37 to 11:02:50), so it is genuinely up to date. It planned
   before the move only because `build_intrinsic` would have re-run and made the grids
   newer. Confirmed by mtime, not inferred.

**The product Snakefile plans the expected jobs**: 16 and 7, as tabulated above.

**Import smoke test: 63 of 63 import, no regressions.** Two entries change, both the move
itself: the test enumerates files matching `^\s*(from|import)\s+(common|aos_state|...)`,
and `aos/code/miw_corner_intrinsic.py` drops out because the shim no longer imports
`aos_state` while `rubinwork/miw_corner.py` appears because it does. Both shims were then
imported directly and both load cleanly.

**Tests: 146 passed** across `rubinwork/products/` — the 56 new `miw` tests plus all 90
pre-existing `fam_tables`, `cwfs_tables` and `catalog` tests, none of which needed changing.

### 3a step 4 — the comparison: DONE, exact match

The reference rerun with the moved code at commit `b9756aa`, **449.6 s wall clock**
(reference 454.2 s): 138.9 and 135.2 s for the two bins, 58.6 s split, 14.7 s sidecar,
13.4 s corner sidecar, 88.8 s refit. Same visit counts, same `|O|` 0.0542 and `|C|` 0.0130
µm of wavefront, same 475,128 and 34,631 donuts reconstructed, same 0 of 169 visits flagged
bad_fit. Output in `_phase2_refbuild/miw/moved/`.

`python -m rubinwork.products.miw.compare_builds` at **zero tolerance**
(`rtol=0.0, atol=0.0`), over **12 tables** discovered by walking both builds:

| table | rows | columns compared | only in one side | differing |
|---|---|---|---|---|
| `intrinsic_split_maps` | 3,985 | 44 | 0 | **0** |
| `intrinsic_split_decomp` | 21 | 10 | 0 | **0** |
| `intrinsic_split_rms` | 21 | 12 | 0 | **0** |
| `zk_intrinsic` | 475,128 | 8 | 0 | **0** |
| `wfs/refitWcs/zk_intrinsic` | 34,631 | 8 | 0 | **0** |
| `fits` | 169 | 831 | 0 | **0** |
| `build/rot_-3_3/intrinsic_grid` | 3,831 | 9 | 0 | **0** |
| `build/rot_-3_3/dz_fits` | 37 | 683 | 0 | **0** |
| `build/rot_-3_3/intrinsic_cov_edge` | 21 | 4 | 0 | **0** |
| `build/rot_55_65/intrinsic_grid` | 3,843 | 9 | 0 | **0** |
| `build/rot_55_65/dz_fits` | 36 | 683 | 0 | **0** |
| `build/rot_55_65/intrinsic_cov_edge` | 21 | 4 | 0 | **0** |

Every key is unique on both sides, so the alignment is sound rather than accidentally
agreeing. Both per-bin `mi_config.yaml` files are **byte-identical**. All 7 PDFs came out
at identical size (0.00% difference), though they are deliberately not byte-compared.

**Nothing differs — not one column in any of the twelve tables.** `RESULT: builds match`,
exit 0.


## Bridging the study rules to the catalog (commit `fec3741`)

Done before `coadds`, because after the `fam_tables`, `cwfs_tables` and `miw` moves the
`aos/` study rules read product data only from the old tree, so a product rebuilt into the
new tree stays invisible to them until phase 5 — and nine of the twelve MIW variants have
no complete old-tree build, so their study jobs were simply gone.

`aos/Snakefile` now has **one resolver**, used by every study-rule input that reads
`fam_tables`, `cwfs_tables` or `miw` data and by every `--tables-dir` / `--miw-dir` /
`--wfs-dir` argument those rules pass to a script. It returns a DIRECTORY, resolved:

1. the old-tree path, if its required files are all there;
2. else the catalog's `current` build of the matching product variant;
3. else the old-tree path anyway, so the existing gates (`_mi_complete`, `wfs_variants`,
   `_has_corner_sidecar`) see a missing file and skip the variant with a reported reason,
   exactly as before.

**No study output path changes** — outputs stay keyed on the old-tree dir names `{ps}`,
`{d}` and `{cwfs}`, so nothing already built is invalidated or relocated.

Three details that are not obvious:

- **The variant names come from the products' own index fields, not from string surgery on
  the directory names.** A `fam_tables` variant IS its `param_sets.yaml` `dir_name`; a
  `cwfs_tables` variant carries the `{cwfs}` wildcard as its `upstream_name`; a `miw`
  variant carries the second half of `{d}` as its `dir_name`. A renamed variant therefore
  stops resolving instead of silently resolving to a stale old-tree directory.
- **The MIW build grids resolve separately from the MIW tables.** A `build_from` parent
  legitimately has per-bin grids and nothing else — that is the only reason several of them
  exist — so gating the grids on `fits.parquet`/`zk_intrinsic.parquet` would send
  `src_miw_dir` back to the old tree for a parent only the new tree has. `src_miw_dir`
  resolves on the grid files themselves.
- **The corner sidecar changes directory across the trees.** The old tree writes it under
  the CONSUMER's study directory, `output/wfs_dof_compare/<d>/<cwfs>/zk_intrinsic.parquet`;
  the product writes it inside its own build as `wfs/<cwfs variant>/zk_intrinsic.parquet`,
  keyed on the full `cwfs_tables` variant name rather than the bare `wfs_name`.
  `miw_corner_sidecar()` maps between them.

`report_resolved()` prints which source every variant resolved to, once, at DAG time, after
a sweep that resolves every param_set and every planned pair so the report is complete
before the first rule is planned.

**Verified both ways.**

| | jobs | resolutions |
|---|---|---|
| `snakemake -n` in `aos/`, real data root | **37** — today's count, rule for rule | 21: 11 old tree, 0 catalog, 10 unresolved |
| the same with `RUBINWORK_DATA=.../_scratch_3a` and a staged complete `miw/d13v1000-A_50_34_i_5rot@20261010` | **44** | 23: 13 old tree, **1 catalog**, 9 unresolved |

All counts are jobs or variant resolutions (dimensionless). The +7 are exactly that
variant's study jobs reappearing — one each of `bounce`, `build_lut`, `dz_correlations`,
`dz_explained`, `thermal_correlations`, `vmode_correlations`, `wfs_mimic`.
`study_radialbins` stays at 2 because its grids come from the `build_from` parent
`d13v1000-A_50_34_i`, which has grids in neither tree. The planned `wfs_mimic` job reads
its MIW from the catalog and its FAM tables from the old tree — the intended mixed
resolution, since `danish_1_3_v1000`'s old-tree FAM tables are complete and win.

146 product tests pass; the import smoke test is 63 of 63 ok, the same set and the same
result as after the `miw` move.

`aos/code/infra/stage_scratch_product.py` stages a placeholder build (one-row tables plus a
complete manifest) into a scratch root, and **refuses any root not under a `_scratch_3a`
directory** so it can never touch the real products tree.

## 3a, `coadds` — the move

### Scope, agreed 2026-10-10

**The product is `run_coadd_blocks_miw.py` and its four data files.** Everything else in
`aos/code/coadd/` is the coadd-vs-MIW study and stays.

| moves to `rubinwork/products/coadds/builders/` | stays (study) |
|---|---|
| `run_coadd_blocks_miw.py` | `recompute_coadd_metrics`, `analyze_miw_bias_regression`, `analyze_miw_field_order`, `analyze_dz_goodness_of_fit`, `analyze_umode_null_coupling`, `analyze_miw_dz_full_k`, `check_k_truncation`, `run_fam_coadd_miw` |

Product files: `blocks_summary.parquet`, `coadd_metrics.parquet`, `block_grids.npz`,
`block_grids_ext.npz`, plus the two PDFs as build artifacts.

**Every reader, found by grep rather than taken from the plan. There is no code reader
outside `aos/code/coadd/` at all:**

| reader | reads | where |
|---|---|---|
| `recompute_coadd_metrics.py` | `block_grids.npz` | `:563` |
| same | `blocks_summary.parquet`, twice (`program`; then per-block camera temps) | `:264`, `:294` |
| `analyze_miw_bias_regression.py` | `block_grids.npz` | `:115` |
| `analyze_miw_field_order.py` | `block_grids.npz`, `coadd_metrics_rebin3.parquet` | `:43`-`:46`, **at import time** |
| `analyze_dz_goodness_of_fit.py` | `blocks_summary.parquet` | `:107`, default at `:294` |
| `analyze_umode_null_coupling.py` | `blocks_summary.parquet` via `assign_blocks` | `:69`, `:83` |
| `aos/moresnippets.ipynb:5393` | `block_grids.npz` | scratch snippets notebook |

`aos/Snakefile:922` declares `fam_coadd_miw_maps.pdf` as an **output**, not an input; no
rule consumes any coadd file. `aos/code/infra/migrate_output_layout.py:145` only rewrites
paths.

**`fam_coadd_miw` is NOT part of this move, and the next-action note above was wrong about
it.** That rule runs `run_fam_coadd_miw.py`, which writes only `fam_coadd_miw_maps.pdf` — a
plot no code reads back — so it is a study rule and stays. The script that moves,
`run_coadd_blocks_miw.py`, **has no Snakefile rule at all** today: it is run only from
`aos/run_coadd_blocks_miw.sbatch`. So `snakemake -n` in `aos/` must be **unchanged at 37**
across this move, and there is no `aos/`-side coadd target to plan.

Three findings worth carrying forward, all Aaron's calls on 2026-10-10:

1. **`block_grids_ext.npz` has no reader anywhere** — written at
   `run_coadd_blocks_miw.py:911`, read by nothing, 66 MB in the production build. **Kept**
   in the product: it is the extra-pupil-Zernike primitive the retrieval-bias work needs,
   and it is what distinguishes the two variants.
2. **`coadd_metrics.parquet` has no reader either.** The only metrics file anyone reads is
   `coadd_metrics_rebin3.parquet`, which `recompute_coadd_metrics.py:634` writes — a
   **study** output. **Kept**: it is the natural per-coadd table and the rebinned one is
   conceptually derived from it.
3. **Three study readers point at a stale path that does not exist on disk.**
   `recompute_coadd_metrics.py:562`, `analyze_miw_bias_regression.py:114` and
   `analyze_miw_field_order.py:36` build `output/<long param_set>/coadd_50_34/`, the layout
   the tree lost when `dir_name` arrived — the same bug class as `miw_corner.decomp_path`.
   `aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/` does not exist, so **all three
   are broken today**; two also hardwire `coadd_50_34` as a literal. **Fixed in the move
   commit**, as the one behaviour change.

### Variants: two, prefixed by the MIW upstream

A coadd depends on a `fam_tables` variant **and** a `miw` variant. The MIW is the main
upstream — the product is a per-block remnant *against* a given MIW — so the variant name
carries the MIW variant, not the FAM one.

| variant | fam_variant | miw_variant | `ext_terms` | old-tree dir | old-tree output? |
|---|---|---|---|---|---|
| `d12-A_50_34_i_5rot` | `danish_1_2` | `d12-A_50_34_i_5rot` | `secondary` | `coadd/danish_1_2/50_34/` | **yes**, 6 files, newest 2026-09-14 |
| `d12-A_50_34_i_5rot_v2` | `danish_1_2` | `d12-A_50_34_i_5rot` | `all` | `coadd/danish_1_2/50_34_v2/` | **no — `archive/` only, top level empty** |

**Checked before moving, as this handoff requires.** `aos/output/coadd/danish_1_2/50_34_v2/`
holds only `archive/20260903_iband/`; its top level is empty. Nothing is currently planned
for it (`run_coadd_blocks_miw` has no rule), so no job count changes — but the product
Snakefile reports it as unbuilt.

The plan's names `d12-50_34` and `d12-50_34_v2` are **not** used: `50_34` names the
`n_dof`/`n_keep` pair, not the MIW, so they would be ambiguous as soon as a second MIW
variant gets a coadd. The old-tree directory name (`50_34`, `50_34_v2`) is carried as a
`dir_name` field, exactly as `miw` does.

The two variants differ **only** in `ext_terms`, which changes nothing but
`block_grids_ext.npz` and the per-block PDF page count — the archive README records
`block_grids.npz` as byte-identical between the two 2026-09-03 runs.

**Where the block-selection settings live.** Today they are argparse defaults plus a
`DEFAULTS` array in `aos/run_coadd_blocks_miw.sbatch`; there is no config file, and
`analysis_config.yaml`'s `fam_coadd_miw` section belongs to the *other* script.
`coadds/variants.yaml` now owns them per variant (`bands`, `min_visits`, `max_blur`,
`pointing_tol`, `wide_tol`, `max_seq_span`, `bounce_round`, `bounce_programs`,
`wide_programs`, `ext_terms`, `cwfs_inner`/`cwfs_outer`, `n_bins`, `n_iter`, `pct`), with
the argparse defaults left intact so a hand run is unchanged. **No generated file is
needed** — unlike `param_sets.yaml` and `mi_config.yaml`, nothing external reads a coadd
config by a bare relative path.

### 3a step 1 — reference build with the unmoved code: DONE

Built at commit `fec3741` (clean tree), interactive on `sdfiana`, **76.7 s wall clock**.

**One night, 20260316, chosen to give three blocks.** `run_coadd_blocks_miw.py` has **no
block-count limit flag**, so the reference is sized by staging the FAM nights, the same
route `miw` took. Measured with the real `assign_blocks` on the three nights of the `miw`
reference slice (counts are blocks, dimensionless):

| night | FAM visits | blocks detected | built (`n_visits >= 6`) |
|---|---|---|---|
| 20260315 | 62 | 6 | 5 |
| **20260316** | **35** | **3** | **3** |
| 20260317 | 72 | 6 | 6 |
| all three | 169 | 15 | 14 |

Input, staged by the `miw` product's own script (it needed no change), 2.9 s:

```bash
python -m rubinwork.products.miw.refbuild.stage_fam_nights \
    --reference aos/output/fam_processing/danish_1_2 \
    --day-obs 20260316 \
    --out /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/coadds/_fam_20260316
```

Staged slice: `visits.parquet` 35 rows x 405 columns, `fits.parquet` 35 rows x 456 columns,
`donuts.parquet` 98,199 rows x 48 columns **in 35 row groups**. The row-group structure is
load-bearing again: `run_coadd_blocks_miw._visit_row_groups` keys each group on its
`day_obs`/`seq_num` column statistics, exactly as `run_build_intrinsic.load_kept_donuts`
does, so one row group per visit is required.

The MIW it compares against is the completed `miw` reference,
`_phase2_refbuild/miw/ref/zk_intrinsic.parquet` (475,128 donuts over the three nights). The
sidecar is indexed on `(day_obs, seq_num)`, not row position, so a three-night sidecar
covers a one-night block set.

The build command, run from `aos/` (the unmoved script's own path):

```bash
cd ~/notebooks/rubin-work/aos && python code/coadd/run_coadd_blocks_miw.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --mi-name pathA_50_34_i_5rot \
    --ext-terms secondary \
    --tables-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/coadds/_fam_20260316 \
    --miw-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/miw/ref \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/coadds/ref
```

Note `--out-dir` is **required**: without it the script builds
`output/<long param_set>/<out-name>/`, the stale layout of finding 3 above.

Output, in `_phase2_refbuild/coadds/ref/`, 6 files:

| file | contents | bytes |
|---|---|---|
| `blocks_summary.parquet` | 3 rows x 26 columns | 15,653 |
| `coadd_metrics.parquet` | 3 rows x 81 columns | 48,618 |
| `block_grids.npz` | 42 arrays | 835,382 |
| `block_grids_ext.npz` | 6 arrays | 943,315 |
| `coadd_blocks_miw_perblock.pdf` | 3 blocks x [coadd/MIW/diff] | 3,716,991 |
| `coadd_blocks_miw_timeseries.pdf` | metrics + corr-bars + scatter + 3 telemetry + u-mode heatmap, n=3 | 49,316 |

The three blocks, all `T614_triplets` on 20260316, band `i`, elevation 69.8 deg, all
in-family:

| block | ordinal | seq range | visits | donuts | rotator angle (deg, OCS) | FWHM (arcsec) |
|---|---|---|---|---|---|---|
| 0 | 0 | 647-680 | 12 | 33,496 | +60.12 | 1.13 |
| 1 | 1 | 693-726 | 12 | 33,762 | -59.89 | 1.00 |
| 2 | 2 | 737-767 | 11 | 30,941 | +0.15 | 0.97 |

Each block ran the full 3-iteration U-constrained build and converged: RMS change between
iterations 1.24e-03 then 7.09e-04 µm of wavefront on block 1, 1.25e-03 then 7.27e-04 on
block 2. Coadd-vs-MIW agreement, over the full focal plane:

| block | `resid_rms_combined` (µm of wavefront) | `corr_combined` (Pearson r, dimensionless) |
|---|---|---|
| 0 | 0.0351 | 0.946 |
| 1 | 0.0451 | 0.927 |
| 2 | 0.0496 | 0.946 |

So the reference exercises block detection, the iterative build, the MIW comparison, the
per-Zernike and CWFS-annulus metrics, the u-modes, both npz files and both PDFs.

**Comparison keys.** `block` is unique on both tables — 3 of 3 rows on
`blocks_summary.parquet` and on `coadd_metrics.parquet`. The two npz files are compared
array by array by name (42 and 6 arrays), with the per-block axis aligned on `block`.

The PDFs are **not** byte-comparable (matplotlib embeds a creation timestamp), so step 4
compares the two parquet tables and both npz files at zero tolerance and checks only that
each PDF is produced at a comparable size.

### 3a step 2 — the move: DONE (commit `d0e4182`)

`run_coadd_blocks_miw.py` moved to `rubinwork/products/coadds/builders/`, with a shim at
the old path. New in the product: `__init__.py` (with `load` and `load_grids`),
`reader.py`, `variants.yaml`, `Snakefile`, `write_manifest.py`, `compare_builds.py`, and
`tests/test_coadds.py` (40 tests).

The builder needed **no import changes at all** — it imports only stdlib, third-party and
the external package, with no `sys.path` work and no sibling imports. What was added is a
`--variant` route, as `run_wfs_mktable` has: it resolves the param_set, the `mi_name` and
the block-selection settings from `variants.yaml`, while `--param-set` plus `--mi-name`
still works. A flag given explicitly on the command line still wins over the variant's
setting, the sentinel for "not given" being the parser's own default.

**The shim gives module identity**, verified: a bare-name import after a `sys.path`
insert of `aos/code/coadd` returns the *same module object*, and `assign_blocks`,
`block_summary`, `main` and `THERMAL_VARS` are the same objects, not copies. Those are
what `analyze_miw_bias_regression.py`, `analyze_miw_field_order.py` and
`aos/moresnippets.ipynb` import that way.

Four things worth recording:

1. **`dir_name` must be quoted in `variants.yaml`.** An unquoted `50_34` parses as the
   **integer 5034** — YAML treats an underscore as a numeric digit separator. Caught by
   printing the parsed value, and now a test (`test_dir_name_is_a_string_not_a_yaml_integer`).
2. **`write_manifest` refuses to record only one upstream.** `--fam-build` and
   `--miw-build` must come together: the FAM build sets which visits exist and the MIW
   sets what is subtracted, so half the provenance recorded as though it were all of it
   would make a build look reproducible when it is not.
3. **The product Snakefile is one rule, not six.** The script does block detection, the
   per-block iterative build and both renders in one pass over the donut table; splitting
   it per output file would mean reading that multi-GB table more than once.
4. **`recompute_coadd_metrics.py` now requires `--out-dir` with `--variant`.** It writes
   `coadd_metrics_rebin<f>.parquet` and `coadd_timeseries.pdf` into the directory it read,
   which with a catalog build would edit a completed product build — rule 5 of the plan's
   section 2. Reading a build is fine; writing into it is not.

`aos/run_coadd_blocks_miw.sbatch` calls the module rather than the old path, and points at
the product Snakefile as the route that writes a manifest. The three stale-path study
readers were fixed in the same commit, per finding 3 above; `analyze_miw_field_order.py`
has no argument parser (110 lines, no functions, runs top to bottom), so its build is
named by the environment variables `COADD_VARIANT`, `COADD_MIW_VARIANT` and
`COADD_STUDY_DIR` instead.

Planned job counts, verified against staged `fam_tables` and `miw` builds in the scratch
data root (`_phase2_refbuild/_scratch_3a/`), never the real products tree:

| variant | jobs | breakdown |
|---|---|---|
| `d12-A_50_34_i_5rot` | 3 | `coadd_blocks` 1, `manifest` 1, `all` 1 |
| `d12-A_50_34_i_5rot_v2` | 3 | the same, with `--ext-terms all` |

All counts are jobs (dimensionless). A missing upstream build names itself:
`no build '29990101' of miw/d12-A_50_34_i_5rot; known builds: ['20261010', 'current']`.
An external MIW variant is refused before any path resolves, since it has no per-donut
sidecar to compare against.

### 3a step 3 — verification: DONE

**`snakemake -n` in `aos/`: 37 jobs, UNCHANGED**, rule for rule — `aberration_pairs` 1,
`bounce` 3, `build_lut` 3, `dz_correlations` 3, `dz_explained` 3, `dz_fit_check` 1,
`plots` 1, `study_radialbins` 2, `thermal_correlations` 3, `vmode_correlations` 3,
`wfs_corner_compare` 4, `wfs_dof_compare` 6, `wfs_mimic` 3, `all` 1. **This is the first
product whose move changes no job count**, and the reason is structural rather than
lucky: the moved script never had a rule in `aos/Snakefile`, so there was nothing there to
remove. `fam_coadd_miw`, the rule that *is* in that Snakefile, belongs to
`run_fam_coadd_miw.py` and did not move.

**The product Snakefile plans the expected jobs**: 3 and 3, as tabulated above.

**Import smoke test: 63 of 63 import, and `--compare` against the pre-move run reports
`no change: all 63 module import outcomes identical`.** No module's import regressed and
none dropped out of the match set, because the moved file was not in it — it imports no
moved library.

All eight remaining `aos/code/coadd/` scripts were then imported directly in script mode:
every one loads. `analyze_miw_field_order.py` is the exception by design — its reads are
at module level, so with no coadd build in the catalog it now stops with
`ProductNotFound: no variant 'd12-A_50_34_i_5rot' of product 'miw'`, a named error
instead of the previous silent miss on a path that never existed.

**Tests: 186 passed** across `rubinwork/products/` — the 40 new `coadds` tests plus all
146 pre-existing `fam_tables`, `cwfs_tables`, `miw` and `catalog` tests, none of which
needed changing.

### 3a step 4 — the comparison: DONE, exact match

The reference rerun with the moved code at commit `d0e4182`, **77.0 s wall clock**
(reference 76.7 s), through the new `--variant d12-A_50_34_i_5rot` route so that the
variant resolution itself is exercised rather than bypassed. Same 3 blocks, same visit and
donut counts, same convergence (1.24e-03 then 7.09e-04 µm of wavefront on block 1,
1.25e-03 then 7.27e-04 on block 2), same per-block `resid_rms` and Pearson r. Output in
`_phase2_refbuild/coadds/moved/`.

`python -m rubinwork.products.coadds.compare_builds` at **zero tolerance**
(`rtol=0.0, atol=0.0`):

| file | rows or blocks | columns or arrays compared | only in one side | differing |
|---|---|---|---|---|
| `blocks_summary.parquet` | 3 rows | 26 columns | 0 | **0** |
| `coadd_metrics.parquet` | 3 rows | 81 columns | 0 | **0** |
| `block_grids.npz` | 3 blocks | 42 arrays | 0 | **0** |
| `block_grids_ext.npz` | 3 blocks | 6 arrays | 0 | **0** |

`block` is unique on both sides of both tables (3 of 3), and both npz files hold the same
blocks in the same order, so the alignment is sound rather than accidentally agreeing.
Both PDFs came out at identical size (0.00% difference), though they are deliberately not
byte-compared.

**Nothing differs — not one column of either table, not one array of either npz.**
`RESULT: builds match`, exit 0.

The npz comparison is new and needed its own code: most of what this product carries is
arrays, not table rows. `compare_npz` compares array by array **by name**, so an array
added or removed is reported rather than skipped, and it **realigns the per-block axis on
the `block` array** rather than assuming stored order — a test builds two archives holding
the same blocks in different orders and asserts that only `block` itself differs.

### The coadd study scripts read the old tree again (2026-10-11)

The `coadds` move left the three study scripts resolving their build **only** through the
catalog, so until phase 3c they could not read the one complete coadd build that exists —
`aos/output/coadd/danish_1_2/50_34/`, six files, newest 2026-09-14. They now follow the
`aos/Snakefile` resolver's order instead, which is the order the study rules already use:

1. an explicit `--coadd-dir` (or `COADD_DIR` / `MIW_DIR` in the parser-less script);
2. else the old-tree directory built from the variant's **own index fields** —
   `aos/output/coadd/<fam_variant>/<dir_name>/` for a coadd,
   `aos/output/miw/<fam_variant>_<dir_name>/` for a MIW — if every required file is there;
3. else the catalog's build of that variant;
4. else `RuntimeError` naming both places tried.

`resolve_product_dir(product, variant, old_dir, required, build)` in
`recompute_coadd_metrics.py` is the one implementation; `resolve_coadd_dir` wraps it for
the two scripts with a parser and `analyze_miw_field_order.py` calls it directly, twice,
at module level. `COADD_REQUIRED` is `block_grids.npz` + `blocks_summary.parquet`; the MIW
side needs only `intrinsic_split_maps.parquet`.

**Unlike the Snakefile, a study script raises rather than falling back to the old-tree
path.** The Snakefile's fallback exists so a downstream gate sees a missing file and skips
the variant with a reason; a study script has no gate, so a silent bad path would surface
as a `FileNotFoundError` on a path nobody chose.

Two knock-on fixes the old-tree route forced, both in the same commit:

- **The "product builds are not written into" guard moved off `--variant`.** It is now
  `is_catalog_build(cdir)` — a `pathlib.relative_to` test against
  `catalog.products_root()` — so `--out-dir` (`recompute_coadd_metrics`) and `--out`
  (`analyze_miw_bias_regression`) are required only when the resolved directory really is
  a product build. Writing beside the data in the old tree is what a hand run always did
  and is allowed again. `analyze_miw_bias_regression` checks it immediately after
  resolving, not at render time, so it fails before the regression runs.
- **`resolve_visits_path` can no longer raise.** It called `catalog.manifest` whenever
  `--variant` was given, which throws with no catalog build, and its caller only *warns*
  on a missing file. It now takes the old `aos/output/fam_processing/<P>/visits.parquet`
  first and swallows the manifest lookup's failure.

The dead `--param-set` / `--out-name` / `--output-root` flags were dropped from both
parsers — nothing read them after the move.

**Verified on the real data root** (`/sdf/group/rubin/u/roodman/LSST/rubin-work`, which
holds only the `miw` product today):

| check | result |
|---|---|
| `resolve_coadd_dir('d12-A_50_34_i_5rot')` | `aos/output/coadd/danish_1_2/50_34`, `is_catalog_build` False |
| `resolve_visits_path` for it | `aos/output/fam_processing/danish_1_2/visits.parquet` |
| `resolve_coadd_dir('d12-A_50_34_i_5rot_v2')` | `RuntimeError` naming the absent old-tree dir **and** the absent catalog product — correct, it has a build in neither tree |
| MIW resolution | `aos/output/miw/danish_1_2_A_50_34_i_5rot` |
| `recompute_coadd_metrics.py --help`, `analyze_miw_bias_regression.py --help` | both exit 0 |
| `analyze_miw_field_order.py`, module level | `MIW` and `CO` both resolve to the old tree; it stops only on `coadd_metrics_rebin3.parquet`, which is a **study** output `recompute_coadd_metrics` has not been run to produce |
| `snakemake -n` in `aos/` | **37** jobs, unchanged |
| `pytest rubinwork/products/` | 186 passed |

All counts are jobs or tests (dimensionless).

## 3a, `dof_lut` and `bounce_tables` — the move

### Scope and decisions, agreed 2026-10-11

Aaron's three calls, in answer to the investigation below:

1. **`dof_lut` fits the plain FAM DZ, and that is recorded, not changed.** "I used the
   plain FAM DZ then but may want to switch to using the post MIW DZ." Recorded in
   `aos/docs/studies/lut.md` (commit `7d23afd`); no code touched, no 3b fix applied.
2. **The `_july` bounce run is its own variant**, so it can be reviewed later — not a
   build of `d12-A_50_34_i_5rot` as the plan's section 9 says. "I don't recall what the
   _july variant changed."
3. **The bounce tables are promoted to a product, and the shared functions become a
   library, in the move commit** — "when it is most convenient", which is this commit
   because `run_bounce.DEFAULT` has to leave the builder either way.

### `dof_lut` does NOT subtract the MIW: it fits the plain FAM DZ

Established by reading `aos/code/lut/run_build_lut.py`, and the answer to the question
the previous "next concrete action" raised:

| file:line | what it does |
|---|---|
| `aos/Snakefile:794`-`:795`, `:405` | the rule's `fits`/`visits` inputs are `d_tables(w.d)` = `fam_dir(...)`, the **FAM** tables — not `output/miw/<d>/` |
| `run_build_lut.py:132`, `:152` | reads `visits.parquet` and `fits.parquet` from `--tables-dir` |
| `run_build_lut.py:165` | `osv.build_ofc_svd(iZs, k_min, k_max, n_keep_spec, n_dof=n_dof_spec, ...)` |
| `run_build_lut.py:170` | `osv.project_dz_table(fits, prefix, svd)`, whose `W` is `dz_table_to_W(fit_table, ...)` — the **raw** DZ matrix of that table |
| `run_build_lut.py:171`-`:172` | `W_fit = A @ U_eff.T`; `W_resid = W - W_fit`. The only subtraction is the SVD's own reconstruction |
| `run_build_lut.py:99`-`:121`, `:191` | `mi_config` is read for **parameters only**: `n_dof`, `n_keep`, `k_min`/`k_max`, `coord_sys`, `filter`, `programs`, `ofc_normalization_yaml`, `mc.rotator_bins(cfg)` |

No intrinsic map, no `zk_intrinsic`, no MIW path appears in the file. The docstring says
it at `:5`-`:8` ("the **Phase-1** per-visit Double-Zernike fits").

The reference build's own numbers show the consequence: `M2_dy = -2364 µm` of hexapod
decentre (`lut.parquet`, `value` column), far outside any physical stroke, which is what
fitting the **absolute** state without an intrinsic subtracted looks like.

Two side findings, neither fixed:

- The comment at `run_build_lut.py:180`-`:183` says "if MI subtraction leaves
  rotator-coupled residual it gets averaged away here". That describes a MIW build, not
  this script, and is misleading as written.
- `aos/code/processing_compare/compare_fam_processings.py:101` is the one reader outside
  `aos/code/lut/`, and it builds `<base>/<mi>/lut/lut.parquet` — the same dead-layout bug
  class as the `coadds` and `decomp_path` ones. Broken today. Not a 3b fix; flagged.

**Consequence for the variant keying.** `dof_lut`'s real upstream is a `fam_tables`
variant; the MIW variant contributes only SVD and selection settings. The variants are
still named with the MIW-prefixed form below, because a LUT is built per (param_set,
mi_name) pair and the dtag is what the old tree uses — but `variants.yaml` names the
`fam_tables` variant as the upstream the manifest records, with `mi_name` as a settings
key. If Aaron switches to the post-MIW DZ, the upstream becomes `miw` and that is a
variant-definition change, not just a builder change.

### `bounce_tables`: five tables are the product, nine PDFs are the study

`run_bounce.py` writes **5 parquet tables and 9 PDFs** into one directory (`tbl_dir` and
`out_dir` are both `--out-dir` when it is given, `run_bounce.py:272`).

Readers outside `aos/code/bounce/`, found with `rg`:

| file | read by | where |
|---|---|---|
| `bounce_dof_stats.parquet` | `aos/code/cwfs_lut/bounce_compare.py` — a **different study** | `:132`, hardwired `aos/output/bounce/<run>/`; `DEFAULT_BOUNCE_RUN = 'danish_1_2_A_50_34_i_5rot_july'` at `:60` |
| `bounce_fwhm_metric.parquet` | `smatrix/docs/studies/regularized_inversion.md:89` validates against it; the script **recomputes** rather than reads | docs only |
| `bounce_kj_stats.parquet` | the `bounce` rule's declared output and `rule all` (`aos/Snakefile:548`) | no code reader |
| `bounce_fwhm_vs_bvalue.parquet` | nothing | — |

So there is exactly **one** cross-study code reader, and it is the plan's rule 3 case
("a study never reads another study's output"). That is what justifies the product.
`aos/output/bounce/{bending_mode_test_meta,dof_normalization_split}.parquet` sit at the
tree's top level, come from other scripts, and have no code reader; they are not part of
this product.

**The split is inside one script, and the interleaving is worse than `coadds`'.** PDFs
are written at `:702`, `:729`, `:743`, `:775`, `:807`, `:840`, `:907`; then
`bounce_fwhm_metric.parquet` at `:955` with its PDF at `:973`,
`bounce_fwhm_vs_bvalue.parquet` at `:997` with its PDF at `:998`,
`bounce_dof_stats.parquet` at `:1162`, `bounce_kj_stats.parquet` at `:1201`. But **all
the table data comes from one shared analysis block, `:479`-`:700`** — `fwhm_rows`,
`fwhm_bvalue_rows`, the per-night deltas and `df_kj` accumulate there and the plot
sections only consume them. So the builder is `:218`-`:700` plus the four `to_parquet`
calls, the study is the nine `PdfPages` blocks, and no physics moves.

**`bounce` therefore keeps a rule in `aos/Snakefile`**, unlike the three earlier
products: `bounce_summary.pdf` is a study output, so the rule shrinks to its plotting
half and reads the tables from the product. None of `cwfs_tables`, `miw` or `coadds` had
that shape.

### The `run_bounce.py` split, mapped section by section (2026-10-11)

Read off the file before touching it, so the next session does not re-derive it. `main()`
runs `:218`-`:1206`; every one of these is a **top-level** (4-space) section comment inside
it, so the split is by whole sections and no section straddles the boundary.

| lines | section | goes to |
|---|---|---|
| `:218`-`:284` | argparse, config resolution, `out_dir`/`tbl_dir`, matplotlib import | **both** (the builder keeps it; the study rule needs its own copy of the config resolution) |
| `:285`-`:478` | load, bad-fit drop, quality cut, OFC SVD, FWHM tooling, RBR tooling, OIC authority, scheme callables, AOS Trim | **builder** |
| `:479`-`:700` | the bounce analysis loop — `fwhm_rows`, `fwhm_bvalue_rows`, `long_dfs`, `bounce_results`, ending with `df_kj` at `:701` | **builder** |
| `:701`-`:951` | the nine `PdfPages` blocks: DZ vs ordinal, v-mode/DOF vs ordinal, summary, DOF night scatter, DOF per (night, B set), 5/5 camera hexapod | **study** |
| `:952`-`:988` | `bounce_fwhm_metric.parquet` **plus** its own `PdfPages` at `:973` | **split inside**: table to the builder, the bar chart to the study |
| `:989`-`:1043` | `bounce_fwhm_vs_bvalue.parquet` **plus** its own `PdfPages` at `:998` | **split inside**: same shape |
| `:1044`-`:1199` | `bounce_dof_stats.parquet` — **no PDF anywhere in it**, pure table, reads only `bounce_results`, `fit_table`, `DOF_all`, `svd` | **builder** |
| `:1200`-`:1203` | `bounce_kj_stats.parquet` from `df_kj` | **builder** |

So the handoff's earlier "everything to `:700` plus the four `to_parquet` calls" is right in
substance, with two refinements worth having before starting:

- **`bounce_dof_stats` is not a `to_parquet` bolted onto a plot section.** It is a 156-line
  table section of its own at `:1044`-`:1199`, the largest single table block in the file,
  and it is already free of plotting. It moves whole.
- **Only two sections need cutting internally**, `:952`-`:988` and `:989`-`:1043`, and in
  both the table is written *before* its `PdfPages` opens, so the cut is at the `with`
  statement in each.

The state the study half needs from the builder is `bounce_results`, `fit_table`,
`df_kj`, `fwhm_rows`, `fwhm_bvalue_rows`, `DOF_all`, `svd` and the derived tooling — which
is most of the analysis. Plan for that: either the study rule reads the four product
tables back (works for the night-values and FWHM pages, not for the per-ordinal DZ pages,
which need the per-visit arrays), or the shared analysis becomes a library function both
halves call. **Decide which before writing code**; it is the one design choice in this
move that the four earlier products did not face.

### The four wrong-way imports need only three names

`smatrix/code/regularized_inversion/` reaches into `aos/code/bounce/` in four places, and
between them uses **three** names:

| import | file:line | names used |
|---|---|---|
| `import bounce_lib as _bl` | `run_oic_compare.py:231` | `_bl.run_bounce` (`:273`) |
| `import bounce_lib as bl` | `run_oic_compare.py:419` | `bl.run_bounce` (`:461`) |
| `import bounce_lib as bl` | `run_regularized_compare.py:65` | `bl.run_bounce` (`:114`), `bl.paired_deltas_matrix` (`:120`) |
| `import run_bounce` | `run_regularized_compare.py:66` | `run_bounce.DEFAULT` (`:205`) |

**Fix: a library, `rubinwork.bounce`**, carrying the leg-definition and pairing layer
only — `DEFAULT`, `DEFAULT_BOUNCES`, `DOF22`, `CAM_HEX_DOF`, `run_bounce()`,
`form_pairs`, `paired_delta`, `paired_deltas_kj`, `paired_deltas_matrix`,
`filter_visits`, `bounce_program_mask`. The ~20 `plot_*` functions of the 86 KB
`bounce_lib.py` stay with the study.

Why a library and not the product reader: these three callers never read the bounce
*tables*. They recompute the Δ from a fits table to score alternative inversions against
the bounce's own selection, which is library use — pairing logic — not product use.
Routing it through the product reader would make a study import a product's internals to
reach a pure function.

Why not leave it behind the shim until phase 5: the shim preserves the *path*, not the
*direction*. `run_bounce.DEFAULT` would still be a study importing a product **builder**,
which is the thing `/review` is meant to catch, and `DEFAULT` has to leave the builder
anyway.

One wrinkle to fix in the same commit: `bounce_lib.rbr_module()` (`:942`) does a
`sys.path` insert into `smatrix/code` to reach `regularized_inversion`, which is now the
installed `rubinwork.smatrix.regularized_inversion`. The moved library imports it
directly instead.

### Variants: three `dof_lut`, four `bounce_tables`, checked for old-tree output first

Checked before anything moved, as this handoff requires — every one of the twelve MIW
variants, against both old trees:

| product | variants WITH old-tree output | variants with none |
|---|---|---|
| `dof_lut` | `d12-A_50_34_i` (5 files), `d12-A_50_34_i_5rot` (5 files) | the other ten |
| `bounce_tables` | **none, under any MIW-variant directory name** | all twelve |

**`bounce_tables`** — four variants, keyed on the MIW variant **and** the fits table:

| variant | miw_variant | `fits` | old-tree dir | old-tree output? |
|---|---|---|---|---|
| `d12-A_50_34_i_5rot_july` | `d12-A_50_34_i_5rot` | `fits_july.parquet` | `bounce/danish_1_2_A_50_34_i_5rot_july/` | **yes, 13 files** — the tech note's source |
| `d12-A_50_34_i_5rot` | `d12-A_50_34_i_5rot` | `fits.parquet` | `bounce/danish_1_2_A_50_34_i_5rot/` | no |
| `d12-A_50_34_i` | `d12-A_50_34_i` | `fits.parquet` | `bounce/danish_1_2_A_50_34_i/` | no |
| `d13t-A_50_34_i_5rot` | `d13t-A_50_34_i_5rot` | `fits.parquet` | `bounce/danish_1_3_test_A_50_34_i_5rot/` | no |

The last three are the three jobs `aos/Snakefile` plans today. The first is the only one
with data, and **it is a separate variant rather than a build**, Aaron's call: its input
is a different fits table, `output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet` at
3,385 visits over day_obs 20250415–20260713, against the MIW build's own `fits.parquet`
at 1,126 visits over 20260315–20260513. Section 5 says a changed configuration makes a
new variant, and `aos/docs/status/miw_danish_1_3_proposal.md:116` calls `fits_july.parquet`
a "hand-made extra" produced by no rule. The plan's section 9 maps it to
`bounce_tables/d12-A_50_34_i_5rot/`, treating it as a build; that mapping is superseded.

`notes/aos-bounce-test-summary/provenance.md:46` names
`aos/output/bounce/danish_1_2_batoid/` as the intrinsic-choice comparison, but **that
directory does not exist on disk** — so it gets no variant, and the provenance doc points
at a path that is gone.

**`dof_lut`** — three variants, matching the three planned jobs:

| variant | fam_variant | mi_name | old-tree dir | old-tree output? |
|---|---|---|---|---|
| `d12-A_50_34_i_5rot` | `danish_1_2` | `pathA_50_34_i_5rot` | `lut/danish_1_2_A_50_34_i_5rot/` | **yes, 5 files** |
| `d12-A_50_34_i` | `danish_1_2` | `pathA_50_34_i` | `lut/danish_1_2_A_50_34_i/` | **yes, 5 files** |
| `d13t-A_50_34_i_5rot` | `danish_1_3_test` | `pathA_50_34_i_5rot` | `lut/danish_1_3_test_A_50_34_i_5rot/` | no |

Both carry `dir_name` as a quoted string, as `coadds` does. The YAML-integer trap does
not bite here (every name starts with a letter), but the field is quoted anyway and the
`coadds` test is mirrored.

**The `bounce` legs come from an `overrides:` block, not from the code defaults.**
`aos/analysis_config.yaml:180` carries a live `bounce:` override for
`fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` that redefines `bounces` as an elevation
*sweep* (five T720 legs, not one). `DEFAULT_BOUNCES` in `run_bounce.py:65` is only the
fallback. So the product's settings must come through `mc.analysis_section`, which does
accept an explicit `config_path` — unlike `param_sets.yaml` and `mi_config.yaml`, no
generated file is forced, but the override keyed on the **long** param_set key has to keep
resolving.

**Undeclared outputs, both rules.** `build_lut` declares 3 outputs
(`aos/Snakefile:799`-`:801`) but the script writes **5** files: those plus
`lut_by_rotbin.parquet` (`:205`, conditional on a rotator column) and `lut_config.yaml`
(`:241`). `bounce` declares 2 (`:880`-`:881`) and writes **13**. Both get full output
lists in the product Snakefiles.

### 3a step 1 — reference builds with the unmoved code: DONE

Built at commit `7d23afd` (clean tree), interactive on `sdfiana`. **Neither product needed
its input sliced** — unlike `miw` and `coadds`, the full production run IS the smallest
thing that exercises the product code, so both references are the real thing:

| product | wall clock | output |
|---|---|---|
| `dof_lut` | **8.99 s** | 5 files, in `_phase2_refbuild/dof_lut/ref/` |
| `bounce_tables` | **167.6 s** | 13 files, in `_phase2_refbuild/bounce_tables/ref/` |

```bash
cd ~/notebooks/rubin-work/aos && python code/lut/run_build_lut.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --mi-name pathA_50_34_i_5rot \
    --tables-dir output/fam_processing/danish_1_2 \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/dof_lut/ref
```

```bash
cd ~/notebooks/rubin-work/aos && python code/bounce/run_bounce.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --mi-name pathA_50_34_i_5rot \
    --fits output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/bounce_tables/ref \
    --min-detectors 160
```

**`dof_lut` reference.** 970 of 3,385 visits pass the band/program cut (band `i`,
`BLOCK-T614_triplets`), 863 survive the good-fit cut (63 bad-fit dropped); SVD
`U_eff = (126, 34)`, `n_dof = 50`, `n_keep_eff = 34`, `k = 1..6`.

| file | rows x columns | note |
|---|---|---|
| `lut.parquet` | 50 x 7 | one row per DOF, plus 8 fields of parquet **schema metadata** |
| `lut_dz.parquet` | 126 x 6 | one row per (focal k, pupil j) |
| `lut_by_rotbin.parquet` | 450 x 7 | 9 rotator bins x 50 DOF |
| `lut_config.yaml` | — | frozen provenance |
| `lut.pdf` | 45,415 bytes | |

Averaged DZ: raw RMS 0.0329 µm of wavefront, residual RMS 0.0109 µm of wavefront after
the 34-mode reconstruction. Nine populated rotator bins, centres −60 to +60 deg (OCS) in
15 deg steps, 16 to 254 visits per bin.

**`bounce_tables` reference.** Six legs over seven nights — `T720_elevation` at Elev=30,
40, 50, 60, 75 and `T724_rotator` at Rot=60 — with 7 `kind` values in `bounce_dof_stats`
(`dof`, `dof22`, `dof5`, `dof_oic`, `vmode`, `vmode22`, `vmode5`), so all five recovery
schemes are exercised.

| file | rows x columns |
|---|---|
| `bounce_kj_stats.parquet` | 1,890 x 17 |
| `bounce_dof_stats.parquet` | 2,550 x 24 |
| `bounce_fwhm_metric.parquet` | 6 x 9 |
| `bounce_fwhm_vs_bvalue.parquet` | 9 x 11 |
| 9 PDFs | 84 kB to 4.6 MB |

The reference **independently reproduces the published result**: on the `T720_elevation`
`Elev=40` leg, `fwhm_before = 0.2986 arcsec` and `fwhm_after_50_34 = 0.0381 arcsec`,
which is what `smatrix/docs/studies/regularized_inversion.md:89` quotes ("elevation 40
deg at 0.2986 to 0.0381 arcsec"). A rerun against the committed `_july` build also gave
identical shapes and column lists on all four tables.

**Comparison keys, verified unique rather than assumed** — and the plan-shaped guess
would have been wrong on the first table:

| table | key | unique |
|---|---|---|
| `bounce_kj_stats` | `(bounce, night, comparison, k, j)` | **1,890 / 1,890** |
| the same, without `night` | `(bounce, comparison, k, j)` | 756 / 1,890 — **NOT unique** |
| `bounce_dof_stats` | `(bounce, comparison, night, kind, index)` | **2,550 / 2,550** |
| `bounce_fwhm_metric` | `(bounce, comparison)` | **6 / 6** |
| `bounce_fwhm_vs_bvalue` | `(bounce, comparison, night)` | **9 / 9** |
| `lut` | `(dof_index)` | **50 / 50** |
| `lut_dz` | `(k, j)` | **126 / 126** |
| `lut_by_rotbin` | `(rotbin_center, dof_index)` | **450 / 450** |

No key column holds a null on any table. Two schema facts the comparison has to carry:
`night` is a **string** on `bounce_kj_stats` and `bounce_dof_stats` — with a literal
`'all'` row for the pooled result — and an **int64** on `bounce_fwhm_vs_bvalue`; and
`lut.parquet`'s 8 parquet schema-metadata fields (`param_set`, `mi_name`, `reduce`,
`n_dof`, `n_keep`, `n_visits`, `k_min`, `k_max`) are part of the product and are compared
too. No earlier product's comparison checked schema metadata.

The PDFs are not byte-comparable (matplotlib embeds a creation timestamp), so step 4
compares the parquet tables at zero tolerance and checks only that each PDF is produced
at a comparable size.

**Expected `aos/` job count after the move: 37 → 31**, −3 for `build_lut` and −3 for
`bounce`, since both are in `rule all`. `rule all` loses `output/lut/{d}/lut.parquet` and
`output/bounce/{d}/bounce_kj_stats.parquet` as targets it builds; the `bounce` rule itself
survives as the plotting half, so the drop is accounted rule by rule in step 3.

### `dof_lut` 3a steps 2-4 — the move, verification and comparison: DONE (`ed53b39`)

**The move.** `run_build_lut.py` → `rubinwork/products/dof_lut/builders/`, with a shim at
the old path. New: `variants.yaml` (3 variants), `reader.py`, `__init__.py`,
`write_manifest.py`, `Snakefile`, `compare_builds.py`, `tests/test_dof_lut.py` (36 tests).
`rule build_lut` and its `rule all` target left `aos/Snakefile`; the MIW-skip message no
longer names `build_lut` as a planned study job.

**Each variant names one thing, `miw_variant`.** The reader derives `param_set`, `mi_name`
(through `miw.mi_keys`) and `fam_variant` (through `miw.fam_variant`) from it. Restating
them per entry would have been a second place to edit and could drift from the generated
`mi_config.yaml` silently; a test asserts no entry carries a builder knob. `dir_name` is
quoted and a test checks it is `<fam dir_name>_<miw dir_name>`.

**The manifest records one upstream, `fam_tables`.** The MIW entry is settings only, so it
goes under `config` (with the derived `param_set`, `mi_name`, `fam_variant`) and a test
asserts `miw` is *not* in `inputs` — recording it would claim a dependency the build does
not have.

**Two things the comparison checks that no earlier product's did**, both because the
information lives nowhere else: the 8 parquet **schema-metadata** fields on `lut` and
`lut_dz` (`param_set`, `mi_name`, `reduce`, `n_dof`, `n_keep`, `n_visits`, `k_min`,
`k_max`) and the 11 keys of `lut_config.yaml`. A build with a different `n_keep` matches on
every row while being a different measurement; a test builds exactly that case and
confirms the comparison catches it.

**`lut_by_rotbin.parquet` is conditional** on the fit table carrying a rotator-angle
column. The product Snakefile declares it anyway (5 outputs against the 3 the `aos/` rule
declared, which silently wrote 5), so a variant whose fits lack the column fails loudly
instead of producing a short build. `load`'s default skips a missing one but an explicit
`tables=("lut_by_rotbin",)` raises, so asking for it by name cannot silently return
nothing. The comparison treats "absent from both" as agreement and "absent from one" as a
difference.

**The result.** Reference vs moved, at zero tolerance, both the full production run:

| table | rows x columns | key | result |
|---|---|---|---|
| `lut` | 50 x 7 | `dof_index` | identical, incl. 8 metadata fields |
| `lut_dz` | 126 x 6 | `(k, j)` | identical, incl. 8 metadata fields |
| `lut_by_rotbin` | 450 x 7 | `(rotbin_center, dof_index)` | identical |
| `lut_config.yaml` | 11 keys | — | identical |
| `lut.pdf` | 45,415 bytes both sides | — | same size |

Every selection number reproduced too: 970/3,385 visits passing the band/program cut, 863
used after 63 bad-fit dropped, `U_eff = (126, 34)`, `n_dof = 50`, `n_keep_eff = 34`, 9
populated rotator bins. `aos/` `snakemake -n` 37 → 34 jobs; 222 product tests pass (186 +
36); import smoke test unchanged at 63/63 modules.

**One pre-existing defect fixed in passing.**
`aos/code/processing_compare/compare_fam_processings.py`'s DOF page built
`<output_root>/<param_set>/<mi>/lut/lut.parquet` from the long param_set key — a layout the
tree lost when `dir_name` was introduced, so the page was already broken. It now resolves
old tree then catalog, keyed on the `(param_set, mi_name)` pair, and skips with a message
when a reduction has no `dof_lut` variant. The script's own default baseline,
`danish_1_0`, is such a reduction, so the page stays skipped for a default run — honestly
this time, since that reduction never had a LUT built.

**The misleading comment is corrected.** The per-rotator-bin block claimed "if MI
subtraction leaves rotator-coupled residual"; it described a MIW build, not this script.
Comment only — no output change.

## 3c rebuild feasibility, measured 2026-10-10

### Every collection resolves; the embargo chunk has not moved

`python -m rubinwork.products.fam_tables.refbuild.check_collections` probes each chunk
collection and each `wfs_collections` entry in its own Butler repo and counts datasets of
the aggregate dataset type. **All 15 collections resolve and all hold data.** Counts are
datasets of `aggregateAOSVisitTableRaw` (`donutBlitzFamResults` for the blitz variants),
one per FAM visit, in the chunk's `day_obs` range.

**The 20260713 chunk is still in `/repo/embargo` and is not in `/repo/main`**: 24
`aggregateAOSVisitTableRaw` datasets in `/repo/embargo`, **0** in `/repo/main` on the same
collection name. So the change the plan anticipated is **not needed** — `variants.yaml`
stays as it is, and 3c still needs a repo that can read `/repo/embargo`. Re-check before
the rebuild; the embargo period is what would move it.

`danish_1_2` chunks (FAM), `/repo/main` except where noted:

| chunk | Butler visits | nights | day_obs seen | built in old tree | gap |
|---|---|---|---|---|---|
| 20260315_20260327 | 467 | 8 | 20260315..20260327 | 427 | 40 |
| 20260331_20260409 | 480 | 5 | 20260331..20260409 | 480 | 0 |
| 20260418_20260513 | 219 | 7 | 20260418..20260513 | 219 | 0 |
| 20260514_20260731 | 90 | 5 | 20260521..20260711 | 54 | 36 |
| 20260713_20260713 (`/repo/embargo`) | 24 | 1 | 20260713 | 24 | 0 |
| 20250415_20250531 | 644 | 27 | 20250415..20250531 | 637 | 7 |
| 20250601_20250930 | 812 | 24 | 20250601..20250921 | 811 | 1 |
| 20251001_20251115 | 367 | 21 | 20251022..20251115 | 361 | 6 |
| 20251116_20251130 | 74 | 15 | 20251116..20251130 | 53 | 21 |
| 20251201_20251231 | 346 | 23 | 20251201..20251231 | 319 | 27 |
| **total** | **3,523** | | | **3,385** | **138** |

All counts are visits (count, dimensionless). The 138-visit gap is **expected, not a
defect**: the Butler probe applies no program filter and no quality cut, while the build
applies both. The reference chunk is the clearest case — 74 Butler visits, 53 built, and
the `median_blur_arcsec <= 1.2 arcsec` cut is what drops the rest (it dropped 5 of 5 on
20251116 alone in the timing run below). The two biggest gaps, 40 and 36 visits, are on
the 2026 chunks, which carry `fam_programs` while the probe does not.

`danish_1_2` corner-WFS collections (`wfs_collections`), all `/repo/main`, probed
**unbounded** in `day_obs` — the variant-level `day_obs_min/max` covers only its 2026
chunks, so bounding would have reported `refitWcs_2025` as 0:

| entry | datasets | nights | day_obs seen | dataset type |
|---|---|---|---|---|
| `refitWcs` | 1,183 | 20 | 20260315..20260513 | `aggregateAOSVisitTableRaw` |
| `refitWcs_2025` | 2,263 | 110 | 20250415..20251231 | `aggregateAOSVisitTableRaw` |
| `paired_3mm` | 134 | 2 | 20260315..20260317 | `aggregateAOSVisitTableRaw` |
| `ai_donut` | 1,184 | 20 | 20260315..20260513 | `aggregateZernikesRaw` |
| `tarts` | 1,029 | 17 | 20260316..20260428 | `aggregateAOSVisitTableAvg` |

The blitz variants, `/repo/main`, no `wfs_collections`:

| variant | Butler visits | nights | day_obs seen | built in old tree |
|---|---|---|---|---|
| `danish_1_3_test` | 966 | 15 | 20260315..20260619 | 966 |
| `danish_1_3_v1000` | 966 | 15 | 20260315..20260619 | 966 |

Both match their old-tree builds exactly — the blitz path applies no program filter, and
its quality cuts pass 12 of 12 on the reference night.

### Rebuild time, and the 19 s/visit figure in the plan is wrong

Measured on `sdfiana`, one night of the 20251116 chunk, `--workers 8`, by running
`run_mktable.py` three times:

| run | visits | wall clock |
|---|---|---|
| 20251116, `--no-thermal`, cold caches | 5 | 453 s |
| 20251116, with the thermal loop | 5 | 466 s |
| 20251116, `--no-thermal`, warm caches | 5 | 453 s |
| 20251117, `--no-thermal` | 15 | 845 s |

Cold and warm `--no-thermal` agree to 1 s on the 5-visit night, so there is no cache
effect to confuse the comparison, and the two no-thermal points give a clean linear cost:

- **fixed overhead 258 s per `mktable` invocation** (Butler init, the ConsDB query, the
  thermocouple table, the validation plot) — paid once per chunk, not per visit
- **marginal 39.1 s/visit** of donut reading, which is the dominant term
- **the EFD thermal loop adds 2.5 s/visit**, from 466 s against 453 s on the same 5 visits

**The plan's section 10 said the thermal loop costs about 19 s/visit and 17 min of the
reference's 54 visits. That does not hold:** at 2.5 s/visit it is about 3 min of a 74-visit
chunk, roughly 6% of the chunk's cost, not a third of it. The 39.1 s/visit donut read is
what dominates. Dropping the loop is still worth doing — it is what removes the EFD from
`mktable` and lets the chunk run in batch (plan 3b) — but **it is a batch-eligibility
change, not a speed change**. Section 10 of the plan is corrected to say so.

Cross-check: 39.1 s/visit plus the 258 s overhead predicts 52.6 min for the 74-visit
reference chunk, against the phase 2 reference build's 53 min for the whole chunk
including the DZ fit and the telemetry attachment. Consistent.

Per-variant rebuild estimate, from those rates:

| variant | visits | serial `mktable`, no thermal | with thermal | longest chunk |
|---|---|---|---|---|
| `danish_1_2` | 3,523 | 39.0 h | 41.5 h | 8.9 h (the 812-visit chunk) |
| `danish_1_3_test` | 966 | 1.0 h | 1.0 h | 1.0 h |
| `danish_1_3_v1000` | 966 | 1.0 h | 1.0 h | 1.0 h |

The blitz variants are an hour, not a day, because they do **not** use `mktable`: the A4
reference ran `run_blitz_mktable --fit` over 12 visits in 43 s, i.e. 3.6 s/visit including
the DZ fit, reading one pre-aggregated `donutBlitzFamResults` table per visit instead of
per-donut Butler datasets. 966 visits × 3.6 s/visit = 58 min.

`danish_1_2`'s 39 h is serial CPU across ten chunks. The chunks are independent, so with
`-j` the wall clock floor is the **longest chunk, 8.9 h** for the 812-visit
20250601_20250930 chunk. Memory: the `mktable` rule declares
`mem_mb = 3000 × workers = 24 GB` per chunk, so four concurrent chunks need 96 GB.

Downstream of `mktable`, measured in this session's step-5 rerun of the 53-visit reference
chunk (`-j4`, wall clock 2 min 40 s total):

| rule | wall clock |
|---|---|
| `fit` | 29 s |
| `combine_donuts` | 3 s |
| `combine_visits` | 1 s |
| `combine_fits` | 1 s |
| `attach_telemetry` | 157 s |
| `manifest` | 1 s |

`attach_telemetry` is the only expensive one and it scales per visit: 157 s for 53 visits
is 3.0 s/visit, so about 3 h for `danish_1_2`'s 3,385 built visits. It queries the EFD.

### What can run in batch

- **`mktable` with `--no-thermal`: yes**, once plan 3b lands. It is the long pole (39 h of
  the 42 h), it is embarrassingly parallel over chunks, and `--no-thermal` is what removes
  its EFD dependency. Ten chunks as ten Slurm jobs turns 39 h of CPU into an 8.9 h wall
  clock bounded by the largest chunk, or less if that chunk is split by night.
- **`mktable` with the thermal loop: no.** The EFD resolves only from interactive nodes.
- **`fit` and the three `combine_*` rules: yes.** No EFD, no ConsDB, minutes of CPU. Not
  worth a batch job on their own.
- **`attach_telemetry`: no**, for the same EFD reason, which is exactly why the Snakefile
  has `--config attach_telemetry=0` for batch mode. Run it afterwards interactively: about
  3 h for `danish_1_2`.
- **`manifest`: no**, 1 s, and it must run after `attach_telemetry` or the build is
  manifested without its telemetry columns. The rule already depends on the telemetry
  stamp when `attach_telemetry=1`, so a batch run (`attach_telemetry=0`) writes the
  manifest **without** it. For a two-stage batch-then-interactive rebuild, write the
  manifest only in the second stage.
- **The blitz variants: not worth batching**, an hour each, and they query ConsDB.

**No batch job has been submitted.** Doing so needs Aaron's OK on the exact submit
command, per `CLAUDE.md`.

## Open, not yet resolved

- **`rotTelPos` is not applied yet** (plan 3b, handoff section A1). Until it is, a rebuild
  of `danish_1_2` differs from the old-tree build in `rotator_angle` by up to 0.188 deg of
  camera rotator angle on the visits where ConsDB `physical_rotator_angle` is NULL — 19 of
  them in the reference chunk. Any 3c comparison must expect that.
- **A rebuild of `danish_1_2` loses 362 `cam_*` columns** from `fits.parquet` until the
  sidecar is reworked, because `combine_fits` drops them (phase 2, "Step 4 requirements").
  The plan lists it as an expected 3c difference.
- **`backfill_visit_sides.py` mutates a product table in place**, which rule 5 of the
  plan's section 2 forbids. Decide in 3a/3b whether it becomes a build step or is retired.
  It still targets the old tree, deliberately.
- **3b: fold `d12-refitWcs_2025` into `d12-refitWcs`** as a date-keyed collection list.
  Aaron, 2026-10-10: they are one variant, differing only in the collection that covers
  2025. Needs a builder change — `run_wfs_mktable` opens one Butler per `--wfs-name` —
  so it was out of scope for 3a's mechanical move. The two collections are disjoint in
  `day_obs`, so the split is clean. Doing it also removes the never-built variant that
  forced the `wfs_variants()` filter in step 3.
- **3b: retire `danish_1_0` whole**, both the FAM variant and `d10-wep17_3_0`. Aaron,
  2026-10-10. Four live things resolve the long param_set key and must be handled in the
  same commit: the staged-MIW provenance chain
  (`aos/calibration/miw/intrinsic_split_maps_v1.provenance.yaml`, `stage_miw.py`,
  `calibration/README.md`), `notes/aos-measured-intrinsics/make_figures.py:23` and
  `provenance.md:5`, the **live** `overrides:` block at `aos/analysis_config.yaml:148`,
  and `test_fam_tables.py:263`. Both halves are carried as `registered: false` until then,
  which is what keeps `aos/param_sets.yaml` byte-identical.
- **An incomplete build breaks a plain `snakemake -n` in `aos/` once its producer moves.**
  Met twice now: `refitWcs_2025` cost 5 jobs in `cwfs_tables`, and nine incomplete MIW
  builds plus `tarts`'s missing corner sidecar cost 75 in `miw`. Both fixed by filtering on
  what is on disk and **reporting every skip**. `aos/Snakefile` has the idiom in three
  places to copy. `coadds` did not hit it — the moved script had no rule there — but
  `dof_lut` and `bounce_tables` both have rules in `rule all`, so they will.
  **Largely superseded by `fec3741`**: a variant whose build exists in the NEW tree now
  resolves through the catalog instead of being skipped, so the gates only drop a variant
  that is built in neither tree. The skip-and-report idiom still matters for that case.
- **Nine of the twelve MIW builds are incomplete in the old tree**, so the 3c rebuild has
  more to do for this product than a re-run: six have the split but no sidecar or refit,
  and three (`danish_1_3_test_A_50_34_i`, `danish_1_3_v1000_A_50_34_i`,
  `danish_1_3_v1000_A_50_34_i_rbr`) have nothing but their per-bin grids. Decide in 3c
  which are wanted — several are parents that exist only to be `build_from` sources, which
  is a legitimate reason to have grids and nothing else.
- **`refit_mi` reads the whole FAM `donuts.parquet`**, 12.36 GB and 9,083,526 rows for
  `danish_1_2`, at about 89 s per 475,128 donut rows measured on the reference — so roughly
  **28 min per MIW variant** for the full table, before `intrinsic_sidecar`'s share. Twelve
  variants is a 3c cost worth planning rather than discovering.
- **The plan's 24-character variant-name limit is raised to 32** (section 5), because
  `d13v1000-A_50_34_i_rbr_5rot` and `d13v1000-A_50_50_i_rbr_5rot` are 27 and have to
  round-trip to their `dir_name`.
- The timing runs wrote throwaway output to
  `_phase2_refbuild/_timing_step5/{nothermal,thermal,n15}/`. Not deleted — file deletion
  needs Aaron's OK.
- 3a left throwaway and reference directories under `_phase2_refbuild/`, none deleted
  (file deletion needs Aaron's OK):
  - `cwfs_tables/`: `_fam_20251126` the staged FAM night, `refitWcs_2025_20251126` the
    reference, `refitWcs_2025_20251126_moved` the rerun.
  - `miw/`: `_fam3n` the staged three-night FAM slice (507 MB), `ref` the reference, `moved`
    the rerun, `mi_config_ref.yaml` the throwaway config, and
    `_fam_20260315_20260317/` — the **first, unusable** staged slice, whose single-row-group
    donuts.parquet the builder cannot read. That one is pure waste; the rest should stay
    until 3c.
  - `coadds/`: `_fam_20260316` the staged one-night FAM slice (129 MB), `ref` the
    reference (5.4 MB), `moved` the rerun (5.4 MB).
  - `_scratch_3a/` the scratch data root, now holding placeholder `fam_tables`,
    `cwfs_tables`, `miw` and `coadds` manifests used only to plan the product Snakefiles
    and to show the `aos/Snakefile` resolver picking up a new-tree build. Written by
    `aos/code/infra/stage_scratch_product.py`, which refuses any root not under a
    `_scratch_3a` directory.
- `rubinwork/products/miw/tests/__init__.py` was added, which
  `rubinwork/products/{cwfs_tables,coadds}/tests/` do not have. Harmless — pytest collects
  either way — but inconsistent; remove it when a deletion is being approved anyway.
- **`block_grids_ext.npz` and `coadd_metrics.parquet` have no reader at all.** Kept in the
  product on Aaron's call 2026-10-10 — the former is the extra-pupil-Zernike primitive the
  retrieval-bias work needs and is what distinguishes the two coadd variants; the latter is
  the natural per-coadd table, and the only metrics file anything reads
  (`coadd_metrics_rebin3.parquet`) is a **study** output of `recompute_coadd_metrics.py`.
  Worth revisiting in 3c: a full build's `block_grids_ext.npz` is 66 MB that nothing opens.
- ~~**`build_lut` reads its `fits.parquet` from the FAM tables, not from the MIW refit.**~~
  **Resolved.** Aaron confirmed on 2026-10-11 that the plain FAM DZ is what was used and
  that he may want to switch to the post-MIW DZ later; recorded, not changed. The
  `dof_lut` move encodes it: `fam_variant` is the manifest's upstream, the `miw` entry is
  settings only and sits under `config`, and switching would make `miw` the upstream — a
  variant-definition change. See `aos/docs/studies/lut.md` and the product's `__init__`.
- **The coadd variants' `_v2` suffix is carried by `ext_terms` alone.** If
  `block_grids_ext.npz` is ever dropped the two variants become identical and should
  collapse into one.
- **Two coadd study scripts still hardwire `output/coadd/danish_1_2/50_34/`** as a
  `--blocks-summary` default (`analyze_dz_goodness_of_fit.py:293`,
  `analyze_umode_null_coupling.py:68`). Left alone deliberately: that path **exists** and
  works today, and both take the flag explicitly, so a catalog build is reachable by
  passing its path. They need the catalog only when phase 5 retires the old tree.
- **`analyze_miw_field_order.py` has no argument parser** — 110 lines, no functions, it
  runs top to bottom and reads at module level, so merely importing it needs the products
  on disk. Its builds are named by `COADD_VARIANT`, `COADD_MIW_VARIANT` and
  `COADD_STUDY_DIR` because there was nowhere to put a flag. Give it a `main()` when the
  coadd study is reshaped in phase 5.
