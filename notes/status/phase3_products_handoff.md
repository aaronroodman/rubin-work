# Phase 3 handoff: the other products — move, fix, rebuild

> **Status:** 3a and 3b complete; **3c IN PROGRESS**, 5 of 8 comparison-plan items done.
> `fam_tables`, `cwfs_tables` and `miw` are all DONE, compared clean against the old tree
> and `current`: `fam_tables` at `danish_1_2@20261014` and `danish_1_3_{test,v1000}@20261013`,
> all four `cwfs_tables` variants at `20261013`, and both `miw` `d12` variants at
> `20261014` — `d12-A_50_34_i_5rot` promoted 2026-10-15, which is the variant items 5 to 7
> read. Every difference found is on the expected-difference list (now 8 entries); the
> M1M3 gradient defect is fixed and promoted. The MIW calibration `intrinsic_zernikes`
> **v4** is BUILT and Butler-ready at `$RUBINWORK_DATA/calibration/intrinsic_zernikes/v4/`
> (207 files + QA PDF), deliberately **not ingested** and **not staged into git**. What
> remains is items 5 to 7 — `coadds`, `dof_lut`, `bounce_tables`.
> **The stack and package upgrade is COMPLETE, so items 5 to 7 are unblocked.** The live
> weekly is **`w_2026_42`** / `lsst-scipipe-14.0.0-exact` — **42, not the 41 decision 8
> named** (decision 9, Aaron, 2026-10-15; `~/u/LSST/setup.sh` reads `w_2026_42`, and Aaron
> made that edit and the restart himself). Steps 1 to 5 are all done:
> `ts_intrinsic_wavefront` committed, fast-forwarded onto `tickets/RSO-809`, tagged
> `rubinwork-2026-10-15` and pushed; every manifest records the stack; the `w_2026_39`
> reference `dof_lut` built and checksummed; `ts_wep` pulled to `0363c0c4` (28 commits, not
> the 9 the inventory says); and on `w_2026_42` all twelve `setup -r` packages set up, the
> five imports succeed, `pytest rubinwork` gives **361 passed** (not 358 — `ff2a8dd` added
> 3) and the `dof_lut` smoke-test rebuild `20261015w42` is **bit-identical** to the
> `w_2026_39` reference, all three md5 checksums equal. **No library moved from 41 to 42**,
> and neither scipy 1.18.1 nor the 28 `ts_wep` commits reaches the `dof_lut` path, so **the
> `ts_wep` rollback test was not needed and must not be redone**. See "Upgrade step 5".
> **The `ts_intrinsic_wavefront` draft PR (`tickets/RSO-809` into `develop` — base is
> `develop`, NOT `main`) is still not open, and it is Aaron's to open**, `gh` being
> unauthenticated on S3DF; the ready-to-run command and body file are in the step 1 section.
> This file was pruned 2026-10-14; the 3a/3b narratives are in commit `ebbc426`.
> · **Last updated:** 2026-10-15 · **Kind:** working state (handoff)

Phase 3 of `notes/status/organization_plan.md`: move every remaining product's code (3a),
apply the intended fixes (3b), then rebuild every product fresh into the new tree (3c).
Read the Phase 3 part of section 8 there for the step list; this file holds the state
between sessions.


> **History pruned 2026-10-14.** The full 3a per-product move narratives (scope, variant derivations, steps 1-4 for every product) and the per-item 3b records are in commit `ebbc426f9d8077e4b9ef475d147dcb5904e99311`, this file at 4,315 lines. Everything a session needs to resume 3c is kept below: decisions, open defects, the expected-difference lists, the comparison plan, the finished-product results, and the variant tables for the products 3c has still to rebuild.

## Decisions for Aaron

Per section 12 of the plan: questions needing your judgment that do not block the rest of
the step. Answered items move to "Decisions taken".

8. **The package and stack upgrade plan — waiting on your OK** (new 2026-10-15; **items 5
   to 7 are sequenced behind it**). The inventory and the proposed order are in "The
   package and stack inventory" and "The upgrade plan" below; nothing in either was
   executed. The three calls that need you:
   - **`w_2026_40` or `w_2026_41`?** 41 is recommended, but it is a conda-env major bump
     (`lsst-scipipe-13.1.0-exact` to `14.0.0-exact`, scipy 1.16.3 to 1.18.1). 40 stays on
     the current env.
   - **Commit and tag `ts_intrinsic_wavefront`** on `tickets/rubinwork-rottelpos`, so a
     future calibration's `git_sha` is not `-dirty`. It is the only dirty package, one
     file, a self-contained bugfix. This would be a commit to an **external** repo, so it
     needs your word.
   - **Pull `ts_wep`'s 9 pending `develop` commits, before or after the stack move?** The
     plan argues after, as a separate attributable step.

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

**Decision 10 (Aaron, 2026-10-15) — telemetry comes from the value-added catalog, and
per-product re-derivation is the thing to get away from.** Two parts:

1. **For item 5 now:** the ESS temperature and M1M3 gradient differences in
   `coadd_metrics.parquet` are **expected**, not a stop. They are differences 6 and 8
   reaching a new product, not new findings. Recorded against item 5 so the comparison is
   not blocked on them.
2. **As a direction for later:** Aaron's words — "I should have all the relevant
   quantities in the value-added catalog, and I want to get away from having to find and
   refind those telemetry values for each product." The value-added database is the
   intended single source for per-exposure telemetry; products should **read** it rather
   than each re-querying the EFD/ConsDB and each acquiring its own copy of the same
   columns with its own window and its own bugs.

**Why this matters beyond tidiness:** expected differences **6** and **8** both exist
*because* the telemetry is re-derived per product. Difference 6 was a wrong-gradient
defect fixed by joining from `visit_telemetry` instead of recomputing; difference 8 is an
EFD query window that silently narrowed, costing ~9-10% of visits their ESS temperatures
in every FAM-derived product. A single read of the value-added catalog removes the class.

**Not scoped or scheduled**, and deliberately **not** done inside 3c — 3c is a
move-and-rebuild phase, and changing where a product reads its telemetry changes its
values. Item 8 (`value_added`) is where the database itself is addressed. Candidate
follow-up work for after phase 3, to be sized then: audit which products re-derive which
telemetry columns, then convert them to read `value_added` through
`rubinwork.products.catalog`.

**Decision 9 (Aaron, 2026-10-15) — the weekly is `w_2026_42`, not `w_2026_41`.** This
amends decision 8's choice of weekly; everything else in decision 8 stands. `w_2026_42`
was already on cvmfs when step 4 validated 41, and Aaron took the newer one rather than
adopt a weekly that was already superseded. `~/u/LSST/setup.sh` line 3 now reads
`weekly_version=w_2026_42` — **Aaron made that edit himself**, and started the session from
a clean login shell, so the switch step 4 left open is **done**. The conda env is
unchanged at `lsst-scipipe-14.0.0-exact`, and **every library version on 42 is identical to
the ones measured on 41**, so the step 4 validation carries over rather than being redone
from scratch. See "Upgrade step 5" for the validation and the smoke test on 42.

**Decision 8 (Aaron, 2026-10-15) — the stack and package upgrade is AUTHORIZED, with
these choices.** This supersedes the "PROPOSED, NOT EXECUTED" framing of the upgrade plan
below; the plan's step 4 (ts_wep separately from the stack) is explicitly overridden.

- **Weekly: `w_2026_41`**, the `lsst-scipipe-14.0.0-exact` env. The conda major bump and
  scipy 1.16.3 to 1.18.1 are taken deliberately now.
- **`ts_intrinsic_wavefront`:** commit the dirty file, roll **all** of Aaron's rubin-work
  changes into his existing `tickets/RSO-*` branch, tag, push, and open **one draft PR**
  from that branch. Aaron authorizes the push and the draft PR **for this repo only**.
- **`ts_wep`:** pull `develop` and move the stack **in one step, not two.** The plan's
  two-step attribution is given up on purpose; if the smoke test then differs, re-run with
  `ts_wep` back at `1ea1ae1d` to separate the two causes.
- **`summit_utils` and `ts_ofc`:** no PRs now; later.
- **Manifest first:** `rubinwork/products/manifest.py` records the stack before any
  rebuild, and a reference `dof_lut` build is made on `w_2026_39` first so the smoke test
  has something to compare against at zero tolerance.

Answered 2026-10-15, closing out `miw` and setting up the stack upgrade:

- **No Butler ingest of `intrinsic_zernikes` v4 for now.** Aaron may do it separately
  soon. v4 stays in the data tree only and is **not staged into git**. The dry-run
  `ingest` command is kept in the v4 section below for whenever that happens; nothing has
  touched `/repo/main`.
- **`miw/d12-A_50_34_i_5rot` is the variant for downstream work** and was promoted to
  `current@20261014` on 2026-10-15 (A4 done). Confirmed through `catalog.load()`:
  `fits.parquet` reads 3,385 rows (visits) x 830 columns through `current`, and
  `catalog.list_products('miw')` shows both `d12` variants `current` and `complete` at 6
  files each. `dof_lut`'s Snakefile now builds a DAG — `snakemake -n` plans 3 jobs
  (`build_lut`, `manifest`, `all`); it previously failed at `Snakefile:95`.
- **Items 5 to 7 rebuild after the package and stack update, not before**, so all three
  run on one environment. See "The package and stack inventory" and the upgrade plan
  below.
- **Two variants registered from the old tree instead of rebuilt** (A5's recommendation):
  `coadds/d12-A_50_34_i_5rot_v2@old-tree-20260903` and
  `bounce_tables/d12-A_50_34_i_5rot_july@old-tree-20261001`. Each manifest carries a
  `location` pointing into `aos/output/` and a `config.description` recording why it was
  registered rather than rebuilt. **No file was copied or moved, and neither is
  `current`.**


Answered 2026-10-14, both on `miw` item 4:

- **Question 6, `d12-tarts`'s missing intra-focal centroids: option (a)** — `d12-tarts`
  is dropped from the `miw` corner-sidecar set, by the Snakefile's existing
  skip-and-report idiom. Nothing is lost: neither the old tree nor the 3a reference has a
  `tarts` sidecar, `(day_obs, seq_num, detector)` is already unique on all 7,813 rows so
  the centroids were only ever a key tie-breaker, and the planned TARTS-vs-Danish CWFS
  comparison reads the donut tables. See "`d12-tarts`: SETTLED" below.
- **Question 7, the ring continuation: option (a)** — the NaN-rim fill of
  `ts_intrinsic_wavefront` commit `3f66904` is **accepted new behaviour**, now expected
  difference 7 of the comparison plan: field-edge cells at **r >= 1.729 deg of field
  angle** may differ, with the interior bounded at **9.2e-04 µm of wavefront**. Whether
  the edge ring itself is trustworthy is the `miw_hull` study's question, not a 3c
  blocker.

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

### The two per-night `mktable` defects: both SETTLED

Defect A (the `< 5 visit_pairs` filter discarding sparse nights when run per night) and defect B (the `Snakefile`'s 3 GB/worker `mem_mb` figure, 3x too low, which OOM-killed 8 donut workers against a 32 GB request) are both fixed, and the full narratives — the measurements, the 118-of-137 failed submission, and defect A's verification against the old tree — are in commit `ebbc426f9d8077e4b9ef475d147dcb5904e99311`. Defect A was not a data loss. Defect B's fix is the corrected `mem_mb` in the build commands below.

## Done and committed

The 3a and 3b commit list, with the per-commit notes, is in commit `ebbc426f9d8077e4b9ef475d147dcb5904e99311`. 3c's own commits are recorded per product below. The settled negative result on `decomp_path` for `value_added` (no registered variant ever went through the broken default, so nothing is mis-attributed) is in the same commit, and nothing waits on it.

## The per-night `mktable` set is COMPLETE, 2026-10-13

All 137 nights accounted for; the set reproduces the old tree's night and visit counts exactly (3,385 visits over 106 nights with tables). Submission job IDs and the per-night accounting are in commit `ebbc426`.

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
## The three-way gradient check: the NEW build was the outlier, 2026-10-13

**RESOLVED by the fix of 2026-10-13 (commit `bef17ba`) and build `danish_1_2@20261014`.**
The verdict, which is what later comparisons need:

- The `20261013` gradients are **wrong** and the old tree's are right: RMS(new − DB) 2.7e-03
  to 1.3e-02 degC/m against RMS(old − DB) 5.0e-05 to 3.0e-04 degC/m; 1,373 visits of 3,006
  where old agrees with the database and new does not, **0** the other way.
- **The mechanism is the thermocouple count, not noise.** All 1,467 visits where the three
  sources agree have `m1m3_n_sensors = 146` (count of thermocouples, dimensionless); every
  one of the 1,539 where the new build disagrees has fewer (median 125 of 146).
  Spearman rho(`m1m3_n_sensors` [count], max|new − DB| [degC/m]) = **−0.859**, p < 1e-300,
  n = 3,006.
- Every ConsDB-transform telemetry column is **bit-identical** between the builds, so only
  the one live-EFD path carries this.

The full investigation — the three-source join tables, the five-night refetch, the
span-width test, and the four hypotheses tested and excluded (fetch span, non-monotonic
`np.interp` input, span-bound indexing, per-night vs per-chunk looping, a silently failed
fetch) — is in commit `ebbc426f9d8077e4b9ef475d147dcb5904e99311`. Do not redo it.

**Why, in one paragraph.** `fit_plane_gradients`
(`ts_m1m3_utils/.../thermal_gradients.py`) drops non-finite thermocouples per time row and
fits a plane anyway as soon as more than 4 survive, so a 30-second bin with a reduced
sensor set is fitted but badly constrained. On night 20250706 refetched over its true span,
bins with 146 finite thermocouples give `y_gradient` about **−0.050 degC/m**, steady, while
bins with 109–110 give **−0.125 or +0.050 degC/m** — sign flips and 2.5x excursions. The
reduced-sensor bins interleave with good bins, so a visit's interpolated value depends on
which bins bracket it. That is the whole 1,539-visit effect.

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

## `fam_tables` `danish_1_2@20261014` BUILT, CHECKED and PROMOTED, 2026-10-14

The staged rebuild ran in **about 8 minutes** of wall-clock time, 6 of 6 snakemake jobs,
exit 0, no `mktable` and no `fit`. The acceptance check printed
**`RESULT: gradients OK, other columns OK`**:

| check | result |
|---|---|
| four gradient columns vs the value-added database | 3,006 visits each, **max abs difference 0.000e+00 degC/m**, exact; NaN 379 in build and 379 in database on all four |
| `visits` vs `20261013` | 3,385 vs 3,385 rows, 405 columns, **0 non-gradient columns differing**; 4 gradient columns differ, max abs **6.681e-01 degC/m** |
| `fits` vs `20261013` | 2,528 vs 2,528 rows, 830 columns, **0 non-gradient differing**; 4 gradient, max abs **6.681e-01 degC/m** |
| `donuts` vs `20261013` | 9,083,526 vs 9,083,526 rows, 48 columns, **0 differing** |

Every difference is expected difference 6 of the comparison plan and nothing else, so the
comparison is **clean** under section 12 and `catalog set-current fam_tables danish_1_2
20261014` ran: `current -> 20261014`. Confirmed through the catalog —
`catalog.load('fam_tables', 'danish_1_2', file='visits.parquet')` gives **3,385 visits
over 405 columns** with the four gradient columns present.

`fam_tables` is now **DONE** for 3c: `danish_1_2@20261014` and
`danish_1_3_{test,v1000}@20261013` are all `current`.

## `miw` item 4: BUILT but the comparison is NOT clean — STOPPED, 2026-10-14

**`d12-A_50_34_i` build `20261014` is built as far as the sidecars and NOT promoted.**
15 of 18 snakemake jobs done: all 9 `build_intrinsic` grids, `intrinsic_split`, the FAM
`intrinsic_sidecar` and 3 of 4 corner sidecars. **No `set-current` has run for any `miw`
variant**, and `d12-A_50_34_i_5rot` was not started.

**Two blockers, in the order they matter.**

**1. `d12-tarts`'s corner sidecar fails, so `refit_mi` never ran and `fits.parquet` is
missing.** `run_make_intrinsic_sidecar.py` raises
`pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(centroid_x_intra)` — the
`cwfs_tables` `d12-tarts` donut table has no intra-focal centroid columns. This is the
**pre-existing `tarts` missing-corner-sidecar defect** already in "Open, not yet
resolved", surfacing now because the `miw` Snakefile builds a corner sidecar for every
`cwfs_tables` variant of the FAM variant. Neither the old tree nor the 3a reference has a
`tarts` sidecar to compare against: the old tree has **no** `wfs/` sidecars at all for this
variant, and the 3a reference built only `refitWcs`. Decide whether `d12-tarts` is excluded
from the corner-sidecar set or the intra centroids are added to it.

**2. The field-map grid gained 154 valid cells, which is a change in the EXTERNAL builder
and is not on the expected-difference list.**

| table | old rows | new rows | columns | differences |
|---|---|---|---|---|
| `intrinsic_split_maps` | 3,831 | **3,985** cells of the same 71x71 grid | — | row-count MISMATCH, so `compare_builds` compared 0 columns and its `values: identical` is vacuous |
| `intrinsic_split_decomp` | 21 | 21 | 10 | `O_re` max abs **0.8405 µm of wavefront** (19.7x the map's own RMS of 0.0427 µm), `O_im` **0.4865 µm** (11.2x), `C_re` **1.007 µm** (50.1x), `C_im` **1.103 µm** (66.1x) |
| `intrinsic_split_rms` | 21 | 21 | 8 | **4 columns only in the new build**: `n_null_rings`, `n_continued_rings`, `min_ring_coverage_seen`, `max_null_dim`. Values: `O_tel` max abs difference **2.524e-05 µm of wavefront** (max value 0.0895 µm), `C_cam` **5.484e-05 µm** (max 0.0305 µm), `residRMS` **4.605e-05 µm** (max 0.0767 µm), `dataRMS` **5.031e-16 µm** |
| `build/rot_*/dz_fits`, 9 bins | match | match | 683 | 677 columns differ, but every DOF and `dz_corr` difference is float round-off (max abs **4.2e-11 µm** on `dof_M2_dy`). The real one is **`vmode_13`, max abs 0.431 to 1.018 (dimensionless v-mode amplitude) across the 9 bins** |
| `build/rot_*/intrinsic_grid`, `intrinsic_cov_edge` | match | match | 9, 4 | round-off only (max abs **9.3e-15**) |
| `zk_intrinsic` | 3,065,249 | 9,083,526 donut rows | — | row-count MISMATCH; the old sidecar covers a third of the new donut table, which now has all 9,083,526 rows of `fam_tables@20261014` |

**Why this is a stop, not an expected difference.** The comparison plan's difference 2
predicted a CCS-only change of **7.8e-05 to 9.2e-04 µm of wavefront** with OCS at most
**7e-06 µm**, because O is theta-independent. Measured, **O changes too** and by 0.84 µm.
The 4 new `rms` columns name the mechanism — the external builder's **ring continuation**
now keeps rings the old build nulled, which is what adds the 154 cells — so the
`O`/`C` max-abs figures are most likely a **mask-alignment artifact** of comparing
flattened maps over different valid-cell sets, not a physics change. The summary RMS
columns agreeing to **5.5e-05 µm of wavefront** supports that. **It is a hypothesis, not a
result**: the comparison must be redone cell by cell on the shared (`thx_deg`, `thy_deg`)
cells before any `set-current`.

Likewise `vmode_13`: it is one v-mode of the recovered state, differing by up to 1.0
(dimensionless amplitude) on bins whose DOF agree to 4e-11 µm, which is the signature of a
near-null mode rotating within a degenerate subspace rather than a changed fit. Also
unverified.

**Next, before any `miw` promotion:** compare `intrinsic_split_maps` restricted to the
3,831 shared cells, column by column (the pattern the `cwfs_tables` 2025/2026 split used);
compare `zk_intrinsic` on the shared donut key; settle `d12-tarts`; and decide whether the
ring-continuation change is an accepted new behaviour of the external builder, which makes
it an expected difference to be written down, or a regression.

## `miw` validation output for Aaron's review, 2026-10-14

Validation only: **nothing promoted, no `set-current`, neither open question decided.**
Script `rubinwork/products/miw/refbuild/compare_old_tree.py`. Output, outside every build
directory and outside the old tree, one PDF plus `summary.md` each:

- `$RUBINWORK_DATA/studies/phase3c_validation/miw/d12-A_50_34_i/` (13-page PDF)
- `$RUBINWORK_DATA/studies/phase3c_validation/miw/d12-A_50_34_i_5rot/` (12-page PDF)

`d12-A_50_34_i_5rot@20261014` was built through `intrinsic_split` only, in 1m40s of
wall-clock time, by calling `run_intrinsic_split.py` directly with `--src-dir` at the
parent build. The Snakefile route refuses it: `_5rot` is a `build_from` variant and
`catalog.path` requires the parent to have a `manifest.json`, which
`d12-A_50_34_i@20261014` does not have because it stopped before `refit_mi`.

**The shared-cell result: `compare_builds`'s `intrinsic_split_maps` row key does not
join.** The two builds lay their 71x71 lattice on different coordinates — old half-width
`126/73 = 1.7260274 deg` of field angle, new `1.726 deg` exactly, offset at most
**2.740e-05 deg**, a thousandth of the 0.0493 deg cell step. An exact float key matched
**1 of 3,831 cells**, which is why the comparison reported 0 columns and a vacuous
`values: identical`. On the **integer lattice index** the join is exact: **3,831 shared,
154 new-only, 0 old-only** — a clean superset.

| quantity | `d12-A_50_34_i` | `d12-A_50_34_i_5rot` |
|---|---|---|
| shared / new-only / old-only cells (dimensionless) | 3,831 / 154 / 0 | 3,985 / 0 / 0 |
| max abs map difference, all columns (µm of wavefront) | 7.601e-01 | 8.163e-01 |
| RMS map difference, all columns (µm of wavefront) | 9.543e-03 | 9.481e-03 |
| max abs difference, interior r < 1.650 deg (µm of wavefront) | 7.179e-04 | 1.292e-03 |
| max abs difference, r >= 1.729 deg (µm of wavefront) | 7.601e-01 | 8.163e-01 |
| NaN map cells over all columns, old -> new (dimensionless) | 42 -> 0 | 888 -> 0 |
| NaN cells in `Z4_OCS`, old -> new (dimensionless) | 1 -> 0 | 124 -> 0 |
| `vmode_13` max abs difference (dimensionless amplitude) | 1.018 | 0.653 |
| worst DOF difference on those same bins (µm) | 5.78e-11 | 5.78e-11 |
| `intrinsic_split_rms` `C_cam` max abs difference (µm of wavefront) | — | 1.599e-05 |

**What this settles, and what it does not.** The big differences are **entirely at the
field edge**: the interior agrees to 7.2e-04 µm (9-bin) and 1.3e-03 µm (`_5rot`), against
difference 2's CCS bound of 9.2e-04 µm, while r >= 1.729 deg reaches 0.76 to 0.82 µm — a
factor 632 to 1,059. So the earlier worry is answered: the 0.84 µm `O_re` figure was a
**flattened-map artifact of unjoined cell sets**, and the real interior change is at the
expected scale. `vmode_13` is confirmed a **degenerate-subspace rotation**, not a changed
fit — it moves by up to 1.018 on bins where every DOF agrees to 5.8e-11 µm.

**The cause is one commit, and Aaron's hypothesis needs one correction.** `intrinsic_split.py`
changed exactly once between the old builds (2026-07) and these:

- **`3f66904`** (2026-10-05) *Fix the OCS/CCS null space at the field edge in the intrinsic
  split* — stops skipping under-determined outer rings, continues rings below
  `min_ring_coverage` radially, assigns unconstrained singular directions by
  `degen_assignment` instead of 50/50, and adds the 4 `intrinsic_split_rms` columns.

It is a **rim / NaN-ring fix, not a convex-hull or CWFS-outside-hull extension.** The
evidence: the old `_5rot` map carries **124 NaN cells in `Z4_OCS`**, exactly the "124
points" annulus at `r >= 1.729 deg` the commit message names, and the new build has 0. The
CWFS corners sit near 1.19 deg of field angle, already well inside the old valid region, so
nothing here makes MIW newly available outside a hull. The 9-bin variant's cell-count
growth is **two effects stacked**: a lattice change (3,831-cell `126/73` grid -> 3,985-cell
`1.726` grid, which `_5rot` already had in July) plus the rim fill.

Still Aaron's call, unchanged: whether the ring continuation is accepted new behaviour for
the expected-difference list or a regression, and what to do about `d12-tarts`.

## `d12-tarts`: SETTLED — no MIW sidecar, by design (Aaron, 2026-10-14)

**Decision: `d12-tarts` does not need an MIW corner sidecar at all.** Commit `b8ae9aa`
skips every `reader: unpaired` CWFS variant in the `miw` Snakefile's `CWFS_VARIANTS` loop
and prints why. The Snakefile comment already said "the **paired** CWFS variants"; only
the filter was missing.

**Why it was never a physics problem.** `d12-tarts` is TARTS's **in-focus** analysis
(`u/peterma2/tarts_fam_acq_stability_infocus`, `aggregateAOSVisitTableAvg`) — one
ML-combined row per corner half-sensor from a single exposure, 7,813 rows over 977 visits
x 8 half-sensors. No intra/extra-focal pair exists, so `centroid_*_intra` is not missing
by accident; it is meaningless for in-focus data.
`run_make_intrinsic_sidecar.py` requests those columns unconditionally (line 115) and dies
with `ArrowInvalid: No match for FieldRef.Name(centroid_x_intra)`.

The centroids are **only a row key**: read at line 115, carried untouched, written back at
lines 213-214. The reconstruction uses `thx_OCS`/`thy_OCS` and `rotator_angle` and nothing
else. The extra centroids are in the key only to separate FAM donuts that share a detector
in one exposure; on `d12-tarts`, `(day_obs, seq_num, detector)` is **already unique on all
7,813 rows**. So "add the intra centroids" would have invented a meaningless column to
rescue a key that needs no rescuing.

Nothing reads such a sidecar either: `d12-tarts` donuts carry their own
`zk_intrinsic_OCS`, and neither the old tree nor the 3a reference ever built one. Aaron's
planned **TARTS-vs-Danish CWFS comparison** reads the donut tables, not a sidecar, so the
exclusion does not foreclose it.

**The alternative, not taken and still available:** make the sidecar runner's key adapt —
include the centroid columns only when present, else fall back to
`(day_obs, seq_num, detector)`. That fixes every unpaired variant but needs a change in
`ts_intrinsic_wavefront`, which (b) keeps off phase 3c's critical path.

## `miw` MIW VALIDATED by Aaron, 2026-10-14

**Aaron reviewed the validation PDFs and accepted the `danish_1_2` MIW build**: "almost
identical to the previous version, with the only change being the wider convex hull
replacing some outlier and NAN values."

**One correction to carry onto the expected-difference list.** The mechanism is a **NaN-rim
fill, not a wider convex hull**. The field extent does not grow: the valid-cell mask shows
**old-only = 0 cells**, and the added cells are rim points of the *same* 71x71 lattice.
The old `_5rot` map holds **124 NaN cells in `Z4_OCS`** — exactly the "124 points" annulus
at `r >= 1.729 deg` that commit `3f66904` names — and the new build holds **0**. The CWFS
corners sit near 1.19 deg of field angle, already well inside the old valid region, so no
MIW becomes newly available outside any hull. The acceptance stands; only the label
differs.

## A1 and A3 DONE, and the MIW calibration v4 is BUILT, 2026-10-14

Aaron gave advance permission for the batch submission after reviewing the A1-A5 assessment
below, and asked for Butler-ready output plus the calibration-ready product.

**A1 — `refit_mi` for `d12-A_50_34_i_5rot@20261014` is COMPLETE.** Slurm job **40830071**,
elapsed **00:38:38** on 8 CPUs, peak RSS **11.1 GB** of the 96 GB requested. All 7 jobs ran.
`fits.parquet` is **3,385 rows x 830 columns** — identical shape to the sibling
`d12-A_50_34_i` — and `manifest.json` is written, so `catalog.list_products('miw')` now
reports **both `d12` variants `complete`**. The manifest records `git_commit 3a4ae67`,
`git_dirty false`, inputs `fam_tables: danish_1_2@20261014` and parent `miw:
d12-A_50_34_i`, and 9,083,526 donut rows in `zk_intrinsic.parquet`.
**No `set-current` was run** (hard stop), so `_5rot` still has no `current` symlink.

**A3 — the `_5rot` `fits.parquet` comparison is CLEAN.** New `20261014` (3,385 rows over
106 nights) against the old tree `aos/output/miw/danish_1_2_A_50_34_i_5rot/fits.parquet`
(1,126 rows over 19 nights); 456 shared columns, 374 new-only, 197 old-only — the same
schema split as the sibling. **All 1,126 old visits are present, 0 old-only keys.**

| column group | cols | differing | max abs difference | NaN-status mismatches | expected difference |
|---|---|---|---|---|---|
| `image_idx` (row-order index, dimensionless) | 1 | 1 | 2202 | 0 | 5 (known, unfixed) |
| `rotator_angle` (deg of camera rotator) | 1 | 1 | 0.4446 | 0 | 1 |
| ESS temperature (degC), 9 columns incl. the 2 `cam_*` | 9 | 9 | 0.2281 | 999 | **8** |
| M1M3 thermal gradients (degC/m) | 4 | 4 | 0.005892 | 0 | 6 |
| DZ fit coefficients (µm of wavefront; `*_scale` dimensionless) | 422 | 420 | 0.001803 | 0 | 2 |
| pointing / visit metadata (`visit`, `ra`, `dec`, `az`, `alt`, `mjd`) | 6 | **0** | 0 | 0 | — |
| fit quality (`n_donuts_1/2`, `n_detectors`, `n_detectors_with_min_donuts`, `bad_fit`, `median_blur_arcsec`) | 6 | **0** | 0 | 0 | — |

The headline numbers match the sibling almost exactly (`rotator_angle` 0.4446 deg, ESS
0.2281 degC, gradients 0.005892 degC/m), as they should: those come from the shared FAM
input, not from the MIW variant. **Every difference is on the expected list.**

**Expected difference 8 is CONFIRMED on a second, independent variant.** Here
**111 of 1,126 visits** lose their ESS temperatures and **0 gain them**, the same 111 on
every one of the 9 columns, over 16 of 19 nights (999 column-instance mismatches = 9 x 111).
The sibling showed 89 of 960, same one-directional pattern — 9.9% against 9.3% of visits.
Two variants with the same signature make this a systematic EFD-window change, not a
one-build artifact. Still not traced to a commit.

### The MIW calibration product: `intrinsic_zernikes` **v4** is BUILT (Butler-ready, NOT ingested)

Aaron's "calibration ready output" = the MIW as an AOS calibration product, and the
toolchain for it already exists in **`ts_intrinsic_wavefront`** — nothing new was written:

1. `bin.src/run_make_calib_tables.py` — MIW maps -> `lsst.ip.isr.IntrinsicZernikes` source
   tables + `provenance.yaml`
2. `bin.src/plot_calib_tables.py` — QA PDF
3. `bin.src/ingest_calib_tables.py` — Butler ingest as `intrinsicZernikes`, dimensioned
   `(instrument, detector, physical_filter)`. **Defaults to `--dry-run`.**

**The Butler MIW already in use is Guillem's v3**, recorded in this repo as the external
variant `official-gmegias-v3`: repo `/repo/main`, dataset type `intrinsicZernikes`,
collection `u/gmegias/calib/DM-55048/intrinsicZernikes.v3`, 1,230 datasets over six bands,
ticket DM-55048. So v4 is a **new version of an existing dataset type**, not a new design.
Version numbering: v1 is in `aos/calibration/miw/`, v2 is in Guillem's output root, v3 is
the Butler collection — hence **v4** (Aaron's call; do not reuse v2 or v3).

**Built to v2's exact recipe**, read off v2's own `provenance.yaml` rather than guessed:
`--fill-ocs-from-batoid` with `Rubin_v3.14` band `i` `--batoid-fill-jmax 78`,
`--height-source batoid_rubin`, `--height-to-z4-factor 15`, `--ccs-footprint-points 50`.
v2 was itself built from `pathA_50_34_i_5rot`, the same `mi_name`, so v4 is the same recipe
on the new data. Note the input is **`intrinsic_split_maps.parquet` only** — the calibration
does **not** depend on `refit_mi`, so it was built in parallel with A1.

Output: **`/sdf/group/rubin/u/roodman/LSST/rubin-work/calibration/intrinsic_zernikes/v4/`**

| file | content |
|---|---|
| `intrinsic_aberrations_OCS.parquet` | 3,985 field points x 75 Zernikes (Noll 4-78) |
| `intrinsic_aberrations_CCS_det<NNN>.parquet` | 205 per-detector tables, 7x7 = 49 footprint points each |
| `provenance.yaml` | maps source, `git_sha a6394e7`, batoid backfill, height source, maps `.meta` |
| `intrinsic_zernikes_v4_maps.pdf` | 76-page QA, representative detector 102 |

**The height map question, settled (Aaron asked).** v2's provenance names Guillem's
`/sdf/home/g/gmegias/aos/batoid_rubin/.../ccd_height_map`. v4 uses **Aaron's own**
`$BATOID_RUBIN_DATA_DIR/ccd_height_map` instead, which is **byte-identical** — both
7,549,578 bytes, both MD5 `8d8591031c27306121e3caa2e752fcc2`, differing only in mtime. So
the heights are numerically identical to v2/v3 while the provenance points at Aaron's
package rather than another user's home.

**Validation against v2 on the 3,861 shared field points — the MIW fit did not change.**

| Noll group | max abs difference, µm of wavefront | reading |
|---|---|---|
| 20 of the 21 measured (5-19, 22-26) | ~**1e-15** | bit-level identical; the fit is unchanged |
| Noll 4 (defocus) | **4.19e-04** | real but tiny: 1.1% of Z4's own RMS of 0.0386 µm |
| backfilled (20, 21, 27-78) | ~**1e-10** | both from batoid `Rubin_v3.14`, as expected |

So despite v2 coming from a differently-named FAM param set three months earlier, the MIW
solution for `pathA_50_34_i_5rot` is stable. What is genuinely new in v4:

1. **+124 rim field points** (3,985 against v2's 3,861);
2. **8 height-only field-edge detectors, down from v2's 16** — v4's fallback list
   `[0, 20, 27, 65, 123, 161, 168, 188]` is exactly the first 8 of v2's, so detectors
   189, 190, 193, 194, 197, 198, 201, 202 **recovered a real camera field**;
3. Z4 moved by the 4.19e-04 µm above.

All three trace to the rim fill (`ts_intrinsic_wavefront` commit `3f66904`): field-edge CCD
footprints that used to fall past the data now land on it. **The +124 points are the same
124 cells the `miw_hull` study measured** as NaN in the old `_5rot` `Z4_OCS` map — the two
lines of work corroborate each other. Whether the Z4 shift is the rim fill perturbing the
OCS/CCS degeneracy split is **not** established; it is far below any operational threshold.

**Two provenance weaknesses to fix before this is deployed anywhere:**

- `git_describe` is **`v2.1-14-ga6394e7-dirty`** — the `ts_intrinsic_wavefront` tree is
  dirty, so the SHA does not pin the code. v2 was `-dirty` too, so this matches past
  practice, but a clean commit and tag would be better for a calibration.
- v4 exists **only in the data tree, not in git**. The repo convention
  (`aos/calibration/README.md`, `stage_miw.py`) stages a versioned
  `intrinsic_split_maps_v<N>.parquet` + provenance into git, but only `v1` is there — v2
  and v3 were never staged. Not done here: it is a different artifact from the Butler
  source tables (1 maps table against 206 per-detector tables) and adding a tracked
  parquet is Aaron's call.

**NOT DONE — needs Aaron, both hard stops:**

- **No Butler ingest and no `/repo/main` connection, not even the dry run.** The command,
  copy-paste ready (as written it is a dry run; add `--execute` to write):
  ```bash
  python $TS_INTRINSIC_WAVEFRONT_DIR/bin/ingest_calib_tables.py \
      -b /repo/main \
      --tables-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/calibration/intrinsic_zernikes/v4 \
      --output-run u/roodman/calib/intrinsicZernikes.v4/run.20261014 \
      --certify-into u/roodman/calib/intrinsicZernikes.v4
  ```
  The collection name is a **guess** — `u/roodman/...` rather than Guillem's
  `u/gmegias/calib/DM-55048/...`. Confirm which before ingesting.
- **No `set-current` for either `miw` variant.**

## Review of the remaining `miw` steps A1-A5, 2026-10-14 — the assessment Aaron acted on

Aaron asked whether the remaining `miw` steps are needed now. **Review only: no job was
submitted, no comparison run, no `set-current`, no build.** Read-only checks only
(`snakemake -n`, build-directory listings, parquet schema and column reads, `sacct`).

| step | needed by | already known? | cost | recommendation |
|---|---|---|---|---|
| **A1** `refit_mi` for `_5rot@20261014` (7 jobs) | items 5, 6 and 7 — **all three default to the `_5rot` MIW variant** and their Snakefiles cannot even build a DAG without it | no | ~32 min wall-clock on 8 CPUs, ~4.2 CPU-hours; ~$2 of session spend to submit and check | **DONE 2026-10-14** (job 40830071, 00:38:38 actual) |
| **A2** re-attribute the 7 temperature and 2 `cam_*` columns; check the 4 gradients against the database | closing 3c's expected-difference list; no product | **yes, now settled below** — and the gradient half was already answered in this file's header | already spent, ~$1 | **done, drop** |
| **A3** compare `_5rot` `fits.parquet` against the old tree | Aaron's review before `set-current` on `_5rot` | no | ~15 min, ~$2 | **DONE 2026-10-14** — clean, see the section above |
| **A4** `set-current` for both `miw` variants | items 5-7 resolve `miw` through the catalog | **partly done**: `d12-A_50_34_i/current` already points at `20261014` (set Oct 14 10:25). Only `_5rot` is missing | minutes, ~$0 | **defer to after A1+A3**; nothing is stale meanwhile |
| **A5** items 5-7 (`coadds`, `dof_lut`, `bounce_tables`) | the reorganization's "every product rebuilt in the new tree" goal | no | hours of wall-clock each, the largest remaining block | **defer**; 2 of the 9 variants are registerable from the old tree instead of rebuilt |

**A1.** Builds the `_5rot` MIW's `fits.parquet` (the post-MIW DZ refit) plus its donut and
CWFS sidecars and manifest. Its `fits.parquet` is read by **`dof_lut` and
`bounce_tables`**, and the MIW maps it completes are read by **`coadds`** — all three name
`d12-A_50_34_i_5rot` as their default `miw_variant`, so `_5rot` *is* needed as a product,
not just as a coadd input. Proof it is the hard gate: `snakemake -n` on
`rubinwork/products/dof_lut/Snakefile` does not reach a job list at all, failing at
`Snakefile:95` with `ProductNotFound: miw/d12-A_50_34_i_5rot has no current build`.
Dry run confirmed at **7 jobs** today (`intrinsic_sidecar`, 3 `wfs_intrinsic_sidecar`,
`refit_mi`, `manifest`, `all`), with `intrinsic_split` already built. Cost comes from the
comparable `d12-A_50_34_i` run, Slurm job `40799253`: **00:31:33 elapsed on 8 allocated
CPUs**.

**A2 — settled from disk, no new computation needed, and the attribution is neither 6 nor
4.** The 7 columns are `m1m3_air_temp`, `m2_air_temp`, `outside_temp`,
`tma_truss_temp_mxmy`, `tma_truss_temp_pxpy`, `cam_air_temp` and `cam_m1m3_delta_t` — the
2 `cam_*` are **2 of the same 7**, not a separate group, so the handoff's two rows are one
row. These are `attach_telemetry` ESS columns, which **decision 3 (2026-10-12) keeps**, so
difference 4 (`mktable --no-thermal`, which removes columns outright) is not their home
either: the columns are present on both sides and NaN only on some visits. The NaN pattern
is one-directional and identical on all 7 columns: **89 visits of the 960 shared have a
finite old value and NaN new; 0 have the reverse; 871 are finite on both** (7 x 89 = the
623 column-instance mismatches this file reported). The 89 are **scattered across all 15
nights** and are a minority of each night (11 of 61 on `20260315`, 15 of 82 on `20260409`,
1 of 11 on `20260419`), so it is per-visit, not per-night — consistent with a narrower EFD
query window dropping visits near a telemetry gap, the same mechanism that shifts the
surviving values by at most **0.2281 degC** (`outside_temp`, on the both-finite visits).
**This is a new expected difference, not a re-filing of 6:** call it **difference 8, the
ESS telemetry window** — 89 of 960 visits lose their ESS temperatures and the rest move by
up to 0.2281 degC. The **4 M1M3 gradients** need no new check: this file's header already
records them **equal to the value-added database at max abs 0.000e+00 degC/m on all 3,006
visits that have them** for this exact build, `20261014`.

**A3.** The same comparison A2 covers, on the `_5rot` side. Blocked on A1 and worth doing
with it, since its differences should now be the known list plus difference 8 — if `_5rot`
shows a *different* ESS pattern, difference 8's mechanism is wrong.

**A4.** `d12-A_50_34_i` is already `current@20261014`; only `_5rot` lacks one. **Nothing
breaks or goes stale while `miw` has no `current`:** a repo-wide search for
`catalog.load`/`catalog.path` on `"miw"` finds hits **only** in `catalog.py`'s own
docstring and in `products/tests/`. No notebook, no study and no other product reads `miw`
through the catalog today — items 5-7 would be the first, which is why their Snakefiles
are the only thing that fails.

**A5.** Per item, with old-tree output counted on disk today:

- **`coadds`** — 2 variants. `d12-A_50_34_i_5rot` has **7 files** in
  `aos/output/coadd/danish_1_2/50_34/`; `d12-A_50_34_i_5rot_v2` has **1** entry, the
  `archive/` directory only, so it has no comparable build and no rule plans it. Consumed
  today by nothing in the catalog. Rebuild is for the reorganization's completeness, not
  for a consumer.
- **`dof_lut`** — 3 variants. `d12-A_50_34_i` and `d12-A_50_34_i_5rot` each have **5
  files**; `d13t-A_50_34_i_5rot` has **0**. Its comparison is a *shape* check by design
  (absolute vs post-MIW DOF), so a rebuild buys a shape check, not a numerical
  verification.
- **`bounce_tables`** — 4 variants. Only `d12-A_50_34_i_5rot_july` has output, **13
  files**, and it is the tech note's source; the other 3 have none.
- **Could any be skipped or registered rather than rebuilt?** Yes, two:
  `coadds/d12-A_50_34_i_5rot_v2` and `bounce_tables/d12-A_50_34_i_5rot_july`. The `_v2`
  variant differs from `_v2`'s sibling only in `ext_terms`, which changes nothing but
  `block_grids_ext.npz` and a PDF page count, and the archive README records
  `block_grids.npz` as byte-identical — so registering the old tree with a manifest loses
  nothing. `_5rot_july` reads a **hand-made `fits_july.parquet` produced by no rule**, so
  it cannot be rebuilt reproducibly at all; a manifest over the old tree is the only
  honest option. The remaining 7 variants either have no old-tree output (so there is
  nothing to register) or are the real rebuild targets.

## Next concrete action

**The upgrade is DONE (stack `w_2026_42`, decision 9) and item 5 `coadds` is DONE,
compared clean and promoted to `current`. The next action is item 6, then item 7:**

1. ~~Item 5 — `coadds`~~ **DONE 2026-10-15.** Job 40920839, 39:42, build `20261015`,
   compared clean against the **i-band archive** and `set-current`. Every difference on
   the expected list; the maps agree to **1.4e-13 µm of wavefront**, so **no scipy 1.18.1
   or `ts_wep` effect**. See "Item 5 `coadds`: BUILT and COMPARED CLEAN" — and note the
   reference trap recorded there before comparing any other `coadds` build.
2. ~~Item 6 — `dof_lut`~~ **DONE 2026-10-15.** Both `d12` variants built locally (no batch),
   shape check passes, both `set-current`. Note the plan's "5 populated rotator bins" is
   **stale: it is 9**, confirmed against the `w_2026_39` reference. See "Item 6 `dof_lut`".
3. **Item 7 — `bounce_tables`**, the variants with a rule. Expected difference **2**,
   through the `miw` fit table (its upstream is `miw`, not `dof_lut`). The nine PDFs are
   the study half, not part of the comparison. `_5rot_july` is **not** rebuilt — registered
   old-tree variant.

Then **stop at item 8 (`value_added`)**. Section 12 permits `set-current` when a
comparison is clean; **any difference not on the expected list is a stop.** Show Aaron the
exact command for any batch job and wait for his OK.

Three things need Aaron, none blocking items 5 to 7:

- **The `ts_intrinsic_wavefront` draft PR is not open and is Aaron's to open** (`gh`
  unauthenticated on S3DF). Base is **`develop`, not `main`** — see the step 1 section for
  the ready-to-run command and the body file.
- **The 62 remaining upstream-only flake8 findings** — whether to clear them is Aaron's
  call; one (`W605`, an invalid escape sequence) is arguably a real bug. The lint fix
  itself is pushed. See "The `ts_intrinsic_wavefront` lint failure".
- **v4 Butler ingest** — still deferred, Aaron's call.

Do **not** redo the `ts_wep` rollback to `1ea1ae1d`: the smoke test was bit-identical, so
there is no difference to separate.

Nothing else in 3c is blocked:
`fam_tables`, `cwfs_tables` and `miw` are all done, compared and `current`, and the two
old-tree variants are registered.

Pointers, so nothing above is re-asked or re-measured:

- **v4 Butler ingest** — deferred, Aaron's call; the dry-run command is in the v4 section.
- **`set-current` for `_5rot`** — done 2026-10-15, see "Decisions taken".
- **Staging v4 into git** — decided against for now, see "Decisions taken".
- **Expected difference 8 (the ESS telemetry window)** — mechanism found and measured, see
  "Expected difference 8" below. Deliberate change, recoverable without an EFD re-fetch.
- **The `d12-A_50_34_i` `fits.parquet` comparison** and the `_5rot` comparison — both done
  and clean; the tables are below.

**`d12-A_50_34_i@20261014` is COMPLETE at 18 of 18 and COMPARED, 2026-10-14.** A
correction to this handoff's earlier state: the build was **already finished** before the
2026-10-14 afternoon session. `refit_mi` and `manifest` landed at 10:02 that day
(`fits.parquet` 3,385 rows x 830 columns, `manifest.json` present,
`_work/logs/refit_mi.log` ending "Saved: ... 3385 rows x 830 columns"), so the handoff's
"16 of 18" was stale. The resubmitted job **40807575** found "Nothing to be done (all
requested files are present and up to date)" and exited without recomputing or writing
anything. **Lesson:** check the build directory for the output file before resubmitting on
a handoff's word.

### The `d12-A_50_34_i` `fits.parquet` comparison, 2026-10-14

New `20261014` against the old tree `aos/output/miw/danish_1_2_A_50_34_i/fits.parquet`.
`refit_mi` runs `--no-quality-cut`, so the counts are given per side:

| side | visits (rows) | unique (`day_obs`, `seq_num`) | nights | columns |
|---|---|---|---|---|
| new `20261014` | 3,385 | 3,385 | 106 | 830 |
| old tree | 960 | 960 | 17 | 653 |
| shared key | 960 | — | 17 | 456 shared |

**All 960 old visits are in the new build; 0 old-only keys.** The new build adds 2,425
visits over 89 more nights. Differences on the 960 shared visits, 451 numeric shared
columns:

| column group | cols | differing | max abs difference | max RMS difference | NaN-status mismatches | expected difference |
|---|---|---|---|---|---|---|
| `image_idx` (row-order index, dimensionless) | 1 | 1 | 2351 | 2240 | 0 | 5 (known, unfixed) |
| `rotator_angle` (deg of camera rotator) | 1 | 1 | 0.4446 | 0.2132 | 0 | 1 |
| ambient/structure temperature (degC) | 7 | 7 | 0.2281 | 0.0414 | 623 | 6 |
| `cam_*` camera telemetry (degC) | 2 | 2 | 0.0683 | 0.0087 | 178 | 6 |
| M1M3 thermal gradients (degC/m) | 4 | 4 | 0.005892 | 0.000356 | 0 | 6 |
| DZ fit coefficients `z1toz*_*` (µm of wavefront; `*_scale` dimensionless) | 422 | 420 | 0.003857 | 0.000413 | 0 | 2 |
| fit-quality summary (`n_donuts_1/2`, `n_detectors`, `n_detectors_with_min_donuts`, `bad_fit`, `median_blur_arcsec`) | 6 | **0** | 0 | 0 | 0 | — |
| pointing / visit metadata (`visit`, `ra`, `dec`, `az`, `alt`, `mjd`) | 8 | **0** | 0 | 0 | 0 | — |

Schema: the **197 old-only columns** are the dropped `m1m3_dt_MTC*` thermocouples
(expected difference 4). The **374 new-only** are `dof0..49`, `tweak_dof0..49`,
`lut_dof5..9`, 25 `cam_*` and the `m1m3elev_*` telemetry (expected differences 3, 5, 6).

Two things to read correctly in the DZ row. The group's largest three entries are
**dimensionless** `*_scale` columns, not µm; among the **µm-of-wavefront** coefficients the
maximum is `z1toz6_z4_c5` at **1.757e-03 µm of wavefront**, within expected difference 2's
own range. And **`fits.parquet` carries no recovered DOF or v-mode columns on either side**
(`dof_event_id` is an event ID, not a DOF), so there is **no per-DOF axis in this
comparison** — the recovered DOF and `vmode_*` live in `build/rot_*/dz_fits.parquet`, which
the earlier `miw` item 4 section already compared (DOF agreeing to 4.2e-11 µm of wavefront,
`vmode_13` the degenerate-subspace rotation).

**Every difference is on the expected-difference list and nothing is off it.**
`miw` was validated by Aaron and both `d12` variants are now `current@20261014` —
`d12-A_50_34_i_5rot` promoted 2026-10-15.

## Upgrade step 1: `ts_intrinsic_wavefront` committed, rolled in, tagged, pushed, 2026-10-15

Done on the **old** stack, `w_2026_39` / `lsst-scipipe-13.1.0-exact`, as the plan requires.

- **The dirty file is committed:** `529713b` "plot_dof_median_summary: bucket DOF by label,
  not global index", on `tickets/rubinwork-rottelpos`. The tree is clean;
  `git describe` went from `v2.1-14-ga6394e7-dirty` to `v2.1-15-g529713b`.
- **Tests: 10 passed, 0 failed** (`python -m pytest tests`, 1.1 s), run twice — before the
  roll-in on `tickets/rubinwork-rottelpos`, and after it on `tickets/RSO-809`.
- **The RSO branch is `tickets/RSO-809`, and it was the only candidate.** Local head was
  `00d0920` (2026-10-07, "Merge origin/develop into tickets/RSO-809"), 29 commits ahead /
  0 behind `origin/main`; `origin/tickets/RSO-809` was 7 commits behind local at `f4f23b6`
  (2026-09-07). Both were **strict ancestors** of `tickets/rubinwork-rottelpos`, so the
  roll-in was `git merge --ff-only` — **no merge commit, no conflicts, no rewriting.**
  `tickets/RSO-809` moved `00d0920..529713b`. `tickets/rubinwork-rottelpos` is untouched
  and sits at the same commit; no branch was deleted or reset.
- **Tag: `rubinwork-2026-10-15`**, annotated, on the `tickets/RSO-809` head. **Not** a
  version number. The reason differs from the plan's guess: `v2` and `v2.1` are *not*
  default-branch tags — both were cut on the **`develop`** line — but they are still
  mainline release numbers, and claiming a `v2.2` on an unreviewed ticket branch would be
  wrong. `git describe` on the branch head now reads `rubinwork-2026-10-15`, clean.
- **Pushed, no force:** branch `f4f23b6..529713b` and the new tag, both fast-forward.
  `tickets/RSO-809` is in sync with its upstream. Nothing was pushed to `main` or
  `develop`, and nothing was merged into either.
- **Working branch for the stack move is `tickets/RSO-809`**, checked out, tree clean.

### The default branch here is a trap: PR base is `develop`, not `main`

`origin/HEAD` points at `main`, but **`main` is a single commit**, "Initial commit"
(2026-06-26). All real work is on **`develop`**, which is 22+ commits beyond it, and that
is where both release tags were cut. `tickets/RSO-809` is **14 ahead / 0 behind
`origin/develop`** but 31 ahead of `main`, so a PR into `main` would be a whole-history
diff. Aaron chose `develop` as the base.

Four commits on the branch are what rubin-work depends on, all confirmed contained in it:
`a6394e7` (rotTelPos first, ConsDB fallback, source column), `3f66904` (the field-edge
OCS/CCS rim fill, arriving via the `00d0920` merge of `develop`), `f45bfbf` (the ESS
temperature / TMA truss ConsDB source) and `529713b` (this plot fix).

### The draft PR is NOT open: `gh` has no credentials on S3DF

`gh` 2.101.0 is installed but unauthenticated — no `~/.config/gh`, no `GH_TOKEN` or
`GITHUB_TOKEN` — and `gh auth login` is interactive, so Claude could not open it. Git over
SSH works, which is why the branch and tag pushes succeeded. **The body is written and
ready** at `notes/status/pr_bodies/ts_intrinsic_wavefront_RSO-809.md`; Aaron runs, after a
one-time `gh auth login`:

```bash
cd /sdf/home/r/roodman/u/LSST/packages/ts_intrinsic_wavefront && \
gh pr create --draft \
  --base develop \
  --head tickets/RSO-809 \
  --title "RSO-809: MIW build support for the rubin-work AOS products" \
  --body-file /sdf/home/r/roodman/notebooks/rubin-work/notes/status/pr_bodies/ts_intrinsic_wavefront_RSO-809.md
```

No PR list could be checked for an existing PR on this branch, for the same auth reason —
if one already exists, update its description from that file instead of opening a second.

## Upgrade step 2: the manifest records the stack, 2026-10-15 — DONE, `020fb55`

Closes the gap named in the inventory below. Every manifest written from now on carries a
**`stack`** key:

- `eups_tag` — the `lsst_distrib` weekly (`"w_2026_39"` now), parsed as the `w_`-prefixed
  tag from `eups list -s`, not the build hash.
- `conda_env` — from `$CONDA_DEFAULT_ENV` (`"lsst-scipipe-13.1.0-exact"` now).
- `packages` — for each of the six `setup -r` packages in `manifest.STACK_PACKAGES`:
  `path`, `branch`, `commit`, `dirty` and `describe`. A package that is **not** set up is
  **absent** from the dict rather than present-and-null, so its absence is visible.

Implemented as `manifest.stack_state()` and merged in `write` via the `extra` mechanism, so
**no schema migration** and `extra` still overrides (a caller can pin a `stack` block).
Verified live: all six packages found, every field matching the inventory table below, with
`ts_intrinsic_wavefront` correctly showing `tickets/RSO-809` / `529713b` /
`rubinwork-2026-10-15`.

**Tests: 358 passed, 0 failed** (`python -m pytest rubinwork`, 119.9 s) — the 355 baseline
plus 3 net-new (`stack_state` against a faked eups environment, `stack_state` with no eups
at all, and `write` recording the stack and honoring an `extra` override).

### Two things the next session should not relearn

- **The "355 tests in 0.99 s" in the old inventory was the COLLECTION time, not the run
  time.** The suite takes about **120 s** wall clock. Budget for that, and do not read a
  two-minute run as a hang.
- **`stack_state` is `lru_cache`d, and the catalog tests patch it out through an autouse
  fixture.** Both are load-bearing for speed, not tidiness: `write` costs one `eups list`
  plus four git subprocesses per package, and without the cache and the fixture the suite
  went **779 s** instead of 120 s. A test that wants the real function must restore it from
  the `_real_stack_state` captured at import and call `cache_clear()`; a bare `lambda`
  replacement has no `cache_clear`, which is what broke the first attempt (37 teardown
  errors). `write` deep-copies the cached dict so a caller cannot mutate the cache.

## Upgrade step 3: the `w_2026_39` reference `dof_lut` is BUILT, 2026-10-15

The thing the stack-move smoke test compares against. Built on the **old** stack,
`w_2026_39` / `lsst-scipipe-13.1.0-exact`, before the move.

**It is a normal dated build with NO `set-current`**, not a build under
`studies/`. The Snakefile **cannot** write outside the products tree: `BUILD_DIR` comes
from `dof_lut.build_dir(VARIANT, BUILD)`, which is `catalog.products_root() / PRODUCT /
variant / build` with no config knob (`rubinwork/products/dof_lut/reader.py:192`). So the
step's fallback applies.

```bash
cd ~/notebooks/rubin-work && \
snakemake -s rubinwork/products/dof_lut/Snakefile \
  --cores 1 --config variant=d12-A_50_34_i_5rot build=20261015w39
```

- **3 of 3 jobs** (`build_lut`, `manifest`, `all`), as the dry run predicted. Local, about
  12 s wall clock — **no batch job needed.**
- Build `20261015w39` at
  `products/dof_lut/d12-A_50_34_i_5rot/20261015w39/`, 6 files, `status: complete`.
  **`dof_lut` had no directory in the new tree before this**; this is its first build there,
  and the variant has **no `current` symlink**, deliberately.
- Inputs recorded: `fam_tables: danish_1_2@20261014` and
  `miw: d12-A_50_34_i_5rot@20261014`.
- **It is the first product in the tree carrying the step 2 `stack` block**, reading
  `w_2026_39` / `lsst-scipipe-13.1.0-exact` and `ts_intrinsic_wavefront` at
  `rubinwork-2026-10-15`.

### The reference copy and its checksums, for the step 5 comparison

Copied out of the build directory so the comparison is against an immutable snapshot:
`$RUBINWORK_DATA/studies/phase3c_validation/stack_move/w39/`.

| file | md5 | rows (count, dimensionless) | columns (count, dimensionless) |
|---|---|---|---|
| `lut.parquet` | `4a83304b369aa32f303ebe64a9da15f6` | 50 | 7 |
| `lut_dz.parquet` | `54c6bcca9a88f56995a563a008a30536` | 126 | 6 |
| `lut_by_rotbin.parquet` | `1a3b451947d081dacacf9475aa0e3f78` | 450 | 7 |

`lut.parquet` columns: `dof_index`, `dof_label`, `dof_unit`, `value`, `scatter_mad`,
`mean`, `n_visits` — one row per DOF (50 DOF), `n_visits` = 969 visits. First rows, for a
quick eyeball after the move: `M2_dz` value = 528.509872 µm, `M2_dx` = 363.139748 µm,
`M2_dy` = -2348.677584 µm (hexapod rigid-body offsets, µm).

**A number not to be alarmed by:** the Snakefile's header comment quotes `M2_dy` as
-2407.34 µm. This build gives **-2348.677584 µm**, a difference of 58.7 µm. That is a
different upstream `miw`/`fam_tables` build, not a stack effect — the comment describes the
20260920-era build. Step 5 compares **this** build against the `w_2026_41` rebuild, so the
comment's value does not enter.

## Upgrade step 4: `ts_wep` PULLED and `w_2026_41` VALIDATED; the switch is NOT made, 2026-10-15

Two of the three parts are done. The stack itself is still `w_2026_39` in this session and
in Aaron's login shell.

### `ts_wep` is pulled: 28 commits, not the 9 the inventory says

```bash
cd /sdf/home/r/roodman/u/LSST/packages/ts_wep && git fetch origin && \
  git merge --ff-only origin/develop
```

- `1ea1ae1d` → **`0363c0c4`** ("Merge pull request #182 from lsst-ts/main", 2026-10-14),
  clean fast-forward, tree clean, in sync with `origin/develop`.
- `git describe`: `v17.14.1-1-g1ea1ae1d` → **`v17.15.0-1-g0363c0c4`**. New tags `v17.14.2`
  and `v17.15.0` arrived with the fetch.
- **Correction to the inventory below: it was 28 commits behind, not 9.** `develop` moved
  between the 2026-10-05 fetch the inventory used and this one. 49 files changed,
  +1,121 / -205 lines.
- Two of the 28 are worth knowing about beyond the saturated-image guard and the
  `empty()` OOD-column crash fix the inventory already names:
  - **`6253be3f` adds provenance for the intrinsic Zernikes calibration dataset** into the
    zernikes tables metadata — the same calibration as `intrinsic_zernikes` v4. Check
    whether it changes what a v5 ingest should write.
  - `8843a486` "guard against saturated images" and `97cc05b9` "Handle batoid failures
    without crashing task" both touch the FAM donut path.

### `w_2026_41` is validated: all twelve packages set up and import, `summit_utils` included

Tested in a **throwaway clean shell**, without touching Aaron's `setup.sh`. The technique
matters, because it is not obvious: sourcing another weekly's `loadLSST.sh` from a shell
that already has one active **silently fails** — `setup` ends up not found and
`CONDA_DEFAULT_ENV` stays `base`. A clean environment is required:

```bash
timeout 580 env -i HOME=$HOME USER=$USER TERM=dumb PATH=/usr/bin:/bin \
  bash --noprofile --norc ~/notebooks/rubin-work/scratch/w41_setup_test.sh
```

- `lsst_distrib` `g00e868bf88+e0f0dea02a`, tags `current w_latest w_2026_41`, conda env
  `lsst-scipipe-14.0.0-exact`.
- **All twelve `setup -r` packages set up with no output at all** — the six in the inventory
  plus `ts_utils`, `ts_salobj`, `ts_idl`, `ts_observatory_control`, `ts_xml`,
  `ts_m1m3_utils`. **No `scons` and no version build was needed for any of them.**
  **`summit_utils`, the one the plan called the most likely failure** (branch cut from
  `w_2026_34`, 7 weeklies back), set up clean.
- Imports all succeed: `lsst.ts.wep`, `lsst.ts.ofc`, `lsst.summit.utils`,
  `lsst.ts.intrinsic.wavefront`, and `rubinwork` 0.1.0 with its catalog and manifest.
- Library versions on the new env, confirming the inventory's prediction: **scipy
  1.18.1** (from 1.16.3 — the one real numerical change), numpy 2.4.6, astropy 8.0.1,
  pandas 2.3.3 all unchanged, and **danish 1.3.0, batoid 0.9.0, batoid_rubin 0.7.0,
  galsim 2.8.5** all unchanged.
- `w_2026_42` also exists on cvmfs now. Decision 8 names **41**; 42 was not tested.

### A step 2 defect the w_2026_41 test caught, fixed in `ff2a8dd`

`stack_state` recorded **`eups_tag: w_latest`** on the new stack instead of `w_2026_41`.
eups lists a package's tags **unordered**: `w_2026_39` lists as
`"<hash> w_2026_39 w_latest current setup"` but `w_2026_41` lists as
`"<hash> current w_latest w_2026_41 setup"`, so taking the first `w_`-prefixed field picked
up the moving alias and named no weekly at all. Now matched against `w_\d{4}_\d{2}`.
**Every manifest built after the move would have been mislabelled.** Re-verified live:
`w_2026_39` on the old stack, `w_2026_41` on the new one. Tests are now **42 in
`test_catalog.py`**, including a parametrised case pinning both field orders.

### What is left of step 4: the switch itself

**Not done, and it needs a Claude restart rather than an in-session change.** The weekly is
set in **`~/u/LSST/setup.sh` line 3**, `weekly_version=w_2026_39`, which `~/.bashrc`
sources for every login. A running session cannot adopt a new stack by re-sourcing it, so
the sequence is:

1. Change line 3 of `/sdf/home/r/roodman/u/LSST/setup.sh` to `weekly_version=w_2026_41`.
2. Exit Claude, start a new shell (or `source ~/u/LSST/setup.sh`), confirm with
   `eups list lsst_distrib` and `echo $CONDA_DEFAULT_ENV`.
3. Start a fresh Claude session and do step 5 there.

Aaron declined letting Claude edit `setup.sh` in this session and will handle the restart,
so **as of this entry the file still reads `w_2026_39`.** Confirm it before trusting a
shell to be on the new stack.

**The exact setup commands, for reproducing the environment** (this is `~/u/LSST/setup.sh`
with the weekly changed, and the `-j` flags matter — they stop eups pulling the stack's own
copy of a package):

```bash
weekly_version=w_2026_41
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/${weekly_version}/loadLSST.sh
setup lsst_distrib
P=/sdf/home/r/roodman/u/LSST/packages
setup -r $P/summit_utils -j
setup -r $P/ts_utils -j
setup -r $P/ts_salobj -j
setup -r $P/ts_idl -j
setup -r $P/ts_observatory_control -j
setup -r $P/ts_aos_analysis -j
setup -r $P/ts_ofc -j
setup -r $P/ts_wep -j
setup -r $P/ts_xml -j
setup -r $P/ts_config_mttcs -j
setup -r $P/ts_m1m3_utils -j
setup -r $P/ts_intrinsic_wavefront -j
```

`ts_intrinsic_wavefront` must be on **`tickets/RSO-809`** at `529713b`
(`git describe` = `rubinwork-2026-10-15`), which is where step 1 left it.

## Upgrade step 5: `w_2026_42` is LIVE and VALIDATED, and the `dof_lut` smoke test is BIT-IDENTICAL, 2026-10-15

The switch is made (decision 9) and the smoke test is clean, so **the upgrade is done** and
items 5 to 7 are unblocked. No rollback test was needed.

### The environment, confirmed before anything else

- `eups list lsst_distrib` → `g00e868bf88+ec6e0cf799  w_latest current w_2026_42 setup`.
- `CONDA_DEFAULT_ENV` = `lsst-scipipe-14.0.0-exact`; `EUPS_PATH` under the `w_2026_42`
  cvmfs tree. **No `w_2026_39` and no `13.1.0` on `PATH`, `PYTHONPATH` or `EUPS_PATH`.**
- `~/u/LSST/setup.sh` line 3 reads `weekly_version=w_2026_42`.
- **All twelve `setup -r` packages are already set up in the login shell** (`eups list -s`),
  so unlike step 4 **no throwaway clean shell was needed** — the live session *is* the test.
- Package commits, all as step 1/4 left them: `ts_intrinsic_wavefront` `tickets/RSO-809`
  `529713b0`, `ts_wep` `develop` `0363c0c4` (`v17.15.0-1-g0363c0c4`), `ts_ofc`
  `tickets/RSO-1007` `ca1eb91e`, `summit_utils` `u/aaronroodman/guider-fixes` `e57e0f48`,
  the other eight on `develop`. All tracked trees clean.
- **One cosmetic exception:** `ts_observatory_control` reports dirty, but it is an
  **untracked `guider_roi_extras/` directory only** — no tracked modification. The manifest
  writer records it as clean because it does not see untracked files. Not a defect.

### Library versions: NOTHING moved from 41 to 42

| library | w_2026_41 | w_2026_42 | moved? |
|---|---|---|---|
| scipy | 1.18.1 | 1.18.1 | no |
| numpy | 2.4.6 | 2.4.6 | no |
| astropy | 8.0.1 | 8.0.1 | no |
| pandas | 2.3.3 | 2.3.3 | no |
| danish | 1.3.0 | 1.3.0 | no |
| batoid | 0.9.0 | 0.9.0 | no |
| galsim | 2.8.5 | 2.8.5 | no |
| batoid_rubin | 0.7.0 | 0.7.0 | no |

So **42 is numerically the same environment as the validated 41**; the scipy 1.16.3 →
1.18.1 jump is still the only library change relative to `w_2026_39`. All five step 4
imports succeed: `lsst.ts.wep`, `lsst.ts.ofc`, `lsst.summit.utils`,
`lsst.ts.intrinsic.wavefront`, `rubinwork` 0.1.0 with its catalog and manifest.

**A practical note for the next session:** the first import of the stack on a newly-mounted
weekly took **over 2 minutes** (cold cvmfs cache) and tripped a 120 s command timeout. Run
the first `python -c "import lsst..."` with a generous timeout, or in the background; it is
not a hang.

### The fresh manifest records the dated weekly

`manifest.stack_state()` on the live stack gives `eups_tag: w_2026_42` and `conda_env:
lsst-scipipe-14.0.0-exact`, with **all six recorded packages clean**. The `ff2a8dd` fix
(match `w_\d{4}_\d{2}`, never the `w_latest` alias) **holds on 42**, where eups again lists
the alias before the dated tag. The manifest records **6** packages, not 12 — that is the
step 2 design (the products' direct dependencies), unchanged.

### `python -m pytest rubinwork`: 361 passed, and why it is not 358

**361 passed, 1 warning, 145 s.** The handoff expected **358**. The difference is fully
accounted for and is **not** a new or missing test: `ff2a8dd` took
`rubinwork/products/tests/test_catalog.py` to **42 tests** (the parametrised cases pinning
both eups field orders) *after* the 358 figure was written. 358 + 3 = 361. The one warning
is the pre-existing `asyncio` `DeprecationWarning` from `ts_ofc`'s `ofc_data.py:159`.
**Expect 361 from now on.**

### The smoke test: `dof_lut d12-A_50_34_i_5rot@20261015w42` is BIT-IDENTICAL to the w39 reference

```bash
cd ~/notebooks/rubin-work && \
snakemake -s rubinwork/products/dof_lut/Snakefile \
  --cores 1 --config variant=d12-A_50_34_i_5rot build=20261015w42
```

3 of 3 jobs, local, about 12 s. Build `20261015w42`, 6 files, `status: complete`, inputs
`fam_tables: danish_1_2@20261014` and `miw: d12-A_50_34_i_5rot@20261014` — the same inputs
as the reference. `stack` block reads `w_2026_42` / `lsst-scipipe-14.0.0-exact`. **No
`set-current`**, deliberately: it is a smoke-test build, and the variant still has no
`current` symlink.

**All three md5 checksums match the `w_2026_39` reference exactly**, so the comparison is
bit-for-bit and the column table is redundant — recorded anyway because the instruction
asked for it column by column:

| file | md5 (w39 and w42, identical) | columns compared | max abs difference |
|---|---|---|---|
| `lut.parquet` | `4a83304b369aa32f303ebe64a9da15f6` | 7 (50 rows) | 0 in all 5 numeric; 0 string mismatches in `dof_label`, `dof_unit` |
| `lut_dz.parquet` | `54c6bcca9a88f56995a563a008a30536` | 6 (126 rows) | 0 in all 6 numeric (`dz_raw`, `dz_fit`, `dz_resid` in µm of wavefront) |
| `lut_by_rotbin.parquet` | `1a3b451947d081dacacf9475aa0e3f78` | 7 (450 rows) | 0 in all 6 numeric; 0 string mismatches in `dof_label` |

Attribution, per column: **none required — nothing differs.** The DOF values are unchanged
to the last bit, e.g. `M2_dz` = 528.509872 µm, `M2_dx` = 363.139748 µm, `M2_dy` =
-2348.677584 µm (hexapod rigid-body offsets, µm), exactly the reference figures.

**What this rules out, and it is the useful result:** neither **scipy 1.18.1** nor the **28
`ts_wep` commits** reaches the `dof_lut` code path. **The `ts_wep` rollback to `1ea1ae1d`
was therefore NOT run and is not needed** — there is no difference to separate. The next
session should not redo it. This does *not* prove the donut-fitting path is unaffected:
`dof_lut` consumes the already-built `fam_tables` and `miw` parquet, so it never calls
`danish` or `ts_wep`'s FAM estimator. **Items 5 to 7 are where a scipy or `ts_wep`
difference would actually show up**, item 5 (`coadds`) first.

## Item 6 `dof_lut`: BOTH `d12` VARIANTS BUILT, SHAPE-CHECKED and PROMOTED, 2026-10-15

Local, 3 jobs and about 12 s each — **no batch job**. Both variants now `current`:

| variant | build | `current` |
|---|---|---|
| `d12-A_50_34_i_5rot` | `20261015w42` | yes |
| `d12-A_50_34_i` | `20261015` | yes |

`_5rot` is promoted at **`20261015w42`**, the step 5 smoke-test build — the same build that
compared bit-identical to the `w_2026_39` reference, so it is a legitimate `current`
despite the `w42` suffix in its name. No separate rebuild was needed. The third variant
`d13t-A_50_34_i_5rot` has no old-tree output and is not part of item 6.

### The shape check passes on every criterion

| criterion | expected | `_5rot` | `_i` |
|---|---|---|---|
| `lut.parquet` rows | 50 DOF | 50 | 50 |
| `lut_dz.parquet` rows | 126 (k, j) | 126 | 126 |
| `lut_config.yaml` keys | 11 | 11 | 11 |
| `lut_by_rotbin` rows | — | 450 | 450 |
| populated rotator bins | **plan says 5** | **9** | **9** |
| manifest `status` | complete | complete | complete |

**The plan's "5 populated rotator bins" is STALE — the correct number is 9, and this is
not a defect.** All 9 bins are genuinely populated, 16 to 309 visits each, summing to
exactly the 969 visits in `n_visits`; 450 rows = 50 DOF x 9 bins is self-consistent. The
independent `w_2026_39` step 3 reference has the **same 9 bins and 450 rows**, so the grid
is unchanged across the stack move. Bin centres: -60, -45, -30, -15, 0, +15, +30, +45,
+60 deg of camera rotator. **Fix the plan's item-6 row to say 9.**

### Against the old tree: the by-design shift, confirmed

Identical shape (50 x 7, same 7 columns) on both, and the DZ residual **improves**, which
is the point of the absolute -> post-MIW change (`733c01d`):

| variant | max DOF shift | median DOF shift | `dz_resid` RMS old -> new |
|---|---|---|---|
| `_5rot` | **60.8 µm** (`Cam_dz`, camera hexapod piston) | 0.003 µm | **0.0109 -> 0.0049 µm of wavefront** |
| `_i` | **51.6 µm** (`Cam_dz`) | 0.002 µm | **0.0109 -> 0.0031 µm of wavefront** |

Largest movers on both: `Cam_dz`, `M2_dy`, `M2_dz` — the hexapod rigid-body terms, as
expected.

**The plan's "at most about 16 µm of hexapod motion" is not contradicted.** That figure
was measured on the **3a reference slice of 169 visits**; these builds cover **969
visits**, a different and much larger sample, so a larger shift is consistent. The DZ
residual improvement is the criterion that carries over, and it holds on both variants.
**The ill-conditioned-inversion caveat from the 3a section still stands** and is untouched
by this item: the LUT inverts a DZ signal of ~0.037 µm of wavefront RMS into thousands of
µm of hexapod motion, which remains open and is not a 3c question.

## Item 5 `coadds`: BUILT and COMPARED CLEAN, 2026-10-15

Job **40920839 COMPLETED** in **00:39:42**, peak RSS **16.9 GB** of the 128 GB requested
(the Snakefile's header predicted 23.9 GB, so the request has ample headroom and could be
trimmed). All 6 outputs plus `manifest.json`, `status: complete`, recording
`w_2026_42` / `lsst-scipipe-14.0.0-exact` and both upstreams
(`fam_tables danish_1_2@20261014`, `miw d12-A_50_34_i_5rot@20261014`).

### READ THIS FIRST: the comparison reference is the ARCHIVE, not the top level

The plan says "compare against `aos/output/coadd/<dir_name>/`". **Taken literally that is
the wrong reference and manufactures a 40% block loss that is not real.** The top level of
`aos/output/coadd/danish_1_2/50_34/` is an **all-bands** run; the variant is **i-band
only** (`bands: null` -> the `mi_config` `filter: i`). The like-for-like reference is
`archive/20260903_iband/`.

| build | detected blocks | nights | built blocks | bands |
|---|---|---|---|---|
| old **top level** | 698 | 106 | 216 | all bands |
| old **`archive/20260903_iband/`** | 274 | 61 | 130 | **i only** |
| **new `20261015`** | **274** | **61** | **130** | **i only** |

The new build reproduces the i-band archive **exactly** on all three counts. The 45 nights
"missing" against the top level are 2025 nights that are r-band dominated (1,229 r-band
against 689 i-band visits in 2025), so an i-band selection legitimately drops them —
`archive/20260903_iband/README.md` documents this and states 2025 blocks fall 117 -> 40
while 2026 is nearly unchanged at 99 -> 90. The upstream `visits.parquet` has all 106
nights and 3,385 visits, so **nothing was lost upstream**; this was purely the reference
choice. **Fix the plan's item-5 row to name the archive.**

Also note **`blocks_summary` and `coadd_metrics` count different things** and both are
right: `blocks_summary` is every *detected* block (274) and `coadd_metrics` is only those
*built*, i.e. `n_visits >= min_visits` of 6 (130). The `plotted` column carries the flag.
The `>= 6` filter behaves identically on both sides (216 of 698 old top level, 130 of 274
new and archive).

### The comparison: every difference on the expected list

`python -m rubinwork.products.coadds.compare_builds aos/output/coadd/danish_1_2/50_34/archive/20260903_iband <build>`
— 274 vs 274 rows, 130 vs 130 blocks, **same blocks in the same order**.

| group | count | max abs difference | expected difference |
|---|---|---|---|
| ESS + dome/M2 air temperatures, degC | 7 | **0.0743 degC** (`dome_delta_t`) | **8** |
| M1M3 bulk thermal gradients, degC/m | 4 | **0.000384 degC/m** (`z_gradient`) | **6** |
| `rot`, camera rotator angle, deg | 1 | **0.858 deg** | **1** |
| `resid_rms_*`, `corr_*`, `cwfs_*`, `umode_1..34` (µm of wavefront; `corr_*` dimensionless) | 54 | **4.5e-14** (`umode_18`) | **2**, at float round-off |
| `block_grids.npz` maps (`grids`, `miw`, `raw_dz`, `resid_dz`, `U_all`, `U_disc`, `U_eff`, `V`, `umodes`, `sigma`) | 10 | **1.4e-13 µm of wavefront** (`grids`) | **2**, at float round-off |

**The scientific content reproduces to float round-off** — 1.4e-13 µm of wavefront against
difference 2's own bound of 9.2e-04 µm, nine orders of magnitude inside it. Only the
telemetry columns and `rot` move at physically meaningful levels, and all three are on the
expected list. The telemetry ones are expected **by decision 10**, Aaron 2026-10-15.

**The headline result: no scipy 1.18.1 and no `ts_wep` effect.** This was the first
product where either could reach the numbers (`dof_lut` reads built parquet and never
calls `danish` or the FAM estimator). The maps agree to 1e-13, so the stack move is
numerically transparent here too.

### Two archive artifacts that are NOT findings

1. **`block_grids_ext.npz` differs at 8.75e-08**, far above the 1e-13 elsewhere. Cause:
   **the archive stored it `float64`, the new build stores `float32`** — 8.75e-08 is
   **2.97e-08 relative** to the 2.94 µm map scale, right at float32 eps (1.19e-07). The
   cast is deliberate and commented in `run_coadd_blocks_miw.py:935` ("Memory-lean: stack
   into float32 -- these are diagnostic maps"); the archive predates it. **Not** an
   `ext_terms` difference: both sides carry the identical 11 terms
   `[12 13 16 17 18 19 22 23 24 25 26]`, which was the first guess and was wrong.
2. **The PDF size and `MISSING` lines.** The archive's `perblock.pdf` is the 2026-09-03
   i-band render, its `timeseries.pdf` is the **2026-08-24 all-bands** one, and
   `coadd_timeseries_rebin3.pdf` is a `recompute_coadd_metrics.py` product the Snakefile
   does not build at all. All three are described in the archive README. PDFs are
   size-checked only, by design.

So `compare_builds` printing `RESULT: DIFFERENCES FOUND` is expected here; the verdict is
a per-difference judgement, not that line.

## Item 5 `coadds`: the submission record, job 40920839, 2026-10-15

First of the three remaining comparison-plan items, and the first product rebuilt on
`w_2026_42`. **Aaron OK'd the exact submit command**; the job is running, the comparison
is not yet done.

```bash
sbatch --account=rubin:developers@milano --partition=milano --job-name=coadd_d12_5rot \
  --cpus-per-task=8 --mem=128G --time=04:00:00 \
  --output=/sdf/home/r/roodman/notebooks/rubin-work/aos/logs/coadd_blocks_%j.log \
  --wrap='cd /sdf/home/r/roodman/notebooks/rubin-work && snakemake -s rubinwork/products/coadds/Snakefile --cores 8 --config variant=d12-A_50_34_i_5rot build=20261015'
```

- **Job 40920839**, `milano`, `sdfmilan019`, log
  `aos/logs/coadd_blocks_40920839.log`. Build `20261015`, variant
  `d12-A_50_34_i_5rot`. 3 jobs (`coadd_blocks`, `manifest`, `all`).
- Upstreams resolved from `current`: `fam_tables danish_1_2@20261014` and
  `miw d12-A_50_34_i_5rot@20261014` — the promoted builds, as intended.
- Resources from the Snakefile's own `mem_mb=128000` and the header's measured figures
  (216 blocks, 3,991 s wall clock, peak RSS 23.9 GB). 4 h is ~3.6x the measured time.

### Two things that cost time here, so the next session does not repeat them

1. **`sbatch` needs `--account=rubin:developers@milano` on this cluster.** Without it the
   submission is rejected with "Invalid account or account/partition combination", after
   two *non-fatal* warnings about qos and OS_VER that are easy to mistake for the cause.
   The account was recovered from the precedent job's own record
   (`sacct -j 40830071 --format=JobID,Account,QOS,Partition -P`), not guessed.
2. **There is no saved submit script for the new-tree products.** The `miw` A1 precedent
   (job 40830071, `miw_refit_d12_5rot`) was an inline `sbatch --wrap` from the repo root;
   no file in the repo carries that job name. `rubinwork/products/fam_tables/`
   has the only submit helper (`submit_mktable_nights.sh`), and it is per-night and
   `fam_tables`-specific. Reuse the `--wrap` form above.

### The comparison target, confirmed present before submitting

**`aos/output/coadd/danish_1_2/50_34/`** — note the `dir_name` nests under the FAM
variant, so `aos/output/coadd/50_34/` does not exist and looks like a missing build.

| file | scale |
|---|---|
| `blocks_summary.parquet` | 698 rows x 26 columns |
| `coadd_metrics.parquet` | 216 rows x 81 columns (one row per detected block) |
| `block_grids.npz` | 42 arrays |
| `block_grids_ext.npz` | 6 arrays |

`rubinwork.products.coadds.compare_builds` keys both tables on **`block`**, aligns the
per-block axis of stacked npz arrays on the `block` array rather than assuming order, and
**excludes the two PDFs by design** (matplotlib embeds a creation timestamp, so identical
code gives different bytes). Expected difference: **2 only** (the theta rotation, through
the MIW maps it coadds). Anything else is a stop.

**This is the first product where scipy 1.18.1 or the 28 `ts_wep` commits could actually
show.** The step 5 `dof_lut` smoke test could not: it reads already-built parquet and
never calls `danish` or the FAM estimator.

### Predicted: the plan's "difference 2 only" is INCOMPLETE for this item

Read before judging the comparison. The plan's item-5 row lists difference 2 alone, but
`coadd_metrics.parquet` carries **telemetry columns joined from the same upstream that
changed**, so differences 6 and 8 must reach it too. Measured on the old-tree build's
81 columns:

| column group in `coadd_metrics` | count | expected difference |
|---|---|---|
| ESS temperatures, degC (`cam_air_temp`, `m2_air_temp`, `m1m3_air_temp`, `outside_temp`, `cam_m1m3_delta_t`, `tma_truss_temp_pxpy`, `tma_truss_temp_mxmy`) | 7 | **8** (the ESS window: visits going finite-old to NaN-new, and up to 0.2281 degC on both-finite visits) |
| M1M3 bulk thermal gradients, degC/m (`x_`, `y_`, `z_`, `radial_gradient`) | 4 | **6** (now joined from `visit_telemetry`; check against the value-added database at zero tolerance, not against the old tree) |
| `resid_rms_z5..z8`, `corr_z5..z8`, the `cwfs_*` pair, `umode_1..34` | 34 + 20 | **2** (the theta rotation, through the MIW maps) |

So a difference in those 11 telemetry columns is **on the expected list already** (6 and
8), not a new finding — it is just recorded against `miw`/`fam_tables` rather than against
`coadds`. A difference in the `resid_rms`/`corr`/`umode` columns beyond difference 2's own
bound, or in `blocks_summary`'s block detection (698 rows, 216 blocks), **is** a stop.

**Confirmed by Aaron 2026-10-15: treat the telemetry columns as expected** — and the
per-product re-derivation that causes them is what he wants to remove. See decision 10.

`block` is unique on all 216 rows of the old build, so the comparison key is sound.

## The `ts_intrinsic_wavefront` lint failure: FIXED and PUSHED, 2026-10-15

Aaron found the GitHub **lint job red** on `tickets/RSO-809`. It was not on any list —
the handoff recorded only the unopened draft PR. Fixed in commit **`f9bafa2`**, and
**PUSHED** to `origin/tickets/RSO-809` on Aaron's OK (`529713b..f9bafa2`).

### What was actually failing, which is not what it looked like

"A few lines too long" was the right instinct for the cause but not the scale. The lint
job runs `flake8` over the **whole repo**, and the limit doing the damage is
**`max-doc-length = 79`** in `setup.cfg`, not `max-line-length = 110`. W505 applies only
to **comments and docstrings**, so code may run to 110 columns while prose must wrap at
79 — which is why the largest class was all prose.

**The debt is almost entirely upstream's, not this branch's:**

| scope | flake8 findings (count, dimensionless) |
|---|---|
| `origin/develop`, the same 11 files | 239 |
| `tickets/RSO-809` before the fix, same 11 files | 249 |
| net added by RSO-809's 15 commits | **+10**, all W505, in 2 files |
| whole repo on `develop`, its own `setup.cfg` | 340 |
| whole repo on `tickets/RSO-809` now | **63** |

The +10 were new docstring prose in `dz_fitting.py` (+6) and `intrinsics_lib.py` (+6);
`intrinsic_build_plots.py` was **−2**. RSO-809 introduced **no new violation type** —
every code on the branch already occurred on `develop`. Aaron chose to clear **all 249**
rather than only his own 10.

### How each class was cleared

- **W505 (76)** — reflowed **per paragraph**, not per line. See the warning below.
- **E702/E701 (60)** — compound statements split with a tokenizer, so a `;` or `:` inside
  a string literal is never mistaken for a separator.
- **F401/F811 (23)** — unused imports, each confirmed dead by a usage grep first. The
  `scipy.stats` and `Line2D` top-level imports were dead **because the same names are
  re-imported inside the functions that use them** — removing the top-level ones also
  cleared the 4 F811s.
- **F841 (6)** — dead locals: `sigma`/`sigma_mad`/`std` left from an earlier annotation
  that now shows only `med`, an unread colorbar handle, and an unused `camera_id_map`.
- **F541 (7)** — `f"..."` with no placeholders turned into plain strings.
- **W504 + E127/E128/E231/E302/E303/E305/E306/E731 (remainder)** — whitespace,
  continuation indent, and one `lambda` assignment turned into a `def`.

### Two `setup.cfg` decisions, deliberately not source edits

`ignore` gains **E201, E221, E241, E272** and **E741**:

- The first four flag **intentional column alignment** — the `DOF_GROUPS` table, the arrow
  vertex coordinate list (whose leading spaces align `hw`/`sw` against `-sw`/`-hw` so the
  shape reads), and the aligned unit comments. Flattening them to satisfy flake8 makes the
  code worse, and they are upstream's lines. This silenced 55 of the 249 with no source
  change. Aaron chose this over flattening.
- **E741** is `O` and `C` in the intrinsic decomposition: the **OCS/CCS split**, not
  stand-ins for 0 and 1. The one-letter names carry the physics.
- **W503 was already ignored**, so W504 could not also be enforced — a line break has to
  go somewhere. The `DOF_UNITS_50` `+` operators moved to line starts, consistent with
  that existing choice. `DOF_UNITS_50` re-verified at **50 entries** afterwards.

### The warning worth keeping: W505 needs PARAGRAPH reflow, not per-line wrapping

The first attempt wrapped each flagged line **in isolation** and produced garbage —
a long line became a short line plus an orphan (`(default)`, `is`, `set,`) on the next.
The numpydoc blocks were left worse than the warning. **That attempt was reverted
wholesale**; the working script collects the contiguous run of sibling lines at the same
indent and reflows the whole paragraph, skipping aligned structure, numpydoc
`name : type` headers and docstring quote lines. The helper scripts are in
`scratch/lintfix/` (`fix_w505.py`, `fix_stmt.py`).

**`black` was rejected**: not configured in this repo, and it would rewrite **6,430
lines** across 11 files upstream never formatted with it. It also does not fix W505 at
all.

### Verification, and the functional-change question

- `flake8` on the 11 changed files: **249 → 0**.
- `pytest tests/`: **10 passed** (this package has only one test file, so it is thin cover
  — hence the checks below).
- All 8 modules import, including `intrinsic_split` and `calib_tables`.
- `python -m pytest rubinwork`: **361 passed**, unchanged.
- **The strongest check: a `dof_lut` rebuild (`20261015lint`) is bit-identical to the
  `w_2026_39` reference on all three parquet md5 checksums.** The F401/F841 removals and
  the statement splits changed no behaviour on that path.
- A word-level prose audit of all 9 edited Python files confirms **no text was lost**:
  every difference is an intended one (semicolon split, removed import, removed dead
  statistic, dropped `f` prefix, or a deliberate reword). `combine_parquets.py` differs by
  exactly one added `\` line continuation.

**A `git diff -U0` reading of this change is misleading** — it interleaves `-` and `+`
hunks so non-adjacent added lines look contiguous, which made one `combine_parquets.py`
docstring appear to have lost a clause. It had not. Audit prose with a word-level
comparison, not by eye on a zero-context diff.

### The other 62 are now CLEARED too: the whole repo lints clean, `4c77b90`

Aaron asked for them (2026-10-15), so `flake8 .` over the whole package now reports
**0 findings**. Pushed as **`4c77b90`**. What they were, and the one that mattered:

| code | count | disposition |
|---|---|---|
| W505 | 50 | paragraph reflow, plus 7 one-line docstrings shortened by hand |
| E702 | 11 | compound statements split, all in `intrinsic_split.py` |
| W605 | 1 | **a latent bug, not style** — see below |

**The W605 was worth fixing on its own merits.** `bin.src/ingest_calib_tables.py`'s module
docstring contained ``butler.put``\ s — a Sphinx escaped space — in a non-raw string, so
`\ ` was an **invalid escape sequence**. Python emits a `SyntaxWarning` for it today and
**will make it an error** in a future version. Fixed by making that docstring a raw
string (`r"""`), which keeps the Sphinx markup intact. Nothing else in the package raises
a `SyntaxWarning` now, verified with `python -W error::SyntaxWarning`.

**The trap that cost the most here, and will recur: a prose reflow destroys a
copy-pasteable example.** Four needed hand repair after the automated pass, because the
reflow moved a **line-final `\` into the middle of a line** and **split an import after
its comma**, leaving commands and an import that no longer run:

- `ingest_calib_tables.py` — the `butler` ingest example (`-b /repo/main \ --tables-dir`)
- `plot_calib_tables.py` — the plotting example
- `run_mktable.py` — a 103-character single-line command, now broken with a `\`
- `common/zernike_names.py` — `from ... import NOLL_NAMES,` split after the comma; now
  parenthesised so it pastes

All four are rewritten to stay valid **and** fit 79 columns. The reflow script's
`is_structure` guard does not catch these: a `\`-continuation line has no multi-space run,
so it reads as prose. **After any reflow pass, check every example block.** The check
used: assert no `\` appears mid-line outside a string escape, and no `import`/`from` line
ends in a comma.

Verification: **0 flake8 findings** whole-repo, 10 tests pass, **all 12 modules import**
(`intrinsic_split` and `calib_tables` included), `pytest rubinwork` unchanged at **361
passed**, no `SyntaxWarning`, and a word-level prose audit showing every difference is
intended. `intrinsic_split.py` took 11 statement splits and is on the MIW path, which is
why the `rubinwork` suite was re-run rather than trusted.

One harmless surprise: the fixer also edited `bin/` copies of two scripts. **`bin/` is
gitignored** — scons generates it from `bin.src/` — and `setup.cfg` excludes `bin` from
linting, so nothing was committed and the copies regenerate.

### What the 62 were, before the fix (kept for the record)

Every one was in a file `tickets/RSO-809` never touches, so none of it was this branch's
debt.

| code | count | what |
|---|---|---|
| W505 | 50 | doc/comment lines over the 79-column `max-doc-length` |
| E702 | 11 | compound statements on one line (semicolons) |
| W605 | 1 | invalid escape sequence `'\ '` |

| file | findings |
|---|---|
| `python/lsst/ts/intrinsic/wavefront/intrinsic_split.py` | 17 |
| `python/lsst/ts/intrinsic/wavefront/ccd_height.py` | 10 |
| `bin.src/run_make_calib_tables.py` | 9 |
| `python/lsst/ts/intrinsic/wavefront/batoid_intrinsic.py` | 7 |
| `bin.src/ingest_calib_tables.py` | 6 |
| `tests/test_intrinsic_split_nullspace.py` | 4 |
| `python/lsst/ts/intrinsic/wavefront/calib_tables.py` | 4 |
| `bin.src/run_mktable.py` | 2 |
| `wcsutils.py`, `common/zernike_names.py`, `bin.src/plot_calib_tables.py` | 1 each |

### Still for Aaron

**Mention the shared-file churn in the PR description.** The branch now touches **22**
files across the two lint commits, all shared with upstream, so the next `develop` merge
may conflict in the reflowed docstrings. The scripts used are in `scratch/lintfix/`
(`fix_w505.py`, `fix_stmt.py`).

## The package and stack inventory, 2026-10-15 (read-only)

Taken on S3DF with the environment the `20261013`/`20261014` builds ran in. Nothing was
changed, fetched, committed or re-setup.

**Stack in use now:** `lsst_distrib` **`w_2026_39`**, eups tag `g00e868bf88+054f4cc332`,
conda env **`lsst-scipipe-13.1.0-exact`** (Python 3.13).

**Latest weekly on S3DF cvmfs:** **`w_2026_41`**, conda env
**`lsst-scipipe-14.0.0-exact`**. `w_2026_40` is the last weekly still on the 13.1.0 env, so
moving past 40 is a **conda-environment major bump, not just a weekly tag**. The only
numerical library that moves in it is **scipy, 1.16.3 to 1.18.1**; numpy 2.4.6, astropy
8.0.1, pandas 2.3.3 and pybind11 3.0.4 are unchanged between the two envs.

### Packages the products use

All checkouts live under `/sdf/home/r/roodman/u/LSST/packages/`. "ahead/behind" is
`git rev-list --left-right --count @{u}...HEAD`.

| package | branch | commit | `git describe` | clean? | behind / ahead | in the stack |
|---|---|---|---|---|---|---|
| `ts_intrinsic_wavefront` | `tickets/rubinwork-rottelpos` | `a6394e7` | `v2.1-14-ga6394e7-dirty` | **DIRTY**, 1 file | 0 / 0 | not in `lsst_distrib`; local `setup -r` only |
| `ts_wep` | `develop` | `1ea1ae1d` | `v17.14.1-1-g1ea1ae1d` | clean | **9 behind** / 0 | not in `lsst_distrib`; local `setup -r` only |
| `ts_ofc` | `tickets/RSO-1007` | `ca1eb91` | `v4.8.1-11-gca1eb91` | clean | no upstream for this branch | not in `lsst_distrib`; local `setup -r` only |
| `summit_utils` | `u/aaronroodman/guider-fixes` | `e57e0f4` | `w.2026.34-21-ge57e0f4` | clean | 0 / 0 | not in `lsst_distrib`; local `setup -r` only |
| `ts_config_mttcs` | `develop` | `ae5cca1` | `v0.22.5-alpha.1-39-gae5cca1` | clean | 0 / 0 | not in `lsst_distrib`; local `setup -r` only |
| `ts_aos_analysis` | `develop` | `b2a9d02` | `b2a9d02` (no tag) | clean | 0 / 0 | not in `lsst_distrib`; local `setup -r` only |
| `rubin-work` (this repo) | `main` | `89d142f` at session start | `pre-products-reorg-2026-10-09-130-g89d142f` | clean at session start | 0 / 0 | `pip -e`, `rubinwork 0.1.0` |

**`danish`, `batoid`, `batoid_rubin` and `galsim` are NOT local checkouts** — they come from
the stack's conda env, so they move only when the weekly moves. Versions read from each
package's `__version__` / `_version.py`, not from its `dist-info` directory — `batoid` and
`batoid_rubin` both ship a `0.0.0` dist-info in **both** weeklies, a build-metadata
artefact that is not the version:

| package | version in `w_2026_39` (13.1.0 env) | version in `w_2026_41` (14.0.0 env) |
|---|---|---|
| `danish` | 1.3.0 | 1.3.0 |
| `batoid` | 0.9.0 | 0.9.0 |
| `batoid_rubin` | 0.7.0 | 0.7.0 |
| `galsim` | 2.8.5 | 2.8.5 |

**The only dirty package is `ts_intrinsic_wavefront`**, one file,
`python/lsst/ts/intrinsic/wavefront/intrinsic_build_plots.py` (+18 / -5 lines): it rebuckets
`plot_dof_median_summary` off the DOF **labels** instead of the global 50-DOF indices, so a
reduced DOF set (22 or 12 DOF) no longer runs a global index off the end, plus an
empty-panel guard. A self-contained bugfix with no caller-visible API change.

**`ts_wep`'s 9 commits behind `develop`** are one merge plus bugfixes and timing logs,
touching `calcZernikesNeuralTask.py` (+150 lines), `calcZernikesTask.py`,
`generateDonutDirectDetectTask.py` and their tests — a saturated-image guard, an
`empty()` OOD-column crash fix, and runtime logging. **No signature change is visible in
the diffstat.** The checkout's last `git fetch` was **2026-10-05**, 10 days before this
inventory, so `develop` may have moved further.

**A gap worth naming:** a product manifest records `git_commit` and `git_dirty` **for
rubin-work only** — `catalog.manifest()` returns no stack, eups-tag, conda-env or
external-package version at all. A build made on `w_2026_39` and a build made on a newer
weekly are today indistinguishable from their manifests.

## The upgrade plan, 2026-10-15 — AUTHORIZED; step 1 EXECUTED

Kept as written for its reasoning and its "what could break" list. Two parts are now
superseded by decision 8 above: the weekly is settled (**`w_2026_41`**), and step 4's
"pull `ts_wep` separately from the stack move" is **overridden** — both happen in one step.
Step 1 below is done; see the step 1 section above for what actually happened.

### Step 1 — the one dirty package

`ts_intrinsic_wavefront`: **commit it**, do not stash and do not discard. It is a real
bugfix, it is already needed by any reduced-DOF build, and a calibration built against a
`-dirty` tree records a `git_sha` nobody can reproduce. Commit **on the current branch**,
`tickets/rubinwork-rottelpos`, which is where the other 14 commits since `v2.1` already sit
and which is 0 ahead / 0 behind its upstream. Then **tag** it, so the next
`intrinsic_zernikes` version has a reproducible `git_sha` rather than
`v2.1-14-ga6394e7-dirty`.

Every other package is clean — nothing to stash or discard anywhere.

### Step 2 — the order

1. **Commit and tag `ts_intrinsic_wavefront`** on `tickets/rubinwork-rottelpos`. Do this
   **first and on the current stack**, so the tag names a tree that is known to work.
2. **Decide the weekly: `w_2026_40` or `w_2026_41`.** These are not equivalent risks.
   `w_2026_40` keeps `lsst-scipipe-13.1.0-exact`, so it is a stack-code-only move with no
   conda churn. `w_2026_41` brings `lsst-scipipe-14.0.0-exact` and scipy 1.16.3 to 1.18.1.
   **Recommendation: `w_2026_41`**, taking the conda bump deliberately now rather than
   hitting it unprepared on a later weekly — but `w_2026_40` is the low-risk option if
   items 5 to 7 need to land first.
3. **Re-setup the six local packages on the new weekly** (`ts_intrinsic_wavefront`,
   `ts_wep`, `ts_ofc`, `summit_utils`, `ts_config_mttcs`, `ts_aos_analysis`). They are
   `setup -r` checkouts, so each needs its eups table satisfied against the new stack; this
   is where a dependency-version conflict surfaces, not at import time.
4. **Decide `ts_wep` separately from the stack move.** Pulling its 9 commits and moving the
   weekly at once makes a failure ambiguous. **Move the stack first, re-setup `ts_wep` at
   `1ea1ae1d` unchanged, confirm the smoke test, then pull `develop`** and re-run the smoke
   test. Two steps, two attributable outcomes.
5. **Smoke test, in this order:**
   - `python -m pytest rubinwork` — **355 tests** over 10 files
     (`test_{bounce_tables,build_optical_state,catalog,coadds,cwfs_tables,dof_lut,fam_tables,guider_moments,miw,value_added}.py`),
     collected clean this session in 0.99 s. Fast, no Butler, no EFD.
   - **One small rebuild that must match an existing build at zero tolerance.**
     `dof_lut` for `d12-A_50_34_i_5rot` is the right choice: its DAG is 3 jobs, it reads the
     `miw` build through the catalog rather than recomputing it, and it now plans cleanly
     (confirmed by `snakemake -n` this session). Build it under a new build name and
     compare against the `20261015` DAG's declared outputs column by column. **A non-zero
     difference here is the signal to stop and attribute it before rebuilding anything
     else.**
6. **Then items 5 to 7** — `coadds`, `dof_lut`, `bounce_tables` — all on the one new
   environment, which is the whole reason for sequencing them after the upgrade.

### What could break

- **scipy 1.16.3 to 1.18.1 (`w_2026_41` only).** The real numerical risk in the move.
  `danish`'s fitting and `rubinwork.smatrix.regularized_inversion` both sit on scipy
  linear algebra and optimizers; a changed default (tolerance, LAPACK driver, or an
  `optimize` method's convergence) shifts fit coefficients slightly without raising
  anything. This is exactly what the zero-tolerance rebuild in step 5 is for.
- **`ts_wep` against a newer stack.** Its tasks subclass `lsst.pipe.base` PipelineTask and
  use `lsst.daf.butler` connections; a weekly can change a connection class or a task
  config field and break `setup -r` or the first `runQuantum`. Its own 9 pending commits
  touch `calcZernikesNeuralTask` and `generateDonutDirectDetectTask` — **the donut and
  Zernike path the FAM products depend on** — so pull them separately as step 4 says.
- **`danish` 1.3.0 is unchanged in version between the two weeklies**, so no danish API
  change is expected from the move itself. The risk is indirect: danish on a new scipy, and
  danish against a `ts_wep` that has moved. A same-version package on a different numerical
  substrate can still change its output.
- **`summit_utils` is on `w.2026.34-21-ge57e0f4`**, a branch cut from the `w_2026_34`
  weekly. Moving the stack 7 weeklies forward is the most likely `setup -r` failure of the
  six, since it is the one pinned closest to a specific weekly.
- **`ts_ofc` is on `tickets/RSO-1007` with no upstream configured**, so there is no
  remote to compare against and no way to tell from git whether it is current. Worth
  resolving before it matters.

### Products whose manifests should record the stack change

Everything rebuilt after the move, which is **`coadds`, `dof_lut` and `bounce_tables`**
(items 5 to 7). They would then be the only products in the tree built on a different stack
from `fam_tables@20261014`, `cwfs_tables@20261013` and `miw@20261014` — and, per the gap
named in the inventory above, **nothing in the manifest would say so**. Two things follow:

1. **Add the stack to the manifest before the rebuilds**, not after: the eups `lsst_distrib`
   tag, the conda env name, and the commit of each `setup -r` package. `manifest.write`
   already takes an `extra` dict, so this is a small change in
   `rubinwork/products/manifest.py` and needs no schema migration.
2. **`intrinsic_zernikes` v4** is the other one. It was built on `w_2026_39` against
   `ts_intrinsic_wavefront` `a6394e7-dirty`; if it is ever ingested or superseded by a v5
   built after the upgrade, the stack and the package tag from step 1 are what make the two
   comparable.

## Expected difference 8, the ESS telemetry window: MECHANISM FOUND, 2026-10-15

Read-only investigation, no rebuild. **The earlier guess in the A2 writeup — "a narrower
EFD query window" — is wrong.** The window did not narrow. The ESS *source* changed, from a
per-visit raw-EFD query to a bulk ConsDB-transform read, and the loss is NULL cells in the
transform table.

**Where.** Both builds call the same function,
`lsst.ts.intrinsic.wavefront.intrinsics_lib.get_thermal_data`, reached through
`rubinwork/products/fam_tables/builders/run_attach_telemetry.py:356` (`_attach_thermal`,
which delegates rather than reimplementing the query). The change is **in the external
package, not in rubin-work** — `ts_intrinsic_wavefront` commit **`f45bfbf`**, "mktable
thermal: bulk ConsDB transform for ESS temps + TMA truss" (2026-07-28).

| | old path | new path |
|---|---|---|
| function | `get_ess_temperature`, `intrinsics_lib.py:1500` | `_thermal_from_consdb_transform`, `intrinsics_lib.py:1659` |
| source | raw EFD topic `lsst.sal.ESS.temperature`, field `temperatureItem0`, per SAL index | ConsDB `efd_lsstcam.exposure_efd`, columns `mt_salindex{111,112,113,301}_temperature_0_mean` |
| granularity | one query per visit per sensor (4 ESS + 2 truss round-trips per visit) | one bulk query per 800 exposures, keyed on `exposure_id` |
| window | `obs_start` to `obs_end`, `postPadding = DEFAULT_TEMP_TIME_WINDOW_SEC = 0.2 s`, then the mean over that window | whatever window the ConsDB transform used to form its per-exposure mean; not a parameter of this code |

So the window start and end are the same exposure bounds on both sides; what changed is
**who computes the mean** and **the 0.2 s post-padding, which the new path has no equivalent
of**. The old path's padding let a visit whose ESS sample landed just after `obs_end` still
return a value; the transform's per-exposure mean is NULL when no sample fell inside the
exposure.

**Measured on `d12-A_50_34_i`, 960 shared visits.** All four ESS columns lose the **same 89
visits** and gain 0 — `cam_air_temp`, `m2_air_temp`, `m1m3_air_temp` and `outside_temp`
alike (counts, dimensionless). An identical loss set across four independent sensors rules
out a sensor-specific fault. The 89 are scattered over **15 nights** and are a minority of
each (11, 5, 7, 12, 8, 6, 6, 5, 15, 1, 1, 2, 3, 2, 5 visits per night), so it is per-visit,
not a date-range gap. Re-measured this session on `d12-A_50_34_i_5rot` as well: **111 of
1,126 shared visits lost, 0 gained** (counts, dimensionless), the same pattern on an
independent variant.

**Probed against ConsDB directly** (read-only, `efd_lsstcam.exposure_efd`):

| exposure set | exposures asked | `exposure_efd` rows returned | non-null `mt_salindex111_temperature_0_mean` |
|---|---|---|---|
| the 89 lost | 89 | 89 | **0** |
| 200 kept, as a control | 200 | 200 | 200 |

All counts dimensionless. Every lost exposure **has** a row in the transform table; the ESS
mean column in it is simply NULL. The join is doing exactly what it should.

**Deliberate change, not a regression.** `f45bfbf`'s own message states the reason: the
per-visit loop "ran the batch `mktable` out of wall time on large chunks", and it kept the
raw-EFD loops as a fallback. The intent was "same output columns and (`day_obs`,
`seq_num`) keying", and that holds for the 871 of 960 visits the transform can answer. The
89-visit coverage loss is an unintended side effect of a deliberate performance
change — the kind worth recording, not reverting. One real defect is visible in the design,
though: **the fallback is all-or-nothing.** `get_thermal_data` takes the transform path when
`_thermal_from_consdb_transform` returns anything at all (`intrinsics_lib.py:1763`), so a
visit with a NULL transform cell gets NaN and **never** gets a per-visit retry. A per-visit
fallback for the NULL rows, rather than a whole-query one, would close the gap.

**Recoverable without an EFD re-fetch? No — but the re-fetch is small and need not touch
`mktable`.** The values are not anywhere in the new tree or the old build's inputs; the old
tree's `fits.parquet` still *holds* them, so the cheapest repair is a join from the old
table onto the 89 (and 111) keys, with no EFD traffic at all. If instead the values are to
be re-derived, it is 89 visits x 4 sensors = 356 raw-EFD queries on
`lsst.sal.ESS.temperature`, seconds of work, not a `mktable` rerun. **Neither was done this
session** — difference 8 stays on the expected list as a known, bounded coverage loss.

### Standing warnings, kept so the next session does not relearn them

**Do not run this build under a forwarded X display.** The first attempt lost 8 of 9
`build_intrinsic` jobs to `X connection to localhost:10.0 broken` plus a pybind11 GIL
abort, because `DISPLAY` was set and matplotlib opened an X-backed backend. Run every
`miw` build as `env -u DISPLAY MPLBACKEND=Agg snakemake ...`; with that, all 9 grids built
in about 12 min of wall-clock time.

**Spend rule, Aaron 2026-10-14:** the monthly-pace stop is **suspended** (a quota increase
is requested). The **per-session $50 stop stays**. Keep sessions cheap: narrow reads, no
re-reading of long handoff sections, terse results and handoff entries.

**A correction worth keeping**, because it is the kind of error that costs a session: an
earlier version of this section claimed four `cwfs_tables` builds were compared and
"recorded below". **They were not** — two were built, none compared, nothing recorded. Do
not trust a "done and recorded" claim in a handoff without the numbers beside it.

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

7. **The `miw` field-edge ring continuation / NaN-rim fill** (`ts_intrinsic_wavefront`
   commit `3f66904`, 2026-10-05; accepted new behaviour, Aaron 2026-10-14). Field-edge
   cells of the 71x71 map grid at **r >= 1.729 deg of field angle** may differ: the new
   build fills rim cells the old build left NaN (`_5rot` `Z4_OCS`: 124 NaN cells old, 0
   new) and keeps **3,985** valid cells where the old kept **3,831**, with old-only = 0
   cells, so the field extent does not grow. The **interior agrees to 9.2e-04 µm of
   wavefront**, which is difference 2's own bound, and `intrinsic_split_rms` gains four
   columns naming the mechanism (`n_null_rings`, `n_continued_rings`,
   `min_ring_coverage_seen`, `max_null_dim`). Compare `intrinsic_split_maps` on the shared
   cells; a flattened whole-map comparison is a mask-alignment artifact, not a result.
   Whether the edge ring itself is trustworthy is the **`miw_hull` study's** question
   (`aos/docs/status/miw_hull_scope.md`), not a 3c blocker.

8. **The ESS telemetry window** (found 2026-10-14 in the `d12-A_50_34_i` `fits.parquet`
   comparison, A2). The 7 `attach_telemetry` ESS columns — `m1m3_air_temp`,
   `m2_air_temp`, `outside_temp`, `tma_truss_temp_mxmy`, `tma_truss_temp_pxpy`,
   `cam_air_temp`, `cam_m1m3_delta_t` — are present on both sides but NaN on more visits
   in the new build: **89 of the 960 shared visits go finite-old to NaN-new, 0 the
   reverse**, the same 89 on all 7 columns, scattered across all 15 nights and a minority
   of each. On the 871 both-finite visits the values move by at most **0.2281 degC**
   (`outside_temp`). This is **not** difference 6 (gradients and `lut_dof5..9`) and **not**
   difference 4 (which drops columns outright; decision 3 keeps these). Likely a narrower
   EFD query window dropping visits near a telemetry gap. Not yet traced to a commit.

   **CONFIRMED on a second variant, 2026-10-14.** `d12-A_50_34_i_5rot` shows the identical
   signature: **111 of 1,126 shared visits** lose their ESS temperatures, **0 gain them**,
   the same 111 on every column, over 16 of 19 nights. Against the sibling's 89 of 960 that
   is 9.9% versus 9.3% of visits — a systematic change, not one build's artifact. Treat the
   9 columns as one group (the 2 `cam_*` are 2 of the 9), and expect the loss on any
   FAM-derived product rebuilt from `danish_1_2@20261014`.

| # | product / variant | compared against | expected differences | what blocks it |
|---|---|---|---|---|
| 1 | **DONE 2026-10-13** `fam_tables` **`danish_1_2`** | `aos/output/fam_processing/danish_1_2/`, and the phase 2 reference chunk `20251116_20251130` for the column-level check | 1, 3, 4, 5. Row **order** differs too (chunks are now per night, so `combine_*` concatenates 137 inputs rather than 10) — compare keyed on (`day_obs`, `seq_num`), never positionally | the embargo probe, then the 137-job set |
| 2 | **DONE 2026-10-13** `fam_tables` **`danish_1_3_v1000`**, **`danish_1_3_test`** (blitz, `run_blitz_mktable`) | `aos/output/fam_processing/danish_1_3_v1000/`, and the phase 2 A4 blitz reference `day_obs=20260428` | 1 (about **0.30 deg** here, the larger blitz offset), 3, 5. About an hour each, so interactive — no batch ask | item 1 landing; independent of the embargo night |
| 3 | **DONE 2026-10-14** `cwfs_tables` (4 variants, all promoted) | `aos/output/wfs_ingest/danish_1_2/<upstream_name>/` per variant | 1 only, plus the **`d12-refitWcs_2025` fold** (3b item 5): one date-keyed variant now covers 2025 and 2026, so the 2025 rows appear under `d12-refitWcs` and the retired `d10-wep17_3_0` entry is gone | nothing — can run alongside item 1 |
| 4 | `miw` (the 2 `d12` variants first; the 10 others have no complete old-tree build) | `aos/output/miw/<dir_name>/`, and the 3a reference in `_phase2_refbuild/miw/ref/` | 2 (the theta rotation, the headline number), 7 (the field-edge rim fill), and 1 carried through the visits it reads | items 1 and 3 |
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
## Variant tables for the products 3c has still to rebuild

Kept from the pruned 3a sections because items 4-7 of the comparison plan need them.

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
