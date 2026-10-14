# Phase 3 handoff: the other products — move, fix, rebuild

> **Status:** 3a and 3b complete; **3c IN PROGRESS**, 3 of 8 comparison-plan items done.
> **`fam_tables` is DONE**: `danish_1_2@20261014` (3,385 visits over 106 nights, the same
> as the old tree) and `danish_1_3_{test,v1000}@20261013` (966 visits each) are all
> `current`, each confirmed through `catalog.load()`. The **M1M3 gradient defect is FIXED
> and the fix is promoted** (commit `bef17ba`, build `20261014`): the four gradients now
> equal the value-added database at **max abs difference 0.000e+00 degC/m** on all 3,006
> visits that have them, with every other column identical to `20261013` at zero
> tolerance. **`cwfs_tables` is DONE**: all four variants compared clean and promoted to
> `current@20261013`; the only differences are `rotator_angle` (max abs 0.4446 deg of
> camera rotator angle on all four) and, on `d12-refitWcs`, the 2025 fold (+2,168 visit
> and +62,554 donut rows, every old row kept). **`miw` (item 4) now WAITS ON AARON'S
> REVIEW**: the comparison difference that stopped it is **explained and measured** — it is
> the field edge (`ts_intrinsic_wavefront` commit `3f66904`), the interior agrees to
> 7.2e-04 µm of wavefront, and `vmode_13` is a degenerate-subspace rotation. Validation
> PDFs and `summary.md` for both `danish_1_2` variants are in
> `$RUBINWORK_DATA/studies/phase3c_validation/miw/`. `d12-A_50_34_i@20261014` is built as
> far as the sidecars and `d12-A_50_34_i_5rot@20261014` through `intrinsic_split`; neither
> is promoted and **no `miw` variant is `current`**. Two things still need Aaron —
> `d12-tarts` has no intra-focal centroids so its corner sidecar fails and `refit_mi` never
> ran, and whether the ring continuation is accepted new behaviour for the
> expected-difference list. Items 5 to 7 are blocked behind `miw`. Questions 6 and 7 in
> "Decisions for Aaron".
> This file was pruned 2026-10-14; the 3a/3b narratives are in commit `ebbc426`.
> · **Last updated:** 2026-10-14 · **Kind:** working state (handoff)

Phase 3 of `notes/status/organization_plan.md`: move every remaining product's code (3a),
apply the intended fixes (3b), then rebuild every product fresh into the new tree (3c).
Read the Phase 3 part of section 8 there for the step list; this file holds the state
between sessions.


> **History pruned 2026-10-14.** The full 3a per-product move narratives (scope, variant derivations, steps 1-4 for every product) and the per-item 3b records are in commit `ebbc426f9d8077e4b9ef475d147dcb5904e99311`, this file at 4,315 lines. Everything a session needs to resume 3c is kept below: decisions, open defects, the expected-difference lists, the comparison plan, the finished-product results, and the variant tables for the products 3c has still to rebuild.

## Decisions for Aaron

Per section 12 of the plan: questions needing your judgment that do not block the rest of
the step. Answered items move to "Decisions taken".

6. **`d12-tarts` has no intra-focal centroid columns, so `miw` cannot build its corner
   sidecar** (open, found 2026-10-14; **`miw` item 4 and therefore items 5 to 7 are
   waiting on it**). `run_make_intrinsic_sidecar.py` raises
   `ArrowInvalid: No match for FieldRef.Name(centroid_x_intra)`, and because the rule is
   an input to `refit_mi` the whole build stops short of `fits.parquet`. The same defect is
   already in "Open, not yet resolved" as `tarts`'s missing corner sidecar.
   **Options:** (a) drop `d12-tarts` from the `miw` corner-sidecar set, by the same
   skip-and-report idiom the Snakefile already uses for a variant with no build —
   recommended, since neither the old tree nor the 3a reference has a `tarts` sidecar, so
   nothing is lost against the comparison; (b) add the intra centroids to the `d12-tarts`
   `cwfs_tables` build, which is a `cwfs_tables` rebuild and a bigger change.
   **Waiting on it:** `miw`, and items 5 to 7 behind it.

7. **The external builder's ring continuation changed the valid field-grid cell set**
   (open, found 2026-10-14; **`miw` promotion is waiting on it**). The new build keeps
   **3,985** of the 5,041 cells of the same 71x71 grid where the old kept **3,831** — 154
   more — and `intrinsic_split_rms` gains 4 columns that name the mechanism
   (`n_null_rings`, `n_continued_rings`, `min_ring_coverage_seen`, `max_null_dim`).
   The commit is **identified** (`3f66904`, 2026-10-05) and the shared-cell comparison is
   **done** — see the validation section; the difference is confined to r >= 1.729 deg and
   the interior agrees to 7.2e-04 µm of wavefront.
   **Options:** (a) accepted new behaviour of `ts_intrinsic_wavefront`, written onto the
   expected-difference list — recommended, since the commit is a deliberate fix with its
   own tests and the interior is unchanged; (b) a regression. **Waiting on it:** every
   `miw` `set-current`,
   and items 5 to 7. Numbers in "`miw` item 4" below.

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

## Next concrete action

**`miw` item 4 still WAITS ON AARON, 2026-10-14** — on the two questions below and on
`d12-tarts`, not on a crash and not on spend. The comparison difference that caused the
original stop is now **explained and measured** (section above): it is the field edge, from
`ts_intrinsic_wavefront` commit `3f66904`. Nothing is half-written: the
`d12-A_50_34_i@20261014` build directory holds 15 of its 18 outputs,
`d12-A_50_34_i_5rot@20261014` holds its 4 `intrinsic_split` outputs and no manifest, and
**no `miw` variant is `current`**. No process is running and the snakemake lock is free.

**Two questions for Aaron, both in "Decisions for Aaron" above.** Items 5 to 7 (`coadds`,
`dof_lut`, `bounce_tables`) all take `miw` as their upstream, so they are blocked behind
them.

1. **`d12-tarts` has no intra-focal centroids**, so its corner sidecar cannot be built and
   `refit_mi` never ran. Exclude it from the corner-sidecar set, or add the columns?
2. **The external builder's ring continuation changed** (`ts_intrinsic_wavefront` commit
   `3f66904`), filling the NaN rim and, on the 9-bin variant, moving the lattice: 3,831 ->
   3,985 cells, plus 4 new `intrinsic_split_rms` columns. Accepted new behaviour, to be
   written onto the expected-difference list, or a regression?

The re-comparison is **done** — shared-cell `intrinsic_split_maps` and shared-key
`zk_intrinsic`, both variants, numbers and PDFs in the validation section above. What is
left is the judgement call, not the measurement.

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
