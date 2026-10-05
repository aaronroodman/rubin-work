# Item 5 step A — MIW on the v1000 pupil model

> **Status:** current · **Last updated:** 2026-10-05 · **Kind:** status (handoff)

Step A of [todo item 5](../todos/todo-ideas.md#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint):
rebuild the 50 degree-of-freedom / 34 v-mode (50/34) Measured Intrinsic Wavefront (MIW)
on Josh Meyers' v1000 pupil-model processing and compare it against the existing build on
the legacy pupil model. Scheme held fixed at 50/34; only the pupil model varies.

**Where it stands:** **step A is complete.** The build ran 2026-10-05 and all three
comparisons are done; the result is below. Step B (RBR) is next and Q8 is answered — it
runs on `danish_1_3_v1000` only.

## Done and committed

Commit `7e82312`.

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

- `aos/output/miw/danish_1_3_v1000_A_50_34_i/build/rot_*/` — the nine per-rotator-bin
  grids, and `aos/output/miw/danish_1_3_v1000_A_50_34_i_5rot/` — the four split products
  including a 64-page `intrinsic_split.pdf`. Batch log
  `aos/logs/batch_20261005_094450.out`, rule log
  `aos/logs/danish_1_3_v1000_A_50_34_i_5rot.intrinsic_split.log`.
- `aos/output/miw/danish_1_3_legacy_vs_v1000/` — the eight comparison products listed
  above.

## The build, and the step A result

The build ran in batch 2026-10-05 09:45 to 09:51, 10 of 10 steps, no errors: 9
`build_intrinsic` grids plus 1 `intrinsic_split`, no `combine_*` (the hand-built tables are
terminal inputs, as the Snakefile documents). Products in
`output/miw/danish_1_3_v1000_A_50_34_i_5rot/`, 3985 field-point rows x 44 columns, |O|
telescope-fixed RMS 0.0534 µm of wavefront and |C| camera-fixed 0.0106 µm.

Both builds select the **same 205 visits** in the five rotator bins, on a field grid
identical to 1e-12 deg. Per-bin counts 36/48/49/36/36.

The numbers, the two readings of them, and the per-annulus and per-term breakdowns are
written up in **[`aos/docs/studies/miw.md`](../../aos/docs/studies/miw.md)**, section "What
the comparison found" — not duplicated here. The short form: the pupil model moves the MIW
by 0.1027 of amplitude and the inferred FWHM by +0.0033 arcsec, but the difference is
astigmatism-led (Noll 6 and 5 carrying 0.4032 and 0.2969 of its power) with the spherical
terms carrying only 0.0143 — *less* than a pure retrieval change does, so **it is not a
pupil-rim signature**. The subtracted optical state moves as well. Either the baffle is
being absorbed into the fit or the shift is the confounded `danish` version change; these
products cannot separate the two.

Products: `output/miw/danish_1_3_legacy_vs_v1000/` — `miw_legacy_vs_v1000_OCS.pdf` (21
pages, one per Noll term), the matching `_CCS.pdf` (near-empty by construction, see below),
two `_summary.parquet`, `pupil_radial_profile_OCS.parquet`,
`pupil_term_share_OCS.parquet`, `build_dof_compare.parquet`,
`build_vmode_compare.parquet`.

## Next concrete action

**Step B — RBR on the MIW, `danish_1_3_v1000` only** (Q8, answered by Aaron 2026-10-05:
the two MIW agree too closely for two arms to be worth it). The scope is the step B bullet
list in [item 5](../todos/todo-ideas.md). Key constraints already settled by the item's
answered questions: call the shared solver in
`smatrix/code/regularized_inversion.py` rather than copying it, use
`RBR_DEFAULTS = {'kappa': 4.0, 'power': 3}` to begin with (Q2), and compare arms on the
achieved residual `regularized_inversion.achieved_residual(dW, d, svd)` rather than the
subspace projection (Q5). The RBR penalty goes inside the build's per-visit optical-state
recovery, which lives in the **external** `ts_intrinsic_wavefront` package, not in
`aos/code/` — so it needs a `scons` re-run after editing.

Optional and not required: the `_legacy` third build would separate the pupil model from the
code version. Aaron declined it once (below); only revisit if step B produces a result that
actually turns on the attribution.

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
- **Passing the target path behind `--` in batch mode.** Dies on
  `MissingRuleException: No rule to produce --config`. In **batch** mode `run_snake.sh`
  already hoists flags ahead of its own `--config` and appends the `--` itself, so a
  passed-through `--` lands as a positional target. Pass the path bare:
  `./run_snake.sh --mode batch output/miw/<P>_<M>/intrinsic_split_maps.parquet`. Local mode
  is the opposite — it passes everything straight through, so there a target *does* need its
  own `--`. The script's usage header now says so.
- **Reading the CCS comparison as a result.** `compare_miw_versions.py --coord CCS` returns
  0.0000 µm of wavefront on all 20 terms except Z4. That is not a null result: the split
  log's `OCS-only (C forced to 0)` line lists exactly those terms, so this build
  configuration permits a camera-fixed component only on Z4. OCS is the informative product
  for this pair; do not quote the CCS zeros as agreement.

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

Step B — Range-Bounded Recovery (RBR) on the MIW — is now unblocked, and runs on
`danish_1_3_v1000` only (Q8). See **Next concrete action** above for the settled
constraints. `aos/code/miw/check_dof_ranges.py` already measures how far a build's states
fall outside `r_j` and changes nothing; it is the before-picture for step B.

## Expectations to carry into step B

The step A result sets up two things worth knowing before RBR runs:

- The bending modes that move most between the two pupil models — B1_16 at 1.1276 of its
  allowed range `r_j`, B1_12 at 0.4968 — are the same modes `check_dof_ranges.py` finds
  sitting far outside range in both builds. RBR acts hardest exactly where the two builds
  disagree most, so a step B "improvement" could partly be RBR suppressing the pupil-model
  sensitivity rather than a real gain. Report both arms on the same visits.
- The 0.0033 arcsec of inferred FWHM between the pupil models is the scale any RBR-induced
  FWHM change should be compared against. A change much below it is not separable from the
  pupil-model choice.
