# Item 5 — MIW on the v1000 pupil model, then under the RBR constraint

> **Status:** current · **Last updated:** 2026-10-05 · **Kind:** status (handoff)

Step A of [todo item 5](../todos/todo-ideas.md#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint):
rebuild the 50 degree-of-freedom / 34 v-mode (50/34) Measured Intrinsic Wavefront (MIW)
on Josh Meyers' v1000 pupil-model processing and compare it against the existing build on
the legacy pupil model. Scheme held fixed at 50/34; only the pupil model varies.

**Where it stands:** **step A is complete** (build 2026-10-05, all three comparisons done).
**Step B is wired and measured on one rotator bin**; what remains is the nine-bin batch
build, after deciding `n_iter`. Q8 is answered — RBR runs on `danish_1_3_v1000` only.

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

**Q8 answered by Aaron 2026-10-05: RBR runs on `danish_1_3_v1000` only**, the two MIW
agreeing too closely for two arms to be worth it.

RBR no longer lives in this repository. Per Aaron (2026-10-05), since
`ts_intrinsic_wavefront` is becoming an official LSST package, RBR had to go somewhere
equally official — `ts_ofc`, which MTAOS would ultimately use and which already holds the
quadratic `motion_penalty`. Two branches, both committed locally, **neither pushed**:

| repo | branch | commit | what |
|---|---|---|---|
| `ts_ofc` | `tickets/RSO-1007` | `979ec73` | `DoubleZernikeStateEstimator`, `range_bounded_recovery`, `OFCData.dof_ranges`, 29 tests |
| `ts_intrinsic_wavefront` | `tickets/RSO-809` | `957a54f` | `build_ofc_svd` delegates its SVD to `ts_ofc` |

`smatrix/code/regularized_inversion.py` stays as the derivation and cross-check reference,
**not** the implementation. `aos/code/miw/test_rbr_against_prototype.py` is the bridge
test — the one place that may import both.

Verified: the `ts_ofc` estimator reproduces `build_ofc_svd` bit for bit across the 50-DOF,
non-contiguous-keep and 22-DOF configurations; rebuilding the `danish_1_3_v1000`
`rot_-3_3` grid end to end reproduces the committed `intrinsic_grid.parquet` exactly;
RBR agrees with the prototype to 1.7e-13; the pre-existing `ts_ofc` suite passes untouched
(73 tests, 87 subtests).

**What remains is the full nine-bin build.** Steps 1, 3 and 4 below are done (see "Step B"
further down); step 2 is not, and it is a batch job.

1. ~~Decide how the build requests RBR.~~ Done: `build.range_bounded_recovery` in
   `mi_config.yaml`, entering at `_apply_uconstraint`.
2. **Build the `danish_1_3_v1000` RBR arm over all nine rotator bins.** Batch, so a hard
   MUST-ASK — the submit and monitor commands are ready and were handed over 2026-10-05.
   `n_iter` is already set to 15 on the RBR entries (see the caveat below); the
   unconstrained arm stays at 3 so its cached grids remain valid. A dry run plans 10 jobs
   and leaves the unconstrained arm alone.
3. ~~Compare arms on the achieved residual.~~ Done, `compare_rbr_arms.py`, using
   `lsst.ts.ofc.achieved_residual`.
4. ~~Report the MIW as inferred FWHM, DOF against `r_j`, and the change per DOF and
   v-mode.~~ Done on one bin; rerun on nine.

**The convergence caveat is the live issue.** RBR converges much more slowly than the
unconstrained recovery. At the configured `n_iter` 3 the unconstrained arm had settled to
8.92e-04 µm of wavefront between iterations (inside the 1.0e-03 µm tolerance) while the RBR
arm was still moving by 5.42e-03 µm. At `n_iter` 8 it was at 1.37e-03 µm, decaying roughly
as 1/iteration, having moved the MIW a further 0.0483 µm of wavefront (inferred FWHM to
0.3550 arcsec). The effect's direction and scale are robust — 0.3077 µm between arms
against 0.0483 µm of drift — but the one-bin numbers are an iterate. The change decays as
`n_iter**-1.066` (fitted over 8 iterations), crossing the 1.0e-03 µm tolerance near 10, so
the RBR entries are set to **`n_iter: 15`** (predicted 6.2e-04 µm). The unconstrained arm
stays at 3: it converged there, and raising it would invalidate its cached grids and the
step A comparison. Each iteration costs roughly 15 s per bin, so 15 is affordable.

Optional and not required: the `_legacy` third build would separate the pupil model from the
code version. Aaron declined it once (below); only revisit if step B produces a result that
actually turns on the attribution.

## Tried and rejected, and why

- **Putting RBR at the runner's `svd.dof(A_last)` call.** That was the plan, and it is the
  wrong place: the runner's DOF are a *diagnostic* written to `dz_fits.parquet` and never
  fed back into the build. The wavefront the build subtracts is formed inside
  `_apply_uconstraint`, per visit per iteration, as `U_eff @ (U_eff.T @ w)`. Changing only
  the runner would have reported range-bounded DOF beside an unchanged MIW.
- **Routing the unconstrained path through the new forward map.** Mathematically it is the
  same thing — `wavefront_from_dof(invert_truncated(w))` equals `U_eff @ (U_eff.T @ w)`, and
  that identity is now a `ts_ofc` test — but only to 9e-15 µm of wavefront, not bit for bit,
  because the round trip through physical DOF associates the products differently. Sharing
  the expression would have broken reproducibility of the committed build for no gain, so
  `_apply_uconstraint` keeps the direct projection when RBR is off.
- **Reading `dz_raw_*` columns from `dz_fits.parquet`.** There are none. The table has
  `dz_corr_*` (the subtracted fit) and `dz_resid_* = raw - corr`; the raw fitted wavefront
  is their sum. Stacking a non-existent prefix silently yields zeros, which made the first
  achieved-residual comparison measure everything against a zero reference and report RBR as
  *improving* the residual by 0.1216. Confirmed the reconstruction by checking the
  unconstrained arm satisfies `dz_corr == U U^T raw` to 1e-15 µm of wavefront.
- **Comparing both arms' residuals against one shared raw wavefront.** They do not have one.
  The build iterates, re-interpolating the MIW onto the donuts each pass, so the arms' fitted
  wavefronts diverge after iteration 1 — by up to 0.1174 µm of wavefront on `rot_-3_3`. Each
  arm's residual is now evaluated against its own fit, and reported as a fraction of it.

- **Swapping `OFCSvd` for `ts_ofc`'s existing `StateEstimator`.** The goal was one SVD in
  the code base, and the two looked equivalent. They are not interchangeable: `OFCSvd`
  decomposes a **double-Zernike slab** (focal order k by pupil Noll j), `StateEstimator`
  the sensitivity at **discrete sensor positions** with a noise covariance. The slab omits
  only 1.0e-05 of the matrix power, and v1–v9 agree at `|dot|` = 1.000000 — which is what
  made the swap look safe — but **past mode 9 the individual v-modes are unrelated**: v20
  at `|dot|` = 0.0040, v28 at 0.00019, singular values differing by up to 0.48 relative.
  The leading-34 subspace overlap is 0.9156, because the singular values cluster and
  `n_keep = 34` cuts through a cluster. So a sensor-position estimator cannot reproduce the
  DZ build. Aaron's call: keep the DZ basis and `k = 1..6` and get identical results, which
  is why `ts_ofc` gained a DZ estimator rather than the build changing basis. Do not
  re-propose the direct swap. Also: `truncate_index` is **12** in the shipped config, not
  34 — anything using `StateEstimator` defaults truncates far harder than this build.
- **Offsetting the sensitivity matrix's pupil axis by `znmin`.** Written that way first,
  and it is wrong but *silent* — it produces a well-formed decomposition of the wrong pupil
  terms, with `Sigma` off by 50.3. Both axes are **1-based index spaces with entry 0
  unused**: pupil column j is Noll j, focal row k is DZ order k, and row 0 and column 0 are
  identically zero. `ofc_svd.py`'s own docstring says so ("axis index equals the Noll
  index"); it was right and the offset was the error. Note this also means **`k_min = 1`**,
  not 0.
- **Reading `(k, j) = (1, 4)` as the camera-piston signature.** Used as the probe for the
  index convention, on the reasoning that piston gives field-constant defocus. The real
  peak is `(1, 1)` — piston moves mean OPD, so Noll 1 dominates and Noll 4 is about a
  quarter of it. The convention test now asserts what is actually true (index 0 zero on
  both axes, piston response at low k and low j) rather than that cell.
- **Trusting a docstring's filename for the weight convention.** The shipped weights file
  is `range0.5_fwhm-0.15.yaml`, but the exponent is **-0.5, not -0.15** — Aaron flagged the
  typo. Checked: `w = r^0.5 f^-0.5` reproduces the shipped weights to 8.2e-16 relative,
  while `f^-0.15` is off by a factor of 3.4. Renaming the file is a config change and was
  left alone. Also note `ts_ofc`'s configured default is `range-fwhm.yaml` (`w = r * f`),
  which is **not** what is in use — the two differ by up to 6868 in DOF units, so the DZ
  estimator has to be handed the weights explicitly.
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

## Step B: wired, measured on one bin, full build outstanding

The RBR code is written, committed and verified (branches above). The build can now
request it, and it has been run on one rotator bin. `aos/code/miw/check_dof_ranges.py`
measures how far a build's states fall outside `r_j` and changes nothing; it is the
before-picture.

**How the build requests RBR** (answers step 1 of the previous "next concrete action"):
`mi_config.yaml` gained `build.range_bounded_recovery`, with `build.rbr_kappa` and
`build.rbr_power`, **default off**. Two new `mi_name` entries on `danish_1_3_v1000`,
`pathA_50_34_i_rbr` and `pathA_50_34_i_rbr_5rot`, so the unconstrained arm's outputs
stay. Snakemake picks them up with no Snakefile change — a dry run plans the same 10 jobs
(9 `build_intrinsic` + 1 `intrinsic_split`) as step A.

RBR enters at `_apply_uconstraint` in `measured_intrinsic.py`, **not** at the
`svd.dof(A_last)` diagnostic in the runner. That diagnostic is written to
`dz_fits.parquet` and never fed back; the wavefront the build actually subtracts is
`U_eff @ (U_eff.T @ w)`, formed per visit per iteration. RBR replaces it with
`wavefront_from_dof(dof_range_bounded(w))` — a new `ts_ofc` forward map, commit
`6e61aa4`. The runner's diagnostic DOF now also report the range-bounded state when RBR
is on, so the table matches the MIW.

Verified RBR-off is a true no-op: rebuilding `rot_-3_3` reproduces the committed
`intrinsic_grid.parquet` bit for bit, and `dz_fits.parquet` on 671 of 682 numeric
columns, the other 11 being the same Z16/Z19 `_err` columns as the RSO-809 test at
5.4e-20 µm of wavefront (one ULP on error bars of order 1e-04 µm; the coefficients are
exact). This is pre-existing fit non-determinism, not from the RBR change.

**The result on `rot_-3_3`, 49 visits** — written up in
[`aos/docs/studies/miw.md`](../../aos/docs/studies/miw.md), "What RBR does to the MIW",
not duplicated here. Short form: RBR works (worst `|d_j|/r_j` 60.25 → 1.713, B1_20's
median 47.28 → 0.881) and the MIW grows a lot — 0.2764 µm of wavefront RMS over the
field, inferred FWHM 0.1774 → 0.3147 arcsec, which is **50 times the 0.0053 µm the pupil
model moved it**. So the recovery constraint dominates the pupil-model choice, and step
A's attribution caveat is not the limiting uncertainty for step B. Mechanism: RBR moves
the smallest-sigma retained v-modes (v33 at sigma 0.0231 by a median 3.85), subtracts
0.8784 of the amplitude the unconstrained arm removed, and leaves the rest in the MIW.

New code: `aos/code/miw/compare_rbr_arms.py`, the arm comparison — DOF against `r_j`,
achieved residual, MIW as inferred FWHM, and the change per DOF and per v-mode.

### Carried commits for step B

| repo | branch | commit | what |
|---|---|---|---|
| `ts_ofc` | `tickets/RSO-1007` | `6e61aa4` | `wavefront_from_dof` + estimator method, 5 tests |
| `ts_intrinsic_wavefront` | `tickets/RSO-809` | `a4716a4` | the build can subtract a range-bounded state |

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
