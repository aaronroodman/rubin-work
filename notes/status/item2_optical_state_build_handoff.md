# Item 2, open-loop and deviation-recovered optical state: handoff

> **Status:** current · **Last updated:** 2026-10-03 · **Kind:** handoff (implementing session → batch submission)

The specification is **item 2 of [`notes/todos/todo-ideas.md`](../todos/todo-ideas.md)** —
read the whole item, not a summary. All 13 of its open questions are answered.

**The code is done and verified on one night; the full-span batch builds are what remain.**
This document holds what the code and git history do not say: the three decisions Aaron took
at implementation kickoff that override the written spec, the findings behind the scoping
decisions, what was found during implementation, and what was tried and rejected. Read the
kickoff decisions first — a reader of the older spec text would otherwise assume the opposite
sign and the wrong image-quality metric.

## Done and committed

The code is **complete and verified on one night**. Only the full-span batch builds remain,
and they need Aaron to submit them.

| commit | what |
| --- | --- |
| `bbc0f64` | answered Q11, added Q13 for the RBR measurement-space mismatch, corrected item 2's scope |
| `cfedee2` | batoid-only variant list, comparison on DOF and image quality, full rerun of the 50/34 variant |
| `196d930` | **spec corrections**: the OLR sign, the CWFS-median image-quality metric, no DZ state |
| `9d8e39d` | `aos/code/open_loop.py` + `aos_state.CornerSvdShim` + `test_open_loop.py` (10 tests) |
| `ce6b5b6` | the builder: `50_34_rbr` with its guard, the `_olr` and `fwhm_cwfs_arcsec` columns, `--telemetry-db` |
| `a0096de` | archived the superseded `olr/code/olr.py`, left a guard module in its place |
| `eef744a` | same-file `--telemetry-db` fix; `schema.md` and `build_progress.md` |
| `c426a8f` | `olr/docs/scheme_comparison.md`; fixed the sign in `olr/README.md` |

## In progress

Nothing uncommitted of this item's. The working tree carries unrelated `thermal_focus/`
modifications and untracked files, plus `notes/README.md`; none belong here.

Two things sit in the data area, outside git:

- `value_added/output/scratch/archive/v50_34__batoid__consdb_v1_pre_item2.parquet` — all
  90,695 pre-item-2 rows, count-verified, 95 MB. **Delete when this work closes.**
- `value_added/output/scratch/item2_cost_{22_12,50_34,50_34_rbr}.duckdb` and
  `item2_5034.duckdb` — the costing databases behind every number in
  `olr/docs/scheme_comparison.md`. Disposable once the full-span build lands.

`v50_34_rbr__batoid__consdb_v1` is registered in the **main** database and holds one real
night (20260318, 916 rows) from the costing run. The full-span rerun will overwrite it.

## Next concrete action

**Hand Aaron the three batch submissions** (commands are in the session summary; regenerate
with `--dry-run` if lost). Before the `50_34` one, get his go-ahead to delete the 90,695
archived rows. Then merge the shards, refresh `build_progress.md` with realized counts, and
redo `olr/docs/scheme_comparison.md` on the full sample — it is one night today.

Order matters only for `50_34`: the other two variants are empty, so they can go first or in
parallel.

## Three decisions taken at the start of the implementing session, 2026-10-02

These came from Aaron during implementation kickoff and **override the specification where
they conflict**. Item 2's text has been updated to match; they are repeated here because
each reverses or narrows something a reader of the older text would otherwise assume.

**A. The OLR sign. `olr/code/olr.py` is wrong, so the move is a sign fix, not a port.**
Two quantities differ by an overall sign and both are built:

| quantity | value | what it is |
| --- | --- | --- |
| optical state | `Trim − Deviation` | the DOF defining the visit's optical state; the `thermal_focus` convention (`MEASURED_SIGN = -1.0`) generalized from v1 to all v-modes |
| **OLR output** | `Deviation − Trim` | what would have been present with the loop open — the sign the OLR columns carry, in DOF, v-mode and Zernike space alike |

Aaron's reasoning, which settles it without reference to any implementation: a deviation is
equivalent to some DOF vector, and the Trim is applied with the **opposite** sign to drive
that deviation toward zero. So `Trim − Deviation` is the optical state and the open-loop
reconstruction is its negative. `olr/code/olr.py` instead *adds* the correction
(`zk_opd + sens_mat @ trim`).

**Do not take `run_olr.py`'s identity check as validating the sign.** The identity
`olr_deviation == olr_opd - intrinsic` holds for *either* sign, because `intrinsic` is
carried through unchanged and cancels. It still catches basis and zero-padding errors, so
carry it across — but it is not sign evidence, and the earlier handoff text implying the
forward direction was "verified" was wrong on this point.

**B. The image-quality metric is the median over the four CWFS, not over the focal plane.**
Applies to the **deviation-recovered** arm only, not the OLR arm. This work is CWFS-only
with no FAM, and four corners do not uniquely determine a DZ field, so the optical state is
evaluated from the corner wavefronts directly with **no DZ state anywhere in the item** —
which also retires the old scope line about "the DZ optical state is new code". Rationale:
the optical state in the science sensors is poorly known, and extrapolating there would fold
that uncertainty into the IQ number, so it is evaluated only where the wavefront is measured.

This is **not** the metric any other wavefront-IQ number in the repository uses. All of
those go through `aos/code/aos_fwhm.py`, whose `fp_grid` / `focal_basis` / `fp_fwhm`
evaluate a DZ field on a focal-plane grid to `FP_RADIUS = 1.75 deg` and median over *that
grid*; its callers (`run_bounce.py`, `run_wfs_dof_compare.py`) have a DZ `dW` from FAM or a
fitted field. The ts_wep `convertZernikesToPsfWidth` conversion is shared, the evaluation
domain is not, and the two numbers must not be substituted for each other. Stored as a
column on `optical_state` (additive migration, as `v_modes_lut`/`v_modes_trim` were) so the
comparison stays a self-join.

**C. `$TS_CONFIG_MTTCS_DIR` is fine in batch — no change needed.** It resolves to
`/sdf/home/r/roodman/u/LSST/packages/ts_config_mttcs`, and `/sdf/home/r/roodman/u` is a
symlink to `/sdf/group/rubin/u/roodman` (→ `/sdf/data/rubin/user/roodman`), a real shared
path visible on batch nodes. The root `CLAUDE.md` warning concerns the RSP-internal
`/home/r/roodman/u/...` form, not this one. `aos_state.resolve_ofc_config_dir` stays as is.

## Three code findings

These came from reading code during scoping. The first is context; the second and third are
load-bearing for correctness and worth re-checking rather than inherited.

**1. The 22/12 and 50/34 v-mode subspaces differ a lot, and that is expected.** The
principal angle between the retained degree-of-freedom (DOF) subspaces is 4.768 deg for
`standard_22`/12 but 89.951 deg for `all_50`/34 — effectively orthogonal. Both numbers are
quoted from a docstring in `aos/code/aos_state.py` rather than measured during scoping.

**This does not gate anything.** Aaron's decision (2026-10-02) is to compare the schemes on
**recovered image quality first and DOF values second**, which are the quantities he cares
about; the schemes' v-mode subspaces being different is fine and is the reason v-modes are
not the metric of record. Report v-modes for continuity with what the summit reports. If
you want the angle as a descriptive number on the as-built estimators, it is cheap to
compute, but no decision waits on it and a different value changes nothing.

The reason the angle is large: `recover_optical_state` is **hybrid by design**.
It inverts in the basis from `corner_recovery_basis` — the singular value decomposition
(SVD) of the 84-row corner-evaluated, Zernike-selected sensitivity matrix — but reports
v-modes in the `make_state_estimator` basis, which is `StateEstimator.Vh` over the full
899-row Double Zernike (DZ) slab with no field evaluation and no pupil-Zernike selection.
Those are different bases. That is not a bug to fix here.

**2. The shim is safe only because `kj_grid` is never read.** Q13's answer is to present
`corner_recovery_basis` through the solver's interface. The mapping, one-to-one onto the
basis dict:

| `smatrix/code/regularized_inversion.py` reads | `corner_recovery_basis` returns |
| --- | --- |
| `U_eff` | `U` |
| `Sigma` | `s` |
| `V` | `V` |
| `n_keep_eff` | `n_modes` |
| `normalization_weights` | `norm_vector` |
| `dof_idx` | `dof_indices` |
| `kj_grid` | nothing — see below |

`kj_grid` has no counterpart, and the shim leaves it `None`. That is sound **only** because
`invert_range_penalty`, `dof_range_vector` and `achieved_residual` never touch it. That was
established by grepping `svd\.[A-Za-z_]*` across the module and reading the three
functions; it is a property of the code as of commit `cfedee2`, not a guarantee. Re-run that
grep. If any of the three has gained a `kj_grid` read, the shim needs a real grid or the
option-3 fallback in Q13 applies.

Two supporting points, both checked during scoping and both worth a glance:

- `dof_range_vector` back-derives the range as `r_j = w_j^2 * f_j`, where `f_j` is the
  **field-averaged quadrature over the full 50 DOF**. It therefore does not depend on which
  field rows the sensitivity was evaluated at, which is why the corner basis can use it
  unchanged — provided `dof_idx` is the corner basis's own `dof_indices`.
- Assert the shim's weights are the `REQUIRED_NORM_YAML` ones. A bare `OFCData('lsst')`
  carries the obsolete normalization, which **rotates** the v-mode basis rather than
  rescaling it, and fails silently.

Inverting in `StateEstimator.Vh` instead of the corner basis leaves an irreducible residual
floor of 2.3e-02 µm of wavefront root-mean-square; the corner basis closes to 2.1e-14 µm of
wavefront. So the corner basis is the correct inversion space, which is what makes option 1
"the standard approach for finding the optical state from the CWFS" (Aaron's phrasing in
Q13's answer).

**3. `corner_recovery_basis` caches on `id(state_estimator)`.** The cache is a plain dict
keyed by `id()`. CPython reuses an `id` after garbage collection, so a short-lived estimator
built per scheme could in principle be handed another scheme's basis. With three schemes
live in one build this is the subtlest correctness trap in the item: it would silently mix
the 22-DOF and 50-DOF bases and produce a plausible-looking wrong answer. Hold the
estimators for the life of each shard and check it explicitly. The cache exists for a real
reason — `get_sensitivity_matrix` costs about 270 ms per call, so rebuilding per visit would
cost roughly 5.8 hours over 76,577 visits — so do not simply remove it.

## Found during implementation — non-obvious, and not in the specification

Five things the spec did not anticipate. The first two were silent-wrong-answer bugs.

**1. A shard has no `visit_telemetry`, so the whole open-loop arm was NaN.** The builder
reads the Trim from the same connection it writes to, and `run_build.sh` gives each shard its
own fresh database. So a sharded build would have produced NaN in every `_olr` column *and*
in `v_modes_lut` / `v_modes_trim`, with no error. (The existing 90,695 rows have commanded
terms, so the original build cannot have been sharded.) Fixed with `--telemetry-db`, which
`run_build.sh` now always passes for `--what state`, and the builder **refuses to run** on an
empty telemetry source rather than writing NaN columns.

Consequence worth knowing: `--resume` used to find populated nights by joining
`visit_telemetry`, which in a shard is empty, so it would silently rebuild everything. It now
derives `day_obs` as `visit_id // 100000`, verified to hold for all 213,704 rows.

**2. `--telemetry-db` pointing at the database already open raises.** DuckDB refuses a second
connection to one file under a different read-only setting. The telemetry reader reuses the
write connection when the paths resolve equal (`_same_db`). This bites only in the
non-sharded case, which is how a by-hand single-night run is done.

**3. The solver needs `U_eff` sliced, not the full `U`.** `regularized_inversion` takes
`n_kept = U_eff.shape[1]` as the retained-mode count and raises on `rank > n_kept`. Handing
over the corner basis's full `U` (84 x 50) would have made the penalty-off limit a full-rank
solve rather than the truncated one the comparison needs — a plausible-looking wrong answer,
not a crash. `CornerSvdShim` slices to `n_keep`.

The handoff's `kj_grid` claim re-verified: it appears in that module in **docstrings only**,
never as a code read, across all of `invert_range_penalty`, `dof_range_vector` and
`achieved_residual`. `n_keep_eff` likewise — the solvers compute the default from
`U_eff.shape[1]` rather than reading the attribute. Only those three functions were audited;
`invert_oic` and the damped route were not.

**4. The carried-over identity check was vacuous as first written.** Running
`check_olr_identity(x, x + 0, 0)` with a zero intrinsic placeholder passes unconditionally. It
now runs against the real intrinsic, recovered as `opd - z_dev`, which is why
`measured_deviation` returns the raw OPD as a third value.

**5. A module-level `__getattr__` guard loses its message on `from X import Y`.** CPython
converts the `AttributeError` into an `ImportError` and discards the text, so the caller sees
only "cannot import name". Measured on this interpreter — and it applies to the **existing**
`aos_state` guard for `build_geom_svd` / `project_dofs_to_vmodes`, whose docstring claims the
message survives. That docstring is wrong; left alone rather than changed as a side effect of
this item, but worth a separate fix. The new `olr/code/olr.py` guard binds real callables
instead, so the import succeeds and the error arrives at the call site where it is actionable.

## Results so far, one night

`day_obs` 20260318, 915 of 916 visits recovered in all three schemes. Full tables in
[`olr/docs/scheme_comparison.md`](../../olr/docs/scheme_comparison.md).

| scheme | CWFS-median FWHM [arcsec] | median max `\|d_j\|/r_j` (dimensionless) | s/night (916 exp) |
| --- | --- | --- | --- |
| `22_12` | 0.4372 | — | 39.8 |
| `50_34` | 0.2428 | 52.44 | 39.1 |
| `50_34_rbr` | 0.2650 | 2.33 | 59.0 |

The physics headline: 22/12 to 50/34 buys 0.1929 arcsec on 99.9% of visits, but unconstrained
50/34 asks for DOF at a **median 52x and up to 227x the force-limited range**, so that image
quality belongs to a correction that cannot be applied. RBR bounds it to 2.33x for 0.0281
arcsec, about 15% of the gain. `kappa = 4` was taken from the bounce test for continuity, not
tuned for science visits — the parameter most worth revisiting on the full sample.

## Tried and rejected, and why

- **Projecting the corner measurement into DZ space** to reach `invert_range_penalty`
  through its existing interface. Rejected on physics: four field points cannot constrain
  the focal-plane DZ orders `build_ofc_svd` uses, so a regularized fit would feed a
  regularized solve, and the Range-Bounded Recovery (RBR)-versus-truncated DOF difference
  would then have two inseparable causes. This is Q13 option 2.
- **Writing a separate corner-space range-penalty solver.** Duplicates the iteratively
  reweighted least squares (IRLS) and contradicts the scope line about calling the shared
  solver. Q13 option 3, the fallback only if the shim needs real surgery.
- **Keeping the open-loop sensitivity matrix at 22 columns everywhere** (Q11). Rejected on
  its merits, not on cost: it would put the 50/34 open-loop state and its
  deviation-recovered state in different subspaces, defeating the comparison the item exists
  for. Related: `DEFAULT_DOF_INDICES` in `olr/code/olr.py` is a hand-written duplicate of
  `DOF_SETS['standard_22']` that can drift silently — it goes away rather than being moved.
- **A fourth variant-name axis for RBR** (Q5). Rejected in favour of the pseudo-scheme
  `50_34_rbr`, so no schema change. Accepted consequence: `scheme` no longer determines
  `n_dof` and `n_modes` uniquely.
- **Building the MIW intrinsic route here** (Q12). Deferred to item 6. Note the live trap
  this leaves: `efd_db.optical_state('v50_34__miw__consdb_v1')` returns an empty DataFrame
  rather than raising, so an analysis naming the unbuilt variant gets zero rows and no
  error.
- **Extending `v50_34__batoid__consdb_v1` night-by-night.** Rejected in favour of a complete
  rerun, so all three variants come from identical code against identical inputs and a
  scheme-to-scheme difference cannot be a build-vintage artifact.

## Rejected during implementation

- **Archiving `run_olr.py` along with `olr.py`.** `olr/Snakefile` has a live rule depending on
  both, so moving the driver would break the pipeline, and repurposing `olr/` is separate
  scope. Only the superseded functions moved; the module stayed as a guard, and the pipeline
  now fails loudly at the first call instead of emitting wrong-sign output.
- **Deleting rather than archiving.** Aaron's instruction (2026-10-03): move superseded things
  to an archive for later deletion. No archive directory existed, so `scratch/archive/item2/`
  was created for code and `value_added/output/scratch/archive/` for the row export.
- **Changing `aos_state.resolve_ofc_config_dir` for batch safety.** Unnecessary:
  `$TS_CONFIG_MTTCS_DIR` resolves to `/sdf/home/r/roodman/u/...`, and `/sdf/home/r/roodman/u`
  symlinks to `/sdf/group/rubin/u/roodman`, a real shared path. The `CLAUDE.md` warning is
  about the RSP-internal `/home/r/roodman/u/...` form.
- **Storing the image quality as a focal-plane median via `aos_fwhm.fp_fwhm`.** This is where
  the implementation was heading before Aaron's correction. It needs a DZ field, which four
  corners do not determine. See decision B.

## Two constraints that will interrupt the work

Both are standing rules in `CLAUDE.md`, repeated here because this item hits them.

- **Deleting anything needs Aaron's go-ahead at the time.** This item deletes twice: the
  superseded functions in `olr/code/olr.py` after the move, and the 90,695 existing rows of
  `v50_34__batoid__consdb_v1` before the rerun. Ask; do not fold either into a larger step.
- **Never submit the batch builds.** Hand Aaron the submit command and a monitoring command,
  both copy-paste-ready. Measure the per-night cost on one night first — three full-span
  builds over 366 nights is not a sizing to guess.

## What is narrower than it looks

`olr/code/` is **Zernike space only**: `build_olr_sensitivity_matrix`, `apply_trim`, the
corner stacking, and the pipeline around them. There is no DOF recovery, no v-mode
projection and **no DZ code at all**. So the move is those two functions and nothing more;
the v-mode half already lives in `build_optical_state.py` and `aos_state.py`. Per decision
B above there is **no DZ optical state to write** — the earlier claim that Q10 required one
is retired. Three defects to fix in the same move: the bare `OFCData(name='lsst')` with no
normalization assertion; field angles named `field_angles_ccs` where `aos_state` requires
the Optical Coordinate System (OCS) at rotator zero; and the **sign** (decision A).

Carry `olr/code/run_olr.py`'s identity assertion `olr_deviation == olr_opd - intrinsic`
into the moved code and run it per visit at build time — but see decision A: it is
insensitive to the sign, so it is a basis-and-padding check, not a sign check.
