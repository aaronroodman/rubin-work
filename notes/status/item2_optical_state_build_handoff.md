# Item 2, open-loop and deviation-recovered optical state: handoff to the implementing session

> **Status:** current · **Last updated:** 2026-10-02 · **Kind:** handoff (scoping session → implementing session)

The specification is **item 2 of [`notes/todos/todo-ideas.md`](../todos/todo-ideas.md)** —
read the whole item, not a summary. All 13 of its open questions are answered and the item
is unblocked. This document holds only what the scoping conversation established that the
item itself does not say: the findings behind three of its decisions, so the implementing
session does not re-derive them or quietly contradict them.

## Done and committed

No implementation has started. Two commits, both to the specification only:

| commit | what |
| --- | --- |
| `bbc0f64` | answered Q11, added Q13 for the RBR measurement-space mismatch, corrected item 2's scope |
| `cfedee2` | batoid-only variant list, comparison on DOF and image quality, full rerun of the 50/34 variant |

## In progress

Nothing. The working tree carries unrelated `thermal_focus/` modifications and untracked
files; none belong to this item.

## Next concrete action

Start the build order in the item's scope: move the two `olr/` functions into `aos/code/`
(fixing the sign, see below), write the corner-basis shim, add the guarded `50_34_rbr` entry
to `SCHEMES`, add the CWFS-median IQ column, cost one night, hand Aaron the three batch
submissions. Nothing gates this.

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
