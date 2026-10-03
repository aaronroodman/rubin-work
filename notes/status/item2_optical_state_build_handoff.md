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

Confirm the principal-angle claim (next section) on the as-built state estimators. It is
step 0 because the DOF-versus-v-mode decision rests on it. Then the build order in the
item's scope: move the two `olr/` functions into `aos/code/`, write the corner-basis shim,
add the guarded `50_34_rbr` entry to `SCHEMES`, write the DZ optical state, cost one night,
hand Aaron the three batch submissions.

## Three findings to verify, not trust

These came from reading code during scoping. Each one is load-bearing for a decision
already written into the item, and each is worth re-checking rather than inherited.

**1. The principal angle is quoted, not measured.** The item's caveat section says a
22/12-versus-50/34 comparison read off `v_modes` is not like-for-like, because the
principal angle between the retained degree-of-freedom (DOF) subspaces is 4.768 deg for
`standard_22`/12 but 89.951 deg for `all_50`/34 — effectively orthogonal. Both numbers come
from a **docstring in `aos/code/aos_state.py`**, not from a measurement made during
scoping. The decision to compare on DOF and image quality (Aaron, 2026-10-02) follows from
them. Measure the angle on the estimators as actually built. If it does not reproduce, stop
and raise it rather than proceeding, because the metric of record changes.

The underlying reason the angle is large: `recover_optical_state` is **hybrid by design**.
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
the v-mode half already lives in `build_optical_state.py` and `aos_state.py`, and the DZ
optical state Q10 asks for has to be written from scratch. Two defects to fix in the same
move: the bare `OFCData(name='lsst')` with no normalization assertion, and field angles
named `field_angles_ccs` where `aos_state` requires the Optical Coordinate System (OCS) at
rotator zero.

One reference implementation is worth keeping: `olr/code/run_olr.py` asserts the identity
`olr_deviation == olr_opd - intrinsic` on every corner row and prints the check into its
log. Carry that assertion into the moved code and run it per visit at build time.
