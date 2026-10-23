# ts_ofc RSO-1007: double-Zernike state estimator and Range-Bounded Recovery

Status: PR open, not merged (2026-10-23).
PR: https://github.com/lsst-ts/ts_ofc/pull/124 (`tickets/RSO-1007` into `develop`).
Ticket: https://rubinobs.atlassian.net/browse/RSO-1007

Commits: `56fac64` (estimator and RBR), `f21cef5` (forward map), `fbb53ed` (barrier
weight and full-DOF solve).

Terms: DOF is degree of freedom, the 50-element optical state in the ts_ofc ordering
(0-4 M2 hexapod, 5-9 camera hexapod, 10-29 M1M3 bending, 30-49 M2 bending). DZ is
double Zernike: focal-field order `k` by pupil Noll index `j`. RBR is Range-Bounded
Recovery. IRLS is iteratively reweighted least squares.

`StateEstimator` is unchanged. RBR is opt-in; nothing changes unless it is called.

## Files

| file | change |
|---|---|
| `python/lsst/ts/ofc/dz_state_estimator.py` | new: `DoubleZernikeStateEstimator`, `LABELS_50DOF`, `DOF_UNITS_50` |
| `python/lsst/ts/ofc/range_bounded_recovery.py` | new: RBR solver and helpers |
| `python/lsst/ts/ofc/ofc_data/ofc_data.py` | new method `OFCData.dof_ranges()` |
| `python/lsst/ts/ofc/scripts/generate_normalization_weights.py` | uses `dof_ranges()` instead of its own copy |
| `python/lsst/ts/ofc/__init__.py` | exports the two new modules |
| `tests/test_dz_state_estimator.py` | new, 15 tests |
| `tests/test_range_bounded_recovery.py` | new, 27 tests |
| `doc/news/RSO-1007.feature.rst` | news fragment |

## `DoubleZernikeStateEstimator`

Estimates the optical state from DZ wavefront coefficients, instead of from the
wavefront at discrete sensor positions as `StateEstimator` does. It takes the SVD of
`S @ diag(w)`, where `S` is the DZ sensitivity slab over `k_min..k_max` and the chosen
pupil Noll indices, and `w` the normalization weights.

Why a new class: the two decompositions share only 0.9156 of their leading-34-mode
subspace (dimensionless, mean of sum cos² over principal angles). The singular values
cluster, and the truncation boundary cuts through a cluster. Compare the two by
subspace, not mode by mode, past about mode 9.

Constructor: `DoubleZernikeStateEstimator(ofc_data, iZs, k_min, k_max, n_keep,
n_dof=None, normalization_weights=None, log=None)`.

- `iZs`: pupil Noll indices; need not be contiguous. They index the pupil axis
  directly, with no `znmin` offset.
- `n_keep`: an `int` keeps modes `0..n-1`; a list keeps exactly those modes, so
  single v-modes can be dropped from the middle.
- `n_dof`: `None` (all 50), a count, or a list of DOF indices.
- `normalization_weights`: an array, a yaml path, or `None` for
  `OFCData.normalization_weights`.

Methods:

| method | returns |
|---|---|
| `project_amplitudes(wavefront)` | u-mode amplitudes, µm of wavefront |
| `vmodes(amplitudes)` | v-mode amplitudes `a_i / sigma_i`, dimensionless |
| `dof(amplitudes)` | DOF by truncated SVD |
| `dof_range_bounded(wavefront, ...)` | DOF by RBR, one solve per visit |
| `wavefront_from_dof(dof)` | DZ wavefront a state reproduces, µm |
| `forward_operator()` | sensitivity per physical DOF over retained modes, `(n_kj, n_dof)` |
| `dof_ranges()` | allowed range per used DOF |
| `dof_labels()` | DOF names and units |

The input wavefront is `(n_visits, n_kj)` in `kj_grid` order. Non-finite entries count
as zero.

## Range-Bounded Recovery

Truncation alone lets a recovered DOF go far outside the stroke the hardware can reach.
RBR minimizes, over normalized DOF `x` with physical `d = w * x`:

    || dW - S x ||^2 + tikhonov^2 ||x||^2 + lambda * sum_j (|d_j| / (kappa r_j))^(2p)

The barrier is well below 1 (dimensionless) while `|d_j| / r_j < kappa`, and rises
steeply past it. It leaves DOF inside their range almost untouched, unlike a quadratic
penalty.

Defaults: `kappa = 4.0` (dimensionless, knee as a multiple of range), `p = 3`
(dimensionless exponent), `lambda = 1.0` (µm² of wavefront). `dof_range_bounded` takes
each one from the call, then from the controller config keys `rbr_kappa`, `rbr_power`
and `rbr_penalty_weight`, then from these defaults. None of the shipped config files
set these keys yet.

Effect, from the commit message, on one representative wavefront: the largest
`|d_j| / r_j` (dimensionless) falls from 59.30 under truncation to 1.28 under RBR. The
achieved residual rises by 5.5e-02 (dimensionless, fractional increase).

Solver (`invert_range_penalty`): IRLS with a backtracking line search. A fixed half-step
at `p = 3` falls into a limit cycle and stalls at a relative change of 1.6e-02
(dimensionless).

Two solve modes:

- **Retained-mode** (default): solves in retained mode coefficients, starting from the
  truncated answer. With the penalty off it returns the truncated solution exactly. But
  confining `d` to the span of the retained `V` couples the DOF. Bounding one DOF that
  is far out of range drags others that are well inside theirs, even ones whose own
  barrier term is around 1e-10 (dimensionless).
- **Full-DOF** (`full_dof=True`, or `forward_operator=` to the solver): solves over all
  used DOF, with the barrier as the only regularizer, from a cold start at `d = 0`.
  This removes the coupling. The barrier is flat below the knee, so different starts
  give DOF differing by a few tenths of `r_j` while the wavefront `S d` agrees to well
  under 1 %. Use it to subtract a wavefront, not to quote a DOF amplitude.

`lambda` matters: the data term is µm² of wavefront and the barrier is dimensionless.
`lambda = 1` suits inputs whose worst DOF is about 10 times its range. At about 50 times,
the barrier exceeds the data term by about nine orders of magnitude, so lower `lambda`.
Scaling `lambda` is the same as moving the knee:
`kappa_eff = kappa * lambda^(-1 / (2p))`.

Other public functions in `range_bounded_recovery`:

- `invert_truncated`: the unregularized truncated-SVD inversion.
- `wavefront_from_dof`: the forward map `U Sigma V^T N^-1 d`. For a truncated
  recovery this equals the projection onto the retained u-modes; for an RBR recovery it
  does not, because RBR cannot reach the part of the wavefront that needs out-of-range
  motion.
- `achieved_residual`: RMS of `dW - S d` through the same rank-limited operator, in µm
  of wavefront. Use this, not the projection residual, to compare RBR with truncation.

## `OFCData.dof_ranges()`

Returns the allowed range `r_j` over all 50 DOF. The units are µm for hexapod
translations and bending-mode amplitudes, and arcsec for hexapod rotations. Rigid-body
DOF use `rb_stroke`. For bending modes it is the force range / 20, divided by the
largest absolute entry of that mode's force column. The 20 is there because the shipped
bending-mode force matrices are normalized to 20 µm amplitude.

`generate_normalization_weights` now calls this, so there is one definition. The
weights are unchanged. The commit message reports that two independent routes to `r_j`
agree to 1.5e-15 (dimensionless, relative difference).
