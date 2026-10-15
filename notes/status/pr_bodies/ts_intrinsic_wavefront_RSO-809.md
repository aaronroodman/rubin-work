## Ticket

RSO-809 — measured intrinsic wavefront (MIW) build support for the rubin-work AOS products.

## What this is

The `tickets/RSO-809` branch, rolled forward to `529713b`. It is 14 commits ahead of
`develop` and 0 behind. The last two commits are new; the other 12 were already on the
branch and are included here because the branch has not been reviewed yet.

Changes, oldest first:

- `923adb3` Add per-donut measured-intrinsic sidecar (Phase 2 step 3)
- `9ac08c1` run_dz_fit: opt-in --min-detectors to relax the per-visit CCD-count cut
- `7ce5729` run_dz_fit: add --no-quality-cut (fit all visits, cut at use)
- `f45bfbf` mktable thermal: bulk ConsDB transform for ESS temps + TMA truss
- `9f9a98f` run_build_intrinsic: honor day_obs_min/max from mi_config
- `f4f23b6` combine_parquets: drop 0-row sentinel inputs before schema unification
- `aead18c` mi_config: add dir_name, and let the runners be told their paths
- `db057dc` Encode the filter band as a dot on the arrow shaft, and add a 75 deg elevation bin
- `957a54f` Take the sensitivity-matrix SVD from ts_ofc
- `a4716a4` Let the MIW build subtract a range-bounded optical state
- `5c5ccfa` Add the 50/50 RBR build scheme
- `00d0920` Merge remote-tracking branch 'origin/develop' into tickets/RSO-809
- `a6394e7` rotator_angle: rotTelPos first, ConsDB fallback, source column
- `529713b` plot_dof_median_summary: bucket DOF by label, not global index

## Tests

`python -m pytest tests` in the package, on `lsst_distrib` `w_2026_39`
(eups tag `g00e868bf88+054f4cc332`, conda env `lsst-scipipe-13.1.0-exact`, Python 3.13):
**10 tests passed, 0 failed**, run twice — once before the roll-in on
`tickets/rubinwork-rottelpos`, and once after the fast-forward on `tickets/RSO-809` at
`529713b`.

The branch head is also tagged `rubinwork-2026-10-15` (annotated, non-release) as the
reference point the rubin-work phase-3c products were built against, before that work
moves to `w_2026_41`.

## What rubin-work depends on

Four commits on this branch are load-bearing for the rubin-work AOS product builds:

- `a6394e7` **rotator_angle: rotTelPos first, ConsDB fallback, source column.** The
  products need the rotator angle taken from `rotTelPos` with ConsDB only as a fallback,
  and need the `source` column to record which one was used.
- `3f66904` **Fix the OCS/CCS null space at the field edge in the intrinsic split**
  (reached via the `00d0920` merge of `develop`). The field-edge rim fill in the
  measured-intrinsic tables.
- `f45bfbf` **mktable thermal: bulk ConsDB transform for ESS temps + TMA truss.** The ESS
  temperature source used by the thermal columns.
- `529713b` **plot_dof_median_summary: bucket DOF by label, not global index.** Without
  it, a reduced DOF set (22 or 12 DOF) indexes a 50-DOF array off the end and the build's
  summary plot raises.

## Notes for the reviewer

- Base is `develop`, not `main`: `main` holds a single "Initial commit" and the `v2` and
  `v2.1` release tags were both cut on the `develop` line.
- Draft, for review of the accumulated branch rather than of a single change.

Generated with AI

Co-Authored-By: SLAC AI
