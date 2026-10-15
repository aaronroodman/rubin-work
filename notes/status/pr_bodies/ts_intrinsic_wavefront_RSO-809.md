## Ticket

RSO-809 — measured intrinsic wavefront (MIW) build support for the rubin-work AOS products.

## What this is

The `tickets/RSO-809` branch, rolled forward to `4c77b90`. It is 16 commits ahead of
`develop` and 0 behind. The last four commits are new; the other 12 were already on the
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
- `f9bafa2` Fix flake8 lint across the files RSO-809 touches
- `4c77b90` Clear the remaining flake8 findings: whole repo now lints clean

The last two are lint-only and change no numerical result. `f9bafa2` covers the files this
branch touches; `4c77b90` clears the rest of the repo, including one latent `W605` invalid
escape sequence. `flake8 .` over the whole repo now reports **0 findings**.

## Tests

`python -m pytest tests` in the package: **10 tests passed, 0 failed** on every run below.

- At the branch head `4c77b90`, on `lsst_distrib` `w_2026_42` (eups tag
  `g00e868bf88+ec6e0cf799`, conda env `lsst-scipipe-14.0.0-exact`, Python 3.13.16):
  **10 passed, 0 failed** in **1.5 s** of wall clock.
- Earlier, at `529713b` on `lsst_distrib` `w_2026_39` (eups tag
  `g00e868bf88+054f4cc332`, conda env `lsst-scipipe-13.1.0-exact`, Python 3.13):
  **10 passed, 0 failed**, run twice — once before the roll-in on
  `tickets/rubinwork-rottelpos`, and once after the fast-forward on `tickets/RSO-809`.

So the suite passes both across the lint commits and across the stack move from
`w_2026_39` to `w_2026_42`.

## The tag marks `529713b`, not the branch head

The annotated, non-release tag **`rubinwork-2026-10-15` marks `529713b`** — the tree the
rubin-work phase-3c MIW product builds actually used. It is deliberately **not** moved onto
the head: `f9bafa2` and `4c77b90` landed after those builds, so moving the tag would
misstate which tree produced them. `git describe` on the head reads
`rubinwork-2026-10-15-2-g4c77b90`.

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
- This is review of the accumulated branch rather than of a single change. The PR title
  still names only the first commit (the per-donut sidecar); the commit list above is the
  actual scope.

Generated with AI

Co-Authored-By: SLAC AI
