## Ticket

RSO-809 — measured intrinsic wavefront (MIW) build support for the rubin-work AOS products.

## What this is

The `tickets/RSO-809` branch, rolled forward to `181a51e`. It is 21 commits ahead of
`develop` (20 plus one merge of `develop`) and 0 behind.

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
- `0f0a2ad` bad_fit: more than 5 coefficients with |DZ| > 2 um, not any one
- `f2e63da` Intrinsic sidecar: support unpaired corner tables (no pixel centroids)
- `ee7160c` DOF_UNITS_50: hexapod tilts are deg, not arcsec
- `787c778` build_ofc_svd, run_build_intrinsic: optional OFC config dir
- `181a51e` intrinsic_split: add Z27, Z28 to the Noll (n, m) table

`f9bafa2` and `4c77b90` are lint-only and change no numerical result. `f9bafa2` covers the
files this branch touches; `4c77b90` clears the rest of the repo, including one latent
`W605` invalid escape sequence. `flake8 .` over the whole repo now reports **0 findings**.

@gmegh: `4c77b90` reflows comments and splits compound statements in your
`intrinsic_split.py`, `calib_tables.py` and `tests/test_intrinsic_split_nullspace.py`. It is
there only because the PR lint check runs flake8 over the whole repo, and `develop` is not
linted. No logic in those files changes.

The five newest commits:

- `0f0a2ad`: a single large coefficient is usually a deliberate hexapod or bending-mode
  perturbation, so a fit is flagged bad only when more than `max_large_coeffs` (default 5)
  coefficients exceed 2 um. Added to `flag_bad_fits`, `run_dz_fit` and the MIW build.
- `f2e63da`: the sidecar takes the CCD height at the pixel each donut's CCS field angle maps
  to on its own half-sensor (new `ccd_height.field_to_pixel`), keyed on `thx_OCS`, `thy_OCS`.
  Paired tables are unchanged.
- `ee7160c`: label only. The hexapod tilt DOF (3, 4, 8, 9) are in deg, as in ts_ofc v4.0.0,
  the Trim and the EFD; no value changes.
- `787c778`: optional OFC config directory, to select another sensitivity matrix. The
  default is unchanged.
- `181a51e`: Noll indices 27 and 28 in `NOLL_NM`.

## Tests

`python -m pytest tests` in the package: **10 tests passed, 0 failed** on every run below.

- At the branch head `181a51e`, on `lsst_distrib` `w_2026_47` (eups tag
  `g00e868bf88+1a8aebc4ba`, conda env `lsst-scipipe-14.1.0-exact`, Python 3.13.18):
  **10 passed, 0 failed** in **1.3 s** of wall clock. The PR lint and rebase checks pass.

- At `4c77b90`, on `lsst_distrib` `w_2026_42` (eups tag
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
the head: the later commits landed after those builds, so moving the tag would misstate
which tree produced them. `git describe` on the head reads
`rubinwork-2026-10-15-7-g181a51e`.

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

