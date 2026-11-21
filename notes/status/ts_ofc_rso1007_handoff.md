# ts_ofc RSO-1007: PR #124 handoff

> **Status:** PR open, review requested from gmegh; branch pushed, CI green · **Last updated:** 2026-11-21 · **Kind:** handoff

What the PR adds: [`notes/ts-ofc-rso-1007/note.md`](../ts-ofc-rso-1007/note.md).
PR: https://github.com/lsst-ts/ts_ofc/pull/124 (`tickets/RSO-1007` into `develop`).
Ticket: https://rubinobs.atlassian.net/browse/RSO-1007
Checkout: `~/u/LSST/package## Done

- 2026-11-20: force-pushed the rebased branch (`ad5a56f...1cf0f47`): the four PR commits on
  `develop` `3cd8929` (`2f51613`, `78419ec`, `108a9b0`, `328ed83`) plus `1cf0f47` (tilt
  units deg). Then `fb2bb26` "Reformat DOF_UNITS_50 with ruff", a normal push.
- CI on `fb2bb26`: GitHub lint, news fragments and Jenkins all pass. Full local suite on
  `w_2026_47`: 115 passed, 87 subtests passed.
- PR description updated (tilt-unit bullet, rebase base, test results).
- 2026-11-21: gmegh requested as reviewer. Old history kept in `backup/RSO-1007-pre-rebase`.

## In progress

Nothing. No uncommitted files in ts_ofc; local equals `origin/tickets/RSO-1007`.

## Next

1. Wait for gmegh's review; answer comments.
2. `develop` moved again on 2026-11-20 (PR #123 RSO-1009, PR #126 release merge); the branch
   is 7 commits behind it. Aaron (2026-11-21): do not rebase now, wait for the review. Rebase
   only if the reviewer asks or a conflict appears; it needs `--force-with-lease`.

n's call).

## Constraints and rejected options

- **No comment-only or style-only hunks** in files the PR does not otherwise change. Comments
  change only where code changes. `1cf0f47` touches only code the PR adds, so it is fine.
- Run CI's lint on the PR files only. Local mypy and ruff use different settings from
  CI's pre-commit (`.ts_pre_commit_config.yaml`: `line-length = 110`,
  `max-doc-length = 79`). A full-repo local run flags things CI does not.
- Local mypy flags `range_bounded_recovery.py` (`v_retained` may be `None`); CI does not.
  Left alone on purpose.
- Hexapod tilts (DOF 3, 4, 8, 9) are deg, not arcsec. Never multiply by 3600.
- Shortening a string can make `ruff format` collapse a multi-line expression: `1cf0f47`
  (arcsec -> deg) failed both the lint check and Jenkins (which runs `pre-commit run --all`
  before the tests) until `fb2bb26`. Run `ruff format --check .` before pushing.
