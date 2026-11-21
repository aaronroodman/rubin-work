# ts_ofc RSO-1007: PR #124 handoff

> **Status:** PR open, awaiting review; local branch not pushed · **Last updated:** 2026-11-20 · **Kind:** handoff

What the PR adds: [`notes/ts-ofc-rso-1007/note.md`](../ts-ofc-rso-1007/note.md).
PR: https://github.com/lsst-ts/ts_ofc/pull/124 (`tickets/RSO-1007` into `develop`).
Ticket: https://rubinobs.atlassian.net/browse/RSO-1007
Checkout: `~/u/LSST/packages/ts_ofc`.

## Done

- On GitHub (`origin/tickets/RSO-1007`): `56fac64`, `f21cef5`, `fbb53ed`, then `ad5a56f`,
  which fixed the CI failures (mypy, ruff W505 doc length, ruff-format). All checks passed
  on `ad5a56f` (GitHub lint, news fragments, Jenkins). No reviewer requested, no reviews.

## In progress

The local branch is 8 commits ahead and 4 behind `origin/tickets/RSO-1007`. It is the
same four commits rebased onto `develop` after PR #125 (`2f51613`, `78419ec`, `108a9b0`,
`328ed83`), plus `1cf0f47` "Hexapod tilt DOF units are deg, not arcsec". That commit
changes `DOF_UNITS_50` and the docstrings of the new RBR code; it is not pushed.

`develop` has moved again (PR #123 RSO-1009, PR #126). A test merge of the local branch
onto `origin/develop` has no conflicts (2026-11-20). No uncommitted files in ts_ofc.

## Next

1. `gh` is logged out on this node; run `gh auth login` first.
2. `git rebase origin/develop`, run `pytest tests/test_dz_state_estimator.py
   tests/test_range_bounded_recovery.py`, then `git push --force-with-lease`. The force
   push is needed because of the rebase.
3. Wait for CI, then request a reviewer: `gh pr edit 124 --add-reviewer <login>` (who
   reviews is Aaron's call).

## Constraints and rejected options

- **No comment-only or style-only hunks** in files the PR does not otherwise change. Comments
  change only where code changes. `1cf0f47` touches only code the PR adds, so it is fine.
- Run CI's lint on the PR files only. Local mypy and ruff use different settings from
  CI's pre-commit (`.ts_pre_commit_config.yaml`: `line-length = 110`,
  `max-doc-length = 79`). A full-repo local run flags things CI does not.
- Local mypy flags `range_bounded_recovery.py` (`v_retained` may be `None`); CI does not.
  Left alone on purpose.
- Hexapod tilts (DOF 3, 4, 8, 9) are deg, not arcsec. Never multiply by 3600.
