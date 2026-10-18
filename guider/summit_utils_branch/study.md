---
study: summit_utils_branch
topic: guider
status: done
question: "Does the summit_utils branch u/aaronroodman/guider-fixes contain every guider commit that the summit runs on deploy-summit?"
answer: "Yes: guider-fixes holds all 8 deploy-summit guider commits by patch-equivalence, so no cherry-pick is needed; origin/main has none of them."
uses: {}  # none
started: 2026-08-06
---

# `summit_utils_branch`: the summit_utils guider-fixes branch against deploy-summit

> **Reference:** [`summit_utils_branch_state.md`](summit_utils_branch_state.md) (git-verified state) · [`deploy_summit_merge_note.md`](deploy_summit_merge_note.md) (why the summit's detection is more robust than a weekly's)

## Question

Does the summit_utils branch u/aaronroodman/guider-fixes contain every guider commit that the summit runs on deploy-summit?

## Current answer

Yes: guider-fixes holds all 8 deploy-summit guider commits by patch-equivalence, so no cherry-pick is needed; origin/main has none of them.

## Log

### 2026-10-11 — backfilled from docs and history

Backfilled 2026-10-18 from the two reference docs (formerly `guider/docs/status/` and `guider/docs/`). Done in the phase 5 triage; the branch-state doc stays as reference. Verified with `git cherry` on 2026-10-11, remote refs as of the 2026-09-03 fetch.

The merge note counted 9 `deploy-summit` commits; 8 of them touch `guiders/`, the count the branch-state doc uses. The note now says so.
