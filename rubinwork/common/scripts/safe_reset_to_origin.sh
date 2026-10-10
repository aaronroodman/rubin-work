#!/usr/bin/env bash
# ----------------------------------------------------------------------
# safe_reset_to_origin.sh
#
# Discard local commits / staged / working-tree changes and reset the
# current branch to match origin/<branch>.  Idempotently aborts any
# rebase or merge in progress.  **Leaves untracked files alone** —
# never runs `git clean`.
#
# Useful on the RSP after `gitpull` (or similar) leaves the repo in
# a "needs merge" state because of local notebook output edits.
#
# Usage:
#   ./safe_reset_to_origin.sh             # current branch
#   ./safe_reset_to_origin.sh main        # explicit branch
#   REPO=/path/to/repo ./safe_reset_to_origin.sh
# ----------------------------------------------------------------------
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "REPO not found: $REPO" >&2; exit 1; }

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
echo "Repo:   $REPO"
echo "Branch: $BRANCH"

# Abort anything in progress (ignore failures)
git rebase --abort  2>/dev/null
git merge  --abort  2>/dev/null
git cherry-pick --abort 2>/dev/null

git fetch origin --prune

echo
echo "Resetting tracked files to origin/$BRANCH (untracked files will be kept)..."
git reset --hard "origin/$BRANCH"

# Drop any stashes (likely created by an auto-stash pull wrapper).
if [[ -n "$(git stash list)" ]]; then
    echo
    echo "Existing stashes (these were likely created by your pull script):"
    git stash list
    echo "Dropping all stashes — Ctrl-C now to keep them."
    sleep 3
    git stash clear
fi

echo
echo "Done.  Working tree state:"
git status
git log --oneline -3
