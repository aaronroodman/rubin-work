#!/bin/bash
# sync.sh — Quick pull/push helper for RSP terminals
cd "$(dirname "$0")"

case "$1" in
    pull)
        echo "Pulling latest from GitHub..."
        STASHED=false
        if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
            echo "Stashing local changes..."
            git stash push -m "sync.sh auto-stash"
            STASHED=true
        fi
        if git pull --rebase origin main; then
            echo "✓ Up to date"
        else
            echo "✗ Rebase conflict — these files need manual resolution:" >&2
            git diff --name-only --diff-filter=U 2>/dev/null
            echo ""
            echo "To resolve:"
            echo "  1. Edit the conflicted files (look for <<<<<<< markers)"
            echo "  2. git add <resolved-files>"
            echo "  3. git rebase --continue"
            echo ""
            echo "Or to abort and return to your previous state:"
            echo "  git rebase --abort"
            if $STASHED; then
                echo ""
                echo "Note: your unstaged changes are saved in git stash."
                echo "After resolving, run: git stash pop"
            fi
            exit 1
        fi
        if $STASHED; then
            echo "Restoring stashed changes..."
            if git stash pop; then
                echo "✓ Local changes restored"
            else
                echo "✗ Conflict restoring stash — resolve manually, then: git stash drop" >&2
                exit 1
            fi
        fi
        ;;
    push)
        MSG="${2:-Update from RSP $(hostname) $(date +%Y-%m-%d)}"
        echo "Staging all changes..."
        git add -A
        echo ""
        git status --short
        echo ""
        echo "Committing: $MSG"
        if git commit -m "$MSG"; then
            echo "✓ Committed"
        else
            echo "(No new changes to commit)"
        fi
        # Push even if commit had nothing new — there may be unpushed commits
        AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
        if [ "$AHEAD" -eq 0 ]; then
            echo "✓ Nothing to push — already in sync with GitHub"
        elif git push origin main; then
            echo "✓ Pushed $AHEAD commit(s) to GitHub"
        else
            echo "✗ Push failed — commit is local only" >&2
            exit 1
        fi
        ;;
    status)
        git status
        ;;
    log)
        git log --oneline -15
        ;;
    *)
        echo "Usage: ./sync.sh [pull|push|status|log]"
        echo ""
        echo "  pull                     Pull latest changes from GitHub"
        echo "  push 'my message'        Stage all, commit with message, and push"
        echo "  status                   Show current git status"
        echo "  log                      Show last 15 commits"
        ;;
esac
