#!/bin/bash
# sync.sh — Quick pull/push helper for RSP terminals
cd "$(dirname "$0")"

case "$1" in
    pull)
        echo "Pulling latest from GitHub..."
        if git pull --rebase origin main; then
            echo "✓ Up to date"
        else
            echo "✗ Pull failed" >&2
            exit 1
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
        git commit -m "$MSG" || { echo "✗ Nothing to commit"; exit 1; }
        if git push origin main; then
            echo "✓ Pushed to GitHub"
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
