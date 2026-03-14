#!/bin/bash
# sync.sh — Quick pull/push helper for RSP terminals
cd "$(dirname "$0")"

case "$1" in
    pull)
        echo "Pulling latest from GitHub..."
        git pull --rebase origin main
        echo "✓ Up to date"
        ;;
    push)
        MSG="${2:-Update from RSP $(hostname) $(date +%Y-%m-%d)}"
        echo "Staging all changes..."
        git add -A
        echo ""
        git status --short
        echo ""
        echo "Committing: $MSG"
        git commit -m "$MSG"
        git push origin main
        echo "✓ Pushed to GitHub"
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
