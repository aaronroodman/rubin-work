#!/bin/bash
# sync.sh — Quick pull/push helper for RSP terminals
cd "$(dirname "$0")"

case "$1" in
    pull)
        echo "Pulling latest from GitHub..."

        # Step 1: Fetch remote state
        git fetch origin main

        # Step 2: Find files changed locally (staged + unstaged)
        LOCAL_CHANGED=$(git diff --name-only 2>/dev/null; git diff --name-only --cached 2>/dev/null)
        LOCAL_CHANGED=$(echo "$LOCAL_CHANGED" | sort -u | grep -v '^$')

        # Step 3: Find files changed on remote
        REMOTE_CHANGED=$(git diff --name-only HEAD origin/main 2>/dev/null)

        # Step 4: Find conflicts (files changed in both places)
        CONFLICTS=""
        if [ -n "$LOCAL_CHANGED" ] && [ -n "$REMOTE_CHANGED" ]; then
            CONFLICTS=$(comm -12 <(echo "$LOCAL_CHANGED") <(echo "$REMOTE_CHANGED"))
        fi

        # Step 5: If conflicts, prompt per-file
        OVERWRITE_FILES=""
        KEEP_FILES=""
        if [ -n "$CONFLICTS" ]; then
            echo ""
            echo "⚠ These files have been changed both locally and on GitHub:"
            echo "$CONFLICTS" | while read -r f; do echo "  $f"; done
            echo ""
            for f in $CONFLICTS; do
                while true; do
                    read -p "  $f — (o)verwrite with remote, (k)eep local, (a)bort? " ans < /dev/tty
                    case "$ans" in
                        o|O)
                            OVERWRITE_FILES="$OVERWRITE_FILES $f"
                            break
                            ;;
                        k|K)
                            KEEP_FILES="$KEEP_FILES $f"
                            break
                            ;;
                        a|A)
                            echo "Aborted — no changes made."
                            exit 0
                            ;;
                        *)
                            echo "    Please enter o, k, or a"
                            ;;
                    esac
                done
            done
            echo ""
        fi

        # Step 6: Overwrite chosen files (discard local changes)
        for f in $OVERWRITE_FILES; do
            echo "  Overwriting $f with remote version..."
            git checkout origin/main -- "$f"
        done

        # Step 7: Stash remaining local changes, pull, restore
        STASHED=false
        if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
            echo "Stashing remaining local changes..."
            git stash push -m "sync.sh auto-stash"
            STASHED=true
        fi

        if git pull --rebase origin main; then
            echo "✓ Up to date with GitHub"
        else
            echo "✗ Rebase conflict — these files need manual resolution:" >&2
            git diff --name-only --diff-filter=U 2>/dev/null
            echo ""
            echo "To abort and return to your previous state:"
            echo "  git rebase --abort"
            if $STASHED; then
                echo "  git stash pop"
            fi
            exit 1
        fi

        if $STASHED; then
            echo "Restoring local changes..."
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
        # Collect changed/untracked files
        FILES=$(git status --porcelain | awk '{print $NF}')
        if [ -n "$FILES" ]; then
            echo "Files to stage:"
            for f in $FILES; do
                read -p "  $f [Y/n] " ans < /dev/tty
                case "$ans" in
                    n|N) echo "    skipped" ;;
                    *)   git add "$f" ;;
                esac
            done
            echo ""
        fi
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
