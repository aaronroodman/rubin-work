#!/usr/bin/env bash
# ----------------------------------------------------------------------
# archive_output.sh — snapshot every <topic>/output/ to a dated archive.
#
# COPIES (does not move) all `<topic>/output/` directories in the repo to
# $ARCHIVE_ROOT/<label>/<topic>/output/, records the git commit/tag that
# produced them, and hardlinks files unchanged since the previous snapshot
# (via rsync --link-dest) so repeat snapshots cost only the changed files.
#
# Run AFTER your RSP->laptop output sync, so the laptop output/ is current.
#
# Usage:
#   common/scripts/archive_output.sh                 # label = today's date
#   common/scripts/archive_output.sh pre-reorg        # explicit label
#   ARCHIVE_ROOT=/Volumes/ext/rubin-archive common/scripts/archive_output.sh
# ----------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/rubin-archive}"
LABEL="${1:-$(date +%Y-%m-%d)}"
DEST="$ARCHIVE_ROOT/$LABEL"

echo "Repo:    $REPO"
echo "Archive: $DEST"
mkdir -p "$DEST"

# Most-recent prior snapshot -> hardlink unchanged files (cheap incremental).
prev="$(ls -1dt "$ARCHIVE_ROOT"/*/ 2>/dev/null | grep -v "/$LABEL/$" | head -n1 || true)"
if [[ -n "$prev" ]]; then
    echo "Incremental: hardlinking unchanged files against $prev"
fi

# Record the code state this snapshot corresponds to.
{
    echo "label:   $LABEL"
    echo "date:    $(date)"
    echo "commit:  $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "describe: $(git -C "$REPO" describe --tags --always --dirty 2>/dev/null || echo '?')"
} > "$DEST/GIT_STATE.txt"

shopt -s nullglob
for outdir in "$REPO"/*/output/; do
    topic="$(basename "$(dirname "$outdir")")"
    dst="$DEST/$topic/output"
    mkdir -p "$dst"
    link=()
    [[ -n "$prev" && -d "$prev/$topic/output" ]] && link=(--link-dest "$prev/$topic/output")
    echo "  syncing $topic/output ..."
    rsync -a "${link[@]}" "$outdir" "$dst/"
done

echo
echo "Per-topic sizes:"
du -sh "$DEST"/*/output 2>/dev/null || true
echo "Snapshot total (apparent): $(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo "Archive root on disk:      $(du -sh "$ARCHIVE_ROOT" 2>/dev/null | cut -f1)  (hardlinks shared across snapshots)"
echo "Done -> $DEST"
