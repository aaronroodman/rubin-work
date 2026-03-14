#!/bin/bash
# list_notebooks.sh — Inventory all Jupyter notebooks on this RSP instance
# Run on Summit or USDF to see what needs organizing.
#
# Usage:
#   ./list_notebooks.sh              # List all notebooks under ~/notebooks
#   ./list_notebooks.sh /path/to/dir # List notebooks under a specific directory

set -e

SEARCH_DIR="${1:-$HOME/notebooks}"
HOSTNAME=$(hostname)
DATE=$(date +%Y-%m-%d)

echo "============================================================"
echo "Notebook inventory: $HOSTNAME"
echo "Date: $DATE"
echo "Search directory: $SEARCH_DIR"
echo "============================================================"
echo ""

# Count total
TOTAL=$(find "$SEARCH_DIR" -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" 2>/dev/null | wc -l | tr -d ' ')
echo "Total notebooks found: $TOTAL"
echo ""

# Already organized (inside rubin-work)
echo "--- In rubin-work (organized) ---"
find "$SEARCH_DIR" -path "*/rubin-work/*" -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" 2>/dev/null | sort | while read -r f; do
    SIZE=$(du -h "$f" | cut -f1)
    MOD=$(stat -f "%Sm" -t "%Y-%m-%d" "$f" 2>/dev/null || stat -c "%y" "$f" 2>/dev/null | cut -d' ' -f1)
    REL=$(echo "$f" | sed "s|$SEARCH_DIR/||")
    printf "  %-60s %6s  %s\n" "$REL" "$SIZE" "$MOD"
done
echo ""

# Not yet organized (outside rubin-work)
echo "--- Outside rubin-work (needs triage) ---"
find "$SEARCH_DIR" -name "*.ipynb" -not -path "*/rubin-work/*" -not -path "*/.ipynb_checkpoints/*" 2>/dev/null | sort | while read -r f; do
    SIZE=$(du -h "$f" | cut -f1)
    MOD=$(stat -f "%Sm" -t "%Y-%m-%d" "$f" 2>/dev/null || stat -c "%y" "$f" 2>/dev/null | cut -d' ' -f1)
    REL=$(echo "$f" | sed "s|$SEARCH_DIR/||")
    printf "  %-60s %6s  %s\n" "$REL" "$SIZE" "$MOD"
done
echo ""

# Summary of other file types that might need organizing
echo "--- Other data files outside rubin-work ---"
OTHER_COUNT=$(find "$SEARCH_DIR" \( -name "*.fits" -o -name "*.parquet" -o -name "*.hdf5" -o -name "*.h5" -o -name "*.csv" -o -name "*.png" -o -name "*.pdf" \) -not -path "*/rubin-work/*" 2>/dev/null | wc -l | tr -d ' ')
echo "  FITS/Parquet/HDF5/CSV/PNG/PDF files: $OTHER_COUNT"
if [ "$OTHER_COUNT" -gt 0 ]; then
    echo ""
    find "$SEARCH_DIR" \( -name "*.fits" -o -name "*.parquet" -o -name "*.hdf5" -o -name "*.h5" -o -name "*.csv" -o -name "*.png" -o -name "*.pdf" \) -not -path "*/rubin-work/*" 2>/dev/null | sort | while read -r f; do
        SIZE=$(du -h "$f" | cut -f1)
        REL=$(echo "$f" | sed "s|$SEARCH_DIR/||")
        printf "  %-60s %6s\n" "$REL" "$SIZE"
    done
fi

echo ""
echo "============================================================"
echo "Suggested next steps:"
echo "  1. Move relevant notebooks into rubin-work/<topic>/notebooks/"
echo "  2. Move large data files into ~/notebooks/rubin-data/<topic>/"
echo "  3. Delete obsolete notebooks and checkpoints"
echo "  4. Run: ./sync.sh push 'organized notebooks from $HOSTNAME'"
echo "============================================================"
