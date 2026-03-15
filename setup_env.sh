#!/bin/bash
# setup_env.sh — Run once after cloning on a new machine or RSP instance
set -e
cd "$(dirname "$0")"

echo "=== Setting up rubin-work environment ==="

echo ""
echo "--- Configuring git credential cache ---"
git config credential.helper 'cache --timeout=86400'
echo "✓ Git credentials will be cached for 24 hours"

if [ -f requirements.txt ]; then
    echo ""
    echo "--- Installing Python dependencies ---"
    pip install --user -r requirements.txt 2>/dev/null || pip install -r requirements.txt
    echo "✓ Python dependencies installed"
fi

echo ""
echo "--- Ensuring directory structure ---"
for topic in aos camera psf guider starcolor des survey wcs blocks common scratch; do
    if [ "$topic" = "common" ] || [ "$topic" = "scratch" ]; then
        mkdir -p "$topic"
    else
        mkdir -p "$topic/code"
        mkdir -p "$topic/output"
    fi
done
echo "✓ Directory structure verified"

echo ""
echo "--- Adding shell functions ---"
# Use ~/notebooks/... so it works regardless of whether $HOME is
# /home/r/roodman (JupyterHub) or /sdf/home/r/roodman (SSH)
FUNC_BLOCK='# rubin-work shortcuts
function gitpull { ~/notebooks/rubin-work/sync.sh pull; }
function gitpush { ~/notebooks/rubin-work/sync.sh push "$@"; }'

for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rcfile" ] || [ "$(basename "$rcfile")" = ".$(basename "$SHELL")rc" ]; then
        if grep -q "rubin-work shortcuts" "$rcfile" 2>/dev/null; then
            # Remove old alias-based block and replace with functions
            sed -i.bak '/# rubin-work shortcuts/,+2d' "$rcfile"
            rm -f "${rcfile}.bak"
        fi
        echo "" >> "$rcfile"
        echo "$FUNC_BLOCK" >> "$rcfile"
        echo "✓ Added gitpull/gitpush functions to $rcfile"
    fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Source your shell config:     source ~/.bashrc  (or restart terminal)"
echo "  2. Copy the notebook template:   cp common/notebook_template.ipynb aos/my_notebook.ipynb"
echo "  3. Start working:                jupyter lab"
echo "  4. When done:                    gitpush 'my changes'"
echo ""
echo "Shell shortcuts (available after sourcing):"
echo "  gitpull                          Pull latest changes from GitHub"
echo "  gitpush 'my message'             Stage all, commit, and push"
