#!/bin/bash
# setup_env.sh — Run once after cloning on a new machine or RSP instance
set -e
cd "$(dirname "$0")"

echo "=== Setting up rubin-work environment ==="

echo ""
echo "--- Installing nbstripout ---"
pip install --user nbstripout 2>/dev/null || pip install nbstripout
nbstripout --install
echo "✓ nbstripout installed and configured as git filter"

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
        mkdir -p "$topic/notebooks"
        mkdir -p "$topic/scripts"
    fi
done
echo "✓ Directory structure verified"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy the notebook template:  cp common/notebook_template.ipynb aos/notebooks/my_notebook.ipynb"
echo "  2. Start working:               jupyter lab"
echo "  3. When done:                    ./sync.sh push 'my changes'"
