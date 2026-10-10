#!/usr/bin/env bash
# ----------------------------------------------------------------------
# relink_output_dirs.sh
#
# Rebuild the per-topic `output/` symlinks on the USDF RSP after they
# get clobbered (most commonly by a stray `git clean -fd` during a
# rebase recovery).  Each topic dir's `output/` points at the
# persistent storage area:
#
#   ~/notebooks/rubin-work/<topic>/output  ->  $TARGET/<topic>/output
#
# Idempotent: skips any topic where the persistent dir is missing, the
# local topic dir is missing, or `output` is already present.
#
# Usage:
#   ./relink_output_dirs.sh
#   TARGET=/some/other/persistent/root ./relink_output_dirs.sh
# ----------------------------------------------------------------------
set -u

TARGET="${TARGET:-/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work}"
REPO="${REPO:-$HOME/notebooks/rubin-work}"

TOPICS=(aos camera psf guider starcolor des survey wcs blocks)

if [[ ! -d "$REPO" ]]; then
    echo "error: REPO not found: $REPO" >&2
    exit 1
fi
cd "$REPO"

for d in "${TOPICS[@]}"; do
    if [[ -d "$d" && ! -e "$d/output" && -d "$TARGET/$d/output" ]]; then
        ( cd "$d" && ln -s "$TARGET/$d/output" )
        echo "  linked $d/output -> $TARGET/$d/output"
    fi
done

echo
echo "Current output symlinks:"
ls -la */output 2>/dev/null
