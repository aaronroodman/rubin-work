#!/usr/bin/env bash
# ----------------------------------------------------------------------
# update_lsst_packages.sh
#
# Fast-forward the develop branch in every manually-cloned LSST package
# under $PACKAGES_DIR (default: ~/u/LSST/packages).  Designed for the
# USDF RSP where Aaron maintains local git clones of the ts_* and
# friends used by various notebooks.
#
# Behaviour per package:
#   1.  Skip if there are uncommitted changes.
#   2.  Switch to `develop` if not already there (only if the working
#       tree is clean).
#   3.  `git fetch origin`, then `git pull --ff-only origin develop`.
#   4.  Report how many commits were applied (or "already up to date").
#
# Usage:
#   ./update_lsst_packages.sh                  # default packages dir
#   PACKAGES_DIR=/some/other/path ./update_lsst_packages.sh
# ----------------------------------------------------------------------
set -u

PACKAGES_DIR="${PACKAGES_DIR:-$HOME/u/LSST/packages}"

PACKAGES=(
    summit_utils
    ts_utils
    ts_salobj
    ts_idl
    ts_observatory_control
    ts_aos_analysis
    ts_ofc
    ts_wep
    ts_xml
    ts_config_mttcs
    ts_m1m3_utils
    batoid_rubin
)

okay=()
skipped=()
failed=()

# Plain ASCII bullets so this script is happy in any locale.
hr() { printf '%s\n' '=================================================='; }

for pkg in "${PACKAGES[@]}"; do
    path="$PACKAGES_DIR/$pkg"
    echo
    hr
    printf ' %s\n' "$pkg"
    printf '   path: %s\n' "$path"
    hr

    if [[ ! -d "$path/.git" ]]; then
        echo "  (no git checkout - skipping)"
        skipped+=("$pkg (no checkout)")
        continue
    fi

    pushd "$path" > /dev/null || { failed+=("$pkg (cd failed)"); continue; }

    # Refuse to touch a dirty tree.
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "  WARNING: uncommitted changes - skipping"
        skipped+=("$pkg (uncommitted changes)")
        popd > /dev/null
        continue
    fi

    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '???')
    if [[ "$current_branch" != "develop" ]]; then
        echo "  on '$current_branch' - switching to develop"
        if ! git checkout develop 2>&1 | sed 's/^/    /'; then
            echo "  could not switch to develop"
            failed+=("$pkg (checkout failed)")
            popd > /dev/null
            continue
        fi
    fi

    echo "  fetching..."
    if ! git fetch origin 2>&1 | sed 's/^/    /'; then
        failed+=("$pkg (fetch failed)")
        popd > /dev/null
        continue
    fi

    before=$(git rev-parse HEAD)
    if git pull --ff-only origin develop 2>&1 | sed 's/^/    /'; then
        after=$(git rev-parse HEAD)
        if [[ "$before" == "$after" ]]; then
            echo "  already up to date"
        else
            n_commits=$(git rev-list --count "$before..$after")
            echo "  pulled $n_commits commit(s)"
        fi
        okay+=("$pkg")
    else
        echo "  pull failed (non-FF? diverged?)"
        failed+=("$pkg (pull failed)")
    fi

    popd > /dev/null
done

echo
hr
echo " Summary"
hr
printf '  OK      : %d\n' "${#okay[@]}";
[[ ${#okay[@]}    -gt 0 ]] && printf '            %s\n' "${okay[@]}"
printf '  Skipped : %d\n' "${#skipped[@]}";
[[ ${#skipped[@]} -gt 0 ]] && printf '            %s\n' "${skipped[@]}"
printf '  Failed  : %d\n' "${#failed[@]}";
[[ ${#failed[@]}  -gt 0 ]] && printf '            %s\n' "${failed[@]}"

# Exit non-zero if anything failed (skipped is fine).
[[ ${#failed[@]} -eq 0 ]]
