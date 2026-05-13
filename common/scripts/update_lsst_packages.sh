#!/usr/bin/env bash
# ----------------------------------------------------------------------
# update_lsst_packages.sh
#
# Fast-forward the *default* branch in every manually-cloned LSST
# package under $PACKAGES_DIR (default: ~/u/LSST/packages).  Designed
# for the USDF RSP where Aaron maintains local git clones of the ts_*
# and friends used by various notebooks.
#
# The default branch is auto-detected per-repo from origin/HEAD —
# e.g. ts_ofc uses 'develop' while summit_utils uses 'main'.
#
# Behaviour per package:
#   1.  Skip if there are uncommitted changes.
#   2.  `git fetch origin --prune`.
#   3.  Resolve origin/HEAD -> default branch.
#   4.  Switch to it if not already there.
#   5.  `git pull --ff-only origin <default>`.
#   6.  Report how many commits were applied (or "already up to date").
#
# Usage:
#   ./update_lsst_packages.sh                  # default packages dir
#   PACKAGES_DIR=/some/other/path ./update_lsst_packages.sh
# ----------------------------------------------------------------------
set -u
set -o pipefail        # so `git pull ... | sed` propagates git's exit code

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

hr() { printf '%s\n' '=================================================='; }

# Return the upstream default branch (e.g. 'main' or 'develop') as
# tracked by origin/HEAD.  Auto-repairs the ref if missing.  Prints
# the branch name on stdout; empty string + nonzero exit on failure.
resolve_default_branch() {
    local b
    b=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null) || b=''
    if [[ -z "$b" ]]; then
        git remote set-head origin --auto >/dev/null 2>&1 || true
        b=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null) || b=''
    fi
    if [[ -z "$b" ]]; then
        return 1
    fi
    printf '%s' "${b#origin/}"
}

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

    echo "  fetching..."
    if ! git fetch origin --prune 2>&1 | sed 's/^/    /'; then
        failed+=("$pkg (fetch failed)")
        popd > /dev/null
        continue
    fi

    if ! default=$(resolve_default_branch); then
        echo "  could not resolve origin/HEAD - skipping"
        failed+=("$pkg (no default branch)")
        popd > /dev/null
        continue
    fi
    echo "  default branch: $default"

    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '???')
    if [[ "$current_branch" != "$default" ]]; then
        echo "  on '$current_branch' - switching to '$default'"
        if ! git checkout "$default" 2>&1 | sed 's/^/    /'; then
            echo "  could not switch to $default"
            failed+=("$pkg (checkout failed)")
            popd > /dev/null
            continue
        fi
    fi

    before=$(git rev-parse HEAD)
    if git pull --ff-only origin "$default" 2>&1 | sed 's/^/    /'; then
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
