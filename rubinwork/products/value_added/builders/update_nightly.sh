#!/usr/bin/env bash
# Run the nightly value_added update (update_nightly.py), locally or as one Slurm job.
#
# Usage:
#   ./update_nightly.sh --mode local --dry-run
#   ./update_nightly.sh --mode batch
#   ./update_nightly.sh --mode batch --through 20261110 --redo 3
#
# Every argument other than --mode goes to update_nightly.py. The log is
# value_added/logs/update_<timestamp>.out in both modes.
#
# Batch tunables (env vars; defaults in parens):
#   SB_PARTITION (milano)  SB_MEM (16G)  SB_TIME (08:00:00)
#   SB_ACCOUNT (rubin:developers@milano)  SB_QOS (normal)
# SB_TIME covers the trim tail: one night of about 1,800 exposures has taken 53 min.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$repo/value_added"

mode=local; args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)   mode="$2"; shift 2;;
        --mode=*) mode="${1#*=}"; shift;;
        *)        args+=("$1"); shift;;
    esac
done

mkdir -p logs
log="$PWD/logs/update_$(date +%Y%m%d_%H%M%S).out"
cmd=(python -u -m rubinwork.products.value_added.builders.update_nightly "${args[@]}")

case "$mode" in
    local)
        echo "log: $log"
        "${cmd[@]}" > "$log" 2>&1
        ;;
    batch)
        command -v sbatch >/dev/null || {
            echo "error: sbatch not found; submit from an s3df node, not an RSP pod" >&2
            exit 2; }
        sbatch --partition="${SB_PARTITION:-milano}" \
               --account="${SB_ACCOUNT:-rubin:developers@milano}" \
               --qos="${SB_QOS:-normal}" --cpus-per-task=1 --mem="${SB_MEM:-16G}" \
               --time="${SB_TIME:-08:00:00}" --job-name=va_update_nightly \
               --output="$log" --wrap "cd '$PWD' && ${cmd[*]}"
        echo "log: $log"
        echo "follow: tail -f $log"
        ;;
    *) echo "error: --mode must be local or batch" >&2; exit 2;;
esac
