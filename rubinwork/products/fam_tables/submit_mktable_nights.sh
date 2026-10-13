#!/bin/bash
# Submit the phase 3c `danish_1_2` mktable rebuild as one Slurm job per night.
#
# Phase 3c of notes/status/organization_plan.md rebuilds `fam_tables` `danish_1_2` into
# the products tree.  `mktable` is the expensive stage -- 258 s fixed per run plus
# 39.1 s/visit of donut reading, about 39 h of serial CPU over 3,523 Butler visits -- and
# since phase 3b it needs no Engineering Facility Database (EFD) connection, so it can
# run on a batch node.  `--no-thermal` is what removed the EFD: the `thermal` and
# `gradients` columns come from `run_attach_telemetry`'s sidecar instead.
#
# Why per night rather than per chunk: the largest chunk alone is an 8.9 h wall-clock
# floor.  One job per night turns that into 137 independent jobs, so the whole rebuild
# finishes in the time of its longest night.
#
# What this does NOT do: `fit`, `combine_donuts`, `combine_fits`, `combine_visits`,
# `attach_telemetry` and `manifest` are all run afterwards, interactively, from a node
# where the EFD resolves -- `attach_telemetry` still needs it for the `gradients`,
# `camera`, `hexlut` and `trim` groups.  No manifest is written until
# `attach_telemetry` has run, so an interrupted rebuild stays invisible to the catalog.
#
# Usage:
#   ./submit_mktable_nights.sh --dry-run            # print every sbatch line, submit none
#   ./submit_mktable_nights.sh                      # submit all nights
#   ./submit_mktable_nights.sh --night 20251126     # one night, repeatable
#   ./submit_mktable_nights.sh --nights 20251126,20251127   # a comma-separated list
#   ./submit_mktable_nights.sh --nights-file nights.txt     # one night per line, # comments ok
#   ./submit_mktable_nights.sh --chunk 20251116_20251130    # every night of one chunk
#
# Memory per job, 176G by default (see MEM below).  To trade margin for queue time:
#   MEM=96G ./submit_mktable_nights.sh --nights-file aos/logs/mktable_todo_nights_20261013.txt
#
# Partition and account default to milano; set both to use roma instead:
#   PARTITION=roma ACCOUNT='rubin:developers@roma' ./submit_mktable_nights.sh
#
# Submit only from an s3df node (slacrd / sdfiana); there is no Slurm on an RSP pod.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VARIANT=danish_1_2
# NOT named BUILD: the conda environment exports BUILD=x86_64-conda-linux-gnu, which a
# ${BUILD:-default} would silently pick up as the build name.
FAM_BUILD="${FAM_BUILD:-$(date -u +%Y%m%d)}"
LOGDIR="$REPO/aos/logs"
# milano by default: its `rubin:developers@milano` association has no GrpTRES cap, while
# `rubin:developers@roma` caps the whole account at 960 CPUs, which other group members
# saturate.  Override both together for roma:
#   PARTITION=roma ACCOUNT='rubin:developers@roma' ./submit_mktable_nights.sh
PARTITION="${PARTITION:-milano}"
ACCOUNT="${ACCOUNT:-rubin:developers@milano}"
WORKERS="${WORKERS:-8}"
# Whole-job memory request, overridable: MEM=96G ./submit_mktable_nights.sh ...
# 176G is 22 GB per donut-reading worker at the default 8 workers, and 2.1x the largest
# measured peak.  Measured 2026-10-13 on danish_1_2, all 70 nights of the 96G set at 8
# workers: MaxRSS 63.48 to 83.79 GB over 21,712 to 379,458 donuts per night, i.e. 9.2 to
# 10.8 GB per worker, so 32G killed 87 of 137 nights and 64G killed 70 of 87 while 96G
# killed none.  96G would be enough, but it clears the largest peak by only 14%, and the
# whole 32G/64G episode cost two batch rounds; the 2x margin here is deliberately bought
# with queue time rather than with another OOM round.  A variant with larger stamps, such
# as the danish_1_3 blitz, needs at least this.
# Per-night rows: aos/logs/mktable_memory_20261013.csv.
MEM="${MEM:-176G}"
# The longest single night is well under this; the margin covers a cold Butler.
TIME=08:00:00

DRY_RUN=0
ONLY_NIGHTS=()
ONLY_CHUNK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --night)   ONLY_NIGHTS+=("$2"); shift 2 ;;
        --nights)  IFS=',' read -r -a _list <<< "$2"
                   ONLY_NIGHTS+=("${_list[@]}"); shift 2 ;;
        --nights-file)
                   while read -r _line; do
                       _line="${_line%%#*}"; _line="${_line//[[:space:]]/}"
                       [[ -n "$_line" ]] && ONLY_NIGHTS+=("$_line")
                   done < "$2"
                   shift 2 ;;
        --chunk)   ONLY_CHUNK="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ $DRY_RUN -eq 0 ]] && ! command -v sbatch >/dev/null; then
    echo "ERROR: no sbatch on this host -- submit from slacrd or sdfiana, not an RSP pod" >&2
    exit 1
fi

# The Snakefile builds every runner path from $TS_INTRINSIC_WAVEFRONT_DIR, and the
# interactive shells set it to the /sdf/home/r/roodman/u/... form, which the root
# CLAUDE.md flags as RSP-only.  It is a symlink to /sdf/group/rubin/u/roodman, so resolve
# it here and pass the resolved path to the job rather than relying on the symlink being
# mounted on a compute node.
if [[ -z "${TS_INTRINSIC_WAVEFRONT_DIR:-}" ]]; then
    echo "ERROR: TS_INTRINSIC_WAVEFRONT_DIR is not set" >&2
    exit 1
fi
TS_DIR="$(readlink -f "$TS_INTRINSIC_WAVEFRONT_DIR")"
if [[ ! -x "$TS_DIR/bin/run_mktable.py" ]]; then
    echo "ERROR: $TS_DIR/bin/run_mktable.py missing -- run scons in the package" >&2
    exit 1
fi

mkdir -p "$LOGDIR"

# The night list comes from the variant's own chunk ranges, queried against the
# Consolidated Database (ConsDB) rather than hardcoded, so a chunk gaining a night does
# not silently drop it here.  Each line is "<night> <chunk_dmin> <chunk_dmax>", because
# mktable takes its collection override from the CHUNK the night belongs to.
# The build directory comes from the product reader, not from a path built here, so the
# data root stays the one place that knows it (RUBINWORK_DATA).
build_dir() {
    python -c "from rubinwork.products import fam_tables; \
print(fam_tables.build_dir('$VARIANT', '$FAM_BUILD'))"
}


night_plan() {
    python - "$REPO" <<'PYEOF'
import os, sys, yaml
from pathlib import Path
os.environ.setdefault('no_proxy', '')
if '.consdb' not in os.environ['no_proxy']:
    os.environ['no_proxy'] += ',.consdb'
from lsst.summit.utils import ConsDbClient

repo = Path(sys.argv[1])
url = 'https://usdf-rsp.slac.stanford.edu/consdb'
token = Path.home() / '.lsst' / 'consdb_token'
if token.exists():
    url = url.replace('://', f'://user:{token.read_text().strip()}@', 1)
client = ConsDbClient(url)

doc = yaml.safe_load(open(repo / 'rubinwork/products/fam_tables/variants.yaml'))

# One line per night, never per (night, chunk) pair. The chunk ranges of a composite
# variant overlap -- day_obs 20260713 sits in both the broad [20260514, 20260731]
# /repo/main chunk and its own single-night /repo/embargo chunk -- and both would write
# into the same chunks/<night>_<night>/ directory, so two jobs would race on one output
# and the /repo/main one would find no data. The narrowest matching range wins: it is the
# chunk written for that night, and it carries the right repository and phrase.
best = {}
for chunk in doc['variants']['danish_1_2']['chunks']:
    lo, hi = chunk['day_obs']
    rows = client.query(
        "SELECT day_obs FROM cdb_lsstcam.visit1 "
        f"WHERE day_obs >= {lo} AND day_obs <= {hi} AND img_type = 'cwfs' "
        "GROUP BY day_obs ORDER BY day_obs").to_pandas()
    for night in rows.day_obs:
        night = int(night)
        span = hi - lo
        if night not in best or span < best[night][0]:
            best[night] = (span, lo, hi)
for night in sorted(best):
    _, lo, hi = best[night]
    print(f'{night} {lo} {hi}')
PYEOF
}

# Snakemake's own budget tracks the Slurm request, so raising MEM raises both.  Without
# this the mem_mb ceiling stays at the Snakefile's estimate and throttles the workers
# below what the job was granted.
case "$MEM" in
    *G|*g) MEM_MB=$(( ${MEM%[Gg]} * 1000 )) ;;
    *M|*m) MEM_MB=${MEM%[Mm]} ;;
    *)     echo "ERROR: cannot parse MEM=$MEM -- use e.g. 64G" >&2; exit 2 ;;
esac

FAM_BUILD_DIR="$(build_dir)"

echo "repo      $REPO"
echo "variant   $VARIANT   build $FAM_BUILD"
echo "build dir $FAM_BUILD_DIR"
echo "resources partition=$PARTITION account=$ACCOUNT mem=$MEM time=$TIME workers=$WORKERS"
echo

n_submitted=0
MATCHED=()
while read -r night dmin dmax; do
    [[ -z "${night:-}" ]] && continue
    if [[ -n "$ONLY_CHUNK" && "${dmin}_${dmax}" != "$ONLY_CHUNK" ]]; then continue; fi
    if [[ ${#ONLY_NIGHTS[@]} -gt 0 ]]; then
        found=0
        for want in "${ONLY_NIGHTS[@]}"; do [[ "$want" == "$night" ]] && found=1; done
        [[ $found -eq 0 ]] && continue
        MATCHED+=("$night")
    fi

    # `nights=` makes this night its own chunk, inheriting its parent chunk's collection
    # override, so the per-night outputs never collide and a failed night is resubmitted
    # on its own.  mktable is driven through the product Snakefile rather than called
    # directly precisely so the override is applied for us.  The target is named
    # explicitly and placed last behind `--`, because `--config` is greedy.
    target="$FAM_BUILD_DIR/_work/chunks/${night}_${night}/donuts.parquet"
    cmd=(sbatch
         --job-name="mkt_${night}"
         --partition="$PARTITION"
         --account="$ACCOUNT"
         --qos=normal
         --cpus-per-task="$WORKERS"
         --mem="$MEM"
         --time="$TIME"
         --output="$LOGDIR/mktable_${night}_%j.log"
         --wrap "export TS_INTRINSIC_WAVEFRONT_DIR=$TS_DIR && cd $REPO && \
snakemake -s rubinwork/products/fam_tables/Snakefile \
--cores $WORKERS --resources mem_mb=$MEM_MB \
--config variant=$VARIANT build=$FAM_BUILD mktable_workers=$WORKERS \
mktable_no_thermal=1 nights=$night \
-- $target")

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
    n_submitted=$((n_submitted + 1))
done < <(night_plan)

# A night named on the command line but absent from the plan is a typo or a night with no
# ConsDB cwfs visits; either way it would otherwise vanish without a word.
if [[ ${#ONLY_NIGHTS[@]} -gt 0 ]]; then
    for want in "${ONLY_NIGHTS[@]}"; do
        found=0
        for got in ${MATCHED[@]+"${MATCHED[@]}"}; do [[ "$got" == "$want" ]] && found=1; done
        [[ $found -eq 0 ]] && echo "WARNING: night $want is not in the plan, not submitted" >&2
    done
fi

echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN: $n_submitted jobs would be submitted, none were."
else
    echo "submitted $n_submitted jobs (dimensionless count)"
    echo "monitor:  squeue -u roodman"
    echo "          tail -f \"\$(ls -t $LOGDIR/mktable_*.log | head -1)\""
fi
