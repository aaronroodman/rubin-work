#!/usr/bin/env bash
# Build the value-added database in parallel shards, locally or as Slurm batch jobs.
#
# Both builders are per-night and independent, so a long backfill is split across shards
# that each write their OWN database file and are merged afterwards with
# merge_db_shards.py.  Sharding is necessary rather than merely faster: DuckDB's file lock
# is process-wide and excludes readers as well as writers, so several processes cannot
# write one database file.
#
# WHICH MODE EACH BUILDER NEEDS
#
#   --what telemetry   Engineering Facility Database (EFD) bound.  The EFD resolves from
#                      the Rubin Science Platform (RSP) and from s3df INTERACTIVE nodes
#                      (slacrd / sdfiana*), but NOT from milano/torino/roma batch compute
#                      nodes.  So this runs in --mode local only; batch would fetch
#                      nothing.  Local shards are ordinary background processes, so no
#                      Slurm submission is involved.
#
#   --what state       Consolidated Database (ConsDB) bound, through the public
#                      token-injected endpoint, which does resolve on a batch node.  It
#                      also needs the AOS environment (lsst.ts.ofc, and for the Measured
#                      Intrinsic Wavefront route lsst.ts.intrinsic.wavefront), which the
#                      job inherits from the submitting shell via --export=ALL.
#
# Usage:
#   ./run_build.sh --what telemetry --day-obs 20251103-20260418 --chunk 24
#   ./run_build.sh --what state --variant v50_34__miw__consdb_v1 \
#       --day-obs 20251102-20260913 --chunk 24 --img-type science,acq
#   ./run_build.sh --what state --variant v50_34__miw__consdb_v1 \
#       --day-obs 20251102-20260913 --chunk 24 --mode batch
#   ./run_build.sh --what telemetry --day-obs 20251103-20260418 --chunk 24 --dry-run
#
# Options:
#   --what telemetry|state   which builder (required)
#   --day-obs SPEC           a night, an inclusive RANGE, or a comma list (required)
#   --chunk N                nights per shard (default 24, about 20 min for telemetry)
#   --mode local|batch       default local; 'batch' is rejected for --what telemetry
#   --variant ID             registered variant id, for --what state
#   --img-type LIST          ConsDB img_types, comma separated, for --what state
#   --groups LIST            EFD/value-added groups, for --what telemetry (default all)
#   --resume                 pass --resume to the builder (skip what is already done)
#   --shard-dir DIR          where shard databases go (default output/value_added/shards)
#   --dry-run                print the shard plan and the commands, launch nothing
#
# Batch tunables (env vars; defaults in parens), matching aos/run_snake.sh:
#   SB_PARTITION (milano)  SB_CPUS (1)  SB_MEM (16G)  SB_TIME (01:00:00)
#   SB_ACCOUNT (rubin:developers@milano)  SB_QOS (normal)
#
# After every shard finishes, merge them into the main database:
#   python common/scripts/merge_db_shards.py --shards '<shard-dir>/<tag>_*.duckdb'
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
repo=$PWD

what=""; day_obs=""; chunk=24; mode=local; variant=""; img_type=""
groups="all"; resume=0; shard_dir="output/value_added/shards"; dry=0
while [ $# -gt 0 ]; do
    case "$1" in
        --what)       what="$2"; shift 2;;
        --what=*)     what="${1#*=}"; shift;;
        --day-obs)    day_obs="$2"; shift 2;;
        --day-obs=*)  day_obs="${1#*=}"; shift;;
        --chunk)      chunk="$2"; shift 2;;
        --chunk=*)    chunk="${1#*=}"; shift;;
        --mode)       mode="$2"; shift 2;;
        --mode=*)     mode="${1#*=}"; shift;;
        --variant)    variant="$2"; shift 2;;
        --variant=*)  variant="${1#*=}"; shift;;
        --img-type)   img_type="$2"; shift 2;;
        --img-type=*) img_type="${1#*=}"; shift;;
        --groups)     groups="$2"; shift 2;;
        --groups=*)   groups="${1#*=}"; shift;;
        --shard-dir)  shard_dir="$2"; shift 2;;
        --shard-dir=*) shard_dir="${1#*=}"; shift;;
        --resume)     resume=1; shift;;
        --dry-run)    dry=1; shift;;
        *) echo "error: unknown argument '$1'" >&2; exit 2;;
    esac
done

[ -n "$what" ]    || { echo "error: --what telemetry|state is required" >&2; exit 2; }
[ -n "$day_obs" ] || { echo "error: --day-obs is required" >&2; exit 2; }
case "$what" in
    telemetry) builder=common/scripts/build_efd_db.py; tag=telemetry;;
    state)     builder=common/scripts/build_optical_state.py; tag=state;;
    *) echo "error: --what must be 'telemetry' or 'state'" >&2; exit 2;;
esac
case "$mode" in
    local) ;;
    batch)
        if [ "$what" = telemetry ]; then
            echo "error: --what telemetry cannot run in batch — the EFD does not resolve" \
                 "from a Slurm compute node, so the fetch would return nothing." \
                 "Use --mode local on an s3df interactive node (slacrd) or the RSP." >&2
            exit 2
        fi
        command -v sbatch >/dev/null || {
            echo "error: sbatch not found — submit batch mode from an s3df interactive" \
                 "node (e.g. slacrd), not an RSP pod, which has no Slurm." >&2
            exit 2; }
        ;;
    *) echo "error: unknown --mode '$mode' (use 'local' or 'batch')" >&2; exit 2;;
esac
if [ "$what" = state ] && [ -z "$variant" ]; then
    echo "error: --what state needs --variant naming a registered variant id" >&2
    exit 2
fi

# ---- resolve the variant's registration flags -------------------------------
# Each shard writes its OWN database, whose state_variant table starts empty, so passing
# only --variant makes the builder exit with "variant is not registered".  Look the
# variant up in the main database and pass the defining flags instead; the builder derives
# the same variant_id from them, so every shard registers the identical variant and the
# merge sees one row in state_variant.
variant_flags=()
if [ "$what" = state ]; then
    mapfile -t vrow < <(python - "$variant" <<'PY'
import pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
from common import efd_db
con = efd_db.open_db('output/value_added/aos_efd.duckdb', readonly=True)
r = con.execute('SELECT scheme, intrinsic_route, opd_version, intrinsic_ref, '
                'ofc_config_version FROM state_variant WHERE variant_id = ?',
                [sys.argv[1]]).fetchall()
con.close()
if not r:
    sys.exit(f'variant {sys.argv[1]!r} is not registered in the main database')
for v in r[0]:
    print('' if v is None else v)
PY
    ) || { echo "error: could not resolve variant '$variant'" >&2; exit 2; }
    variant_flags=(--scheme "${vrow[0]}" --intrinsic "${vrow[1]}"
                   --opd-version "${vrow[2]}")
    [ -n "${vrow[3]}" ] && variant_flags+=(--intrinsic-ref "${vrow[3]}")
    [ -n "${vrow[4]}" ] && variant_flags+=(--ofc-version "${vrow[4]}")
    echo "variant $variant -> ${variant_flags[*]}"
fi

mkdir -p logs "$shard_dir"

# ---- expand --day-obs to the nights that actually have exposures ------------
# The builders' own parse_day_obs is reused, so a range means the same set of nights here
# as it does inside a shard, and an 18-month range does not iterate over empty nights.
echo "expanding --day-obs $day_obs (ConsDB query for nights with exposures) ..."
mapfile -t nights < <(python - "$day_obs" <<'PY'
import pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
sys.path.insert(0, str(pathlib.Path.cwd() / 'common' / 'scripts'))
from build_efd_db import parse_day_obs
from common.telemetry_clients import make_consdb_client
spec = sys.argv[1]
cdb = make_consdb_client('auto') if '-' in spec else None
for d in parse_day_obs(spec, cdb=cdb):
    print(d)
PY
)
n_nights=${#nights[@]}
[ "$n_nights" -gt 0 ] || { echo "error: --day-obs matched no nights" >&2; exit 2; }

ts=$(date +%Y%m%d_%H%M%S)
n_shards=$(( (n_nights + chunk - 1) / chunk ))
echo "$n_nights night(s) -> $n_shards shard(s) of up to $chunk night(s), mode $mode"
echo

pids=(); shards=()
for ((s = 0; s < n_shards; s++)); do
    slice=("${nights[@]:s*chunk:chunk}")
    list=$(IFS=,; echo "${slice[*]}")
    idx=$(printf '%02d' "$((s + 1))")
    shard_db="$shard_dir/${tag}_${ts}_${idx}.duckdb"
    log="logs/build_${tag}_${ts}_${idx}.log"
    shards+=("$shard_db")

    # -u is required, not cosmetic: with stdout redirected to a log file Python block-buffers
    # it, and the accumulated writes fail to flush at shutdown, killing the shard partway
    # through with exit code 120, no traceback, and a log holding only the stderr warnings.
    # Unbuffered output writes through immediately and the same 24-night slice runs clean.
    cmd=(python -u "$builder" --day-obs "$list" --db "$shard_db")
    if [ "$what" = telemetry ]; then
        cmd+=(--groups "$groups")
    else
        cmd+=("${variant_flags[@]}")
        [ -n "$img_type" ] && cmd+=(--img-type "$img_type")
    fi
    [ "$resume" = 1 ] && cmd+=(--resume)

    echo "shard $idx/$n_shards: ${#slice[@]} night(s) ${slice[0]}..${slice[-1]}"
    echo "  db:  $shard_db"
    echo "  log: $log"
    if [ "$dry" = 1 ]; then
        echo "  cmd: ${cmd[*]}"
        continue
    fi
    case "$mode" in
        local)
            nohup "${cmd[@]}" > "$log" 2>&1 &
            pids+=("$!")
            echo "  pid: ${pids[-1]}"
            ;;
        batch)
            jlog="logs/batch_${tag}_${ts}_${idx}.out"
            sbatch --partition="${SB_PARTITION:-milano}" \
                   --account="${SB_ACCOUNT:-rubin:developers@milano}" \
                   --qos="${SB_QOS:-normal}" \
                   --cpus-per-task="${SB_CPUS:-1}" --mem="${SB_MEM:-16G}" \
                   --time="${SB_TIME:-01:00:00}" \
                   --job-name="build_${tag}_${idx}" --output="$jlog" \
                   --wrap "cd '$repo' && ${cmd[*]}"
            echo "  job log: $jlog"
            ;;
    esac
done

if [ "$dry" = 1 ]; then
    echo
    echo "dry run — nothing launched."
    exit 0
fi

echo
merge_glob="$shard_dir/${tag}_${ts}_*.duckdb"
case "$mode" in
    local)
        echo "$n_shards shard(s) launched in the background on $(hostname)."
        echo "  follow all:   tail -f logs/build_${tag}_${ts}_*.log"
        echo "  check alive:  ps -o pid,etime,cmd -p ${pids[*]}"
        ;;
    batch)
        echo "$n_shards job(s) submitted."
        echo "  follow all:   tail -f logs/batch_${tag}_${ts}_*.out"
        echo "  queue state:  squeue -u \$USER"
        ;;
esac
echo
echo "when every shard has finished, merge into the main database:"
echo "  cd $repo && python common/scripts/merge_db_shards.py --shards '$merge_glob'"
