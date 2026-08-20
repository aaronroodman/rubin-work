#!/usr/bin/env bash
# Launch the guider moment pipeline, locally or as a Slurm batch job.
#
# Modes (--mode, default local):
#   local  — run snakemake on THIS node, detached via nohup (survives a dropped
#            SSH/network connection).  Tuned for the RSP terminal (~4 cores).
#            Use from the USDF JupyterLab RSP (which cannot submit to Slurm).
#   batch  — submit ONE sbatch job PER night (each runs snakemake with many
#            parallel slots; a roma/milano node has far more than 4 cores).  So
#            --day-obs A,B,C submits 3 independent jobs -> parallel across the
#            cluster, per-night logs, and easy re-submit of a single failed
#            night.  Submit from an s3df interactive node (e.g. the `slacrd` ssh
#            alias) where sbatch + the LSST stack are available.  (With no
#            --day-obs, submits one job for the default `rule all`.)
#
# Scope:
#   (no --day-obs)      build `rule all` (the static datasets in the config)
#   --day-obs LIST      process whole nights: LIST is one dayObs or a
#                       comma-separated list (YYYYMMDD[,YYYYMMDD...]).  Each
#                       night's guider exposures are discovered from the Butler
#                       and built into output/night_<dayObs>/ (partitioned
#                       dataset + validation plot).
#   --limit N           smoke test: keep only the first N exposures per night
#                       (0 = all).  Run limited first, then re-run without
#                       --limit to build (and overwrite with) the full night(s).
#
# Usage:
#   ./run_snake.sh --day-obs 20260709                        # local, one night
#   ./run_snake.sh --day-obs 20260709 --limit 5 -n           # dry-run, 5-exp test
#   ./run_snake.sh --day-obs 20260709,20260710 --mode batch  # batch, two nights
#   ./run_snake.sh                                            # local, static datasets
#
# Batch tunables (env vars; defaults in parens):
#   SB_PARTITION (roma)  SB_CPUS (32)  SB_MEM (96G)  SB_TIME (08:00:00)
#   SB_RESMEM (90000)    SB_ACCOUNT (rubin:developers@roma)  SB_QOS (normal)
#
# Remember to `git pull` first to pick up code changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

mode=local
dayObs=""
limit=""
passthru=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)     mode="$2"; shift 2;;
        --mode=*)   mode="${1#*=}"; shift;;
        --day-obs)  dayObs="$2"; shift 2;;
        --day-obs=*) dayObs="${1#*=}"; shift;;
        --limit)    limit="$2"; shift 2;;
        --limit=*)  limit="${1#*=}"; shift;;
        *)          passthru+=("$1"); shift;;
    esac
done

# Parse the comma-separated nights into an array.
days=()
if [ -n "$dayObs" ]; then
    IFS=',' read -ra _days <<< "$dayObs"
    for d in "${_days[@]}"; do
        d="${d// /}"   # strip stray spaces
        [ -n "$d" ] && days+=("$d")
    done
fi

# The per-night targets (one night's full product set).
day_targets() {
    local d="$1"
    echo "output/night_${d}/guider_moments_${d}.parquet" \
         "output/night_${d}/guider_psfmoments_${d}.parquet" \
         "output/night_${d}/guider_plots_${d}.pdf" \
         "output/night_${d}/guider_movies_${d}.txt"
}

# All targets together (local mode / default rule all).
targets=()
for d in "${days[@]}"; do
    targets+=($(day_targets "$d"))
done

# --limit becomes a Snakemake config override (read as config["limit"]).
cfg=()
if [ -n "$limit" ]; then
    cfg=(--config "limit=${limit}")
fi

ts=$(date +%Y%m%d_%H%M%S)
tag=$(echo "${dayObs:-all}" | tr ', ' '__')   # filename-safe tag for logs

case "$mode" in
    local)
        log="logs/run_${tag}_${ts}.log"
        # --config is greedy, so it must come last (after targets).
        args=(-j 4 --resources mem_mb=14000 --keep-going "${passthru[@]}" "${targets[@]}" "${cfg[@]}")
        nohup snakemake "${args[@]}" > "$log" 2>&1 &
        echo "snakemake (local) launched: pid $!"
        echo "  targets: ${targets[*]:-<rule all>}"
        echo "  log:     $log   (tail -f $log)"
        ;;
    batch)
        command -v sbatch >/dev/null || {
            echo "error: sbatch not found — submit batch mode from an s3df" \
                 "interactive node (e.g. slacrd), not the RSP pod." >&2
            exit 2; }
        part=${SB_PARTITION:-roma}
        cpus=${SB_CPUS:-32}
        mem=${SB_MEM:-96G}
        tlim=${SB_TIME:-08:00:00}
        resmem=${SB_RESMEM:-90000}
        acct=${SB_ACCOUNT:-rubin:developers@roma}
        qos=${SB_QOS:-normal}

        # Submit ONE sbatch job per night (per-night isolation, parallel across
        # the cluster, easy re-submit of a single failed night). --config is
        # greedy, so it must come last (after targets).
        submit_one() {
            local jtag="$1"; shift
            local jlog="logs/batch_${jtag}_${ts}.out"
            local smk="snakemake -j ${cpus} --resources mem_mb=${resmem} --keep-going ${passthru[*]} $* ${cfg[*]}"
            sbatch --partition="$part" --account="$acct" --qos="$qos" \
                   --cpus-per-task="$cpus" --mem="$mem" --time="$tlim" \
                   --job-name="guider_${jtag}" --output="$jlog" \
                   --wrap "cd '$PWD' && ${smk}"
            echo "  ${jtag}: job log $jlog"
        }

        if [ ${#days[@]} -gt 0 ]; then
            echo "submitting ${#days[@]} job(s), one per night -> '$part'" \
                 "(${cpus} cpus, ${mem}, ${tlim} each)"
            for d in "${days[@]}"; do
                submit_one "$d" $(day_targets "$d")
            done
        else
            echo "submitting 1 job (rule all) -> '$part' (${cpus} cpus, ${mem}, ${tlim})"
            submit_one "$tag" "${targets[@]}"
        fi
        ;;
    *)
        echo "error: unknown --mode '$mode' (use 'local' or 'batch')" >&2
        exit 2;;
esac
