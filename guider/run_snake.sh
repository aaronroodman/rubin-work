#!/usr/bin/env bash
# Launch the guider moment pipeline, locally or as a Slurm batch job.
#
# Modes (--mode, default local):
#   local  — run snakemake on THIS node, detached via nohup (survives a dropped
#            SSH/network connection).  Tuned for the RSP terminal (~4 cores).
#            Use from the USDF JupyterLab RSP (which cannot submit to Slurm).
#   batch  — submit ONE sbatch job to s3df that runs snakemake with many
#            parallel slots (a roma node has far more than 4 cores).  Submit
#            from an s3df interactive node (e.g. the `slacrd` ssh alias) where
#            sbatch + the LSST stack are available.  Best for a full night.
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

# Targets: one or more whole nights (comma-separated), or the default `rule all`.
targets=()
if [ -n "$dayObs" ]; then
    IFS=',' read -ra _days <<< "$dayObs"
    for d in "${_days[@]}"; do
        d="${d// /}"   # strip stray spaces
        [ -n "$d" ] || continue
        targets+=("output/night_${d}/moments" "output/night_${d}/plots/validation.png")
    done
fi

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
        jlog="logs/batch_${tag}_${ts}.out"
        # --config is greedy, so it must come last (after targets).
        smk="snakemake -j ${cpus} --resources mem_mb=${resmem} --keep-going ${passthru[*]} ${targets[*]} ${cfg[*]}"
        sbatch --partition="$part" --account="$acct" --qos="$qos" \
               --cpus-per-task="$cpus" --mem="$mem" --time="$tlim" \
               --job-name="guider_${tag}" --output="$jlog" \
               --wrap "cd '$PWD' && ${smk}"
        echo "submitted batch job -> '$part' (${cpus} cpus, ${mem}, ${tlim})"
        echo "  targets: ${targets[*]:-<rule all>}"
        echo "  job log: $jlog   (tail -f $jlog)"
        ;;
    *)
        echo "error: unknown --mode '$mode' (use 'local' or 'batch')" >&2
        exit 2;;
esac
