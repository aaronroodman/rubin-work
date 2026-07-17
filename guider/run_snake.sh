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
#   --day-obs YYYYMMDD  process a whole night: discover its guider exposures
#                       from the Butler and build the partitioned dataset +
#                       validation plot (output/night_YYYYMMDD/).
#
# Usage:
#   ./run_snake.sh --day-obs 20260709                 # local, one night
#   ./run_snake.sh --day-obs 20260709 -n              # dry-run
#   ./run_snake.sh --day-obs 20260709 --mode batch    # batch on roma
#   ./run_snake.sh                                     # local, static datasets
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
passthru=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)     mode="$2"; shift 2;;
        --mode=*)   mode="${1#*=}"; shift;;
        --day-obs)  dayObs="$2"; shift 2;;
        --day-obs=*) dayObs="${1#*=}"; shift;;
        *)          passthru+=("$1"); shift;;
    esac
done

# Targets: a whole night, or the default `rule all`.
targets=()
if [ -n "$dayObs" ]; then
    targets=("output/night_${dayObs}/moments" "output/night_${dayObs}/plots/validation.png")
fi

ts=$(date +%Y%m%d_%H%M%S)
tag="${dayObs:-all}"

case "$mode" in
    local)
        log="logs/run_${tag}_${ts}.log"
        args=(-j 4 --resources mem_mb=14000 --keep-going "${passthru[@]}" "${targets[@]}")
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
        smk="snakemake -j ${cpus} --resources mem_mb=${resmem} --keep-going ${passthru[*]} ${targets[*]}"
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
