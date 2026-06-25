#!/bin/bash
#SBATCH --job-name=wfs_refit
#SBATCH --partition=roma
#SBATCH --account=rubin:developers@roma
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wfs_refit_%j.log
#
# Ensemble corner-WFS Danish-1.2 triangle refit (run_wfs_refit_ensemble.py) on s3df.
# Needs danish >= 1.2 (DonutTriangleFactory), which the /sdf shared env does NOT have --
# so we source the cvmfs weekly that ships it.
#
# Submit from an s3df node (slacrd), from the aos/ dir (so logs/ exists):
#     cd ~/notebooks/rubin-work/aos
#     sbatch run_wfs_refit.sbatch                       # default args (below)
#     sbatch run_wfs_refit.sbatch --max-visits 30 --alphas 0.0,0.02,0.05   # override
#
set -euo pipefail
cd "$(dirname "$0")"            # aos/
mkdir -p logs

# ---- danish 1.2 stack from cvmfs (pinned weekly; newer ones also work) ----
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_25/loadLSST.sh
setup lsst_distrib
# If you need your local ts_wep clone instead of the weekly's, uncomment (verify it is
# compatible with this weekly's danish 1.2):
# setup -r ~/u/LSST/packages/ts_wep -j

export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# fail fast if the env isn't what we need
python - <<'PY'
import danish
assert hasattr(danish, "DonutTriangleFactory"), f"danish {danish.__version__} lacks DonutTriangleFactory"
import lsst.ts.wep, batoid_rubin   # noqa: F401
print(f"OK: danish {danish.__version__}, ts_wep + batoid_rubin importable")
PY

# default args if none given on the sbatch command line
if [ $# -eq 0 ]; then
    set -- --param-set fam_danish_1_0_wep17_3_0_bin2x \
           --b-min 20 --alphas 0.0,0.01,0.02,0.03,0.05 \
           --height-map-dir /sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data
fi
echo "running: code/run_wfs_refit_ensemble.py $*"
python code/run_wfs_refit_ensemble.py "$@"
