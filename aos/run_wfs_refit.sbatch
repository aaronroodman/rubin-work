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
set -eo pipefail               # NOT -u: the lsst/conda (de)activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"   # submit dir (= aos/); $0 is Slurm's spool copy under sbatch
mkdir -p logs

# ---- LSST env: reuse the user's own setup.sh (cvmfs w_2026_25 danish 1.2 + local
#      ts_wep/ts_ofc clones + BUTLER_CONFIG).  set +e around it: its many `setup -r`
#      / conda-activate lines can return nonzero without being fatal.
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}   # override setup.sh's 1, silences the warning
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# fail fast if the env isn't what we need
python - <<'PY'
import danish
assert hasattr(danish, "DonutTriangleFactory"), f"danish {danish.__version__} lacks DonutTriangleFactory"
import lsst.ts.wep, batoid_rubin   # noqa: F401
print(f"OK: danish {danish.__version__}, ts_wep + batoid_rubin importable")
PY

# default args if none given on the sbatch command line
# Baked-in param-set + baseline knobs; ANY args on the sbatch line override these
# (argparse uses the last occurrence of a repeated option). --height-map-dir defaults
# to ~/u/LSST/packages/batoid_rubin_data inside the python.
DEFAULTS=(--param-set fam_danish_1_0_wep17_3_0_bin2x --b-min 20 --alphas 0.0,0.01,0.02,0.03,0.05)
echo "running: code/run_wfs_refit_ensemble.py ${DEFAULTS[*]} $*"
python code/run_wfs_refit_ensemble.py "${DEFAULTS[@]}" "$@"
