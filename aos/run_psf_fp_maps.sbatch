#!/bin/bash
#SBATCH --job-name=psf_fp
#SBATCH --partition=roma
#SBATCH --account=rubin:developers@roma
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/psf_fp_%j.log
#
# GalSim/HSM focal-plane PSF maps (code/run_psf_fp_maps.py) on s3df.  One job per
# option combination — pass the run-specific options on the sbatch line; they go
# straight to the python (argparse last-wins over the baked DEFAULTS below).
#
# Needs galsim + ts_wep + ts_ofc (+ TS_CONFIG_MTTCS_DIR for the OFC norm yaml) +
# batoid_rubin + cameraGeom — the cvmfs w_2026_25 stack via the user's setup.sh.
#
# Submit from an s3df node (slacrd), from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos
#   sbatch run_psf_fp_maps.sbatch --case loop50 --intrinsic tabulated --order ordered
#   sbatch run_psf_fp_maps.sbatch --case loop50 --intrinsic miw      --order ordered
#   sbatch run_psf_fp_maps.sbatch --case loop --max-visits 0 --order random --intrinsic tabulated
#   sbatch run_psf_fp_maps.sbatch -n            # NOT a thing; use --max-visits 4 for a quick check
#
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

set +e
source "$HOME/u/LSST/setup.sh"
set -e
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# fail fast if the env can't do the work
python - <<'PY'
import galsim                                          # noqa: F401
import lsst.afw.cameraGeom, lsst.obs.lsst              # noqa: F401
import lsst.ts.wep, lsst.ts.ofc, batoid_rubin          # noqa: F401
import os
print(f"OK: galsim {galsim.__version__}, ts_wep/ts_ofc/batoid_rubin importable")
if not os.environ.get("TS_CONFIG_MTTCS_DIR"):
    print("WARNING: TS_CONFIG_MTTCS_DIR unset -> build_ofc_svd will fail for loop/fam/mimic cases "
          "(setup ts_config_mttcs).")
PY

# Baked defaults: the new collection, full visit set, 1000 stars.  ANY option on the
# sbatch line overrides (argparse last occurrence wins).
DEFAULTS=(--ps fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x --band i --max-visits 0 --n-stars 1000)
echo "running: code/run_psf_fp_maps.py ${DEFAULTS[*]} $*"
python code/run_psf_fp_maps.py "${DEFAULTS[@]}" "$@"
