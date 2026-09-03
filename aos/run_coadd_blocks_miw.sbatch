#!/bin/bash
#SBATCH --job-name=coadd_miw
#SBATCH --partition=roma
#SBATCH --account=rubin:developers@roma
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/coadd_miw_%j.log
#
# Per-FAM-block MIW-style coadds (code/run_coadd_blocks_miw.py) on s3df.
#
# Why batch: the job holds every block's donut table plus the stacked coadd/MIW
# map arrays, and now also the DZ primitives and the extra secondary-Zernike
# maps -- enough to OOM an RSP notebook pod.  128G is generous headroom; the
# observed peak is a few tens of GB, dominated by the per-block donut loads and
# the 221-page per-block PDF render.
#
# Needs lsst.ts.intrinsic.wavefront + lsst.ts.ofc (+ TS_CONFIG_MTTCS_DIR for the
# OFC normalization yaml) and the mi_name MIW sidecar (zk_intrinsic.parquet).
#
# Submit from an s3df node (e.g. the `slacrd` ssh alias), from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos
#   sbatch run_coadd_blocks_miw.sbatch
#   sbatch run_coadd_blocks_miw.sbatch --ext-terms all
#   sbatch run_coadd_blocks_miw.sbatch --out-name coadd_50_34_v2   # keep the old products
#
# Watch:  squeue --me    |    tail -f logs/coadd_miw_<jobid>.log
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
import os
import lsst.ts.ofc                                     # noqa: F401
from lsst.ts.intrinsic.wavefront import ofc_svd        # noqa: F401
from lsst.ts.intrinsic.wavefront import measured_intrinsic   # noqa: F401
print("OK: ts_ofc + ts_intrinsic_wavefront importable")
if not os.environ.get("TS_CONFIG_MTTCS_DIR"):
    raise SystemExit("FATAL: TS_CONFIG_MTTCS_DIR unset -> build_ofc_svd and the "
                     "residual subspace bases will fail (setup ts_config_mttcs).")
PY

# Baked defaults: the current collection and the 5rot 50/34 build knobs, with the
# secondary/tertiary pupil Zernikes mapped into block_grids_ext.npz.  ANY option on
# the sbatch line overrides (argparse last occurrence wins).
DEFAULTS=(--param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
          --mi-name pathA_50_34_i_5rot
          --ext-terms secondary)
echo "running: code/run_coadd_blocks_miw.py ${DEFAULTS[*]} $*"
python code/run_coadd_blocks_miw.py "${DEFAULTS[@]}" "$@"
