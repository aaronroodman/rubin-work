#!/bin/bash
#SBATCH --job-name=rubinwork_import
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/rubinwork_import_%j.log
#
# Check that the rubinwork package and the phase 1 compatibility shims import on a
# batch node.  A Slurm job gets a non-interactive shell, so the stack is not set up
# by .bashrc and the user-site install of rubinwork has to be found through the
# stack's python; and the RSP-only /home/r/roodman/u/... paths do not resolve here.
#
# Submit from an s3df node (slacrd), never from an RSP pod:
#   cd ~/notebooks/rubin-work
#   sbatch rubinwork/common/scripts/check_batch_import.sl
#
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

set +e
source /sdf/group/rubin/u/roodman/LSST/setup.sh
set -e

REPO=/sdf/home/r/roodman/notebooks/rubin-work

echo "host    = $(hostname)"
echo "python  = $(which python)"
python -c "import sys; print('version =', sys.version.split()[0])"

echo
echo "--- new-style package imports ---"
set +e
python -c "
import rubinwork, rubinwork.common, rubinwork.products.catalog
print('rubinwork from', rubinwork.__file__)
from rubinwork.common.utils import nmad
print('rubinwork.common.utils.nmad ok')
from rubinwork.products import catalog
root = catalog.data_root()
print('data root =', root, '| exists:', root.is_dir())
print('n_builds =', len(catalog.list_products()), '(count, dimensionless)')
"
new_status=$?

echo
echo "--- shimmed old-style imports (repo root on sys.path, as the scripts do) ---"
cd "$REPO/aos/code" || exit 1
PYTHONPATH="$REPO:$REPO/aos/code" python -c "
from common.utils import nmad
print('from common.utils import nmad ok')
import aos_state
print('import aos_state ok ->', aos_state.__name__)
"
shim_status=$?

echo
if [ "$new_status" -eq 0 ] && [ "$shim_status" -eq 0 ]; then
    echo "RESULT: pass (package and shim both import on the batch node)"
    exit 0
fi
echo "RESULT: fail (new_status=$new_status shim_status=$shim_status)"
exit 1
