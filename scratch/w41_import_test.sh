source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_41/loadLSST.sh
setup lsst_distrib
P=/sdf/home/r/roodman/u/LSST/packages
for pkg in summit_utils ts_utils ts_salobj ts_idl ts_observatory_control \
           ts_aos_analysis ts_ofc ts_wep ts_xml ts_config_mttcs ts_m1m3_utils \
           ts_intrinsic_wavefront; do
  setup -r $P/$pkg -j
done
export OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
echo "### scipy/numpy and the key imports"
python -c "
import scipy, numpy, astropy, pandas
print('### scipy', scipy.__version__, '| numpy', numpy.__version__, '| astropy', astropy.__version__, '| pandas', pandas.__version__)
import danish, batoid, batoid_rubin, galsim
print('### danish', danish.__version__, '| batoid', batoid.__version__, '| batoid_rubin', batoid_rubin.__version__, '| galsim', galsim.__version__)
import lsst.ts.wep, lsst.ts.ofc, lsst.summit.utils
import lsst.ts.intrinsic.wavefront as iw
print('### ts_wep, ts_ofc, summit_utils, ts_intrinsic_wavefront all import')
import rubinwork
from rubinwork.products import catalog, manifest
print('### rubinwork', rubinwork.__version__ if hasattr(rubinwork,'__version__') else '(no __version__)')
s = manifest.stack_state()
print('### stack_state eups_tag:', s['eups_tag'], '| conda:', s['conda_env'], '| n_pkgs:', len(s['packages']))
"
