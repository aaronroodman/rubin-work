# Test the six local packages against w_2026_41. Does not touch ~/u/LSST/setup.sh.
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_41/loadLSST.sh
setup lsst_distrib
echo "### lsst_distrib: $(eups list lsst_distrib | tr -s ' ')"
echo "### conda env: $CONDA_DEFAULT_ENV"
P=/sdf/home/r/roodman/u/LSST/packages
for pkg in summit_utils ts_utils ts_salobj ts_idl ts_observatory_control \
           ts_aos_analysis ts_ofc ts_wep ts_xml ts_config_mttcs ts_m1m3_utils \
           ts_intrinsic_wavefront; do
  out=$(setup -r $P/$pkg -j 2>&1)
  if [ -n "$out" ]; then
    echo "### FAIL $pkg: $out"
  else
    echo "### ok   $pkg"
  fi
done
