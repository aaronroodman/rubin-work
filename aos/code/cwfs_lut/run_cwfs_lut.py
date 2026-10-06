"""Run the pointing-dependence study: open-loop DOF against elevation and rotator angle.

Products, written to ``aos/output/cwfs_lut/``:

* ``trend_<variant>_<angle>.parquet`` -- one row per degree of freedom (DOF): the Huber robust
  slope against one pointing angle, its formal error, the robust residual scatter, and both
  Pearson r and Spearman rho. Written for every variant and both angles.
* ``intrinsic_spread_<angle>.parquet`` -- the same slopes on the two intrinsic routes side by
  side, batoid against the measured intrinsic wavefront (MIW), with the solver held fixed.
* ``bounce_comparable_<angle>.parquet`` -- the subset a bounce test can be compared against,
  in the bounce test's own units.

The quantity fitted is the stored open-loop state ``Deviation - Trim``, which is what a look-up
table has to supply. Trends are **absolute**, not within-night paired differences: a typical
science night sweeps elevation over roughly 33 to 80 deg and rotator angle over roughly -80 to
+79 deg within that one night, far more leverage than the bounce test's paired +-3 deg legs. The
cost is that an absolute elevation trend confounds gravity with thermal drift that tracks
elevation, which the study document states as a caveat rather than pairing away.

**The tilt DOF are stored in deg, and the bounce test reports them in arcsec.** The
``bounce_comparable`` product applies 3600 arcsec/deg to DOF 3, 4, 8 and 9 and nothing to the
other 46; `cwfs_lut_lib.to_bounce_units` is the only place that conversion lives. Getting it
wrong is silent -- it leaves the dominant decentre terms correct and corrupts only the tilts.

Invocation::

    python code/cwfs_lut/run_cwfs_lut.py
    python code/cwfs_lut/run_cwfs_lut.py --day-obs-range 20260101 20260713
    python code/cwfs_lut/run_cwfs_lut.py --variants v50_34_rbr__batoid__consdb_v1

Reads the value-added DuckDB only -- no Butler, no ConsDB, no network.
"""
import argparse
import pathlib
import sys
import warnings

import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                   # -> cwfs_lut_lib
_ROOT = _HERE.parents[2]                                         # repo root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'value_added' / 'code'))           # -> efd_db

import cwfs_lut_lib as C                                         # noqa: E402
import efd_db                                                    # noqa: E402

# optical_state(wide=True) inserts one column at a time, so pandas warns once per expanded
# column -- hundreds of lines that bury the result.
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

#: Both pointing angles, each fitted independently. A joint fit is not used: the two angles are
#: correlated through the observing pattern, so a joint slope is not the LUT axis it looks like.
ANGLES = ('elevation_deg', 'rotator_angle_deg')


def load(variant, day_obs_range, verbose=True):
    """Read one variant's recovered states with both pointing angles present.

    Parameters
    ----------
    variant : `str`
        Recovered-optical-state variant id.
    day_obs_range : `tuple` [`int`] or `None`
        Inclusive ``(lo, hi)`` night range as ``YYYYMMDD``, or `None` for everything.
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        One row per recovered visit carrying the 50 ``dof*_olr`` columns and both angles.
    """
    st = efd_db.optical_state(variant, day_obs_range=day_obs_range, wide=True, ok_only=True)
    n_read = len(st)
    for angle in ANGLES:
        if angle not in st.columns:
            raise KeyError(f'{variant} lacks {angle}; the pointing columns arrived in 090b185')
    have = [c for c in C.olr_columns() if c in st.columns]
    if len(have) < 50:
        raise KeyError(f'{variant} carries {len(have)} of 50 dof*_olr columns')

    st = st.dropna(subset=list(ANGLES))
    if verbose:
        print(f'{variant}: {n_read} recovered rows, {len(st)} with both pointing angles, '
              f'{st["day_obs"].nunique()} nights')
        for angle in ANGLES:
            print(f'  {angle} spans {st[angle].min():.2f} to {st[angle].max():.2f} deg')
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--variants', nargs='+', default=None,
                    help='variant ids; default the RBR variant plus both intrinsic routes')
    ap.add_argument('--day-obs-range', nargs=2, type=int, default=None,
                    metavar=('LO', 'HI'), help='inclusive night range YYYYMMDD')
    ap.add_argument('--output-dir', default=None,
                    help='where to write; default aos/output/cwfs_lut')
    args = ap.parse_args()

    variants = args.variants or [C.RBR_VARIANT, C.INTRINSIC_VARIANTS['batoid'],
                                 C.INTRINSIC_VARIANTS['miw']]
    out_dir = (pathlib.Path(args.output_dir) if args.output_dir
               else _ROOT / 'aos' / 'output' / 'cwfs_lut')
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = tuple(args.day_obs_range) if args.day_obs_range else None

    tabs, frames = {}, {}
    for variant in variants:
        print(f'\n=== {variant} ===')
        df = frames[variant] = load(variant, rng)
        for angle in ANGLES:
            print()
            tab = C.trend_table(df, angle)
            tabs[(variant, angle)] = tab
            path = out_dir / f'trend_{variant}_{angle}.parquet'
            tab.to_parquet(path, index=False)
            print(f'wrote {path}')

    bat, miw = C.INTRINSIC_VARIANTS['batoid'], C.INTRINSIC_VARIANTS['miw']
    for angle in ANGLES:
        if (bat, angle) not in tabs or (miw, angle) not in tabs:
            continue
        print(f'\n=== intrinsic routes against {angle} ===')
        cmp = C.intrinsic_spread(tabs[(bat, angle)], tabs[(miw, angle)])
        path = out_dir / f'intrinsic_spread_{angle}.parquet'
        cmp.to_parquet(path, index=False)
        print(f'wrote {path}')

    # All ten rigid-body axes in the bounce test's own units, which is arcsec on the four tilts
    # and deg nowhere. `comparable` marks the six decentres and pistons, the only terms both
    # retrievals measure the same way -- the bounce test uses full-focal-plane wavefronts and
    # this uses four corner sensors.
    for angle in ANGLES:
        if C.RBR_VARIANT not in frames:
            continue
        print(f'\n=== rigid-body DOF against {angle}, bounce units ===')
        tab = C.trend_table(frames[C.RBR_VARIANT], angle, bounce_units=True)
        tab['comparable'] = tab['dof'].isin(C.BOUNCE_COMPARABLE_DOF)
        path = out_dir / f'bounce_comparable_{angle}.parquet'
        tab.to_parquet(path, index=False)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
