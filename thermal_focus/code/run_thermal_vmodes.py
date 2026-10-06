"""Run the all-v-mode thermal study: per-mode skill, the noise floor, the two intrinsic routes.

Three products, written to ``thermal_focus/output/thermal_vmodes/``:

* ``mode_table_<variant>.parquet`` -- one row per v-mode on the primary variant: night-grouped
  Huber skill against a median-intercept null, the Benjamini-Hochberg flag over the 34
  simultaneous tests, and the per-feature coefficients.
* ``noise_floor.parquet`` -- per-mode within-night against between-night scatter, which is what
  says where a fitted slope stops being interpretable.
* ``intrinsic_comparison.parquet`` -- per-mode skill on the two unconstrained 50/34 variants,
  batoid against the measured intrinsic wavefront (MIW), with the solver held fixed.

The sample is `run_thermal_focus.load_science`, so the selection funnel is the published one --
LUT-epoch night exclusion, 20 degC truss cut, science exposures in the fitted bands. ``load_science``
carries the 34 ``v*_olr`` columns through via ``keep_extra``; the ``y`` it sets is v-mode 1 in
physical units and is overwritten per mode by `thermal_vmodes.attach_mode_response`.

Invocation::

    python code/run_thermal_vmodes.py
    python code/run_thermal_vmodes.py --no-intrinsic
    python code/run_thermal_vmodes.py --day-obs-range 20251103 20260713

This stage needs the network: the truss temperature is derived on a ConsDB join rather than
stored, so each variant costs one round trip per night. ``--no-intrinsic`` skips the two extra
loads when only the primary table is wanted.
"""
import argparse
import pathlib
import sys
import warnings

import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                   # -> thermal_vmodes
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))                                   # repo root -> common/

import run_thermal_focus as RTF                                  # noqa: E402
import thermal_focus_lib as L                                    # noqa: E402
import thermal_vmodes as TV                                      # noqa: E402

# optical_state(wide=True) inserts one column at a time, so pandas warns once per expanded
# column -- hundreds of lines that bury the result.
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def load(variant, day_obs_range, verbose=True):
    """Load one variant's science sample with the open-loop v-mode columns attached.

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
        One row per science visit, carrying the 34 ``v*_olr`` columns and the thermal features.
    """
    v1_per_um_dz = L.v1_per_um_dz_value()
    df = RTF.load_science(variant, day_obs_range, v1_per_um_dz, verbose=verbose,
                          keep_extra=TV.response_columns())
    have = [c for c in TV.response_columns() if c in df.columns]
    if len(have) < TV.N_MODES:
        raise KeyError(f'{variant} carries {len(have)} of {TV.N_MODES} v*_olr columns; the '
                       f'variant is either not 50/34 or predates the open-loop schema')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--variant', default=TV.PRIMARY_VARIANT,
                    help='primary variant; default the range-bounded recovery')
    ap.add_argument('--day-obs-range', nargs=2, type=int, default=None,
                    metavar=('LO', 'HI'), help='inclusive night range YYYYMMDD')
    ap.add_argument('--n-modes', type=int, default=TV.N_MODES)
    ap.add_argument('--no-intrinsic', action='store_true',
                    help='skip the batoid-against-MIW comparison')
    ap.add_argument('--output-dir', default=None,
                    help='where to write; default thermal_focus/output/thermal_vmodes')
    args = ap.parse_args()

    out_dir = (pathlib.Path(args.output_dir) if args.output_dir
               else _ROOT / 'thermal_focus' / 'output' / 'thermal_vmodes')
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = tuple(args.day_obs_range) if args.day_obs_range else None

    print(f'=== primary variant {args.variant} ===')
    df = load(args.variant, rng)
    print(f'{len(df)} visits, {df["day_obs"].nunique()} nights')

    print('\n=== per-mode thermal skill ===')
    tab = TV.mode_table(df, variant=args.variant, n_modes=args.n_modes)
    path = out_dir / f'mode_table_{args.variant}.parquet'
    tab.to_parquet(path, index=False)
    print(f'wrote {path}')

    print('\n=== per-mode noise floor ===')
    floor = TV.noise_floor_table(df, n_modes=args.n_modes)
    floor_path = out_dir / 'noise_floor.parquet'
    floor.to_parquet(floor_path, index=False)
    print(f'wrote {floor_path}')

    if not args.no_intrinsic:
        tabs = {}
        for variant in TV.INTRINSIC_PAIR:
            print(f'\n=== intrinsic route {variant} ===')
            d = load(variant, rng, verbose=False)
            print(f'{len(d)} visits, {d["day_obs"].nunique()} nights')
            tabs[variant] = TV.mode_table(d, variant=variant, n_modes=args.n_modes)

        print('\n=== batoid against MIW, solver fixed ===')
        cmp = TV.intrinsic_comparison(tabs[TV.INTRINSIC_PAIR[0]], tabs[TV.INTRINSIC_PAIR[1]])
        cmp_path = out_dir / 'intrinsic_comparison.parquet'
        cmp.to_parquet(cmp_path, index=False)
        print(f'wrote {cmp_path}')


if __name__ == '__main__':
    main()
