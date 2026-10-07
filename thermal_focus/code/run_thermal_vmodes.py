"""Run the all-v-mode thermal study: per-mode skill, the noise floor, the two intrinsic routes.

Three products, written to ``thermal_focus/output/thermal_vmodes/``:

* ``mode_table_<variant>.parquet`` -- one row per v-mode on the primary variant: night-grouped
  Huber skill against a median-intercept null, the Benjamini-Hochberg flag over the 34
  simultaneous tests, and the per-feature coefficients.
* ``noise_floor.parquet`` -- per-mode within-night against between-night scatter, which is what
  says where a fitted slope stops being interpretable.
* ``intrinsic_comparison.parquet`` -- per-mode skill on the two unconstrained 50/34 variants,
  batoid against the measured intrinsic wavefront (MIW), with the solver held fixed.
* ``thermal_vmodes.pdf`` -- the figures, one page per `thermal_vmodes_figures` function.

The sample is `run_thermal_focus.load_science`, so the selection funnel is the published one --
LUT-epoch night exclusion, 20 degC truss cut, science exposures in the fitted bands.
``load_science`` carries the 34 ``v*_olr`` columns through via ``keep_extra``; the ``y`` it sets
is v-mode 1 in physical units and is overwritten per mode by
`thermal_vmodes.attach_mode_response`.

Invocation::

    python code/run_thermal_vmodes.py
    python code/run_thermal_vmodes.py --no-intrinsic
    python code/run_thermal_vmodes.py --no-figures
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
import thermal_vmodes_channel_figures as TVCF                    # noqa: E402
import thermal_vmodes_channels as TVC                            # noqa: E402
import thermal_vmodes_figures as TVF                             # noqa: E402

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
    ap.add_argument('--no-figures', action='store_true', help='write the tables only')
    ap.add_argument('--no-channels', action='store_true',
                    help='skip the per-channel screen over the full thermal telemetry set')
    ap.add_argument('--rho-strong', type=float, default=TVC.RHO_STRONG,
                    help='|Spearman rho| at which a mode gets a combined follow-up fit')
    ap.add_argument('--n-lead', type=int, default=TVC.N_LEAD,
                    help='channels combined for a mode that clears --rho-strong')
    ap.add_argument('--dup-rho', type=float, default=TVC.CHANNEL_DUP_RHO,
                    help='|rho| between channels above which the weaker is skipped as a '
                         'duplicate when assembling a leading set')
    ap.add_argument('--pdf-name', default='thermal_vmodes.pdf', help='PDF filename')
    ap.add_argument('--channel-pdf-name', default='thermal_vmodes_channels.pdf',
                    help='PDF filename for the per-channel screen')
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

    cmp = None
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

    if not args.no_figures:
        print('\n=== figures ===')
        pdf_path = TVF.write_pdf(out_dir / args.pdf_name, df, tab, floor, cmp=cmp)
        print(f'wrote {pdf_path}')

    if not args.no_channels:
        print('\n=== per-channel screen over the full thermal telemetry set ===')
        dfc = TVC.attach_differences(df)
        grid = TVC.channel_grid(dfc, n_modes=args.n_modes)
        grid_path = out_dir / f'channel_grid_{args.variant}.parquet'
        grid.to_parquet(grid_path, index=False)
        print(f'wrote {grid_path}')

        print('\n=== M1M3 shape channels against the S-matrix prediction ===')
        exp = TVC.expectation_check(grid)
        if len(exp):
            exp_path = out_dir / 'channel_expectation.parquet'
            exp.to_parquet(exp_path, index=False)
            print(f'wrote {exp_path}')

        # Which v-mode carries which Zernike is not the mode index -- v12 is Z15, not spherical
        # -- so the prediction is tested against the modes that actually hold Z4, Z11 and Z22.
        print()
        try:
            content = TVC.vmode_zernike_content(n_modes=args.n_modes)
            pred = TVC.prediction_table(grid, content)
            cpath = out_dir / 'vmode_zernike_content.parquet'
            content.to_parquet(cpath)
            print(f'wrote {cpath}')
            ppath = out_dir / 'channel_prediction.parquet'
            pred.to_parquet(ppath, index=False)
            print(f'wrote {ppath}')
        except Exception as exc:                                  # noqa: BLE001
            print(f'v-mode Zernike content unavailable ({type(exc).__name__}: {exc}); '
                  f'skipping the prediction test')
            content, pred = None, None

        print(f'\n=== combined fits for modes past |rho| {args.rho_strong} ===')
        comb, results = TVC.combined_table(dfc, grid, rho_strong=args.rho_strong,
                                           n_lead=args.n_lead, dup_rho=args.dup_rho)
        if len(comb):
            comb_path = out_dir / 'channel_combined.parquet'
            comb.to_parquet(comb_path, index=False)
            print(f'wrote {comb_path}')
            summary = TVC.nmad_summary(results)
            sum_path = out_dir / 'channel_nmad_summary.parquet'
            summary.to_parquet(sum_path, index=False)
            print(f'wrote {sum_path}')
        else:
            print(f'no mode reaches |Spearman rho| {args.rho_strong} on any channel; '
                  f'no combined fit to do')

        if not args.no_figures:
            cpdf = TVCF.write_pdf(out_dir / args.channel_pdf_name, grid, results,
                                  exp=exp, df=dfc, pred=pred, content=content)
            print(f'wrote {cpdf}')


if __name__ == '__main__':
    main()
