"""Run the pointing-dependence study: open-loop DOF against elevation and rotator angle.

Products, written to ``aos/output/cwfs_lut/``:

* ``trend_<variant>_<angle>.parquet`` -- one row per degree of freedom (DOF): the Huber robust
  slope against one pointing angle, its formal error, the robust residual scatter, and both
  Pearson r and Spearman rho. Written for every variant and both angles.
* ``intrinsic_spread_<angle>.parquet`` -- the same slopes on the two intrinsic routes side by
  side, batoid against the measured intrinsic wavefront (MIW), with the solver held fixed.
* ``bounce_comparable_<angle>.parquet`` -- the survey side of the bounce comparison: the ten
  rigid-body axes, with `comparable` marking the six the two retrievals measure alike.
* ``bounce_compare_per_dof_<angle>.parquet``, ``bounce_compare_lateral_sum_<angle>.parquet``,
  ``bounce_compare_subspace_<angle>.parquet``, ``bounce_compare_vmode_<angle>.parquet`` and
  ``bounce_compare_elevation_legs.parquet`` -- the comparison itself, in the three spaces, with
  every (bounce arm, survey arm, variant) pairing in one long frame.
* ``cwfs_lut.pdf`` -- the figures, one page per `cwfs_lut_figures` function.

The quantity fitted is the stored open-loop state ``Deviation - Trim``, which is what a look-up
table has to supply. Trends are **absolute**, not within-night paired differences: a typical
science night sweeps elevation over roughly 33 to 80 deg and rotator angle over roughly -80 to
+79 deg within that one night, far more leverage than the bounce test's paired +-3 deg legs. The
cost is that an absolute elevation trend confounds gravity with thermal drift that tracks
elevation, which the study document states as a caveat rather than pairing away.

The tilt DOF are stored in deg and the bounce test stores them in deg too, so **nothing
converts** -- see `cwfs_lut_lib`'s module docstring for why the arcsec label on both sides is
wrong. What the comparison does have to match is the **arm**: the bounce Δ is a paired
difference of the recovered deviation, while ``dof*_olr`` is ``Deviation - Trim``.

Invocation::

    python code/cwfs_lut/run_cwfs_lut.py
    python code/cwfs_lut/run_cwfs_lut.py --no-figures
    python code/cwfs_lut/run_cwfs_lut.py --no-bounce
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

import bounce_compare as B                                       # noqa: E402
import cwfs_lut_figures as CF                                    # noqa: E402
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
        One row per recovered visit carrying both pointing angles, the 50 ``dof*_olr`` columns
        and the 50 ``dof*`` deviation columns the bounce comparison matches against.
    """
    st = efd_db.optical_state(variant, day_obs_range=day_obs_range, wide=True, ok_only=True)
    n_read = len(st)
    for angle in ANGLES:
        if angle not in st.columns:
            raise KeyError(f'{variant} lacks {angle}; the pointing columns arrived in 090b185')
    have = [c for c in C.olr_columns() if c in st.columns]
    if len(have) < 50:
        raise KeyError(f'{variant} carries {len(have)} of 50 dof*_olr columns')
    # The bounce Δ is a deviation-arm quantity, so the comparison needs `dofN` as well as
    # `dofN_olr`. Fail here rather than silently comparing two different quantities.
    deviation = [f'dof{j}' for j in range(50) if f'dof{j}' in st.columns]
    if len(deviation) < 50:
        raise KeyError(f'{variant} carries {len(deviation)} of 50 dof* deviation columns; the '
                       f'bounce comparison matches the deviation arm, not the open-loop one')

    st = st.dropna(subset=list(ANGLES))
    if verbose:
        print(f'{variant}: {n_read} recovered rows, {len(st)} with both pointing angles, '
              f'{st["day_obs"].nunique()} nights')
        for angle in ANGLES:
            print(f'  {angle} spans {st[angle].min():.2f} to {st[angle].max():.2f} deg')
    return st


def run_bounce_comparison(frames, out_dir, run=B.DEFAULT_BOUNCE_RUN,
                          min_significance=B.MIN_SIGNIFICANCE):
    """Compare every variant against the bounce test and write the five products.

    Parameters
    ----------
    frames : `dict` [`str`, `pandas.DataFrame`]
        Per-visit frames keyed by variant id.
    out_dir : `pathlib.Path`
        Where to write.
    run : `str`, optional
        Bounce run directory under ``aos/output/bounce``.
    min_significance : `float`, optional
        Bounce |Δ|/error floor for the subspace statistics.

    Returns
    -------
    bounce : `dict` or `None`
        Products for `cwfs_lut_figures.write_pdf`, or `None` if the bounce run is absent.

    Notes
    -----
    Returns `None` rather than raising when the bounce products are missing: they belong to the
    `bounce` study and will not exist in every checkout, and this study's own results do not
    depend on them.
    """
    try:
        stats = B.load_bounce_stats(run=run)
    except FileNotFoundError as exc:
        print(f'\n=== bounce comparison skipped ===\n{exc}')
        return None

    # The rotator leg is the sharp test: one 60.017 deg throw with elevation pinned to 0.003
    # deg, so its recovered change is rotator-only.
    angle = B.BOUNCE_ANGLE['T724_rotator']
    print(f'\n=== bounce comparison, {run} ===')
    for name in B.BOUNCE_ANGLE:
        legs = B.bounce_legs(stats, bounce=name)
        print(f'{name}: {len(legs)} leg(s) throwing {B.BOUNCE_ANGLE[name]}')
        for _, r in legs.iterrows():
            print(f'  {r["comparison"]:<10s} throw {r["throw_deg"]:+7.2f} deg, '
                  f'cross-throw {r["cross_throw_deg"]:+6.2f} deg, '
                  f'{int(r["n_pairs"])} pairs')

    res = B.compare_all(stats, frames, angle, min_significance=min_significance)
    head = (res['subspace']['bounce_arm'].eq('svd')
            & res['subspace']['survey_arm'].eq('deviation'))
    print('\nagreement by subspace, bounce svd against survey deviation (the headline):')
    print(f'  {"subspace":<20s} {"cosine":>8s} {"scale":>8s} {"n sig":>6s}  variant')
    for _, r in res['subspace'][head].iterrows():
        print(f'  {r["subspace"]:<20s} {r["cosine_similarity"]:>+8.3f} '
              f'{r["scale"]:>+8.3f} {int(r["n_significant"]):>6d}  {r["variant"]}')
    print('  cosine and scale are dimensionless; scale is survey over bounce amplitude')

    lat = res['lateral_sum']
    lat_head = lat['bounce_arm'].eq('svd') & lat['survey_arm'].eq('deviation')
    print('\nlateral sums over both hexapods [µm per deg of rotator angle]:')
    for _, r in lat[lat_head].iterrows():
        print(f'  M2 + camera {r["axis"]}: bounce {r["slope_bounce"]:+8.3f} '
              f'survey {r["slope_survey"]:+8.3f} ratio {r["ratio"]:+7.3f} (dimensionless)')

    legs, fits = B.compare_elevation_legs(stats, frames[next(iter(frames))])
    print('\nelevation legs, weighted linear fit through the origin:')
    for _, r in fits.iterrows():
        print(f'  {r["label"]:<18s} {r["slope_linear"]:+8.3f} +- {r["slope_linear_err"]:.3f} '
              f'{r["unit"]}/deg, chi2/dof {r["chi2_linear"]:6.2f} '
              f'(dof {int(r["dof_linear"])}), cos-fit chi2/dof {r["chi2_cos"]:6.2f}')
    print('  chi2/dof above 1 reflects turbulence the per-leg errors do not capture; the '
          'linear term is a first-order approximation, not a rejected model')

    products = {'per_dof': res['per_dof'], 'lateral_sum': res['lateral_sum'],
                'subspace': res['subspace'], 'vmode': res['vmode'],
                'elevation_legs': legs, 'elevation_fits': fits}
    for name, tab in products.items():
        if tab is None or not len(tab):
            continue
        suffix = '' if name.startswith('elevation') else f'_{angle}'
        path = out_dir / f'bounce_compare_{name}{suffix}.parquet'
        tab.to_parquet(path, index=False)
        print(f'wrote {path}')

    # Slope-summary overlay: one slope per axis. The rotator program is a single throw, so its
    # per-leg slope *is* that number; the elevation program has five legs, where the weighted
    # fit across them is the comparable slope and any single leg would be an arbitrary pick.
    slopes = {B.BOUNCE_ANGLE['T724_rotator']:
              B.bounce_slope(stats, bounce='T724_rotator', kind='dof', arm='svd')}
    all_legs, all_fits = B.compare_elevation_legs(stats, frames[next(iter(frames))],
                                                  dof_indices=range(10))
    if len(all_fits):
        slopes[B.BOUNCE_ANGLE['T720_elevation']] = all_fits.rename(
            columns={'slope_linear': 'slope', 'slope_linear_err': 'slope_err'})
    return dict(products, angle=angle, slopes=slopes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--variants', nargs='+', default=None,
                    help='variant ids; default the RBR variant plus both intrinsic routes')
    ap.add_argument('--day-obs-range', nargs=2, type=int, default=None,
                    metavar=('LO', 'HI'), help='inclusive night range YYYYMMDD')
    ap.add_argument('--no-figures', action='store_true', help='write the tables only')
    ap.add_argument('--no-bounce', action='store_true',
                    help='skip the comparison against the measured bounce test')
    ap.add_argument('--bounce-run', default=B.DEFAULT_BOUNCE_RUN,
                    help='bounce run directory under aos/output/bounce')
    ap.add_argument('--bounce-min-significance', type=float, default=B.MIN_SIGNIFICANCE,
                    help='bounce |delta|/error floor for the subspace statistics')
    ap.add_argument('--pdf-name', default='cwfs_lut.pdf', help='PDF filename')
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

    # The survey side of the comparison, in stored units -- nothing converts, since the bounce
    # test stores the same units. `comparable` marks the six pistons and decentres, the terms
    # both retrievals measure alike; the bounce test uses full-focal-plane wavefronts and this
    # uses four corner sensors, which the bending-subspace agreement below quantifies.
    for angle in ANGLES:
        if C.RBR_VARIANT not in frames:
            continue
        print(f'\n=== rigid-body DOF against {angle} ===')
        tab = C.trend_table(frames[C.RBR_VARIANT], angle)
        tab['comparable'] = tab['dof'].isin(C.BOUNCE_COMPARABLE_DOF)
        path = out_dir / f'bounce_comparable_{angle}.parquet'
        tab.to_parquet(path, index=False)
        print(f'wrote {path}')

    bounce = None
    if not args.no_bounce:
        bounce = run_bounce_comparison(frames, out_dir, run=args.bounce_run,
                                       min_significance=args.bounce_min_significance)

    if not args.no_figures:
        print('\n=== figures ===')
        # The headline panels take the strongest rigid-body trend from the data rather than a
        # fixed DOF, so the page stays the headline if the dominant term ever changes.
        primary = C.RBR_VARIANT if C.RBR_VARIANT in frames else variants[0]
        strongest = max(
            ((abs(r['pearson_r']), int(r['dof']), angle)
             for angle in ANGLES if (primary, angle) in tabs
             for _, r in tabs[(primary, angle)].iterrows()
             if int(r['dof']) in CF.RIGID_BODY_DOF and pd.notna(r['pearson_r'])),
            default=(0.0, 1, ANGLES[1]))
        _, dof, angle = strongest
        name, _unit = C.dof_label(dof)
        print(f'headline panels: {name} against {angle}, '
              f'|Pearson r| {strongest[0]:.3f} (dimensionless)')
        pdf_path = CF.write_pdf(out_dir / args.pdf_name, frames, tabs, ANGLES,
                                headline_dof=dof, headline_angle=angle, bounce=bounce)
        print(f'wrote {pdf_path}')


if __name__ == '__main__':
    main()
