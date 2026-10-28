"""Build the four ``bounce_tables`` product tables.

Calls `rubinwork.bounce.analysis.run_analysis` for the whole reduction and writes
its four tables.  The nine PDFs are the `bounce` study's half of the same
analysis and stay in ``aos/bounce/code/run_bounce.py``, which calls the same
function; see `rubinwork.products.bounce_tables` for why both halves run it.

Moved out of ``aos/bounce/code/run_bounce.py`` in phase 3a of
``notes/status/organization_plan.md``.  The table-writing code moved verbatim, so
a build reproduces the old script's tables exactly.

Usage, from the repository root (``aos/`` as the working directory, so
``mi_config.yaml`` and ``analysis_config.yaml`` resolve by their relative default
paths)::

    python -m rubinwork.products.bounce_tables.builders.build_bounce_tables \
        --variant d12-A_50_34_i_5rot_july \
        --fits <path>/fits_july.parquet --out-dir <build dir> \
        --min-detectors 160
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lsst.ts.intrinsic.wavefront.ofc_svd import LABELS_50DOF, DOF_UNITS_50

from rubinwork.bounce import analysis as ba
from rubinwork.bounce import legs as bl
from rubinwork.bounce.legs import DOF22
from rubinwork.products import bounce_tables


def resolve_config(variant, **kwargs):
    """The resolved bounce configuration for one product variant.

    Parameters
    ----------
    variant : `str`
        A `bounce_tables` variant name.
    **kwargs
        Passed through to `rubinwork.bounce.legs.resolve_config`:
        `analysis_config`, `rbr_kappa`, `rbr_power`, `no_rbr`.

    Returns
    -------
    cfg : `dict`
        Every knob the analysis reads.

    Notes
    -----
    The variant supplies only the two keys the configuration is looked up by.
    The knobs themselves stay in `analysis_config.yaml`, whose ``bounce:``
    override is keyed on the **long** param_set, so moving them here would be a
    second place to edit.
    """
    param_set, mi_name = bounce_tables.mi_keys(variant)
    return bl.resolve_config(param_set, mi_name, **kwargs)


def write_tables(result, cfg, out_dir):
    """Write the four product tables from one bounce analysis.

    Parameters
    ----------
    result : `rubinwork.bounce.analysis.BounceAnalysis`
        The reduction, from `rubinwork.bounce.analysis.run_analysis`.
    cfg : `dict`
        The resolved bounce configuration, used only for the printed scheme
        labels and for the Range-Bounded Recovery (RBR) settings quoted in them.
    out_dir : `pathlib.Path`
        Build directory to write into.

    Returns
    -------
    written : `list` [`str`]
        File names written, in write order.

    Notes
    -----
    `bounce_fwhm_metric.parquet` and `bounce_fwhm_vs_bvalue.parquet` are written
    only when the correctable-FWHM metric was available (it needs
    ``lsst.ts.wep``), and `bounce_dof_stats.parquet` only when the Optical
    Feedback Control (OFC) singular-value decomposition (SVD) built.  The
    product Snakefile declares all four, so a build missing one fails loudly
    rather than registering short.
    """
    out_dir = Path(out_dir)
    written = []
    fit_table = result.fit_table
    bounce_results = result.bounce_results
    df_kj = result.df_kj
    fwhm_rows = result.fwhm_rows
    fwhm_bvalue_rows = result.fwhm_bvalue_rows
    DOF_all = result.DOF_all
    dof_ranges = result.dof_ranges
    _svd_ok = result.svd_ok

    # ---- differential correctable-FWHM metric ----
    if fwhm_rows:
        fdf = pd.DataFrame(fwhm_rows)
        fdf.to_parquet(out_dir / 'bounce_fwhm_metric.parquet')
        written.append('bounce_fwhm_metric.parquet')
        print('  correctable-FWHM metric [arcsec, median over focal plane]:')
        print('   ' + fdf.to_string(index=False).replace('\n', '\n   '))
        print(f'  wrote bounce_fwhm_metric.parquet ({len(fdf)} rows)')

    # ---- correctable FWHM vs B-set position, per (night, leg) ----
    if fwhm_bvalue_rows:
        bdf = pd.DataFrame(fwhm_bvalue_rows)
        bdf.to_parquet(out_dir / 'bounce_fwhm_vs_bvalue.parquet')
        written.append('bounce_fwhm_vs_bvalue.parquet')
        print(f'  wrote bounce_fwhm_vs_bvalue.parquet ({len(bdf)} rows)')
        # Image-quality cost of each scheme relative to the unregularized 50/34
        # recovery, as the difference of two achieved-residual FWHM in arcsec.
        for _col, _tag in (('fwhm_after_rbr',
                            bl.scheme_label('rbr', cfg)),
                           ('fwhm_after_22_12',
                            f'{len(DOF22)}/{cfg["n_keep_reduced"]} reduced'),
                           ('fwhm_after_oic',
                            f'OIC 50/34 (rho={cfg["oic_rho"]:g} dimensionless)'),
                           ('fwhm_after_5_5', '5/5 camera hexapod')):
            if _col not in bdf.columns:
                continue
            _c = bdf.dropna(subset=['fwhm_after_default', _col])
            if not len(_c):
                continue
            _cost = _c[_col] - _c['fwhm_after_default']
            print(f'  {_tag} FWHM cost over the 50/34 truncated recovery '
                  f'[arcsec]: median={_cost.median():.5f}, '
                  f'min={_cost.min():.5f}, max={_cost.max():.5f} '
                  f'over {len(_c)} (night, leg) points')

    # ---- per-DOF and per-v-mode Δ table ----
    # The DOF Δ was previously visible only inside the PDFs, so any number
    # quoted from it had to be read off a plot.  One row per (bounce, leg,
    # night, quantity); `unit` is µm for translations and bending-mode
    # amplitudes, arcsec for hexapod rotations, and dimensionless for v-modes.
    # This table is what reproduces bounce_dof_night_values.pdf, and is the
    # form shared externally, so it carries the *measured* elevation and
    # rotator angle of both legs rather than the nominal config window.
    if _svd_ok and DOF_all is not None:
        dof_rows = []
        for name, br in bounce_results.items():
            ref_pt_all = bl.leg_pointing(fit_table, br['reference_mask'])
            for clabel, cb in br['comparisons'].items():
                # Pointing per (leg, night), plus the pooled 'all' row.
                pt_cache = {}

                def _pointing(night, _cb=cb, _br=br, _cache=pt_cache,
                              _ref_all=ref_pt_all):
                    if night not in _cache:
                        d = None if night == 'all' else int(night)
                        cp = bl.leg_pointing(fit_table, _cb['comp_mask'], day_obs=d)
                        rp = (_ref_all if night == 'all'
                              else bl.leg_pointing(fit_table, _br['reference_mask'],
                                                   day_obs=d))
                        _cache[night] = (cp, rp)
                    return _cache[night]

                # RBR Δ, keyed the same way, so the `dof` rows can carry it as
                # two extra columns beside the default recovery's Δ rather than
                # as a separate `kind` a reader would have to pivot.
                rbr_blocks = {'all': cb.get('rbr_deltas') or {}}
                rbr_blocks.update({str(int(d)): (v or {}) for d, v
                                   in (cb.get('rbr_deltas_by_night') or {}).items()})

                for kind, pooled, by_night, labels, units in (
                        ('dof', cb.get('dof_deltas'), cb.get('dof_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50),
                        ('vmode', cb.get('vmode_deltas'), cb.get('vmode_deltas_by_night'),
                         None, None),
                        # The 5-DOF / 5-v-mode camera-hexapod-only scheme, present
                        # only on a camera_hexapod_only bounce.  Its DOF indices are
                        # the camera-hexapod entries of the same 50-DOF labelling.
                        ('dof5', cb.get('dof5_deltas'), cb.get('dof5_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50),
                        ('vmode5', cb.get('vmode5_deltas'),
                         cb.get('vmode5_deltas_by_night'), None, None),
                        # The 22-DOF / 12-v-mode reduced set, the scheme the AOS
                        # is expected to operate in.  Its DOF rows are keyed by
                        # global DOF index (DOF22), so they share the 50-DOF
                        # labelling; its v-mode rows are its own 12 v-modes and
                        # are *not* the first 12 of the 50/34 scheme's.
                        ('dof22', cb.get('dof22_deltas'),
                         cb.get('dof22_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50),
                        ('vmode22', cb.get('vmode22_deltas'),
                         cb.get('vmode22_deltas_by_night'), None, None),
                        # The 50-DOF recovery with the ts_ofc OIC quadratic
                        # motion penalty at cfg['oic_rho'].  No v-mode kind:
                        # like RBR it is solved in the retained-mode
                        # coefficients but reported as DOF.
                        ('dof_oic', cb.get('dof_oic_deltas'),
                         cb.get('dof_oic_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50)):
                    blocks = {'all': pooled} if pooled else {}
                    blocks.update({str(int(d)): v for d, v in (by_night or {}).items()})
                    for night, block in blocks.items():
                        cp, rp = _pointing(night)
                        # Only the 50-DOF `dof` kind has an RBR counterpart:
                        # RBR is a 50-DOF recovery, and v-modes are the step it
                        # replaces rather than a quantity it produces.
                        rb = rbr_blocks.get(night, {}) if kind == 'dof' else {}
                        for q, v in (block or {}).items():
                            rv = rb.get(int(q)) or {}
                            dof_rows.append({
                                'bounce': name, 'comparison': clabel,
                                'reference': br['reference_label'],
                                'block': cp['block'] or rp['block'],
                                'night': night,
                                'day_obs': (pd.NA if night == 'all' else int(night)),
                                'elevation_deg': cp['elevation_deg'],
                                'rot_angle_deg': cp['rot_angle_deg'],
                                'ref_elevation_deg': rp['elevation_deg'],
                                'ref_rot_angle_deg': rp['rot_angle_deg'],
                                'n_visits': cp['n_visits'],
                                'n_pairs': (len(cb.get('pairs', [])) if night == 'all'
                                            else cb.get('n_pairs_by_night', {}).get(
                                                night, 0)),
                                'kind': kind, 'index': int(q),
                                'label': (labels[int(q)] if labels is not None
                                          else f'v{int(q) + 1}'),
                                'unit': (units[int(q)] if units is not None
                                         else 'dimensionless'),
                                'delta': float(v.get('delta', np.nan)),
                                'delta_err': float(v.get('err', np.nan)),
                                # Range-Bounded Recovery Δ for the same DOF,
                                # same unit and same median / median-SEM error
                                # definition; NaN for the v-mode kinds.
                                'delta_rbr': float(rv.get('delta', np.nan)),
                                'delta_rbr_err': float(rv.get('err', np.nan)),
                            })
        if dof_rows:
            ddf = pd.DataFrame(dof_rows)
            ddf['significance'] = (ddf['delta'].abs() / ddf['delta_err']).replace(
                [np.inf, -np.inf], np.nan)
            # Allowed range and the two amplitude-over-range ratios, so a reader
            # can see which recovery exceeds what the telescope can apply
            # without having to re-derive r_j from the normalization weights.
            if dof_ranges is not None:
                is_dof = ddf['kind'].isin(['dof', 'dof5', 'dof22', 'dof_oic'])
                rj = ddf['index'].map(
                    lambda i: (float(dof_ranges[int(i)])
                               if 0 <= int(i) < len(dof_ranges) else np.nan))
                ddf['dof_range'] = rj.where(is_dof)
                ddf['ratio_to_range'] = (ddf['delta'].abs() / ddf['dof_range']
                                         ).replace([np.inf, -np.inf], np.nan)
                ddf['ratio_to_range_rbr'] = (
                    ddf['delta_rbr'].abs() / ddf['dof_range']
                ).replace([np.inf, -np.inf], np.nan)
            ddf.to_parquet(out_dir / 'bounce_dof_stats.parquet')
            print(f'  wrote bounce_dof_stats.parquet ({len(ddf)} rows)')
            if 'ratio_to_range' in ddf.columns:
                # One line per recovery scheme: how many of its DOF rows ask for
                # more than the telescope can apply.  Per-night and leg-pooled
                # rows are split because they describe the same physics and a
                # single total would double-count it.  The `dof` kind also
                # carries RBR as columns, so RBR is counted from
                # `ratio_to_range_rbr` on those rows.
                print('    DOF rows over the allowed range (|Δ|/r_j > 1, '
                      'dimensionless), per scheme:')
                _schemes = [('50/34 truncated', 'dof', 'ratio_to_range'),
                            (bl.scheme_label('rbr', cfg),
                             'dof', 'ratio_to_range_rbr'),
                            (f'{len(DOF22)}/{cfg["n_keep_reduced"]} reduced',
                             'dof22', 'ratio_to_range'),
                            (f'OIC 50/34 (rho={cfg["oic_rho"]:g} '
                             f'dimensionless)',
                             'dof_oic', 'ratio_to_range'),
                            ('5/5 camera hexapod', 'dof5', 'ratio_to_range')]
                for _tag, _kind, _col in _schemes:
                    if _col not in ddf.columns:
                        continue
                    _d = ddf[(ddf['kind'] == _kind) & ddf[_col].notna()]
                    if not len(_d):
                        continue
                    _parts = []
                    for _lab, _s in (
                            ('per (leg, night)',
                             _d[_d['night'].astype(str) != 'all']),
                            ('leg-pooled',
                             _d[_d['night'].astype(str) == 'all'])):
                        if len(_s):
                            _parts.append(f'{_lab} {int((_s[_col] > 1).sum())} '
                                          f'of {len(_s)}')
                    print(f'      {_tag:42s} ' + '; '.join(_parts))
        if dof_rows:
            written.append('bounce_dof_stats.parquet')

    # ---- long-format table (cell 28) ----
    df_kj.to_parquet(out_dir / 'bounce_kj_stats.parquet')
    written.append('bounce_kj_stats.parquet')
    print(f'  wrote bounce_kj_stats.parquet ({len(df_kj)} rows)')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', required=True,
                    help='bounce_tables variant name (see variants.yaml)')
    ap.add_argument('--fits', required=True,
                    help='the per-visit Double-Zernike fit table to read')
    ap.add_argument('--out-dir', required=True,
                    help='build directory to write the four tables into')
    ap.add_argument('--analysis-config', default=None,
                    help='override the aos/analysis_config.yaml path')
    ap.add_argument('--min-detectors', type=int, default=None,
                    help='Per-visit quality cut: keep visits with '
                         'n_detectors_with_min_donuts >= this (relaxes ONLY the '
                         'CCD-count cut; blur cut kept). Default: None -> use '
                         'the precomputed visit_quality_pass (nd>=170).')
    ap.add_argument('--rbr-kappa', type=float, default=None,
                    help='Range-Bounded Recovery: ratio |d_j|/r_j '
                         '(dimensionless, recovered amplitude over allowed '
                         'range) at which the penalty reaches unit weight. '
                         'Default from config (4.0).')
    ap.add_argument('--rbr-power', type=int, default=None,
                    help='Range-Bounded Recovery: penalty exponent; the penalty '
                         'goes as the ratio to the 2*power. Default from '
                         'config (3).')
    ap.add_argument('--no-rbr', action='store_true',
                    help='Skip Range-Bounded Recovery and report only the '
                         'default truncated recovery.')
    args = ap.parse_args()

    cfg = resolve_config(args.variant, analysis_config=args.analysis_config,
                         rbr_kappa=args.rbr_kappa, rbr_power=args.rbr_power,
                         no_rbr=args.no_rbr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ba.run_analysis(args.fits, cfg, min_detectors=args.min_detectors)
    written = write_tables(result, cfg, out_dir)
    print(f'[bounce_tables] wrote {len(written)} tables into {out_dir}')
    print('[bounce_tables] done.')


if __name__ == '__main__':
    main()
