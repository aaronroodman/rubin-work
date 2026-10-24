"""Check the standalone calculator against the thermal-focus pipeline it was fitted with.

`trim_calculator` reads its coefficients, rounded to two decimals, from
``trim_coefficients.yaml``. That copy can drift from the fit silently: if the model is
re-fitted and the file is not updated, the maximum difference grows from rounding noise to
something that matters. This refits the deliverable model on the science sample and compares.

The sample is the cached ``thermal_focus.parquet`` the ``thermal_focus`` build writes, with
the analysis-stage cuts (truss temperature, complete features) re-applied. ``--from-product``
rebuilds it from the value-added product instead, one ConsDB round trip per night.

Invocation::

    python check_against_fit.py
    python check_against_fit.py --day-obs-range 20251103 20260713
    python check_against_fit.py --from-product
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

from rubinwork.common.utils import repo_root
from rubinwork.thermal_focus import fit as F
from rubinwork.thermal_focus import lib as L

from test_trim_calculator import load_test_cases
from trim_calculator import TrimCalculator

#: The calculator's five channels, in its argument order.
CALCULATOR_COLS = ['truss_temp_mean_c', 'm1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
                   'm1m3_radial_gradient_c_per_m', 'm1m3_x_gradient_c_per_m']


def load_sample(sci_path=None, day_obs_range=None, from_product=False):
    """Load the science sample the deliverable model is fitted on.

    Parameters
    ----------
    sci_path : `pathlib.Path`, optional
        Cached ``thermal_focus.parquet``; ``thermal_focus/output/`` by default.
    day_obs_range : `tuple` [`int`], optional
        Inclusive ``(lo, hi)`` night range as ``YYYYMMDD``.
    from_product : `bool`, optional
        Rebuild the sample from the value-added product rather than the cache.

    Returns
    -------
    sci : `pandas.DataFrame`
        One row per science visit with every deliverable feature finite.
    features : `list` [`str`]
        The deliverable feature columns, in fitted order.
    """
    features = L.resolve_features(L.DELIVERABLE_GROUPS)
    if from_product:
        from rubinwork.products import value_added
        from rubinwork.thermal_focus import sample as S
        sci = S.load_science(value_added, L.DEFAULT_VARIANT, day_obs_range,
                             L.v1_per_um_dz_value(verbose=False))
    else:
        sci_path = sci_path or repo_root() / 'thermal_focus' / 'output' / 'thermal_focus.parquet'
        sci = pd.read_parquet(sci_path)
        if day_obs_range:
            sci = sci[sci['day_obs'].between(*day_obs_range)]
    # The analysis stage re-applies the truss cut so an older cache gives the same sample.
    sci = sci[~(sci['truss_temp_mean_c'] > L.TRUSS_TEMP_MAX_C)]
    sci = sci[sci[features].notna().all(axis=1)].reset_index(drop=True)
    return sci, features


def compare(sci, features, model='huber', verbose=True):
    """Compare the calculator against the pipeline fitted on ``sci``.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science sample, from `load_sample`.
    features : `list` [`str`]
        Deliverable feature columns, in fitted order.
    model : `str`, optional
        Key for `rubinwork.thermal_focus.fit.make_model`.
    verbose : `bool`, optional
        Print the comparison.

    Returns
    -------
    out : `dict`
        ``max_abs_diff_um`` and ``median_diff_um`` over the sample, ``worked_max_abs_diff_um``
        over the worked cases in ``trim_test_cases.yaml`` [all µm of equivalent hexapod dz],
        ``n`` (visits) and ``n_worked_cases``. Empty if the fitted features are not the
        calculator's five channels.
    """
    if list(features) != CALCULATOR_COLS:
        if verbose:
            print('the fitted feature set is not the calculator\'s five channels; skipped')
        return {}
    full = F.fit_full(sci, features, model=model, verbose=False)
    calculator = TrimCalculator()
    _, pred_um, _ = calculator.predict_trim(*[sci[c].to_numpy(float) for c in CALCULATOR_COLS],
                                            warn_extrapolation=False)
    diff = pred_um - np.asarray(full['pred'], float)
    _, test_cases = load_test_cases()
    worked = max(abs(calculator.predict_trim(**inp, warn_extrapolation=False)[1] - exp)
                 for _, inp, exp in test_cases)
    out = {'n': int(len(sci)),
           'max_abs_diff_um': float(np.nanmax(np.abs(diff))),
           'median_diff_um': float(np.nanmedian(diff)),
           'worked_max_abs_diff_um': float(worked),
           'n_worked_cases': len(test_cases)}
    if verbose:
        print(f'calculator against the fitted pipeline over {out["n"]} visits: '
              f'max |difference| {out["max_abs_diff_um"]:.4f}, median '
              f'{out["median_diff_um"]:+.4f} um of equivalent hexapod dz')
        print(f'its {len(test_cases)} worked cases agree with their stated values to '
              f'{out["worked_max_abs_diff_um"]:.4f} um of equivalent hexapod dz')
        print(f'coefficients from {calculator.path.name}: intercept '
              f'{calculator.intercept_um:+.2f} um, truss {calculator.truss_um_per_c:+.2f} um '
              f'per deg C; fitted intercept {full["intercept"]:+.2f} um')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sci-path', type=pathlib.Path, default=None,
                    help='cached thermal_focus.parquet')
    ap.add_argument('--day-obs-range', nargs=2, type=int, default=None, metavar=('LO', 'HI'))
    ap.add_argument('--from-product', action='store_true',
                    help='rebuild the sample from the value-added product')
    ap.add_argument('--model', default='huber', choices=F.MODEL_NAMES)
    args = ap.parse_args()
    day_obs_range = tuple(args.day_obs_range) if args.day_obs_range else None
    sci, features = load_sample(args.sci_path, day_obs_range, args.from_product)
    compare(sci, features, model=args.model)


if __name__ == '__main__':
    main()
