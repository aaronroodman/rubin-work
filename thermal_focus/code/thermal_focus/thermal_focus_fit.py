"""Model fitting for the thermal-focus study: night-grouped evaluation and the physical equation.

The one rule this module exists to enforce is that a focus model is scored with **whole nights
held out**. Only 1.9% of the Telescope Mount Assembly (TMA) truss temperature's variance is
within-night while 83.8% of the response's variance is between nights, so a visit-level split
lets a model memorise the night it is predicting. Measured optimism against a night-grouped
split is a factor of 3.1 (dimensionless) for boosted trees and about 1.04 for the Huber linear
fit, which has too few coefficients to memorise a night.

Fits are Huber robust linear (`sklearn.linear_model.HuberRegressor`) by default. That is not a
default of convenience: the response has a one-sided positive tail in every band, with 2-3% of
visits beyond +3 normalized median absolute deviations (nMAD) against 0.3% beyond −3, versus a
Gaussian's 0.135% on each side, so least squares would be pulled by the tail.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

import thermal_focus_lib as L                                    # noqa: E402
from common.utils import nmad                                    # noqa: E402

#: Models available to the comparison table. ``huber`` is the deliverable.
MODEL_NAMES = ('huber', 'ridge', 'histgb', 'rf')

#: Folds in the night-grouped cross-validation.
N_SPLITS = 5

#: Measured optimism of a visit-level split over a night-grouped one for a high-capacity model
#: [dimensionless, visit-level residual nMAD over night-grouped]. Quoted in the warning that
#: accompanies any leaky score.
LEAK_FACTOR = 3.1

#: Units of each feature, for reading a coefficient.
FEATURE_UNITS = {
    'truss_temp_mean_c': 'deg C',
    'cam_AverageTemp': 'deg C',
    'cam_AmbAirtemp': 'deg C',
    'm1m3_z_gradient_c_per_m': 'deg C per m',
    'm1m3_y_gradient_c_per_m': 'deg C per m',
    'm1m3_x_gradient_c_per_m': 'deg C per m',
    'm1m3_radial_gradient_c_per_m': 'deg C per m',
    'wind_speed_ms': 'm per s',
    'into_wind_deg': 'deg',
    'altitude_deg': 'deg',
    'cum_hex_dz_um': 'um',
    'recent_hex_dz_um': 'um',
    'n_moves_night': 'moves',
}

#: Commanded truss slope from the Full Array Mode (FAM) Double Zernike fits [dimensionless
#: v-mode-1 amplitude per deg C]. This is the slope of v-mode 1 **of the commanded Trim** against
#: truss temperature, so the like-for-like science-image comparison is `commanded_truss_slope`,
#: not the fitted coefficient of the response. The response is Trim minus the measured state, a
#: different quantity, and its coefficient lands about 16% away — which is a statement about the
#: measured term, not a disagreement between the two engines.
FAM_TRUSS_SLOPE = 0.09634


def make_model(name):
    """Build a named sklearn pipeline.

    Parameters
    ----------
    name : `str`
        One of `MODEL_NAMES`.

    Returns
    -------
    model : `sklearn.base.BaseEstimator`
        Unfitted pipeline. Every pipeline imputes missing features at the training-fold median,
        so a fold never sees a NaN and the imputation is fitted inside the fold rather than on
        the whole sample.

    Raises
    ------
    ValueError
        If `name` is not in `MODEL_NAMES`.

    Notes
    -----
    ``huber`` is the recommendation: measured best under a night-grouped split, and its
    coefficients read directly in µm of equivalent hexapod dz per feature unit. The tree models
    are kept for the comparison table, where they lose — they fit night-specific structure that
    does not transfer across held-out nights.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy='median')
    if name == 'huber':
        return make_pipeline(imp, StandardScaler(), HuberRegressor(max_iter=2000))
    if name == 'ridge':
        return make_pipeline(imp, StandardScaler(), RidgeCV())
    if name == 'histgb':
        return HistGradientBoostingRegressor(max_iter=200, max_depth=2, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=0)
    if name == 'rf':
        return make_pipeline(imp, RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=0))
    raise ValueError(f'unknown model {name!r}; choose from {", ".join(MODEL_NAMES)}')


def _linear_coefficients(fitted, features):
    """Slopes of a fitted linear pipeline in physical units, or None.

    Parameters
    ----------
    fitted : `sklearn.pipeline.Pipeline`
        A fitted pipeline.
    features : `list` [`str`]
        The fitted column order.

    Returns
    -------
    coef : `numpy.ndarray` or `None`
        One coefficient per feature [µm of equivalent hexapod dz per feature unit], or None when
        the final estimator is not linear.

    Notes
    -----
    Only the slopes are un-scaled here. The intercept needs an additional
    ``- coef . scaler.mean_`` shift, which `_physical_intercept` applies.
    """
    try:
        steps = list(fitted.named_steps.values())
    except AttributeError:
        return None
    est = steps[-1]
    if not hasattr(est, 'coef_') or len(getattr(est, 'coef_', [])) != len(features):
        return None
    scaler = next((s for s in steps if hasattr(s, 'scale_')), None)
    coef = np.asarray(est.coef_, float)
    if scaler is not None and getattr(scaler, 'scale_', None) is not None:
        coef = coef / scaler.scale_
    return coef


def _physical_intercept(fitted, coef_physical):
    """Intercept of a fitted linear pipeline in physical units.

    Parameters
    ----------
    fitted : `sklearn.pipeline.Pipeline`
        A fitted pipeline.
    coef_physical : `numpy.ndarray` or `None`
        Slopes already divided by ``scaler.scale_``, from `_linear_coefficients`.

    Returns
    -------
    intercept : `float`
        Response at zero in every feature [µm of equivalent hexapod dz], or NaN.

    Notes
    -----
    `sklearn.preprocessing.StandardScaler` centres as well as scales, so the estimator's own
    ``intercept_`` is the response at the training-fold **mean**. The slopes need only
    ``/ scaler.scale_``, but the intercept additionally needs ``- coef . scaler.mean_`` —
    without it the whole curve is offset by the model's prediction at the feature means, about
    1500 µm of equivalent hexapod dz on this sample.
    """
    try:
        steps = list(fitted.named_steps.values())
    except AttributeError:
        return float('nan')
    est = steps[-1]
    b = getattr(est, 'intercept_', None)
    if b is None:
        return float('nan')
    b = float(np.asarray(b).ravel()[0]) if np.ndim(b) else float(b)
    scaler = next((s for s in steps if hasattr(s, 'mean_')), None)
    if scaler is not None and coef_physical is not None \
            and getattr(scaler, 'mean_', None) is not None:
        b = b - float(np.dot(np.asarray(coef_physical, float), np.asarray(scaler.mean_, float)))
    return b


def verify_equation(fitted, df, features, intercept, coef, tol=1e-6):
    """Check the physical equation reproduces the pipeline's own prediction.

    Parameters
    ----------
    fitted : `sklearn.pipeline.Pipeline`
        The fitted pipeline.
    df : `pandas.DataFrame`
        The fitted sample.
    features : `list` [`str`]
        The fitted column order.
    intercept : `float`
        From `_physical_intercept` [µm of equivalent hexapod dz].
    coef : `numpy.ndarray`
        From `_linear_coefficients`.
    tol : `float`, optional
        Tolerance [µm of equivalent hexapod dz].

    Returns
    -------
    max_abs_diff : `float`
        Largest disagreement over the sample [µm of equivalent hexapod dz].

    Raises
    ------
    AssertionError
        If the equation and the pipeline disagree by more than `tol`.

    Notes
    -----
    This is what stops a written-out equation from silently disagreeing with the model it
    claims to describe — the failure mode that makes a published coefficient unusable. Rows with
    a missing feature are skipped, since the pipeline imputes them and the bare equation cannot.
    """
    X = df[features].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    if not ok.any():
        return float('nan')
    by_hand = intercept + X[ok] @ np.asarray(coef, float)
    by_pipe = fitted.predict(X[ok])
    max_abs_diff = float(np.max(np.abs(by_hand - by_pipe)))
    assert max_abs_diff < tol, (
        f'the physical equation disagrees with the fitted pipeline by {max_abs_diff:.3e} um of '
        f'equivalent hexapod dz, over the tolerance {tol:.0e}')
    return max_abs_diff


def evaluate(df, features, model='huber', n_splits=N_SPLITS, leaky=False, verbose=True):
    """Out-of-fold prediction with whole nights held out.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``y``, ``day_obs`` and the feature columns.
    features : `list` [`str`]
        Feature column names.
    model : `str`, optional
        Key for `make_model`.
    n_splits : `int`, optional
        Cross-validation folds.
    leaky : `bool`, optional
        Split on visits instead of nights. Only for demonstrating the leak; the returned numbers
        are optimistic and must not be quoted as performance.
    verbose : `bool`, optional
        Print the score.

    Returns
    -------
    res : `dict`
        ``pred`` (out-of-fold prediction [µm of equivalent hexapod dz]), ``resid``, ``coefs``
        (per-fold coefficient array, or None for the tree models), ``nmad``, ``r2``,
        ``n_splits`` and ``grouped``.

    Notes
    -----
    `sklearn.model_selection.GroupKFold` on ``day_obs`` is what makes the score meaningful; see
    the module docstring for the measured size of the visit-level leak.
    """
    from sklearn.model_selection import GroupKFold, KFold

    X = df[features].to_numpy(float)
    y = df['y'].to_numpy(float)
    groups = df['day_obs'].to_numpy()

    if leaky:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y)
    else:
        splits = GroupKFold(n_splits=n_splits).split(X, y, groups=groups)

    pred = np.full(len(y), np.nan)
    coefs = []
    for train, test in splits:
        m = make_model(model)
        m.fit(X[train], y[train])
        pred[test] = m.predict(X[test])
        c = _linear_coefficients(m, features)
        if c is not None:
            coefs.append(c)
    resid = y - pred
    res = dict(pred=pred, resid=resid, coefs=np.array(coefs) if coefs else None,
               nmad=nmad(resid), r2=1.0 - np.nanvar(resid) / np.nanvar(y),
               n_splits=n_splits, grouped=not leaky)
    if verbose:
        kind = 'visit-level (LEAKY)' if leaky else 'night-grouped on day_obs'
        print(f'{model}, {n_splits}-fold {kind}: residual nMAD {res["nmad"]:.1f} um of '
              f'equivalent hexapod dz, R2 {res["r2"]:.3f} (dimensionless)')
        if leaky:
            print(f'  WARNING: whole nights were NOT held out, so this is not a performance '
                  f'estimate. Optimism is up to a factor of {LEAK_FACTOR} (dimensionless) for '
                  f'boosted trees, about 1.04 for a Huber linear fit.')
    return res


def fit_full(df, features, model='huber', verbose=True):
    """Fit on every night — the deliverable model.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``y`` and the feature columns.
    features : `list` [`str`]
        Feature column names.
    model : `str`, optional
        Key for `make_model`.
    verbose : `bool`, optional
        Print the coefficients.

    Returns
    -------
    res : `dict`
        ``model``, ``pred``, ``resid``, ``coef``, ``nmad``, ``intercept`` [µm of equivalent
        hexapod dz], ``features`` and ``equation_max_abs_diff``.

    Notes
    -----
    The in-sample nMAD of this fit is **not** a performance estimate — use `evaluate`. It is
    reported only so the gap from the out-of-fold number shows how much the fit depends on which
    nights it saw.
    """
    X = df[features].to_numpy(float)
    y = df['y'].to_numpy(float)
    m = make_model(model)
    m.fit(X, y)
    pred = m.predict(X)
    resid = y - pred
    coef = _linear_coefficients(m, features)
    intercept = _physical_intercept(m, coef)
    res = dict(model=m, pred=pred, resid=resid, coef=coef, nmad=nmad(resid),
               intercept=intercept, features=list(features),
               equation_max_abs_diff=float('nan'))
    if coef is not None and np.isfinite(intercept):
        res['equation_max_abs_diff'] = verify_equation(m, df, features, intercept, coef)
    if verbose:
        print(f'full fit ({model}, all {df["day_obs"].nunique()} nights): in-sample residual '
              f'nMAD {res["nmad"]:.1f} um of equivalent hexapod dz (not a performance estimate)')
        if coef is not None:
            print(f'  intercept {res["intercept"]:+.2f} um of equivalent hexapod dz '
                  f'(physical, response at zero in every feature)')
            for f, c in zip(features, coef):
                print(f'  {f:32s} {c:+10.2f} um per {FEATURE_UNITS.get(f, "?")}')
            print(f'  equation reproduces pipeline.predict to '
                  f'{res["equation_max_abs_diff"]:.2e} um (tolerance 1e-06)')
    return res


def coefficient_table(coefs, features, v1_per_um_dz, verbose=True):
    """Per-fold coefficient mean and scatter, with the FAM truss cross-check.

    Parameters
    ----------
    coefs : `numpy.ndarray`
        Shape ``(n_folds, n_features)`` [µm of equivalent hexapod dz per feature unit].
    features : `list` [`str`]
        Feature column names, in the fitted order.
    v1_per_um_dz : `float`
        Converts the truss coefficient into a dimensionless v-mode-1 amplitude per deg C for
        comparison with `FAM_TRUSS_SLOPE`.
    verbose : `bool`, optional
        Print the table.

    Returns
    -------
    out : `pandas.DataFrame`
        Per feature: ``feature``, ``unit``, ``mean``, ``std``, ``n_folds``, ``sign_stable``.

    Notes
    -----
    A coefficient whose sign flips between folds is flagged: it means the feature is not
    carrying a reproducible physical effect, whatever its mean says.
    """
    rows = []
    for i, f in enumerate(features):
        c = coefs[:, i]
        rows.append(dict(feature=f,
                         unit=f'um equiv hexapod dz per {FEATURE_UNITS.get(f, "?")}',
                         mean=float(c.mean()), std=float(c.std()), n_folds=len(c),
                         sign_stable=bool(len(set(np.sign(c))) == 1)))
    out = pd.DataFrame(rows)
    if verbose:
        print(f'coefficients across {coefs.shape[0]} folds '
              f'[um of equivalent hexapod dz per feature unit]')
        for _, r in out.iterrows():
            flag = '' if r.sign_stable else '   <-- SIGN FLIPS across folds'
            print(f'  {r.feature:32s} {r["mean"]:+10.2f} +/- {r["std"]:8.2f} '
                  f'per {FEATURE_UNITS.get(r.feature, "?"):12s}{flag}')
        if 'truss_temp_mean_c' in features:
            i = features.index('truss_temp_mean_c')
            v1_per_c = coefs[:, i].mean() * v1_per_um_dz
            print(f'  the response truss term as a v-mode-1 slope: {v1_per_c:+.5f} '
                  f'dimensionless v-mode-1 amplitude per deg C')
            print(f'    not directly comparable to the FAM commanded slope '
                  f'{FAM_TRUSS_SLOPE:+.5f} per deg C: see commanded_truss_slope')
    return out


def commanded_truss_slope(df, v1_per_um_dz, verbose=True):
    """The commanded truss slope, the like-for-like comparison against the FAM value.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``v1_trim`` [dimensionless v-mode-1 amplitude of the commanded Trim],
        ``truss_temp_mean_c`` [°C] and ``band``.
    v1_per_um_dz : `float`
        Conversion [dimensionless v-mode-1 amplitude per µm of total hexapod dz travel]. Used
        only to report the slope in both units.
    verbose : `bool`, optional
        Print the per-band comparison.

    Returns
    -------
    out : `pandas.DataFrame`
        Per band and pooled: ``band``, ``n``, ``slope`` [dimensionless v-mode-1 amplitude per
        °C], ``slope_err``, ``difference`` from `FAM_TRUSS_SLOPE`, ``n_sigma`` and
        ``slope_um_per_c`` [µm of equivalent hexapod dz per °C].

    Notes
    -----
    `FAM_TRUSS_SLOPE` is the slope of v-mode 1 **of the commanded Trim** against truss
    temperature, measured from the FAM Double Zernike fits. So the science-image quantity that
    tests it is ``v1_trim`` against the same temperature — not the fitted coefficient of the
    response, which is Trim minus the measured state and therefore a different quantity. The
    two agree here to well inside the 0.03 per °C tolerance the original study set, which is
    what says the two independent engines see the same commanded thermal response.
    """
    rows = []
    for band, g in list(df.groupby('band')) + [('all', df)]:
        if len(g) < 200:
            continue
        r = huber_line(g['truss_temp_mean_c'], g['v1_trim'])
        if not np.isfinite(r['slope']):
            continue
        d = r['slope'] - FAM_TRUSS_SLOPE
        rows.append(dict(band=band, n=r['n'], slope=r['slope'], slope_err=r['slope_err'],
                         pearson_r=r['pearson_r'], spearman_rho=r['spearman_rho'],
                         difference=d,
                         n_sigma=abs(d) / r['slope_err'] if r['slope_err'] else float('nan'),
                         slope_um_per_c=r['slope'] / v1_per_um_dz))
    out = pd.DataFrame(rows)
    if verbose:
        print(f'commanded truss slope, v1 of the Trim against truss temperature, against the '
              f'FAM value {FAM_TRUSS_SLOPE:+.5f}')
        print('  [dimensionless v-mode-1 amplitude per deg C]')
        for _, r in out.iterrows():
            print(f'  {r.band:>4s}  n {int(r.n):6d}  {r.slope:+.5f} +/- {r.slope_err:.5f}  '
                  f'difference {r.difference:+.5f} ({r.n_sigma:.1f} standard errors)  '
                  f'= {r.slope_um_per_c:+.1f} um of equivalent hexapod dz per deg C')
        pooled = out[out.band == 'all']
        per_band = out[out.band != 'all']
        if len(pooled):
            d = float(pooled.difference.iloc[0])
            print(f'  pooled difference {d:+.5f} per deg C: '
                  f'{"consistent with FAM" if abs(d) < 0.03 else "DEVIATES from FAM"} '
                  f'(tolerance 0.03 per deg C) -- this is the number the study reports')
        if len(per_band):
            worst = per_band.loc[per_band.difference.abs().idxmax()]
            print(f'  per-band slopes span {per_band.slope.min():+.5f} to '
                  f'{per_band.slope.max():+.5f} per deg C; furthest is {worst.band} at '
                  f'{worst.difference:+.5f}')
            print(f'    the per-band spread is far larger than any formal error, so a single '
                  f'band is not an\n    independent measurement of this slope: the bands differ '
                  f'in sample size and in how well\n    the focus is determined, and the '
                  f'pooled fit is what averages that out')
    return out


def model_comparison(df, features, models=MODEL_NAMES, n_splits=N_SPLITS, verbose=True):
    """Night-grouped score for each model, plus the uncorrected baseline.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``y``, ``day_obs`` and the feature columns.
    features : `list` [`str`]
        Feature column names.
    models : `iterable` [`str`], optional
        Keys for `make_model`.
    n_splits : `int`, optional
        Cross-validation folds.
    verbose : `bool`, optional
        Print the table.

    Returns
    -------
    out : `pandas.DataFrame`
        Per model: ``model``, ``resid_nmad`` [µm of equivalent hexapod dz], ``r2``
        (dimensionless) and ``improvement`` (dimensionless, uncorrected nMAD over residual
        nMAD).

    Notes
    -----
    The uncorrected row is the scatter of the response itself, which is what any model has to
    beat to be worth running.
    """
    base = nmad(df['y'].to_numpy(float))
    rows = [dict(model='uncorrected', resid_nmad=base, r2=0.0, improvement=1.0)]
    for name in models:
        r = evaluate(df, features, model=name, n_splits=n_splits, verbose=False)
        rows.append(dict(model=name, resid_nmad=r['nmad'], r2=r['r2'],
                         improvement=base / r['nmad'] if r['nmad'] else float('nan')))
    out = pd.DataFrame(rows)
    if verbose:
        print(f'night-grouped model comparison, {n_splits} folds, {len(features)} features, '
              f'n {len(df)} visits over {df["day_obs"].nunique()} nights')
        for _, r in out.iterrows():
            print(f'  {r.model:12s} residual nMAD {r.resid_nmad:7.1f} um of equivalent hexapod '
                  f'dz   R2 {r.r2:+.3f}   improvement {r.improvement:.2f}x (dimensionless)')
    return out


def split_comparison(df, features, model='huber', n_splits=N_SPLITS, verbose=True):
    """The night-grouped score beside the visit-level one, for the same model.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``y``, ``day_obs`` and the feature columns.
    features : `list` [`str`]
        Feature column names.
    model : `str`, optional
        Key for `make_model`.
    n_splits : `int`, optional
        Cross-validation folds.
    verbose : `bool`, optional
        Print the comparison.

    Returns
    -------
    out : `dict`
        ``grouped_nmad``, ``leaky_nmad`` [µm of equivalent hexapod dz] and ``optimism``
        (dimensionless, night-grouped nMAD over visit-level nMAD).

    Notes
    -----
    Makes the leak legible rather than asserted: the same model, the same features, one number
    that is a performance estimate and one that is not.
    """
    g = evaluate(df, features, model=model, n_splits=n_splits, verbose=False)
    k = evaluate(df, features, model=model, n_splits=n_splits, leaky=True, verbose=False)
    out = dict(grouped_nmad=g['nmad'], leaky_nmad=k['nmad'],
               optimism=g['nmad'] / k['nmad'] if k['nmad'] else float('nan'))
    if verbose:
        print(f'{model}: night-grouped residual nMAD {out["grouped_nmad"]:.1f} um of equivalent '
              f'hexapod dz vs visit-level {out["leaky_nmad"]:.1f} um, '
              f'optimism {out["optimism"]:.2f}x (dimensionless, grouped over visit-level)')
    return out


def residual_tail(df, resid, n_sigma=3.0, verbose=True):
    """Asymmetry of the residual tails, per band.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``band``.
    resid : `array_like`
        Out-of-fold residual [µm of equivalent hexapod dz].
    n_sigma : `float`, optional
        Tail threshold in units of the residual nMAD.
    verbose : `bool`, optional
        Print the table.

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``band``, ``n``, ``resid_nmad``, ``frac_above``, ``frac_below`` (per cent) and
        ``ratio`` (dimensionless, above over below).

    Notes
    -----
    A Gaussian puts 0.135% beyond 3 sigma on each side, so a ratio far from 1 is the signal.
    Which residual is passed in decides what the answer means: about a **truss-only per-band**
    fit the tail is strongly one-sided positive, with a ratio of 5 to 25 (dimensionless) in the
    same direction in every band — a population of visits whose commanded focus sat above the
    truss relation. About the **full five-feature** fit that asymmetry is largely absorbed, and
    what remains is heavy on both sides. Both are reasons to fit robustly; only the first is a
    statement about the truss relation itself.
    """
    r = np.asarray(resid, float)
    rows = []
    for band, idx in df.groupby('band').groups.items():
        rb = r[df.index.get_indexer(idx)]
        rb = rb[np.isfinite(rb)]
        if len(rb) < 10:
            continue
        s = nmad(rb)
        above = 100.0 * np.mean(rb > n_sigma * s)
        below = 100.0 * np.mean(rb < -n_sigma * s)
        rows.append(dict(band=band, n=len(rb), resid_nmad=s, frac_above=above,
                         frac_below=below,
                         ratio=above / below if below else float('inf')))
    out = pd.DataFrame(rows)
    if verbose:
        print(f'residual tails beyond +/-{n_sigma:.0f} nMAD, per band '
              f'(a Gaussian gives 0.135% each side)')
        for _, x in out.iterrows():
            print(f'  {x.band}  n {int(x.n):6d}  nMAD {x.resid_nmad:6.1f} um  '
                  f'above {x.frac_above:5.2f}%  below {x.frac_below:5.2f}%  '
                  f'ratio {x.ratio:5.1f} (dimensionless, above over below)')
    return out


def huber_line(x, y):
    """Huber robust straight-line fit with Pearson and Spearman statistics.

    Parameters
    ----------
    x, y : `array_like`
        Paired values in any units; the slope carries ``y`` units per ``x`` unit.

    Returns
    -------
    res : `dict`
        ``n``, ``slope``, ``slope_err``, ``intercept``, ``pearson_r``, ``spearman_rho`` and
        ``resid_nmad``, or NaNs when fewer than three finite pairs remain.

    Notes
    -----
    Both correlation statistics are reported because the repository's convention requires it:
    Pearson answers how linear the relation is, Spearman how monotonic, and they disagree
    exactly where a tail is doing the work.
    """
    import statsmodels.api as sm
    from scipy import stats

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()), slope=float('nan'), slope_err=float('nan'),
                    intercept=float('nan'), pearson_r=float('nan'),
                    spearman_rho=float('nan'), resid_nmad=float('nan'))
    x, y = x[ok], y[ok]
    X = sm.add_constant(x)
    rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    pred = rlm.params[0] + rlm.params[1] * x
    return dict(n=int(len(x)), slope=float(rlm.params[1]),
                slope_err=float(rlm.bse[1]), intercept=float(rlm.params[0]),
                pearson_r=float(stats.pearsonr(x, y)[0]),
                spearman_rho=float(stats.spearmanr(x, y)[0]),
                resid_nmad=float(nmad(y - pred)))


# ----------------------------------------------------------------- elevation and band changes

#: A per-night slope below this many visits is not worth fitting.
MIN_VISITS_NIGHT = 40

#: Likewise for one direction of one night.
MIN_VISITS_LEG = 25

#: Visits in the centred rolling median of elevation used to label slew direction.
DIRECTION_WINDOW = 21

#: Minimum |d(elevation)| per visit to count as slewing [deg per visit]. Smaller excursions are
#: tracking, not a slew.
DIRECTION_DEADBAND = 0.02

#: Elevation at which a night's offset is read, inside the observed range [deg].
REF_ELEV_DEG = 60.0


def huber_slope(x, y, min_n=MIN_VISITS_LEG, min_span=5.0):
    """Robust straight-line fit against elevation, with the slope's standard error.

    Parameters
    ----------
    x, y : `array_like`
        Elevation [deg] and response [µm of equivalent hexapod dz].
    min_n : `int`, optional
        Return None below this many finite pairs.
    min_span : `float`, optional
        Return None if elevation spans less than this [deg].

    Returns
    -------
    out : `dict` or `None`
        ``n``, ``slope`` and ``slope_err`` [µm of equivalent hexapod dz per deg],
        ``intercept`` [µm of equivalent hexapod dz], ``pearson_r``, ``spearman_rho``,
        ``resid_nmad``, ``elev_min`` and ``elev_max`` [deg]. None if under-determined.

    Notes
    -----
    The `min_span` guard matters: over a few deg of elevation a fitted slope is an
    extrapolation dressed as a measurement, and its formal error does not say so.
    """
    import statsmodels.api as sm
    from scipy import stats

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return None
    x, y = x[ok], y[ok]
    if np.ptp(x) < min_span:
        return None
    X = sm.add_constant(x)
    try:
        res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    except Exception:
        return None
    resid = y - res.predict(X)
    return dict(n=int(ok.sum()), slope=float(res.params[1]), slope_err=float(res.bse[1]),
                intercept=float(res.params[0]),
                pearson_r=float(stats.pearsonr(x, y)[0]),
                spearman_rho=float(stats.spearmanr(x, y)[0]),
                resid_nmad=float(nmad(resid)),
                elev_min=float(x.min()), elev_max=float(x.max()))


def label_direction(df, window=DIRECTION_WINDOW, deadband=DIRECTION_DEADBAND):
    """Label each visit of one night as taken on a rising or falling elevation leg.

    Parameters
    ----------
    df : `pandas.DataFrame`
        One night, any order; needs ``obs_start_mjd`` and ``altitude_deg`` [deg].
    window : `int`, optional
        Visits in the centred rolling median of elevation.
    deadband : `float`, optional
        Minimum |d(elevation)| per visit to count as slewing [deg per visit].

    Returns
    -------
    direction : `pandas.Series`
        ``'up'``, ``'down'`` or ``'flat'``, indexed like `df`.

    Notes
    -----
    The rolling median is what makes the label meaningful rather than noise. Consecutive visits
    are about 0.7 min apart with sub-2 deg steps whose raw sign alternates while tracking, so
    the sign of a per-visit elevation difference reports hundreds of direction changes per night
    instead of the few tens of real elevation legs.
    """
    d = df.sort_values('obs_start_mjd')
    sm_el = d['altitude_deg'].rolling(window, center=True, min_periods=3).median()
    de = sm_el.diff()
    out = pd.Series('flat', index=d.index, dtype=object)
    out[de > deadband] = 'up'
    out[de < -deadband] = 'down'
    return out.reindex(df.index)


def per_night_elevation(df, ycol='resid', ref_elev_deg=REF_ELEV_DEG, verbose=True):
    """Per-night elevation slopes, all points and each leg, with the hysteresis test.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit table with ``day_obs``, ``altitude_deg`` [deg], ``direction`` and `ycol`.
    ycol : `str`, optional
        Column to fit [µm of equivalent hexapod dz].
    ref_elev_deg : `float`, optional
        Elevation at which each night's offset is read [deg].
    verbose : `bool`, optional
        Print the summary.

    Returns
    -------
    out : `pandas.DataFrame`
        One row per fitted night: ``slope_*``, ``err_*`` [µm of equivalent hexapod dz per deg],
        ``offset_*`` at `ref_elev_deg`, ``difference`` (rising minus falling) and
        ``difference_sigma`` in units of the combined standard error.

    Notes
    -----
    ``offset_*`` rather than ``intercept_*`` is the quantity to compare night to night: the 0
    deg intercept lies about 60 deg outside the observed elevation range, so its scatter is
    dominated by the slope error propagated over that lever arm rather than by any real offset.
    """
    from scipy import stats

    rows = []
    for day, d in df.groupby('day_obs'):
        if len(d) < MIN_VISITS_NIGHT:
            continue
        fits = {}
        for key, sub, min_n in (('all', d, MIN_VISITS_NIGHT),
                                ('up', d[d.direction == 'up'], MIN_VISITS_LEG),
                                ('down', d[d.direction == 'down'], MIN_VISITS_LEG)):
            fits[key] = huber_slope(sub['altitude_deg'], sub[ycol], min_n=min_n)
        if fits['all'] is None:
            continue
        r = dict(day_obs=int(day), n=len(d))
        for key, f in fits.items():
            r[f'slope_{key}'] = f['slope'] if f else np.nan
            r[f'err_{key}'] = f['slope_err'] if f else np.nan
            r[f'n_{key}'] = f['n'] if f else 0
            r[f'offset_{key}'] = (f['intercept'] + f['slope'] * ref_elev_deg if f else np.nan)
        r['resid_nmad_all'] = fits['all']['resid_nmad']
        r['spearman_rho_all'] = fits['all']['spearman_rho']
        r['pearson_r_all'] = fits['all']['pearson_r']
        rows.append(r)
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out['difference'] = out.slope_up - out.slope_down
    comb = np.sqrt(out.err_up ** 2 + out.err_down ** 2)
    out['difference_sigma'] = out.difference / comb.replace(0, np.nan)

    if verbose:
        a = out.slope_all.dropna().to_numpy(float)
        print(f'per-night elevation slope of {ycol}, all points '
              f'[um of equivalent hexapod dz per deg]')
        print(f'  nights fitted       : {len(a)}')
        print(f'  median slope        : {np.median(a):+.3f} um per deg')
        print(f'  nMAD of the slopes  : {nmad(a):.3f} um per deg')
        print(f'  median formal error : {out.err_all.median():.3f} um per deg')
        o = out.offset_all.dropna().to_numpy(float)
        within = out.resid_nmad_all.median()
        if len(o):
            print(f'per-night offset at {ref_elev_deg:.0f} deg elevation '
                  f'[um of equivalent hexapod dz]: median {np.median(o):+.1f}, '
                  f'nMAD {nmad(o):.1f}, n {len(o)} nights')
            if within and np.isfinite(within) and within > 0:
                print(f'  within-night residual nMAD {within:.1f} um; night-to-night over '
                      f'within-night {nmad(o) / within:.2f} (dimensionless, offset nMAD over '
                      f'residual nMAD)')
        both = out.dropna(subset=['difference'])
        if len(both):
            med = float(both.difference.median())
            pos = int((both.difference > 0).sum())
            p = float(stats.binomtest(pos, len(both), 0.5).pvalue)
            n_sig = int((both.difference_sigma.abs() > 3).sum())
            print(f'rising minus falling slope, {len(both)} nights with both legs')
            print(f'  median difference {med:+.3f} um of equivalent hexapod dz per deg, '
                  f'nMAD {nmad(both.difference.to_numpy(float)):.3f} um per deg')
            print(f'  nights differing by more than 3 combined standard errors: '
                  f'{n_sig} of {len(both)}')
            print(f'  rising steeper than falling on {pos} of {len(both)} nights '
                  f'(sign test p {p:.3g} dimensionless)')
            print(f'  -> {"hysteresis" if p < 0.01 else "no consistent direction dependence"}')
    return out


def band_change_step(df, col, verbose=True, label=''):
    """Median absolute step in a residual across a filter change, against same-band steps.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``day_obs``, ``seq_num``, ``band`` and `col`.
    col : `str`
        Residual column [µm of equivalent hexapod dz].
    verbose : `bool`, optional
        Print the line.
    label : `str`, optional
        Name used in the printed line.

    Returns
    -------
    res : `dict`
        ``n_change``, ``median_change``, ``n_same``, ``median_same`` [µm of equivalent hexapod
        dz] and their ``ratio`` (dimensionless, band-change over same-band median step).

    Notes
    -----
    This is the metric that argues for a band-independent correction. Fitting each band
    separately makes every filter change inject a step into the corrected residual, because the
    coefficients swap while nothing physical happens. Steps are taken within a night only,
    between consecutive ``seq_num``, so the daytime gap is never crossed. Residual excess over
    the same-band step is a real per-band focus offset, which a shared slope cannot remove.
    """
    chg, same = [], []
    for _, g in df.groupby('day_obs'):
        g = g.sort_values('seq_num')
        step = g[col].diff().abs()
        changed = g.band.ne(g.band.shift(1))
        ok = step.notna()
        chg.append(step[ok & changed])
        same.append(step[ok & ~changed])
    chg = pd.concat(chg) if chg else pd.Series(dtype=float)
    same = pd.concat(same) if same else pd.Series(dtype=float)
    res = dict(n_change=int(len(chg)),
               median_change=float(chg.median()) if len(chg) else float('nan'),
               n_same=int(len(same)),
               median_same=float(same.median()) if len(same) else float('nan'))
    res['ratio'] = (res['median_change'] / res['median_same']
                    if res['median_same'] else float('nan'))
    if verbose:
        print(f'  {label:26s} band change {res["median_change"]:7.1f} um (n {res["n_change"]}) '
              f'  same band {res["median_same"]:7.1f} um (n {res["n_same"]})   '
              f'ratio {res["ratio"]:.2f} (dimensionless, band-change over same-band)')
    return res


def per_band_fit(df, features, model='huber', n_splits=N_SPLITS, verbose=True):
    """The shared band-independent model scored per band, beside a per-band refit.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``y``, ``band``, ``day_obs`` and the feature columns.
    features : `list` [`str`]
        Feature column names.
    model : `str`, optional
        Key for `make_model`.
    n_splits : `int`, optional
        Cross-validation folds.
    verbose : `bool`, optional
        Print the table.

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``band``, ``n``, ``uncorrected_nmad``, ``shared_nmad``, ``own_nmad``
        [µm of equivalent hexapod dz] and ``own_truss`` [µm of equivalent hexapod dz per °C].

    Notes
    -----
    The band-independent model is the deliverable, so the per-band refit is here to be
    compared against, not adopted: the per-band truss coefficients spread enough that swapping
    between them at every filter change injects a step (see `band_change_step`), which is a
    worse defect than the small per-band gain in scatter.
    """
    shared = evaluate(df, features, model=model, n_splits=n_splits, verbose=False)
    rows = []
    for band, g in df.groupby('band'):
        if len(g) < 200:
            continue
        idx = df.index.get_indexer(g.index)
        own = evaluate(g.reset_index(drop=True), features, model=model,
                       n_splits=min(n_splits, g['day_obs'].nunique()), verbose=False)
        f = fit_full(g.reset_index(drop=True), features, model=model, verbose=False)
        truss = (float(f['coef'][features.index('truss_temp_mean_c')])
                 if f['coef'] is not None and 'truss_temp_mean_c' in features else float('nan'))
        rows.append(dict(band=band, n=len(g), uncorrected_nmad=nmad(g['y'].to_numpy(float)),
                         shared_nmad=nmad(shared['resid'][idx]), own_nmad=own['nmad'],
                         own_truss=truss))
    out = pd.DataFrame(rows)
    if verbose:
        print('per band, night-grouped [um of equivalent hexapod dz]')
        for _, r in out.iterrows():
            print(f'  {r.band}  n {int(r.n):6d}  uncorrected {r.uncorrected_nmad:6.1f}  '
                  f'shared model {r.shared_nmad:6.1f}  own model {r.own_nmad:6.1f}  '
                  f'own truss coefficient {r.own_truss:+8.2f} um per deg C')
        if len(out) > 1:
            t = out.own_truss.dropna()
            print(f'  per-band truss coefficient spread: {t.min():+.2f} to {t.max():+.2f} '
                  f'um of equivalent hexapod dz per deg C, nMAD {nmad(t.to_numpy()):.2f}')
    return out


def within_set_scatter(df, set_col, cols, verbose=True):
    """Peak-to-peak scatter within each observing set, per column.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying `set_col` and `cols`.
    set_col : `str`
        Column identifying a set — a FAM block, or a night.
    cols : `dict`
        ``{column: unit string}``, each column a quantity to reduce.
    verbose : `bool`, optional
        Print the medians.

    Returns
    -------
    out : `pandas.DataFrame`
        One row per set, with ``<col>_p2p`` and ``<col>_nmad`` per input column and ``n``.

    Notes
    -----
    Peak-to-peak rather than a robust width because a set holds only a handful of visits, where
    an nMAD is not better determined than the range and is harder to reason about. Sets with
    fewer than three finite values in a column give NaN for that column rather than 0.
    """
    rows = []
    for key, g in df.groupby(set_col):
        r = {set_col: key, 'n': len(g)}
        for c in cols:
            v = g[c].to_numpy(float)
            v = v[np.isfinite(v)]
            r[f'{c}_p2p'] = float(v.max() - v.min()) if len(v) >= 3 else float('nan')
            r[f'{c}_nmad'] = nmad(v) if len(v) >= 3 else float('nan')
        rows.append(r)
    out = pd.DataFrame(rows)
    if verbose:
        for c, unit in cols.items():
            p = out[f'{c}_p2p'].dropna()
            if len(p):
                print(f'  {c:22s} within-set peak-to-peak: median {p.median():.4f} {unit}, '
                      f'n {len(p)} sets')
    return out


# --------------------------------------------------------------------- FAM blocks and sets

#: Visits per selected set: 12 triplets, one in-focus ``acq`` each.
DEFAULT_SET_SIZE = 12

#: ``seq_num`` step between consecutive ``acq`` visits of a triplet sequence, each triplet being
#: intra-focal, extra-focal and in-focus [dimensionless, a sequence-number difference].
DEFAULT_SEQ_STEP = 3

#: Pointing tolerance within one block [deg], applied to altitude, azimuth and rotator angle.
#: The selected sets hold pointing to about 0.01 deg, so 2.0 deg and the coadd study's 5.0 deg
#: give the same sets.
DEFAULT_POINTING_TOL = 2.0

#: A new block starts once ``seq_num`` reaches this far past the block's first visit
#: [dimensionless, a sequence-number difference]. A 12-triplet block spans exactly 33, so 36
#: keeps one whole and splits a back-to-back repeat.
DEFAULT_MAX_SEQ_SPAN = 36.0


def _wrapdiff(a, b):
    """Circular difference between two angles [deg].

    Parameters
    ----------
    a, b : `float`
        Angles in deg.

    Returns
    -------
    d : `float`
        Smallest absolute difference in deg, in [0, 180].
    """
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def assign_blocks(df, seq_col='acq_seq_num', pointing_tol=DEFAULT_POINTING_TOL,
                  max_seq_span=DEFAULT_MAX_SEQ_SPAN):
    """Label contiguous fixed-pointing Full Array Mode blocks.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must carry ``science_program``, ``day_obs``, `seq_col`, ``altitude_deg`` [deg],
        ``azimuth_deg_consdb`` [deg] and ``rotator_angle_deg`` [deg].
    seq_col : `str`, optional
        Sequence-number column defining observing order.
    pointing_tol : `float`, optional
        Tolerance in deg, applied to all three angles.
    max_seq_span : `float`, optional
        Maximum sequence-number span of one block [dimensionless].

    Returns
    -------
    df : `pandas.DataFrame`
        A copy sorted by ``(science_program, day_obs, seq_col)`` with an integer ``block``
        column. Rows lacking any pointing angle are dropped, since a block is defined by held
        pointing and a row without it cannot be placed in one.

    Raises
    ------
    KeyError
        If a required column is absent, naming it. A block is a **derived** quantity, not a
        stored one, so falling back to grouping by night would silently substitute a whole
        night for a 12-triplet block and quietly change what every within-set number means.

    Notes
    -----
    Within one ``(science_program, day_obs)`` the visits are walked in sequence order and a new
    block starts when the sequence number reaches `max_seq_span` past the block's first visit,
    or when altitude, azimuth or rotator angle drifts beyond `pointing_tol` from it. Missing
    triplets inside the span are tolerated. Azimuth is compared circularly so a block spanning
    360 deg does not split.
    """
    need = ('science_program', 'day_obs', seq_col, 'altitude_deg', 'azimuth_deg_consdb',
            'rotator_angle_deg')
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f'assign_blocks needs {", ".join(miss)}, absent from the table; a FAM '
                       f'block is derived from held pointing and cannot be substituted by a '
                       f'coarser grouping')
    v = df.dropna(subset=['altitude_deg', 'azimuth_deg_consdb', 'rotator_angle_deg']).copy()
    v = v.sort_values(['science_program', 'day_obs', seq_col]).reset_index(drop=True)

    block = np.full(len(v), -1, dtype=int)
    seq = v[seq_col].to_numpy(float)
    alt = v['altitude_deg'].to_numpy(float)
    az = np.mod(v['azimuth_deg_consdb'].to_numpy(float), 360.0)
    rot = v['rotator_angle_deg'].to_numpy(float)

    nb = 0
    for _, pos in v.groupby(['science_program', 'day_obs']).indices.items():
        pos = np.sort(np.asarray(pos))
        cur, start_seq, ref = None, None, None
        for p in pos:
            new = (start_seq is None
                   or (seq[p] - start_seq) >= max_seq_span
                   or abs(alt[p] - ref[0]) > pointing_tol
                   or _wrapdiff(az[p], ref[1]) > pointing_tol
                   or abs(rot[p] - ref[2]) > pointing_tol)
            if new:
                cur = nb
                nb += 1
                start_seq = seq[p]
                ref = (alt[p], az[p], rot[p])
            block[p] = cur
    v['block'] = block
    return v


def select_sets(df, seq_col='acq_seq_num', set_size=DEFAULT_SET_SIZE,
                seq_step=DEFAULT_SEQ_STEP, drop_lut_epoch=True, verbose=True):
    """Keep only blocks that are a clean run of `set_size` triplets.

    Parameters
    ----------
    df : `pandas.DataFrame`
        An `assign_blocks` result.
    seq_col : `str`, optional
        Sequence-number column defining observing order.
    set_size : `int`, optional
        Required visits per set.
    seq_step : `int` or `None`, optional
        Required sequence-number step [dimensionless]. None skips the check.
    drop_lut_epoch : `bool`, optional
        Drop nights running a different hexapod look-up-table configuration.
    verbose : `bool`, optional
        Print the cut-by-cut counts.

    Returns
    -------
    df : `pandas.DataFrame`
        The kept visits, with ``set_id`` numbering surviving sets from 0 in
        ``(day_obs, seq_col)`` order.
    info : `dict`
        Counts at each cut.

    Notes
    -----
    Three cuts in order: exactly `set_size` visits in the block; a constant sequence step of
    `seq_step`, which validates the intra-focal / extra-focal / in-focus triplet structure; and
    by default no night in `thermal_focus_lib.LUT_EPOCH_OFFSET_NIGHTS`, whose different
    commanded baseline puts it thousands of µm of equivalent hexapod dz from the rest.
    """
    sized = [b for b, g in df[df.block >= 0].groupby('block') if len(g) == set_size]
    n_sized = len(sized)

    kept, n_bad_step = [], 0
    for b in sized:
        s = np.sort(df.loc[df.block == b, seq_col].to_numpy(int))
        if seq_step is not None and not np.all(np.diff(s) == seq_step):
            n_bad_step += 1
            continue
        kept.append(b)

    out = df[df.block.isin(kept)].copy()
    n_before_epoch, nights_before = len(kept), out.day_obs.nunique()
    dropped_nights = sorted(set(int(d) for d in out.day_obs.unique())
                            & set(L.LUT_EPOCH_OFFSET_NIGHTS))
    if drop_lut_epoch and dropped_nights:
        bad = out.day_obs.isin(dropped_nights)
        n_dropped_sets = out.loc[bad, 'block'].nunique()
        out = out[~bad].copy()
    else:
        n_dropped_sets, dropped_nights = 0, []

    order = (out.groupby('block')[['day_obs', seq_col]].min()
             .sort_values(['day_obs', seq_col]).index.tolist())
    out['set_id'] = out.block.map({b: i for i, b in enumerate(order)})
    out = out.sort_values(['set_id', seq_col]).reset_index(drop=True)

    info = dict(n_blocks=int(df[df.block >= 0].block.nunique()), n_sized=n_sized,
                n_bad_step=n_bad_step, n_before_epoch=n_before_epoch,
                nights_before=int(nights_before), n_dropped_sets=int(n_dropped_sets),
                dropped_nights=dropped_nights, n_sets=int(out.set_id.nunique()),
                n_visits=len(out), n_nights=int(out.day_obs.nunique()),
                set_size=set_size, seq_step=seq_step)
    if verbose:
        print(f'  blocks {info["n_blocks"]} -> exactly {set_size} visits: {n_sized}'
              f' -> constant step of {seq_step}: {n_before_epoch}'
              f' (rejected {n_bad_step} for an irregular step)')
        if n_dropped_sets:
            print(f'  dropping {len(dropped_nights)} LUT-epoch nights removes '
                  f'{n_dropped_sets} sets')
        print(f'  -> {info["n_sets"]} sets, {info["n_visits"]} visits, '
              f'{info["n_nights"]} nights')
    return out, info
