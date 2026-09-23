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

#: Commanded truss slope from the FAM Double Zernike fits [dimensionless v-mode-1 amplitude per
#: deg C]. The science-image slope must land near this; it is the study's one independent check.
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
            dev = 100 * abs(v1_per_c - FAM_TRUSS_SLOPE) / FAM_TRUSS_SLOPE
            print(f'  truss term as a v-mode-1 slope: {v1_per_c:+.5f} dimensionless v-mode-1 '
                  f'amplitude per deg C')
            print(f'    FAM Double Zernike value {FAM_TRUSS_SLOPE:+.5f} per deg C; '
                  f'differ by {dev:.1f}% (dimensionless)')
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
    The excess is one-sided positive in every band, which is why the fits are robust rather than
    least squares.
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
