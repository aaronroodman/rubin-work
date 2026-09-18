#!/usr/bin/env python3
"""One band-independent thermal correction for the v-mode-1 Trim, fitted jointly.

The correction chain in ``run_science_lut_report.py`` fits Telescope Mount Assembly (TMA)
truss temperature and then the M1M3 z thermal gradient **sequentially and per band**. Because
the coefficients differ band to band, every filter change injects a step into the corrected
residual: on 20260706 at seq_num 213 to 214 the band goes r to g and the corrected residual
jumps by -104.5 µm of equivalent camera-hexapod dz while the commanded Trim is unchanged
(-411.66 µm on both sides), the truss temperature moves +0.060 deg C and the M1M3 z gradient
+0.0036 deg C per m. Nothing physical happens; the coefficients swap. The band-independent fit
below reduces that step to -4.3 µm, and the median absolute step at a band change pooled over
all nights from 32.9 µm (n=963) to 22.0 µm (n=1166).

This script fits **one band-independent model on all the thermal telemetry at once** instead,
which removes that discontinuity by construction and uses the four M1M3 gradients rather than
the z gradient alone.

Whole nights are held out
-------------------------
Within a night the thermal telemetry drifts slowly, so consecutive visits are near-duplicates
in feature space: only 1.9% of the truss temperature's variance is within-night, while 83.8%
of the target variance is *between* nights. A visit-level split therefore lets a model
identify the night from its temperature and memorise that night's offset, scoring the
interpolation between neighbouring visits rather than a thermal response. Measured with
boosted trees on 6 features, residual normalized median absolute deviation (nMAD): 26.5 µm of
equivalent camera-hexapod dz visit-level against 83.1 µm night-grouped, a factor of 3.1
(dimensionless) optimism.

How large that leak is depends on the model's capacity, not on the split alone. For the Huber
linear fit this script actually uses it is small -- 66.0 µm visit-level against 68.4 µm
night-grouped, a factor of 1.04 (dimensionless) -- because five coefficients cannot memorise a
night. The night-grouped split is still the only one reported, since which model will be used
is not known in advance.

**Every reported number therefore comes from a `sklearn.model_selection.GroupKFold` split
grouped on ``day_obs``.** ``--leaky-split`` reproduces the visit-level number, with a warning,
so the size of the trap stays documented rather than folklore.

What the data selected
----------------------
The response is close to linear in the telemetry: under a night-grouped split a Huber linear
fit reaches 68.4 µm where boosted trees reach 90.8 µm and a random forest 98.8 µm, against an
uncorrected baseline of 332.4 µm. The trees overfit night-specific structure that does not
transfer to held-out nights, so the "machine learning method" here is regularised linear
regression, which also reads directly as a coefficient per deg C.

The default feature set is the TMA truss temperature plus the four M1M3 thermal gradients. The
truss is the load path setting the M1M3-to-camera spacing, so its temperature is the
mechanistically motivated regressor; ``cam_AverageTemp`` correlates marginally better against
the target (Pearson r +0.804 against +0.767) but the two are collinear at Pearson r 0.9663, so
that is proxy behaviour rather than evidence of cause, and including it is slightly *worse*
under Huber (73.0 against 68.4 µm). Excluding it also makes the truss coefficient stable and
physically checkable: +124.3 +/- 1.5 µm of equivalent camera-hexapod dz per deg C across
folds, which is +0.11197 dimensionless v-mode-1 amplitude per deg C against the independent
Full Array Mode (FAM) value ``FAM_TRUSS_SLOPE = +0.09634`` -- agreement to 16%. With
``cam_AverageTemp`` included the two collinear temperatures split the term and the truss
coefficient wanders from +1.0 to +17.1 µm per deg C.

Tested and adding nothing beyond that set, all available through ``--features``: the air
temperatures (collinear with each other, 1.6 to 3.1% within-night variance), the wind channels,
and elevation (83.7 against 83.1 µm -- the hexapod look-up table already handles elevation).
``danish_epoch`` is excluded on principle: 0.0% within-night variance makes it a pure
night-label.

Usage
-----
Run from ``aos/``::

    python code/science_lut/run_thermal_model.py
    python code/science_lut/run_thermal_model.py --v1-per-um-dz 9.008514e-04
    python code/science_lut/run_thermal_model.py --model ridge --n-splits 8
    python code/science_lut/run_thermal_model.py --features truss grads camtemp
    python code/science_lut/run_thermal_model.py --leaky-split   # documents the trap only

Reads ``output/science_lut/science_lut.parquet`` (``run_science_lut.py``). Writes
``thermal_model.pdf``, ``thermal_model_coeffs.parquet`` and ``thermal_model_predictions.parquet``
alongside it; the predictions file carries the out-of-fold correction per ``visit_id`` and is
what ``run_science_lut_report.py --chain-order ml-elev`` consumes. That chain writes
``v1_dzequiv_mlcorr`` into its results parquet, which ``run_visit_elevation.py --chain-tag
ml_elev_trim_meas_nolutepoch --ycol v1_dzequiv_mlcorr`` then draws as the corrected series.

Notes
-----
The 8 nights in ``LUT_EPOCH_OFFSET_NIGHTS`` are dropped by default. Their hexapod look-up-table
term sits far below the rest at the same elevation, so a different LUT configuration was loaded
at the time.

Visits missing a feature are imputed at the training-fold median rather than dropped, because
the truss temperature reaches only about 89% of visits while the M1M3 gradients reach 99.9%;
dropping would discard a ninth of the sample to a channel the gradients largely cover.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))                       # repo root -> common/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common.utils import nmad                                    # noqa: E402
from run_science_lut import (BAND_COLORS, FAM_TRUSS_SLOPE,        # noqa: E402
                            LUT_EPOCH_OFFSET_NIGHTS, MEASURED_SIGN)

#: Default variant, matching the rest of the study.
DEFAULT_VARIANT = 'v50_34__batoid__consdb_v1'

#: Band order for the per-band tables, blue to red.
BAND_ORDER = ('u', 'g', 'r', 'i', 'z', 'y')

#: Named feature groups, selectable by ``--features``. ``truss`` and ``grads`` together are the
#: default: mechanistically motivated and, measured, the best configuration. The rest are kept
#: reachable so the negative results in the module docstring stay reproducible.
FEATURE_GROUPS = {
    'truss': ['truss_temp_mean_c'],
    'grads': ['m1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
              'm1m3_radial_gradient_c_per_m', 'm1m3_x_gradient_c_per_m'],
    'zgrad': ['m1m3_z_gradient_c_per_m'],
    'camtemp': ['cam_AverageTemp'],
    'airtemps': ['cam_air_temp', 'm2_air_temp', 'm1m3_air_temp', 'outside_temp',
                 'sonic_temperature'],
    'wind': ['wind_speed_ms', 'wind_inside_maxmag', 'into_wind_deg'],
    'elev': ['altitude_deg'],
}

#: The default feature set: truss temperature plus the four M1M3 thermal gradients.
DEFAULT_FEATURES = ('truss', 'grads')

#: The per-band correction chain this model replaces, used as the like-for-like baseline in the
#: filter-change comparison. Its results document is built by
#: ``run_science_lut_report.py --chain-order truss-grad-elev --response trim-meas
#: --drop-lut-epoch-offset-nights``.
PER_BAND_CHAIN_TAG = 'truss_grad_elev_trim_meas_nolutepoch'
PER_BAND_CHAIN_COL = 'v1_dzequiv_truss_grad_corr'

#: Units per feature, for every table and axis label. Physical units are mandatory here -- a
#: coefficient of -3605 means nothing without "µm of equivalent camera-hexapod dz per
#: (deg C per m)".
FEATURE_UNITS = {
    'truss_temp_mean_c': 'deg C',
    'cam_AverageTemp': 'deg C',
    'cam_air_temp': 'deg C',
    'm2_air_temp': 'deg C',
    'm1m3_air_temp': 'deg C',
    'outside_temp': 'deg C',
    'sonic_temperature': 'deg C',
    'm1m3_z_gradient_c_per_m': 'deg C per m',
    'm1m3_y_gradient_c_per_m': 'deg C per m',
    'm1m3_x_gradient_c_per_m': 'deg C per m',
    'm1m3_radial_gradient_c_per_m': 'deg C per m',
    'wind_speed_ms': 'm per s',
    'wind_inside_maxmag': 'm per s',
    'into_wind_deg': 'deg',
    'altitude_deg': 'deg',
}

#: Ratio of the night-grouped to the visit-level residual nMAD for boosted trees on 6 features
#: (83.1 um against 26.5 um of equivalent camera-hexapod dz). Quoted only as the worst case
#: seen: the leak is a property of the MODEL, not of the split alone. For the Huber linear fit
#: used here it is about 1.04 (dimensionless), because five coefficients cannot memorise a
#: night. ``--leaky-split`` prints whichever ratio the run actually measures.
LEAK_FACTOR = 3.1


def make_model(name):
    """Build a named sklearn pipeline.

    Parameters
    ----------
    name : `str`
        One of ``huber``, ``ridge``, ``spline_huber``, ``histgb``, ``rf``.

    Returns
    -------
    model : `sklearn.base.BaseEstimator`
        Unfitted pipeline. Every pipeline imputes missing features at the training-fold median,
        so a fold never sees a NaN and the imputation is fitted inside the fold rather than on
        the whole sample.

    Notes
    -----
    ``huber`` is the default and the recommendation: measured best under a night-grouped split
    on the default feature set, and its coefficients read directly in µm of equivalent
    camera-hexapod dz per feature unit. The tree models are kept for the comparison table, where
    they lose -- they fit night-specific structure that does not transfer across held-out nights.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    imp = SimpleImputer(strategy='median')
    if name == 'huber':
        return make_pipeline(imp, StandardScaler(), HuberRegressor(max_iter=2000))
    if name == 'ridge':
        return make_pipeline(imp, StandardScaler(), RidgeCV())
    if name == 'spline_huber':
        return make_pipeline(imp, SplineTransformer(n_knots=4, degree=3), StandardScaler(),
                             HuberRegressor(max_iter=3000))
    if name == 'histgb':
        return HistGradientBoostingRegressor(max_iter=200, max_depth=2, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=0)
    if name == 'rf':
        return make_pipeline(imp, RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=0))
    raise ValueError(f'unknown model {name!r}')


MODEL_NAMES = ('huber', 'ridge', 'spline_huber', 'histgb', 'rf')


def resolve_features(groups):
    """Expand ``--features`` group names into a de-duplicated column list.

    Parameters
    ----------
    groups : `iterable` [`str`]
        Keys of `FEATURE_GROUPS`.

    Returns
    -------
    features : `list` [`str`]
        Column names, in the order the groups were given, each appearing once.
    """
    out = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise SystemExit(f'unknown feature group {g!r}; choose from '
                             f'{", ".join(sorted(FEATURE_GROUPS))}')
        for c in FEATURE_GROUPS[g]:
            if c not in out:
                out.append(c)
    return out


def load_target(in_dir, variant=DEFAULT_VARIANT, v1_per_um_dz=None, features=None,
                drop_lut_epoch=True, verbose=True):
    """Read the ``Trim - measured`` response and its telemetry features per visit.

    Parameters
    ----------
    in_dir : `pathlib.Path`
        Directory holding ``science_lut.parquet``.
    variant : `str`, optional
        ``optical_state`` variant id.
    v1_per_um_dz : `float`
        Dimensionless v-mode-1 amplitude per µm of camera-hexapod dz, used to express every
        amplitude as an equivalent hexapod motion.
    features : `list` [`str`], optional
        Feature columns to carry through. Missing columns raise.
    drop_lut_epoch : `bool`, optional
        Drop `LUT_EPOCH_OFFSET_NIGHTS`, the default.
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit_id``, ``day_obs``, ``seq_num``, ``band``, ``altitude_deg`` [deg], the requested
        features, and ``y`` -- the response ``v1(Trim) - v1(measured)`` in µm of equivalent
        camera-hexapod dz.

    Notes
    -----
    The response matches the ``trim-meas`` numerator in ``run_science_lut_report.RESPONSES``,
    which leaves the hexapod look-up-table baseline out. The LUT carries essentially the whole
    elevation dependence, so this response is what the closed loop and the wavefront sensors do
    on their own.
    """
    path = in_dir / 'science_lut.parquet'
    if not path.exists():
        raise SystemExit(f'{path} is missing; build it first with '
                         f'python code/science_lut/run_science_lut.py')
    b = pd.read_parquet(path)
    if variant not in set(b.variant.unique()):
        raise SystemExit(f'variant {variant!r} not in {path}; present: '
                         f'{", ".join(sorted(b.variant.unique()))}')
    b = b[b.variant == variant]
    n_all, nights_all = len(b), b.day_obs.nunique()

    if drop_lut_epoch:
        drop = b.day_obs.isin(LUT_EPOCH_OFFSET_NIGHTS)
        # Count the nights present before the drop; intersecting afterwards always gives 0.
        n_nights_dropped = b.loc[drop, 'day_obs'].nunique()
        b = b[~drop]
        if verbose:
            print(f'dropped {int(drop.sum())} visits on {n_nights_dropped} '
                  f'flagged nights running a different hexapod LUT configuration '
                  f'({", ".join(str(d) for d in LUT_EPOCH_OFFSET_NIGHTS)})')

    features = list(features or resolve_features(DEFAULT_FEATURES))
    missing = [c for c in features if c not in b.columns]
    if missing:
        raise SystemExit(f'{path} lacks feature columns {missing}')

    keep = ['visit_id', 'day_obs', 'seq_num', 'band', 'altitude_deg',
            'v1_trim', 'v1_meas'] + [c for c in features if c not in ('altitude_deg',)]
    df = b[list(dict.fromkeys(keep))].dropna(subset=['v1_trim', 'v1_meas']).copy()
    df['y'] = (df.v1_trim + MEASURED_SIGN * df.v1_meas) / v1_per_um_dz
    df = df.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)

    # The truss temperature is the primary regressor, so a visit without one must be
    # dropped and not imputed: the imputer substitutes the run-wide median, which is a
    # per-night bias of order 1 deg C -- about 124 um of equivalent camera-hexapod dz --
    # rather than noise. Scattered single-exposure gaps are already filled upstream by
    # common.efd_db.interpolate_within_night, so what survives here sits hours outside
    # its own night's valid span.
    if 'truss_temp_mean_c' in df.columns:
        no_truss = df.truss_temp_mean_c.isna()
        if no_truss.any():
            if verbose:
                by_night = df.loc[no_truss, 'day_obs'].value_counts()
                print(f'no truss temperature after in-night interpolation: dropped '
                      f'{int(no_truss.sum())} visits on {len(by_night)} nights '
                      f'(worst: {", ".join(f"{int(d)} n={int(k)}" for d, k in by_night.head(3).items())})')
            df = df[~no_truss].reset_index(drop=True)

    if verbose:
        print(f'variant {variant}: {len(df)} visits over {df.day_obs.nunique()} nights '
              f'(from {n_all} visits / {nights_all} nights before cuts)')
        print(f'response v1(Trim) - v1(measured): median {df.y.median():+.1f} um, '
              f'nMAD {nmad(df.y.to_numpy(float)):.1f} um of equivalent camera-hexapod dz')
        print(f'features ({len(features)}):')
        for c in features:
            v = df[c].to_numpy(float)
            f = np.isfinite(v)
            print(f'  {c:32s} {f.sum():6d} finite ({100 * f.mean():5.1f}%), '
                  f'{v[f].min():+8.3f} to {v[f].max():+8.3f} {FEATURE_UNITS.get(c, "?")}')
    return df, features


def evaluate(df, features, model='huber', n_splits=5, leaky=False, verbose=True):
    """Out-of-fold prediction with whole nights held out.

    Parameters
    ----------
    df : `pandas.DataFrame`
        From `load_target`, carrying ``y`` and the feature columns.
    features : `list` [`str`]
    model : `str`, optional
        Key for `make_model`.
    n_splits : `int`, optional
    leaky : `bool`, optional
        Split on visits instead of nights. Only for reproducing the leak; the returned numbers
        are optimistic -- by up to `LEAK_FACTOR` for a high-capacity model, and by about 1.04
        (dimensionless) for the Huber linear fit -- and must not be quoted as performance.
    verbose : `bool`, optional

    Returns
    -------
    res : `dict`
        ``pred`` (out-of-fold prediction [µm of equivalent camera-hexapod dz]), ``resid``,
        ``coefs`` (per-fold coefficient array, or None for the tree models), ``nmad``, ``r2``,
        ``n_splits``, ``grouped``.

    Notes
    -----
    `sklearn.model_selection.GroupKFold` on ``day_obs`` is what makes the score meaningful: see
    the module docstring for the measured size of the visit-level leak.
    """
    from sklearn.model_selection import GroupKFold, KFold

    X = df[features].to_numpy(float)
    y = df.y.to_numpy(float)
    groups = df.day_obs.to_numpy()

    if leaky:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = splitter.split(X, y)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups=groups)

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
        kind = ('visit-level (LEAKY)' if leaky
                else f'night-grouped on day_obs')
        print(f'{model}, {n_splits}-fold {kind}: residual nMAD {res["nmad"]:.1f} um of '
              f'equivalent camera-hexapod dz, R2 {res["r2"]:.3f} (dimensionless)')
        if leaky:
            print('  WARNING: whole nights were NOT held out, so this score is not a '
                  'performance estimate and must not be quoted as one. How optimistic it is '
                  'depends on the model: up to a factor of '
                  f'{LEAK_FACTOR} (dimensionless) for boosted trees on 6 features, but only '
                  'about 1.04 (dimensionless) for a Huber linear fit, which has too few '
                  'coefficients to memorise a night. Compare against the night-grouped number '
                  'printed above.')
    return res


def _linear_coefficients(fitted, features):
    """Coefficients of a fitted linear pipeline in physical units, or None.

    Parameters
    ----------
    fitted : `sklearn.pipeline.Pipeline`
    features : `list` [`str`]

    Returns
    -------
    coef : `numpy.ndarray` or `None`
        One coefficient per feature, in µm of equivalent camera-hexapod dz per feature unit.
        None when the final estimator is not linear, or when a spline expansion means the
        coefficients no longer map one-to-one onto features.
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


def per_band_residual(df, resid, verbose=True):
    """Residual scatter per band from the band-independent fit.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``band``.
    resid : `array_like`
        Out-of-fold residual [µm of equivalent camera-hexapod dz].
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``n``, ``median``, ``resid_nmad`` [µm of equivalent camera-hexapod dz].

    Notes
    -----
    A band-independent model is the point of this script, so the check that matters is that no
    single band is left badly served. A per-band median well away from zero is a genuine
    band-dependent focus offset -- filter thickness -- rather than a defect of the fit.
    """
    d = df.assign(resid=resid)
    rows = []
    for band in BAND_ORDER:
        g = d[d.band == band]
        if not len(g):
            continue
        rows.append(dict(band=band, n=len(g), median=float(g.resid.median()),
                         resid_nmad=float(nmad(g.resid.to_numpy(float)))))
    out = pd.DataFrame(rows)
    if verbose and len(out):
        print('per-band residual of the band-independent model '
              '[um of equivalent camera-hexapod dz]')
        print(f'  {"band":5s} {"n":>7s} {"median":>9s} {"nMAD":>9s}')
        for _, r in out.iterrows():
            print(f'  {r.band:5s} {int(r.n):7d} {r["median"]:+9.1f} {r.resid_nmad:9.1f}')
    return out


def filter_change_step(df, col, verbose=True, label=''):
    """Median absolute step in a residual across a filter change, against same-band steps.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``day_obs``, ``seq_num``, ``band`` and `col`.
    col : `str`
        Residual column [µm of equivalent camera-hexapod dz].
    verbose : `bool`, optional
    label : `str`, optional
        Name used in the printed line.

    Returns
    -------
    res : `dict`
        ``n_change``, ``median_change``, ``n_same``, ``median_same`` -- median |step| in µm of
        equivalent camera-hexapod dz, and the ``ratio`` of the two (dimensionless).

    Notes
    -----
    Steps are taken within a night only, between consecutive ``seq_num``, so the daytime gap is
    never crossed. A band-independent correction should bring the band-change step down towards
    the same-band step; residual excess is a real per-band focus offset, which a shared slope
    cannot remove.
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
    res = dict(n_change=int(len(chg)), median_change=float(chg.median()) if len(chg) else np.nan,
               n_same=int(len(same)), median_same=float(same.median()) if len(same) else np.nan)
    res['ratio'] = (res['median_change'] / res['median_same']
                    if res['median_same'] else np.nan)
    if verbose:
        print(f'  {label:34s} band change {res["median_change"]:7.1f} um '
              f'(n={res["n_change"]}), same band {res["median_same"]:7.1f} um '
              f'(n={res["n_same"]}), ratio {res["ratio"]:.2f} (dimensionless)')
    return res


def coefficient_table(coefs, features, v1_per_um_dz, verbose=True):
    """Per-fold coefficient mean and scatter, with the FAM truss cross-check.

    Parameters
    ----------
    coefs : `numpy.ndarray`
        Shape ``(n_folds, n_features)``, in µm of equivalent camera-hexapod dz per feature unit.
    features : `list` [`str`]
    v1_per_um_dz : `float`
        Used to convert the truss coefficient into a dimensionless v-mode-1 amplitude per deg C
        for comparison with `FAM_TRUSS_SLOPE`.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per feature: ``feature``, ``unit``, ``mean``, ``std``, ``n_folds``, and ``sign_stable``.
    """
    rows = []
    for i, f in enumerate(features):
        c = coefs[:, i]
        rows.append(dict(feature=f, unit=f'um equiv camhex dz per {FEATURE_UNITS.get(f, "?")}',
                         mean=float(c.mean()), std=float(c.std()), n_folds=len(c),
                         sign_stable=bool(len(set(np.sign(c))) == 1)))
    out = pd.DataFrame(rows)
    if verbose:
        print(f'coefficients across {coefs.shape[0]} folds '
              f'[um of equivalent camera-hexapod dz per feature unit]')
        for _, r in out.iterrows():
            flag = '' if r.sign_stable else '   <-- SIGN FLIPS across folds'
            print(f'  {r.feature:32s} {r["mean"]:+10.1f} +/- {r["std"]:8.1f} '
                  f'per {FEATURE_UNITS.get(r.feature, "?"):12s}{flag}')
        if 'truss_temp_mean_c' in features:
            i = features.index('truss_temp_mean_c')
            v1_per_c = coefs[:, i].mean() * v1_per_um_dz
            dev = 100 * abs(v1_per_c - FAM_TRUSS_SLOPE) / FAM_TRUSS_SLOPE
            print(f'  truss term as a v-mode-1 slope: {v1_per_c:+.5f} dimensionless v-mode-1 '
                  f'amplitude per deg C')
            print(f'    FAM Double Zernike value {FAM_TRUSS_SLOPE:+.5f} per deg C; '
                  f'differ by {dev:.0f}% (dimensionless)')
    return out


def compare_models(df, features, models=MODEL_NAMES, n_splits=5, verbose=True):
    """Night-grouped residual nMAD for each named model.

    Parameters
    ----------
    df : `pandas.DataFrame`
    features : `list` [`str`]
    models : `iterable` [`str`], optional
    n_splits : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per model: ``model``, ``resid_nmad`` [µm of equivalent camera-hexapod dz],
        ``frac_of_baseline`` (dimensionless), ``r2`` (dimensionless).
    """
    base = nmad(df.y.to_numpy(float))
    rows = []
    for m in models:
        r = evaluate(df, features, model=m, n_splits=n_splits, verbose=False)
        rows.append(dict(model=m, resid_nmad=r['nmad'],
                         frac_of_baseline=r['nmad'] / base, r2=r['r2']))
    out = pd.DataFrame(rows).sort_values('resid_nmad').reset_index(drop=True)
    if verbose:
        print(f'model comparison, {n_splits}-fold night-grouped; uncorrected baseline nMAD '
              f'{base:.1f} um of equivalent camera-hexapod dz')
        print(f'  {"model":14s} {"nMAD[um]":>9s} {"frac base":>10s} {"R2":>7s}')
        for _, r in out.iterrows():
            print(f'  {r.model:14s} {r.resid_nmad:9.1f} {r.frac_of_baseline:10.3f} '
                  f'{r.r2:7.3f}')
        print('  frac base and R2 are dimensionless')
    return out


def compare_feature_sets(df_all, group_sets, model='huber', n_splits=5, in_dir=None,
                         variant=DEFAULT_VARIANT, v1_per_um_dz=None, verbose=True):
    """Night-grouped residual nMAD for each feature-group combination.

    Parameters
    ----------
    df_all : `pandas.DataFrame`
        Table already carrying every candidate column, from `load_target` called with the union
        of the groups being compared.
    group_sets : `list` [`tuple` [`str`]]
        Each entry a tuple of `FEATURE_GROUPS` keys.
    model : `str`, optional
    n_splits : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per set: ``groups``, ``n_features``, ``resid_nmad`` [µm of equivalent camera-hexapod dz],
        ``r2`` (dimensionless).
    """
    base = nmad(df_all.y.to_numpy(float))
    rows = []
    for gs in group_sets:
        feats = resolve_features(gs)
        if any(c not in df_all.columns for c in feats):
            continue
        r = evaluate(df_all, feats, model=model, n_splits=n_splits, verbose=False)
        rows.append(dict(groups='+'.join(gs), n_features=len(feats),
                         resid_nmad=r['nmad'], frac_of_baseline=r['nmad'] / base, r2=r['r2']))
    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f'feature-set comparison, {model}, {n_splits}-fold night-grouped; baseline nMAD '
              f'{base:.1f} um of equivalent camera-hexapod dz')
        print(f'  {"feature groups":34s} {"n":>3s} {"nMAD[um]":>9s} {"R2":>7s}')
        for _, r in out.iterrows():
            print(f'  {r.groups:34s} {int(r.n_features):3d} {r.resid_nmad:9.1f} {r.r2:7.3f}')
    return out


def fit_full(df, features, model='huber', verbose=True):
    """Fit on every night, the deliverable model.

    Parameters
    ----------
    df : `pandas.DataFrame`
    features : `list` [`str`]
    model : `str`, optional
    verbose : `bool`, optional

    Returns
    -------
    res : `dict`
        ``model`` (the fitted estimator), ``pred`` (in-sample prediction [µm of equivalent
        camera-hexapod dz]), ``resid``, ``coef`` (or None), ``nmad``, ``intercept``.

    Notes
    -----
    The in-sample nMAD of this fit is **not** a performance estimate -- use `evaluate`. It is
    reported only so the difference from the out-of-fold number shows how much the fit depends on
    which nights it saw.
    """
    X = df[features].to_numpy(float)
    y = df.y.to_numpy(float)
    m = make_model(model)
    m.fit(X, y)
    pred = m.predict(X)
    resid = y - pred
    coef = _linear_coefficients(m, features)
    est = list(m.named_steps.values())[-1] if hasattr(m, 'named_steps') else m
    res = dict(model=m, pred=pred, resid=resid, coef=coef, nmad=nmad(resid),
               intercept=float(getattr(est, 'intercept_', np.nan)))
    if verbose:
        print(f'full fit ({model}, all {df.day_obs.nunique()} nights): in-sample residual nMAD '
              f'{res["nmad"]:.1f} um of equivalent camera-hexapod dz (not a performance '
              f'estimate -- see the night-grouped number above)')
        if coef is not None:
            print(f'  intercept {res["intercept"]:+.1f} um of equivalent camera-hexapod dz')
            for f, c in zip(features, coef):
                print(f'  {f:32s} {c:+10.1f} um per {FEATURE_UNITS.get(f, "?")}')
    return res


# --------------------------------------------------------------------------- PDF pages

def page_text(pdf, title, blocks, subtitle=None):
    """Portrait text page: a title and a list of ``(heading, monospace body)`` blocks.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    title : `str`
    blocks : `list` [`tuple` [`str`, `str`]]
    subtitle : `str`, optional
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.955, title, fontsize=15, va='top', weight='bold')
    y = 0.925
    if subtitle:
        fig.text(0.06, y, subtitle, fontsize=9.5, va='top', style='italic')
        y -= 0.030
    for head, body in blocks:
        fig.text(0.06, y, head, fontsize=11, va='top', weight='bold')
        y -= 0.022
        fig.text(0.07, y, body, fontsize=8.2, va='top', family='monospace')
        y -= 0.0205 * (body.count('\n') + 1) + 0.016
    pdf.savefig(fig)
    plt.close(fig)


def _table_text(out, cols, fmts, widths):
    """Render a DataFrame as fixed-width monospace text for `page_text`.

    Parameters
    ----------
    out : `pandas.DataFrame`
    cols : `list` [`str`]
        Columns to show, in order.
    fmts : `list` [`str`]
        A format spec per column, e.g. ``'.1f'`` or ``'s'``.
    widths : `list` [`int`]

    Returns
    -------
    text : `str`
    """
    lines = ['  '.join(f'{c:>{w}s}' for c, w in zip(cols, widths))]
    for _, r in out.iterrows():
        cells = []
        for c, f, w in zip(cols, fmts, widths):
            v = r[c]
            if f == 's':
                cells.append(f'{v:>{w}s}')
            else:
                # A sign flag must precede the width in a format spec, so '+.1f'
                # with width 9 becomes '>+9.1f' and not the invalid '>9+.1f'.
                sign, rest = (f[0], f[1:]) if f[:1] in '+- ' else ('', f)
                cells.append(f'{v:>{sign}{w}{rest}}')
        lines.append('  '.join(cells))
    return '\n'.join(lines)


def page_coefficients(pdf, coefs, features, v1_per_um_dz):
    """Per-fold coefficient scatter, one panel per feature, plus the FAM truss cross-check.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    coefs : `numpy.ndarray`
        Shape ``(n_folds, n_features)`` [µm of equivalent camera-hexapod dz per feature unit].
    features : `list` [`str`]
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm of hexapod dz].
    """
    import matplotlib.pyplot as plt

    n = len(features)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.3, 3.2 * nrow), squeeze=False)
    for i, f in enumerate(features):
        ax = axes[i // ncol][i % ncol]
        c = coefs[:, i]
        ax.plot(np.arange(1, len(c) + 1), c, 'o-', color='tab:blue')
        ax.axhline(c.mean(), color='0.4', ls='--', lw=1)
        ax.axhline(0.0, color='k', lw=0.8)
        ax.set_xlabel('held-out night fold')
        ax.set_ylabel(f'um equiv camhex dz\nper {FEATURE_UNITS.get(f, "?")}', fontsize=8)
        ax.set_title(f'{f}\nmean {c.mean():+.1f} +/- {c.std():.1f}', fontsize=9)
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    extra = ''
    if 'truss_temp_mean_c' in features:
        i = features.index('truss_temp_mean_c')
        v1_per_c = coefs[:, i].mean() * v1_per_um_dz
        dev = 100 * abs(v1_per_c - FAM_TRUSS_SLOPE) / FAM_TRUSS_SLOPE
        extra = (f'  |  truss term {v1_per_c:+.5f} dimensionless v1 per deg C against FAM '
                 f'{FAM_TRUSS_SLOPE:+.5f} per deg C, differ by {dev:.0f}% (dimensionless)')
    fig.suptitle(f'Coefficients across {coefs.shape[0]} night-grouped folds{extra}',
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_resid_vs_features(pdf, df, features, resid, extra_cols=('altitude_deg',)):
    """Out-of-fold residual against each feature, binned medians over the point cloud.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    features : `list` [`str`]
    resid : `array_like`
        [µm of equivalent camera-hexapod dz].
    extra_cols : `tuple` [`str`], optional
        Columns plotted beside the features though not fitted. ``altitude_deg`` is here to show
        the elevation dependence is already handled by the hexapod look-up table.

    Notes
    -----
    A flat binned-median trend means the model has taken out that feature's dependence; a
    surviving trend in a column that was *not* fitted, such as elevation, would be a missing term.
    """
    import matplotlib.pyplot as plt

    cols = list(features) + [c for c in extra_cols
                             if c in df.columns and c not in features]
    n = len(cols)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.3, 3.4 * nrow), squeeze=False)
    for i, c in enumerate(cols):
        ax = axes[i // ncol][i % ncol]
        x = df[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(resid)
        ax.plot(x[m], resid[m], '.', ms=1, alpha=0.15, color='0.5')
        if m.sum() > 200:
            q = np.linspace(1, 99, 16)
            edges = np.percentile(x[m], q)
            edges = np.unique(edges)
            idx = np.digitize(x[m], edges)
            xs, ys = [], []
            for k in range(len(edges) + 1):
                s = idx == k
                if s.sum() >= 30:
                    xs.append(np.median(x[m][s]))
                    ys.append(np.median(resid[m][s]))
            ax.plot(xs, ys, 'o-', color='tab:red', ms=4, lw=1.5)
        ax.axhline(0.0, color='k', lw=0.8)
        ax.set_xlabel(f'{c} [{FEATURE_UNITS.get(c, "?")}]', fontsize=8)
        ax.set_ylabel('out-of-fold residual\n[um equiv camhex dz]', fontsize=8)
        lo, hi = np.percentile(resid[m], [1, 99]) if m.sum() else (-1, 1)
        ax.set_ylim(lo, hi)
        tag = ' (not fitted)' if c not in features else ''
        ax.set_title(f'{c}{tag}', fontsize=9)
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('Out-of-fold residual against each feature; red is the binned median. '
                 'A flat trend means that dependence has been removed', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_band_residual(pdf, df, resid, baseline=None):
    """Per-band histogram of the out-of-fold residual from the band-independent fit.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carrying ``band``.
    resid : `array_like`
        [µm of equivalent camera-hexapod dz].
    baseline : `array_like`, optional
        The uncorrected response, drawn behind for scale [µm of equivalent camera-hexapod dz].
    """
    import matplotlib.pyplot as plt

    d = df.assign(resid=resid)
    if baseline is not None:
        d = d.assign(base=np.asarray(baseline, float))
    bands = [b for b in BAND_ORDER if (d.band == b).any()]
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.5), squeeze=False)
    lo, hi = np.nanpercentile(resid, [0.5, 99.5])
    for i, b in enumerate(bands[:6]):
        ax = axes[i // 3][i % 3]
        g = d[d.band == b]
        v = g.resid.to_numpy(float)
        v = v[np.isfinite(v)]
        if baseline is not None:
            bv = g.base.to_numpy(float)
            bv = bv[np.isfinite(bv)]
            ax.hist(bv, bins=60, range=(lo, hi), color='0.8',
                    label=f'uncorrected, nMAD {nmad(bv):.0f} um')
        ax.hist(v, bins=60, range=(lo, hi), color=BAND_COLORS.get(b, 'tab:blue'),
                alpha=0.85, label=f'corrected, nMAD {nmad(v):.0f} um')
        ax.axvline(np.median(v), color='k', ls='--', lw=1)
        ax.set_title(f'{b}: n = {len(v)}, median {np.median(v):+.0f} um', fontsize=9)
        ax.set_xlabel('out-of-fold residual [um equiv camhex dz]', fontsize=8)
        ax.set_ylabel('visits', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(bands), 6):
        axes[j // 3][j % 3].axis('off')
    fig.suptitle('Per-band residual of the single band-independent model. A per-band median '
                 'away from zero is a real filter-thickness focus offset', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_night_series(pdf, df, resid, day_obs, features):
    """One night's uncorrected and corrected residual against ``seq_num``, bands marked.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    resid : `array_like`
        [µm of equivalent camera-hexapod dz].
    day_obs : `int`
    features : `list` [`str`]

    Notes
    -----
    This is the page the whole exercise is judged on: the filter changes are drawn as vertical
    lines, and the corrected trace should cross them without a step.
    """
    import matplotlib.pyplot as plt

    d = df.assign(resid=resid)
    d = d[d.day_obs == int(day_obs)].sort_values('seq_num')
    if not len(d):
        print(f'no visits on {day_obs}, skipping the night-series page')
        return
    changes = d.seq_num[d.band.ne(d.band.shift(1))].to_numpy()[1:]
    panels = [('y', 'uncorrected v1(Trim) - v1(measured)\n[um equiv camhex dz]', 'tab:gray'),
              ('resid', 'band-independent model residual\n[um equiv camhex dz]', 'tab:blue')]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(13.3, 9.0), sharex=True)
    for ax, (col, ylabel, color) in zip(axes, panels):
        for b in [x for x in BAND_ORDER if (d.band == x).any()]:
            g = d[d.band == b]
            ax.plot(g.seq_num, g[col], '.', ms=3, color=BAND_COLORS.get(b, color), label=b)
        v = d[col].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.set_title(f'{ylabel.splitlines()[0]}: nMAD {nmad(v):.1f} um, '
                     f'span {np.ptp(v):.1f} um', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, ncol=6, loc='upper right')
        ax.grid(alpha=0.3)
    ax = axes[-1]
    for c, color in zip(features[:2], ('tab:red', 'tab:purple')):
        a = ax if c == features[0] else ax.twinx()
        a.plot(d.seq_num, d[c], '-', color=color, lw=1)
        a.set_ylabel(f'{c}\n[{FEATURE_UNITS.get(c, "?")}]', color=color, fontsize=8)
        a.tick_params(axis='y', labelcolor=color)
    ax.set_xlabel('seq_num')
    ax.grid(alpha=0.3)
    for a in axes:
        for s in changes:
            a.axvline(s, color='0.7', lw=0.8, ls=':')
    fig.suptitle(f'{day_obs}: the band-independent correction across filter changes '
                 f'(dotted lines). {len(d)} visits', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf(out_path, df, features, res, full, model, v1_per_um_dz, model_cmp, feat_cmp,
              band_tab, coef_tab, steps, night, leaky_nmad=None, variant=DEFAULT_VARIANT):
    """Assemble the thermal-model PDF.

    Parameters
    ----------
    out_path : `pathlib.Path`
    df : `pandas.DataFrame`
        From `load_target`.
    features : `list` [`str`]
    res : `dict`
        From `evaluate`, the night-grouped result.
    full : `dict`
        From `fit_full`.
    model : `str`
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm of hexapod dz].
    model_cmp, feat_cmp, band_tab, coef_tab : `pandas.DataFrame` or `None`
        Tables from `compare_models`, `compare_feature_sets`, `per_band_residual`,
        `coefficient_table`.
    steps : `dict`
        ``{label: filter_change_step result}``.
    night : `int`
        ``day_obs`` for the series page.
    leaky_nmad : `float`, optional
        Visit-level residual nMAD [µm of equivalent camera-hexapod dz], for the warning block.
    variant : `str`, optional
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    base = nmad(df.y.to_numpy(float))
    nights = np.sort(df.day_obs.unique())
    with PdfPages(out_path) as pdf:
        blocks = [
            ('What this document measures',
             'One band-independent thermal correction for the v-mode-1 Trim, fitted\n'
             'jointly on all thermal telemetry at once, replacing the sequential\n'
             'per-band truss-then-z-gradient chain. Because that chain fits a\n'
             'different slope per band, every filter change injects a step into the\n'
             'corrected residual; a single band-independent model has none by\n'
             'construction.'),
            ('The response variable',
             'v1_trim_meas = v1(Trim) + MEASURED_SIGN * v1(measured)\n'
             f'MEASURED_SIGN = {MEASURED_SIGN:+.0f}, i.e. Trim - MEASURED. The hexapod\n'
             'look-up-table (LUT) baseline is left out; it carries essentially the\n'
             'whole elevation dependence.\n'
             'Expressed throughout as equivalent camera-hexapod dz [um]:\n'
             f'   v1_dzequiv = v1_trim_meas / {v1_per_um_dz:.6e} per um'),
            ('Sample',
             f'visits  : {len(df)}\n'
             f'nights  : {len(nights)}  ({int(nights.min())} to {int(nights.max())})\n'
             f'variant : {variant}\n'
             f'dropped : the {len(LUT_EPOCH_OFFSET_NIGHTS)} nights running a different '
             'hexapod LUT\n'
             '          configuration (LUT_EPOCH_OFFSET_NIGHTS)'),
            ('Whole nights are held out, and this is not optional',
             'Within a night the thermal telemetry is near-constant: only 1.9% of the\n'
             'truss temperature variance is within-night, while 83.8% of the target\n'
             'variance is between nights. A visit-level split therefore lets a model\n'
             'identify the night from its temperature and memorise that night offset.\n'
             'Every number here uses GroupKFold grouped on day_obs.\n'
             + (f'Measured leak, same model and features:\n'
                f'  visit-level (LEAKY) : {leaky_nmad:6.1f} um equiv camhex dz\n'
                f'  night-grouped       : {res["nmad"]:6.1f} um equiv camhex dz\n'
                f'  ratio               : {res["nmad"] / leaky_nmad:6.2f} '
                '(dimensionless), i.e. the leaky\n'
                f'                        score is optimistic by '
                f'{res["nmad"] / leaky_nmad:.2f}x'
                if leaky_nmad else
                f'Worst-case leak factor from the model scan: {LEAK_FACTOR} (dimensionless), '
                f'boosted trees on 6\nfeatures. A Huber linear fit leaks far less.')),
            ('Result',
             f'model              : {model}\n'
             f'features ({len(features):d})       : ' + '\n'.rjust(22).join(features) + '\n'
             f'folds              : {res["n_splits"]} night-grouped\n'
             f'uncorrected nMAD   : {base:7.1f} um of equivalent camera-hexapod dz\n'
             f'out-of-fold nMAD   : {res["nmad"]:7.1f} um of equivalent camera-hexapod dz\n'
             f'fraction remaining : {res["nmad"] / base:7.3f} (dimensionless, corrected '
             'over uncorrected)\n'
             f'R2                 : {res["r2"]:7.3f} (dimensionless)\n'
             f'in-sample nMAD     : {full["nmad"]:7.1f} um (NOT a performance estimate)'),
            ('Filter-change step, the metric this exercise is about',
             'Median |step| in the residual between consecutive visits within a night\n'
             '[um of equivalent camera-hexapod dz]:\n'
             + '\n'.join(
                 f'  {k:24s} band change {v["median_change"]:7.1f} (n={v["n_change"]:5d}), '
                 f'same band {v["median_same"]:7.1f}, ratio {v["ratio"]:.2f}'
                 for k, v in steps.items())
             + '\n  ratio is dimensionless (band-change over same-band median step)'),
            ('Provenance',
             'per-visit  : aos/output/science_lut/science_lut.parquet\n'
             'this script : aos/code/science_lut/run_thermal_model.py\n'
             f'FAM commanded truss slope for comparison: {FAM_TRUSS_SLOPE:+.5f}\n'
             '            dimensionless v-mode-1 amplitude per deg C, from\n'
             '            code/correlations/run_dz14_truss.py'),
        ]
        page_text(pdf, 'A band-independent thermal correction for v-mode-1 Trim',
                  blocks,
                  subtitle=(f'produced {pd.Timestamp.today().date().isoformat()} by '
                            'code/science_lut/run_thermal_model.py'))

        tabs = []
        if model_cmp is not None and len(model_cmp):
            tabs.append(('Model comparison, night-grouped',
                         _table_text(model_cmp, ['model', 'resid_nmad', 'frac_of_baseline', 'r2'],
                                     ['s', '.1f', '.3f', '.3f'], [14, 10, 10, 7])
                         + f'\nresid_nmad [um of equivalent camera-hexapod dz]; '
                           f'frac_of_baseline and r2\ndimensionless; baseline nMAD {base:.1f} um.'
                           '\nThe thermal response is close to linear, so the tree models '
                           'lose:\nthey fit night-specific structure that does not transfer '
                           'to held-out nights.'))
        if feat_cmp is not None and len(feat_cmp):
            tabs.append(('Feature-set ablation, night-grouped',
                         _table_text(feat_cmp,
                                     ['groups', 'n_features', 'resid_nmad', 'r2'],
                                     ['s', 'd', '.1f', '.3f'], [30, 3, 10, 7])
                         + '\nresid_nmad [um of equivalent camera-hexapod dz]; r2 '
                           'dimensionless.'))
        if band_tab is not None and len(band_tab):
            tabs.append(('Per-band residual of the band-independent model',
                         _table_text(band_tab, ['band', 'n', 'median', 'resid_nmad'],
                                     ['s', 'd', '+.1f', '.1f'], [5, 7, 9, 10])
                         + '\nmedian and resid_nmad [um of equivalent camera-hexapod dz].'))
        if coef_tab is not None and len(coef_tab):
            tabs.append(('Coefficients, mean and scatter across folds',
                         _table_text(coef_tab, ['feature', 'mean', 'std'],
                                     ['s', '+.1f', '.1f'], [32, 10, 9])
                         + '\n[um of equivalent camera-hexapod dz per feature unit: '
                           'deg C for the\ntruss temperature, deg C per m for each M1M3 '
                           'gradient]'))
        if tabs:
            page_text(pdf, 'Model selection and coefficients', tabs)

        if res['coefs'] is not None:
            page_coefficients(pdf, res['coefs'], features, v1_per_um_dz)
        page_resid_vs_features(pdf, df, features, res['resid'])
        page_band_residual(pdf, df, res['resid'], baseline=df.y.to_numpy(float))
        page_night_series(pdf, df, res['resid'], night, features)


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--variant', default=DEFAULT_VARIANT,
                   help='optical-state variant to fit (default %(default)s)')
    p.add_argument('--in-dir', default=None,
                   help='directory holding science_lut.parquet; default aos/output/science_lut')
    p.add_argument('--out-dir', default=None, help='output directory; default --in-dir')
    p.add_argument('--v1-per-um-dz', type=float, default=None,
                   help='conversion [dimensionless v-mode-1 amplitude per um of hexapod dz]; '
                        'computed from aos_state via run_science_lut_report if omitted, which '
                        'needs lsst.ts.ofc')
    p.add_argument('--model', default='huber', choices=MODEL_NAMES,
                   help='estimator for the deliverable fit (default %(default)s, measured best '
                        'under a night-grouped split)')
    p.add_argument('--features', nargs='+', default=list(DEFAULT_FEATURES),
                   metavar='GROUP',
                   help='feature groups to fit; choose from '
                        + ', '.join(sorted(FEATURE_GROUPS))
                        + ' (default: %(default)s -- truss temperature plus the four M1M3 '
                          'thermal gradients, the best configuration measured)')
    p.add_argument('--n-splits', type=int, default=5,
                   help='number of night-grouped folds (default %(default)s)')
    p.add_argument('--night', type=int, default=20260706,
                   help='day_obs for the series page, where the filter-change steps are '
                        'visible (default %(default)s)')
    p.add_argument('--leaky-split', action='store_true',
                   help='ALSO evaluate with a visit-level split, which does NOT hold whole '
                        'nights out and so is not a performance estimate. The optimism is '
                        f'up to {LEAK_FACTOR}x (dimensionless) for boosted trees but only '
                        'about 1.04x for the Huber linear fit used here. Present only so the '
                        'size of that trap stays documented')
    p.add_argument('--keep-lut-epoch-offset-nights', action='store_true',
                   help=f'keep the {len(LUT_EPOCH_OFFSET_NIGHTS)} nights that ran a different '
                        'hexapod look-up-table configuration (LUT_EPOCH_OFFSET_NIGHTS in '
                        'run_science_lut.py). They are dropped by default')
    p.add_argument('--no-model-scan', action='store_true',
                   help='skip the model and feature-set comparison tables, which refit the '
                        'whole cross-validation once per row')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)
    verbose = not a.quiet

    aos_dir = pathlib.Path(__file__).resolve().parents[2]
    in_dir = pathlib.Path(a.in_dir) if a.in_dir else aos_dir / 'output' / 'science_lut'
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_per_um_dz = a.v1_per_um_dz
    if v1_per_um_dz is None:
        from run_science_lut_report import v1_per_um_dz_value
        v1_per_um_dz = v1_per_um_dz_value(verbose=verbose)

    features = resolve_features(a.features)
    # The ablation table needs every candidate column present in one frame, so load the union
    # and select per row rather than re-reading the parquet for each feature set.
    scan_sets = [('truss',), ('truss', 'camtemp'), ('truss', 'zgrad'), ('truss', 'grads'),
                 ('truss', 'grads', 'camtemp'), ('truss', 'grads', 'airtemps'),
                 ('truss', 'grads', 'wind'), ('truss', 'grads', 'elev')]
    union = list(dict.fromkeys(features + resolve_features(
        sorted({g for s in scan_sets for g in s}))))
    df, _ = load_target(in_dir, variant=a.variant, v1_per_um_dz=v1_per_um_dz,
                        features=union, drop_lut_epoch=not a.keep_lut_epoch_offset_nights,
                        verbose=verbose)

    if verbose:
        print()
    res = evaluate(df, features, model=a.model, n_splits=a.n_splits, verbose=verbose)
    leaky_nmad = None
    if a.leaky_split:
        leaky_nmad = evaluate(df, features, model=a.model, n_splits=a.n_splits,
                              leaky=True, verbose=verbose)['nmad']

    coef_tab = None
    if res['coefs'] is not None:
        if verbose:
            print()
        coef_tab = coefficient_table(res['coefs'], features, v1_per_um_dz, verbose=verbose)

    if verbose:
        print()
    band_tab = per_band_residual(df, res['resid'], verbose=verbose)

    model_cmp = feat_cmp = None
    if not a.no_model_scan:
        if verbose:
            print()
        model_cmp = compare_models(df, features, n_splits=a.n_splits, verbose=verbose)
        if verbose:
            print()
        feat_cmp = compare_feature_sets(df, scan_sets, model=a.model, n_splits=a.n_splits,
                                        verbose=verbose)

    if verbose:
        print()
    full = fit_full(df, features, model=a.model, verbose=verbose)

    if verbose:
        print('\nfilter-change step, median |step| between consecutive visits within a night '
              '[um of equivalent camera-hexapod dz]')
        print('  the baseline that matters is the PER-BAND chain, not the uncorrected '
              'response: the\n  per-band coefficients are what inject a step at a filter '
              'change, and a correction of\n  any kind raises the same-band step by adding '
              'the model\'s own visit-to-visit noise.')
    d = df.assign(resid=res['resid'])
    steps = {
        'uncorrected': filter_change_step(d, 'y', verbose=verbose, label='uncorrected'),
    }
    # The per-band chain, read from its published results parquet, is the like-for-like
    # baseline. Absent when that document has not been built, in which case the row is
    # simply omitted rather than substituting the uncorrected response for it.
    per_band_path = in_dir / f'science_lut_results_{PER_BAND_CHAIN_TAG}.parquet'
    if per_band_path.exists():
        pb = pd.read_parquet(per_band_path)
        if PER_BAND_CHAIN_COL in pb.columns:
            pb = pb[['visit_id', 'day_obs', 'band', PER_BAND_CHAIN_COL]].copy()
            pb['seq_num'] = pb.visit_id % 100000
            steps['per-band truss-grad-elev chain'] = filter_change_step(
                pb, PER_BAND_CHAIN_COL, verbose=verbose,
                label='per-band truss-grad-elev chain')
    elif verbose:
        print(f'  (per-band chain baseline unavailable: {per_band_path.name} not built)')
    steps[f'{a.model} band-independent'] = filter_change_step(
        d, 'resid', verbose=verbose, label=f'{a.model} band-independent')

    pred_path = out_dir / 'thermal_model_predictions.parquet'
    pd.DataFrame({'visit_id': df.visit_id.to_numpy(),
                  'day_obs': df.day_obs.to_numpy(),
                  'band': df.band.to_numpy(),
                  'y_dzequiv_um': df.y.to_numpy(float),
                  'pred_oof_dzequiv_um': res['pred'],
                  'resid_oof_dzequiv_um': res['resid'],
                  'pred_full_dzequiv_um': full['pred'],
                  'resid_full_dzequiv_um': full['resid']}).to_parquet(pred_path)

    coef_path = out_dir / 'thermal_model_coeffs.parquet'
    if coef_tab is not None:
        meta = coef_tab.assign(model=a.model, n_splits=a.n_splits, variant=a.variant,
                               intercept_um=full['intercept'],
                               oof_resid_nmad_um=res['nmad'],
                               baseline_nmad_um=nmad(df.y.to_numpy(float)),
                               v1_per_um_dz=v1_per_um_dz)
        if full['coef'] is not None:
            meta['full_fit'] = full['coef']
        meta.to_parquet(coef_path)

    pdf_path = out_dir / 'thermal_model.pdf'
    build_pdf(pdf_path, df, features, res, full, a.model, v1_per_um_dz, model_cmp, feat_cmp,
              band_tab, coef_tab, steps, a.night, leaky_nmad=leaky_nmad, variant=a.variant)

    print(f'\nwrote {pdf_path}')
    print(f'wrote {pred_path}')
    if coef_tab is not None:
        print(f'wrote {coef_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
