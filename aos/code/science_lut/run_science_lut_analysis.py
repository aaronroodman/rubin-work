#!/usr/bin/env python3
"""A focus look-up table from science exposures: the analysis document for the science_lut study.

Fits the Rubin telescope's uniform-defocus error against thermal telemetry and writes one PDF
describing the result. The response is the v-mode-1 amplitude of the commanded Trim minus the
optical state measured at the four Corner Wavefront Sensors (CWFS), expressed as equivalent
hexapod dz [µm], and the model is one band-independent Huber robust linear fit on the Telescope
Mount Assembly (TMA) truss temperature plus the four M1M3 thermal gradients, scored on whole
held-out nights.

Reads ``output/science_lut/science_lut.parquet``, built by ``run_science_lut.py`` -- the only
script in the study that touches the Consolidated Database (ConsDB) or the Engineering Facility
Database (EFD). This script needs no network access.

Usage
-----
Run from ``aos/``::

    python code/science_lut/run_science_lut_analysis.py
    python code/science_lut/run_science_lut_analysis.py --all-nights
    python code/science_lut/run_science_lut_analysis.py --day-obs 20260706 20260604
    python code/science_lut/run_science_lut_analysis.py --features truss grads camtemp
    python code/science_lut/run_science_lut_analysis.py --no-model-scan

Writes, all in ``output/science_lut/``:

* ``science_lut_analysis.pdf`` -- the document;
* ``science_lut_model.parquet`` -- the fitted coefficients, per fold and full-sample, with units;
* ``science_lut_predictions.parquet`` -- out-of-fold prediction and residual per visit;
* ``science_lut_nights.parquet`` -- per-night elevation slope and offset at ``--ref-elev-deg``.

Notes
-----
Whole nights are held out by `sklearn.model_selection.GroupKFold` on ``day_obs``. Within a night
the thermal telemetry drifts slowly, so consecutive visits are near-duplicates in feature space:
only 1.9% of the truss temperature's variance is within-night while 83.8% of the response
variance is between nights. A visit-level split therefore lets a model identify the night from
its temperature and memorise that night's offset. Measured with boosted trees on 6 features the
residual normalized median absolute deviation (nMAD) is 26.5 µm of equivalent hexapod dz
visit-level against 83.1 µm night-grouped, a factor of `LEAK_FACTOR` (dimensionless). For the
Huber linear fit used here the factor is about 1.04, because five coefficients cannot memorise a
night. ``--leaky-split`` reproduces the visit-level number, with a warning.

The 8 nights in `LUT_EPOCH_OFFSET_NIGHTS` are dropped by default: their hexapod look-up-table
(LUT) term sits far below the rest at the same elevation, so a different LUT configuration was
loaded at the time. ``--keep-lut-epoch-offset-nights`` keeps them.

Visits missing a feature are imputed at the training-fold median rather than dropped, because the
truss temperature reaches only about 89% of visits while the M1M3 gradients reach 99.9%. A visit
with no truss temperature at all is dropped instead of imputed: the imputer would substitute the
run-wide median, which is a per-night bias of order 1 deg C -- about 124 µm of equivalent hexapod
dz -- rather than noise.
"""
import argparse
import pathlib
import sys
import textwrap

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))                                        # repo root -> common/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # aos/code
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # this study

from common.utils import nmad                                          # noqa: E402
from run_science_lut import (BAND_COLORS, FAM_TRUSS_SLOPE,             # noqa: E402
                            LUT_EPOCH_OFFSET_NIGHTS, MEASURED_SIGN,
                            huber_fit, theilsen_fit)

# --------------------------------------------------------------------------- constants

#: Default ``optical_state`` variant id, matching the rest of the study.
DEFAULT_VARIANT = 'v50_34__batoid__consdb_v1'

#: Band order for every multi-panel page and table, bluest first.
BAND_ORDER = ('u', 'g', 'r', 'i', 'z', 'y')

#: Named feature groups, selectable by ``--features``. ``truss`` and ``grads`` together are the
#: default: mechanistically motivated and, measured, the best configuration. The rest are kept
#: reachable so the negative results stay reproducible.
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

#: Units per feature, for every table, axis label and equation line. Physical units are
#: mandatory here -- a coefficient of -3335 means nothing without "µm of equivalent hexapod dz
#: per (deg C per m)".
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
#: (83.1 µm against 26.5 µm of equivalent hexapod dz). Quoted only as the worst case seen: the
#: leak is a property of the MODEL, not of the split alone. For the Huber linear fit used here it
#: is about 1.04 (dimensionless). ``--leaky-split`` prints whichever ratio the run measures.
LEAK_FACTOR = 3.1

#: The response column, in µm of equivalent hexapod dz, after the thermal correction.
YCOL = 'y'

#: Nights per page of the rising/falling panels, as 4 columns by 3 rows.
NIGHTS_PER_PAGE = 12

#: Panel height for those pages [µm of equivalent hexapod dz], centred on each night's own
#: median. The per-night 1st-to-99th percentile span is 769 µm at the median night and 1174 µm at
#: the 95th percentile of nights, so this holds nearly every visit while keeping one common scale
#: for comparing slopes panel to panel.
PANEL_YSPAN = 1200.0

#: Colours for the elevation direction, shared by every page so the legend need not repeat.
DIR_COLOUR = {'up': 'tab:red', 'down': 'tab:blue', 'flat': '0.6'}

#: Minimum visits for a per-night slope, and for one direction of one night.
MIN_VISITS_NIGHT = 40
MIN_VISITS_LEG = 25

#: Visits in the centred rolling median of elevation used to label rising/falling legs, and the
#: minimum |d(elevation)| per visit [deg] that counts as slewing rather than tracking.
DIRECTION_WINDOW = 21
DIRECTION_DEADBAND = 0.02

#: Plotted window for the uncorrected measured v-mode-1 amplitude [µm of equivalent hexapod dz].
#: Every band's 0.5th-to-99.5th percentile falls inside ±100 µm, while a handful of visits reach
#: ±900 µm; the robust statistics use every visit regardless.
MEAS_LIM = (-100.0, 100.0)

#: Histogram bins for the measured amplitude, spanning `MEAS_LIM`.
MEAS_BINS = 40

#: Plotted window for the per-visit elevation change [deg] against the previous seq_num. The
#: 1st-to-99th percentile is -6.3 to +7.4 deg with tails to ±55 deg from slews between fields;
#: the correlation statistics use every retained visit regardless.
DELTA_ELEV_LIM = (-12.0, 12.0)

#: Plotted window for the per-night elevation slope of the thermally corrected response [µm of
#: equivalent hexapod dz per deg]. With the look-up-table term excluded the nights sit at zero
#: (median -0.072 µm per deg); autoscaling instead lets a single failed fit near +50 compress
#: every other night into a few pixels. The fits and reported statistics use every night.
SLOPE_LIM = (-10.0, 10.0)

#: Histogram bins for the all-points slope, spanning the slope window.
SLOPE_BINS = 10

#: Polynomial orders in elevation fitted to the look-up-table term alone. The first is drawn; the
#: rest are reported as residual nMAD so any curvature is quantified.
LUT_ELEV_ORDERS = (1, 2, 3)

#: Elevation at which each per-night fit is evaluated to give that night's offset [deg]. The
#: sample median elevation is 61.60 deg, so a round 60 deg sits inside the bulk of every night.
#: The fitted intercept at 0 deg is an extrapolation roughly 60 deg outside the data and is
#: therefore strongly anti-correlated with the slope -- Pearson r -0.705 against +0.378 for the
#: offset -- which is exactly what a night-to-night comparison must avoid.
REF_ELEV_DEG = 60.0

#: Plotted window for the per-night offset [µm of equivalent hexapod dz]. The 5th-to-95th
#: percentile of the nights is -113.0 to +71.0 µm, while three nights reach -1622.6, -1492.9 and
#: +843.8 µm -- the first two have 47 and 171 visits and within-night residual nMAD near 80 µm,
#: the third a fitted slope of +49.4 µm per deg, so all three are failed or barely-determined
#: fits rather than real offsets. Each panel reports how many nights fall outside, and the
#: medians, nMADs and trends use every night regardless.
OFFSET_LIM = (-250.0, 250.0)

#: Elevation window the polynomial is fitted over [deg]. Outside it the sampling is sparse and a
#: high-order polynomial swings, so the fit is restricted and the plot says so.
ELEV_FIT_RANGE = (30.0, 80.0)

#: Polynomial orders scanned for the elevation dependence (dimensionless).
ELEV_ORDERS = (1, 2, 3, 4)

#: Fixed y-limits for the six-panel grid pages [µm of equivalent hexapod dz]. A pooled
#: 1st-to-99th percentile clips the tails of the widest band; this window is wide enough to hold
#: nearly every visit in any band, so panels can be compared directly. Each panel reports the
#: count falling outside, and the fits always use every point.
GRID_YLIM_DZEQUIV = (-2000.0, 4000.0)

#: M1M3 thermal-gradient columns and their labels [°C/m].
GRAD_COLS = (('m1m3_z_gradient_c_per_m', 'M1M3 z thermal gradient'),
             ('m1m3_radial_gradient_c_per_m', 'M1M3 radial thermal gradient'),
             ('m1m3_x_gradient_c_per_m', 'M1M3 x thermal gradient'),
             ('m1m3_y_gradient_c_per_m', 'M1M3 y thermal gradient'))

#: Nights drawn as a visit-by-visit time series by default. One PDF page each.
DAY_OBS_SERIES = [20260706]

#: Per-axis v-mode-1 response, filled by `v1_per_um_dz_value` and keyed by degree-of-freedom
#: (DOF) index: 5 is the camera-hexapod dz axis, 0 the M2-hexapod dz axis [dimensionless
#: v-mode-1 amplitude per µm]. Kept so the document can state what the mean of the two means
#: physically without hardcoding the numbers.
V1_PER_UM_DZ_AXES = {}

#: Model names available to ``--model`` and to the comparison table.
MODEL_NAMES = ('huber', 'ridge', 'spline_huber', 'histgb', 'rf')

#: Feature-group sets compared in the ablation table, from the truss temperature alone up to the
#: deliverable set plus one group the deliverable set leaves out.
ABLATION_SETS = (('truss',), ('truss', 'zgrad'), ('truss', 'grads'),
                 ('truss', 'grads', 'elev'), ('truss', 'grads', 'wind'))

#: Fixed y-limits for the residual-against-modulator pages [µm of equivalent hexapod dz]. The
#: residual is centred near zero, so a window narrower than `GRID_YLIM_DZEQUIV` is readable.
RESID_YLIM_DZEQUIV = (-600.0, 600.0)

#: Second-order factors tested for a residual dependence the five thermal channels do not carry,
#: as ``(column, label, unit)``. Drawn only by ``--modulators``; a column absent from the assembled
#: table is skipped rather than raising, since several were added part-way through the run.
MODULATOR_CANDIDATES = (
    ('cum_hex_dz_um', 'cumulative |hexapod dz| since night start', 'um'),
    ('recent_hex_dz_um', 'trailing-window |hexapod dz|', 'um'),
    ('n_moves_night', 'commanded hexapod moves so far this night', 'count'),
    ('into_wind_deg', 'wind direction relative to azimuth (0 = into the wind)', 'deg'),
    ('wind_speed_ms', 'weather-station wind speed', 'm per s'),
    ('wind_inside_maxmag', 'salIndex 110 sonic anemometer maximum speed', 'm per s'),
    ('turb126_speed_mag_ms', 'salIndex 126 sonic anemometer speed magnitude', 'm per s'),
    ('cam_AverageTemp', 'camera average temperature', 'deg C'),
    ('outside_temp', 'outside air temperature', 'deg C'),
)


def day_obs_to_date(day_obs):
    """Convert an integer ``day_obs`` to a calendar date.

    Parameters
    ----------
    day_obs : `array_like` [`int`]
        Nights as ``YYYYMMDD``.

    Returns
    -------
    dates : `pandas.DatetimeIndex`
    """
    return pd.to_datetime(np.asarray(day_obs).astype(int).astype(str), format='%Y%m%d')


def v1_per_um_dz_value(dof_set='all_50', n_modes=34, verbose=True):
    """v-mode-1 amplitude per µm of hexapod dz [per µm].

    Parameters
    ----------
    dof_set : `str`, optional
        ts_ofc degree-of-freedom (DOF) set name for the DOF to v-mode projection.
    n_modes : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    v1_per_um_dz : `float`
        Mean magnitude over the camera-hexapod (DOF 5) and M2-hexapod (DOF 0) dz axes
        [dimensionless v-mode-1 amplitude per µm].

    Notes
    -----
    Derived here rather than copied, so the document's conversion factor cannot drift from the
    stored v-modes.

    The projection is `aos_state.vmodes_from_dofs` through `aos_state.make_state_estimator` --
    the single sanctioned v-mode engine, the basis the measured state's v-modes are reported in
    and the commanded terms are stored in.

    v1 is the camera-hexapod dz (DOF 5) and M2-hexapod dz (DOF 0) combination plus small
    mirror-bending terms, and is the same mode in both schemes, so this factor comes out equal to
    5 decimal places at ``standard_22``/12 and ``all_50``/34.

    What the mean magnitude means physically: the two coefficients carry the **same sign**, so
    the two axes add rather than cancel, and their sum over their mean is 2.00000 (dimensionless)
    to five decimal places. Moving 0.5 µm on each hexapod -- 1 µm of **total** dz travel --
    therefore produces exactly this factor's worth of v1. So a value reported in these units is
    µm of total defocus travel, split evenly between the camera and M2 hexapods, and not µm of
    camera-hexapod motion with M2 held still. Per unit v-mode-1 amplitude that is 1110.1 µm of
    total travel shared, 555.0 µm on each hexapod, against 1121.8 µm if the camera hexapod moves
    alone -- the two differ by only 1.1% (dimensionless), because the two coefficients agree to
    2.1%.
    """
    import aos_state
    se = aos_state.make_state_estimator(dof_set=dof_set, n_modes=n_modes)
    c = {}
    for k in (0, 5):
        d = np.zeros(50)
        d[k] = 1.0
        c[k] = float(aos_state.vmodes_from_dofs(d, se, n_modes=n_modes)[0, 0])
    val = 0.5 * (abs(c[5]) + abs(c[0]))
    # Keep the two per-axis coefficients reachable, so the document can state what the mean
    # means physically without hardcoding numbers that would drift with the scheme.
    V1_PER_UM_DZ_AXES.update({5: c[5], 0: c[0]})
    if verbose:
        print(f'v1 per um camera-hexapod dz (DOF 5) = {c[5]:+.7e} per um')
        print(f'v1 per um M2-hexapod dz     (DOF 0) = {c[0]:+.7e} per um')
        print(f'mean magnitude = {val:.5e} per um; axes agree to '
              f'{100 * abs(c[5] - c[0]) / val:.1f}% (dimensionless)')
        # Same sign on both axes, so sum/mean = 2 exactly: the mean magnitude is what 1 um of
        # total dz travel produces when it is split evenly between the two hexapods.
        print(f'  the two axes share a sign, so their sum over their mean is '
              f'{abs(c[5] + c[0]) / val:.5f} (dimensionless): this factor is 1 um of TOTAL '
              f'dz travel, 0.5 um on each hexapod')
    return val


def _unit_reading_text(v1_per_um_dz):
    """Prose explaining what the dz-equivalent unit means physically.

    Parameters
    ----------
    v1_per_um_dz : `float`
        The conversion in use [dimensionless v-mode-1 amplitude per µm of hexapod dz].

    Returns
    -------
    text : `str`
        Plain-text paragraph, using the per-axis coefficients from `V1_PER_UM_DZ_AXES` when
        `v1_per_um_dz_value` has been called and a shorter form when it has not.
    """
    # 1/v1_per_um_dz is the TOTAL travel per unit v1 when the two hexapods share it equally,
    # i.e. half that figure on each axis.
    shared_total = 1.0 / v1_per_um_dz
    if len(V1_PER_UM_DZ_AXES) == 2:
        c5, c0 = V1_PER_UM_DZ_AXES[5], V1_PER_UM_DZ_AXES[0]
        cam_alone = 1.0 / abs(c5)
        return (f'The two dz axes carry the SAME sign, so they add: v1 per um is\n'
                f'{c5:+.6e} for the camera hexapod (DOF 5) and {c0:+.6e} for\n'
                f'the M2 hexapod (DOF 0), and their sum over their mean is\n'
                f'{abs(c5 + c0) / v1_per_um_dz:.5f} (dimensionless). Moving 0.5 um on each\n'
                f'hexapod -- 1 um of TOTAL dz travel -- therefore gives exactly the\n'
                f'conversion factor above.\n'
                f'So a value in these units is um of total defocus travel split evenly\n'
                f'between the two hexapods, NOT um of camera-hexapod motion with M2 held\n'
                f'still. Per unit v-mode-1 amplitude: {shared_total:.1f} um of total travel\n'
                f'shared ({0.5 * shared_total:.1f} um on each hexapod), against\n'
                f'{cam_alone:.1f} um if the camera hexapod moves alone.')
    return ('The conversion is the mean magnitude of the camera-hexapod (DOF 5) and\n'
            'M2-hexapod (DOF 0) dz coefficients. Those two carry the same sign and are\n'
            'nearly equal, so the mean is what 1 um of TOTAL dz travel produces when it is\n'
            'split evenly between the two hexapods. A value in these units is therefore um\n'
            f'of total defocus travel: {shared_total:.1f} um of total travel shared\n'
            f'({0.5 * shared_total:.1f} um on each hexapod).')


# --------------------------------------------------------------------------- robust fits

def robust_poly(x, y, order, min_n=20):
    """Huber polynomial fit.

    Parameters
    ----------
    x, y : `array_like`
        Paired values, each in a single unit.
    order : `int`
        Polynomial order (dimensionless); 1 is a straight line.
    min_n : `int`, optional
        Return None below this many finite pairs.

    Returns
    -------
    res : `dict` or `None`
        ``coef`` (highest power first, as `numpy.polyval` wants), ``n``, ``resid_nmad`` in
        y-units, and ``predict``, a callable on x-units.

    Notes
    -----
    The design matrix is built on a centred and scaled predictor, ``(x - x0) / xs``, because a
    raw fourth power of an elevation in deg spans eight orders of magnitude and the normal
    equations lose conditioning. The returned ``predict`` undoes the scaling, so callers work in
    the original units.
    """
    import statsmodels.api as sm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return None
    x, y = x[m], y[m]
    if np.ptp(x) == 0.0:
        return None
    x0, xs = float(np.mean(x)), float(np.std(x)) or 1.0
    u = (x - x0) / xs
    X = np.column_stack([u ** k for k in range(1, order + 1)])
    r = sm.RLM(y, sm.add_constant(X), M=sm.robust.norms.HuberT()).fit()
    p = np.asarray(r.params, dtype=float)

    def predict(xq):
        uq = (np.asarray(xq, dtype=float) - x0) / xs
        return p[0] + sum(p[k] * uq ** k for k in range(1, order + 1))

    return dict(coef=p, n=int(x.size), order=int(order),
                resid_nmad=float(nmad(y - predict(x))), predict=predict)


def scan_poly_order(x, y, orders=ELEV_ORDERS, min_n=20):
    """Robust residual scatter for each polynomial order.

    Parameters
    ----------
    x, y : `array_like`
    orders : `sequence` [`int`], optional
    min_n : `int`, optional

    Returns
    -------
    out : `dict`
        Order (`int`) to ``resid_nmad`` in y-units, omitting orders that could not be fitted.
    """
    out = {}
    for o in orders:
        r = robust_poly(x, y, o, min_n=min_n)
        if r is not None:
            out[o] = r['resid_nmad']
    return out


def huber_slope(x, y, min_n=MIN_VISITS_LEG):
    """Robust straight-line fit, returning the slope and its standard error.

    Parameters
    ----------
    x, y : `array_like`
        Predictor and response; `x` in deg of elevation, `y` in µm of equivalent hexapod dz.
    min_n : `int`, optional
        Return None below this many finite pairs.

    Returns
    -------
    out : `dict` or `None`
        ``n``, ``slope`` and ``slope_err`` [response unit per deg], ``intercept``,
        ``pearson_r``, ``spearman_rho``, ``resid_nmad``, ``elev_min``, ``elev_max``. None if
        under-determined or if elevation spans under 5 deg, where a slope is an extrapolation
        rather than a measurement.
    """
    import statsmodels.api as sm
    from scipy import stats

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return None
    x, y = x[ok], y[ok]
    if np.ptp(x) < 5.0:
        return None
    X = sm.add_constant(x)
    try:
        res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    except Exception:
        return None
    resid = y - res.predict(X)
    return dict(n=int(ok.sum()), slope=float(res.params[1]),
                slope_err=float(res.bse[1]), intercept=float(res.params[0]),
                pearson_r=float(stats.pearsonr(x, y)[0]),
                spearman_rho=float(stats.spearmanr(x, y)[0]),
                resid_nmad=float(nmad(resid)),
                elev_min=float(x.min()), elev_max=float(x.max()))


def label_direction(df, window=DIRECTION_WINDOW, deadband=DIRECTION_DEADBAND):
    """Label each visit as taken on a rising or falling elevation leg.

    Parameters
    ----------
    df : `pandas.DataFrame`
        One night, any order; needs ``obs_start_mjd`` and ``altitude_deg`` [deg].
    window : `int`, optional
        Visits in the centred rolling median of elevation.
    deadband : `float`, optional
        Minimum |d(elevation)| per visit [deg] to count as slewing.

    Returns
    -------
    direction : `pandas.Series`
        ``'up'``, ``'down'`` or ``'flat'``, indexed like `df`.

    Notes
    -----
    The rolling median is what makes this meaningful. Consecutive visits are ~0.7 min apart with
    sub-2 deg steps whose raw sign alternates while tracking, so a per-visit difference would
    report hundreds of direction changes per night instead of the few tens of real elevation legs.
    """
    d = df.sort_values('obs_start_mjd')
    sm_el = (d['altitude_deg'].rolling(window, center=True, min_periods=3)
             .median())
    de = sm_el.diff()
    out = pd.Series('flat', index=d.index, dtype=object)
    out[de > deadband] = 'up'
    out[de < -deadband] = 'down'
    return out.reindex(df.index)


# --------------------------------------------------------------------------- the model

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
        so a fold never sees a NaN and the imputation is fitted inside the fold rather than on the
        whole sample.

    Notes
    -----
    ``huber`` is the default and the recommendation: measured best under a night-grouped split on
    the default feature set, and its coefficients read directly in µm of equivalent hexapod dz per
    feature unit. The tree models are kept for the comparison table, where they lose -- they fit
    night-specific structure that does not transfer across held-out nights.
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


def _linear_coefficients(fitted, features):
    """Slopes of a fitted linear pipeline in physical units, or None.

    Parameters
    ----------
    fitted : `sklearn.pipeline.Pipeline`
    features : `list` [`str`]

    Returns
    -------
    coef : `numpy.ndarray` or `None`
        One coefficient per feature, in µm of equivalent hexapod dz per feature unit. None when
        the final estimator is not linear, or when a spline expansion means the coefficients no
        longer map one-to-one onto features.

    Notes
    -----
    Only the slopes are un-scaled here. The intercept needs the additional
    ``- coef_physical . scaler.mean_`` shift, which `_physical_intercept` applies; see the
    note there for why the two cannot be un-scaled the same way.
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
    coef_physical : `numpy.ndarray` or `None`
        Slopes already divided by ``scaler.scale_``, from `_linear_coefficients`.

    Returns
    -------
    intercept : `float`
        Response at zero in every feature [µm of equivalent hexapod dz], or NaN when the
        estimator has no intercept or the slopes are unavailable.

    Notes
    -----
    `sklearn.preprocessing.StandardScaler` centres as well as scales, so the estimator's own
    ``intercept_`` is the response at the training-fold **mean**, in standardized space. The
    slopes need only ``/ scaler.scale_``, but the intercept additionally needs
    ``- coef_physical . scaler.mean_`` -- without it the whole curve is offset by the model's
    prediction at the feature means, about 1500 µm of equivalent hexapod dz on this sample.
    `verify_equation` asserts the shifted value reproduces `sklearn`'s own ``predict``.
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
    """Assert the physical equation reproduces the pipeline's own prediction.

    Parameters
    ----------
    fitted : `sklearn.pipeline.Pipeline`
    df : `pandas.DataFrame`
        Carrying the feature columns.
    features : `list` [`str`]
    intercept : `float`
        [µm of equivalent hexapod dz].
    coef : `numpy.ndarray`
        Per feature [µm of equivalent hexapod dz per feature unit].
    tol : `float`, optional
        Maximum allowed disagreement [µm of equivalent hexapod dz].

    Returns
    -------
    max_abs_diff : `float`
        Largest |predict - (intercept + X . coef)| over the visits with no missing feature
        [µm of equivalent hexapod dz].

    Raises
    ------
    AssertionError
        If the two disagree by more than `tol`, which means the intercept or the slopes are not
        in physical units and the equation printed in the document would be wrong.

    Notes
    -----
    Restricted to visits with every feature present, because the pipeline imputes a missing
    feature at the training median while the hand-built equation cannot know that value. Those
    visits agree by construction once the present ones do.
    """
    X = df[features].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    if not ok.any():
        return float('nan')
    hand = intercept + X[ok] @ np.asarray(coef, float)
    ref = fitted.predict(X[ok])
    d = float(np.max(np.abs(hand - ref)))
    assert d < tol, (f'the physical equation disagrees with the fitted pipeline by {d:.3e} um '
                     f'of equivalent hexapod dz over {int(ok.sum())} visits, tolerance '
                     f'{tol:.0e}; the intercept or the slopes are not in physical units')
    return d


# --------------------------------------------------------------------------- data

def load_target(in_dir, variant=DEFAULT_VARIANT, v1_per_um_dz=None, features=None,
                extra_cols=(), drop_lut_epoch=True, verbose=True):
    """Read the ``Trim - measured`` response and its telemetry features per visit.

    Parameters
    ----------
    in_dir : `pathlib.Path`
        Directory holding ``science_lut.parquet``.
    variant : `str`, optional
        ``optical_state`` variant id.
    v1_per_um_dz : `float`
        Dimensionless v-mode-1 amplitude per µm of hexapod dz, used to express every amplitude as
        an equivalent hexapod motion.
    features : `list` [`str`], optional
        Feature columns to carry through. Missing columns raise.
    extra_cols : `sequence` [`str`], optional
        Further columns to carry for the ablation and modulator pages, which fit them without the
        model using them. A column absent from the table is skipped rather than raising, since
        several telemetry channels start part-way through the run.
    drop_lut_epoch : `bool`, optional
        Drop `LUT_EPOCH_OFFSET_NIGHTS`, the default.
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit_id``, ``day_obs``, ``seq_num``, ``band``, ``altitude_deg`` [deg],
        ``obs_start_mjd`` [d], the commanded and measured v-mode-1 terms, the requested features,
        and ``y`` -- the response ``v1(Trim) - v1(measured)`` in µm of equivalent hexapod dz.
    features : `list` [`str`]
        The resolved feature list, so a caller need not repeat `resolve_features`.

    Notes
    -----
    The hexapod look-up-table (LUT) baseline is deliberately left out of the response. The LUT
    carries essentially the whole elevation dependence and about 37x the measured term's scatter,
    so this response is what the closed loop and the wavefront sensors do on their own.
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

    # obs_start_mjd and v1_lut are carried for the elevation and measured-state pages; both are
    # optional in older builds of science_lut.parquet, so keep whichever are present.
    optional = [c for c in ('obs_start_mjd', 'v1_lut', 'v1_lut_trim') if c in b.columns]
    optional += [c for c in extra_cols if c in b.columns and c not in optional]
    keep = (['visit_id', 'day_obs', 'seq_num', 'band', 'altitude_deg', 'v1_trim', 'v1_meas']
            + optional + [c for c in features if c != 'altitude_deg'])
    df = b[list(dict.fromkeys(keep))].dropna(subset=['v1_trim', 'v1_meas']).copy()
    df['y'] = (df.v1_trim + MEASURED_SIGN * df.v1_meas) / v1_per_um_dz
    # The measured state alone, for the per-band pages; same unit as the response.
    df['meas_dzequiv'] = df.v1_meas / v1_per_um_dz
    if 'v1_lut' in df.columns:
        df['lut_dzequiv'] = df.v1_lut / v1_per_um_dz
    df = df.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)

    # The truss temperature is the primary regressor, so a visit without one must be dropped and
    # not imputed: the imputer substitutes the run-wide median, which is a per-night bias of order
    # 1 deg C -- about 124 um of equivalent hexapod dz -- rather than noise. Scattered
    # single-exposure gaps are already filled upstream by common.efd_db.interpolate_within_night,
    # so what survives here sits hours outside its own night's valid span.
    if 'truss_temp_mean_c' in df.columns:
        no_truss = df.truss_temp_mean_c.isna()
        if no_truss.any():
            if verbose:
                by_night = df.loc[no_truss, 'day_obs'].value_counts()
                worst = ', '.join(f'{int(d)} n={int(k)}' for d, k in by_night.head(3).items())
                print(f'no truss temperature after in-night interpolation: dropped '
                      f'{int(no_truss.sum())} visits on {len(by_night)} nights (worst: {worst})')
            df = df[~no_truss].reset_index(drop=True)

    if verbose:
        print(f'variant {variant}: {len(df)} visits over {df.day_obs.nunique()} nights '
              f'(from {n_all} visits / {nights_all} nights before cuts)')
        print(f'response v1(Trim) - v1(measured): median {df.y.median():+.1f} um, '
              f'nMAD {nmad(df.y.to_numpy(float)):.1f} um of equivalent hexapod dz')
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
        Split on visits instead of nights. Only for reproducing the leak; the returned numbers are
        optimistic -- by up to `LEAK_FACTOR` for a high-capacity model, and by about 1.04
        (dimensionless) for the Huber linear fit -- and must not be quoted as performance.
    verbose : `bool`, optional

    Returns
    -------
    res : `dict`
        ``pred`` (out-of-fold prediction [µm of equivalent hexapod dz]), ``resid``, ``coefs``
        (per-fold coefficient array, or None for the tree models), ``nmad``, ``r2``, ``n_splits``,
        ``grouped``.

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
        kind = 'visit-level (LEAKY)' if leaky else 'night-grouped on day_obs'
        print(f'{model}, {n_splits}-fold {kind}: residual nMAD {res["nmad"]:.1f} um of '
              f'equivalent hexapod dz, R2 {res["r2"]:.3f} (dimensionless)')
        if leaky:
            print('  WARNING: whole nights were NOT held out, so this score is not a '
                  'performance estimate and must not be quoted as one. How optimistic it is '
                  'depends on the model: up to a factor of '
                  f'{LEAK_FACTOR} (dimensionless) for boosted trees on 6 features, but only '
                  'about 1.04 (dimensionless) for a Huber linear fit, which has too few '
                  'coefficients to memorise a night.')
    return res


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
        ``model`` (the fitted estimator), ``pred`` (in-sample prediction [µm of equivalent hexapod
        dz]), ``resid``, ``coef`` (or None), ``nmad``, ``intercept`` [µm of equivalent hexapod dz,
        physical], ``features`` (the fitted column order, so ``coef`` can be read without the
        caller repeating it), and ``equation_max_abs_diff`` -- the verified agreement between the
        printed equation and the pipeline [µm].

    Notes
    -----
    The in-sample nMAD of this fit is **not** a performance estimate -- use `evaluate`. It is
    reported only so the difference from the out-of-fold number shows how much the fit depends on
    which nights it saw.

    ``intercept`` is the response at zero in every feature, in physical units: `_physical_intercept`
    shifts the estimator's standardized ``intercept_`` by ``scaler.mean_``. `verify_equation` then
    asserts that the intercept and slopes together reproduce the pipeline's own ``predict``, so the
    equation page cannot silently disagree with the model it describes.
    """
    X = df[features].to_numpy(float)
    y = df.y.to_numpy(float)
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
        print(f'full fit ({model}, all {df.day_obs.nunique()} nights): in-sample residual nMAD '
              f'{res["nmad"]:.1f} um of equivalent hexapod dz (not a performance estimate -- '
              f'see the night-grouped number above)')
        if coef is not None:
            print(f'  intercept {res["intercept"]:+.2f} um of equivalent hexapod dz '
                  f'(physical, response at zero in every feature)')
            for f, c in zip(features, coef):
                print(f'  {f:32s} {c:+10.2f} um per {FEATURE_UNITS.get(f, "?")}')
            print(f'  equation reproduces pipeline.predict to '
                  f'{res["equation_max_abs_diff"]:.2e} um (tolerance 1e-06)')
    return res


# --------------------------------------------------------------------------- summaries

def coefficient_table(coefs, features, v1_per_um_dz, verbose=True):
    """Per-fold coefficient mean and scatter, with the FAM truss cross-check.

    Parameters
    ----------
    coefs : `numpy.ndarray`
        Shape ``(n_folds, n_features)``, in µm of equivalent hexapod dz per feature unit.
    features : `list` [`str`]
    v1_per_um_dz : `float`
        Used to convert the truss coefficient into a dimensionless v-mode-1 amplitude per deg C
        for comparison with `FAM_TRUSS_SLOPE`.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per feature: ``feature``, ``unit``, ``mean``, ``std``, ``n_folds``, ``sign_stable``.
    """
    rows = []
    for i, f in enumerate(features):
        c = coefs[:, i]
        rows.append(dict(feature=f, unit=f'um equiv hexapod dz per {FEATURE_UNITS.get(f, "?")}',
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


def per_band_residual(df, resid, verbose=True):
    """Residual scatter per band from the band-independent fit.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``band``.
    resid : `array_like`
        Out-of-fold residual [µm of equivalent hexapod dz].
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``n``, ``median``, ``resid_nmad`` [µm of equivalent hexapod dz].

    Notes
    -----
    A band-independent model is the point, so the check that matters is that no single band is left
    badly served. A per-band median well away from zero is a genuine band-dependent focus offset --
    filter thickness -- rather than a defect of the fit.
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
        print('per-band residual of the band-independent model [um of equivalent hexapod dz]')
        print(f'  {"band":5s} {"n":>7s} {"median":>9s} {"nMAD":>9s}')
        for _, r in out.iterrows():
            print(f'  {r.band:5s} {int(r.n):7d} {r["median"]:+9.1f} {r.resid_nmad:9.1f}')
    return out


def prediction_calibration(df, pred, bands=BAND_ORDER, verbose=True):
    """Robust calibration of the thermal estimate against the measured response, per band.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``band`` and ``y`` [µm of equivalent hexapod dz].
    pred : `array_like`
        Out-of-fold prediction in the same unit.
    bands : `sequence` [`str`], optional
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        One row for ``all`` then one per band: ``n``, ``pearson_r`` and ``spearman_rho``
        (dimensionless), ``slope`` (dimensionless, measured response over predicted response),
        ``slope_err``, ``intercept`` [µm of equivalent hexapod dz] and ``resid_nmad`` in the same
        unit.

    Notes
    -----
    The slope is the quantity to read: at 1.0 the model is calibrated, meaning a predicted µm of
    defocus corresponds to a measured µm. It is fitted with Huber on the measured response against
    the prediction, so the two axes are response-on-y and prediction-on-x throughout.
    """
    d = df.assign(pred=np.asarray(pred, float))
    rows = []
    for key in ['all'] + [b for b in bands if (d.band == b).any()]:
        g = d if key == 'all' else d[d.band == key]
        x = g.pred.to_numpy(float)
        y = g.y.to_numpy(float)
        h = huber_fit(x, y)
        m = np.isfinite(x) & np.isfinite(y)
        rows.append(dict(band=key, n=int(m.sum()),
                         pearson_r=h['pearson_r'] if h else np.nan,
                         spearman_rho=h['spearman_rho'] if h else np.nan,
                         slope=h['slope'] if h else np.nan,
                         slope_err=h['slope_err'] if h else np.nan,
                         intercept=h['intercept'] if h else np.nan,
                         resid_nmad=float(nmad((y - x)[m])) if m.any() else np.nan))
    out = pd.DataFrame(rows)
    if verbose:
        print('thermal estimate against measured response '
              '[response and residual in um of equivalent hexapod dz; slope dimensionless, '
              'measured over predicted]')
        print(f'  {"band":5s} {"n":>7s} {"Pearson r":>10s} {"Spearman":>9s} {"slope":>8s} '
              f'{"nMAD":>8s}')
        for _, r in out.iterrows():
            print(f'  {r.band:5s} {int(r.n):7d} {r.pearson_r:+10.4f} {r.spearman_rho:+9.4f} '
                  f'{r.slope:+8.3f} {r.resid_nmad:8.1f}')
    return out


def filter_change_step(df, col, verbose=True, label=''):
    """Median absolute step in a residual across a filter change, against same-band steps.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``day_obs``, ``seq_num``, ``band`` and `col`.
    col : `str`
        Residual column [µm of equivalent hexapod dz].
    verbose : `bool`, optional
    label : `str`, optional
        Name used in the printed line.

    Returns
    -------
    res : `dict`
        ``n_change``, ``median_change``, ``n_same``, ``median_same`` -- median |step| in µm of
        equivalent hexapod dz -- and the ``ratio`` of the two (dimensionless).

    Notes
    -----
    Steps are taken within a night only, between consecutive ``seq_num``, so the daytime gap is
    never crossed. A band-independent correction should bring the band-change step down towards the
    same-band step; residual excess is a real per-band focus offset, which a shared slope cannot
    remove.
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
        Per model: ``model``, ``resid_nmad`` [µm of equivalent hexapod dz],
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
              f'{base:.1f} um of equivalent hexapod dz')
        print(f'  {"model":14s} {"nMAD[um]":>9s} {"frac base":>10s} {"R2":>7s}')
        for _, r in out.iterrows():
            print(f'  {r.model:14s} {r.resid_nmad:9.1f} {r.frac_of_baseline:10.3f} {r.r2:7.3f}')
        print('  frac base and R2 are dimensionless')
    return out


def compare_feature_sets(df_all, group_sets, model='huber', n_splits=5, verbose=True):
    """Night-grouped residual nMAD for each feature-group combination.

    Parameters
    ----------
    df_all : `pandas.DataFrame`
        Table already carrying every candidate column, from `load_target` called with the union of
        the groups being compared.
    group_sets : `list` [`tuple` [`str`]]
        Each entry a tuple of `FEATURE_GROUPS` keys.
    model : `str`, optional
    n_splits : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per set: ``groups``, ``n_features``, ``resid_nmad`` [µm of equivalent hexapod dz],
        ``frac_of_baseline`` and ``r2`` (dimensionless).
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
              f'{base:.1f} um of equivalent hexapod dz')
        print(f'  {"feature groups":34s} {"n":>3s} {"nMAD[um]":>9s} {"R2":>7s}')
        for _, r in out.iterrows():
            print(f'  {r.groups:34s} {int(r.n_features):3d} {r.resid_nmad:9.1f} {r.r2:7.3f}')
    return out


def per_night_direction_slopes(df, ycol=YCOL, ref_elev_deg=REF_ELEV_DEG, verbose=True):
    """Per-night elevation slope of a residual, split by the direction elevation is moving.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit table with ``day_obs``, ``altitude_deg`` [deg], ``direction`` and `ycol`.
    ycol : `str`, optional
        Response column [µm of equivalent hexapod dz].
    ref_elev_deg : `float`, optional
        Elevation at which each night's fit is evaluated to give ``offset_*`` [deg].
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        One row per night with ``slope_all``, ``slope_up``, ``slope_down`` and their standard
        errors [µm of equivalent hexapod dz per deg], the counts behind each, ``intercept_*`` at
        0 deg and ``offset_*`` at `ref_elev_deg` [µm of equivalent hexapod dz], the night median
        of `ycol` in the same unit, and ``difference`` = rising minus falling with
        ``difference_sigma`` in units of the combined standard error.

    Notes
    -----
    Huber (`statsmodels` `RLM` with `HuberT`) throughout, matching the rest of the study.

    ``offset_*`` rather than ``intercept_*`` is the quantity to compare night to night: the 0 deg
    intercept lies about 60 deg outside the observed elevation range, so its scatter is dominated
    by the slope error propagated over that lever arm.
    """
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
            r[f'intercept_{key}'] = f['intercept'] if f else np.nan
            # The night's offset, read at ref_elev_deg rather than at the 0 deg intercept, so it
            # is a value inside the data instead of a 60 deg extrapolation.
            r[f'offset_{key}'] = (f['intercept'] + f['slope'] * ref_elev_deg if f else np.nan)
        r[f'median_{ycol}'] = float(np.nanmedian(d[ycol].to_numpy(float)))
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
        print(f'\nper-night elevation slope of {ycol} '
              f'[um of equivalent hexapod dz per deg], all points')
        print(f'  nights fitted       : {len(a)}')
        print(f'  median slope        : {np.median(a):+.3f} um per deg')
        print(f'  nMAD of the slopes  : {nmad(a):.3f} um per deg')
        print(f'  full range          : {a.min():+.3f} to {a.max():+.3f} um per deg')
        print(f'  median formal error : {out.err_all.median():.3f} um per deg')

        o = out.offset_all.dropna().to_numpy(float)
        if len(o):
            print(f'\nper-night offset of {ycol} at {ref_elev_deg:.0f} deg elevation '
                  f'[um of equivalent hexapod dz], all points')
            print(f'  nights fitted       : {len(o)}')
            print(f'  median offset       : {np.median(o):+.1f} um')
            print(f'  nMAD of the offsets : {nmad(o):.1f} um')
            print(f'  full range          : {o.min():+.1f} to {o.max():+.1f} um')
            # The night-to-night scatter of the offset against the within-night scatter is the
            # question: a large ratio says the residual is a per-night constant.
            within = out.resid_nmad_all.median()
            if within and np.isfinite(within) and within > 0:
                print(f'  median within-night residual nMAD : {within:.1f} um')
                print(f'  night-to-night over within-night  : {nmad(o) / within:.2f} '
                      f'(dimensionless, offset nMAD over residual nMAD)')
        both = out.dropna(subset=['difference'])
        if len(both):
            from scipy import stats
            med = both.difference.median()
            pos = int((both.difference > 0).sum())
            p = stats.binomtest(pos, len(both), 0.5).pvalue
            print(f'\nrising minus falling slope, {len(both)} nights with both legs')
            print(f'  median difference   : {med:+.3f} um of equivalent hexapod dz per deg')
            print(f'  nMAD of differences : '
                  f'{nmad(both.difference.to_numpy(float)):.3f} um per deg')
            n_sig = int((both.difference_sigma.abs() > 3).sum())
            print(f'  nights differing by more than 3 combined standard errors: '
                  f'{n_sig} of {len(both)}')
            print(f'  nights with rising steeper than falling: {pos} of {len(both)} '
                  f'(sign-test p = {p:.3g})')
            if p < 0.01:
                print('  -> a consistent direction-dependent offset, i.e. hysteresis, rather '
                      'than symmetric night-to-night scatter')
            else:
                print('  -> no consistent direction dependence; the rising-minus-falling '
                      'difference scatters about zero')
    return out


# --------------------------------------------------------------------------- shared page parts

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
                # A sign flag must precede the width in a format spec, so '+.1f' with width 9
                # becomes '>+9.1f' and not the invalid '>9+.1f'.
                sign, rest = (f[0], f[1:]) if f[:1] in '+- ' else ('', f)
                cells.append(f'{v:>{sign}{w}{rest}}')
        lines.append('  '.join(cells))
    return '\n'.join(lines)


def stat_title(band, n, h, t=None, unit='um dz', xunit=None, extra=''):
    """Panel title carrying the fit statistics.

    Parameters
    ----------
    band : `str`
    n : `int`
        Visits in the fit.
    h : `dict` or `None`
        Huber result from `run_science_lut.huber_fit`.
    t : `dict` or `None`, optional
        Theil-Sen result, quoted as a leverage check.
    unit : `str`, optional
        Response unit, for the slope's numerator.
    xunit : `str`, optional
        Predictor unit, for the slope's denominator.
    extra : `str`, optional
        Appended verbatim on its own line.

    Returns
    -------
    title : `str`

    Notes
    -----
    Slope and error are rounded so the error carries two significant figures and the slope matches
    its decimal place, which is one fewer than the raw fit prints.
    """
    if h is None:
        return f'{band} band   n = {n}   fit under-determined'
    err = h['slope_err']
    nd = 2 if err <= 0 or not np.isfinite(err) else max(0, 1 - int(np.floor(np.log10(err))))
    su = f'{unit} per {xunit}' if xunit else unit
    line = (f'{band} band   n = {h["n"]}   slope {h["slope"]:+.{nd}f} '
            f'+/- {err:.{nd}f} {su}')
    # The residual nMAD spans a dimensionless v-mode-1 amplitude of order 0.01 and a dz-equivalent
    # of order 60 um, so the precision follows the magnitude rather than being fixed.
    rn = h['resid_nmad']
    rnd = 1 if (np.isfinite(rn) and abs(rn) >= 1.0) else 4
    line2 = (f'Pearson r = {h["pearson_r"]:+.3f}   '
             f'Spearman rho = {h["spearman_rho"]:+.3f}   '
             f'nMAD = {rn:.{rnd}f} {unit}')
    if t is not None:
        line2 += f'\nTheil-Sen slope {t["slope"]:+.{nd}f} {su}'
    return line + '\n' + line2 + (('\n' + extra) if extra else '')


def hist_title(band, v, unit='um dz'):
    """Panel title for a residual histogram.

    Parameters
    ----------
    band : `str`
    v : `array_like`
        Residuals in `unit`.
    unit : `str`, optional

    Returns
    -------
    title : `str`
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return f'{band} band   no finite entries'
    return (f'{band} band   n = {v.size}\n'
            f'median {np.median(v):+.4f} {unit}   robust RMS (nMAD) {nmad(v):.4f} {unit}')


# --------------------------------------------------------------------------- pages

def page_documentation(pdf, meta):
    """Opening page: what the study is, what it measures, how it works and what it finds.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    meta : `dict`
        ``title``, ``subtitle`` and ``sections``, a list of ``(heading, body)`` pairs.
    """
    page_text(pdf, meta['title'], meta['sections'], subtitle=meta.get('subtitle'))


def page_visits_per_night(pdf, df, bands):
    """Stacked histogram of visits per night by band, on a calendar axis.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``day_obs`` and ``band``.
    bands : `sequence` [`str`]
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.2))
    nights = np.sort(df.day_obs.unique())
    dates = day_obs_to_date(nights)
    bottom = np.zeros(len(nights))
    for b in bands:
        c = (df[df.band == b].groupby('day_obs').size()
             .reindex(nights, fill_value=0).to_numpy(float))
        ax.bar(dates, c, bottom=bottom, width=1.0, label=f'{b} ({int(c.sum())})',
               color=BAND_COLORS.get(b, 'grey'), edgecolor='none')
        bottom += c
    ax.set_xlabel('day_obs [calendar date of the observing night]')
    ax.set_ylabel('science visits per night [count]')
    ax.set_title(f'science visits per night by band: n = {len(df)} visits over '
                 f'{len(nights)} nights, {int(nights.min())} to {int(nights.max())}')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.legend(title='band (visits)', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_response_histograms(pdf, df, v1_per_um_dz):
    """Histograms of the response in both the dimensionless and the hexapod-dz unit.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``y`` [µm of equivalent hexapod dz].
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm], quoted in the title.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    dz = df.y.to_numpy(float)
    for ax, v, unit, lab in (
            (axes[0], dz * v1_per_um_dz, 'dimensionless',
             'v1(Trim) - v1(measured) [dimensionless v-mode-1 amplitude]'),
            (axes[1], dz, 'um dz',
             'v1(Trim) - v1(measured) as equivalent hexapod dz [um]')):
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.5, 99.5])
        ax.hist(v, bins=80, range=(lo, hi), color='tab:blue', alpha=0.8)
        ax.set_xlabel(lab, fontsize=9)
        ax.set_ylabel('visits [count]')
        ax.set_title(f'n = {v.size}\nmedian {np.median(v):+.4f} {unit}   '
                     f'robust RMS (nMAD) {nmad(v):.4f} {unit}', fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f'The response, before any correction;   conversion {v1_per_um_dz:.5e} '
                 f'dimensionless v-mode-1 amplitude per um of hexapod dz', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_grid_fits(pdf, df, xcol, ycol, bands, xlabel, ylabel, title, xunit,
                   shared_ylim=True, order=None, fit_range=None, rows=2, cols=3,
                   theilsen=False, ylim=None):
    """One page of per-band scatter panels with a robust fit.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    xcol, ycol : `str`
    bands : `sequence` [`str`]
    xlabel, ylabel : `str`
    title : `str`
    xunit : `str`
        Predictor unit, for the slope denominator in each panel title.
    shared_ylim : `bool`, optional
        Match the y-axis across panels, from the pooled 1st to 99th percentile. Ignored when
        `ylim` is given.
    order : `int`, optional
        Polynomial order. None or 1 fits a line.
    fit_range : `tuple` [`float`], optional
        Restrict the fit (not the plot) to this predictor range.
    rows, cols : `int`, optional
    theilsen : `bool`, optional
        Quote a Theil-Sen slope beside the Huber one. Theil-Sen forms all pairwise slopes, so it
        costs O(n^2) -- minutes per panel at tens of thousands of visits -- while contributing only
        a leverage cross-check line to the title. Default False.
    ylim : `tuple` [`float`], optional
        Explicit ``(low, high)`` y-limits in the response unit, used for every panel instead of the
        pooled percentile window. Each panel then reports the count falling outside it.

    Returns
    -------
    fits : `dict`
        Band to the fit result, so a caller can subtract it.

    Notes
    -----
    The fits use every finite point in `fit_range` regardless of the y-limits; the limits set what
    is drawn, not what is fitted, so a slope never depends on the plotting window.
    """
    import matplotlib.pyplot as plt

    yall = df[ycol].to_numpy(float)
    yall = yall[np.isfinite(yall)]
    if ylim is None:
        ylim = tuple(np.percentile(yall, [1, 99])) if (shared_ylim and yall.size) else None

    fig, axes = plt.subplots(rows, cols, figsize=(15, 8.5), squeeze=False)
    out = {}
    for k, b in enumerate(bands):
        ax = axes[k // cols][k % cols]
        d = df[df.band == b]
        x, y = d[xcol].to_numpy(float), d[ycol].to_numpy(float)
        ax.plot(x, y, '.', ms=1.5, alpha=0.35, color=BAND_COLORS.get(b, 'grey'))
        m = np.isfinite(x) & np.isfinite(y)
        if fit_range is not None:
            m &= (x >= fit_range[0]) & (x <= fit_range[1])
        if order and order > 1:
            r = robust_poly(x[m], y[m], order)
            out[b] = r
            if r is not None:
                xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 200)
                ax.plot(xs, r['predict'](xs), '-', color='k', lw=1.6)
                ax.set_title(f'{b} band   n = {r["n"]}   order {order} over '
                             f'{fit_range[0]:.0f}-{fit_range[1]:.0f} {xunit}\n'
                             f'robust RMS (nMAD) {r["resid_nmad"]:.4f} um dz', fontsize=8.5)
            else:
                ax.set_title(f'{b} band   fit under-determined', fontsize=8.5)
        else:
            h = huber_fit(x[m], y[m])
            t = theilsen_fit(x[m], y[m]) if theilsen else None
            out[b] = h
            if h is not None:
                xs = np.array([np.nanmin(x[m]), np.nanmax(x[m])])
                ax.plot(xs, h['intercept'] + h['slope'] * xs, '-', color='k', lw=1.6)
            n_out = (int(((y[m] < ylim[0]) | (y[m] > ylim[1])).sum())
                     if ylim is not None else 0)
            ax.set_title(stat_title(b, int(m.sum()), h, t, xunit=xunit,
                                    extra=(f'{n_out} points outside the plotted window'
                                           if n_out else '')),
                         fontsize=8)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for k in range(len(bands), rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle(title, fontsize=11)
    # A multi-line title needs its own band reserved, or tight_layout puts the panel titles under it.
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.022 * title.count('\n')))
    pdf.savefig(fig)
    plt.close(fig)
    return out


def page_model_equation(pdf, full, coef_tab, features, df, v1_per_um_dz, ev):
    """The fitted model written out as an equation, with every coefficient's unit.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    full : `dict`
        From `fit_full`: the full-sample deliverable model, with a physical ``intercept``.
    coef_tab : `pandas.DataFrame`
        From `coefficient_table`: the per-fold mean and scatter.
    features : `list` [`str`]
    df : `pandas.DataFrame`
        The fitted sample, for the feature means and the sample description.
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm of hexapod dz].
    ev : `dict`
        From `evaluate`, for the out-of-fold scatter quoted beside the fit.

    Notes
    -----
    Every number is read from the fit at run time. The intercept is the physical one -- the response
    at zero in every feature -- and `fit_full` has already asserted that the intercept and slopes
    together reproduce the pipeline's own ``predict``, so the equation on this page cannot disagree
    with the model that produced the rest of the document.

    The truss coefficient appears twice on purpose, and both are labelled: the full-sample value is
    what the equation quotes, since that is the deliverable model, while the mean over folds and its
    scatter describe how stable the fit is against which nights it saw.
    """
    coef = full['coef']
    b = full['intercept']
    unit = 'um of equivalent total hexapod dz, 0.5 um on each hexapod'

    eq = [f'v1_dz [{unit}]', '']
    eq.append(f'  v1_dz = {b:+11.2f}')
    for f, c in zip(features, coef):
        eq.append(f'          {c:+11.2f} * {f:30s} [per {FEATURE_UNITS.get(f, "?")}]')
    equation = '\n'.join(eq)

    # Each term is referenced to its own feature mean, so a reader can see which part of the
    # intercept each term buys back over the observed range.
    rows = []
    for i, f in enumerate(features):
        mu = float(np.nanmean(df[f].to_numpy(float)))
        srow = coef_tab[coef_tab.feature == f]
        std = float(srow['std'].iloc[0]) if len(srow) else np.nan
        fmean = float(srow['mean'].iloc[0]) if len(srow) else np.nan
        rows.append(dict(feature=f, unit=FEATURE_UNITS.get(f, '?'),
                         full=float(coef[i]), fold_mean=fmean, fold_std=std,
                         feat_mean=mu, term=float(coef[i]) * mu))
    tab = pd.DataFrame(rows)
    coef_block = _table_text(
        tab, ['feature', 'unit', 'full', 'fold_mean', 'fold_std', 'feat_mean', 'term'],
        ['s', 's', '+.2f', '+.2f', '.2f', '+.5f', '+.1f'],
        [30, 12, 10, 10, 8, 10, 9])
    coef_note = (f'{coef_block}\n\n'
                 f'full      = the deliverable full-sample coefficient, as in the equation\n'
                 f'fold_mean = mean over the {int(coef_tab.n_folds.iloc[0])} night-grouped folds; '
                 f'fold_std is its scatter\n'
                 f'feat_mean = the sample mean of that feature, in its own unit\n'
                 f'term      = full * feat_mean, the contribution at the mean [um of\n'
                 f'            equivalent hexapod dz]')

    lines = []
    if 'truss_temp_mean_c' in features:
        i = features.index('truss_temp_mean_c')
        full_v1 = float(coef[i]) * v1_per_um_dz
        srow = coef_tab[coef_tab.feature == 'truss_temp_mean_c']
        fm = float(srow['mean'].iloc[0])
        fs = float(srow['std'].iloc[0])
        fold_v1 = fm * v1_per_um_dz
        d_full = 100 * abs(full_v1 - FAM_TRUSS_SLOPE) / FAM_TRUSS_SLOPE
        d_fold = 100 * abs(fold_v1 - FAM_TRUSS_SLOPE) / FAM_TRUSS_SLOPE
        lines = [
            f'full sample : {coef[i]:+.2f} um of equivalent hexapod dz per deg C',
            f'              = {full_v1:+.5f} dimensionless v-mode-1 amplitude per deg C',
            f'                differs from FAM by {d_full:.1f}% (dimensionless)',
            '',
            f'fold mean   : {fm:+.2f} +/- {fs:.2f} um of equivalent hexapod dz per deg C',
            f'              = {fold_v1:+.5f} dimensionless v-mode-1 amplitude per deg C',
            f'                differs from FAM by {d_fold:.1f}% (dimensionless)',
            '',
            f'FAM Double Zernike (independent) : {FAM_TRUSS_SLOPE:+.5f} dimensionless',
            f'                                  v-mode-1 amplitude per deg C',
        ]
    fam_block = '\n'.join(lines) if lines else 'the truss temperature is not in this feature set'

    base = nmad(df.y.to_numpy(float))
    sample = (f'visits fitted            : {len(df)}\n'
              f'nights fitted            : {df.day_obs.nunique()}\n'
              f'night range              : {int(df.day_obs.min())} to '
              f'{int(df.day_obs.max())}\n'
              f'bands                    : '
              f'{" ".join(b for b in BAND_ORDER if (df.band == b).any())}\n'
              f'uncorrected response nMAD: {base:.1f} um of equivalent hexapod dz\n'
              f'out-of-fold residual nMAD: {ev["nmad"]:.1f} um, R2 {ev["r2"]:.3f} '
              f'(dimensionless)\n'
              f'                           = {100 * ev["nmad"] / base:.0f}% of the '
              f'uncorrected scatter\n'
              f'in-sample residual nMAD  : {full["nmad"]:.1f} um (not a performance estimate)\n'
              f'equation vs pipeline     : max |difference| '
              f'{full["equation_max_abs_diff"]:.2e} um, asserted below 1e-06')

    page_text(pdf, 'The fitted thermal model',
              [('The equation', equation),
               ('Coefficients, their fold-to-fold stability, and the term at the sample mean',
                coef_note),
               ('The truss temperature against the independent FAM measurement', fam_block),
               ('Sample', sample)],
              subtitle='One band-independent Huber robust linear fit; the deliverable model is '
                       'fitted on every night.')


def _calibration_panel(ax, x, y, band, lim, cal_row, compact=False):
    """Draw one prediction-against-response panel with its 1:1 line.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
    x, y : `array_like`
        Prediction and measured response [µm of equivalent hexapod dz].
    band : `str`
        Panel label; ``'all'`` for the pooled panel.
    lim : `tuple` [`float`]
        Shared axis window in the same unit.
    cal_row : `pandas.Series`
        The matching row of `prediction_calibration`.
    compact : `bool`, optional
        Break the title over four short lines and drop the units gloss, for the six-panel page
        where one wide line runs under the neighbouring panels.

    Returns
    -------
    hexbin : `matplotlib.collections.PolyCollection`
        The density collection, so a caller can attach a colourbar.
    """
    m = np.isfinite(x) & np.isfinite(y)
    # Tens of thousands of markers overplot into a solid block, so the density is drawn as a
    # hexbin and the 1:1 line stays visible on top of it.
    hb = ax.hexbin(x[m], y[m], gridsize=60, extent=(lim[0], lim[1], lim[0], lim[1]),
                   cmap='viridis', bins='log', mincnt=1)
    ax.plot(lim, lim, '-', color='tab:red', lw=1.4, label='1:1')
    if np.isfinite(cal_row.slope):
        xs = np.array(lim, float)
        ax.plot(xs, cal_row.intercept + cal_row.slope * xs, '--', color='w', lw=1.2,
                label=f'Huber slope {cal_row.slope:+.3f}')
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.25)
    n_out = int(((x[m] < lim[0]) | (x[m] > lim[1]) | (y[m] < lim[0]) | (y[m] > lim[1])).sum())
    extra = f'\n{n_out} visits outside the plotted window' if n_out else ''
    if compact:
        # Four short lines: on a 3-column page a single statistics line is about twice the panel
        # width and matplotlib centres the overflow under the neighbouring panels.
        ax.set_title(f'{band}   n = {int(cal_row.n)}\n'
                     f'Pearson r {cal_row.pearson_r:+.4f}   '
                     f'Spearman rho {cal_row.spearman_rho:+.4f}\n'
                     f'Huber slope {cal_row.slope:+.3f} +/- {cal_row.slope_err:.3f}   '
                     f'residual nMAD {cal_row.resid_nmad:.1f} um{extra}', fontsize=8)
    else:
        ax.set_title(f'{band}   n = {int(cal_row.n)}   Pearson r {cal_row.pearson_r:+.4f}   '
                     f'Spearman rho {cal_row.spearman_rho:+.4f}\n'
                     f'Huber slope {cal_row.slope:+.3f} +/- {cal_row.slope_err:.3f} '
                     f'(dimensionless, measured over predicted)   '
                     f'residual nMAD {cal_row.resid_nmad:.1f} um{extra}', fontsize=8)
    return hb


def page_prediction_vs_response(pdf, df, pred, cal, lim=(-1500.0, 2000.0)):
    """The thermal estimate against the measured response, all bands on one panel.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carrying ``y`` [µm of equivalent hexapod dz].
    pred : `array_like`
        Out-of-fold prediction in the same unit.
    cal : `pandas.DataFrame`
        From `prediction_calibration`.
    lim : `tuple` [`float`], optional
        Shared axis window [µm of equivalent hexapod dz]; each panel reports what falls outside.

    Notes
    -----
    Prediction on x and measured response on y throughout, so a slope of 1.0 means a predicted µm
    of defocus corresponds to a measured µm.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    row = cal[cal.band == 'all'].iloc[0]
    hb = _calibration_panel(ax, np.asarray(pred, float), df.y.to_numpy(float), 'all bands', lim,
                            row)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('visits per hexagonal bin (log scale)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel('thermal estimate of v1_dz, out-of-fold [um of equivalent hexapod dz]')
    ax.set_ylabel('measured v1(Trim) - v1(measured) [um of equivalent hexapod dz]')
    fig.suptitle('The thermal estimate against the measured response, all bands. Colour is the '
                 'visit density (log)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def page_prediction_vs_response_bands(pdf, df, pred, cal, bands=BAND_ORDER,
                                      lim=(-1500.0, 2000.0)):
    """The thermal estimate against the measured response, one panel per band.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    pred : `array_like`
        Out-of-fold prediction [µm of equivalent hexapod dz].
    cal : `pandas.DataFrame`
        From `prediction_calibration`.
    bands : `sequence` [`str`], optional
    lim : `tuple` [`float`], optional
        Shared axis window, the same on every panel so bands compare directly.
    """
    import matplotlib.pyplot as plt

    d = df.assign(pred=np.asarray(pred, float))
    present = [b for b in bands if (d.band == b).any()]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4), squeeze=False)
    for k, b in enumerate(present[:6]):
        ax = axes[k // 3][k % 3]
        g = d[d.band == b]
        sub = cal[cal.band == b]
        if not len(sub):
            ax.axis('off')
            continue
        _calibration_panel(ax, g.pred.to_numpy(float), g.y.to_numpy(float), f'{b} band', lim,
                           sub.iloc[0], compact=True)
        ax.set_xlabel('thermal estimate [um equiv hexapod dz]', fontsize=8)
        ax.set_ylabel('measured response [um equiv hexapod dz]', fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(len(present), 6):
        axes[k // 3][k % 3].axis('off')
    fig.suptitle('The thermal estimate against the measured response, per band. One shared window '
                 'and one shared 1:1 line, so the bands compare directly.\nHuber slope is '
                 'dimensionless, measured response over predicted; residual nMAD and both axes '
                 'are um of equivalent hexapod dz. Colour is the visit density (log)',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def page_coefficients(pdf, coefs, features, v1_per_um_dz):
    """Per-fold coefficient scatter, one panel per feature, plus the FAM truss cross-check.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    coefs : `numpy.ndarray`
        Shape ``(n_folds, n_features)`` [µm of equivalent hexapod dz per feature unit].
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
        ax.set_ylabel(f'um equiv hexapod dz\nper {FEATURE_UNITS.get(f, "?")}', fontsize=8)
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
                 f'{FAM_TRUSS_SLOPE:+.5f} per deg C, differ by {dev:.1f}% (dimensionless)')
    fig.suptitle(f'Coefficients across {coefs.shape[0]} night-grouped folds{extra}', fontsize=10.5)
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
        [µm of equivalent hexapod dz].
    baseline : `array_like`, optional
        The uncorrected response, drawn behind for scale, in the same unit.
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
        ax.set_xlabel('out-of-fold residual [um equiv hexapod dz]', fontsize=8)
        ax.set_ylabel('visits', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(bands), 6):
        axes[j // 3][j % 3].axis('off')
    fig.suptitle('Per-band residual of the single band-independent model. A per-band median away '
                 'from zero is a real filter-thickness focus offset', fontsize=10.5)
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
        [µm of equivalent hexapod dz].
    extra_cols : `tuple` [`str`], optional
        Columns plotted beside the features though not fitted. ``altitude_deg`` is here because a
        surviving elevation trend would be a missing term.

    Notes
    -----
    A flat binned-median trend means the model has taken out that feature's dependence; a surviving
    trend in a column that was *not* fitted, such as elevation, would be a missing term.
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
            edges = np.unique(np.percentile(x[m], q))
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
        ax.set_ylabel('out-of-fold residual\n[um equiv hexapod dz]', fontsize=8)
        lo, hi = np.percentile(resid[m], [1, 99]) if m.sum() else (-1, 1)
        ax.set_ylim(lo, hi)
        tag = ' (not fitted)' if c not in features else ''
        ax.set_title(f'{c}{tag}', fontsize=9)
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('Out-of-fold residual against each feature; red is the binned median. A flat '
                 'trend means that dependence has been removed', fontsize=10.5)
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
        [µm of equivalent hexapod dz].
    day_obs : `int`
    features : `list` [`str`]

    Notes
    -----
    Filter changes are drawn as vertical dotted lines, and the corrected trace should cross them
    without a step.
    """
    import matplotlib.pyplot as plt

    d = df.assign(resid=resid)
    d = d[d.day_obs == int(day_obs)].sort_values('seq_num')
    if not len(d):
        print(f'no visits on {day_obs}, skipping the night-series page')
        return
    changes = d.seq_num[d.band.ne(d.band.shift(1))].to_numpy()[1:]
    panels = [('y', 'uncorrected v1(Trim) - v1(measured)\n[um equiv hexapod dz]', 'tab:gray'),
              ('resid', 'band-independent model residual\n[um equiv hexapod dz]', 'tab:blue')]
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
    fig.suptitle(f'{day_obs}: the band-independent correction across filter changes (dotted '
                 f'lines). {len(d)} visits', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def page_night_temperatures(pdf, df, day_obs, features):
    """The driving temperatures for one night against ``seq_num``, elevation twinned.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carrying ``day_obs``, ``seq_num``, ``altitude_deg`` [deg] and the feature columns.
    day_obs : `int`
    features : `list` [`str`]
        The first two are drawn, one panel each.

    Notes
    -----
    These are the quantities the correction is fitted against, so seeing them in observing order
    says whether a within-night focus drift tracks a temperature that is itself drifting, or moves
    independently of both.
    """
    import matplotlib.pyplot as plt

    d = df[df.day_obs == int(day_obs)].sort_values('seq_num')
    if not len(d):
        print(f'no visits on {day_obs}, skipping the temperature page')
        return
    cols = [c for c in features[:2] if c in d.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(13.3, 7.5), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, c, colour in zip(axes, cols, ('tab:red', 'tab:purple')):
        ax.plot(d.seq_num, d[c], '.-', ms=3, lw=0.6, color=colour)
        ax.set_ylabel(f'{c}\n[{FEATURE_UNITS.get(c, "?")}]', fontsize=8, color=colour)
        ax.tick_params(axis='y', labelsize=7, labelcolor=colour)
        v = d[c].to_numpy(float)
        v = v[np.isfinite(v)]
        rng = (f'{v.min():+.3f} to {v.max():+.3f} {FEATURE_UNITS.get(c, "?")}, '
               f'range {np.ptp(v):.3f}' if len(v) else 'no finite values')
        ax.set_title(f'{c}: {rng}', fontsize=8, loc='left')
        ax.grid(alpha=0.3)
        # Elevation on a twin axis rather than an extra panel, so the eye reads the lag between a
        # slew and the response off one shared x axis.
        axr = ax.twinx()
        axr.plot(d.seq_num, d.altitude_deg, '-', lw=1.0, color='0.35', alpha=0.8)
        axr.set_ylabel('elevation [deg]', fontsize=8, color='0.35')
        axr.tick_params(axis='y', labelsize=7, labelcolor='0.35')
    axes[-1].set_xlabel('seq_num', fontsize=9)
    fig.suptitle(f'{day_obs}: the driving temperatures against sequence number, elevation in grey '
                 f'on the right axis ({len(d)} visits)', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_night_panels(pdf, df, slopes, ycol=YCOL, n_per_page=NIGHTS_PER_PAGE,
                      max_nights=None, ylim=None, yspan=PANEL_YSPAN):
    """One panel per night: the corrected focus error against elevation, split by direction.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``day_obs``, ``altitude_deg`` [deg], ``direction`` and `ycol`.
    slopes : `pandas.DataFrame`
        Output of `per_night_direction_slopes`.
    ycol : `str`, optional
        Residual column plotted [µm of equivalent hexapod dz].
    n_per_page : `int`, optional
        Panels per page, drawn as 4 columns by 3 rows.
    max_nights : `int`, optional
        Draw only the first this many nights.
    ylim : `tuple` [`float`], optional
        Absolute shared y-limits [µm of equivalent hexapod dz]. Overrides `yspan`.
    yspan : `float`, optional
        Panel height [µm of equivalent hexapod dz], centred on each night's own median. None
        autoscales each panel independently.

    Notes
    -----
    The default is a shared *span* about each night's own median rather than one absolute
    window, because the corrected residual still carries a per-night offset of order 1000 µm
    that the thermal fit does not remove; a single absolute window wide enough to hold every
    night would leave each panel's slope too small to read, while a common span makes the
    slopes directly comparable panel to panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    days = list(slopes.day_obs)
    if max_nights:
        days = days[:max_nights]
    handles = [Line2D([], [], marker='o', ls='', ms=4, color=DIR_COLOUR['up'],
                      label='rising elevation'),
               Line2D([], [], marker='o', ls='', ms=4, color=DIR_COLOUR['down'],
                      label='falling elevation'),
               Line2D([], [], color='k', lw=1.5, label='all points')]
    for start in range(0, len(days), n_per_page):
        chunk = days[start:start + n_per_page]
        fig, axes = plt.subplots(3, 4, figsize=(15, 9.5))
        for ax, day in zip(axes.ravel(), chunk):
            d = df[df.day_obs == day]
            r = slopes[slopes.day_obs == day].iloc[0]
            for direction in ('up', 'down'):
                s = d[d.direction == direction]
                if len(s):
                    ax.scatter(s.altitude_deg, s[ycol], s=3, alpha=0.4,
                               color=DIR_COLOUR[direction])
            el = np.linspace(d.altitude_deg.min(), d.altitude_deg.max(), 10)
            bits = []
            for key, colour, lw in (('all', 'k', 1.6),
                                    ('up', DIR_COLOUR['up'], 1.1),
                                    ('down', DIR_COLOUR['down'], 1.1)):
                if np.isfinite(r[f'slope_{key}']):
                    ax.plot(el, r[f'intercept_{key}'] + r[f'slope_{key}'] * el, '-',
                            lw=lw, color=colour)
                    bits.append(f'{key} {r[f"slope_{key}"]:+.1f}')
            ax.set_title(f'{day}   n = {int(r.n)}\n' + '  '.join(bits), fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
            if ylim is not None:
                ax.set_ylim(*ylim)
            elif yspan:
                mid = float(np.nanmedian(d[ycol].to_numpy(float)))
                ax.set_ylim(mid - 0.5 * yspan, mid + 0.5 * yspan)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis('off')
        fig.supxlabel('elevation [deg]', fontsize=9, y=0.045)
        fig.supylabel(f'{ycol} [um of equivalent hexapod dz]', fontsize=9)
        fig.suptitle('per-night elevation dependence of the thermally corrected residual; '
                     'slopes in um of equivalent hexapod dz per deg', fontsize=10)
        fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)


def page_night_offsets(pdf, slopes, ycol=YCOL, ref_elev_deg=REF_ELEV_DEG,
                       offset_lim=OFFSET_LIM, slope_lim=SLOPE_LIM):
    """The per-night offset against ``day_obs``, its distribution, and its slope correlation.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    slopes : `pandas.DataFrame`
        Output of `per_night_direction_slopes`, carrying ``offset_all``, ``slope_all``,
        ``err_all`` and ``resid_nmad_all``.
    ycol : `str`, optional
        Response column name, for the labels.
    ref_elev_deg : `float`, optional
        Elevation the offset was evaluated at [deg].
    offset_lim : `tuple` [`float`], optional
        Plotted offset window [µm of equivalent hexapod dz]. None autoscales.
    slope_lim : `tuple` [`float`], optional
        Plotted slope window [µm of equivalent hexapod dz per deg], applied to the
        offset-against-slope scatter so it matches `page_night_slopes`. None autoscales.

    Notes
    -----
    Two pages, the counterpart of `page_night_slopes` for the offset rather than the slope.
    The first is the time series against ``day_obs`` with the within-night residual nMAD as
    the error bar, which answers whether the surviving residual is a per-night constant that
    moves from night to night. The second is the distribution of the offset together with the
    offset against the slope, since a correlation between the two would mean the split
    between them is not identified by the elevation range each night covers.

    `offset_lim` and `slope_lim` change only what is drawn; every night enters the median,
    the nMAD, the trend and the correlations, and each panel says how many nights fall
    outside the window it plots.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    unit = 'um of equivalent hexapod dz'
    s = slopes.dropna(subset=['offset_all']).sort_values('day_obs')
    if not len(s):
        return

    o = s.offset_all.to_numpy(float)
    day_str = [str(int(d)) for d in s.day_obs]
    n_out = (0 if offset_lim is None
             else int(((o < offset_lim[0]) | (o > offset_lim[1])).sum()))
    out_note = (f'; {n_out} nights outside the plotted window' if n_out else '')

    # Page: the offset per night against day_obs.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))
    x = np.arange(len(s))
    # The within-night robust scatter of the residual, divided down by the night's count, is
    # the honest error on a night mean; the Huber slope error does not cover the offset.
    n_all = s.n_all.to_numpy(float)
    err = s.resid_nmad_all.to_numpy(float) / np.sqrt(np.maximum(n_all, 1.0))
    axes[0].errorbar(x, o, yerr=err, fmt='o', ms=3.5, lw=0.8, color='tab:green')
    axes[0].axhline(0, color='grey', lw=0.8)
    axes[0].axhline(np.median(o), color='k', ls='--', lw=0.9,
                    label=f'median {np.median(o):+.1f} {unit}')
    axes[0].set_xticks(x[::2])
    axes[0].set_xticklabels(day_str[::2], rotation=90, fontsize=6)
    axes[0].set_xlabel('day_obs')
    axes[0].set_ylabel(f'offset at {ref_elev_deg:.0f} deg elevation\n[{unit}]', fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    if offset_lim is not None:
        axes[0].set_ylim(*offset_lim)
    axes[0].set_title(f'{len(s)} nights; error bars are the within-night residual nMAD '
                      f'over the square root of the night count' + out_note, fontsize=9)

    # Same series against the night index, with a Theil-Sen trend, to separate a drift across
    # the run from night-to-night scatter about a constant.
    ts = stats.theilslopes(o, x)
    axes[1].plot(x, o, 'o', ms=3.5, color='tab:green', alpha=0.8)
    axes[1].plot(x, ts[1] + ts[0] * x, 'k-', lw=1.2,
                 label=f'Theil-Sen {ts[0]:+.2f} {unit} per night in sequence')
    axes[1].axhline(np.median(o), color='grey', ls='--', lw=0.8)
    axes[1].set_xticks(x[::2])
    axes[1].set_xticklabels(day_str[::2], rotation=90, fontsize=6)
    axes[1].set_xlabel('day_obs')
    axes[1].set_ylabel(f'offset [{unit}]', fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    if offset_lim is not None:
        axes[1].set_ylim(*offset_lim)
    rho, p_rho = stats.spearmanr(x, o)
    axes[1].set_title(f'trend across the run: Spearman rho {rho:+.3f} (p = {p_rho:.2g}), '
                      f'n = {len(s)} nights' + out_note, fontsize=9)
    fig.suptitle(f'per-night offset of {ycol} at {ref_elev_deg:.0f} deg elevation, '
                 f'against day_obs', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page: the offset distribution, and the offset against the slope.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    # Binned over the plotted window rather than the data range, so three failed fits cannot
    # push every other night into one bin.
    bins = (np.linspace(offset_lim[0], offset_lim[1], 25) if offset_lim is not None else 24)
    axes[0].hist(o, bins=bins, color='tab:green', alpha=0.8)
    axes[0].axvline(0, color='grey', lw=0.8)
    axes[0].axvline(np.median(o), color='k', ls='--',
                    label=f'median {np.median(o):+.1f} {unit}')
    axes[0].set_xlabel(f'per-night offset at {ref_elev_deg:.0f} deg elevation [{unit}]')
    axes[0].set_ylabel('nights')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    if offset_lim is not None:
        axes[0].set_xlim(*offset_lim)
    within = float(np.nanmedian(s.resid_nmad_all.to_numpy(float)))
    ratio = (nmad(o) / within) if within > 0 else np.nan
    # Two short lines rather than one long one, so the title does not run under the right panel.
    axes[0].set_title(f'median {np.median(o):+.1f}, nMAD {nmad(o):.1f} {unit}{out_note}\n'
                      f'median within-night residual nMAD {within:.1f} {unit} '
                      f'-- ratio {ratio:.2f} (dimensionless)', fontsize=8)

    sl = s.slope_all.to_numpy(float)
    fin = np.isfinite(sl) & np.isfinite(o)
    axes[1].plot(sl[fin], o[fin], 'o', ms=3.5, color='tab:purple', alpha=0.8)
    axes[1].axhline(0, color='grey', lw=0.8)
    axes[1].axvline(0, color='grey', lw=0.8)
    axes[1].set_xlabel(f'per-night elevation slope [{unit} per deg]')
    axes[1].set_ylabel(f'offset at {ref_elev_deg:.0f} deg elevation [{unit}]')
    axes[1].grid(alpha=0.3)
    if offset_lim is not None:
        axes[1].set_ylim(*offset_lim)
    if slope_lim is not None:
        axes[1].set_xlim(*slope_lim)
    # A night is hidden here if either coordinate leaves its own window.
    hid = np.zeros(fin.sum(), dtype=bool)
    if offset_lim is not None:
        hid |= (o[fin] < offset_lim[0]) | (o[fin] > offset_lim[1])
    if slope_lim is not None:
        hid |= (sl[fin] < slope_lim[0]) | (sl[fin] > slope_lim[1])
    # On its own line: appended to the statistics line it runs off the right edge of the page.
    hid_note = (f'\n{int(hid.sum())} nights outside the plotted window' if hid.any() else '')
    if fin.sum() > 3:
        r, p_r = stats.pearsonr(sl[fin], o[fin])
        rho2, p_rho2 = stats.spearmanr(sl[fin], o[fin])
        axes[1].set_title(f'Pearson r {r:+.3f} (p = {p_r:.2g}), Spearman rho {rho2:+.3f} '
                          f'(p = {p_rho2:.2g}), n = {int(fin.sum())} nights{hid_note}\n'
                          f'a strong correlation would mean the offset and the slope are '
                          f'not separately identified', fontsize=8)
    fig.suptitle(f'distribution of the per-night offset of {ycol}, and its relation to the '
                 f'per-night slope', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_night_slopes(pdf, slopes, ycol=YCOL, slope_lim=SLOPE_LIM, slope_bins=SLOPE_BINS):
    """The rising and falling slopes per night, and the distributions of both statistics.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    slopes : `pandas.DataFrame`
        Output of `per_night_direction_slopes`.
    ycol : `str`, optional
        Residual column name, for the labels.
    slope_lim : `tuple` [`float`], optional
        Plotted slope window [µm of equivalent hexapod dz per deg], shared by the per-night
        slope scatter and the all-points slope histogram. None autoscales.
    slope_bins : `int`, optional
        Histogram bins for the all-points slope, spanning `slope_lim` when it is set.

    Notes
    -----
    Two pages: the per-night rising and falling slopes with the distribution of their
    difference, then the distribution of the all-points slope.

    `slope_lim` changes only what is drawn. Every night still enters the median, nMAD and
    sign test, and each panel reports how many nights fall outside the window, so a night
    whose fit failed stays visible in the numbers rather than being silently dropped.
    """
    import matplotlib.pyplot as plt

    unit = 'um of equivalent hexapod dz per deg'
    both = slopes.dropna(subset=['difference']).sort_values('day_obs')

    def outside(v):
        """Count finite values falling outside the plotted slope window."""
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if slope_lim is None:
            return 0
        return int(((v < slope_lim[0]) | (v > slope_lim[1])).sum())

    # Page: rising and falling slope per night, and the difference distribution.
    if len(both):
        fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))
        x = np.arange(len(both))
        axes[0].errorbar(x - 0.15, both.slope_up, yerr=both.err_up, fmt='o', ms=3,
                         lw=0.8, color=DIR_COLOUR['up'], label='rising elevation')
        axes[0].errorbar(x + 0.15, both.slope_down, yerr=both.err_down, fmt='s', ms=3,
                         lw=0.8, color=DIR_COLOUR['down'], label='falling elevation')
        axes[0].axhline(0, color='grey', lw=0.8)
        axes[0].set_xticks(x[::2])
        axes[0].set_xticklabels([str(d) for d in both.day_obs][::2], rotation=90,
                                fontsize=6)
        axes[0].set_xlabel('day_obs')
        axes[0].set_ylabel(f'slope\n[{unit}]', fontsize=8)
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        if slope_lim is not None:
            axes[0].set_ylim(*slope_lim)
        n_out = outside(both.slope_up) + outside(both.slope_down)
        axes[0].set_title(f'{len(both)} nights with both legs fitted; error bars are the '
                          f'Huber standard error'
                          + (f'; {n_out} leg slopes outside the plotted window'
                             if n_out else ''), fontsize=9)

        d = both.difference.to_numpy(float)
        axes[1].hist(d, bins=24, color='tab:purple', alpha=0.8)
        axes[1].axvline(0, color='grey', lw=0.8)
        axes[1].axvline(np.median(d), color='k', ls='--',
                        label=f'median {np.median(d):+.2f} {unit}')
        axes[1].set_xlabel(f'rising minus falling slope [{unit}]')
        axes[1].set_ylabel('nights')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        n_sig = int((both.difference_sigma.abs() > 3).sum())
        axes[1].set_title(f'nMAD {nmad(d):.2f} {unit}; {n_sig} of {len(both)} nights '
                          f'differ by more than 3 combined standard errors', fontsize=9)
        fig.suptitle(f'elevation hysteresis in {ycol}: rising against falling legs',
                     fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Page: the all-points slope distribution.
    a = slopes.slope_all.dropna().to_numpy(float)
    if len(a):
        fig, ax = plt.subplots(figsize=(11, 7))
        # Binned over the plotted window rather than over the data range, so the bin edges are
        # the round numbers the window names and one bad fit cannot stretch them.
        bins = (np.linspace(slope_lim[0], slope_lim[1], slope_bins + 1)
                if slope_lim is not None else slope_bins)
        ax.hist(a, bins=bins, color='tab:blue', alpha=0.8)
        ax.axvline(0, color='grey', lw=0.8)
        ax.axvline(np.median(a), color='k', ls='--',
                   label=f'median {np.median(a):+.2f} {unit}')
        ax.set_xlabel(f'per-night elevation slope, all points [{unit}]')
        ax.set_ylabel('nights')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        if slope_lim is not None:
            ax.set_xlim(*slope_lim)
        spread = (nmad(a) / slopes.err_all.median()
                  if slopes.err_all.median() > 0 else np.nan)
        n_out = outside(a)
        ax.set_title(f'{len(a)} nights; median {np.median(a):+.2f}, nMAD {nmad(a):.2f} '
                     f'{unit}\nnight-to-night spread over the median formal error '
                     f'{spread:.1f} (dimensionless)'
                     + (f'; {n_out} nights outside the plotted window'
                        if n_out else ''), fontsize=9)
        fig.suptitle(f'distribution of the per-night fitted slope of {ycol}', fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def page_measured_state(pdf, df, variant, bands=BAND_ORDER, lim=MEAS_LIM, bins=MEAS_BINS,
                        verbose=True):
    """Per-band histogram of the uncorrected measured amplitude, with a robust RMS.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``band`` and ``meas_dzequiv`` [µm of equivalent hexapod dz].
    variant : `str`
        ``optical_state`` variant id, for the title.
    bands : `sequence` [`str`], optional
        Bands drawn, one panel each, as 3 columns by 2 rows.
    lim : `tuple` [`float`], optional
        Plotted window [µm of equivalent hexapod dz]; None autoscales per panel.
    bins : `int`, optional
        Bins spanning `lim`.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``n``, ``median``, ``nmad`` and ``rms`` [µm of equivalent hexapod dz], and
        the count outside the plotted window.

    Notes
    -----
    ``nmad`` is the robust scatter and is the number to quote; the plain ``rms`` about the
    median is reported beside it because a handful of visits reach several hundred µm, and the
    ratio of the two says how much the tail inflates a non-robust estimate.

    This is the *measured* state on its own -- no Trim term and no thermal correction.
    """
    import matplotlib.pyplot as plt

    unit = 'um of equivalent hexapod dz'
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, band in zip(axes.ravel(), bands):
        v = df.loc[df.band == band, 'meas_dzequiv'].to_numpy(float)
        v = v[np.isfinite(v)]
        if not v.size:
            ax.axis('off')
            continue
        med, sig = float(np.median(v)), float(nmad(v))
        rms = float(np.sqrt(np.mean((v - med) ** 2)))
        n_out = (int(((v < lim[0]) | (v > lim[1])).sum()) if lim is not None else 0)
        rows.append(dict(band=band, n=int(v.size), median=med, nmad=sig, rms=rms,
                         n_outside=n_out))
        edges = (np.linspace(lim[0], lim[1], bins + 1) if lim is not None else bins)
        ax.hist(v, bins=edges, color='tab:blue', alpha=0.8)
        ax.axvline(0, color='grey', lw=0.8)
        ax.axvline(med, color='k', ls='--', lw=1.2)
        if lim is not None:
            ax.set_xlim(*lim)
        ax.set_title(f'{band}   n = {v.size}\nmedian {med:+.1f}, robust RMS (nMAD) '
                     f'{sig:.1f} um\nplain RMS {rms:.1f} um'
                     + (f', {n_out} outside' if n_out else ''), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(bands):]:
        ax.axis('off')
    fig.supxlabel(f'measured v-mode-1 amplitude, uncorrected [{unit}]', fontsize=9)
    fig.supylabel('visits', fontsize=9)
    fig.suptitle(f'measured optical state by band, no Trim term and no thermal correction; '
                 f'robust RMS is the nMAD -- {variant}', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f'\nuncorrected measured v-mode-1 amplitude by band [{unit}]')
        print(f'  {"band":5s} {"n":>7s} {"median":>9s} {"robust RMS":>11s} '
              f'{"plain RMS":>10s} {"ratio":>7s}')
        for _, r in out.iterrows():
            print(f'  {r.band:5s} {int(r.n):7d} {r["median"]:+9.1f} {r["nmad"]:11.1f} '
                  f'{r["rms"]:10.1f} {r["rms"] / r["nmad"]:7.2f}')
        allv = df.meas_dzequiv.to_numpy(float)
        allv = allv[np.isfinite(allv)]
        print(f'  {"all":5s} {allv.size:7d} {np.median(allv):+9.1f} {nmad(allv):11.1f} '
              f'{np.sqrt(np.mean((allv - np.median(allv)) ** 2)):10.1f}')
        print('  ratio is plain RMS over robust RMS (dimensionless); above ~1.5 the tail '
              'dominates a\n  non-robust estimate')
    return out


def page_lut_vs_elevation(pdf, df, variant, bands=BAND_ORDER, orders=LUT_ELEV_ORDERS,
                          verbose=True):
    """Per-band scatter of the look-up-table term alone against elevation.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``band``, ``altitude_deg`` [deg] and ``lut_dzequiv`` [µm of equivalent
        hexapod dz].
    variant : `str`
        ``optical_state`` variant id, for the title.
    bands : `sequence` [`str`], optional
        Bands drawn, one panel each, as 3 columns by 2 rows.
    orders : `sequence` [`int`], optional
        Polynomial orders fitted in elevation; the first is drawn.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``n``, the median [µm of equivalent hexapod dz], the Huber linear ``slope``
        [µm of equivalent hexapod dz per deg] with its standard error, Pearson r and Spearman
        rho (both dimensionless), and the residual nMAD [µm] after each fitted order.

    Notes
    -----
    This is the hexapod look-up-table (LUT) term on its own -- no Trim, no measured state and
    no thermal correction -- so the slope here is the elevation dependence the control system
    is *already* applying, which is what the response studied elsewhere in this document
    deliberately leaves out.

    Each panel autoscales, because the per-band medians span more than 4000 µm and one shared
    window would compress most panels to a few pixels.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    unit = 'um of equivalent hexapod dz'
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, band in zip(axes.ravel(), bands):
        d = df[(df.band == band) & df.lut_dzequiv.notna() & df.altitude_deg.notna()]
        if not len(d):
            ax.axis('off')
            continue
        x = d.altitude_deg.to_numpy(float)
        y = d.lut_dzequiv.to_numpy(float)
        r = dict(band=band, n=len(d), median=float(np.median(y)))
        ax.scatter(x, y, s=2, alpha=0.2, color='tab:orange', edgecolors='none')
        h = huber_slope(x, y, min_n=MIN_VISITS_LEG) if len(d) >= MIN_VISITS_LEG else None
        r['slope'] = h['slope'] if h else np.nan
        r['slope_err'] = h['slope_err'] if h else np.nan
        r['pearson_r'] = np.nan
        r['spearman_rho'] = np.nan
        if len(d) >= 3:
            r['pearson_r'] = float(stats.pearsonr(x, y)[0])
            r['spearman_rho'] = float(stats.spearmanr(x, y)[0])
            xg = np.linspace(x.min(), x.max(), 100)
            for i, order in enumerate(orders):
                try:
                    c = np.polyfit(x, y, order)
                except Exception:
                    continue
                r[f'resid_nmad_order{order}'] = float(nmad(y - np.polyval(c, x)))
                if i == 0:
                    ax.plot(xg, np.polyval(c, xg), 'k-', lw=1.6)
        rows.append(r)
        slope_txt = (f'{h["slope"]:+.2f} +/- {h["slope_err"]:.2f} um per deg'
                     if h else 'no fit')
        ax.set_title(f'{band}   n = {len(d)}\nHuber slope {slope_txt}\n'
                     f'Pearson r {r["pearson_r"]:+.3f}, '
                     f'Spearman rho {r["spearman_rho"]:+.3f}', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(bands):]:
        ax.axis('off')
    fig.supxlabel('elevation [deg]', fontsize=9)
    fig.supylabel(f'look-up-table term only [{unit}]', fontsize=9)
    fig.suptitle(f'elevation dependence already carried by the hexapod look-up table: the '
                 f'LUT term alone, no Trim and no thermal correction -- {variant}',
                 fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f'\nlook-up-table term alone against elevation (no Trim, no thermal '
              f'correction)')
        print(f'  {"band":5s} {"n":>7s} {"median":>9s} {"slope":>9s} {"err":>6s} '
              f'{"Pearson r":>10s} {"Spearman rho":>13s}')
        for _, r in out.iterrows():
            print(f'  {r.band:5s} {int(r.n):7d} {r["median"]:+9.1f} {r.slope:+9.2f} '
                  f'{r.slope_err:6.2f} {r.pearson_r:+10.3f} {r.spearman_rho:+13.3f}')
        print(f'  median and slope in {unit} and {unit} per deg; the correlations are '
              f'dimensionless')
    return out


def page_modulators(pdf, df, resid, variant, bands=BAND_ORDER,
                    modulators=MODULATOR_CANDIDATES):
    """One page per candidate modulator: the corrected residual against the factor, per band.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        The fitted sample.
    resid : `array_like`
        Out-of-fold residual after the thermal model [µm of equivalent hexapod dz], aligned
        with the rows of `df`.
    variant : `str`
        ``optical_state`` variant id, for the titles.
    bands : `sequence` [`str`], optional
    modulators : `sequence` [`tuple`], optional
        ``(column, label, unit)`` triples. A column absent from `df` is skipped.

    Notes
    -----
    These are the second-order factors tested for a residual dependence the five thermal
    channels do not carry. Each page is a per-band scatter with a Huber fit, so a real
    dependence shows as a slope reproduced across bands rather than in one panel.
    """
    d = df.copy()
    d['thermal_resid'] = np.asarray(resid, float)
    for col, label, unit in modulators:
        if col not in d.columns or not d[col].notna().any():
            continue
        page_grid_fits(pdf, d, col, 'thermal_resid', bands, f'{label} [{unit}]',
                       'residual after the thermal model\n[um of equivalent hexapod dz]',
                       f'Candidate modulator: {label} -- {variant}', unit,
                       order=1, ylim=RESID_YLIM_DZEQUIV)


# --------------------------------------------------------------------- opening page

def opening_page_meta(df, ev, full, cal, v1_per_um_dz, baseline_nmad):
    """Assemble the opening page: what the study is, how it works and what it finds.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The fitted sample, for the visit and night counts and the day_obs range.
    ev : `dict`
        From `evaluate`: the out-of-fold ``nmad`` [µm of equivalent hexapod dz] and ``r2``.
    full : `dict`
        From `fit_full`: the deliverable model's coefficients and physical intercept.
    cal : `pandas.DataFrame`
        From `prediction_calibration`, whose first row pools every band.
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm of hexapod dz].
    baseline_nmad : `float`
        Robust scatter of the uncorrected response [µm of equivalent hexapod dz].

    Returns
    -------
    meta : `dict`
        ``title``, ``subtitle`` and ``sections``, ready for `page_documentation`.
    """
    truss_um = float('nan')
    if full.get('coef') is not None and 'truss_temp_mean_c' in full['features']:
        truss_um = float(full['coef'][full['features'].index('truss_temp_mean_c')])
    truss_dimensionless = truss_um * v1_per_um_dz
    fam_pct = (100.0 * abs(truss_dimensionless - FAM_TRUSS_SLOPE) / abs(FAM_TRUSS_SLOPE)
               if np.isfinite(truss_dimensionless) else float('nan'))
    resid = float(ev['nmad'])
    frac_pct = 100.0 * resid / baseline_nmad if baseline_nmad > 0 else float('nan')
    pooled = cal.iloc[0] if len(cal) else None
    slope_txt = (f'{pooled.slope:+.3f}' if pooled is not None else 'n/a')

    sections = [
        ('The goal',
         'Predict the telescope\'s uniform-defocus error from temperature telemetry\n'
         'alone, so focus can be set open-loop from a look-up table instead of being\n'
         'driven by the wavefront sensors.'),
        ('What is measured',
         'For every science visit the Consolidated Database (ConsDB) records the\n'
         'optical state recovered at the four Corner Wavefront Sensors (CWFS). The\n'
         'quantity studied is the first singular vector of the Active Optics System\n'
         '(AOS) sensitivity matrix -- v-mode 1, essentially uniform defocus -- as\n'
         '\n'
         '    response = v1(commanded Trim) - v1(measured state)\n'
         '\n'
         'the focus error the closed loop had accumulated but not yet corrected. It\n'
         'is expressed throughout as equivalent hexapod dz [um]: the total defocus\n'
         'travel, shared as 0.5 um on the camera hexapod and 0.5 um on the M2\n'
         'hexapod.'),
        ('How it works',
         'The response is fitted against five thermal telemetry channels at once --\n'
         'the Telescope Mount Assembly (TMA) truss temperature and the four M1M3\n'
         'thermal gradients -- with one Huber robust linear model, band independent,\n'
         'evaluated on whole held-out nights.'),
        # Wrapped at run time, not by hand: every sentence carries an interpolated number, so a
        # hand-placed break lands in a different column each time the fit moves.
        ('What it finds', textwrap.fill(
            f'The five thermal channels predict the focus error to {resid:.1f} um of '
            f'equivalent hexapod dz, from an uncorrected {baseline_nmad:.1f} um: '
            f'{frac_pct:.0f}% of the original scatter. The truss temperature carries most '
            f'of it at {truss_um:+.1f} um per deg C, which agrees to {fam_pct:.1f}% with '
            f'the independent Full Array Mode (FAM) measurement of {FAM_TRUSS_SLOPE:+.5f} '
            f'dimensionless v-mode-1 amplitude per deg C. The estimate is calibrated: the '
            f'robust slope of response against prediction is {slope_txt} (dimensionless, '
            f'measured over predicted). Once the thermal correction is applied no elevation '
            f'dependence remains, so temperature alone sets the table.', width=70)),
        ('Sample',
         f'{len(df)} visits over {df.day_obs.nunique()} nights, '
         f'{int(df.day_obs.min())} to {int(df.day_obs.max())}, '
         f'bands {" ".join(BAND_ORDER)}.'),
    ]
    return dict(title='A focus look-up table from science exposures',
                subtitle=None, sections=sections)


# --------------------------------------------------------------------------- driver

def build_pdf(out_pdf, df, features, ev, full, coef_tab, cal, band_resid, steps,
              slopes, v1_per_um_dz, variant, model_cmp=None, ablation=None,
              day_obs_series=DAY_OBS_SERIES, all_nights=False, modulators=False,
              ref_elev_deg=REF_ELEV_DEG, verbose=True):
    """Write the whole document, one page per section of the argument.

    Parameters
    ----------
    out_pdf : `pathlib.Path`
        Destination PDF.
    df : `pandas.DataFrame`
        The fitted sample from `load_target`, carrying ``direction`` and ``thermal_resid``.
    features : `list` [`str`]
    ev : `dict`
        From `evaluate`.
    full : `dict`
        From `fit_full`.
    coef_tab : `pandas.DataFrame`
        From `coefficient_table`.
    cal : `pandas.DataFrame`
        From `prediction_calibration`.
    band_resid : `pandas.DataFrame`
        From `per_band_residual`.
    steps : `list` [`tuple` [`str`, `dict`]]
        ``(label, result)`` pairs from `filter_change_step`, uncorrected first.
    slopes : `pandas.DataFrame`
        From `per_night_direction_slopes` on the corrected residual.
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm of hexapod dz].
    variant : `str`
        ``optical_state`` variant id.
    model_cmp, ablation : `pandas.DataFrame`, optional
        From `compare_models` and `compare_feature_sets`. Omitted with ``--no-model-scan``.
    day_obs_series : `sequence` [`int`], optional
        Nights drawn as a visit-by-visit series, three pages each.
    all_nights : `bool`, optional
        Add the per-night elevation grid, ``ceil(n_nights / NIGHTS_PER_PAGE)`` pages.
    modulators : `bool`, optional
        Add one page per candidate second-order modulator.
    ref_elev_deg : `float`, optional
        Elevation the per-night offset was evaluated at [deg].
    verbose : `bool`, optional

    Returns
    -------
    n_pages : `int`
        Pages written.

    Notes
    -----
    The page order is the order of the argument: what the sample is, what the response looks
    like uncorrected, the model and its coefficients, how well the estimate tracks the
    response, what the residual still contains, and finally the elevation null result and the
    per-night behaviour.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    baseline = float(nmad(df.y.to_numpy(float)))
    resid = df.thermal_resid.to_numpy(float)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # 1. What the study is.
        page_documentation(pdf, opening_page_meta(df, ev, full, cal, v1_per_um_dz, baseline))

        # 2-3. The sample and the uncorrected response.
        page_visits_per_night(pdf, df, BAND_ORDER)
        page_response_histograms(pdf, df, v1_per_um_dz)

        # 4-5. The uncorrected response against the two drivers, per band.
        page_grid_fits(pdf, df, 'truss_temp_mean_c', 'y', BAND_ORDER,
                       'TMA truss temperature [deg C]',
                       'v1(Trim) - v1(measured)\n[um of equivalent hexapod dz]',
                       f'Uncorrected response against the truss temperature -- {variant}',
                       'deg C', order=1, ylim=GRID_YLIM_DZEQUIV)
        page_grid_fits(pdf, df, 'altitude_deg', 'y', BAND_ORDER, 'elevation [deg]',
                       'v1(Trim) - v1(measured)\n[um of equivalent hexapod dz]',
                       f'Uncorrected response against elevation -- {variant}', 'deg',
                       order=1, fit_range=ELEV_FIT_RANGE, ylim=GRID_YLIM_DZEQUIV)

        # 6-7. The model itself.
        page_model_equation(pdf, full, coef_tab, features, df, v1_per_um_dz, ev)
        if ev.get('coefs') is not None:
            page_coefficients(pdf, ev['coefs'], features, v1_per_um_dz)

        # 8-9. How well the thermal estimate tracks the response.
        page_prediction_vs_response(pdf, df, ev['pred'], cal)
        page_prediction_vs_response_bands(pdf, df, ev['pred'], cal)

        # 10-11. What the residual still contains.
        page_band_residual(pdf, df, resid, baseline=baseline)
        page_resid_vs_features(pdf, df, features, resid)

        # 12. The elevation null result: drawn, never subtracted.
        d = df.copy()
        d['thermal_resid'] = resid
        # The per-band panels are the picture; the pooled fit is the claim, so it goes in the
        # page title rather than being left for the reader to combine by eye.
        pooled = huber_fit(df.altitude_deg.to_numpy(float), resid)
        null_note = ''
        if pooled is not None:
            flat = resid - (pooled['intercept']
                            + pooled['slope'] * df.altitude_deg.to_numpy(float))
            null_note = ('\n' + textwrap.fill(
                f'pooled over all bands: slope {pooled["slope"]:+.3f} +/- '
                f'{pooled["slope_err"]:.3f} um of equivalent hexapod dz per deg, Pearson r '
                f'{pooled["pearson_r"]:+.3f} -- removing it moves the residual nMAD only from '
                f'{nmad(resid):.1f} to {nmad(flat):.1f} um, so no elevation term is '
                f'subtracted', width=118))
        page_grid_fits(pdf, d, 'altitude_deg', 'thermal_resid', BAND_ORDER,
                       'elevation [deg]',
                       'residual after the thermal model\n[um of equivalent hexapod dz]',
                       f'Residual against elevation after the thermal correction: no '
                       f'dependence remains -- {variant}{null_note}', 'deg', order=1,
                       fit_range=ELEV_FIT_RANGE, ylim=RESID_YLIM_DZEQUIV)

        # 13. Model and feature-set comparison.
        if model_cmp is not None or ablation is not None:
            blocks = []
            if model_cmp is not None:
                blocks.append(('Model comparison, night-grouped out-of-fold',
                               _table_text(model_cmp,
                                           ['model', 'resid_nmad', 'frac_of_baseline', 'r2'],
                                           ['s', '.1f', '.3f', '.3f'],
                                           [14, 11, 17, 7])
                               + f'\n\nresid_nmad in um of equivalent hexapod dz; '
                                 f'frac_of_baseline and R2\nare dimensionless, the fraction '
                                 f'being residual nMAD over the\nuncorrected '
                                 f'{baseline:.1f} um.'))
            if ablation is not None:
                blocks.append(('Feature-set ablation, same folds',
                               _table_text(ablation,
                                           ['groups', 'n_features', 'resid_nmad',
                                            'frac_of_baseline', 'r2'],
                                           ['s', 'd', '.1f', '.3f', '.3f'],
                                           [22, 11, 11, 17, 7])))
            page_text(pdf, 'Model choice and feature contribution', blocks)

        # 14. The filter-change step.
        rows = []
        for label, r in steps:
            rows.append(dict(series=label, n_change=r['n_change'],
                             median_change=r['median_change'], n_same=r['n_same'],
                             median_same=r['median_same'], ratio=r['ratio']))
        page_text(pdf, 'Focus step across a filter change',
                  [('Median |step| between consecutive visits within a night',
                    _table_text(pd.DataFrame(rows),
                                ['series', 'n_change', 'median_change', 'n_same',
                                 'median_same', 'ratio'],
                                ['s', 'd', '.1f', 'd', '.1f', '.2f'],
                                [24, 10, 15, 8, 13, 7])
                    + '\n\nmedian_change and median_same in um of equivalent hexapod dz; '
                      'ratio is\nthe band-change step over the same-band step '
                      '(dimensionless). A band-\nindependent correction cannot remove a '
                      'per-band focus offset, so an\nexcess surviving the correction is a '
                      'real offset between filters.')])

        # 15-17. One night in detail.
        for day in day_obs_series:
            if day not in set(df.day_obs):
                if verbose:
                    print(f'day_obs {day} is not in the sample; its series pages are skipped')
                continue
            page_night_series(pdf, df, resid, day, features)
            page_night_temperatures(pdf, df, day, features)

        # 18-21. Per-night behaviour of the corrected residual.
        page_night_offsets(pdf, slopes, ycol='thermal_resid', ref_elev_deg=ref_elev_deg)
        page_night_slopes(pdf, slopes, ycol='thermal_resid')
        if all_nights:
            page_night_panels(pdf, d, slopes, ycol='thermal_resid')

        # 22. The measured state and the look-up table on their own.
        page_measured_state(pdf, df, variant, verbose=verbose)
        if 'lut_dzequiv' in df.columns and df.lut_dzequiv.notna().any():
            page_lut_vs_elevation(pdf, df, variant, verbose=verbose)

        # Optional: the second-order factors.
        if modulators:
            page_modulators(pdf, df, resid, variant)

        n_pages = pdf.get_pagecount()
    if verbose:
        print(f'\nwrote {out_pdf} ({n_pages} pages)')
    return n_pages


def write_tables(out_dir, df, features, ev, full, coef_tab, slopes, v1_per_um_dz, variant,
                 verbose=True):
    """Write the three tables the document is built from.

    Parameters
    ----------
    out_dir : `pathlib.Path`
        Destination directory.
    df : `pandas.DataFrame`
        The fitted sample, carrying ``thermal_resid``.
    features : `list` [`str`]
    ev : `dict`
        From `evaluate`.
    full : `dict`
        From `fit_full`.
    coef_tab : `pandas.DataFrame`
        From `coefficient_table`.
    slopes : `pandas.DataFrame`
        From `per_night_direction_slopes`.
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm of hexapod dz].
    variant : `str`
    verbose : `bool`, optional

    Returns
    -------
    paths : `dict`
        ``model``, ``predictions`` and ``nights`` mapped to the files written.

    Notes
    -----
    ``science_lut_model.parquet`` carries the deliverable coefficients in physical units --
    the intercept is the response at zero in every feature, already shifted off the
    standardized scale -- so it can be read straight into a look-up table without refitting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, f in enumerate(features):
        r = coef_tab[coef_tab.feature == f]
        rows.append(dict(
            variant=variant, term=f, unit=f'um of equivalent hexapod dz per '
                                        f'{FEATURE_UNITS.get(f, "?")}',
            coef_full=(float(full['coef'][i]) if full.get('coef') is not None else np.nan),
            coef_fold_mean=(float(r['mean'].iloc[0]) if len(r) else np.nan),
            coef_fold_std=(float(r['std'].iloc[0]) if len(r) else np.nan),
            feature_mean=float(df[f].mean()), feature_unit=FEATURE_UNITS.get(f, '?')))
    rows.append(dict(variant=variant, term='intercept',
                     unit='um of equivalent hexapod dz',
                     coef_full=float(full.get('intercept', np.nan)),
                     coef_fold_mean=np.nan, coef_fold_std=np.nan,
                     feature_mean=np.nan, feature_unit='dimensionless'))
    model = pd.DataFrame(rows)
    model['v1_per_um_dz'] = v1_per_um_dz
    model['n_visits'] = len(df)
    model['n_nights'] = int(df.day_obs.nunique())
    model['resid_nmad_oof'] = float(ev['nmad'])
    model['r2_oof'] = float(ev['r2'])
    model['baseline_nmad'] = float(nmad(df.y.to_numpy(float)))
    model['equation_max_abs_diff'] = float(full.get('equation_max_abs_diff', np.nan))

    pred_cols = ['visit_id', 'day_obs', 'seq_num', 'band', 'altitude_deg', 'y']
    predictions = df[[c for c in pred_cols if c in df.columns]].copy()
    predictions['pred'] = np.asarray(ev['pred'], float)
    predictions['thermal_resid'] = df.thermal_resid.to_numpy(float)
    predictions['variant'] = variant

    paths = {}
    for key, name, tab in (('model', 'science_lut_model.parquet', model),
                           ('predictions', 'science_lut_predictions.parquet', predictions),
                           ('nights', 'science_lut_nights.parquet', slopes)):
        path = out_dir / name
        tab.to_parquet(path, index=False)
        paths[key] = path
        if verbose:
            print(f'wrote {path} ({len(tab)} rows)')
    return paths


def main(argv=None):
    """Run the analysis and write one PDF plus its three tables.

    Parameters
    ----------
    argv : `list` [`str`], optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    code : `int`
        0 on success.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in-dir', type=pathlib.Path, default=None,
                   help='directory holding science_lut.parquet; default '
                        'aos/output/science_lut')
    p.add_argument('--out', type=pathlib.Path, default=None,
                   help='destination PDF; default <out-dir>/science_lut_analysis.pdf')
    p.add_argument('--out-dir', type=pathlib.Path, default=None,
                   help='directory for the PDF and the three tables; default '
                        'aos/output/science_lut')
    p.add_argument('--variant', default=DEFAULT_VARIANT,
                   help='optical_state variant id (default: %(default)s)')
    p.add_argument('--features', nargs='+', default=list(DEFAULT_FEATURES),
                   choices=sorted(FEATURE_GROUPS), metavar='GROUP',
                   help=f'feature groups to fit; choose from '
                        f'{", ".join(sorted(FEATURE_GROUPS))} (default: '
                        f'{" ".join(DEFAULT_FEATURES)})')
    p.add_argument('--model', default='huber', choices=MODEL_NAMES,
                   help='regressor (default: %(default)s)')
    p.add_argument('--n-splits', type=int, default=5,
                   help='night-grouped folds (default: %(default)s)')
    p.add_argument('--leaky-split', action='store_true',
                   help='shuffle visits across nights instead of holding whole nights out; '
                        'inflates the score by about a factor LEAK_FACTOR and is for '
                        'demonstrating the leak, not for reporting')
    p.add_argument('--keep-lut-epoch-offset-nights', action='store_true',
                   help=f'keep the {len(LUT_EPOCH_OFFSET_NIGHTS)} nights running a different '
                        f'hexapod LUT configuration, dropped by default')
    p.add_argument('--v1-per-um-dz', type=float, default=None,
                   help='override the dimensionless v-mode-1 amplitude per um of hexapod dz')
    p.add_argument('--ref-elev-deg', type=float, default=REF_ELEV_DEG,
                   help='elevation the per-night offset is evaluated at [deg] '
                        '(default: %(default)s)')
    p.add_argument('--day-obs', type=int, nargs='+', default=list(DAY_OBS_SERIES),
                   help='nights drawn as a visit-by-visit series, two pages each '
                        '(default: %(default)s)')
    p.add_argument('--all-nights', action='store_true',
                   help='add the per-night elevation grid, 12 panels per page')
    p.add_argument('--modulators', action='store_true',
                   help='add one page per candidate second-order modulator')
    p.add_argument('--no-model-scan', action='store_true',
                   help='skip the model comparison and feature ablation, which refit every '
                        'model on every fold')
    p.add_argument('--quiet', action='store_true', help='suppress the printed tables')
    a = p.parse_args(argv)

    verbose = not a.quiet
    default_dir = _ROOT / 'aos' / 'output' / 'science_lut'
    in_dir = a.in_dir or default_dir
    out_dir = a.out_dir or default_dir
    out_pdf = a.out or (out_dir / 'science_lut_analysis.pdf')

    v1_per_um_dz = (a.v1_per_um_dz if a.v1_per_um_dz is not None
                    else v1_per_um_dz_value(verbose=verbose))
    features = resolve_features(a.features)
    # The ablation and the modulator pages fit columns the model itself does not use, so they are
    # carried alongside the features rather than added to them.
    extra = list(dict.fromkeys(
        [c for gs in ABLATION_SETS for c in resolve_features(gs)]
        + [c for c, _, _ in MODULATOR_CANDIDATES]))
    df, features = load_target(in_dir, variant=a.variant, v1_per_um_dz=v1_per_um_dz,
                               features=features, extra_cols=extra,
                               drop_lut_epoch=not a.keep_lut_epoch_offset_nights,
                               verbose=verbose)

    ev = evaluate(df, features, model=a.model, n_splits=a.n_splits, leaky=a.leaky_split,
                  verbose=verbose)
    full = fit_full(df, features, model=a.model, verbose=verbose)
    df['thermal_resid'] = np.asarray(ev['resid'], float)

    coef_tab = (coefficient_table(ev['coefs'], features, v1_per_um_dz, verbose=verbose)
                if ev.get('coefs') is not None else pd.DataFrame(columns=['feature']))
    cal = prediction_calibration(df, ev['pred'], verbose=verbose)
    band_resid = per_band_residual(df, df.thermal_resid.to_numpy(float), verbose=verbose)
    steps = [('uncorrected response', filter_change_step(df, 'y', verbose=verbose,
                                                         label='uncorrected response')),
             (f'after the {a.model} thermal model',
              filter_change_step(df, 'thermal_resid', verbose=verbose,
                                 label=f'after the {a.model} thermal model'))]

    model_cmp = ablation = None
    if not a.no_model_scan:
        model_cmp = compare_models(df, features, n_splits=a.n_splits, verbose=verbose)
        ablation = compare_feature_sets(df, ABLATION_SETS, model=a.model,
                                        n_splits=a.n_splits, verbose=verbose)

    # The per-night pages describe the residual that survives the thermal model, so the
    # direction labels and the slopes are computed on that column, not on the raw response.
    # Per night, so the centred rolling median never spans the daytime gap.
    df['direction'] = pd.concat(
        [label_direction(d) for _, d in df.groupby('day_obs')]).reindex(df.index)
    slopes = per_night_direction_slopes(df, ycol='thermal_resid',
                                       ref_elev_deg=a.ref_elev_deg, verbose=verbose)

    build_pdf(out_pdf, df, features, ev, full, coef_tab, cal, band_resid, steps, slopes,
              v1_per_um_dz, a.variant, model_cmp=model_cmp, ablation=ablation,
              day_obs_series=a.day_obs, all_nights=a.all_nights, modulators=a.modulators,
              ref_elev_deg=a.ref_elev_deg, verbose=verbose)
    write_tables(out_dir, df, features, ev, full, coef_tab, slopes, v1_per_um_dz, a.variant,
                 verbose=verbose)
    return 0


if __name__ == '__main__':
    sys.exit(main())
