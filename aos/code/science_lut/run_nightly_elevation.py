"""Per-night structure of the truss-corrected v-mode-1 elevation dependence.

The pooled fit over all nights gives one elevation slope near -0.014 dimensionless
v-mode-1 amplitude per deg, but the night-to-night spread of that slope is large: the
question this script addresses is whether each night has its own slope, whether a night
behaves differently going up in elevation than coming down (hysteresis), and what
telemetry the per-night slope correlates with.

Four analyses, in order:

1. **Per-night slopes.** A robust (Huber) fit of the truss-corrected residual against
   elevation, one night at a time, with the distribution of those slopes.
2. **Up versus down.** Each night split into rising and falling elevation legs, fitted
   separately. A systematic up/down difference is hysteresis -- backlash or thermal lag --
   rather than a static flexure the look-up table could absorb.
3. **Median trend.** The elevation dependence of the median residual in elevation bins,
   pooled over nights, which is the shape a look-up table would have to reproduce and is
   insensitive to how any one night is sampled.
4. **What the per-night slope correlates with.** The 52 per-night slopes are correlated
   against per-night telemetry summaries, ranked by Spearman rho.

Run from ``aos/``::

    python code/science_lut/run_nightly_elevation.py --variant v50_34__batoid__consdb_v1

Reads ``output/science_lut/science_lut.parquet`` written by ``run_science_lut.py``, so it
needs no Engineering Facility Database (EFD) or Consolidated Database (ConsDB) access.
Writes ``nightly_elevation.pdf``, ``nightly_elevation_slopes.parquet`` and
``nightly_elevation_correlations.parquet`` alongside it.

Notes
-----
Slew direction cannot be taken from the sign of the per-visit elevation difference:
consecutive science visits are about 0.7 min apart, the median absolute step is under
2 deg, and the sign flips constantly while tracking a field. The direction is therefore
taken from a centred rolling median of elevation (default 21 visits) with a deadband, which
recovers tens of genuine legs per night rather than hundreds of spurious reversals.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from common.utils import nmad                                    # noqa: E402

# Telemetry summarized per night and correlated against the per-night elevation slope.
# Each entry is (column, label, unit, how) where `how` is the per-night reduction.
NIGHT_FACTORS = [
    ('truss_temp_mean_c', 'TMA truss temperature, night median', 'deg C', 'median'),
    ('truss_temp_mean_c', 'TMA truss temperature, night range', 'deg C', 'ptp'),
    ('m1m3_x_gradient_c_per_m', 'M1M3 x thermal gradient, median', 'deg C/m', 'median'),
    ('m1m3_y_gradient_c_per_m', 'M1M3 y thermal gradient, median', 'deg C/m', 'median'),
    ('m1m3_z_gradient_c_per_m', 'M1M3 z thermal gradient, median', 'deg C/m', 'median'),
    ('m1m3_radial_gradient_c_per_m', 'M1M3 radial thermal gradient, median',
     'deg C/m', 'median'),
    ('cam_air_temp', 'camera air temperature, median', 'deg C', 'median'),
    ('m2_air_temp', 'M2 air temperature, median', 'deg C', 'median'),
    ('m1m3_air_temp', 'M1M3 air temperature, median', 'deg C', 'median'),
    ('outside_temp', 'outside temperature, median', 'deg C', 'median'),
    ('cam_AverageTemp', 'camera body temperature, median', 'deg C', 'median'),
    ('wind_speed_ms', 'weather-station wind speed, median', 'm/s', 'median'),
    ('wind_inside_maxmag', 'dome inside wind, median of maximum magnitude',
     'm/s', 'median'),
    ('into_wind_deg', 'into-wind angle, median absolute', 'deg', 'absmedian'),
    ('cum_hex_dz_um', 'cumulative |hexapod dz|, night maximum', 'um', 'max'),
    ('n_moves_night', 'commanded hexapod moves, night maximum', 'count', 'max'),
    ('altitude_deg', 'elevation range covered', 'deg', 'ptp'),
    ('altitude_deg', 'elevation, night median', 'deg', 'median'),
    ('psf_sigma_median', 'PSF sigma, night median', 'pixels', 'median'),
    ('seeing_zenith_500nm_median', 'zenith seeing at 500 nm, median', 'arcsec', 'median'),
]

MIN_VISITS_NIGHT = 40      # a per-night slope below this is not worth fitting
MIN_VISITS_LEG = 25        # likewise for one direction of one night
DIRECTION_WINDOW = 21      # visits in the centred rolling median of elevation
DIRECTION_DEADBAND = 0.02  # deg per visit; smaller excursions count as tracking, not slewing

# A night whose robust residual scatter exceeds this is excluded from the slope
# distribution and from the telemetry correlations. Typical nights sit at 0.03-0.12
# dimensionless v-mode-1 amplitude, so this cut removes only nights where the optical-state
# recovery itself failed rather than nights with an unusual elevation dependence.
MAX_NIGHT_RESID_NMAD = 0.35

# The pooled elevation trend is curved: the local slope steepens towards the horizon, so a
# night sampling only high elevation measures a shallower slope than one reaching the
# horizon even if the underlying behaviour is identical. Per-night slopes are therefore also
# reported over this common elevation window, where every retained night has coverage.
COMMON_ELEV = (35.0, 75.0)


def huber_slope(x, y, min_n=MIN_VISITS_LEG):
    """Robust straight-line fit, returning the slope and its standard error.

    Parameters
    ----------
    x, y : `array_like`
        Predictor and response; `x` in deg of elevation, `y` a dimensionless v-mode
        amplitude.
    min_n : `int`, optional
        Return None below this many finite pairs.

    Returns
    -------
    out : `dict` or `None`
        ``n``, ``slope`` and ``slope_err`` [response unit per deg], ``intercept``,
        ``pearson_r``, ``spearman_rho``, ``resid_nmad``, ``elev_min``, ``elev_max``.
        None if under-determined or if elevation spans under 5 deg, where a slope is
        an extrapolation rather than a measurement.
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
        One night, any order; needs `obs_start_mjd` and `altitude_deg`.
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
    The rolling median is what makes this meaningful. Consecutive visits are ~0.7 min
    apart with sub-2 deg steps whose raw sign alternates while tracking, so a per-visit
    difference would report hundreds of direction changes per night instead of the few
    tens of real elevation legs.
    """
    d = df.sort_values('obs_start_mjd')
    sm_el = (d['altitude_deg'].rolling(window, center=True, min_periods=3)
             .median())
    de = sm_el.diff()
    out = pd.Series('flat', index=d.index, dtype=object)
    out[de > deadband] = 'up'
    out[de < -deadband] = 'down'
    return out.reindex(df.index)


def per_night_slopes(df, ycol, verbose=True):
    """Fit the elevation dependence one night at a time, and per direction.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit table with `day_obs`, `altitude_deg`, `direction` and `ycol`.
    ycol : `str`
        Residual column [dimensionless v-mode-1 amplitude].
    verbose : `bool`, optional

    Returns
    -------
    rows : `pandas.DataFrame`
        One row per (day_obs, direction), direction being ``'all'``, ``'up'`` or
        ``'down'``, with the slope in dimensionless v-mode-1 amplitude per deg.
    """
    rows = []
    for day, d in df.groupby('day_obs'):
        if len(d) < MIN_VISITS_NIGHT:
            continue
        # The same fit restricted to the common elevation window, so that nights sampling
        # different elevation ranges are compared over identical leverage.
        dc = d[d.altitude_deg.between(*COMMON_ELEV)]
        fc = huber_slope(dc['altitude_deg'], dc[ycol], min_n=MIN_VISITS_NIGHT)
        for direction in ('all', 'up', 'down'):
            sub = d if direction == 'all' else d[d['direction'] == direction]
            f = huber_slope(sub['altitude_deg'], sub[ycol],
                            min_n=MIN_VISITS_NIGHT if direction == 'all'
                            else MIN_VISITS_LEG)
            if f is None:
                continue
            rows.append(dict(
                day_obs=int(day), direction=direction, **f,
                slope_common=(fc['slope'] if fc else np.nan),
                slope_common_err=(fc['slope_err'] if fc else np.nan),
                n_common=(fc['n'] if fc else 0)))
    out = pd.DataFrame(rows)
    if len(out):
        # Flag rather than drop, so the excluded nights stay visible in the parquet and in
        # the per-night panel pages.
        bad = set(out.loc[(out.direction == 'all')
                          & (out.resid_nmad > MAX_NIGHT_RESID_NMAD), 'day_obs'])
        out['night_ok'] = ~out.day_obs.isin(bad)
        if verbose and bad:
            print(f'\nnights excluded for a robust residual scatter above '
                  f'{MAX_NIGHT_RESID_NMAD} dimensionless v-mode-1 amplitude:')
            for day in sorted(bad):
                r = out[(out.day_obs == day) & (out.direction == 'all')].iloc[0]
                print(f'  {day}: resid nMAD {r.resid_nmad:.3f}, slope {r.slope:+.5f} '
                      f'per deg, Pearson r {r.pearson_r:+.3f}, n = {int(r.n)} '
                      f'-- the optical-state recovery, not the elevation dependence')
    if verbose and len(out):
        a = out[(out.direction == 'all') & out.night_ok]
        print(f'\nper-night elevation slopes of {ycol} '
              f'[dimensionless v-mode-1 amplitude per deg]')
        print(f'  nights fitted            : {len(a)}')
        print(f'  median slope             : {a.slope.median():+.5f} per deg')
        print(f'  nMAD of the slopes       : {nmad(a.slope.to_numpy()):.5f} per deg')
        print(f'  full range               : {a.slope.min():+.5f} to '
              f'{a.slope.max():+.5f} per deg')
        print(f'  median formal error      : {a.slope_err.median():.5f} per deg')
        spread_ratio = (nmad(a.slope.to_numpy()) / a.slope_err.median()
                        if a.slope_err.median() > 0 else np.nan)
        print(f'  night-to-night spread over the median formal error: '
              f'{spread_ratio:.1f} (dimensionless)')
        if spread_ratio > 3:
            print('  -> the spread is far larger than the fit errors, so the nights '
                  'genuinely differ; a single static elevation slope does not describe '
                  'the data')
        n_neg = int((a.slope < 0).sum())
        print(f'  nights with a negative slope: {n_neg} of {len(a)}')
        c = a[np.isfinite(a.slope_common)]
        if len(c):
            print(f'\n  restricted to the common window '
                  f'{COMMON_ELEV[0]:.0f}-{COMMON_ELEV[1]:.0f} deg '
                  f'({len(c)} nights):')
            print(f'    median slope       : {c.slope_common.median():+.5f} per deg')
            print(f'    nMAD of the slopes : '
                  f'{nmad(c.slope_common.to_numpy()):.5f} per deg')
            print(f'    spread over the median formal error: '
                  f'{nmad(c.slope_common.to_numpy()) / c.slope_common_err.median():.1f} '
                  f'(dimensionless)')
            print('    -- if the spread shrinks here, part of the night-to-night '
                  'variation was nights\n       sampling different elevation ranges '
                  'against a curved trend, not a real difference')
    return out


def updown_summary(slopes, verbose=True):
    """Compare the rising-leg and falling-leg slope on each night.

    Parameters
    ----------
    slopes : `pandas.DataFrame`
        Output of `per_night_slopes`.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        One row per night having both directions, with ``slope_up``, ``slope_down``,
        their difference [dimensionless v-mode-1 amplitude per deg] and the difference
        in units of the combined standard error.
    """
    ok = slopes[slopes.night_ok] if 'night_ok' in slopes.columns else slopes
    up = ok[ok.direction == 'up'].set_index('day_obs')
    dn = ok[ok.direction == 'down'].set_index('day_obs')
    common = up.index.intersection(dn.index)
    if not len(common):
        return pd.DataFrame()
    out = pd.DataFrame({
        'day_obs': common,
        'n_up': up.loc[common, 'n'].to_numpy(),
        'n_down': dn.loc[common, 'n'].to_numpy(),
        'slope_up': up.loc[common, 'slope'].to_numpy(),
        'slope_down': dn.loc[common, 'slope'].to_numpy(),
        'err_up': up.loc[common, 'slope_err'].to_numpy(),
        'err_down': dn.loc[common, 'slope_err'].to_numpy(),
    })
    out['difference'] = out.slope_up - out.slope_down
    comb = np.sqrt(out.err_up ** 2 + out.err_down ** 2)
    out['difference_sigma'] = out.difference / comb.replace(0, np.nan)
    if verbose:
        med = out.difference.median()
        print(f'\nrising minus falling elevation slope, {len(out)} nights with both')
        print(f'  median difference   : {med:+.5f} dimensionless v-mode-1 amplitude '
              f'per deg')
        print(f'  nMAD of differences : {nmad(out.difference.to_numpy()):.5f} per deg')
        n_sig = int((out.difference_sigma.abs() > 3).sum())
        print(f'  nights differing by more than 3 combined standard errors: '
              f'{n_sig} of {len(out)}')
        # Sign test: is the up-down difference consistent in sign across nights, which is
        # what hysteresis would produce, or symmetric scatter?
        from scipy import stats
        pos = int((out.difference > 0).sum())
        p = stats.binomtest(pos, len(out), 0.5).pvalue if len(out) else np.nan
        print(f'  nights with up steeper than down: {pos} of {len(out)} '
              f'(sign-test p = {p:.3g})')
        if p < 0.01:
            print('  -> a consistent direction-dependent offset, i.e. hysteresis, not '
                  'symmetric night-to-night scatter')
        else:
            print('  -> no consistent direction dependence; the up/down difference '
                  'scatters about zero')
    return out


def median_trend(df, ycol, bands, n_bins=14, verbose=True):
    """Median residual in elevation bins, pooled over nights.

    Parameters
    ----------
    df : `pandas.DataFrame`
    ycol : `str`
    bands : `list` [`str`]
    n_bins : `int`, optional
        Equal-count elevation bins.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per bin and per band: elevation centre [deg], median residual [dimensionless],
        `nmad`, and n. Band ``'all'`` pools the bands, which is legitimate here because
        the residual has already had its per-band truss fit removed.
    """
    rows = []
    for b in list(bands) + ['all']:
        d = df if b == 'all' else df[df.band == b]
        d = d[np.isfinite(d[ycol]) & np.isfinite(d.altitude_deg)]
        if len(d) < n_bins * 10:
            continue
        try:
            cut = pd.qcut(d.altitude_deg, n_bins, duplicates='drop')
        except ValueError:
            continue
        for interval, g in d.groupby(cut, observed=True):
            rows.append(dict(
                band=b, elev_centre=float(interval.mid),
                elev_lo=float(interval.left), elev_hi=float(interval.right),
                median=float(np.median(g[ycol])),
                nmad=float(nmad(g[ycol].to_numpy())), n=int(len(g))))
    out = pd.DataFrame(rows)
    if verbose and len(out):
        a = out[out.band == 'all']
        print(f'\nmedian {ycol} in {len(a)} equal-count elevation bins, bands pooled')
        print(f'  {"elev [deg]":>12s} {"median":>10s} {"nMAD":>9s} {"n":>7s}')
        for _, r in a.iterrows():
            print(f'  {r.elev_centre:12.1f} {r["median"]:+10.4f} {r["nmad"]:9.4f} '
                  f'{int(r.n):7d}')
        print(f'  peak-to-peak of the binned median: '
              f'{a["median"].max() - a["median"].min():.4f} dimensionless v-mode-1 '
              f'amplitude over {a.elev_centre.min():.0f} to {a.elev_centre.max():.0f} deg')
        # Is the pooled trend a straight line, or does it steepen towards the horizon?
        x = a.elev_centre.to_numpy()
        y = a['median'].to_numpy()
        c1 = np.polyfit(x, y, 1)
        c2 = np.polyfit(x, y, 2)
        rms1 = float(np.std(y - np.polyval(c1, x)))
        rms2 = float(np.std(y - np.polyval(c2, x)))
        print(f'\n  shape of the binned median trend:')
        print(f'    linear    : slope {c1[0]:+.5f} per deg, residual RMS {rms1:.5f} '
              f'dimensionless')
        print(f'    quadratic : curvature {c2[0]:+.3e} per deg^2, residual RMS '
              f'{rms2:.5f} dimensionless')
        lo = 2 * c2[0] * x.min() + c2[1]
        hi = 2 * c2[0] * x.max() + c2[1]
        print(f'    local slope from the quadratic: {lo:+.5f} per deg at '
              f'{x.min():.0f} deg elevation, {hi:+.5f} per deg at {x.max():.0f} deg')
        if rms2 < 0.7 * rms1:
            print(f'    -> the trend is curved, steepening towards the horizon by a '
                  f'factor of {lo / hi:.1f} (dimensionless);')
            print('       a single straight-line elevation slope under-describes it, and a '
                  'night sampling only')
            print('       high elevation necessarily measures a shallower slope than one '
                  'reaching the horizon')
    return out


def night_summaries(df, factors=NIGHT_FACTORS):
    """Reduce the per-visit telemetry to one value per night.

    Parameters
    ----------
    df : `pandas.DataFrame`
    factors : `list` [`tuple`], optional
        ``(column, label, unit, how)`` per factor.

    Returns
    -------
    out : `pandas.DataFrame`
        Indexed by `day_obs`, one column per factor, named ``<column>__<how>``.
    """
    out = {}
    for col, _label, _unit, how in factors:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce')
        g = s.groupby(df.day_obs)
        if how == 'median':
            v = g.median()
        elif how == 'max':
            v = g.max()
        elif how == 'ptp':
            v = g.max() - g.min()
        elif how == 'absmedian':
            v = s.abs().groupby(df.day_obs).median()
        else:
            continue
        out[f'{col}__{how}'] = v
    return pd.DataFrame(out)


def correlate_slopes(slopes, summ, factors=NIGHT_FACTORS, slope_col='slope',
                     verbose=True):
    """Correlate the per-night elevation slope against per-night telemetry.

    Parameters
    ----------
    slopes : `pandas.DataFrame`
        Output of `per_night_slopes`; only the ``'all'`` rows are used.
    summ : `pandas.DataFrame`
        Output of `night_summaries`.
    factors : `list` [`tuple`], optional
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        One row per factor: Pearson r, Spearman rho, n nights, and the Huber slope of
        the per-night elevation slope against that factor, ranked by |Spearman rho|.

    Notes
    -----
    With of order 50 nights these are weak tests, and the factors are themselves
    correlated (every temperature tracks every other). A high rho identifies a candidate
    to model, not a cause.
    """
    from scipy import stats

    ok = slopes[slopes.night_ok] if 'night_ok' in slopes.columns else slopes
    a = ok[ok.direction == 'all'].set_index('day_obs')
    rows = []
    for col, label, unit, how in factors:
        key = f'{col}__{how}'
        if key not in summ.columns:
            continue
        j = pd.DataFrame({'slope': a[slope_col]}).join(summ[[key]], how='inner').dropna()
        if len(j) < 10:
            continue
        x = j[key].to_numpy(float)
        y = j.slope.to_numpy(float)
        if np.ptp(x) == 0:
            continue
        f = huber_slope(x, y, min_n=10) if np.ptp(x) >= 5.0 else None
        rho = float(stats.spearmanr(x, y)[0])
        t = rho * np.sqrt((len(j) - 2) / max(1e-12, 1 - rho ** 2))
        rows.append(dict(
            factor=key, label=label, unit=unit, n_nights=len(j),
            pearson_r=float(stats.pearsonr(x, y)[0]),
            spearman_rho=rho,
            spearman_p=float(2 * stats.t.sf(abs(t), len(j) - 2)),
            slope_per_unit=(f['slope'] if f else np.nan),
            slope_per_unit_err=(f['slope_err'] if f else np.nan),
            x_min=float(x.min()), x_max=float(x.max())))
    out = pd.DataFrame(rows)
    if len(out):
        out['abs_rho'] = out.spearman_rho.abs()
        out = out.sort_values('abs_rho', ascending=False).drop(columns='abs_rho')
        out = out.reset_index(drop=True)
    if verbose and len(out):
        print(f'\nper-night elevation slope ({slope_col}) against per-night telemetry, '
              f'ranked by |Spearman rho|')
        print(f'  {"Pearson":>8s} {"Spearman":>9s} {"p":>7s} {"n":>4s}  factor')
        for _, r in out.iterrows():
            print(f'  {r.pearson_r:+8.3f} {r.spearman_rho:+9.3f} {r.spearman_p:7.3f} '
                  f'{int(r.n_nights):4d}  {r.label} [{r.unit}]')
        # With this many factors tested at once, the uncorrected p-value is not the right
        # bar: state the threshold a rho must clear to mean anything.
        n_med = int(out.n_nights.median())
        from scipy import stats as _st
        thr_raw = _st.norm.ppf(0.975) / np.sqrt(n_med - 1)
        thr_bon = _st.norm.ppf(1 - 0.025 / len(out)) / np.sqrt(n_med - 1)
        best = out.spearman_rho.abs().max()
        print(f'\n  {len(out)} factors tested at n = {n_med} nights. |Spearman rho| must '
              f'exceed {thr_raw:.3f} for p < 0.05')
        print(f'  uncorrected, and {thr_bon:.3f} after a Bonferroni correction for '
              f'{len(out)} tests. The largest here')
        print(f'  is {best:.3f} (dimensionless).')
        if best < thr_bon:
            print('  -> no factor survives the multiple-comparison correction: the '
                  'night-to-night slope')
            print('     variation is real but this telemetry set does not explain it. '
                  'The factors are also')
            print('     mutually correlated, so these rank as candidates to model, not '
                  'as causes.')
    return out


def plot_pages(pdf, df, slopes, ud, trend, corr, ycol, bands, variant):
    """Write the figure pages.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    slopes, ud, trend, corr : `pandas.DataFrame`
    ycol : `str`
    bands : `list` [`str`]
    variant : `str`
    """
    import matplotlib.pyplot as plt

    ok = slopes[slopes.night_ok] if 'night_ok' in slopes.columns else slopes
    a = ok[ok.direction == 'all']

    # Page 1 -- the distribution of per-night slopes, and each night's slope in time.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    axes[0].hist(a.slope, bins=24, color='tab:blue', alpha=0.8)
    axes[0].axvline(a.slope.median(), color='k', ls='--',
                    label=f'median {a.slope.median():+.5f} per deg')
    axes[0].set_xlabel('per-night elevation slope '
                       '[dimensionless v-mode-1 amplitude per deg]')
    axes[0].set_ylabel('nights')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f'{len(a)} nights, nMAD of slopes '
                      f'{nmad(a.slope.to_numpy()):.5f} per deg, median formal error '
                      f'{a.slope_err.median():.5f} per deg')
    x = np.arange(len(a))
    axes[1].errorbar(x, a.slope, yerr=a.slope_err, fmt='o', ms=3, lw=0.8,
                     color='tab:blue')
    axes[1].axhline(a.slope.median(), color='k', ls='--')
    axes[1].axhline(0, color='grey', lw=0.8)
    axes[1].set_xticks(x[::4])
    axes[1].set_xticklabels([str(d) for d in a.day_obs][::4], rotation=90, fontsize=6)
    axes[1].set_xlabel('day_obs')
    axes[1].set_ylabel('slope [per deg]')
    axes[1].grid(alpha=0.3)
    fig.suptitle(f'per-night elevation slope of {ycol} — {variant}')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2 -- up versus down.
    if len(ud):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        lim = [min(ud.slope_up.min(), ud.slope_down.min()),
               max(ud.slope_up.max(), ud.slope_down.max())]
        axes[0].errorbar(ud.slope_down, ud.slope_up, xerr=ud.err_down,
                         yerr=ud.err_up, fmt='o', ms=4, lw=0.7, color='tab:purple')
        axes[0].plot(lim, lim, 'k--', lw=1, label='equal slopes')
        axes[0].set_xlabel('falling-elevation slope [per deg]')
        axes[0].set_ylabel('rising-elevation slope [per deg]')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[0].set_title('one point per night')
        axes[1].hist(ud.difference, bins=20, color='tab:purple', alpha=0.8)
        axes[1].axvline(0, color='grey', lw=0.8)
        axes[1].axvline(ud.difference.median(), color='k', ls='--',
                        label=f'median {ud.difference.median():+.5f} per deg')
        axes[1].set_xlabel('rising minus falling slope [per deg]')
        axes[1].set_ylabel('nights')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.suptitle(f'elevation hysteresis: rising versus falling legs — {variant}')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Page 3 -- the pooled median trend, and every night's own line.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    t = trend[trend.band == 'all']
    if len(t):
        axes[0].errorbar(t.elev_centre, t['median'], yerr=t['nmad'] / np.sqrt(t.n),
                         fmt='o', ms=5, color='k', label='median, bands pooled')
        xg = np.linspace(t.elev_centre.min(), t.elev_centre.max(), 100)
        c1 = np.polyfit(t.elev_centre, t['median'], 1)
        c2 = np.polyfit(t.elev_centre, t['median'], 2)
        axes[0].plot(xg, np.polyval(c1, xg), 'r--', lw=1.2,
                     label=f'linear {c1[0]:+.5f} per deg')
        axes[0].plot(xg, np.polyval(c2, xg), 'g-', lw=1.5,
                     label=f'quadratic, curvature {c2[0]:+.2e} per deg$^2$')
    for b in bands:
        tb = trend[trend.band == b]
        if len(tb):
            axes[0].plot(tb.elev_centre, tb['median'], '.-', ms=3, alpha=0.7, label=b)
    axes[0].axhline(0, color='grey', lw=0.8)
    axes[0].set_xlabel('elevation [deg]')
    axes[0].set_ylabel(f'median {ycol} [dimensionless]')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title('median trend in equal-count elevation bins')
    el = np.linspace(df.altitude_deg.min(), df.altitude_deg.max(), 10)
    for _, r in a.iterrows():
        axes[1].plot(el, r.intercept + r.slope * el, '-', lw=0.6, alpha=0.35,
                     color='tab:blue')
    axes[1].plot(el, a.intercept.median() + a.slope.median() * el, 'k-', lw=2.2,
                 label='median night')
    axes[1].set_xlabel('elevation [deg]')
    axes[1].set_ylabel(f'{ycol} [dimensionless]')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title(f'each of {len(a)} nights, own fitted line')
    fig.suptitle(f'elevation dependence of {ycol} — {variant}')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_correlate_pages(pdf, slopes, summ, corr, n_show=6):
    """Scatter the per-night slope against its strongest correlates.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    slopes : `pandas.DataFrame`
    summ : `pandas.DataFrame`
    corr : `pandas.DataFrame`
    n_show : `int`, optional
    """
    import matplotlib.pyplot as plt

    ok = slopes[slopes.night_ok] if 'night_ok' in slopes.columns else slopes
    a = ok[ok.direction == 'all'].set_index('day_obs')
    if 'slope_col' in corr.columns:
        corr = corr[corr.slope_col == 'slope']
    top = corr.head(n_show)
    if not len(top):
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        j = pd.DataFrame({'slope': a.slope}).join(summ[[r.factor]], how='inner').dropna()
        ax.scatter(j[r.factor], j.slope, s=18, alpha=0.8, color='tab:green')
        ax.set_title(f'{r.label}\nSpearman rho = {r.spearman_rho:+.3f}, '
                     f'Pearson r = {r.pearson_r:+.3f}, n = {int(r.n_nights)}',
                     fontsize=8)
        ax.set_xlabel(f'{r.label} [{r.unit}]', fontsize=7)
        ax.set_ylabel('night elevation slope\n[per deg]', fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(top):]:
        ax.axis('off')
    fig.suptitle('per-night elevation slope against its strongest per-night correlates')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_night_pages(pdf, df, slopes, ycol, n_per_page=12, max_nights=None):
    """One small panel per night: the residual against elevation, split by direction.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    slopes : `pandas.DataFrame`
    ycol : `str`
    n_per_page : `int`, optional
    max_nights : `int`, optional
        Limit the number of nights drawn; None draws all.
    """
    import matplotlib.pyplot as plt

    a = slopes[slopes.direction == 'all']
    days = list(a.day_obs)
    if max_nights:
        days = days[:max_nights]
    up = slopes[slopes.direction == 'up'].set_index('day_obs')
    dn = slopes[slopes.direction == 'down'].set_index('day_obs')
    for start in range(0, len(days), n_per_page):
        chunk = days[start:start + n_per_page]
        fig, axes = plt.subplots(3, 4, figsize=(14, 9))
        for ax, day in zip(axes.ravel(), chunk):
            d = df[df.day_obs == day]
            for direction, colour in (('up', 'tab:red'), ('down', 'tab:blue')):
                s = d[d.direction == direction]
                if len(s):
                    ax.scatter(s.altitude_deg, s[ycol], s=3, alpha=0.4, color=colour,
                               label=direction)
            el = np.linspace(d.altitude_deg.min(), d.altitude_deg.max(), 10)
            r = a[a.day_obs == day].iloc[0]
            ax.plot(el, r.intercept + r.slope * el, 'k-', lw=1.5)
            bits = [f'all {r.slope:+.4f}']
            if day in up.index:
                bits.append(f'up {up.loc[day, "slope"]:+.4f}')
            if day in dn.index:
                bits.append(f'dn {dn.loc[day, "slope"]:+.4f}')
            ax.set_title(f'{day}\n' + '  '.join(bits), fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis('off')
        fig.supxlabel('elevation [deg]', fontsize=9)
        fig.supylabel(f'{ycol} [dimensionless v-mode-1 amplitude]', fontsize=9)
        fig.suptitle('per-night elevation dependence; slopes in per deg, '
                     'red rising / blue falling', fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', default=None,
                   help='variant to analyse; default the first in the table')
    p.add_argument('--in-dir', default=None,
                   help='directory holding science_lut.parquet; default aos/output/science_lut')
    p.add_argument('--ycol', default='v1_resid_truss',
                   help='residual column to explain against elevation')
    p.add_argument('--bands', nargs='+', default=['g', 'r', 'i', 'z'])
    p.add_argument('--max-night-panels', type=int, default=None,
                   help='limit the per-night panel pages')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)
    verbose = not a.quiet

    root = pathlib.Path(__file__).resolve().parents[3]
    in_dir = pathlib.Path(a.in_dir) if a.in_dir else (root / 'aos' / 'output'
                                                     / 'science_lut')
    pv = pd.read_parquet(in_dir / 'science_lut.parquet')
    variant = a.variant or sorted(pv.variant.unique())[0]
    df = pv[pv.variant == variant].copy()
    if not len(df):
        print(f'no rows for variant {variant}; present: {sorted(pv.variant.unique())}')
        return 1
    df = df[df.band.isin(a.bands)]
    if verbose:
        print(f'variant {variant}: {len(df)} visits over {df.day_obs.nunique()} nights, '
              f'bands {a.bands}')
        print(f'residual column: {a.ycol}')

    # Direction is labelled per night, since the rolling median must not span a night gap.
    df['direction'] = pd.concat([label_direction(d) for _, d in df.groupby('day_obs')])
    if verbose:
        vc = df.direction.value_counts()
        print(f'\ndirection labelling (rolling median of {DIRECTION_WINDOW} visits, '
              f'deadband {DIRECTION_DEADBAND} deg per visit):')
        for k in ('up', 'down', 'flat'):
            print(f'  {k:5s} {int(vc.get(k, 0)):7d} visits')

    slopes = per_night_slopes(df, a.ycol, verbose=verbose)
    if not len(slopes):
        print('no night had enough visits to fit')
        return 1
    ud = updown_summary(slopes, verbose=verbose)
    trend = median_trend(df, a.ycol, a.bands, verbose=verbose)
    summ = night_summaries(df)
    corr = correlate_slopes(slopes, summ, slope_col='slope', verbose=verbose)
    # Repeat over the common elevation window. A factor that survives here is not simply
    # restating which elevations the night happened to observe.
    corr_common = correlate_slopes(slopes, summ, slope_col='slope_common',
                                   verbose=verbose)
    if len(corr) and len(corr_common):
        corr['slope_col'] = 'slope'
        corr_common['slope_col'] = 'slope_common'
        corr = pd.concat([corr, corr_common], ignore_index=True)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = in_dir / 'nightly_elevation.pdf'
    with PdfPages(pdf_path) as pdf:
        plot_pages(pdf, df, slopes, ud, trend, corr, a.ycol, a.bands, variant)
        plot_correlate_pages(pdf, slopes, summ, corr)
        plot_night_pages(pdf, df, slopes, a.ycol,
                         max_nights=a.max_night_panels)

    sl_path = in_dir / 'nightly_elevation_slopes.parquet'
    co_path = in_dir / 'nightly_elevation_correlations.parquet'
    slopes.to_parquet(sl_path)
    if len(corr):
        corr.to_parquet(co_path)
    if len(ud):
        ud.to_parquet(in_dir / 'nightly_elevation_updown.parquet')
    trend.to_parquet(in_dir / 'nightly_elevation_trend.parquet')

    print(f'\nwrote {len(slopes)} slope rows -> {sl_path}')
    print(f'wrote {len(corr)} correlation rows -> {co_path}')
    print(f'wrote {pdf_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
