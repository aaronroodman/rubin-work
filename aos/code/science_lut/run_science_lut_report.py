#!/usr/bin/env python3
"""Results document for the science-image focus look-up table (LUT) study.

Builds ``science_lut_results_<chain-order>.pdf``: the main results of the study in the order
the analysis takes them, with every panel carrying its own statistics so the figure stands
alone. This is the reading document, distinct from ``science_lut.pdf``, which is the
per-variant diagnostic dump `run_science_lut.py` writes as it fits.

The response is the v-mode-1 amplitude expressed as **equivalent camera-hexapod dz** [µm],
``v1_dzequiv = v1 / v1_per_um_dz``, because µm of hexapod dz is the quantity a focus LUT is
written in. ``v1_per_um_dz`` is the mean magnitude of the v-mode-1 response to camera (degree
of freedom, DOF, 5) and M2 (DOF 0) hexapod dz, recomputed here from `aos_state` so the
conversion in the document is the same number the fits used.

Correction chain, each stage subtracting a robust fit of the previous residual. The default
order is ``truss-grad-elev``:

1. ``v1_dzequiv`` against mean Telescope Mount Assembly (TMA) truss temperature [°C], linear;
2. the residual against the M1M3 z thermal gradient [°C/m], linear;
3. that residual against elevation [deg], polynomial over 30 to 80 deg.

``--chain-order truss-elev-grad`` swaps the last two. Elevation and the z gradient are
correlated, so whichever is fitted first absorbs the shared variance; the gradient is fitted
first by default, leaving the elevation stage what the gradient cannot explain.

All fits are Huber M-estimators, with a Theil-Sen slope quoted alongside as a leverage check
unless ``--no-theilsen`` is given (Theil-Sen is O(n^2) and dominates the run time at tens of
thousands of visits per band).
Fits are per band throughout, because filter thickness changes the camera-hexapod dz LUT and a
pooled fit would leave a band-to-band focus offset as six offset clusters.

Usage
-----
::

    python code/science_lut/run_science_lut_report.py \\
        --variant v50_34__batoid__consdb_v1 --bands u g r i z y

``--response trim-meas`` rebuilds every page against ``v1(Trim) - v1(measured)``, leaving the
hexapod LUT baseline out, and tags the output name with ``_trim_meas``. The LUT term carries
essentially the whole elevation dependence and about 37x the measured term's scatter, so that
variant shows what the closed loop and the wavefront sensors see on their own.

Notes
-----
Reads ``science_lut.parquet`` and ``science_lut_fits.parquet`` only, so it needs no
Engineering Facility Database (EFD) or Consolidated Database (ConsDB) access and runs from
synced output. It does need `aos_state` for the dz conversion, which needs the Active Optics
System (AOS) environment; ``--v1-per-um-dz`` supplies the number directly when that is
unavailable.

The elevation stage scans polynomial orders 1 to 4 per band and reports the robust residual
scatter for each, so the order used is a measurement rather than an assertion. Over 30 to 80
deg and fitted per band, order 1 wins: it captures the whole dependence, and order 2 moves the
residual by under 1% in either direction. A pooled fit over a wider window does show
curvature, but that is an artefact of stacking six bands with different focus offsets.
"""
import argparse
import datetime
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # aos/code

from common.utils import nmad                                          # noqa: E402

from run_science_lut import (BAND_COLORS, FAM_TRUSS_SLOPE,                 # noqa: E402
                            LUT_EPOCH_OFFSET_NIGHTS, MEASURED_SIGN,
                            huber_fit, theilsen_fit)

#: Elevation window the polynomial is fitted over [deg]. Outside it the sampling is sparse and
#: a high-order polynomial swings, so the fit is restricted and the plot says so.
ELEV_FIT_RANGE = (30.0, 80.0)

#: Polynomial orders scanned for the elevation stage (dimensionless).
ELEV_ORDERS = (1, 2, 3, 4)

#: Order drawn on the scatter plots and subtracted (dimensionless). Quadratic, because the
#: elevation dependence is quadratic within a night: the effective slope changes from night to
#: night, so pooling every night into one fit flattens the curvature and makes order 1 look as
#: good as order 2 on the stacked sample even though single nights clearly favour order 2. The
#: scan page reports every order over the pooled sample, which is why its numbers show a
#: smaller order-1-to-2 gain than a per-night fit would.
ELEV_ORDER_PLOT = 2

#: Band order for every multi-panel page, bluest first.
BAND_ORDER = ('u', 'g', 'r', 'i', 'z', 'y')

#: Response variable definitions, selected by ``--response``. Each entry gives the columns of
#: `science_lut.parquet` summed to form the numerator of ``v1_dzequiv``, the sign applied to
#: each, a short label, and the formula as prose for the documentation page. Every downstream
#: page reads only ``v1_dzequiv``, so a new entry here changes the whole document.
#:
#: ``lut-trim-meas`` is ``v1_total`` itself -- the full focus error, what the study fits.
#: ``trim-meas`` drops the look-up-table baseline and keeps the accumulated closed-loop Trim
#: offset minus the measured optical state. The LUT term carries essentially all of the
#: elevation dependence and about 37x the measured term's scatter, so removing it isolates
#: what the closed loop and the wavefront sensors see, independent of the commanded baseline.
RESPONSES = {
    'lut-trim-meas': dict(
        terms=(('v1_lut', +1.0), ('v1_trim', +1.0), ('v1_meas', MEASURED_SIGN)),
        label='v1_total = v1(LUT) + v1(Trim) - v1(measured)',
        short='v1_total',
        prose=('v1_total = v1(hexapod LUT) + v1(Trim) + MEASURED_SIGN * v1(measured)\n'
               f'MEASURED_SIGN = {MEASURED_SIGN:+.0f}, i.e. LUT + Trim - MEASURED.\n'
               'The LUT is the elevation- and temperature-dependent hexapod baseline;\n'
               'the Trim is the accumulated closed-loop offset; the measured term is the\n'
               'optical state recovered at the four Corner Wavefront Sensors (CWFS).')),
    'trim-meas': dict(
        terms=(('v1_trim', +1.0), ('v1_meas', MEASURED_SIGN)),
        label='v1_trim_meas = v1(Trim) - v1(measured)',
        short='v1_trim_meas',
        prose=('v1_trim_meas = v1(Trim) + MEASURED_SIGN * v1(measured)\n'
               f'MEASURED_SIGN = {MEASURED_SIGN:+.0f}, i.e. Trim - MEASURED.\n'
               'The hexapod look-up-table (LUT) baseline is deliberately LEFT OUT, so this\n'
               'is the accumulated closed-loop Trim offset against the optical state\n'
               'recovered at the four Corner Wavefront Sensors (CWFS), with no commanded\n'
               'elevation/temperature baseline in it. The LUT term carries essentially the\n'
               'whole elevation dependence and about 37x the measured term scatter, so the\n'
               'temperature and elevation dependences found here are those of the closed\n'
               'loop and the wavefront sensors alone.')),
}

#: Fixed y-limits for the six-panel v1_dzequiv grid pages [µm of equivalent camera-hexapod dz].
#: The pooled 1st-to-99th percentile these pages used before clips the tails of the widest
#: band; this window is wide enough to hold nearly every visit in any band, so the panels can
#: be compared directly. Each panel reports the count falling outside, and the fits always use
#: every point, so the window changes only what is drawn.
GRID_YLIM_DZEQUIV = (-2000.0, 4000.0)

#: M1M3 thermal-gradient columns and their labels [°C/m].
GRAD_COLS = (('m1m3_z_gradient_c_per_m', 'M1M3 z thermal gradient'),
             ('m1m3_radial_gradient_c_per_m', 'M1M3 radial thermal gradient'),
             ('m1m3_x_gradient_c_per_m', 'M1M3 x thermal gradient'),
             ('m1m3_y_gradient_c_per_m', 'M1M3 y thermal gradient'))


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


#: Per-axis v-mode-1 response, filled by `v1_per_um_dz_value` and keyed by DOF index: 5 is
#: the camera-hexapod dz axis, 0 the M2-hexapod dz axis [dimensionless v-mode-1 amplitude per
#: µm]. Kept so the document can state what the mean of the two means physically without
#: hardcoding the numbers.
V1_PER_UM_DZ_AXES = {}


def v1_per_um_dz_value(dof_set='all_50', n_modes=34, verbose=True):
    """v-mode-1 amplitude per µm of hexapod dz [per µm].

    Parameters
    ----------
    dof_set : `str`, optional
        ts_ofc DOF-set name for the DOF to v-mode projection.
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

    The projection is `aos_state.vmodes_from_dofs` through `aos_state.make_state_estimator` —
    the single sanctioned v-mode engine, the basis the measured state's v-modes are reported
    in and the commanded terms are stored in. The 12-mode cap that once argued against it was
    `StateEstimator.truncate_index`, a controller-yaml default rather than a limit.

    v1 is the camera-hexapod dz (DOF 5) and M2-hexapod dz (DOF 0) combination plus small
    mirror-bending terms, and is the same mode in both schemes, so this factor comes out equal
    to 5 decimal places at ``standard_22``/12 and ``all_50``/34.

    What the mean magnitude means physically: the two coefficients carry the **same sign**, so
    the two axes add rather than cancel, and their sum over their mean is 2.00000
    (dimensionless) to five decimal places. Moving 0.5 µm on each hexapod — 1 µm of **total**
    dz travel — therefore produces exactly this factor's worth of v1. So a value reported in
    these units is µm of total defocus travel, split evenly between the camera and M2
    hexapods, and not µm of camera-hexapod motion with M2 held still. Per unit v-mode-1
    amplitude that is 1110.1 µm of total travel shared, 555.0 µm on each hexapod, against
    1121.8 µm if the camera hexapod moves alone — the two differ by only 1.1%
    (dimensionless), because the two coefficients agree to 2.1%. The unit label elsewhere in
    this study says "equivalent camera-hexapod dz" for continuity with the earlier documents;
    read it as equivalent total hexapod dz.
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
    """Prose for the title page explaining what the dz-equivalent unit means physically.

    Parameters
    ----------
    v1_per_um_dz : `float`
        The conversion in use [dimensionless v-mode-1 amplitude per µm of hexapod dz].

    Returns
    -------
    text : `str`
        Plain-text paragraph, using the per-axis coefficients from `V1_PER_UM_DZ_AXES` when
        `v1_per_um_dz_value` has been called and a shorter form when it has not.

    Notes
    -----
    The conversion is the **mean magnitude** of the camera-hexapod (DOF 5) and M2-hexapod
    (DOF 0) dz coefficients. Because those two carry the same sign and are nearly equal, that
    mean is what 1 µm of total dz travel produces when split evenly between the two hexapods,
    so a reported µm is total travel and not one hexapod moving alone.
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
                f'{cam_alone:.1f} um if the camera hexapod moves alone. Axis labels\n'
                f'elsewhere in this document say "equivalent camera-hexapod dz" for\n'
                f'continuity with the earlier versions; read them as equivalent total\n'
                f'hexapod dz.')
    return ('The conversion is the mean magnitude of the camera-hexapod (DOF 5) and\n'
            'M2-hexapod (DOF 0) dz coefficients. Those two carry the same sign and are\n'
            'nearly equal, so the mean is what 1 um of TOTAL dz travel produces when it is\n'
            'split evenly between the two hexapods. A value in these units is therefore um\n'
            'of total defocus travel, NOT um of camera-hexapod motion with M2 held still.\n'
            f'Per unit v-mode-1 amplitude: {shared_total:.1f} um of total travel shared\n'
            f'({0.5 * shared_total:.1f} um on each hexapod), against about the same\n'
            f'{shared_total:.0f} um if one hexapod moves alone, since the two coefficients\n'
            'agree to about 2%. Axis labels elsewhere say "equivalent camera-hexapod dz"\n'
            'for continuity; read them as equivalent total hexapod dz.')


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
    equations lose conditioning. The returned ``predict`` undoes the scaling, so callers work
    in the original units.
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
    Slope and error are rounded so the error carries two significant figures and the slope
    matches its decimal place, which is one fewer than the raw fit prints.
    """
    if h is None:
        return f'{band} band   n = {n}   fit under-determined'
    err = h['slope_err']
    nd = 2 if err <= 0 or not np.isfinite(err) else max(0, 1 - int(np.floor(np.log10(err))))
    su = f'{unit} per {xunit}' if xunit else unit
    line = (f'{band} band   n = {h["n"]}   slope {h["slope"]:+.{nd}f} '
            f'+/- {err:.{nd}f} {su}')
    line2 = (f'Pearson r = {h["pearson_r"]:+.3f}   '
             f'Spearman rho = {h["spearman_rho"]:+.3f}   '
             f'nMAD = {h["resid_nmad"]:.4f} {unit}')
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
    """Opening page: what this document is and how it was made.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    meta : `dict`
        Text blocks to render, in order.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.955, 'Focus look-up table from science exposures',
             fontsize=16, va='top', weight='bold')
    fig.text(0.06, 0.925, meta['subtitle'], fontsize=9.5, va='top', style='italic')
    y = 0.885
    for head, body in meta['sections']:
        fig.text(0.06, y, head, fontsize=11, va='top', weight='bold')
        y -= 0.022
        fig.text(0.07, y, body, fontsize=8.2, va='top', family='monospace')
        y -= 0.021 * (body.count('\n') + 1) + 0.016
    pdf.savefig(fig)
    plt.close(fig)
    return None


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
    return None


def page_v1_histograms(pdf, df, v1_per_um_dz, response='lut-trim-meas'):
    """Histograms of the response in both the dimensionless and the hexapod-dz unit.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``v1_response`` and ``v1_dzequiv``.
    v1_per_um_dz : `float`
        [per µm], quoted in the title.
    response : `str`, optional
        Key into `RESPONSES`, naming which v-mode-1 combination is plotted.
    """
    import matplotlib.pyplot as plt

    spec = RESPONSES[response]
    short = spec['short']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, unit, lab in (
            (axes[0], 'v1_response', 'dimensionless',
             f'{short} [dimensionless v-mode-1 amplitude]'),
            (axes[1], 'v1_dzequiv', 'um dz',
             f'{short} as equivalent camera-hexapod dz [um]')):
        v = df[col].to_numpy(float)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.5, 99.5])
        ax.hist(v, bins=80, range=(lo, hi), color='tab:blue', alpha=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel('visits [count]')
        ax.set_title(f'n = {v.size}\nmedian {np.median(v):+.4f} {unit}   '
                     f'robust RMS (nMAD) {nmad(v):.4f} {unit}', fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{spec["label"]};   '
                 f'conversion {v1_per_um_dz:.5e} dimensionless per um of hexapod dz',
                 fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return None


def page_grid_fits(pdf, df, xcol, ycol, bands, xlabel, ylabel, title, xunit,
                   shared_ylim=True, order=None, fit_range=None, rows=2, cols=3,
                   theilsen=True, ylim=None):
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
        Polynomial order. None or 1 fits a line and quotes Theil-Sen alongside.
    fit_range : `tuple` [`float`], optional
        Restrict the fit (not the plot) to this predictor range.
    rows, cols : `int`, optional
    theilsen : `bool`, optional
        Quote a Theil-Sen slope beside the Huber one. Theil-Sen forms all pairwise slopes,
        so it costs O(n^2) -- minutes per panel at tens of thousands of visits -- while
        contributing only the leverage cross-check line to the title. Set False to skip it.
    ylim : `tuple` [`float`], optional
        Explicit ``(low, high)`` y-limits in the response unit, used for every panel instead
        of the pooled percentile window. The pooled 1st-to-99th percentile clips the tails of
        the widest band, so a fixed window is what shows the full spread; each panel then
        reports the count falling outside it.

    Returns
    -------
    fits : `dict`
        Band to the fit result, so a caller can subtract it.

    Notes
    -----
    The fits use every finite point in `fit_range` regardless of the y-limits; the limits set
    what is drawn, not what is fitted, so a slope never depends on the plotting window.
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
                             f'robust RMS (nMAD) {r["resid_nmad"]:.4f} um dz',
                             fontsize=8.5)
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
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)
    return out


def page_single_band_fits(pdf, df, xcol, ycol, bands, xlabel, ylabel, title, xunit,
                          clip=(1, 99), theilsen=True, ylim=None):
    """One page per band, y-limits set from that band alone to show the detail.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    xcol, ycol : `str`
    bands : `sequence` [`str`]
    xlabel, ylabel, title : `str`
    xunit : `str`
    clip : `tuple` [`float`], optional
        Percentiles setting the per-band y-limits. Ignored when `ylim` is given.
    theilsen : `bool`, optional
        Quote a Theil-Sen slope beside the Huber one; see `page_grid_fits`.
    ylim : `tuple` [`float`], optional
        Fixed ``(low, high)`` y-limits in the response unit, shared by every band, instead
        of the per-band percentile clip. Each panel title then reports how many points fall
        outside the window, so a fixed scale cannot hide a band silently.

    Notes
    -----
    The fit uses every finite point regardless of `ylim`; the limits set what is drawn, not
    what is fitted, so a slope never depends on the plotting window.
    """
    import matplotlib.pyplot as plt

    for b in bands:
        d = df[df.band == b]
        x, y = d[xcol].to_numpy(float), d[ycol].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            continue
        h = huber_fit(x[m], y[m])
        t = theilsen_fit(x[m], y[m]) if theilsen else None
        fig, ax = plt.subplots(figsize=(9.5, 6.4))
        ax.plot(x, y, '.', ms=2.5, alpha=0.4, color=BAND_COLORS.get(b, 'grey'))
        if h is not None:
            xs = np.array([np.nanmin(x[m]), np.nanmax(x[m])])
            ax.plot(xs, h['intercept'] + h['slope'] * xs, '-', color='k', lw=1.8)
        if ylim is not None:
            ax.set_ylim(*ylim)
            n_out = int(((y[m] < ylim[0]) | (y[m] > ylim[1])).sum())
            extra = (f'{n_out} of {int(m.sum())} points outside the plotted '
                     f'{ylim[0]:.0f} to {ylim[1]:.0f} window (fit uses all)'
                     if n_out else 'all points inside the plotted window')
        else:
            ax.set_ylim(*np.percentile(y[m], clip))
            extra = ''
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\n'
                     + stat_title(b, int(m.sum()), h, t, xunit=xunit, extra=extra),
                     fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return None


def page_hists(pdf, df, col, bands, xlabel, title, rows=2, cols=3, clip=(0.5, 99.5)):
    """Per-band histograms of a residual, with n and the robust RMS.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    col : `str`
    bands : `sequence` [`str`]
    xlabel, title : `str`
    rows, cols : `int`, optional
    clip : `tuple` [`float`], optional
        Percentiles setting the shared histogram range.
    """
    import matplotlib.pyplot as plt

    v_all = df[col].to_numpy(float)
    v_all = v_all[np.isfinite(v_all)]
    rng = tuple(np.percentile(v_all, clip)) if v_all.size else None

    fig, axes = plt.subplots(rows, cols, figsize=(15, 8.5), squeeze=False)
    for k, b in enumerate(bands):
        ax = axes[k // cols][k % cols]
        v = df.loc[df.band == b, col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=60, range=rng, color=BAND_COLORS.get(b, 'grey'), alpha=0.85)
        ax.set_title(hist_title(b, v), fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('visits [count]', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for k in range(len(bands), rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)
    return None


def page_order_scan(pdf, scans, bands):
    """Table page: robust residual scatter against polynomial order.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    scans : `dict`
        Band to a dict of order to ``resid_nmad`` [µm dz].
    bands : `sequence` [`str`]
    """
    import matplotlib.pyplot as plt

    lines = [f'Elevation polynomial order scan over '
             f'{ELEV_FIT_RANGE[0]:.0f} to {ELEV_FIT_RANGE[1]:.0f} deg',
             'robust residual scatter, nMAD [um of equivalent hexapod dz]', '',
             'band  ' + '  '.join(f'order {o}' for o in ELEV_ORDERS) + '   plotted']
    for b in bands:
        s = scans.get(b, {})
        cells = '  '.join(f'{s[o]:7.4f}' if o in s else '      -' for o in ELEV_ORDERS)
        lines.append(f'{b:>4}  {cells}   order {ELEV_ORDER_PLOT}')
    lines += ['',
              f'Order {ELEV_ORDER_PLOT} is drawn on the elevation scatter pages and is the',
              'order subtracted to form the residual.',
              '',
              'The scan above pools every night of the sample into one fit per band, and on',
              'that pooled sample order 2 improves the robust residual by under 1 percent',
              'over order 1. That understates the curvature rather than measuring it: the',
              'effective elevation slope changes from night to night, so stacking the nights',
              'averages together lines of differing slope and flattens the quadratic term.',
              'Fitted within a single night the elevation dependence is clearly quadratic,',
              'which is why order 2 is used here despite the pooled numbers.',
              '',
              'Orders 3 and 4 add nothing beyond order 2 and are not used.']

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.94, 'Choice of elevation polynomial order', fontsize=13,
             va='top', weight='bold')
    fig.text(0.06, 0.86, '\n'.join(lines), fontsize=9.5, va='top', family='monospace')
    pdf.savefig(fig)
    plt.close(fig)
    return None


#: Correction-chain stage columns per order, in the order they are applied. Used for the
#: closing nMAD summary and to pick the columns written to the results parquet.
CHAIN_STAGES = {
    'truss-elev-grad': ('v1_dzequiv', 'v1_dzequiv_trusscorr',
                        'v1_dzequiv_truss_elev_corr', 'v1_dzequiv_final'),
    'truss-grad-elev': ('v1_dzequiv', 'v1_dzequiv_trusscorr',
                        'v1_dzequiv_truss_grad_corr', 'v1_dzequiv_final'),
    'ml-elev': ('v1_dzequiv', 'v1_dzequiv_mlcorr', 'v1_dzequiv_final'),
}

#: Human-readable name of each stage, parallel to `CHAIN_STAGES`, for the summary header.
CHAIN_LABELS = {
    'truss-elev-grad': ('raw', '-truss', '-elev', '-gradz'),
    'truss-grad-elev': ('raw', '-truss', '-gradz', '-elev'),
    'ml-elev': ('raw', '-ml', '-elev'),
}

#: Feature groups of the band-independent thermal model used by the ``ml-elev`` chain.
#: Truss temperature plus the four M1M3 thermal gradients -- the configuration measured best
#: under a night-grouped split (68.4 um of equivalent hexapod dz, against 162.8 um for the
#: truss alone). See `run_thermal_model`.
ML_FEATURE_GROUPS = ('truss', 'grads')


def _chain_ml(pdf, df, bands, theilsen=True, verbose=True):
    """One band-independent thermal stage, then elevation.

    Replaces the sequential per-band truss and M1M3-z-gradient stages with a single
    band-independent fit of the truss temperature and all four M1M3 thermal gradients at once,
    then fits elevation per band to what is left.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Must carry ``v1_dzequiv`` [µm of equivalent camera-hexapod dz], ``day_obs``, ``band``,
        ``altitude_deg`` [deg] and the thermal feature columns.
    bands : `sequence` [`str`]
    theilsen : `bool`, optional
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        With ``v1_dzequiv_mlcorr`` and ``v1_dzequiv_final`` added, both in µm of equivalent
        camera-hexapod dz.

    Notes
    -----
    Fitting one set of coefficients for every band is the point: the per-band chain re-fits at
    each filter change, which injects a step into the corrected residual where nothing physical
    has changed. A band-independent stage cannot produce that step by construction.

    The coefficients come from the whole sample rather than out-of-fold, so the residual here is
    in-sample. `run_thermal_model` carries the honest night-grouped score -- 68.4 um of
    equivalent hexapod dz against a 332.4 um uncorrected baseline -- and the ~2 um difference
    from the in-sample number is the measured cost of that distinction.
    """
    import run_thermal_model as tm

    features = tm.resolve_features(ML_FEATURE_GROUPS)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f'--chain-order ml-elev needs the thermal feature columns '
                         f'{", ".join(missing)}, which are absent from the input frame')

    # ------------------------------------------- one band-independent joint thermal stage
    if verbose:
        print('\nstage 1 (ml order): band-independent joint fit of the truss temperature '
              'and the four M1M3 thermal gradients')
    fit = df[['v1_dzequiv'] + features].copy()
    fit['y'] = fit.v1_dzequiv
    fit['day_obs'] = df.day_obs
    ok = np.isfinite(fit.y.to_numpy(float))
    # A visit with no truss temperature is dropped, not imputed: the pipeline's median
    # imputer would substitute the run-wide median, a per-night bias of order 1 deg C and
    # so about 124 um of equivalent camera-hexapod dz. The scattered single-exposure gaps
    # are already filled in time by common.efd_db.interpolate_within_night.
    if 'truss_temp_mean_c' in fit.columns:
        has_truss = np.isfinite(fit.truss_temp_mean_c.to_numpy(float))
        if verbose and (ok & ~has_truss).any():
            print(f'  dropped {int((ok & ~has_truss).sum())} visits with no truss '
                  f'temperature after in-night interpolation')
        ok = ok & has_truss
    full = tm.fit_full(fit[ok], features, model='huber', verbose=verbose)

    df['v1_dzequiv_mlcorr'] = np.nan
    df.loc[ok, 'v1_dzequiv_mlcorr'] = full['resid']
    page_hists(pdf, df, 'v1_dzequiv_mlcorr', bands,
               'v1 equiv hexapod dz after the band-independent thermal correction [um]',
               'v1 equivalent hexapod dz after one band-independent fit of the truss '
               'temperature and the four M1M3 thermal gradients')

    # The per-band residual of a single band-independent fit -- the check that no band is
    # badly served by shared coefficients.
    if verbose:
        print('\nper-band residual of the band-independent stage '
              '[um of equivalent camera-hexapod dz]')
        for b in bands:
            r = df.loc[df.band == b, 'v1_dzequiv_mlcorr'].to_numpy(float)
            r = r[np.isfinite(r)]
            if len(r):
                print(f'  {b}: median {np.median(r):+8.1f}, nMAD {nmad(r):8.1f}, n = {len(r)}')

    # ------------------------------------------------------- elevation, after the ML stage
    if verbose:
        print('\nstage 2 (ml order): elevation, polynomial order scan '
              f'over {ELEV_FIT_RANGE[0]:.0f}-{ELEV_FIT_RANGE[1]:.0f} deg [nMAD, um dz]')
    scans = {}
    for b in bands:
        d = df[df.band == b]
        x = d.altitude_deg.to_numpy(float)
        y = d.v1_dzequiv_mlcorr.to_numpy(float)
        m = ((x >= ELEV_FIT_RANGE[0]) & (x <= ELEV_FIT_RANGE[1])
             & np.isfinite(x) & np.isfinite(y))
        scans[b] = scan_poly_order(x[m], y[m])
        if verbose and scans[b]:
            print(f'  {b}: ' + '  '.join(f'order {o} {v:.4f}'
                                         for o, v in sorted(scans[b].items())))
    page_order_scan(pdf, scans, bands)
    el = page_grid_fits(
        pdf, df, 'altitude_deg', 'v1_dzequiv_mlcorr', bands, 'elevation [deg]',
        'v1 equiv hexapod dz after the band-independent thermal correction [um]',
        'Thermally corrected v1 equivalent hexapod dz against elevation, with a robust '
        f'order-{ELEV_ORDER_PLOT} polynomial over '
        f'{ELEV_FIT_RANGE[0]:.0f}-{ELEV_FIT_RANGE[1]:.0f} deg',
        'deg', order=ELEV_ORDER_PLOT, fit_range=ELEV_FIT_RANGE, theilsen=theilsen)

    df['v1_dzequiv_final'] = np.nan
    for b in bands:
        r = el.get(b)
        if r is None:
            continue
        s = df.band == b
        el_b = df.loc[s, 'altitude_deg'].to_numpy(float)
        # page_grid_fits returns a huber_fit dict at order 1 and a robust_poly dict above it;
        # only the latter carries a predict callable.
        model = (r['predict'](el_b) if 'predict' in r
                 else r['intercept'] + r['slope'] * el_b)
        df.loc[s, 'v1_dzequiv_final'] = df.loc[s, 'v1_dzequiv_mlcorr'] - model
    page_hists(pdf, df, 'v1_dzequiv_final', bands,
               'v1 equiv hexapod dz after the band-independent thermal and elevation '
               'correction [um]',
               'v1 equivalent hexapod dz after removing the band-independent thermal model '
               'and the elevation dependence')
    return df


def _chain_grad_first(pdf, df, bands, theilsen=True, verbose=True):
    """Stages 3 and 2 of the chain in the order M1M3 z gradient, then elevation.

    Fits the M1M3 z thermal gradient to the truss-corrected residual, subtracts it, then fits
    elevation to what is left -- the reverse of the default order. Adds the pages for both
    stages and their residual histograms to `pdf`.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Must already carry ``v1_dzequiv_trusscorr`` [µm of equivalent camera-hexapod dz].
    bands : `sequence` [`str`]
    theilsen : `bool`, optional
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        With ``v1_dzequiv_truss_grad_corr`` and ``v1_dzequiv_final`` added, both in µm of
        equivalent camera-hexapod dz.

    Notes
    -----
    Reversing the order matters when the two predictors are correlated in the sample: whichever
    is fitted first absorbs the shared variance. Comparing the two documents' final nMAD and
    per-stage slopes is what shows how much of the elevation dependence is really the M1M3
    thermal gradient, and vice versa.

    All four M1M3 gradients are plotted for context, but only the z gradient is subtracted --
    the others are not independent of it and subtracting each in turn would double-count.
    """
    # ------------------------------------------------------------ M1M3 gradients, after truss
    if verbose:
        print('\nstage 2 (grad-first order): M1M3 thermal gradients against the '
              'truss-corrected residual')
    gz = {}
    for col, label in GRAD_COLS:
        if col not in df.columns:
            continue
        f = page_grid_fits(
            pdf, df, col, 'v1_dzequiv_trusscorr', bands, f'{label} [deg C/m]',
            'v1 equiv hexapod dz after truss temp correction [um]',
            f'Truss-corrected v1 equivalent hexapod dz against the {label}',
            'deg C/m', theilsen=theilsen)
        if col == 'm1m3_z_gradient_c_per_m':
            gz = f
        if verbose:
            for b in bands:
                h = f.get(b)
                if h is not None:
                    print(f'  {label}, {b}: slope {h["slope"]:+.4f} um dz per deg C/m, '
                          f'Spearman rho {h["spearman_rho"]:+.3f}, n = {h["n"]}')

    df['v1_dzequiv_truss_grad_corr'] = np.nan
    for b in bands:
        h = gz.get(b)
        if h is None:
            continue
        s = df.band == b
        df.loc[s, 'v1_dzequiv_truss_grad_corr'] = (
            df.loc[s, 'v1_dzequiv_trusscorr']
            - (h['intercept'] + h['slope'] * df.loc[s, 'm1m3_z_gradient_c_per_m']))
    page_hists(pdf, df, 'v1_dzequiv_truss_grad_corr', bands,
               'v1 equiv hexapod dz after truss temp and M1M3 z gradient correction [um]',
               'v1 equivalent hexapod dz after removing the truss temperature and the '
               'M1M3 z thermal gradient dependence')

    # ------------------------------------------------- elevation, after truss and z gradient
    if verbose:
        print('\nstage 3 (grad-first order): elevation, polynomial order scan '
              f'over {ELEV_FIT_RANGE[0]:.0f}-{ELEV_FIT_RANGE[1]:.0f} deg [nMAD, um dz]')
    scans = {}
    for b in bands:
        d = df[df.band == b]
        x = d.altitude_deg.to_numpy(float)
        y = d.v1_dzequiv_truss_grad_corr.to_numpy(float)
        m = ((x >= ELEV_FIT_RANGE[0]) & (x <= ELEV_FIT_RANGE[1])
             & np.isfinite(x) & np.isfinite(y))
        scans[b] = scan_poly_order(x[m], y[m])
        if verbose and scans[b]:
            print(f'  {b}: ' + '  '.join(f'order {o} {v:.4f}'
                                         for o, v in sorted(scans[b].items())))
    page_order_scan(pdf, scans, bands)
    el = page_grid_fits(
        pdf, df, 'altitude_deg', 'v1_dzequiv_truss_grad_corr', bands, 'elevation [deg]',
        'v1 equiv hexapod dz after truss and M1M3 z grad correction [um]',
        'Truss- and M1M3-z-gradient-corrected v1 equivalent hexapod dz against elevation, '
        f'with a robust order-{ELEV_ORDER_PLOT} polynomial over '
        f'{ELEV_FIT_RANGE[0]:.0f}-{ELEV_FIT_RANGE[1]:.0f} deg',
        'deg', order=ELEV_ORDER_PLOT, fit_range=ELEV_FIT_RANGE, theilsen=theilsen)

    df['v1_dzequiv_final'] = np.nan
    for b in bands:
        r = el.get(b)
        if r is None:
            continue
        s = df.band == b
        el_b = df.loc[s, 'altitude_deg'].to_numpy(float)
        # page_grid_fits returns a huber_fit dict at order 1 and a robust_poly dict above it;
        # only the latter carries a predict callable.
        model = (r['predict'](el_b) if 'predict' in r
                 else r['intercept'] + r['slope'] * el_b)
        df.loc[s, 'v1_dzequiv_final'] = df.loc[s, 'v1_dzequiv_truss_grad_corr'] - model
    page_hists(pdf, df, 'v1_dzequiv_final', bands,
               'v1 equiv hexapod dz after truss, M1M3 z gradient and elevation '
               'correction [um]',
               'v1 equivalent hexapod dz after removing the truss temperature, the M1M3 z '
               'thermal gradient and the elevation dependence')
    return df


def _chain_summary(df, bands, chain_order, verbose=True):
    """Print the robust residual scatter at each stage of the chain.

    Parameters
    ----------
    df : `pandas.DataFrame`
    bands : `sequence` [`str`]
    chain_order : `str`
        Key into `CHAIN_STAGES`.
    verbose : `bool`, optional
        Print nothing when False.
    """
    if not verbose:
        return
    cols = CHAIN_STAGES[chain_order]
    labels = CHAIN_LABELS[chain_order]
    print(f'\nrobust RMS (nMAD) through the {chain_order} correction chain '
          f'[um of equivalent hexapod dz]:')
    for b in bands:
        s = df.band == b
        cells = []
        for c in cols:
            v = df.loc[s, c].to_numpy(float) if c in df.columns else np.array([])
            v = v[np.isfinite(v)]
            cells.append(f'{nmad(v):.4f}' if v.size else '     -')
        print(f'  {b}: ' + '  '.join(f'{lab} {cell}'
                                     for lab, cell in zip(labels, cells)))


def build_report(pv, fits, variant, bands, v1_per_um_dz, out_path, db_path, verbose=True,
                 theilsen=True, grid_ylim=GRID_YLIM_DZEQUIV,
                 chain_order='truss-grad-elev', response='lut-trim-meas',
                 dropped_lut_epoch=False):
    """Assemble the results PDF.

    Parameters
    ----------
    pv : `pandas.DataFrame`
        Per-visit rows from ``science_lut.parquet``, one variant already selected.
    fits : `pandas.DataFrame`
        Fit rows from ``science_lut_fits.parquet``.
    variant : `str`
    bands : `sequence` [`str`]
    v1_per_um_dz : `float`
        [dimensionless v-mode-1 amplitude per µm of hexapod dz].
    out_path : `pathlib.Path`
    db_path : `str`
        Database path, quoted on the documentation page.
    verbose : `bool`, optional
    theilsen : `bool`, optional
        Quote a Theil-Sen slope beside each Huber slope as a leverage check. It is O(n^2) in
        the visits per band, so it dominates the run time on the larger bands.
    grid_ylim : `tuple` [`float`], optional
        Shared y-limits for the six-panel ``v1_dzequiv`` grid pages -- the truss page and the
        uncorrected elevation page -- [µm of equivalent camera-hexapod dz]. Pass None to fall
        back to the pooled 1st-to-99th percentile.
    chain_order : `str`, optional
        ``'truss-grad-elev'`` (default) fits truss temperature, then the M1M3 z thermal
        gradient, then elevation. ``'truss-elev-grad'`` fits elevation before the z gradient.
        The two orders divide shared variance differently when the predictors are correlated,
        so comparing them is a measurement, not a cosmetic choice.
    response : `str`, optional
        Key into `RESPONSES`, naming which v-mode-1 combination every page fits.
    dropped_lut_epoch : `bool`, optional
        Whether the caller already removed `LUT_EPOCH_OFFSET_NIGHTS` from `pv`. Recorded on
        the documentation page; this function does no filtering of its own.

    Returns
    -------
    pv : `pandas.DataFrame`
        With the dz-equivalent and residual columns added.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    spec = RESPONSES[response]
    df = pv.copy()
    # The response is summed from its named terms rather than read from v1_total, so that
    # dropping the LUT term is a change of numerator only: every page below reads v1_dzequiv.
    # NaN in any term propagates, as it does in v1_total, so a visit missing one term is
    # excluded from the fits instead of entering with that term silently set to zero.
    missing = [c for c, _ in spec['terms'] if c not in df.columns]
    if missing:
        raise SystemExit(f'science_lut.parquet lacks {missing} needed for --response '
                         f'{response}; rebuild it with run_science_lut.py')
    df['v1_response'] = sum(sign * df[col] for col, sign in spec['terms'])
    df['v1_dzequiv'] = df['v1_response'] / v1_per_um_dz

    with PdfPages(out_path) as pdf:
        # ---------------------------------------------------------------- 1. documentation
        nights = np.sort(df.day_obs.unique())
        vr = fits[fits.variant == variant]
        short = spec['short']
        bandline = '  '.join(
            f'{b}:{int(df[(df.band == b) & df.v1_response.notna()].shape[0])}' for b in bands)
        meta = dict(
            subtitle=(f'produced {datetime.date.today().isoformat()} by '
                      f'code/science_lut/run_science_lut_report.py'),
            sections=[
                ('What this study measures',
                 'The uniform-defocus state of the telescope, from ordinary science\n'
                 'exposures rather than dedicated Full Array Mode (FAM) sequences, as a\n'
                 'function of TMA truss temperature and then of elevation. The purpose is a\n'
                 'focus look-up table (LUT) derived from science images.'),
                ('The response variable',
                 spec['prose'] + '\n'
                 'Plotted throughout as equivalent hexapod dz [um]:\n'
                 f'   v1_dzequiv = {short} / {v1_per_um_dz:.6e} per um\n'
                 'the mean v-mode-1 response magnitude of the camera (DOF 5) and M2\n'
                 f'(DOF 0) hexapod dz axes. Every term of {short} and this\n'
                 'conversion factor are projected in one basis, the 50/34 scheme\n'
                 '(all_50 / n_modes=34): the commanded LUT and Trim v-modes are stored\n'
                 'per variant by build_optical_state, so no term is reprojected here.'),
                ('Reading the unit: total dz travel, not one hexapod alone',
                 _unit_reading_text(v1_per_um_dz)),
                ('Sample',
                 f'visits with finite {short:<10s}: '
                 f'{int(df.v1_response.notna().sum())}\n'
                 f'rows in the table           : {len(df)}\n'
                 f'nights                      : {len(nights)}  '
                 f'({int(nights.min())} to {int(nights.max())})\n'
                 f'per band (finite response)  : {bandline}\n'
                 "selection                   : img_type = 'science', all science_programs"
                 + (f'\nnights dropped              : the '
                    f'{len(LUT_EPOCH_OFFSET_NIGHTS)} nights running a different\n'
                    '            hexapod LUT configuration '
                    '(LUT_EPOCH_OFFSET_NIGHTS)' if dropped_lut_epoch else '')),
                ('Optical state',
                 f'variant        : {variant}\n'
                 'scheme         : 50 degrees of freedom, 34 v-modes (all_50 / n_modes=34)\n'
                 'intrinsic      : batoid ray trace from lsst.ts.ofc (ofc_v13)\n'
                 'Zernike basis  : Noll 4 to 26 excluding 20 and 21 (21 indices)\n'
                 'measured OPD   : ConsDB ccdvisit1_quicklook, detectors 191/195/199/203\n'
                 'rotator angle  : ConsDB physical_rotator_angle\n'
                 'The Measured Intrinsic Wavefront (MIW) route exists as a second variant\n'
                 'and gives the same slopes to within 0.06 of their own errors; it is not\n'
                 'the subject of this document.'),
                ('Correction chain, each stage on the previous residual',
                 (f'order          : {chain_order}\n'
                  '1. v1_dzequiv          vs mean TMA truss temperature [C], linear\n'
                  '2. v1_dzequiv_trusscorr    vs elevation [deg], polynomial 30-80 deg\n'
                  '3. v1_dzequiv_truss_elev_corr  vs M1M3 z thermal gradient [C/m], linear\n'
                  if chain_order == 'truss-elev-grad' else
                  f'order          : {chain_order}\n'
                  '1. v1_dzequiv          vs mean TMA truss temperature [C], linear\n'
                  '2. v1_dzequiv_trusscorr    vs M1M3 z thermal gradient [C/m], linear\n'
                  '3. v1_dzequiv_truss_grad_corr  vs elevation [deg], polynomial 30-80 deg\n'
                  'The z gradient is fitted before elevation here; the two predictors are\n'
                  'correlated, so whichever is fitted first absorbs the shared variance.\n')
                 + 'All fits Huber M-estimator (statsmodels RLM, HuberT)'
                 + (', with a Theil-Sen\nslope quoted alongside as a check on leverage.'
                    if theilsen else
                    '. The Theil-Sen leverage\ncheck was skipped (--no-theilsen).')
                 + ' Scatter is reported as\n'
                 'nMAD, a robust RMS. Fits are per band because filter thickness changes\n'
                 'the camera-hexapod dz LUT.'),
                ('Reference and provenance',
                 f'database   : {db_path}\n'
                 'per-visit  : aos/output/science_lut/science_lut.parquet\n'
                 'fit table  : aos/output/science_lut/science_lut_fits.parquet\n'
                 'diagnostic : aos/output/science_lut/science_lut.pdf (per-variant dump)\n'
                 f'FAM commanded truss slope for comparison: {FAM_TRUSS_SLOPE:+.5f}\n'
                 '            dimensionless v-mode-1 amplitude per deg C, from\n'
                 '            code/correlations/run_dz14_truss.py'),
            ])
        page_documentation(pdf, meta)

        # ---------------------------------------------------------------- 2. validation
        page_visits_per_night(pdf, df[df.v1_response.notna()], bands)
        page_v1_histograms(pdf, df, v1_per_um_dz, response=response)

        # ------------------------------------------- 3. elevation raw, then the ML chain
        # The ml-elev order has no per-band truss or z-gradient stage to build on, so it
        # branches before them; the raw elevation page is still shown for continuity with
        # the other orders.
        if chain_order == 'ml-elev':
            page_grid_fits(
                pdf, df, 'altitude_deg', 'v1_dzequiv', bands, 'elevation [deg]',
                'v1 equivalent hexapod dz [um]',
                'v1 equivalent hexapod dz against elevation, before any correction', 'deg',
                theilsen=theilsen, ylim=grid_ylim)
            page_grid_fits(
                pdf, df, 'truss_temp_mean_c', 'v1_dzequiv', bands,
                'mean TMA truss temperature [deg C]',
                'v1 equivalent hexapod dz [um]',
                f'{short} as equivalent hexapod dz against mean TMA truss temperature '
                f'({spec["label"].split(" = ", 1)[1]})', 'deg C',
                theilsen=theilsen, ylim=grid_ylim)
            df = _chain_ml(pdf, df, bands, theilsen=theilsen, verbose=verbose)
            _chain_summary(df, bands, chain_order, verbose=verbose)
            return df

        # ---------------------------------------------------------------- 3. truss, grid
        if verbose:
            print('\nstage 1: truss temperature')
        tr = page_grid_fits(
            pdf, df, 'truss_temp_mean_c', 'v1_dzequiv', bands,
            'mean TMA truss temperature [deg C]',
            'v1 equivalent hexapod dz [um]',
            f'{short} as equivalent hexapod dz against mean TMA truss temperature '
            f'({spec["label"].split(" = ", 1)[1]})', 'deg C',
            theilsen=theilsen, ylim=grid_ylim)

        # ---------------------------------------------------------------- 4. truss, per band
        page_single_band_fits(
            pdf, df, 'truss_temp_mean_c', 'v1_dzequiv', bands,
            'mean TMA truss temperature [deg C]',
            'v1 equivalent hexapod dz [um]',
            'v1 equivalent hexapod dz against mean TMA truss temperature', 'deg C',
            theilsen=theilsen)

        # ---------------------------------------------------------------- 5. truss residual
        df['v1_dzequiv_trusscorr'] = np.nan
        for b in bands:
            h = tr.get(b)
            if h is None:
                continue
            s = df.band == b
            df.loc[s, 'v1_dzequiv_trusscorr'] = (
                df.loc[s, 'v1_dzequiv']
                - (h['intercept'] + h['slope'] * df.loc[s, 'truss_temp_mean_c']))
        page_hists(pdf, df, 'v1_dzequiv_trusscorr', bands,
                   'v1 equivalent hexapod dz after truss temp correction [um]',
                   'v1 equivalent hexapod dz after removing the linear TMA truss '
                   'temperature dependence')

        # ---------------------------------------------------------------- 6. elevation, raw
        page_grid_fits(
            pdf, df, 'altitude_deg', 'v1_dzequiv', bands, 'elevation [deg]',
            'v1 equivalent hexapod dz [um]',
            'v1 equivalent hexapod dz against elevation, before any correction', 'deg',
            theilsen=theilsen, ylim=grid_ylim)

        if chain_order == 'truss-grad-elev':
            df = _chain_grad_first(pdf, df, bands, theilsen=theilsen, verbose=verbose)
            _chain_summary(df, bands, chain_order, verbose=verbose)
            return df

        # ---------------------------------------------------------------- 7. elevation, corr
        if verbose:
            print('\nstage 2: elevation, polynomial order scan '
                  f'over {ELEV_FIT_RANGE[0]:.0f}-{ELEV_FIT_RANGE[1]:.0f} deg '
                  '[nMAD, um dz]')
        scans = {}
        for b in bands:
            d = df[df.band == b]
            x, y = d.altitude_deg.to_numpy(float), d.v1_dzequiv_trusscorr.to_numpy(float)
            m = ((x >= ELEV_FIT_RANGE[0]) & (x <= ELEV_FIT_RANGE[1])
                 & np.isfinite(x) & np.isfinite(y))
            scans[b] = scan_poly_order(x[m], y[m])
            if verbose and scans[b]:
                print(f'  {b}: ' + '  '.join(f'order {o} {v:.4f}'
                                             for o, v in sorted(scans[b].items())))
        page_order_scan(pdf, scans, bands)
        el = page_grid_fits(
            pdf, df, 'altitude_deg', 'v1_dzequiv_trusscorr', bands, 'elevation [deg]',
            'v1 equiv hexapod dz after truss temp correction [um]',
            'Truss-corrected v1 equivalent hexapod dz against elevation, with a robust '
            f'order-{ELEV_ORDER_PLOT} polynomial over '
            f'{ELEV_FIT_RANGE[0]:.0f}-{ELEV_FIT_RANGE[1]:.0f} deg',
            'deg', order=ELEV_ORDER_PLOT, fit_range=ELEV_FIT_RANGE, theilsen=theilsen)

        # ---------------------------------------------------------------- 8. elev residual
        df['v1_dzequiv_truss_elev_corr'] = np.nan
        for b in bands:
            r = el.get(b)
            if r is None:
                continue
            s = df.band == b
            el_b = df.loc[s, 'altitude_deg'].to_numpy(float)
            # page_grid_fits returns a huber_fit dict at order 1 and a robust_poly dict above
            # it; only the latter carries a predict callable.
            model = (r['predict'](el_b) if 'predict' in r
                     else r['intercept'] + r['slope'] * el_b)
            df.loc[s, 'v1_dzequiv_truss_elev_corr'] = (
                df.loc[s, 'v1_dzequiv_trusscorr'] - model)
        page_hists(pdf, df, 'v1_dzequiv_truss_elev_corr', bands,
                   'v1 equiv hexapod dz after truss temp and elevation correction [um]',
                   'v1 equivalent hexapod dz after removing the truss temperature and '
                   'the elevation dependence')

        # ---------------------------------------------------------------- 9. M1M3 gradients
        if verbose:
            print('\nstage 3: M1M3 thermal gradients against the truss- and '
                  'elevation-corrected residual')
        gz = {}
        for col, label in GRAD_COLS:
            if col not in df.columns:
                continue
            f = page_grid_fits(
                pdf, df, col, 'v1_dzequiv_truss_elev_corr', bands, f'{label} [deg C/m]',
                'v1 equiv hexapod dz, truss and elev corrected [um]',
                f'Truss- and elevation-corrected v1 equivalent hexapod dz against the '
                f'{label}', 'deg C/m', theilsen=theilsen)
            if col == 'm1m3_z_gradient_c_per_m':
                gz = f
            if verbose:
                for b in bands:
                    h = f.get(b)
                    if h is not None:
                        print(f'  {label}, {b}: slope {h["slope"]:+.4f} um dz per deg C/m, '
                              f'Spearman rho {h["spearman_rho"]:+.3f}, n = {h["n"]}')

        # ---------------------------------------------------------------- 10. final residual
        df['v1_dzequiv_final'] = np.nan
        for b in bands:
            h = gz.get(b)
            if h is None:
                continue
            s = df.band == b
            df.loc[s, 'v1_dzequiv_final'] = (
                df.loc[s, 'v1_dzequiv_truss_elev_corr']
                - (h['intercept']
                   + h['slope'] * df.loc[s, 'm1m3_z_gradient_c_per_m']))
        page_hists(pdf, df, 'v1_dzequiv_final', bands,
                   'v1 equiv hexapod dz after truss, elevation and M1M3 z gradient '
                   'correction [um]',
                   'v1 equivalent hexapod dz after removing the truss temperature, the '
                   'elevation and the M1M3 z thermal gradient dependence')

    _chain_summary(df, bands, chain_order, verbose=verbose)
    return df


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--variant', default='v50_34__batoid__consdb_v1',
                   help='optical-state variant to report')
    p.add_argument('--bands', nargs='+', default=list(BAND_ORDER))
    p.add_argument('--in-dir', default=None,
                   help='directory holding science_lut.parquet; default aos/output/science_lut')
    p.add_argument('--out', default=None, help='output PDF path')
    p.add_argument('--v1-per-um-dz', type=float, default=None,
                   help='conversion [dimensionless per um]; computed from aos_state if omitted')
    p.add_argument('--db', default=None, help='database path, for the documentation page')
    p.add_argument('--grid-ylim', nargs=2, type=float, default=GRID_YLIM_DZEQUIV,
                   metavar=('LOW', 'HIGH'),
                   help='y-limits for the six-panel v1_dzequiv grid pages -- truss and '
                        'uncorrected elevation -- [um of equivalent hexapod dz], shared by '
                        'all panels (default %(default)s). The fits always use every point; '
                        'this sets only what is drawn')
    p.add_argument('--chain-order', default='truss-grad-elev',
                   choices=sorted(CHAIN_STAGES),
                   help='order the correction stages are fitted and subtracted in. '
                        'truss-grad-elev (default) is truss temperature, the M1M3 z thermal '
                        'gradient, then elevation; truss-elev-grad swaps the last two. '
                        'Elevation and the z gradient are correlated, so whichever is fitted '
                        'first absorbs the shared variance -- fitting the gradient first '
                        'leaves the elevation stage the variance the gradient cannot explain. '
                        'Every order names itself in the output so the sets coexist')
    p.add_argument('--no-theilsen', action='store_true',
                   help='skip the Theil-Sen leverage check quoted beside each Huber slope. '
                        'Theil-Sen forms all pairwise slopes, so it costs O(n^2) and dominates '
                        'the run time at tens of thousands of visits per band')
    p.add_argument('--drop-lut-epoch-offset-nights', action='store_true',
                   help=f'drop the {len(LUT_EPOCH_OFFSET_NIGHTS)} nights that ran a different '
                        'hexapod look-up-table configuration (see LUT_EPOCH_OFFSET_NIGHTS in '
                        'run_science_lut.py). Their commanded baseline is not comparable to '
                        'the rest, which matters even for --response trim-meas, where the LUT '
                        'term itself is left out but the Trim accumulated against that other '
                        'baseline. The output name gains _nolutepoch')
    p.add_argument('--response', default='lut-trim-meas', choices=sorted(RESPONSES),
                   help='which v-mode-1 combination every page fits. lut-trim-meas (default) '
                        'is v1_total, the full focus error. trim-meas leaves the hexapod '
                        'look-up-table baseline out and fits Trim - MEASURED alone. A '
                        'non-default response gets its own output filenames')
    a = p.parse_args(argv)

    aos_dir = pathlib.Path(__file__).resolve().parents[2]
    in_dir = pathlib.Path(a.in_dir) if a.in_dir else aos_dir / 'output' / 'science_lut'
    # Every chain order names itself in the output, including the default one: a consumer that
    # wants a particular chain (run_visit_elevation.py wants truss_grad_elev) then asks for it
    # by name and cannot be handed a different chain's residual by a change of default here.
    # A non-default response adds its own tag so the documents coexist.
    suffix = '_' + a.chain_order.replace('-', '_')
    if a.response != 'lut-trim-meas':
        suffix += '_' + a.response.replace('-', '_')
    if a.drop_lut_epoch_offset_nights:
        suffix += '_nolutepoch'
    out_path = (pathlib.Path(a.out) if a.out
                else in_dir / f'science_lut_results{suffix}.pdf')

    pv = pd.read_parquet(in_dir / 'science_lut.parquet')
    fits = pd.read_parquet(in_dir / 'science_lut_fits.parquet')
    if a.variant not in set(pv['variant']):
        print(f'variant {a.variant!r} not in {in_dir / "science_lut.parquet"}; '
              f'available: {sorted(pv["variant"].unique())}')
        return 1
    pv = pv[pv.variant == a.variant].copy()
    if a.drop_lut_epoch_offset_nights:
        drop = pv.day_obs.isin(LUT_EPOCH_OFFSET_NIGHTS)
        present = sorted(int(d) for d in pv.day_obs[drop].unique())
        pv = pv[~drop].copy()
        print(f'dropped {int(drop.sum())} visits on {len(present)} nights running a different '
              f'hexapod LUT configuration: '
              f'{", ".join(str(d) for d in present) if present else "none present"}')
    bands = [b for b in a.bands if b in set(pv['band'])]
    missing = [b for b in a.bands if b not in bands]
    if missing:
        print(f'bands with no rows in this table, skipped: {", ".join(missing)}')
    print(f'variant {a.variant}: n = {len(pv)} rows, {pv.day_obs.nunique()} nights, '
          f'bands {", ".join(bands)}')

    v1 = a.v1_per_um_dz
    if v1 is None:
        v1 = v1_per_um_dz_value()

    db = a.db
    if db is None:
        from common import efd_db
        db = str(efd_db.default_db_path()) if hasattr(efd_db, 'default_db_path') else \
            'output/value_added/aos_efd.duckdb'

    out = build_report(pv, fits, a.variant, bands, v1, out_path, db,
                       theilsen=not a.no_theilsen,
                       grid_ylim=tuple(a.grid_ylim),
                       chain_order=a.chain_order, response=a.response,
                       dropped_lut_epoch=a.drop_lut_epoch_offset_nights)
    out_pq = in_dir / f'science_lut_results{suffix}.parquet'
    keep = (['visit_id', 'day_obs', 'band', 'altitude_deg', 'truss_temp_mean_c',
             'v1_response', 'v1_total']
            + [c for c, _ in GRAD_COLS]
            + list(CHAIN_STAGES[a.chain_order]))
    out[[c for c in keep if c in out.columns]].to_parquet(out_pq)
    print(f'\nwrote {out_path}')
    print(f'wrote {out_pq}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
