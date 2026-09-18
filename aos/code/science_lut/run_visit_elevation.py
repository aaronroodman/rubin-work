"""Visit-by-visit focus against elevation: the within-night time series and its hysteresis.

The pooled and per-night fits in ``run_science_lut_report.py`` and
``run_nightly_elevation.py`` both treat a night as a cloud of points. This script keeps the
observing order instead, which is what shows the two effects a static look-up table (LUT)
cannot represent:

1. **The time series.** For a chosen night, the v-mode-1 amplitude expressed as equivalent
   camera-hexapod dz [µm] against sequence number, with elevation on the right-hand axis.
   Three panels split the same quantity into its terms -- the total focus error, the
   commanded LUT plus Trim, and the measured optical state -- so that a lag between the
   measured focus and the elevation it is responding to is visible directly. A lag means the
   correction never catches up with the elevation change, which is a different defect from a
   wrong LUT slope. Each night gets three pages: the top two panels carrying the
   truss-temperature and M1M3 z-thermal-gradient corrections, matching the rest of the
   document, so what remains against elevation is not a thermal term in disguise; the same
   terms uncorrected, so the size of the correction is visible; and the two driving
   temperatures themselves against sequence number. ``--series-uncorrected`` draws only the
   uncorrected focus page, keeping the temperature page.
2. **Rising against falling legs, per night.** The truss-temperature- and M1M3
   z-thermal-gradient-corrected focus error against elevation, one panel per night, with
   visits coloured by the direction elevation is moving and the two directions fitted
   separately. A systematic rising-minus-falling slope difference is hysteresis.

3. **The measured state alone, by band.** The uncorrected measured amplitude per band, as a
   histogram with its robust RMS, and against the elevation change from the immediately
   preceding exposure. The second of these asks whether a single slew leaves a focus error
   behind, which is the per-exposure form of the same catching-up question.

4. **The per-night offset against ``day_obs``.** Each night's fit evaluated at
   ``--ref-elev-deg`` rather than at its 0 deg intercept, which is where the surviving
   residual after the thermal correction would show up as a night-to-night offset rather
   than as a wrong elevation slope. Plotted with the within-night residual scatter as the
   error bar, with a Theil-Sen trend across the run, and against the slope, since a strong
   offset-slope correlation would mean the two are not separately identified by the elevation
   range a night covers.

Figure groups in order: the per-night time series pages (three per requested night --
corrected focus terms, uncorrected focus terms, driving temperatures), the two per-band pages
of the uncorrected measured state, the 12-nights-per-page rising/falling panels, the
per-night rising and falling slopes with the distribution of their difference, the
distribution of the all-points slope, and the per-night offset against ``day_obs`` with its
distribution and its relation to the slope.

Run from ``aos/``::

    python code/science_lut/run_visit_elevation.py
    python code/science_lut/run_visit_elevation.py --day-obs 20260706 20260707
    python code/science_lut/run_visit_elevation.py --day-obs 20260706 \
        --seq-num 0 200 --series-only --out output/science_lut/visit_series_20260706.pdf

Reads two parquet files written earlier in the study, joined on ``visit_id``:

* ``output/science_lut/science_lut.parquet`` (``run_science_lut.py``) for ``seq_num``,
  ``obs_start_mjd``, ``v1_lut_trim`` and ``v1_meas``;
* ``output/science_lut/science_lut_results_<chain-tag>.parquet``
  (``run_science_lut_report.py --chain-order truss-grad-elev``) for
  ``v1_dzequiv_truss_grad_corr``, the focus error after both temperature corrections.

Writes ``visit_elevation.pdf``, ``visit_elevation_slopes.parquet`` and
``visit_elevation_measured_by_band.parquet`` alongside them.

``--response trim-meas --chain-tag truss_grad_elev_trim_meas_nolutepoch`` builds the same
document for ``v1(Trim) - v1(measured)`` -- the hexapod LUT baseline left out -- with the
nights running a different LUT configuration dropped, and tags every output name to match.
The look-up-table term carries essentially the whole elevation dependence, so that variant
shows what the closed loop and the wavefront sensors do on their own. Both flags must agree:
the response terms come from ``science_lut.parquet`` and are selected by ``--response``,
while the corrected residual is read from the ``--chain-tag`` file as already computed.

The plotted window for the per-night elevation slope also follows ``--response``, since
excluding the LUT term moves that distribution from about -17 to zero µm of equivalent
camera-hexapod dz per deg; ``--slope-lim`` overrides it.

Notes
-----
The reordered-chain parquet is required rather than optional: the default chain order
removes elevation before the M1M3 z gradient, so its residual column has already had the
elevation dependence this script is measuring subtracted out.

Every v-mode-1 amplitude is converted to µm of equivalent camera-hexapod dz by dividing by
``v1_per_um_dz``, taken from ``run_science_lut_report.v1_per_um_dz_value`` so the factor
cannot drift from the one in the main document. That import needs ``lsst.ts.ofc``; pass
``--v1-per-um-dz`` to run without the AOS environment.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.utils import nmad                                       # noqa: E402
from run_nightly_elevation import (                                 # noqa: E402
    DIRECTION_DEADBAND, DIRECTION_WINDOW, MIN_VISITS_LEG, MIN_VISITS_NIGHT,
    huber_slope, label_direction)
from run_science_lut import (LUT_EPOCH_OFFSET_NIGHTS,               # noqa: E402
                            MEASURED_SIGN)

#: Default chain parquet read for the corrected residual, as
#: ``science_lut_results_<CHAIN_TAG>.parquet``. The truss then M1M3-z-gradient then elevation
#: order is required: the other order removes elevation first, leaving a residual with the
#: elevation dependence already subtracted, which is the quantity this document is about.
CHAIN_TAG = 'truss_grad_elev'

#: Panel labels per response, so the series page names the quantity it actually draws.
#: ``lut-trim-meas`` is the full focus error; ``trim-meas`` leaves the look-up-table baseline
#: out and shows the closed-loop Trim against the measured state.
RESPONSE_LABELS = {
    'lut-trim-meas': dict(total='total focus error',
                          commanded='commanded LUT + Trim',
                          formula='LUT + Trim - measured'),
    'trim-meas': dict(total='Trim - measured',
                      commanded='commanded Trim',
                      formula='Trim - measured, no LUT'),
}

#: Nights drawn as a visit-by-visit time series. One PDF page each, three stacked panels.
DAY_OBS_SERIES = [20260706]

#: Focus error after the truss-temperature and M1M3 z-thermal-gradient corrections
#: [µm of equivalent camera-hexapod dz], from the reordered correction chain.
YCOL = 'v1_dzequiv_truss_grad_corr'

#: Nights per page of the rising/falling panels, as 4 columns by 3 rows.
NIGHTS_PER_PAGE = 12

#: Panel height for those pages [µm of equivalent camera-hexapod dz], centred on each
#: night's own median. The per-night 1st-to-99th percentile span is 769 µm at the median
#: night and 1174 µm at the 95th percentile of nights, so this holds nearly every visit
#: while keeping one common scale for comparing slopes panel to panel.
PANEL_YSPAN = 1200.0

#: Colours for the elevation direction, shared by every page so the legend need not repeat.
DIR_COLOUR = {'up': 'tab:red', 'down': 'tab:blue', 'flat': '0.6'}

#: Band order for the per-band grid pages, blue to red.
BAND_ORDER = ('u', 'g', 'r', 'i', 'z', 'y')

#: Plotted window for the uncorrected measured v-mode-1 amplitude [µm of equivalent
#: camera-hexapod dz]. Every band's 0.5th-to-99.5th percentile falls inside ±100 µm, while a
#: handful of visits reach ±900 µm; the robust statistics use every visit regardless.
MEAS_LIM = (-100.0, 100.0)

#: Histogram bins for the measured amplitude, spanning `MEAS_LIM`.
MEAS_BINS = 40

#: Plotted window for the per-visit elevation change [deg] against the previous seq_num.
#: The 1st-to-99th percentile is -6.3 to +7.4 deg with tails to ±55 deg from slews between
#: fields; the correlation statistics use every retained visit regardless.
DELTA_ELEV_LIM = (-12.0, 12.0)

#: Plotted window for the per-night elevation slope [µm of equivalent camera-hexapod dz
#: per deg], per response, shared by the per-night slope scatter and the all-points slope
#: histogram. The two responses put the slope in different places, so one window cannot
#: serve both: with the look-up-table (LUT) term included the nights sit between -27 and
#: -10, carrying the LUT's own elevation term, while with it excluded they sit at zero
#: (median -0.072 µm per deg). Autoscaling instead lets a single failed fit -- near +30 for
#: the full response, +50.5 for ``trim-meas`` -- compress every other night into a few
#: pixels. The fits and the reported statistics use every night regardless.
SLOPE_LIM_BY_RESPONSE = {
    'lut-trim-meas': (-30.0, 0.0),
    'trim-meas': (-10.0, 10.0),
}

#: Default window, for the full response; `SLOPE_LIM_BY_RESPONSE` overrides it per response.
SLOPE_LIM = SLOPE_LIM_BY_RESPONSE['lut-trim-meas']

#: Histogram bins for the all-points slope, spanning the response's slope window.
SLOPE_BINS = 10

#: Polynomial orders in elevation fitted to the look-up-table term alone. The first is the
#: one drawn; the rest are reported as residual nMAD so any curvature is quantified.
LUT_ELEV_ORDERS = (1, 2, 3)

#: Elevation at which the per-night fit is evaluated to give that night's offset [deg]. The
#: sample median elevation is 61.60 deg, so a round 60 deg sits inside the bulk of every
#: night. The fitted intercept at 0 deg is an extrapolation roughly 60 deg outside the data
#: and is therefore strongly anti-correlated with the slope: a night's offset read there
#: mixes in its slope error, which is exactly what a night-to-night comparison must avoid.
REF_ELEV_DEG = 60.0

#: Plotted window for the per-night offset [µm of equivalent camera-hexapod dz]. The 5th-to-95th
#: percentile of the nights is -113.0 to +71.0 µm, while three nights reach -1622.6, -1492.9 and
#: +843.8 µm — the first two have 47 and 171 visits and within-night residual nMAD near 80 µm,
#: the third a fitted slope of +49.4 µm per deg, so all three are failed or barely-determined
#: fits rather than real offsets. Autoscaling to them compresses every other night into a few
#: pixels. Each panel reports how many nights fall outside, and the medians, nMADs and trends
#: use every night regardless.
OFFSET_LIM = (-250.0, 250.0)

#: Nights running a different hexapod look-up-table configuration; see
#: `run_science_lut.LUT_EPOCH_OFFSET_NIGHTS`, which owns the list and the reasoning.
LUT_EPOCH_EARLY_NIGHTS = LUT_EPOCH_OFFSET_NIGHTS


def load_visits(in_dir, variant=None, bands=None, v1_per_um_dz=None,
                chain_tag=CHAIN_TAG, response='lut-trim-meas', ycol=YCOL,
                verbose=True):
    """Join the per-visit table to the corrected-chain table and label slew direction.

    Parameters
    ----------
    in_dir : `pathlib.Path`
        Directory holding both parquet files.
    variant : `str`, optional
        Optical-state variant id; default the first present.
    bands : `list` [`str`], optional
        Bands to keep; default every band present.
    v1_per_um_dz : `float`
        Dimensionless v-mode-1 amplitude per µm of hexapod dz. Required.
    chain_tag : `str`, optional
        Names the chain parquet read for `ycol`, as
        ``science_lut_results_<chain_tag>.parquet``. The join with it is inner, so a chain
        built with nights dropped drops them here too.
    response : `str`, optional
        ``'lut-trim-meas'`` (default) or ``'trim-meas'``. Selects what ``dz_total`` and
        ``dz_commanded`` hold; must match the response the chain parquet was built with, or
        the uncorrected panels and the corrected one describe different quantities.
    ycol : `str`, optional
        Corrected-residual column to read from the chain parquet [µm of equivalent
        camera-hexapod dz]. ``v1_dzequiv_truss_grad_corr`` for the per-band chain,
        ``v1_dzequiv_mlcorr`` for the band-independent thermal model.
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        One row per visit, sorted by time, carrying ``seq_num``, ``altitude_deg`` [deg],
        ``direction``, the dz-equivalent terms ``dz_total``, ``dz_commanded``, ``dz_lut``,
        ``dz_lut_trim`` and ``dz_meas`` [µm], their truss and M1M3-z-gradient corrected
        counterparts ``dz_total_corr`` and ``dz_commanded_corr`` with the correction itself as
        ``dz_corr_applied`` [µm], `YCOL` [µm], and ``delta_elev_deg`` [deg] -- the elevation
        change from ``seq_num - 1``, NaN unless that exposure is present in the same night.
    variant : `str`
        The variant actually used.

    Notes
    -----
    ``dz_total = dz_commanded - dz_meas`` by construction, for either response, and the same
    identity holds for the corrected pair; the three panels of the time series therefore
    decompose one quantity rather than showing three independent measurements.
    """
    base = pd.read_parquet(in_dir / 'science_lut.parquet')
    chain_path = in_dir / f'science_lut_results_{chain_tag}.parquet'
    if not chain_path.exists():
        report_args = '--chain-order truss-grad-elev'
        if 'trim_meas' in chain_tag:
            report_args += ' --response trim-meas'
        if 'nolutepoch' in chain_tag:
            report_args += ' --drop-lut-epoch-offset-nights'
        raise SystemExit(
            f'{chain_path} is missing. Produce it first with:\n'
            f'  python code/science_lut/run_science_lut_report.py {report_args}\n'
            f'The chain order matters: truss-elev-grad removes elevation before the M1M3 z '
            f'gradient, so its residual has the elevation dependence already subtracted.')
    chain = pd.read_parquet(chain_path)

    variant = variant or sorted(base.variant.unique())[0]
    base = base[base.variant == variant]
    keep = ['visit_id', 'day_obs', 'seq_num', 'obs_start_mjd', 'band', 'altitude_deg',
            'v1_total', 'v1_lut_trim', 'v1_lut', 'v1_trim', 'v1_meas']
    # An inner join on the chain table carries its row selection across: a chain built with
    # --drop-lut-epoch-offset-nights has no rows on those nights, so they are dropped here too
    # without this script needing to know which nights those were.
    # The two driving temperatures come from the chain table rather than science_lut.parquet so
    # they are by construction the same values the correction was fitted against.
    if ycol not in chain.columns:
        raise SystemExit(
            f'{chain_path.name} has no column {ycol!r}; its correction-chain columns are '
            + ', '.join(c for c in chain.columns if c.startswith('v1_dzequiv')))
    df = base[keep].merge(chain[['visit_id', ycol, 'truss_temp_mean_c',
                                 'm1m3_z_gradient_c_per_m']],
                          on='visit_id', how='inner')
    if bands:
        df = df[df.band.isin(bands)]
    df = df.sort_values('obs_start_mjd').reset_index(drop=True)

    # Every term shares the one conversion to equivalent camera-hexapod dz. The commanded term
    # and the total depend on the response: lut-trim-meas totals LUT + Trim - measured, while
    # trim-meas leaves the look-up-table baseline out and totals Trim - measured.
    df['dz_meas'] = df.v1_meas / v1_per_um_dz
    df['dz_lut'] = df.v1_lut / v1_per_um_dz
    if response == 'trim-meas':
        df['dz_commanded'] = df.v1_trim / v1_per_um_dz
        df['dz_total'] = (df.v1_trim + MEASURED_SIGN * df.v1_meas) / v1_per_um_dz
    else:
        df['dz_commanded'] = df.v1_lut_trim / v1_per_um_dz
        df['dz_total'] = df.v1_total / v1_per_um_dz
    # Kept under its old name too, so the look-up-table page reads the same column either way.
    df['dz_lut_trim'] = df.v1_lut_trim / v1_per_um_dz

    # The truss and M1M3-z-gradient corrected versions of the same two terms. The chain parquet
    # stores only the corrected RESIDUAL of the whole response (`YCOL`), not the per-band model
    # coefficients, so the correction itself is recovered as the difference and then applied to
    # the commanded term as well. That keeps dz_total_corr = dz_commanded_corr - dz_meas exact,
    # and attributes the whole correction to the commanded term -- which is where it belongs:
    # the truss temperature and the mirror thermal gradient act on the telescope, not on the
    # wavefront sensors' estimate of it. NaN wherever a correction stage had no telemetry.
    df['dz_corr_applied'] = df.dz_total - df[ycol]
    df['dz_total_corr'] = df[ycol]
    df['dz_commanded_corr'] = df.dz_commanded - df.dz_corr_applied

    # Direction is labelled per night: the rolling median must not span the gap between
    # nights, where elevation jumps with no slew in between.
    df['direction'] = pd.concat(
        [label_direction(d) for _, d in df.groupby('day_obs')]).reindex(df.index)

    # Elevation change from the immediately preceding exposure, for the scatter page. Only
    # a visit whose seq_num - 1 is present in the SAME night qualifies: a gap means the
    # previous exposure is missing from this table -- a non-science image, a different
    # variant, or a cut band -- so the difference would span an unknown interval, and
    # across a day_obs boundary it would span the daytime gap. Everything else gets NaN.
    df = df.sort_values(['day_obs', 'seq_num'])
    grp = df.groupby('day_obs')
    contiguous = (df.seq_num - grp.seq_num.shift(1)) == 1
    df['delta_elev_deg'] = (df.altitude_deg - grp.altitude_deg.shift(1)).where(contiguous)
    df = df.sort_values('obs_start_mjd').reset_index(drop=True)

    if verbose:
        print(f'variant {variant}: {len(df)} visits over {df.day_obs.nunique()} nights, '
              f'bands {sorted(df.band.unique())}')
        vc = df.direction.value_counts()
        print(f'direction labelling (centred rolling median of {DIRECTION_WINDOW} visits, '
              f'deadband {DIRECTION_DEADBAND} deg per visit):')
        for k in ('up', 'down', 'flat'):
            print(f'  {k:5s} {int(vc.get(k, 0)):7d} visits')
        for col, label in (('dz_total', RESPONSE_LABELS[response]['total']),
                           ('dz_commanded', RESPONSE_LABELS[response]['commanded']),
                           ('dz_meas', 'measured optical state')):
            v = df[col].to_numpy(float)
            v = v[np.isfinite(v)]
            print(f'  {label:24s}: median {np.median(v):+9.1f} um, '
                  f'nMAD {nmad(v):8.1f} um of equivalent camera-hexapod dz')
        n_ok = int(df.delta_elev_deg.notna().sum())
        print(f'elevation change from seq_num - 1 in the same night: {n_ok} of {len(df)} '
              f'visits qualify ({100 * n_ok / len(df):.1f}%), '
              f'{len(df) - n_ok} dropped as a night start or a seq_num gap')
    return df, variant


def _stacked_series_figure(d, panels, suptitle):
    """Stacked per-visit time series against ``seq_num``, elevation twinned on every panel.

    Shared by the focus-term pages and the driving-temperature page so all of them keep one
    shape, one x axis and one elevation reference curve.

    Parameters
    ----------
    d : `pandas.DataFrame`
        One night, sorted by time, carrying ``seq_num``, ``altitude_deg`` [deg] and every
        column named in `panels`.
    panels : `list` [`tuple`]
        One ``(column, ylabel, colour, note)`` per panel, top to bottom. The ylabel carries
        the units, since the panels do not share a y scale.
    suptitle : `str`

    Returns
    -------
    fig : `matplotlib.figure.Figure`

    Notes
    -----
    The page is landscape 13.3 by 7.5 inch, the 16:9 shape of a presentation slide, so the
    wide panels transfer to a slide without rescaling.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(panels), 1, figsize=(13.3, 7.5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (col, ylabel, colour, note) in zip(axes, panels):
        ax.plot(d.seq_num, d[col], '.-', ms=3, lw=0.6, color=colour)
        ax.set_ylabel(ylabel, fontsize=8, color=colour)
        ax.tick_params(axis='y', labelsize=7, labelcolor=colour)
        ax.grid(alpha=0.3)
        ax.set_title(note, fontsize=8, loc='left')
        # Elevation on a twin axis rather than an extra panel, so the eye reads the lag
        # between a slew and the response off one shared x axis.
        axr = ax.twinx()
        axr.plot(d.seq_num, d.altitude_deg, '-', lw=1.0, color='0.35', alpha=0.8)
        axr.set_ylabel('elevation [deg]', fontsize=8, color='0.35')
        axr.tick_params(axis='y', labelsize=7, labelcolor='0.35')
    axes[-1].set_xlabel('seq_num', fontsize=9)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    return fig


def night_temperature_page(pdf, d, day_obs, variant, verbose=True):
    """The two driving temperatures for one night, against sequence number.

    These are the quantities the focus correction is fitted against, so seeing them in
    observing order says whether a within-night focus drift tracks a temperature that is
    itself drifting, or moves independently of both temperatures.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    d : `pandas.DataFrame`
        One night, sorted by time, carrying ``truss_temp_mean_c`` [°C] and
        ``m1m3_z_gradient_c_per_m`` [°C per m].
    day_obs : `int`
    variant : `str`
    verbose : `bool`, optional

    Notes
    -----
    Both columns are read from the chain parquet, so they are the same values the correction
    was fitted against rather than an independently re-derived pair.
    """
    import matplotlib.pyplot as plt

    panels = [
        ('truss_temp_mean_c', 'TMA truss temperature\n[deg C]', 'tab:red',
         'truss_temp_mean_c: mean of the +x+y and -x-y truss sensors, the first correction '
         'term'),
        ('m1m3_z_gradient_c_per_m', 'M1M3 z thermal gradient\n[deg C per m]', 'tab:purple',
         'm1m3_z_gradient_c_per_m: front-to-back mirror gradient, the second correction term'),
    ]
    fig = _stacked_series_figure(
        d, panels,
        suptitle=(f'{day_obs}: the two driving temperatures against sequence number, '
                  f'elevation in grey on the right axis ({len(d)} visits, seq_num '
                  f'{int(d.seq_num.min())} to {int(d.seq_num.max())}) — {variant}'))
    pdf.savefig(fig)
    plt.close(fig)

    if verbose:
        for col, unit in (('truss_temp_mean_c', 'deg C'),
                          ('m1m3_z_gradient_c_per_m', 'deg C per m')):
            v = d[col].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                print(f'  {col}: no finite values this night')
                continue
            print(f'  {col:26s}: {v.min():+8.3f} to {v.max():+8.3f} {unit}, '
                  f'range {v.max() - v.min():7.3f} {unit}, n = {len(v)} visits')


def night_series_page(pdf, d, day_obs, variant, response='lut-trim-meas', corrected=True):
    """Three stacked time series for one night, elevation on the right-hand axis.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    d : `pandas.DataFrame`
        One night, sorted by time.
    day_obs : `int`
    variant : `str`
    response : `str`, optional
        Key into `RESPONSE_LABELS`, naming the terms the top two panels draw.
    corrected : `bool`, optional
        Draw the truss-temperature and M1M3-z-thermal-gradient corrected terms
        (``dz_total_corr``, ``dz_commanded_corr``) in the top two panels, the default. False
        draws the uncorrected ``dz_total`` and ``dz_commanded``. The measured panel is
        uncorrected either way -- both corrections are applied to the commanded term.

    Notes
    -----
    The page is landscape 13.3 by 7.5 inch, the 16:9 shape of a presentation slide, so the
    three wide panels transfer to a slide without rescaling.
    """
    import matplotlib.pyplot as plt

    lab = RESPONSE_LABELS[response]
    if response == 'trim-meas':
        top_note = 'v1_trim - v1_measured: no look-up-table baseline'
        mid_note = 'v1_trim: the accumulated closed-loop offset'
    else:
        top_note = 'v1_total: commanded minus measured'
        mid_note = 'v1_lut + v1_trim: what the control system asked for'
    if corrected:
        corr_tag = ', truss + M1M3 z-gradient corrected'
        top_note += corr_tag
        mid_note += corr_tag
        top_col, mid_col = 'dz_total_corr', 'dz_commanded_corr'
        top_lab = f'{lab["total"]}\ncorrected [um dz]'
        mid_lab = f'{lab["commanded"]}\ncorrected [um dz]'
    else:
        top_col, mid_col = 'dz_total', 'dz_commanded'
        top_lab = f'{lab["total"]}\n[um dz]'
        mid_lab = f'{lab["commanded"]}\n[um dz]'
    panels = [
        (top_col, top_lab, 'tab:green', top_note),
        (mid_col, mid_lab, 'tab:orange', mid_note),
        ('dz_meas', 'measured state\n[um dz]', 'tab:blue',
         'v1_measured: what the wavefront sensors saw'),
    ]
    fig = _stacked_series_figure(
        d, panels,
        suptitle=(f'{day_obs}: v-mode-1 focus terms as equivalent camera-hexapod dz '
                  f'against sequence number ({lab["formula"]}'
                  + ('; top two panels truss + M1M3 z-gradient corrected'
                     if corrected else '; no thermal correction')
                  + f'), elevation in grey on the right axis ({len(d)} visits, seq_num '
                  f'{int(d.seq_num.min())} to {int(d.seq_num.max())}) — {variant}'))
    pdf.savefig(fig)
    plt.close(fig)


def series_lag(d, verbose=True):
    """Cross-correlate the measured focus against elevation to estimate the response lag.

    Parameters
    ----------
    d : `pandas.DataFrame`
        One night, sorted by time, with ``dz_meas`` [µm] and ``altitude_deg`` [deg].
    verbose : `bool`, optional

    Returns
    -------
    out : `dict` or `None`
        ``lag_visits`` at the strongest absolute correlation, that ``rho`` (Spearman,
        dimensionless), and ``rho_zero`` at zero lag. None if the night is too short.

    Notes
    -----
    The lag is in visits, not seconds, because it is read off the observing sequence; at the
    typical 0.7 min between consecutive science visits a lag of one visit is under a minute.
    Only the sign and rough size matter here -- a positive lag means the measured focus
    follows elevation rather than leading it.
    """
    from scipy import stats

    d = d.dropna(subset=['dz_meas', 'altitude_deg'])
    if len(d) < MIN_VISITS_NIGHT:
        return None
    y = d.dz_meas.to_numpy(float)
    e = d.altitude_deg.to_numpy(float)
    rhos = {}
    for lag in range(0, 31):
        if len(y) - lag < MIN_VISITS_LEG:
            break
        # Elevation is advanced against the focus, so a positive lag asks whether the
        # measured focus at visit i+lag tracks the elevation at visit i.
        rho = float(stats.spearmanr(e[:len(e) - lag] if lag else e,
                                    y[lag:] if lag else y)[0])
        if np.isfinite(rho):
            rhos[lag] = rho
    if 0 not in rhos:
        return None
    best_lag = max(rhos, key=lambda k: abs(rhos[k]))
    out = dict(lag_visits=best_lag, rho=rhos[best_lag], rho_zero=rhos[0])
    if verbose:
        print(f'  measured focus against elevation: strongest Spearman rho '
              f'{out["rho"]:+.3f} (dimensionless) at a lag of {out["lag_visits"]} visits, '
              f'against {out["rho_zero"]:+.3f} at zero lag')
    return out


def per_night_direction_slopes(df, ycol=YCOL, ref_elev_deg=REF_ELEV_DEG, verbose=True):
    """Fit the elevation dependence per night, once for all points and once per direction.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit table with ``day_obs``, ``altitude_deg`` [deg], ``direction`` and `ycol`.
    ycol : `str`, optional
        Response column [µm of equivalent camera-hexapod dz].
    ref_elev_deg : `float`, optional
        Elevation at which each night's fit is evaluated to give ``offset_*`` [deg].
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        One row per night with ``slope_all``, ``slope_up``, ``slope_down`` and their standard
        errors [µm of equivalent camera-hexapod dz per deg], the counts behind each,
        ``intercept_*`` at 0 deg and ``offset_*`` at `ref_elev_deg` [µm of equivalent
        camera-hexapod dz], the night median of `ycol` in the same unit, and ``difference``
        = rising minus falling with ``difference_sigma`` in units of the combined standard
        error.

    Notes
    -----
    Huber (`statsmodels` `RLM` with `HuberT`) throughout, matching the rest of the study.

    ``offset_*`` rather than ``intercept_*`` is the quantity to compare night to night: the
    0 deg intercept lies about 60 deg outside the observed elevation range, so its scatter is
    dominated by the slope error propagated over that lever arm.
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
            # The night's offset, read at REF_ELEV_DEG rather than at the 0 deg intercept,
            # so it is a value inside the data instead of a 60 deg extrapolation.
            r[f'offset_{key}'] = (f['intercept'] + f['slope'] * ref_elev_deg
                                  if f else np.nan)
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
              f'[um of equivalent camera-hexapod dz per deg], all points')
        print(f'  nights fitted       : {len(a)}')
        print(f'  median slope        : {np.median(a):+.3f} um per deg')
        print(f'  nMAD of the slopes  : {nmad(a):.3f} um per deg')
        print(f'  full range          : {a.min():+.3f} to {a.max():+.3f} um per deg')
        print(f'  median formal error : {out.err_all.median():.3f} um per deg')

        o = out.offset_all.dropna().to_numpy(float)
        if len(o):
            print(f'\nper-night offset of {ycol} at {ref_elev_deg:.0f} deg elevation '
                  f'[um of equivalent camera-hexapod dz], all points')
            print(f'  nights fitted       : {len(o)}')
            print(f'  median offset       : {np.median(o):+.1f} um')
            print(f'  nMAD of the offsets : {nmad(o):.1f} um')
            print(f'  full range          : {o.min():+.1f} to {o.max():+.1f} um')
            # The night-to-night scatter of the offset against the within-night scatter is
            # the question: a large ratio says the residual is a per-night constant.
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
            print(f'  median difference   : {med:+.3f} um of equivalent camera-hexapod dz '
                  f'per deg')
            print(f'  nMAD of differences : '
                  f'{nmad(both.difference.to_numpy(float)):.3f} um per deg')
            n_sig = int((both.difference_sigma.abs() > 3).sum())
            print(f'  nights differing by more than 3 combined standard errors: '
                  f'{n_sig} of {len(both)}')
            print(f'  nights with rising steeper than falling: {pos} of {len(both)} '
                  f'(sign-test p = {p:.3g})')
            if p < 0.01:
                print('  -> a consistent direction-dependent offset, i.e. hysteresis, '
                      'rather than symmetric night-to-night scatter')
            else:
                print('  -> no consistent direction dependence; the rising-minus-falling '
                      'difference scatters about zero')
    return out


def night_panel_pages(pdf, df, slopes, ycol=YCOL, n_per_page=NIGHTS_PER_PAGE,
                      max_nights=None, ylim=None, yspan=PANEL_YSPAN):
    """One panel per night: the corrected focus error against elevation, split by direction.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    slopes : `pandas.DataFrame`
        Output of `per_night_direction_slopes`.
    ycol : `str`, optional
    n_per_page : `int`, optional
        Panels per page, drawn as 4 columns by 3 rows.
    max_nights : `int`, optional
    ylim : `tuple` [`float`], optional
        Absolute shared y-limits [µm of equivalent camera-hexapod dz]. Overrides `yspan`.
    yspan : `float`, optional
        Panel height [µm of equivalent camera-hexapod dz], centred on each night's own
        median. None autoscales each panel independently.

    Notes
    -----
    The default is a shared *span* about each night's own median rather than one absolute
    window, because the corrected residual still carries a per-night offset of order
    1000 µm that the truss and gradient fits do not remove; a single absolute window wide
    enough to hold every night would leave each panel's slope too small to read, while a
    common span makes the slopes directly comparable panel to panel.
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
        fig.supylabel(f'{ycol} [um of equivalent camera-hexapod dz]', fontsize=9)
        fig.suptitle('per-night elevation dependence after the truss-temperature and M1M3 '
                     'z-thermal-gradient corrections; slopes in um of equivalent '
                     'camera-hexapod dz per deg', fontsize=10)
        fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)


def measured_hist_page(pdf, df, variant, bands=BAND_ORDER, lim=MEAS_LIM, bins=MEAS_BINS,
                       verbose=True):
    """Per-band histogram of the uncorrected measured amplitude, with a robust RMS.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``band`` and ``dz_meas`` [µm of equivalent camera-hexapod dz].
    variant : `str`
    bands : `sequence` [`str`], optional
        Bands drawn, one panel each, as 3 columns by 2 rows.
    lim : `tuple` [`float`], optional
        Plotted window [µm of equivalent camera-hexapod dz]; None autoscales per panel.
    bins : `int`, optional
        Bins spanning `lim`.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``n``, ``median``, ``nmad`` and ``rms`` [µm of equivalent camera-hexapod
        dz], and the count outside the plotted window.

    Notes
    -----
    ``nmad`` is the robust scatter and is the number to quote; the plain ``rms`` about the
    median is reported beside it because a handful of visits reach several hundred µm, and the
    ratio of the two says how much the tail inflates a non-robust estimate.

    This is the *uncorrected* measured state -- no truss-temperature, thermal-gradient or
    elevation term removed.
    """
    import matplotlib.pyplot as plt

    unit = 'um of equivalent camera-hexapod dz'
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, band in zip(axes.ravel(), bands):
        v = df.loc[df.band == band, 'dz_meas'].to_numpy(float)
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
    fig.suptitle(f'uncorrected measured optical state by band; robust RMS is the nMAD '
                 f'— {variant}', fontsize=10)
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
        allv = df.dz_meas.to_numpy(float)
        allv = allv[np.isfinite(allv)]
        print(f'  {"all":5s} {allv.size:7d} {np.median(allv):+9.1f} {nmad(allv):11.1f} '
              f'{np.sqrt(np.mean((allv - np.median(allv)) ** 2)):10.1f}')
        print('  ratio is plain RMS over robust RMS (dimensionless); above ~1.5 the tail '
              'dominates a\n  non-robust estimate')
    return out


def measured_vs_delta_elev_page(pdf, df, variant, bands=BAND_ORDER, lim=MEAS_LIM,
                                xlim=DELTA_ELEV_LIM, verbose=True):
    """Per-band scatter of the measured amplitude against the elevation change per exposure.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``band``, ``dz_meas`` [µm of equivalent camera-hexapod dz] and
        ``delta_elev_deg`` [deg].
    variant : `str`
    bands : `sequence` [`str`], optional
    lim : `tuple` [`float`], optional
        Plotted y window [µm of equivalent camera-hexapod dz].
    xlim : `tuple` [`float`], optional
        Plotted x window [deg].
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``n``, the Huber ``slope`` [µm of equivalent camera-hexapod dz per deg]
        with its standard error, Pearson r and Spearman rho (both dimensionless).

    Notes
    -----
    Only visits whose ``seq_num - 1`` is present in the same night carry a finite
    ``delta_elev_deg``, so the cut is already applied by `load_visits`; no difference spans a
    seq_num gap or a day_obs boundary.

    The band is that of the current visit. The previous exposure may be a different band,
    which is the right comparison here -- what matters is the elevation move immediately
    before this exposure, whatever was observed then.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    unit = 'um of equivalent camera-hexapod dz'
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, band in zip(axes.ravel(), bands):
        d = df[(df.band == band) & df.delta_elev_deg.notna() & df.dz_meas.notna()]
        if not len(d):
            ax.axis('off')
            continue
        x = d.delta_elev_deg.to_numpy(float)
        y = d.dz_meas.to_numpy(float)
        h = huber_slope(x, y, min_n=MIN_VISITS_LEG)
        pear = float(stats.pearsonr(x, y)[0])
        spear = float(stats.spearmanr(x, y)[0])
        rows.append(dict(band=band, n=len(d),
                         slope=(h['slope'] if h else np.nan),
                         slope_err=(h['slope_err'] if h else np.nan),
                         pearson_r=pear, spearman_rho=spear))
        ax.scatter(x, y, s=2, alpha=0.2, color='tab:blue', edgecolors='none')
        ax.axhline(0, color='grey', lw=0.8)
        ax.axvline(0, color='grey', lw=0.8)
        if h is not None:
            xg = np.linspace(*(xlim if xlim else (x.min(), x.max())), 20)
            ax.plot(xg, h['intercept'] + h['slope'] * xg, 'r-', lw=1.4)
        if lim is not None:
            ax.set_ylim(*lim)
        if xlim is not None:
            ax.set_xlim(*xlim)
        slope_txt = (f'{h["slope"]:+.2f} +/- {h["slope_err"]:.2f} um per deg'
                     if h else 'no fit')
        ax.set_title(f'{band}   n = {len(d)}\nHuber slope {slope_txt}\n'
                     f'Pearson r {pear:+.3f}, Spearman rho {spear:+.3f}', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(bands):]:
        ax.axis('off')
    fig.supxlabel('elevation change from the previous seq_num [deg]', fontsize=9)
    fig.supylabel(f'measured v-mode-1 amplitude, uncorrected [{unit}]', fontsize=9)
    fig.suptitle(f'uncorrected measured optical state against the elevation move '
                 f'immediately before the exposure — {variant}', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f'\nuncorrected measured amplitude against the elevation change from '
              f'seq_num - 1')
        print(f'  {"band":5s} {"n":>7s} {"slope":>9s} {"err":>7s} {"Pearson r":>10s} '
              f'{"Spearman rho":>13s}')
        for _, r in out.iterrows():
            print(f'  {r.band:5s} {int(r.n):7d} {r.slope:+9.2f} {r.slope_err:7.2f} '
                  f'{r.pearson_r:+10.3f} {r.spearman_rho:+13.3f}')
        print(f'  slope in {unit} per deg; the correlations are dimensionless')
    return out


def lut_vs_elevation_page(pdf, df, variant, bands=BAND_ORDER, orders=LUT_ELEV_ORDERS,
                          verbose=True):
    """Per-band scatter of the look-up-table term alone against elevation.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
        Carries ``band``, ``altitude_deg`` [deg] and ``dz_lut`` [µm of equivalent
        camera-hexapod dz].
    variant : `str`
    bands : `sequence` [`str`], optional
        Bands drawn, one panel each, as 3 columns by 2 rows.
    orders : `sequence` [`int`], optional
        Polynomial orders fitted in elevation; the first is drawn.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per band: ``n`` and its split into ``n_main``/``n_early``, the median of each epoch
        [µm of equivalent camera-hexapod dz], the main-epoch Huber linear ``slope`` [µm per
        deg] with its standard error, Pearson r and Spearman rho (both dimensionless), and
        the residual nMAD [µm] after each fitted order.

    Notes
    -----
    This is the look-up-table term on its own -- no Trim, no measured state, and no
    truss-temperature or thermal-gradient correction -- so the slope here is the elevation
    dependence the control system is *already* applying. Comparing it against the residual
    slope of -16.87 µm of equivalent camera-hexapod dz per deg found after the temperature
    corrections says how much of the total elevation dependence the look-up table has yet to
    account for.

    The `LUT_EPOCH_EARLY_NIGHTS` are drawn but excluded from every fit; see that constant for
    why. Each panel autoscales, because the per-band medians span -1237 µm (u) to +2892 µm
    (y) and one shared window would compress most panels to a few pixels.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    unit = 'um of equivalent camera-hexapod dz'
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, band in zip(axes.ravel(), bands):
        d = df[(df.band == band) & df.dz_lut.notna() & df.altitude_deg.notna()]
        if not len(d):
            ax.axis('off')
            continue
        early = d.day_obs.isin(LUT_EPOCH_EARLY_NIGHTS)
        main, off = d[~early], d[early]
        # The fit uses the main epoch only: the offset nights are a different look-up-table
        # configuration, and pooling the two turns a 2400 um offset into apparent curvature.
        x = main.altitude_deg.to_numpy(float)
        y = main.dz_lut.to_numpy(float)
        r = dict(band=band, n=len(d), n_main=len(main), n_early=len(off),
                 median_main=float(main.dz_lut.median()),
                 median_early=(float(off.dz_lut.median()) if len(off) else np.nan))
        ax.scatter(x, y, s=2, alpha=0.2, color='tab:orange', edgecolors='none',
                   label=f'main epoch, n = {len(main)}')
        if len(off):
            ax.scatter(off.altitude_deg, off.dz_lut, s=2, alpha=0.35, color='tab:purple',
                       edgecolors='none', label=f'offset epoch, n = {len(off)}')
        h = huber_slope(x, y, min_n=MIN_VISITS_LEG) if len(main) >= MIN_VISITS_LEG else None
        r['slope'] = h['slope'] if h else np.nan
        r['slope_err'] = h['slope_err'] if h else np.nan
        if len(main) >= 3:
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
        corr_txt = (f'Pearson r {r["pearson_r"]:+.3f}, '
                    f'Spearman rho {r["spearman_rho"]:+.3f}'
                    if 'pearson_r' in r else 'no correlation')
        ax.set_title(f'{band}   main epoch n = {len(main)}\nHuber slope {slope_txt}\n'
                     f'{corr_txt}', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if len(off):
            ax.legend(fontsize=6, loc='lower left', markerscale=3, framealpha=0.9)
    for ax in axes.ravel()[len(bands):]:
        ax.axis('off')
    fig.supxlabel('elevation [deg]', fontsize=9)
    fig.supylabel(f'look-up-table term only, uncorrected [{unit}]', fontsize=9)
    fig.suptitle(f'elevation dependence already carried by the look-up table: v1_lut alone, '
                 f'no Trim and no thermal correction — {variant}\n'
                 f'fit uses the main epoch; the offset nights ran a different look-up table '
                 f'at the same slope', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f'\nlook-up-table term alone against elevation (no Trim, no thermal '
              f'correction), main epoch only')
        print(f'  {"band":5s} {"n_main":>7s} {"slope":>9s} {"err":>6s} {"Pearson r":>10s} '
              f'{"Spearman rho":>13s}')
        for _, r in out.iterrows():
            print(f'  {r.band:5s} {int(r.n_main):7d} {r.slope:+9.2f} {r.slope_err:6.2f} '
                  f'{r.pearson_r:+10.3f} {r.spearman_rho:+13.3f}')
        print(f'  slope in {unit} per deg; the correlations are dimensionless')
        cols = [f'resid_nmad_order{o}' for o in orders if f'resid_nmad_order{o}' in out]
        if cols:
            print(f'  residual nMAD after removing an elevation polynomial [{unit}]:')
            print('  ' + f'{"band":5s} ' + ' '.join(f'order {o:>1d}'.rjust(9)
                                                    for o in orders))
            for _, r in out.iterrows():
                print('  ' + f'{r.band:5s} '
                      + ' '.join(f'{r[c]:9.1f}' for c in cols))
        n_off = int(out.n_early.sum())
        print(f'  the {len(LUT_EPOCH_EARLY_NIGHTS)} offset nights '
              f'({n_off} visits) are excluded from these fits; their look-up-table term sits '
              f'about 2400 {unit} lower at the same elevation and the same slope')
    return out


def slope_pages(pdf, slopes, ycol=YCOL, slope_lim=SLOPE_LIM, slope_bins=SLOPE_BINS):
    """The rising and falling slopes per night, and the distributions of both statistics.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    slopes : `pandas.DataFrame`
        Output of `per_night_direction_slopes`.
    ycol : `str`, optional
    slope_lim : `tuple` [`float`], optional
        Plotted slope window [µm of equivalent camera-hexapod dz per deg], shared by the
        per-night slope scatter and the all-points slope histogram. None autoscales.
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

    unit = 'um of equivalent camera-hexapod dz per deg'
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
        # Binned over the plotted window rather than over the data range, so the bin edges
        # are the round numbers the window names and one bad fit cannot stretch them.
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


def offset_pages(pdf, slopes, ycol=YCOL, ref_elev_deg=REF_ELEV_DEG, offset_lim=OFFSET_LIM,
                 slope_lim=SLOPE_LIM):
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
        Plotted offset window [µm of equivalent camera-hexapod dz]. None autoscales.
    slope_lim : `tuple` [`float`], optional
        Plotted slope window [µm of equivalent camera-hexapod dz per deg], applied to the
        offset-against-slope scatter so it matches `slope_pages`. None autoscales.

    Notes
    -----
    Two pages, the counterpart of `slope_pages` for the offset rather than the slope. The
    first is the time series against ``day_obs`` with the within-night residual nMAD as the
    error bar, which answers whether the surviving residual is a per-night constant that
    moves from night to night. The second is the distribution of the offset together with the
    offset against the slope, since a correlation between the two would mean the split
    between them is not identified by the elevation range each night covers.

    `offset_lim` and `slope_lim` change only what is drawn; every night enters the median, the
    nMAD, the trend and the correlations, and each panel says how many nights fall outside the
    window it plots.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    unit = 'um of equivalent camera-hexapod dz'
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

    # Same series against the night index, with a Theil-Sen trend, to separate a drift
    # across the run from night-to-night scatter about a constant.
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


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--day-obs', nargs='+', type=int, default=DAY_OBS_SERIES,
                   help='nights drawn as a visit-by-visit time series, one page each '
                        f'(default {" ".join(str(d) for d in DAY_OBS_SERIES)})')
    p.add_argument('--variant', default=None,
                   help='optical-state variant id; default the first in the table')
    p.add_argument('--in-dir', default=None,
                   help='directory holding the two parquet files; '
                        'default aos/output/science_lut')
    p.add_argument('--out', default=None,
                   help='output PDF path; default <in-dir>/visit_elevation.pdf')
    p.add_argument('--ycol', default=YCOL,
                   help='corrected-residual column read from the --chain-tag parquet and '
                        'used for the per-night elevation fits and the corrected series '
                        f'panels (default {YCOL}). Use v1_dzequiv_mlcorr with a '
                        'ml_elev chain tag, whose band-independent thermal stage replaces '
                        'the per-band truss and M1M3-z-gradient stages')
    p.add_argument('--bands', nargs='+', default=None,
                   help='bands to keep; default every band present')
    p.add_argument('--v1-per-um-dz', type=float, default=None,
                   help='dimensionless v-mode-1 amplitude per um of hexapod dz; default '
                        'derived from the StateEstimator, which needs lsst.ts.ofc')
    p.add_argument('--panel-ylim', nargs=2, type=float, default=None,
                   metavar=('LOW', 'HIGH'),
                   help='absolute shared y-limits for the per-night panels '
                        '[um of equivalent camera-hexapod dz]; overrides --panel-yspan')
    p.add_argument('--panel-yspan', type=float, default=PANEL_YSPAN,
                   help='per-night panel height [um of equivalent camera-hexapod dz], '
                        f'centred on each night median (default {PANEL_YSPAN:.0f}); '
                        'pass 0 to autoscale every panel independently')
    p.add_argument('--max-night-panels', type=int, default=None,
                   help='limit the number of per-night panels drawn')
    p.add_argument('--slope-lim', nargs=2, type=float, default=None,
                   metavar=('LOW', 'HIGH'),
                   help='plotted window for the per-night slope [um of equivalent '
                        'camera-hexapod dz per deg], shared by the slope scatter and the '
                        'slope histogram; the default follows --response ('
                        + ', '.join(f'{r} {lo:.0f} {hi:.0f}' for r, (lo, hi)
                                    in sorted(SLOPE_LIM_BY_RESPONSE.items()))
                        + '). The fits and the reported statistics always use every night')
    p.add_argument('--slope-bins', type=int, default=SLOPE_BINS,
                   help=f'bins in the all-points slope histogram (default {SLOPE_BINS}), '
                        'spanning --slope-lim')
    p.add_argument('--ref-elev-deg', type=float, default=REF_ELEV_DEG,
                   help=f'elevation [deg] at which each night\'s fit is evaluated to give '
                        f'that night\'s offset (default {REF_ELEV_DEG:.0f}, inside the bulk '
                        f'of the data; the 0 deg intercept is an extrapolation and mixes in '
                        f'the slope error)')
    p.add_argument('--seq-num', nargs=2, type=int, default=None,
                   metavar=('FIRST', 'LAST'),
                   help='restrict the time-series pages to this inclusive seq_num window, '
                        'to see the start of a night at readable resolution; the per-night '
                        'fits always use every visit')
    p.add_argument('--series-uncorrected', action='store_true',
                   help='draw only the uncorrected focus page per night, instead of the '
                        'corrected page followed by the uncorrected one; the driving-'
                        'temperature page is written either way')
    p.add_argument('--series-only', action='store_true',
                   help='write only the time-series pages, skipping the per-night panels '
                        'and the slope distributions, and leave the slope parquet alone')
    p.add_argument('--response', default='lut-trim-meas', choices=sorted(RESPONSE_LABELS),
                   help='which v-mode-1 combination the uncorrected panels show. '
                        'lut-trim-meas (default) is the full focus error; trim-meas leaves '
                        'the hexapod look-up-table baseline out. This must match the response '
                        'the --chain-tag parquet was built with, since the corrected residual '
                        'comes from that file')
    p.add_argument('--chain-tag', default=CHAIN_TAG,
                   help='names the chain parquet holding the corrected residual, as '
                        f'science_lut_results_<tag>.parquet (default {CHAIN_TAG}). Use '
                        'truss_grad_elev_trim_meas_nolutepoch for the look-up-table-free '
                        'response with the offset-look-up-table nights dropped. The join is '
                        'inner, so nights absent from the chain are dropped here too')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)
    verbose = not a.quiet

    root = pathlib.Path(__file__).resolve().parents[3]
    in_dir = (pathlib.Path(a.in_dir) if a.in_dir
              else root / 'aos' / 'output' / 'science_lut')

    if a.v1_per_um_dz is not None:
        v1_per_um_dz = a.v1_per_um_dz
        if verbose:
            print(f'v1 per um of hexapod dz = {v1_per_um_dz:.6e} per um (given on the '
                  f'command line)')
    else:
        from run_science_lut_report import v1_per_um_dz_value
        v1_per_um_dz = v1_per_um_dz_value(verbose=verbose)

    # The slope window follows the response unless it was given explicitly: excluding the
    # look-up-table term moves the per-night slope distribution from about -17 to zero.
    if a.slope_lim is None:
        a.slope_lim = list(SLOPE_LIM_BY_RESPONSE[a.response])
        if verbose:
            print(f'slope window {a.slope_lim[0]:+.0f} to {a.slope_lim[1]:+.0f} um of '
                  f'equivalent camera-hexapod dz per deg (default for --response '
                  f'{a.response})')

    df, variant = load_visits(in_dir, variant=a.variant, bands=a.bands,
                              v1_per_um_dz=v1_per_um_dz, chain_tag=a.chain_tag,
                              response=a.response, ycol=a.ycol, verbose=verbose)

    slopes = None
    if not a.series_only:
        slopes = per_night_direction_slopes(df, ycol=a.ycol,
                                            ref_elev_deg=a.ref_elev_deg, verbose=verbose)
        if not len(slopes):
            print('no night had enough visits to fit')
            return 1

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    # A non-default chain tag names itself in the output, so the look-up-table-free document
    # sits beside the full-focus-error one instead of overwriting it.
    if a.chain_tag == CHAIN_TAG:
        tag_suffix = ''
    elif a.chain_tag.startswith(CHAIN_TAG):
        tag_suffix = '_' + a.chain_tag[len(CHAIN_TAG):].lstrip('_')
    else:
        tag_suffix = '_' + a.chain_tag
    out_path = (pathlib.Path(a.out) if a.out
                else in_dir / f'visit_elevation{tag_suffix}.pdf')
    with PdfPages(out_path) as pdf:
        for day in a.day_obs:
            d = df[df.day_obs == day]
            if a.seq_num:
                d = d[d.seq_num.between(*a.seq_num)]
            if not len(d):
                print(f'day_obs {day}: no visits for variant {variant}'
                      + (f' in seq_num {a.seq_num[0]} to {a.seq_num[1]}'
                         if a.seq_num else '') + ', skipped')
                continue
            if verbose:
                print(f'\nday_obs {day}: {len(d)} visits, seq_num '
                      f'{int(d.seq_num.min())} to {int(d.seq_num.max())}, elevation '
                      f'{d.altitude_deg.min():.1f} to {d.altitude_deg.max():.1f} deg')
            series_lag(d, verbose=verbose)
            # Three pages per night, in this order: the corrected focus terms, the same
            # terms uncorrected for comparison, then the two temperatures the correction
            # is fitted against.
            night_series_page(pdf, d, day, variant, response=a.response,
                              corrected=not a.series_uncorrected)
            if not a.series_uncorrected:
                night_series_page(pdf, d, day, variant, response=a.response,
                                  corrected=False)
            night_temperature_page(pdf, d, day, variant, verbose=verbose)
        if not a.series_only:
            meas = measured_hist_page(pdf, df, variant, verbose=verbose)
            dmeas = measured_vs_delta_elev_page(pdf, df, variant, verbose=verbose)
            lut = lut_vs_elevation_page(pdf, df, variant, verbose=verbose)
            night_panel_pages(pdf, df, slopes, ycol=a.ycol,
                              max_nights=a.max_night_panels,
                              ylim=tuple(a.panel_ylim) if a.panel_ylim else None,
                              yspan=a.panel_yspan or None)
            slope_pages(pdf, slopes, ycol=a.ycol,
                        slope_lim=tuple(a.slope_lim) if a.slope_lim else None,
                        slope_bins=a.slope_bins)
            offset_pages(pdf, slopes, ycol=a.ycol, ref_elev_deg=a.ref_elev_deg,
                         slope_lim=tuple(a.slope_lim) if a.slope_lim else None)

    if not a.series_only:
        sl_path = in_dir / f'visit_elevation_slopes{tag_suffix}.parquet'
        slopes.to_parquet(sl_path)
        print(f'\nwrote {len(slopes)} night rows -> {sl_path}')
        # The two tables have different n per band -- the scatter keeps only visits with a
        # contiguous predecessor -- so n is suffixed rather than joined on.
        mb_path = in_dir / f'visit_elevation_measured_by_band{tag_suffix}.parquet'
        meas.merge(dmeas, on='band', suffixes=('', '_vs_delta_elev')).to_parquet(mb_path)
        print(f'wrote {len(meas)} band rows -> {mb_path}')
        lut_path = in_dir / f'visit_elevation_lut_by_band{tag_suffix}.parquet'
        lut.to_parquet(lut_path)
        print(f'wrote {len(lut)} band rows -> {lut_path}')
    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
