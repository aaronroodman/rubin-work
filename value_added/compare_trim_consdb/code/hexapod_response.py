"""Does the hexapod move when the AOS Trim changes during an exposure?

For science exposures whose ``MTAOS.logevent_degreeOfFreedom`` Trim changed between
``obs_start`` and ``obs_end``, plot one hexapod DOF per page:

* top panel — the Trim: ``MTAOS.logevent_degreeOfFreedom.aggregatedDoF<i>`` and the
  hexapod's own record of it, ``MTHexapod.logevent_uncompensatedPosition.<axis>``;
* bottom panel — the hexapod: measured ``MTHexapod.application.position<k>``, commanded
  ``MTHexapod.application.demand<k>`` and ``MTHexapod.logevent_compensatedPosition.<axis>``
  (Trim + LUT), with ``MTHexapod.logevent_inPosition`` marked.

A summary page gives, per page, the delay from the Trim event to the first change in
``application.demand`` and to the measured position settling.

``--census`` instead covers every ``differ`` exposure of ``comparison.parquet``, one EFD
query per night and topic: the delay from ``obs_end`` to the first
``MTHexapod.logevent_uncompensatedPosition`` after the in-exposure Trim event, and whether
the measured ``application.position`` (1 s means, server-side) moved during the exposure.

Run on s3df after ``compare.py``::

    python value_added/compare_trim_consdb/code/hexapod_response.py
    python value_added/compare_trim_consdb/code/hexapod_response.py --census
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.time import Time  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.dof_telemetry import _run_coro  # noqa: E402
from rubinwork.common.runs import data_root  # noqa: E402

# The four science mismatches of the 20261019_report time histories.
DEFAULT_EXPOSURES = [2026022700235, 2025092100134, 2026042100703, 2026071100146]
N_DOF_PER_EXPOSURE = 3
# Smallest Trim step worth a page: below this the hexapod response is lost in noise.
MIN_STEP = {'µm': 1.0, 'deg': 5e-5}
PAD = pd.Timedelta(minutes=2)
# DOF index -> (hexapod, salIndex, axis name, application array index in X,Y,Z,U,V,W)
HEX_DOF = {0: ('M2', 2, 'z', 2), 1: ('M2', 2, 'x', 0), 2: ('M2', 2, 'y', 1),
           3: ('M2', 2, 'u', 3), 4: ('M2', 2, 'v', 4),
           5: ('camera', 1, 'z', 2), 6: ('camera', 1, 'x', 0), 7: ('camera', 1, 'y', 1),
           8: ('camera', 1, 'u', 3), 9: ('camera', 1, 'v', 4)}
DOF_NAME = {0: 'M2 hex dz', 1: 'M2 hex dx', 2: 'M2 hex dy', 3: 'M2 hex rx', 4: 'M2 hex ry',
            5: 'Cam hex dz', 6: 'Cam hex dx', 7: 'Cam hex dy', 8: 'Cam hex rx',
            9: 'Cam hex ry'}


def unit(i):
    return 'deg' if HEX_DOF[i][2] in 'uv' else 'µm'


def study_dir():
    return data_root() / 'studies' / 'value_added' / 'compare_trim_consdb'


def efd_query(efd, topic, fields, t0, t1, index=None):
    """Raw-index (UTC) query; see aos/docs/telemetry.md section 7."""
    kw = {} if index is None else {'index': index}
    df = _run_coro(efd.select_time_series(
        topic, fields, Time(t0.to_pydatetime(), scale='utc'),
        Time(t1.to_pydatetime(), scale='utc'), convert_influx_index=False, **kw))
    if df is None or not len(df):
        return pd.DataFrame(columns=fields)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def first_change(ts, after, before_value, after_value, frac=0.5):
    """First time after `after` where `ts` has moved `frac` of the way to `after_value`."""
    s = ts[ts.index >= after]
    if not len(s) or not np.isfinite(after_value - before_value):
        return None
    step = after_value - before_value
    if step == 0:
        return None
    moved = (s - before_value) / step >= frac
    return s.index[moved.to_numpy()][0] if moved.any() else None


def page(pdf, row, i, dof_ev, t_event, unc, comp, app, inpos, delays):
    hexname, _, ax_name, k = HEX_DOF[i]
    t0, t1 = row['start_utc'], row['end_utc']
    lo, hi = t0 - PAD, t1 + PAD
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for a in (a1, a2):
        a.axvspan(t0, t1, color='#edae49', alpha=0.3, label='exposure (obs_start to obs_end)')
        a.axvline(t_event, color='#d1495b', lw=1, ls='--',
                  label='degreeOfFreedom event during exposure')
    d = dof_ev[(dof_ev['time_utc'] >= lo - pd.Timedelta(hours=1)) & (dof_ev['time_utc'] <= hi)]
    a1.step(d['time_utc'], d[f'dof{i}'], where='post', color='0.2', lw=1.2)
    a1.plot(d['time_utc'], d[f'dof{i}'], 'o', ms=4, color='0.2',
            label=f'lsst.sal.MTAOS.logevent_degreeOfFreedom.aggregatedDoF{i}')
    if len(unc):
        a1.plot(unc.index, unc[ax_name], 'x', ms=7, color='#00798c',
                label=f'lsst.sal.MTHexapod.logevent_uncompensatedPosition.{ax_name} '
                      f'({hexname})')
    vis = d[(d['time_utc'] >= lo)][f'dof{i}'].tolist() + d[d['time_utc'] < lo][f'dof{i}'].tail(1).tolist()
    if vis:
        span = max(vis) - min(vis) or abs(max(vis)) * 0.01 or 1.0
        a1.set_ylim(min(vis) - 0.15 * span, max(vis) + 0.15 * span)
    a1.set_ylabel(f'Trim, dof{i} ({DOF_NAME[i]}) [{unit(i)}]')
    a1.legend(fontsize=7, loc='best')

    if len(app):
        a2.plot(app.index, app[f'position{k}'], '-', lw=1.2, color='#4c9f70',
                label=f'lsst.sal.MTHexapod.application.position{k} (measured, {hexname})')
        a2.plot(app.index, app[f'demand{k}'], '-', lw=0.8, color='0.4',
                label=f'lsst.sal.MTHexapod.application.demand{k} (commanded)')
    if len(comp):
        a2.plot(comp.index, comp[ax_name], 'D', ms=5, color='#d1495b', mfc='none',
                label=f'lsst.sal.MTHexapod.logevent_compensatedPosition.{ax_name}')
    if len(inpos):
        # Only state changes; the event repeats with an unchanged value many times.
        v = inpos['inPosition'].astype(bool)
        change = v[v.ne(v.shift())]
        for t, val in change.items():
            a2.axvline(t, color='#00798c' if val else '#bbbbbb', lw=0.8, ls=':')
        a2.plot([], [], color='#00798c', ls=':',
                label='lsst.sal.MTHexapod.logevent_inPosition -> True')
        a2.plot([], [], color='#bbbbbb', ls=':',
                label='lsst.sal.MTHexapod.logevent_inPosition -> False')
    a2.set_ylabel(f'hexapod position, {hexname} {ax_name} [{unit(i)}]\n(LUT + Trim)')
    a2.legend(fontsize=7, loc='best')
    a2.set_xlim(lo, hi)
    a2.set_xlabel('time [UTC]')

    def fmt(x):
        return 'none' if x is None else f'{x:+.1f} s'
    fig.suptitle(
        f'exposure {row["exposure_id"]} ({row.get("img_type", "science")}, exp_time = {row["exp_time"]:.1f} s), '
        f'dof{i} {DOF_NAME[i]}\nTrim event at obs_start {fmt(delays["event_s"])}; '
        f'demand moves {fmt(delays["demand_s"])} and position moves '
        f'{fmt(delays["position_s"])} after the event', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fetch_hex(efd, sal, lo, hi):
    """Full-rate hexapod topics for one hexapod over [lo, hi], for the example pages."""
    return dict(
        unc=efd_query(efd, 'lsst.sal.MTHexapod.logevent_uncompensatedPosition',
                      list('xyzuv'), lo, hi, index=sal),
        comp=efd_query(efd, 'lsst.sal.MTHexapod.logevent_compensatedPosition',
                       list('xyzuv'), lo, hi, index=sal),
        app=efd_query(efd, 'lsst.sal.MTHexapod.application',
                      [f'position{k}' for k in range(5)]
                      + [f'demand{k}' for k in range(5)], lo, hi, index=sal),
        inpos=efd_query(efd, 'lsst.sal.MTHexapod.logevent_inPosition',
                        ['inPosition'], lo, hi, index=sal))


def dof_delays(h, i, t0, t_event, step):
    """Delays [s] for DOF ``i``: event after obs_start, demand/position/uncomp after event."""
    k = HEX_DOF[i][3]
    app = h['app']
    pre = app[app.index <= t_event]
    delays = {'event_s': (t_event - t0).total_seconds(), 'demand_s': None, 'position_s': None}
    if len(pre):
        for key, col in (('demand_s', f'demand{k}'), ('position_s', f'position{k}')):
            tc = first_change(app[col], t_event, pre[col].iloc[-1], pre[col].iloc[-1] + step)
            delays[key] = None if tc is None else (tc - t_event).total_seconds()
    unc_after = h['unc'][h['unc'].index > t_event]
    delays['uncomp_event_s'] = (None if not len(unc_after) else
                                (unc_after.index[0] - t_event).total_seconds())
    return delays


def examples(args):
    base = study_dir()
    inp = sorted(base.glob('*_inputs'))[-1]
    run = study_run('value_added/compare_trim_consdb', 'hexapod')
    exp = pd.read_parquet(inp / 'consdb_exposure.parquet')
    exp = exp[exp['exposure_id'].isin(args.exposures)].copy()
    exp['start_utc'] = pd.to_datetime(Time(list(exp['obs_start']), scale='tai').utc.isot,
                                      utc=True)
    exp['end_utc'] = pd.to_datetime(Time(list(exp['obs_end']), scale='tai').utc.isot,
                                    utc=True)
    ev = pd.read_parquet(inp / 'efd_dof_events.parquet').sort_values('time_utc')
    steps = np.diff(ev[[f'dof{i}' for i in range(10)]].to_numpy(float), axis=0)
    step_scale = 1.4826 * np.nanmedian(np.abs(steps - np.nanmedian(steps, axis=0)), axis=0)

    from lsst_efd_client import EfdClient
    efd = EfdClient('usdf_efd')

    rows = []
    pages = []
    for _, row in exp.sort_values('start_utc').iterrows():
        t0, t1 = row['start_utc'], row['end_utc']
        during = ev[(ev['time_utc'] > t0) & (ev['time_utc'] <= t1)]
        before = ev[ev['time_utc'] <= t0].iloc[-1]
        if not len(during):
            print(f'{row["exposure_id"]}: no event during the exposure')
            continue
        after = during.iloc[-1]
        t_event = after['time_utc']
        delta = np.array([after[f'dof{i}'] - before[f'dof{i}'] for i in range(10)])
        rank = np.argsort(-np.abs(delta) / np.where(step_scale > 0, step_scale, 1))
        chosen = [int(i) for i in rank
                  if abs(delta[i]) >= MIN_STEP[unit(i)]][:N_DOF_PER_EXPOSURE]
        lo, hi = t0 - PAD, t1 + PAD
        per_hex = {sal: fetch_hex(efd, sal, lo, hi) for sal in {HEX_DOF[i][1] for i in chosen}}
        for i in chosen:
            h = per_hex[HEX_DOF[i][1]]
            app = h['app']
            delays = dof_delays(h, i, t0, t_event, delta[i])
            pages.append((row, i, t_event, h, delays))
            rows.append({'exposure_id': row['exposure_id'], 'dof': f'dof{i}',
                         'name': DOF_NAME[i], 'trim_step': delta[i], 'unit': unit(i),
                         'event_after_obs_start_s': delays['event_s'],
                         'obs_end_after_obs_start_s': (t1 - t0).total_seconds(),
                         'uncompensatedPosition_after_event_s': delays['uncomp_event_s'],
                         'hexapod_commanded_after_obs_end_s': (
                             None if delays['uncomp_event_s'] is None else
                             delays['event_s'] + delays['uncomp_event_s']
                             - (t1 - t0).total_seconds()),
                         'demand_moves_after_event_s': delays['demand_s'],
                         'position_moves_after_event_s': delays['position_s'],
                         'n_application_rows': len(app)})
    summary = pd.DataFrame(rows)
    summary.to_csv(run.path / 'hexapod_response.csv', index=False)

    pdf_path = run.path / 'hexapod_response.pdf'
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.96, 'Hexapod response to a Trim change during the exposure',
                 fontsize=14, weight='bold', va='top')
        notes = (
            'Times are seconds; "moves" = first sample 50% of the way from the pre-event\n'
            'value to pre-event value + Trim step. Trim step in the DOF unit (µm or deg).\n'
            'Measured position: lsst.sal.MTHexapod.application.position<k>, order X,Y,Z,U,V,W.\n'
            'Commanded: lsst.sal.MTHexapod.application.demand<k>. salIndex 1 camera, 2 M2.\n'
            'uncomp-end: first MTHexapod.logevent_uncompensatedPosition after the Trim event,\n'
            'relative to obs_end; positive means the hexapod was commanded after the shutter\n'
            'closed.\n\n')
        cols = ['exposure_id', 'dof', 'trim_step', 'unit', 'event_after_obs_start_s',
                'obs_end_after_obs_start_s', 'uncompensatedPosition_after_event_s',
                'hexapod_commanded_after_obs_end_s',
                'demand_moves_after_event_s', 'position_moves_after_event_s']
        txt = summary[cols].rename(columns={
            'event_after_obs_start_s': 'event-start', 'obs_end_after_obs_start_s': 'end-start',
            'uncompensatedPosition_after_event_s': 'uncomp-event',
            'hexapod_commanded_after_obs_end_s': 'uncomp-end',
            'demand_moves_after_event_s': 'demand-event',
            'position_moves_after_event_s': 'position-event'}).to_string(
                index=False, float_format=lambda x: f'{x:.4g}')
        fig.text(0.04, 0.90, notes + txt, family='monospace', fontsize=7, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        for row, i, t_event, h, delays in pages:
            page(pdf, row, i, ev, t_event, h['unc'], h['comp'], h['app'], h['inpos'],
                 delays)
    print(summary.to_string(index=False))
    print(f'wrote {pdf_path}')


# ---- census of every 'differ' exposure -----------------------------------------------

AXES = 'xyzuv'
# DOF indices per salIndex, in AXES order.
SAL_DOFS = {1: [6, 7, 5, 8, 9], 2: [1, 2, 0, 3, 4]}
SAL_NAME = {1: 'camera', 2: 'M2'}
CENSUS_PAD = pd.Timedelta(minutes=1)
# Longest wait for the first uncompensatedPosition after the in-exposure event.
UNC_SEARCH = pd.Timedelta(minutes=10)
BIN = pd.Timedelta(seconds=1)
BIN_CHUNK = pd.Timedelta(hours=5)
# A move is |last - first 1 s mean| over N_SIGMA times the census nMAD, and >= MIN_STEP.
N_SIGMA = 5.0
N_EXAMPLES = 6


def axis_unit(ax):
    return 'deg' if ax in 'uv' else 'µm'


def iso(t):
    return t.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def efd_binned(efd, topic, fields, t0, t1, index):
    """Server-side 1 s means of `fields`, raw (UTC) influx index, bins labelled by start.

    Queried in chunks: the EFD refuses more than 20000 buckets per query.
    """
    sel = ', '.join(f'mean("{f}") AS "{f}"' for f in fields)
    parts = []
    for c0 in pd.date_range(t0.floor('s'), t1, freq=BIN_CHUNK):
        c1 = min(c0 + BIN_CHUNK - BIN, t1)
        q = (f'SELECT {sel} FROM "{efd._db_name}"."autogen"."{topic}" '
             f"WHERE time >= '{iso(c0)}' AND time <= '{iso(c1)}' AND \"salIndex\" = {index} "
             f'GROUP BY time({int(BIN.total_seconds())}s) fill(none)')
        df = _run_coro(efd._do_query(q, convert_influx_index=False))
        if df is not None and isinstance(df, pd.DataFrame) and len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=fields)
    df = pd.concat(parts)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[~df.index.duplicated()].sort_index()


def census_night(efd, night, ev):
    """One row per (exposure, hexapod) for the 'differ' exposures of one night."""
    lo = night['start_utc'].min() - CENSUS_PAD
    hi = night['end_utc'].max() + UNC_SEARCH
    unc, app = {}, {}
    for sal in SAL_DOFS:
        unc[sal] = efd_query(efd, 'lsst.sal.MTHexapod.logevent_uncompensatedPosition',
                             list(AXES), lo, hi, index=sal)
        app[sal] = efd_binned(efd, 'lsst.sal.MTHexapod.application',
                              [f'position{k}' for k in range(5)], lo,
                              night['end_utc'].max() + CENSUS_PAD, sal)
    rows = []
    for _, row in night.iterrows():
        t0, t1 = row['start_utc'], row['end_utc']
        during = ev[(ev['time_utc'] > t0) & (ev['time_utc'] <= t1)]
        prior = ev[ev['time_utc'] <= t0]
        if not len(during) or not len(prior):
            continue
        before, after = prior.iloc[-1], during.iloc[-1]
        t_event = after['time_utc']
        for sal, dofs in SAL_DOFS.items():
            r = {'exposure_id': row['exposure_id'], 'day_obs': row['day_obs'],
                 'img_type': row['img_type'], 'exp_time': row['exp_time'],
                 'sal_index': sal, 'hexapod': SAL_NAME[sal],
                 'obs_end_after_obs_start_s': (t1 - t0).total_seconds(),
                 'event_after_obs_start_s': (t_event - t0).total_seconds(),
                 'n_events_during': len(during)}
            new = np.array([after[f'dof{i}'] for i in dofs], float)
            step = new - np.array([before[f'dof{i}'] for i in dofs], float)
            for ax, st in zip(AXES, step):
                r[f'step_{ax}'] = st
            u = unc[sal][(unc[sal].index > t_event) & (unc[sal].index <= t_event + UNC_SEARCH)]
            if len(u):
                tu = u.index[0]
                r['unc_after_obs_end_s'] = (tu - t1).total_seconds()
                r['unc_after_event_s'] = (tu - t_event).total_seconds()
                diff = np.abs(u.iloc[0][list(AXES)].to_numpy(float) - new)
                r['unc_minus_trim_max_um'] = diff[:3].max()
                r['unc_minus_trim_max_deg'] = diff[3:].max()
            a = app[sal]
            a = a[(a.index >= t0) & (a.index + BIN <= t1)]
            r['app_n_bins'] = len(a)
            for k, ax in enumerate(AXES):
                col = a[f'position{k}'].to_numpy(float) if len(a) else np.array([])
                ok = col[np.isfinite(col)]
                r[f'app_d_{ax}'] = ok[-1] - ok[0] if len(ok) >= 2 else np.nan
                r[f'app_ptp_{ax}'] = np.ptp(ok) if len(ok) >= 2 else np.nan
            rows.append(r)
    return rows


def classify(cen):
    """Add per-axis noise thresholds and the moved / commanded flags."""
    sigma = {}
    for sal in SAL_DOFS:
        m = cen['sal_index'] == sal
        for ax in AXES:
            d = cen.loc[m, f'app_d_{ax}'].dropna()
            sigma[(sal, ax)] = 1.4826 * np.median(np.abs(d - np.median(d))) if len(d) else np.nan
    moved = np.zeros(len(cen), bool)
    toward = np.full(len(cen), np.nan)
    for ax in AXES:
        thr = np.array([max(N_SIGMA * sigma[(s, ax)], MIN_STEP[axis_unit(ax)])
                        for s in cen['sal_index']])
        cen[f'thr_{ax}'] = thr
        cen[f'moved_{ax}'] = np.abs(cen[f'app_d_{ax}']) > thr
        moved |= cen[f'moved_{ax}'].to_numpy()
    # Fraction of the Trim step reached during the exposure, on the axis with the
    # largest step relative to MIN_STEP.
    steps = cen[[f'step_{ax}' for ax in AXES]].to_numpy(float)
    norm = steps / np.array([MIN_STEP[axis_unit(ax)] for ax in AXES])
    big = np.nanargmax(np.abs(np.nan_to_num(norm)), axis=1)
    idx = np.arange(len(cen))
    has_step = np.abs(norm[idx, big]) >= 1
    dvals = cen[[f'app_d_{ax}' for ax in AXES]].to_numpy(float)
    toward[has_step] = dvals[idx, big][has_step] / steps[idx, big][has_step]
    cen['trim_commanded'] = has_step
    cen['big_axis'] = [AXES[b] for b in big]
    cen['moved'] = moved
    cen['frac_of_step_in_exposure'] = toward
    cen['moved_to_trim'] = toward >= 0.5
    return cen, sigma


def census_pages(pdf, cen, ex, sigma, n_nights):
    cmd = cen[cen['trim_commanded']]
    # Per exposure: the first hexapod command after the event, over hexapods with a step.
    first = cmd.groupby('exposure_id').agg(
        delay=('unc_after_obs_end_s', 'min'), day_obs=('day_obs', 'first'),
        img_type=('img_type', 'first'))
    first['month'] = (first['day_obs'] // 100).astype(int).astype(str)
    q = first.groupby('month')['delay'].describe(percentiles=[.05, .5, .95])
    q['n_before_obs_end'] = first[first['delay'] < 0].groupby('month').size()
    q = q.fillna({'n_before_obs_end': 0})
    mv = cen.groupby('exposure_id').agg(moved=('moved', 'any'),
                                        to_trim=('moved_to_trim', 'any'),
                                        day_obs=('day_obs', 'first'))
    mv['month'] = (mv['day_obs'] // 100).astype(int).astype(str)
    mtab = mv.groupby('month').agg(n=('moved', 'size'), moved=('moved', 'sum'),
                                   moved_to_trim=('to_trim', 'sum'))

    n_exp = ex['exposure_id'].nunique()
    n_row = cen['exposure_id'].nunique()
    no_unc = cmd['unc_after_obs_end_s'].isna().sum()
    match = cmd.dropna(subset=['unc_after_obs_end_s'])
    n_match = ((match['unc_minus_trim_max_um'] < 0.1)
               & (match['unc_minus_trim_max_deg'] < 1e-5)).sum()
    d = first['delay'].dropna()
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.96, 'Census: hexapod response to every in-exposure Trim change',
             fontsize=14, weight='bold', va='top')
    sig_txt = '\n'.join(
        f'  {SAL_NAME[s]:6s} ' + '  '.join(
            f'{ax}: {sigma[(s, ax)]:.3g} {axis_unit(ax)}' for ax in AXES) for s in SAL_DOFS)
    txt = (
        f"'differ' exposures (status == 'differ' in comparison.parquet): {n_exp} (count), "
        f'{n_nights} nights (count)\n'
        f'with a degreeOfFreedom event in (obs_start, obs_end] and an earlier event: '
        f'{n_row} (count)\n\n'
        'Commanded delay = first MTHexapod.logevent_uncompensatedPosition after the last\n'
        'in-exposure Trim event, minus obs_end [s]; per exposure, the earliest over the\n'
        f'hexapods whose Trim step is >= 1 µm or 5e-5 deg on some axis. That is '
        f'{len(cmd)} (count) exposure-hexapod pairs\n'
        f'over {len(first)} exposures (count). No uncompensatedPosition within '
        f'{UNC_SEARCH.total_seconds():.0f} s: {no_unc} pairs (count).\n'
        f'uncompensatedPosition equals the in-exposure Trim (< 0.1 µm, < 1e-5 deg): '
        f'{n_match} of {len(match)} pairs (count).\n\n'
        f'Commanded delay over exposures [s]: min {d.min():.2f}, 5% {d.quantile(.05):.2f}, '
        f'median {d.median():.2f}, 95% {d.quantile(.95):.2f}, max {d.max():.2f}\n'
        f'Commanded before obs_end (delay < 0 s): {(d < 0).sum()} of {len(d)} exposures '
        f'(count).\n\n'
        'Moved during the exposure: measured MTHexapod.application.position, 1 s means;\n'
        'd = last minus first full bin inside [obs_start, obs_end]. An axis moved if\n'
        f'|d| > max({N_SIGMA:.0f} x nMAD of d over the census, 1 µm or 5e-5 deg).\n'
        f'nMAD of d per axis:\n{sig_txt}\n'
        f'Exposures where either hexapod moved: {mv["moved"].sum()} of {len(mv)} (count).\n'
        f'... and moved >= 0.5 of the Trim step (dimensionless; d over Trim step, largest-'
        f'step axis): {mv["to_trim"].sum()} (count).\n')
    fig.text(0.04, 0.90, txt, family='monospace', fontsize=8, va='top')
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.96, 'Commanded delay after obs_end [s] by month (YYYYMM of day_obs)',
             fontsize=12, weight='bold', va='top')
    tab = q[['count', 'min', '5%', '50%', '95%', 'max', 'n_before_obs_end']].rename(
        columns={'count': 'n exposures', '5%': 'p5', '50%': 'median', '95%': 'p95',
                 'n_before_obs_end': 'n delay < 0'})
    mt = mtab.rename(columns={'n': 'n exposures', 'moved_to_trim': 'moved >= 0.5 step'})
    fig.text(0.04, 0.90, 'delay [s]; counts are exposures\n' + tab.to_string(
        float_format=lambda x: f'{x:.3g}') + '\n\nmoved during exposure (count)\n'
        + mt.to_string(), family='monospace', fontsize=8, va='top')
    pdf.savefig(fig)
    plt.close(fig)

    months = sorted(first['month'].unique())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5))
    data = [first.loc[first['month'] == m, 'delay'].dropna().to_numpy() for m in months]
    a1.boxplot(data, labels=months, whis=(5, 95), showfliers=True,
               flierprops={'ms': 2, 'alpha': 0.3})
    a1.axhline(0, color='#d1495b', lw=1, ls='--', label='obs_end')
    a1.set_yscale('symlog', linthresh=2)
    a1.set_ylabel('first uncompensatedPosition\nafter Trim event, minus obs_end [s]')
    a1.set_xlabel('month (YYYYMM of day_obs); box 25-75%, whiskers 5-95%')
    a1.tick_params(axis='x', rotation=45)
    a1.legend(fontsize=8)
    a2.hist(d.clip(-5, 20), bins=np.arange(-5, 20.1, 0.1), color='0.3')
    a2.axvline(0, color='#d1495b', lw=1, ls='--')
    a2.set_yscale('log')
    a2.set_xlabel('commanded delay after obs_end [s] (clipped to [-5, 20])')
    a2.set_ylabel('exposures (count)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5))
    x = np.arange(len(mtab))
    a1.bar(x, mtab['moved'], color='#edae49', label='either hexapod moved')
    a1.bar(x, mtab['moved_to_trim'], color='#d1495b', label='moved >= 0.5 of the Trim step')
    a1.set_xticks(x, mtab.index, rotation=45)
    a1.set_ylabel('exposures (count)')
    a1.set_xlabel('month (YYYYMM of day_obs)')
    a1.legend(fontsize=8)
    c = cen[cen['trim_commanded']]
    a2.scatter(c['event_after_obs_start_s'], c['frac_of_step_in_exposure'], s=3,
               alpha=0.3, color='0.2')
    a2.axhline(0.5, color='#d1495b', lw=1, ls='--')
    a2.set_ylim(-1, 2)
    a2.set_xlabel('Trim event after obs_start [s]')
    a2.set_ylabel('position change in exposure over Trim step\n'
                  '(dimensionless; largest-step axis)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def census(args):
    base = study_dir()
    inp = sorted(base.glob('*_inputs'))[-1]
    cmp_path = sorted(base.glob('*_compare'))[-1] / 'comparison.parquet'
    run = study_run('value_added/compare_trim_consdb',
                    'hexapod_census_test' if args.nights else 'hexapod_census')
    ex = pd.read_parquet(cmp_path)
    ex = ex[ex['status'] == 'differ'].copy()
    if args.nights:
        ex = ex[ex['day_obs'].isin(args.nights)]
    ex['start_utc'] = pd.to_datetime(Time(list(ex['obs_start']), scale='tai').utc.isot,
                                     utc=True)
    ex['end_utc'] = pd.to_datetime(Time(list(ex['obs_end']), scale='tai').utc.isot, utc=True)
    ev = pd.read_parquet(inp / 'efd_dof_events.parquet').sort_values('time_utc')

    from lsst_efd_client import EfdClient
    efd = EfdClient('usdf_efd')

    cache = run.path / 'nights'
    cache.mkdir(exist_ok=True)
    parts = []
    nights = sorted(ex['day_obs'].unique())
    for n, night in enumerate(nights):
        f = cache / f'{night}.parquet'
        if not f.exists():
            sel = ex[ex['day_obs'] == night].sort_values('start_utc')
            pd.DataFrame(census_night(efd, sel, ev)).to_parquet(f)
        parts.append(pd.read_parquet(f))
        print(f'{n + 1}/{len(nights)} {night}: {len(parts[-1])} rows', flush=True)
    cen, sigma = classify(pd.concat(parts, ignore_index=True))
    cen.to_parquet(run.path / 'hexapod_census.parquet')

    # Example pages: the moved exposures with the largest fraction of the Trim step,
    # then the largest moves without one.
    mv = cen[cen['moved']].copy()
    mv['rank'] = mv['frac_of_step_in_exposure'].fillna(-np.inf)
    mv = mv.sort_values('rank', ascending=False).drop_duplicates('exposure_id')
    picks = mv.head(N_EXAMPLES)
    pages = []
    for _, c in picks.iterrows():
        row = ex[ex['exposure_id'] == c['exposure_id']].iloc[0]
        t0, t1 = row['start_utc'], row['end_utc']
        during = ev[(ev['time_utc'] > t0) & (ev['time_utc'] <= t1)]
        before = ev[ev['time_utc'] <= t0].iloc[-1]
        t_event = during.iloc[-1]['time_utc']
        sal = int(c['sal_index'])
        ax = (c['big_axis'] if c['trim_commanded'] else
              max(AXES, key=lambda a: abs(c[f'app_d_{a}']) / c[f'thr_{a}']))
        i = SAL_DOFS[sal][AXES.index(ax)]
        step = during.iloc[-1][f'dof{i}'] - before[f'dof{i}']
        h = fetch_hex(efd, sal, t0 - PAD, t1 + PAD)
        pages.append((row, i, t_event, h, dof_delays(h, i, t0, t_event, step)))

    pdf_path = run.path / 'hexapod_census.pdf'
    with PdfPages(pdf_path) as pdf:
        census_pages(pdf, cen, ex, sigma, len(nights))
        for row, i, t_event, h, delays in pages:
            page(pdf, row, i, ev, t_event, h['unc'], h['comp'], h['app'], h['inpos'],
                 delays)
    print(f'wrote {pdf_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--exposures', type=int, nargs='+', default=DEFAULT_EXPOSURES)
    parser.add_argument('--census', action='store_true',
                        help="every 'differ' exposure, one query per night")
    parser.add_argument('--nights', type=int, nargs='+',
                        help='census: only these day_obs (for testing)')
    args = parser.parse_args()
    census(args) if args.census else examples(args)


if __name__ == '__main__':
    main()
