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

Run on s3df after ``compare.py``::

    python value_added/compare_trim_consdb/code/hexapod_response.py
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
        f'exposure {row["exposure_id"]} (science, exp_time = {row["exp_time"]:.1f} s), '
        f'dof{i} {DOF_NAME[i]}\nTrim event at obs_start {fmt(delays["event_s"])}; '
        f'demand moves {fmt(delays["demand_s"])} and position moves '
        f'{fmt(delays["position_s"])} after the event', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--exposures', type=int, nargs='+', default=DEFAULT_EXPOSURES)
    args = parser.parse_args()

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
        per_hex = {}
        for sal in {HEX_DOF[i][1] for i in chosen}:
            per_hex[sal] = dict(
                unc=efd_query(efd, 'lsst.sal.MTHexapod.logevent_uncompensatedPosition',
                              list('xyzuv'), lo, hi, index=sal),
                comp=efd_query(efd, 'lsst.sal.MTHexapod.logevent_compensatedPosition',
                               list('xyzuv'), lo, hi, index=sal),
                app=efd_query(efd, 'lsst.sal.MTHexapod.application',
                              [f'position{k}' for k in range(5)]
                              + [f'demand{k}' for k in range(5)], lo, hi, index=sal),
                inpos=efd_query(efd, 'lsst.sal.MTHexapod.logevent_inPosition',
                                ['inPosition'], lo, hi, index=sal))
        for i in chosen:
            hexname, sal, ax_name, k = HEX_DOF[i]
            h = per_hex[sal]
            app = h['app']
            pre = app[app.index <= t_event]
            delays = {'event_s': (t_event - t0).total_seconds(),
                      'demand_s': None, 'position_s': None}
            if len(pre):
                for key, col in (('demand_s', f'demand{k}'), ('position_s', f'position{k}')):
                    tc = first_change(app[col], t_event, pre[col].iloc[-1],
                                      pre[col].iloc[-1] + delta[i])
                    delays[key] = None if tc is None else (tc - t_event).total_seconds()
            unc_after = h['unc'][h['unc'].index > t_event]
            delays['uncomp_event_s'] = (None if not len(unc_after) else
                                        (unc_after.index[0] - t_event).total_seconds())
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


if __name__ == '__main__':
    main()
