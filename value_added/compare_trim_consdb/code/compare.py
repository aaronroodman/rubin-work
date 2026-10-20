"""Compare the value-added database Trim with the ConsDB transformed-EFD Trim.

Reads the newest ``*_inputs`` run of this study, classifies every exposure, explains each
mismatch by rebuilding both selection rules from the raw ``degreeOfFreedom`` events, and
writes summary tables, plots and EFD time histories for a few science mismatches.

The two rules:

* **value-added DB** — the last event at or before ``obs_start``.
* **ConsDB** — ``most_recent_value`` over ``[obs_start - 1 h, obs_end]``
  (``start_offset: -1`` in ``config_lsstcam.yaml``), so an event published during the
  exposure wins, and an exposure with no event in the window gets no value.

Run on s3df after ``fetch_inputs.py``::

    python value_added/compare_trim_consdb/code/compare.py
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.time import Time  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.runs import data_root  # noqa: E402

N_DOF = 50
DOFS = [f'dof{i}' for i in range(N_DOF)]
START_OFFSET = pd.Timedelta(hours=1)        # ConsDB start_offset: -1 [h]
RTOL, ATOL = 1e-6, 1e-9                    # agreement tolerance, DOF units
STATUS_ORDER = ['agree', 'differ', 'db_only', 'consdb_only', 'neither']
STATUS_COLOR = {'agree': '#4c9f70', 'differ': '#d1495b', 'db_only': '#edae49',
                'consdb_only': '#00798c', 'neither': '#bbbbbb'}
IMG_ORDER = ['science', 'acq', 'cwfs', 'focus', 'engtest', 'flat', 'dark', 'bias',
             'cbp', 'indome', 'test']
DOF_LABEL = {0: 'M2 hex dz', 1: 'M2 hex dx', 2: 'M2 hex dy', 3: 'M2 hex rx',
             4: 'M2 hex ry', 5: 'Cam hex dz', 6: 'Cam hex dx', 7: 'Cam hex dy',
             8: 'Cam hex rx', 9: 'Cam hex ry'}


def dof_name(i):
    if i in DOF_LABEL:
        return f'dof{i} ({DOF_LABEL[i]})'
    return f'dof{i} (M1M3 bend {i - 9})' if i < 30 else f'dof{i} (M2 bend {i - 29})'


def dof_unit(i):
    """Stored unit of DOF i: µm, except the four hexapod tilts (see study.md)."""
    return 'deg or arcsec' if i in (3, 4, 8, 9) else 'µm'


def latest_inputs():
    base = data_root() / 'studies' / 'value_added' / 'compare_trim_consdb'
    runs = sorted(base.glob('*_inputs'))
    if not runs:
        raise SystemExit(f'no *_inputs run under {base}; run fetch_inputs.py first')
    return runs[-1]


def tai_to_utc(col):
    t = Time(list(col.astype(str)), format='isot', scale='tai').utc
    return pd.to_datetime(t.isot, utc=True)


def rebuild(exposure, ev):
    """Both selection rules from the raw events: index of the chosen event, or -1."""
    et = ev['time_utc'].to_numpy(dtype='datetime64[ns]')
    t0 = exposure['start_utc'].to_numpy(dtype='datetime64[ns]')
    t1 = exposure['end_utc'].to_numpy(dtype='datetime64[ns]')
    tlo = (exposure['start_utc'] - START_OFFSET).to_numpy(dtype='datetime64[ns]')
    i_db = np.searchsorted(et, t0, side='right') - 1           # last event <= start
    i_end = np.searchsorted(et, t1, side='right') - 1          # last event <= end
    i_lo = np.searchsorted(et, tlo, side='left')               # first event >= start-1h
    i_cdb = np.where(i_end >= i_lo, i_end, -1)
    n_during = i_end - i_db                                     # events in (start, end]
    return i_db, i_cdb, n_during


def classify(df):
    db = df[[f'db_{d}' for d in DOFS]].to_numpy(float)
    cd = df[[f'cdb_{d}' for d in DOFS]].to_numpy(float)
    has_db = np.isfinite(db).all(axis=1)
    has_cd = np.isfinite(cd).any(axis=1)
    close = np.isclose(db, cd, rtol=RTOL, atol=ATOL)
    agree = close.all(axis=1)
    status = np.select(
        [has_db & has_cd & agree, has_db & has_cd, has_db, has_cd],
        ['agree', 'differ', 'db_only', 'consdb_only'], default='neither')
    absdiff = np.where(has_db[:, None] & has_cd[:, None], np.abs(db - cd), np.nan)
    df['status'] = status
    df['n_dof_differ'] = np.where(has_db & has_cd, (~close).sum(axis=1), 0)
    with np.errstate(invalid='ignore'):
        df['max_absdiff'] = np.nanmax(np.where(close, np.nan, absdiff), axis=1) \
            if absdiff.size else np.nan
    return df


def explain(df):
    """Cause of every non-agreeing exposure, from the rebuilt rules."""
    cause = np.full(len(df), '', dtype=object)
    s = df['status'].to_numpy()
    during = df['n_events_during'].to_numpy() > 0
    no_window = df['i_cdb'].to_numpy() < 0
    cdb_is_rule = df['cdb_matches_rule'].to_numpy()
    db_is_rule = df['db_matches_rule'].to_numpy()
    m = s == 'differ'
    cause[m & during & cdb_is_rule] = 'event during exposure'
    cause[m & ~(during & cdb_is_rule)] = 'unexplained'
    m = s == 'db_only'
    cause[m & no_window] = 'no event in [start-1h, end]'
    cause[m & ~no_window] = 'ConsDB row missing'
    m = s == 'consdb_only'
    cause[m & ~db_is_rule] = 'DB row missing'
    cause[m & db_is_rule] = 'unexplained'
    df['cause'] = cause
    return df


def rule_match(values, ev_vals, idx):
    out = np.zeros(len(idx), bool)
    ok = idx >= 0
    out[ok] = np.isclose(values[ok], ev_vals[idx[ok]], rtol=RTOL, atol=ATOL).all(axis=1)
    return out


def bar_by_img_type(df, path):
    tab = pd.crosstab(df['img_type'], df['status']).reindex(columns=STATUS_ORDER,
                                                             fill_value=0)
    tab = tab.reindex([t for t in IMG_ORDER if t in tab.index]
                      + [t for t in tab.index if t not in IMG_ORDER])
    frac = tab.div(tab.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    left = np.zeros(len(frac))
    for s in STATUS_ORDER:
        ax.barh(frac.index, frac[s], left=left, color=STATUS_COLOR[s], label=s)
        left += frac[s].to_numpy()
    for k, t in enumerate(frac.index):
        ax.text(1.01, k, f'n = {tab.loc[t].sum():,}', va='center', fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('fraction of exposures (dimensionless; status count over img_type count)')
    ax.set_title('Trim: value-added DB vs ConsDB, by img_type')
    ax.legend(ncol=5, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return tab


def bar_causes(df, path):
    bad = df[df['status'].isin(['differ', 'db_only', 'consdb_only'])]
    tab = pd.crosstab([bad['img_type']], [bad['status'], bad['cause']])
    tab = tab.reindex([t for t in IMG_ORDER if t in tab.index]
                      + [t for t in tab.index if t not in IMG_ORDER])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    left = np.zeros(len(tab))
    cmap = plt.get_cmap('tab10')
    for k, col in enumerate(tab.columns):
        ax.barh(tab.index, tab[col], left=left, color=cmap(k % 10),
                label=f'{col[0]}: {col[1]}')
        left += tab[col].to_numpy()
    ax.invert_yaxis()
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlabel('exposures (count)')
    ax.set_title('Trim mismatches by img_type and cause')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return tab


def status_by_month(df, path, img_type='science'):
    d = df[df['img_type'] == img_type].copy()
    d['month'] = (d['day_obs'] // 100).astype(str)
    tab = pd.crosstab(d['month'], d['status']).reindex(columns=STATUS_ORDER,
                                                         fill_value=0)
    frac = tab.div(tab.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(frac))
    for s in STATUS_ORDER:
        ax.bar(frac.index, frac[s], bottom=bottom, color=STATUS_COLOR[s], label=s)
        bottom += frac[s].to_numpy()
    ax.set_ylim(0, 1)
    ax.set_ylabel(f'fraction of {img_type} exposures\n(dimensionless)')
    ax.set_xlabel('month (YYYYMM of day_obs)')
    ax.tick_params(axis='x', rotation=60)
    ax.set_title(f'Trim comparison by month, img_type = {img_type}')
    ax.legend(ncol=5, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return tab


def scatter_dofs(df, path, dofs=(5, 0, 8, 10)):
    both = df[df['status'].isin(['agree', 'differ'])]
    fig, axes = plt.subplots(2, len(dofs), figsize=(4 * len(dofs), 7.5))
    for k, i in enumerate(dofs):
        x = both[f'db_dof{i}'].to_numpy()
        y = both[f'cdb_dof{i}'].to_numpy()
        diff = both['status'].to_numpy() == 'differ'
        ax = axes[0, k]
        ax.scatter(x[~diff], y[~diff], s=2, color=STATUS_COLOR['agree'], label='agree')
        ax.scatter(x[diff], y[diff], s=4, color=STATUS_COLOR['differ'], label='differ')
        lim = np.nanpercentile(np.r_[x, y], [0.1, 99.9])
        ax.plot(lim, lim, 'k-', lw=0.5)
        ax.set_xlabel(f'value-added DB [{dof_unit(i)}]')
        ax.set_ylabel(f'ConsDB [{dof_unit(i)}]')
        ax.set_title(dof_name(i), fontsize=9)
        ax.legend(fontsize=7, markerscale=3)
        ax = axes[1, k]
        d = np.abs(y - x)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d):
            ax.hist(np.log10(d), bins=60, color=STATUS_COLOR['differ'])
        ax.set_xlabel(f'log10 |ConsDB - DB| [log10 {dof_unit(i)}], nonzero only')
        ax.set_ylabel('exposures (count)')
        ax.set_title(f'{len(d):,} nonzero of {len(x):,} with both', fontsize=9)
    fig.suptitle('Trim per DOF: exposures with a value in both databases')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def lateness_hist(df, ev, path):
    """When the in-exposure event fell, relative to obs_start, for 'differ' exposures."""
    d = df[(df['status'] == 'differ') & (df['cause'] == 'event during exposure')]
    et = ev['time_utc'].to_numpy(dtype='datetime64[ns]')
    lag = (et[d['i_cdb'].to_numpy()] - d['start_utc'].to_numpy(dtype='datetime64[ns]'))
    lag_s = lag.astype('timedelta64[ms]').astype(float) / 1e3
    frac = lag_s / d['exp_time'].clip(lower=1e-3).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(lag_s, bins=60)
    axes[0].set_xlabel('ConsDB event time - obs_start [s]')
    axes[0].set_ylabel('exposures (count)')
    axes[1].hist(np.clip(frac, 0, 1.5), bins=60)
    axes[1].set_xlabel('(event time - obs_start) / exp_time (dimensionless)')
    fig.suptitle(f'Where in the exposure the ConsDB-chosen event fell, n = {len(d):,}')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def history(row, ev, step_scale, path, n_top=2,
            pad_before=pd.Timedelta(minutes=6), pad_after=pd.Timedelta(minutes=3)):
    """EFD Trim events around one exposure: camera and M2 dz, plus the DOFs whose
    ConsDB-minus-DB difference is largest relative to their typical per-event step."""
    db = np.array([row[f'db_{d}'] for d in DOFS], float)
    cd = np.array([row[f'cdb_{d}'] for d in DOFS], float)
    rel = np.abs(cd - db) / step_scale
    top = [int(i) for i in np.argsort(-np.nan_to_num(rel)) if i not in (5, 0)][:n_top]
    show = [5, 0] + top
    t0, t1 = row['start_utc'], row['end_utc']
    lo, hi = t0 - pad_before, t1 + pad_after
    win = ev[(ev['time_utc'] >= lo) & (ev['time_utc'] <= hi)]
    prev = ev[ev['time_utc'] < lo].tail(1).assign(time_utc=lo)
    win = pd.concat([prev, win, win.tail(1).assign(time_utc=hi)])
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 2.2 * len(show)), sharex=True)
    for ax, i in zip(axes, show):
        ax.step(win['time_utc'], win[f'dof{i}'], where='post', color='0.3', lw=1)
        pts = win.iloc[1:-1]
        ax.plot(pts['time_utc'], pts[f'dof{i}'], 'o', ms=4, color='0.3',
                label='EFD degreeOfFreedom event')
        ax.axvspan(t0, t1, color='#edae49', alpha=0.35, label='exposure')
        ax.axhline(db[i], color='#4c9f70', ls='--', label='value-added DB (as of obs_start)')
        ax.axhline(cd[i], color='#d1495b', ls=':', lw=2,
                   label='ConsDB (last event in [start-1h, end])')
        ax.set_ylabel(f'{dof_name(i)}\n[{dof_unit(i)}]', fontsize=8)
    axes[0].legend(fontsize=7, ncol=2, loc='best')
    axes[-1].set_xlim(lo, hi)
    axes[-1].set_xlabel('time [UTC]')
    fig.suptitle(f'exposure {row["exposure_id"]} ({row["img_type"]}, '
                 f'exp_time = {row["exp_time"]:.1f} s): '
                 f'{int(row["n_events_during"])} event(s) during the exposure',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-history', type=int, default=2,
                        help='science mismatches per epoch to plot EFD time histories for')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    inp = latest_inputs()
    run = study_run('value_added/compare_trim_consdb', 'compare')
    print(f'inputs: {inp}\noutput: {run.path}')

    exposure = pd.read_parquet(inp / 'consdb_exposure.parquet')
    cdb = pd.read_parquet(inp / 'consdb_trim.parquet')
    db = pd.read_parquet(inp / 'db_trim.parquet')
    ev = pd.read_parquet(inp / 'efd_dof_events.parquet').sort_values('time_utc')
    ev = ev.reset_index(drop=True)

    exposure = exposure.dropna(subset=['obs_start', 'obs_end']).copy()
    exposure['start_utc'] = tai_to_utc(exposure['obs_start'])
    exposure['end_utc'] = tai_to_utc(exposure['obs_end'])
    df = exposure.merge(
        db[['visit_id'] + DOFS].rename(columns={d: f'db_{d}' for d in DOFS}),
        left_on='exposure_id', right_on='visit_id', how='left')
    df = df.merge(cdb.rename(columns={d: f'cdb_{d}' for d in DOFS}),
                  on='exposure_id', how='left')
    df = df.sort_values('start_utc').reset_index(drop=True)
    n_not_in_db = int(df['visit_id'].isna().sum())

    i_db, i_cdb, n_during = rebuild(df, ev)
    df['i_db'], df['i_cdb'], df['n_events_during'] = i_db, i_cdb, n_during
    ev_vals = ev[DOFS].to_numpy(float)
    df['db_matches_rule'] = rule_match(df[[f'db_{d}' for d in DOFS]].to_numpy(float),
                                       ev_vals, i_db)
    df['cdb_matches_rule'] = rule_match(df[[f'cdb_{d}' for d in DOFS]].to_numpy(float),
                                        ev_vals, i_cdb)
    df = explain(classify(df))

    keep = ['exposure_id', 'day_obs', 'seq_num', 'img_type', 'obs_start', 'obs_end',
            'exp_time', 'science_program', 'observation_reason', 'status', 'cause',
            'n_dof_differ', 'max_absdiff', 'n_events_during', 'db_matches_rule',
            'cdb_matches_rule']
    df[keep].to_parquet(run.path / 'comparison.parquet')

    tab_img = bar_by_img_type(df, run.path / 'status_by_img_type.png')
    tab_cause = bar_causes(df, run.path / 'mismatch_causes_by_img_type.png')
    tab_month = status_by_month(df, run.path / 'status_by_month_science.png')
    scatter_dofs(df, run.path / 'scatter_dof.png')
    if (df['cause'] == 'event during exposure').any():
        lateness_hist(df, ev, run.path / 'event_time_in_exposure.png')
    for name, tab in [('status_by_img_type', tab_img), ('causes', tab_cause),
                      ('status_by_month_science', tab_month)]:
        tab.to_csv(run.path / f'{name}.csv')

    both = df['status'].isin(['agree', 'differ', 'db_only', 'consdb_only'])
    print(f'\nexposures in ConsDB with times: {len(df):,}; not in the DB: {n_not_in_db:,}')
    print('\nstatus by img_type (count):\n', tab_img.to_string())
    print('\ncauses (count):\n', tab_cause.T.to_string())
    print('\nrule reconstruction: DB value equals last event <= obs_start on '
          f'{df.loc[df["db_matches_rule"]].shape[0]:,} of '
          f'{np.isfinite(df["db_dof5"]).sum():,} DB rows; ConsDB value equals last event '
          f'in [start-1h, end] on {df["cdb_matches_rule"].sum():,} of '
          f'{np.isfinite(df["cdb_dof5"]).sum():,} ConsDB rows (counts)')
    print(f'exposures with any Trim value: {both.sum():,} (count)')

    # Typical per-event Trim step (= Tweak) per DOF, nMAD over all events.
    steps = np.diff(ev_vals, axis=0)
    step_scale = 1.4826 * np.nanmedian(np.abs(steps - np.nanmedian(steps, axis=0)), axis=0)
    step_scale = np.where(step_scale > 0, step_scale, np.nanstd(steps, axis=0))

    sci = df[(df['img_type'] == 'science') & (df['status'] == 'differ')]
    # One example per epoch: before and after the 2026-04 drop in the differ fraction.
    for label, sel in [('early', sci['day_obs'] < 20260401),
                       ('late', sci['day_obs'] >= 20260401)]:
        pick = sci[sel]
        if not len(pick):
            continue
        pick = pick.sample(min(args.n_history, len(pick)), random_state=args.seed)
        for _, row in pick.iterrows():
            history(row, ev, step_scale,
                    run.path / f'history_{label}_{row["exposure_id"]}.png')
            print(f'history ({label}): {row["exposure_id"]} n_events_during = '
                  f'{row["n_events_during"]}, cause = {row["cause"]}')
    unexplained = df[df['cause'] == 'unexplained']
    print(f'unexplained exposures: {len(unexplained):,} (count)')


if __name__ == '__main__':
    main()
