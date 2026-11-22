"""Figures and numbers for the elevation-pairs note.

Reads the `aos/elevation_pairs` run ``20261121_bins_d13_22_12_dt5min_del2p5_lut20251209``
(pairs, slopes) and the ``bounce_aba`` product, variant ``d13-batoid-22_12``. Writes PNG
and PDF into ``figures/`` and the numbers the note quotes into ``figures/numbers.json``.

    python make_figures.py
"""
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rubinwork.common.runs import data_root
from rubinwork.common.utils import nmad
from rubinwork.products.bounce_aba import reader as bounce_aba

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / 'figures'
RUN = data_root() / 'studies' / 'aos' / 'elevation_pairs' / \
    '20261121_bins_d13_22_12_dt5min_del2p5_lut20251209'
BOUNCE_VARIANT = 'd13-batoid-22_12'
EL_BINS = [(25.0, 40.0), (40.0, 55.0), (55.0, 75.0)]
BIN_KEYS = [f'{lo:g}-{hi:g}' for lo, hi in EL_BINS]
BOUNCE_REF_EL_DEG = 70.0
PLOT_DEL_MAX_DEG = 30.0
MEDIAN_BIN_DEG = 2.0
MIN_PER_MEDIAN_BIN = 20
DOF_IDX = [*range(10), *range(10, 17), *range(30, 35)]
DOF_LABEL = {0: 'M2 dz', 1: 'M2 dx', 2: 'M2 dy', 3: 'M2 rx', 4: 'M2 ry',
             5: 'Cam dz', 6: 'Cam dx', 7: 'Cam dy', 8: 'Cam rx', 9: 'Cam ry',
             **{k: f'M1M3 B{k - 9}' for k in range(10, 30)},
             **{k: f'M2 B{k - 29}' for k in range(30, 50)}}
DOF_UNIT = {k: ('deg' if k in (3, 4, 8, 9) else 'µm') for k in range(50)}
NUM = {}


def label(q):
    if q.startswith('dof'):
        k = int(q[3:])
        return DOF_LABEL[k], DOF_UNIT[k]
    return q, 'dimensionless'


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=130, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def load():
    pairs = pd.read_parquet(RUN / 'pairs_bins.parquet')
    ps = pairs[pairs['d_dmt_v1'].notna()].reset_index(drop=True)
    ps['el_mid_deg'] = ps['el_1_deg'] + 0.5 * ps['d_el_deg']
    slopes = pd.read_csv(RUN / 'slopes_bins.csv')
    st = bounce_aba.load_table('aba_state', BOUNCE_VARIANT)
    st['quantity'] = np.where(st['kind'] == 'vmode', 'v' + st['index'].astype(str),
                              'dof' + st['index'].astype(str))
    return pairs, ps, slopes, st


def fit_of(slopes, q, s):
    return slopes[(slopes['quantity'] == q) & (slopes['sample'] == s)].iloc[0]


def sample_numbers(pairs, ps):
    NUM['pairs_found'] = len(pairs)
    NUM['pairs_with_state'] = len(ps)
    NUM['nights'] = int(ps['day_obs'].nunique())
    NUM['day_obs_range'] = [int(ps['day_obs'].min()), int(ps['day_obs'].max())]
    vis = np.unique(np.r_[ps['visit_id_1'], ps['visit_id_2']])
    NUM['distinct_visits'] = len(vis)
    NUM['pairs_per_visit_mean'] = 2 * len(ps) / len(vis)
    NUM['pairs_per_night_median'] = float(ps.groupby('day_obs').size().median())
    NUM['bands'] = ps['band'].value_counts().to_dict()
    NUM['programs'] = ps['science_program'].value_counts().to_dict()
    NUM['dt_min_median'] = float(ps['dt_min'].median())
    a = ps['d_el_deg'].abs()
    NUM['abs_del_median_deg'] = float(a.median())
    NUM['abs_del_counts'] = {f'{lo}-{hi}': int(((a >= lo) & (a < hi)).sum())
                             for lo, hi in [(2.5, 5), (5, 10), (10, 20), (20, 90)]}
    NUM['bin_counts'] = {k: int(((ps['el_mid_deg'] >= lo) & (ps['el_mid_deg'] < hi)).sum())
                         for k, (lo, hi) in zip(BIN_KEYS, EL_BINS)}


def fig_sample(ps):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')
    axs[0].hist(ps['d_el_deg'], bins=np.arange(-30, 30.5, 0.5), color='C0')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('elevation change, later minus earlier visit [deg]')
    axs[0].set_ylabel('pairs per 0.5 deg')
    axs[0].set_title('Elevation change (|ΔEl| ≥ 2.5 deg)')
    axs[1].hist(ps['el_mid_deg'], bins=np.arange(20, 90.5, 1), color='C0')
    for lo, hi in EL_BINS:
        axs[1].axvline(lo, color='k', lw=0.8, ls='--')
    axs[1].axvline(EL_BINS[-1][1], color='k', lw=0.8, ls='--')
    axs[1].set_xlabel('pair midpoint elevation [deg]')
    axs[1].set_ylabel('pairs per deg')
    axs[1].set_title('Midpoint elevation; dashed: bin edges')
    axs[2].hist(ps['dt_min'], bins=np.arange(0, 5.1, 0.1), color='C0')
    axs[2].set_xlabel('time between visit starts [min]')
    axs[2].set_ylabel('pairs per 0.1 min')
    axs[2].set_title('Time separation (≤ 5 min)')
    save(fig, 'fig01_sample')


def hexbin_panel(ax, fig, xv, yv, f, ylabel, title):
    ok = np.isfinite(xv) & np.isfinite(yv) & (np.abs(xv) <= PLOT_DEL_MAX_DEG)
    xv, yv = xv[ok], yv[ok]
    ylo, yhi = np.quantile(yv, (0.002, 0.998))
    pad = 0.05 * (yhi - ylo)
    ylo, yhi = ylo - pad, yhi + pad
    hb = ax.hexbin(xv, yv, gridsize=(50, 30), mincnt=1, bins='log', cmap='viridis',
                   extent=(-PLOT_DEL_MAX_DEG, PLOT_DEL_MAX_DEG, ylo, yhi))
    fig.colorbar(hb, ax=ax, label='pairs per cell')
    edges = np.arange(-PLOT_DEL_MAX_DEG, PLOT_DEL_MAX_DEG + MEDIAN_BIN_DEG / 2, MEDIAN_BIN_DEG)
    idx = np.digitize(xv, edges)
    xm, ym, em = [], [], []
    for i in np.unique(idx):
        yy = yv[idx == i]
        if len(yy) >= MIN_PER_MEDIAN_BIN:
            xm.append(np.median(xv[idx == i]))
            ym.append(np.median(yy))
            # Error of a median: 1.253 sigma / sqrt(n), with sigma from nMAD.
            em.append(1.253 * nmad(yy) / np.sqrt(len(yy)))
    ax.errorbar(xm, ym, yerr=em, fmt='o', color='white', mec='k', ms=4, ecolor='k',
                capsize=2, label=f'median per {MEDIAN_BIN_DEG:g} deg')
    xx = np.array([-PLOT_DEL_MAX_DEG, PLOT_DEL_MAX_DEG])
    ax.plot(xx, f['intercept'] + f['slope'] * xx, color='C3', lw=1.5, label='Huber fit')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.axvline(0, color='0.6', lw=0.5)
    ax.set_xlim(-PLOT_DEL_MAX_DEG, PLOT_DEL_MAX_DEG)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel('ΔEl, later minus earlier [deg]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper left')


def fig_bins(ps, slopes, q, name):
    lab, unit = label(q)
    col = f'd_dmt_{q}'
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    x = ps['d_el_deg'].to_numpy()
    for ax, key, (lo, hi) in zip(axs, BIN_KEYS, EL_BINS):
        m = ((ps['el_mid_deg'] >= lo) & (ps['el_mid_deg'] < hi)).to_numpy()
        f = fit_of(slopes, q, key)
        title = (f'midpoint elevation {key} deg, n = {int(f["n"])} pairs\n'
                 f'slope {f["slope"]:+.3g} ± {f["slope_err_cl"]:.2g} {unit}/deg '
                 f'(night-clustered)\nPearson r = {f["r"]:+.3f}, Spearman ρ = {f["rho"]:+.3f}')
        hexbin_panel(ax, fig, x[m], ps.loc[m, col].to_numpy(), f, f'Δ{lab} [{unit}]', title)
    save(fig, name)


def bounce_curve(b):
    c = b.groupby('b_elev_deg')['delta'].mean()
    c.loc[BOUNCE_REF_EL_DEG] = 0.0
    return c.sort_index()


def bounce_panel(ax, st, slopes, q):
    lab, unit = label(q)
    b = st[st['quantity'] == q]
    for (night, el), g in b.groupby(['night', 'b_elev_deg']):
        off = {20260418: -0.8, 20260419: 0.0, 20260513: 0.8}.get(night, 0.0) if el == 40 else 0.0
        ax.errorbar(el + off, g['delta'].iloc[0], yerr=g['err'].iloc[0], fmt='o', color='k',
                    ms=4, capsize=2)
    ax.plot([BOUNCE_REF_EL_DEG], [0.0], 'o', mfc='none', mec='k', ms=6)
    curve = bounce_curve(b)
    ax.plot(curve.index, curve.values, color='0.6', lw=0.8, ls=':', zorder=0)
    for i, (key, (lo, hi)) in enumerate(zip(BIN_KEYS, EL_BINS)):
        f = fit_of(slopes, q, key)
        mid = 0.5 * (lo + hi)
        y0 = np.interp(mid, curve.index, curve.values)
        xx = np.linspace(lo, hi, 20)
        ax.fill_between(xx, y0 + (f['slope'] - f['slope_err_cl']) * (xx - mid),
                        y0 + (f['slope'] + f['slope_err_cl']) * (xx - mid),
                        color=f'C{i}', alpha=0.25)
        ax.plot(xx, y0 + f['slope'] * (xx - mid), color=f'C{i}', lw=2)
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set_title(f'{lab} [{unit}]', fontsize=9)
    ax.tick_params(labelsize=7)


def fig_bounce(st, slopes, qs, ncol, name, what):
    nrow = int(np.ceil(len(qs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), layout='constrained',
                            squeeze=False)
    for ax, q in zip(axs.flat, qs):
        bounce_panel(ax, st, slopes, q)
    for ax in list(axs.flat)[len(qs):]:
        ax.axis('off')
    for ax in axs[-1]:
        ax.set_xlabel('elevation [deg]', fontsize=8)
    handles = [plt.Line2D([], [], color=f'C{i}', lw=2, label=f'pairs, midpoint {k} deg')
               for i, k in enumerate(BIN_KEYS)]
    handles += [plt.Line2D([], [], color='k', marker='o', ls='', label='bounce B − A'),
                plt.Line2D([], [], color='k', marker='o', mfc='none', ls='',
                           label='bounce reference A, 70 deg')]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=8)
    fig.suptitle(f'{what}: bounce B − A against B elevation (points) and pair slopes per '
                 'midpoint-elevation bin (segments, ±1σ night-clustered)', fontsize=10)
    save(fig, name)


def chain(slopes, st, q):
    # Pair slopes integrated from 0 at 70 deg down to 30 deg, against the 30 deg bounce leg.
    steps = [('55-75', 70 - 55), ('40-55', 55 - 40), ('25-40', 40 - 30)]
    fs = [fit_of(slopes, q, k) for k, _ in steps]
    val = -sum(f['slope'] * d for f, (_, d) in zip(fs, steps))
    err = float(np.sqrt(sum((f['slope_err_cl'] * d) ** 2 for f, (_, d) in zip(fs, steps))))
    b = st[(st['quantity'] == q) & (st['b_elev_deg'] == 30.0)].iloc[0]
    return dict(pairs=float(val), pairs_err=err, bounce=float(b['delta']), bounce_err=float(b['err']))


def comparison_numbers(slopes, st):
    qs = [f'v{m}' for m in range(1, 13)] + [f'dof{k}' for k in DOF_IDX]
    n_sig = n_same = 0
    ratios = []
    for q in qs:
        b = st[st['quantity'] == q]
        c = b.groupby('b_elev_deg').agg(m=('delta', 'mean'),
                                         e=('err', lambda e: np.sqrt((e ** 2).sum()) / len(e)))
        c.loc[BOUNCE_REF_EL_DEG] = [0.0, 0.0]
        for key, (a, z) in zip(BIN_KEYS, [(30, 40), (40, 50), (60, 70)]):
            bs = (c.m[z] - c.m[a]) / (z - a)
            be = np.hypot(c.e[a], c.e[z]) / (z - a)
            if abs(bs / be) > 3:
                f = fit_of(slopes, q, key)
                n_sig += 1
                n_same += int(np.sign(bs) == np.sign(f['slope']))
                ratios.append(f['slope'] / bs)
    NUM['secants_above_3sigma'] = n_sig
    NUM['secants_same_sign'] = n_same
    NUM['median_pair_over_bounce_slope'] = float(np.median(ratios))
    NUM['chain_70_to_30'] = {q: chain(slopes, st, q) for q in
                             ['v1', 'dof0', 'dof2', 'dof5', 'dof6', 'dof7', 'dof8', 'dof9']}
    NUM['slopes'] = {q: {k: [float(fit_of(slopes, q, k)['slope']),
                             float(fit_of(slopes, q, k)['slope_err_cl'])] for k in BIN_KEYS}
                     for q in ['v1', 'dz1_4', 'dof6', 'dof8', 'dof2']}
    NUM['cl_over_formal_range'] = [float(slopes['cl_over_formal'].min()),
                                   float(slopes['cl_over_formal'].max())]


def main():
    FIG.mkdir(exist_ok=True)
    pairs, ps, slopes, st = load()
    sample_numbers(pairs, ps)
    fig_sample(ps)
    fig_bins(ps, slopes, 'dof6', 'fig02_cam_dx_bins')
    fig_bins(ps, slopes, 'dof8', 'fig03_cam_rx_bins')
    fig_bounce(st, slopes, [f'v{m}' for m in range(1, 13)], 4, 'fig04_vmodes_bounce',
               '12 v-modes')
    fig_bounce(st, slopes, [f'dof{k}' for k in range(10)], 5, 'fig05_hexapods_bounce',
               'Hexapod DOF')
    fig_bounce(st, slopes, [f'dof{k}' for k in [*range(10, 17), *range(30, 35)]], 4,
               'fig06_bending_bounce', 'Mirror bending-mode DOF')
    comparison_numbers(slopes, st)
    with open(FIG / 'numbers.json', 'w') as f:
        json.dump(NUM, f, indent=1, default=float)
    print('wrote numbers.json')


if __name__ == '__main__':
    main()
