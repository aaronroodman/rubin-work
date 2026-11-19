"""Data against model rim profiles for a few candidate roll-offs, from a ``fit_rolloff`` run.

Candidates (nearest grid points): nominal (no roll-off, no turn-up), the mirror-lab map
(A = 75 nm, w = 27 mm; B = 80 nm, u = 25 mm), and the best fits of all, extra-focal and
intra-focal sides from the run's ``results.json``. Each side takes, per candidate, the
blur that minimizes its M1-inner plus M3-outer cost.

All curves of a bin are drawn in the nominal model's frame: the data aligned and scaled
by the nominal fit, and each candidate model moved by its own fitted shift and scale
relative to the nominal one. So one data curve is compared with every candidate. The
residual panels are data minus each candidate in that candidate's own fit, over its
plateau level.

psi is the image azimuth about the donut center measured from the direction away from
the field center: psi = (OCS image azimuth − (field azimuth + 180 deg)) mod 360, as in
`aos/giant_donut_pupil`.

Writes ``compare_profiles.pdf`` into a new study run.

Usage, from the repo root::

    python aos/mirror_edge_rolloff/code/compare_profiles.py --fit 20261119_fit_rolloff \\
        --models 20261118_model_profiles
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_rolloff as fr  # noqa: E402
from rubinwork.giant_donuts import edge_profiles as ep  # noqa: E402

X_GRID = np.arange(-25, 25.01, 0.5)
PSI_EDGES = np.arange(0, 361, 60)
MIN_BINS = 40
COLORS = {'nominal': 'tab:blue', 'map': 'tab:green', 'best all': 'tab:red',
          'best extra': 'tab:orange', 'best intra': 'tab:purple'}
SET_TITLE = {'m1_inner': 'inner edge set by M1', 'inner_other': 'inner edge set by another element',
             'm3_outer': 'outer edge set by M3', 'baffle_outer': 'outer edge set by M1Baffle1'}


def candidates(grids, results):
    """Grid indices (k for M1, j for M3) of each candidate."""
    A, W, B, U = grids['A_nm'], grids['w_mm'], grids['B_nm'], grids['u_mm']

    def k_of(a, w):
        return int(np.argmin(np.abs(A - a) / 25 + np.abs(np.log(W / w))))

    def j_of(b, u):
        return int(np.argmin(np.abs(B - b) / 40 + np.abs(np.log(U / u))))
    out = {'nominal': (k_of(0, 27), j_of(0, 25)), 'map': (k_of(75, 27), j_of(80, 25))}
    for g in ('all', 'extra', 'intra'):
        if g in results:
            r = results[g]
            out[f'best {g}'] = (k_of(r['A_nm']['value'], r['w_mm']['value']),
                                j_of(r['B_nm']['value'], r['u_mm']['value']))
    return out


def label(name, k, j, grids):
    return (f'{name} (A {grids["A_nm"][k]:g} nm, w {grids["w_mm"][k]:g} mm; '
            f'B {grids["B_nm"][j]:g} nm, u {grids["u_mm"][j]:g} mm)')


def collect(fit_dir, models_dir, keys, sides_tab, cands, C):
    """Per-bin aligned curves for every side, set and candidate, as a list of dicts."""
    rows = []
    info = sides_tab.set_index('key')
    for s_i, key in enumerate(keys):
        d = dict(np.load(pathlib.Path(fit_dir) / 'costs' / f'{key}.npz'))
        m = dict(np.load(pathlib.Path(models_dir) / 'profiles' / f'{key}.npz'))
        r = info.loc[key]
        blur = {}
        for name, (k, j) in cands.items():
            tot = C['m1_inner'][s_i, k] + C['m3_outer'][s_i, j]
            blur[name] = int(np.argmin(tot))
        k0, j0 = cands['nominal']
        for set_name, (edge, _, _) in fr.SETS.items():
            sel = d[f'set_{set_name}'] & d[f'{edge}_ok']
            if not sel.any():
                continue
            grid = 'prof_m1' if edge == 'inner' else 'prof_m3'
            bins = d[f'{edge}_bins'][sel]
            e_nom = (m['e_in'] if edge == 'inner' else m['e_out'])[bins]
            sign = 1.0 if edge == 'inner' else -1.0
            rm = d[f'{edge}_rad_model'][sel]
            rd = d[f'{edge}_rad_data'][sel]
            data = d[f'{edge}_data'][sel]
            f0 = blur['nominal']
            d0 = d[f'{edge}_delta'][k0 if edge == 'inner' else j0, f0, sel]
            s0 = d[f'{edge}_scale'][k0 if edge == 'inner' else j0, f0, sel]
            xm0 = sign * (rm - e_nom[:, None])
            P0 = m[grid][k0 if edge == 'inner' else j0, f0, bins]
            lvl0 = np.nanmedian(np.where((xm0 >= fr.RIM_PLATEAU_PIX[0]) & (xm0 <= fr.RIM_PLATEAU_PIX[1]),
                                         P0, np.nan), axis=1)
            az_ccs = (bins + 0.5) * 360 / ep.NBIN
            psi = np.mod(az_ccs + r.rtp_deg - (r.field_az_ocs_deg + 180.0), 360.0)
            base = dict(key=key, side=r.side, night=int(r.night), set=set_name, psi=psi,
                        x_data=sign * (rd - d0[:, None] - e_nom[:, None]),
                        data=data / s0[:, None] / lvl0[:, None])
            for name, (k, j) in cands.items():
                kk = k if edge == 'inner' else j
                f = blur[name]
                dl = d[f'{edge}_delta'][kk, f, sel]
                sc = d[f'{edge}_scale'][kk, f, sel]
                P = m[grid][kk, f, bins]
                Pd = np.array([np.interp(q - dd, rr, pp) for q, dd, rr, pp in zip(rd, dl, rm, P)])
                lvl = np.nanmedian(np.where((xm0 >= fr.RIM_PLATEAU_PIX[0]) & (xm0 <= fr.RIM_PLATEAU_PIX[1]),
                                            P, np.nan), axis=1)
                rows.append(dict(base, cand=name,
                                 cost=d[f'{edge}_cost'][kk, f, sel],
                                 x_model=sign * (rm + (dl - d0)[:, None] - e_nom[:, None]),
                                 model=P * (sc / s0)[:, None] / lvl0[:, None],
                                 resid=(data / sc[:, None] - Pd) / lvl[:, None]))
    return rows


def binned_median(x, y):
    x, y = np.concatenate(x).ravel(), np.concatenate(y).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    kb = np.digitize(x[good], X_GRID)
    y = y[good]
    med = np.full(len(X_GRID) - 1, np.nan)
    for i in range(1, len(X_GRID)):
        v = y[kb == i]
        if v.size:
            med[i - 1] = np.median(v)
    return 0.5 * (X_GRID[1:] + X_GRID[:-1]), med


def rim(x, p):
    pk = np.nanmax(np.where((x >= 0) & (x <= fr.RIM_PEAK_PIX), p, np.nan))
    pl = np.nanmedian(np.where((x >= fr.RIM_PLATEAU_PIX[0]) & (x <= fr.RIM_PLATEAU_PIX[1]), p, np.nan))
    return pk / pl


def subset(rows, cand, **sel):
    """Rows of one candidate matching the given fields."""
    return [r for r in rows if r['cand'] == cand and all(r[k] == v for k, v in sel.items())]


def psi_cut(rows, lo, hi):
    """Rows restricted to the bins with lo <= psi < hi (deg)."""
    out = []
    for r in rows:
        keep = (r['psi'] >= lo) & (r['psi'] < hi)
        if keep.any():
            out.append({k: (v[keep] if isinstance(v, np.ndarray) and v.ndim else v)
                        for k, v in r.items()})
    return out


def panel(ax, rax, rows, cands, show, title):
    """Data (nominal frame) and candidate model medians; residuals on rax."""
    base = subset(rows, 'nominal')
    n = sum(len(r['psi']) for r in base)
    if not n:
        ax.set_visible(False)
        if rax is not None:
            rax.set_visible(False)
        return n
    x, y = binned_median([r['x_data'] for r in base], [r['data'] for r in base])
    ax.plot(x, y, 'k.', ms=3, label=f'data (rim {rim(x, y):.2f})')
    for name in show:
        sub = subset(rows, name)
        xm, ym = binned_median([r['x_model'] for r in sub], [r['model'] for r in sub])
        cost = np.nanmean(np.concatenate([r['cost'] for r in sub]))
        ax.plot(xm, ym, '-', color=COLORS[name], lw=1,
                label=f'{name} (rim {rim(xm, ym):.2f}, cost {cost:.0f})')
        if rax is not None:
            xr, yr = binned_median([r['x_data'] for r in sub], [r['resid'] for r in sub])
            rax.plot(xr, yr, '-', color=COLORS[name], lw=1)
    ax.set_title(f'{title} ({n} bins)', fontsize=8)
    ax.set_ylabel('profile / plateau (dimensionless)', fontsize=7)
    ax.legend(fontsize=6)
    ax.set_xlim(-20, 22)
    if rax is not None:
        rax.axhline(0, color='0.5', lw=0.5)
        rax.set_xlim(-20, 22)
        rax.set_ylabel('(data - model) / plateau\n(dimensionless)', fontsize=7)
        rax.set_xlabel('distance from nominal edge, bright side positive (pixels)', fontsize=7)
    return n


def summary_table(rows, cands, grids):
    out = []
    for set_name in fr.SETS:
        for side in ('intra', 'extra'):
            base = subset(rows, 'nominal', set=set_name, side=side)
            if not base:
                continue
            x, y = binned_median([r['x_data'] for r in base], [r['data'] for r in base])
            rec = dict(set=set_name, side=side, n_bins=sum(len(r['psi']) for r in base),
                       rim_data=rim(x, y))
            for name in cands:
                sub = subset(rows, name, set=set_name, side=side)
                xm, ym = binned_median([r['x_model'] for r in sub], [r['model'] for r in sub])
                rec[f'rim {name}'] = rim(xm, ym)
                rec[f'cost {name}'] = np.nanmean(np.concatenate([r['cost'] for r in sub]))
            out.append(rec)
    return pd.DataFrame(out)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from rubinwork.common import study_run
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--fit', required=True, help='fit_rolloff run directory name')
    p.add_argument('--models', required=True, help='model_profiles run directory name')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    root = fr._study_out()
    fit_dir, models_dir = root / a.fit, root / a.models
    out = pathlib.Path(a.out) if a.out else study_run(fr.STUDY, 'compare_profiles').path
    out.mkdir(parents=True, exist_ok=True)
    C, N, sides, keys, grids = fr.load_costs(fit_dir, models_dir)
    results = json.loads((fit_dir / 'results.json').read_text())
    cands = candidates(grids, results)
    tab = pd.read_csv(models_dir / 'sides.csv')
    rows = collect(fit_dir, models_dir, keys, tab, cands, C)
    summ = summary_table(rows, cands, grids)
    summ.to_csv(out / 'summary.csv', index=False, float_format='%.4g')
    main_show = ['nominal', 'map', 'best all']
    side_show = ['nominal', 'map', 'best all', 'best extra', 'best intra']
    with PdfPages(out / 'compare_profiles.pdf') as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        txt = ['Data against model rim profiles, candidate roll-offs (nearest grid points):']
        txt += ['  ' + label(n, k, j, grids) for n, (k, j) in cands.items()]
        txt += [f'{len(keys)} sides: {(sides == "intra").sum()} intra, {(sides == "extra").sum()} extra. '
                'rim = peak within 10 px over the 12-25 px plateau; cost = mean Huber cost per bin '
                '(76 points); both dimensionless.', '']
        show = summ.copy()
        for c in show.columns:
            if show[c].dtype.kind == 'f':
                show[c] = show[c].map(lambda v: f'{v:.2f}' if abs(v) < 10 else f'{v:.0f}')
        fig.text(0.03, 0.97, '\n'.join(txt), va='top', family='monospace', fontsize=7)
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.6])
        ax.axis('off')
        t = ax.table(cellText=show.to_numpy(), colLabels=[c.replace(' ', '\n') for c in show.columns],
                     loc='upper center')
        t.auto_set_font_size(False)
        t.set_fontsize(6)
        t.scale(1, 2.2)
        pdf.savefig(fig)
        plt.close(fig)

        # all bins, per set and side, with residuals
        for set_name in fr.SETS:
            fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), gridspec_kw=dict(height_ratios=[3, 1.3, 3, 1.3]))
            for i, side in enumerate(('intra', 'extra')):
                sub = [r for r in rows if r['set'] == set_name and r['side'] == side]
                panel(axes[2 * i], axes[2 * i + 1], sub, cands, side_show, f'{side}-focal')
            fig.suptitle(f'{SET_TITLE[set_name]}: median profiles, all bins', fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # psi and night breakdowns of the two mirror edges
        for set_name in ('m1_inner', 'm3_outer'):
            for side in ('intra', 'extra'):
                fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
                for ax, lo, hi in zip(axes.ravel(), PSI_EDGES[:-1], PSI_EDGES[1:]):
                    sub = psi_cut([r for r in rows if r['set'] == set_name and r['side'] == side],
                                  lo, hi)
                    n = panel(ax, None, sub, cands, main_show, f'psi {lo}-{hi} deg')
                    if n < MIN_BINS:
                        ax.set_visible(False)
                fig.suptitle(f'{SET_TITLE[set_name]}, {side}-focal: by psi (deg)', fontsize=10)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
            nights = sorted({r['night'] for r in rows})
            for row_ax, night in zip(axes, nights):
                for ax, side in zip(row_ax, ('intra', 'extra')):
                    sub = [r for r in rows if r['set'] == set_name and r['side'] == side
                           and r['night'] == night]
                    panel(ax, None, sub, cands, main_show, f'night {night}, {side}-focal')
            fig.suptitle(f'{SET_TITLE[set_name]}: by night', fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # single sides: the side with the most M1-inner bins, intra and extra
        for side in ('intra', 'extra'):
            nb = {}
            for r in rows:
                if r['cand'] == 'nominal' and r['set'] == 'm1_inner' and r['side'] == side:
                    nb[r['key']] = len(r['psi'])
            if not nb:
                continue
            key = max(nb, key=nb.get)
            base = next(r for r in rows if r['key'] == key and r['cand'] == 'nominal'
                        and r['set'] == 'm1_inner')
            pick = np.linspace(0, len(base['psi']) - 1, 6).astype(int)
            fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
            for ax, b in zip(axes.ravel(), pick):
                ax.plot(base['x_data'][b], base['data'][b], 'k.', ms=3, label='data')
                for name in main_show:
                    r = next(r for r in rows if r['key'] == key and r['cand'] == name
                             and r['set'] == 'm1_inner')
                    ax.plot(r['x_model'][b], r['model'][b], '-', color=COLORS[name], lw=1,
                            label=f'{name} (cost {r["cost"][b]:.0f})')
                ax.set_title(f'psi {base["psi"][b]:.0f} deg', fontsize=8)
                ax.set_xlim(-20, 22)
                ax.legend(fontsize=6)
                ax.set_xlabel('distance from nominal edge (pixels)', fontsize=7)
                ax.set_ylabel('profile / plateau (dimensionless)', fontsize=7)
            fig.suptitle(f'{key} ({side}-focal), single 2 deg bins of the M1 inner edge', fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print('wrote', out / 'compare_profiles.pdf', flush=True)


if __name__ == '__main__':
    main()
