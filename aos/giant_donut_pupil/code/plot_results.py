"""Results PDF for the giant-donut fits (`fit_donuts.py`).

Pages:

1. Summary: residual RMS per donut and side for each pupil, and how many donuts
   v4.02 fits better, for all donuts and the clean subset.
2. Z5-Z8 focal-plane maps (OCS, deg), total wavefront (batoid reference plus
   fitted deviation), each pupil and Zernike set. In the Z4 + Z11 + Z22 set,
   Z5-Z8 are not fitted, so the map is the reference alone.
3. Z5-Z8 fitted deviation (blitz set) maps, OCS.
4. Z5-Z8 total and deviation against field radius (deg).
5. Intra against extra, deviations, both pupils.
6. Per donut and side: data, model and residual for both pupils (blitz set),
   and the v1000 residual minus the v4.02 residual.

Markers: circle = clean; triangle = beyond the 1.725 deg field radius; square =
corner donut not isolated on that visit; diamond = both flags.

Run::

    python aos/giant_donut_pupil/code/plot_results.py --run <fits run directory>
"""
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PUPILS = ('v1000', 'v4.02')
ZSETS = {'z4_11_22': 'Z4+Z11+Z22', 'blitz': 'blitz set (Noll 4-19, 22-26)'}
ZK = (5, 6, 7, 8)
PUPIL_COLOURS = {'v1000': 'tab:blue', 'v4.02': 'tab:red'}
FLAG_LEGEND = [Line2D([], [], marker=m, ls='', color='k', label=lab)
               for m, lab in (('o', 'clean'), ('^', 'beyond 1.725 deg'),
                              ('s', 'corner not isolated'), ('D', 'both flags'))]


def marker(row):
    if row.beyond_fov and row.corner_not_isolated:
        return 'D'
    if row.beyond_fov:
        return '^'
    if row.corner_not_isolated:
        return 's'
    return 'o'


def add_total(df):
    df = df.copy()
    for j in (4, 5, 6, 7, 8, 11, 22):
        df[f'z{j}_tot_ocs_um'] = df[f'z{j}_ref_ocs_um'] + df[f'z{j}_ocs_um'].fillna(0.0)
    df['marker'] = [marker(r) for r in df.itertuples()]
    df['label'] = df.cand_id + ' ' + df.side
    return df


def scatter_flags(ax, x, y, sub, **kw):
    for m in 'o^sD':
        s = sub.marker.to_numpy() == m
        if s.any():
            ax.scatter(np.asarray(x)[s], np.asarray(y)[s], marker=m, **kw)


def page_summary(pdf, df):
    fig = plt.figure(figsize=(17, 11))
    lines = ['Residual RMS inside 1.1 r_out, unmasked pixels (electrons per pixel); '
             'v4.02 better = lower residual RMS', '']
    for zset, zname in ZSETS.items():
        piv = df[df.zset == zset].pivot_table(index=['label'], columns='pupil',
                                             values='resid_rms_e')
        clean = df[df.zset == zset].groupby('label').clean.first()
        better = piv['v4.02'] < piv['v1000']
        ratio = (piv['v4.02'] / piv['v1000'])
        lines.append(f'{zname}: v4.02 better on {better.sum()} of {len(piv)} donut sides '
                     f'(all); {better[clean].sum()} of {clean.sum()} (clean). '
                     f'Median v4.02/v1000 residual RMS ratio (dimensionless): '
                     f'{ratio.median():.4f} all, {ratio[clean].median():.4f} clean.')
    lines.append('')
    piv = {z: df[df.zset == z].pivot_table(index='label', columns='pupil',
                                           values=['resid_rms_e', 'chi2_per_dof'])
           for z in ZSETS}
    first = df.groupby('label').first()
    head = (f'{"donut side":<16}{"det":<9}{"r (deg)":>8} {"flags":<8}'
            f'{"Z4+11+22 v1000":>15}{"v4.02":>8}{"blitz v1000":>13}{"v4.02":>8}'
            f'{"chi2/dof v1000":>16}{"v4.02":>8}')
    lines.append(head)
    for label in piv['blitz'].index:
        f = first.loc[label]
        flag = ('F' if f.beyond_fov else '-') + ('C' if f.corner_not_isolated else '-')
        a = piv['z4_11_22'].loc[label]
        b = piv['blitz'].loc[label]
        lines.append(f'{label:<16}{f.detector:<9}{f.field_r_deg:>8.3f} {flag:<8}'
                     f'{a[("resid_rms_e", "v1000")]:>15.1f}{a[("resid_rms_e", "v4.02")]:>8.1f}'
                     f'{b[("resid_rms_e", "v1000")]:>13.1f}{b[("resid_rms_e", "v4.02")]:>8.1f}'
                     f'{b[("chi2_per_dof", "v1000")]:>16.3f}{b[("chi2_per_dof", "v4.02")]:>8.3f}')
    lines.append('')
    lines.append('flags: F = beyond 1.725 deg field radius; C = corner donut not isolated '
                 '(iso_class != isolated, overlap_frac > 0.02 or annulus_ratio > 0.05). '
                 'Residual RMS columns in electrons per pixel; chi2/dof for the blitz set.')
    fig.text(0.02, 0.98, '\n'.join(lines), family='monospace', fontsize=7.5, va='top')
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (zset, zname) in zip(axes, ZSETS.items()):
        sub = df[df.zset == zset].pivot_table(index='label', columns='pupil',
                                              values='resid_rms_e')
        meta = df[(df.zset == zset) & (df.pupil == 'v1000')].set_index('label').loc[sub.index]
        scatter_flags(ax, sub['v1000'], sub['v4.02'], meta, c='k', s=30)
        lim = [0, 1.05 * np.nanmax(sub.to_numpy())]
        ax.plot(lim, lim, 'k:', lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('v1000 residual RMS (electrons per pixel)')
        ax.set_ylabel('v4.02 residual RMS (electrons per pixel)')
        ax.set_title(f'{zname}: below the line, v4.02 fits better')
        ax.legend(handles=FLAG_LEGEND, fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_maps(pdf, df, kind):
    """kind 'tot' (both sets) or 'dev' (blitz set only)."""
    sets = list(ZSETS) if kind == 'tot' else ['blitz']
    for zset in sets:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
        for i, p in enumerate(PUPILS):
            sub = df[(df.zset == zset) & (df.pupil == p)]
            for k, j in enumerate(ZK):
                ax = axes[i, k]
                col = f'z{j}_{kind}_ocs_um' if kind == 'tot' else f'z{j}_ocs_um'
                vals = sub[col].to_numpy()
                lim = robust_lim(df[df.zset == zset], col)
                for m in 'o^sD':
                    s = sub.marker.to_numpy() == m
                    sc = ax.scatter(sub.field_x_ocs_deg[s], sub.field_y_ocs_deg[s],
                                    c=vals[s], cmap='RdBu_r', vmin=-lim, vmax=lim,
                                    marker=m, s=60, edgecolor='k', lw=0.4)
                ax.add_patch(plt.Circle((0, 0), 1.725, fill=False, ls=':', color='gray'))
                ax.set_aspect('equal')
                ax.set_xlim(-2.1, 2.1)
                ax.set_ylim(-2.1, 2.1)
                ax.set_title(f'{p}: Z{j} {"total" if kind == "tot" else "deviation"}')
                ax.set_xlabel('field x, OCS (deg)')
                ax.set_ylabel('field y, OCS (deg)')
                fig.colorbar(sc, ax=ax, shrink=0.8, label=f'Z{j} (µm of wavefront)')
        what = ('total wavefront = batoid reference + fitted deviation'
                if kind == 'tot' else 'fitted deviation from the batoid reference')
        note = ' (Z5-Z8 not fitted in this set: reference only)' if zset == 'z4_11_22' else ''
        fig.suptitle(f'{ZSETS[zset]}: {what}{note}; both visits; dotted circle 1.725 deg; '
                     'colour scale from the clean rows, saturates beyond')
        axes[0, 0].legend(handles=FLAG_LEGEND, fontsize=7, loc='lower left')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def robust_lim(df, col):
    """Symmetric limit: 1.2 x the largest |value| among clean rows."""
    v = np.abs(df.loc[df.clean, col].dropna())
    if v.empty:
        v = np.abs(df[col].dropna())
    return float(1.2 * v.max()) if len(v) and v.max() > 0 else 1.0


def page_radius(pdf, df):
    for zset in ZSETS:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        dev_terms = ZK if zset == 'blitz' else (4, 11, 22)
        for row, kind in enumerate(('tot', 'dev')):
            for k, j in enumerate(ZK if kind == 'tot' else dev_terms):
                ax = axes[row, k]
                col = f'z{j}_tot_ocs_um' if kind == 'tot' else f'z{j}_ocs_um'
                sel = df[df.zset == zset]
                vals = sel[col].dropna()
                c = sel.loc[sel.clean, col].dropna()
                lo, hi = (c.min(), c.max()) if len(c) else (vals.min(), vals.max())
                pad = 0.3 * (hi - lo) + 1e-3
                lo, hi = lo - pad, hi + pad
                n_off = int(((vals < lo) | (vals > hi)).sum())
                for p in PUPILS:
                    for side, fill in (('extra', True), ('intra', False)):
                        sub = df[(df.zset == zset) & (df.pupil == p) & (df.side == side)]
                        scatter_flags(ax, sub.field_r_deg, sub[col], sub, s=30,
                                      facecolors=PUPIL_COLOURS[p] if fill else 'none',
                                      edgecolors=PUPIL_COLOURS[p])
                ax.axvline(1.725, color='gray', ls=':')
                ax.set_ylim(lo, hi)
                ax.set_xlim(0, 2.05)
                ax.set_xlabel('field radius (deg)')
                ax.set_ylabel(f'Z{j} OCS (µm of wavefront)')
                ax.set_title(f'Z{j} {"total" if kind == "tot" else "deviation"}'
                             f' ({n_off} of {len(vals)} fits off scale)')
            for k in range(len(dev_terms) if kind == 'dev' else 4, 4):
                axes[row, k].axis('off')
        handles = FLAG_LEGEND + [
            Line2D([], [], marker='o', ls='', color=PUPIL_COLOURS[p], label=p) for p in PUPILS
        ] + [Line2D([], [], marker='o', ls='', mfc='none', color='k', label='intra (open)'),
             Line2D([], [], marker='o', ls='', color='k', label='extra (filled)')]
        axes[0, 0].legend(handles=handles, fontsize=7)
        fig.suptitle(f'{ZSETS[zset]}: against field radius; top Z5-Z8 total (reference + '
                     f'deviation), bottom fitted deviation of '
                     f'{", ".join("Z%d" % j for j in dev_terms)}; y range from the clean rows')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def page_intra_extra(pdf, df):
    combos = [('blitz', j) for j in ZK] + [('z4_11_22', j) for j in (4, 11, 22)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (zset, j) in zip(axes.ravel(), combos):
        for p in PUPILS:
            sub = df[(df.zset == zset) & (df.pupil == p)]
            piv = sub.pivot_table(index='cand_id', columns='side', values=f'z{j}_ocs_um')
            if not {'extra', 'intra'} <= set(piv.columns):
                continue
            piv = piv.dropna()
            meta = sub[sub.side == 'extra'].set_index('cand_id').loc[piv.index]
            scatter_flags(ax, piv['extra'], piv['intra'], meta, c=PUPIL_COLOURS[p], s=30)
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], 'k:', lw=1)
        ax.set_xlabel(f'extra-focal Z{j} deviation, OCS (µm)')
        ax.set_ylabel(f'intra-focal Z{j} deviation, OCS (µm)')
        ax.set_title(f'Z{j}, {ZSETS[zset]}')
    axes[-1, -1].axis('off')
    axes[-1, -1].legend(handles=FLAG_LEGEND + [
        Line2D([], [], marker='o', ls='', color=PUPIL_COLOURS[p], label=p) for p in PUPILS],
        loc='center')
    fig.suptitle('Intra against extra, fitted deviation (µm of wavefront), donuts on both '
                 'visits; dotted line intra = extra')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_donuts(pdf, df, run):
    sub = df[df.zset == 'blitz']
    for label, grp in sub.groupby('label', sort=True):
        r = grp.set_index('pupil')
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        res = {}
        for i, p in enumerate(PUPILS):
            f = np.load(run / 'images' / f'{label.replace(" ", "_")}_{p}_blitz.npz')
            img, mod, msk = f['img'], f['model'], f['masked']
            resid = np.where(msk, np.nan, img - mod)
            res[p] = resid
            vmax = np.nanpercentile(img, 99.5)
            rl = np.nanpercentile(np.abs(resid), 99)
            for k, (a, title, kw) in enumerate((
                    (np.where(msk, np.nan, img), 'data', dict(vmin=0, vmax=vmax, cmap='gray')),
                    (mod, 'model', dict(vmin=0, vmax=vmax, cmap='gray')),
                    (resid, 'data - model', dict(vmin=-rl, vmax=rl, cmap='RdBu_r')))):
                im = axes[i, k].imshow(a, origin='lower', **kw)
                axes[i, k].set_title(f'{p} {title}')
                fig.colorbar(im, ax=axes[i, k], shrink=0.75, label='electrons per pixel')
            rr = r.loc[p]
            axes[0, 3].text(0, 1.0 - 0.5 * i, f'{p}\nresidual RMS {rr.resid_rms_e:.1f} electrons per pixel'
                            f'\nchi2/dof {rr.chi2_per_dof:.3f} (dof {int(rr.dof)})'
                            f'\nblur FWHM {rr.fwhm_arcsec:.3f} arcsec'
                            f'{" (at bound)" if rr.fwhm_at_bound else ""}'
                            f'\nZ5..Z8 OCS (µm): {rr.z5_ocs_um:+.3f} {rr.z6_ocs_um:+.3f} '
                            f'{rr.z7_ocs_um:+.3f} {rr.z8_ocs_um:+.3f}',
                            va='top', fontsize=9, transform=axes[0, 3].transAxes)
        d = np.abs(res['v1000']) - np.abs(res['v4.02'])
        rl = np.nanpercentile(np.abs(d), 99)
        axes[0, 3].axis('off')
        ax = axes[1, 3]
        im = ax.imshow(d, origin='lower', cmap='PuOr', vmin=-rl, vmax=rl)
        ax.set_title('|resid v1000| - |resid v4.02|\n(>0: v4.02 closer)', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label='electrons per pixel')
        f0 = grp.iloc[0]
        flags = ', '.join(n for n, v in (('beyond 1.725 deg', f0.beyond_fov),
                                         ('corner not isolated', f0.corner_not_isolated)) if v)
        fig.suptitle(f'{label} {f0.detector}, field radius {f0.field_r_deg:.3f} deg, '
                     f'blitz set, CCS stamp; neighbour mask '
                     f'{"applied" if f0.mask_applied else "none"} '
                     f'(ring fraction masked {f0.masked_ring_frac:.3f}, dimensionless)'
                     f'{"; FLAGS: " + flags if flags else "; clean"}', fontsize=10)
        pdf.savefig(fig, dpi=90)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    df = pd.read_parquet(run / 'fits.parquet')
    df = add_total(df[df.error == ''])
    out = run / 'giant_donut_pupil_results.pdf'
    with PdfPages(out) as pdf:
        page_summary(pdf, df)
        page_maps(pdf, df, 'tot')
        page_maps(pdf, df, 'dev')
        page_radius(pdf, df)
        page_intra_extra(pdf, df)
        page_donuts(pdf, df, run)
    print('wrote', out)


if __name__ == '__main__':
    main()
