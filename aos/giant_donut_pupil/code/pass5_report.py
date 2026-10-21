"""Pupil pass 5 report: ``pupil_pass5.pdf`` (steps 2, 3, 4/6c, 6 and 7) and
``gallery_spiders.pdf`` (step 5).

Reads the pass 5 run (`pass5_spiders.py`, `pass5_blur.py`, `pass5_vignetting.py`,
`pass5_analysis.py`) and the pass 4 run (fits, edges, spiders) for the before/after
counts and the step 3 pages. Every empty panel prints its reason
(`report_utils.empty_panel`).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass5_report.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass5_common as pc  # noqa: E402
import pass5_analysis as pa  # noqa: E402
import pass5_vignetting as pv  # noqa: E402
import pupil_pass4_report as rep4  # noqa: E402
from report_utils import empty_panel  # noqa: E402

from rubinwork.common.utils import nmad  # noqa: E402

RUN = pc.p3.STUDY_DIR / 'output' / f'{pc.RUN_DATE}_{pc.RUN_LABEL}'
VANE_COLOURS = plt.get_cmap('tab10')
SURFACE_COLOURS = {'M1': 'tab:blue', 'M1Baffle1': 'tab:cyan', 'M2': 'tab:green',
                   'M3': 'tab:olive', 'CameraSilhouette': 'tab:red', 'TMASection': 'tab:purple',
                   'M2BaffleCAD': 'tab:orange', 'TopEndCentre': 'tab:brown', 'M2CellTop': 'pink',
                   'TopEndRing': 'tab:gray', 'spider': 'gold', 'none': '0.8'}
SIDE_COLOURS = rep4.SIDE_COLOURS
AARON_RANGES = {'intra': (330.0, 30.0), 'extra': (160.0, 200.0)}


def text_page(pdf, title, text, fontsize=7):
    rep4.text_page(pdf, title, text, fontsize)


def _in_range(psi, lo, hi):
    return (psi >= lo) | (psi < hi) if lo > hi else (psi >= lo) & (psi < hi)


# ---------------------------------------------------------------- step 2

def psi_stack_pass4():
    """Pass 4 step 2 median data − fit c per 2 deg psi bin, science 1.6-1.725 deg, per side."""
    fits = pd.read_parquet(pc.PASS4_RUN / 'fits.parquet')
    bins = pd.read_parquet(pc.PASS4_RUN / 'edges_bins.parquet')
    b_all = rep4.add_psi(bins, fits)
    out = {}
    for side in ('intra', 'extra'):
        f = rep4.psi_select(fits, 'c', ('science', side))
        b = b_all.merge(f[['sample_id', 'side']], on=['sample_id', 'side'])
        _, _, med = rep4.psi_stats(b, 'c', side)
        out[side] = (med, len(f))
    return out


def _bands(ax, psi, surf):
    s = np.asarray(surf)
    i0 = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[i0]:
            ax.axvspan(psi[i0] - 0.5, psi[i - 1] + 0.5, color=SURFACE_COLOURS.get(s[i0], 'k'),
                       alpha=0.18, lw=0, zorder=0)
            i0 = i


def page_vignetting(pdf, vb, imgs, stack, side):
    g = vb[vb.side == side].sort_values('psi_bin').copy()
    for edge in ('inner', 'outer'):
        g[f'surface_{edge}'] = pv._mode_filter(g[f'surface_{edge}'].to_numpy())
        g.loc[g.spider, f'r_{edge}_batoid_pix'] = np.nan
    fig = plt.figure(figsize=(17, 9.5))
    ax0 = fig.add_axes([0.03, 0.08, 0.36, 0.8])
    img = imgs[side]
    ax0.imshow(np.sqrt(img), origin='lower', cmap='gray', extent=(-400, 400, -400, 400))
    phi = pv.FIELD_AZ_DEG
    az = np.deg2rad(g.psi_deg.to_numpy() + phi + 180.0)
    for edge in ('inner', 'outer'):
        r = g[f'r_{edge}_batoid_pix'].to_numpy()
        for s, col in SURFACE_COLOURS.items():
            m = g[f'surface_{edge}'].to_numpy() == s
            ax0.plot(r[m] * np.cos(az[m]), r[m] * np.sin(az[m]), '.', color=col, ms=3)
        rd = g[f'r_{edge}_danish_pix'].to_numpy()
        ax0.plot(rd * np.cos(az), rd * np.sin(az), '-', color='yellow', lw=0.5)
    ax0.annotate('psi 0', (0, 0), xytext=(370 * np.cos(np.deg2rad(phi + 180)),
                                         370 * np.sin(np.deg2rad(phi + 180))),
                 color='w', fontsize=8, ha='center')
    ax0.set_xlabel('x, CCS = OCS (pixels from the ray centroid)')
    ax0.set_ylabel('y (pixels)')
    ax0.set_title(f'(a) Batoid v4.02 full model, {side}-focal, field 1.7 deg at azimuth 45 deg, '
                  'rotator 0.\nEdge points coloured by the surface that vignettes rays just '
                  'outside the edge; yellow: danish v4.02g mask edges', fontsize=8)
    med, n_sides = stack[side]
    lo, hi = AARON_RANGES[side]
    for k, edge in enumerate(('inner', 'outer')):
        ax = fig.add_axes([0.46, 0.53 - 0.45 * k, 0.48, 0.37])
        _bands(ax, g.psi_deg.to_numpy(), g[f'surface_{edge}'].to_numpy())
        ax.plot(g.psi_deg, g[f'r_{edge}_batoid_pix'], 'k-', lw=1.0, label='Batoid v4.02')
        ax.plot(g.psi_deg, g[f'r_{edge}_danish_pix'], '-', color='tab:red', lw=0.8,
                label='danish v4.02g mask')
        for x in ((lo, 360), (0, hi)) if lo > hi else ((lo, hi),):
            ax.axvspan(*x, ymin=0.97, ymax=1.0, color='k', lw=0)
        ax.set_xlim(0, 360)
        ax.set_ylabel(f'{edge} edge radius (pixels)')
        ax2 = ax.twinx()
        ax2.plot(med.psi_deg, med[edge], '-', color='tab:blue', lw=1.0, alpha=0.8,
                 label=f'pass 4 data − fit c, median of {n_sides} science sides 1.6-1.725 deg')
        ax2.axhline(0, color='tab:blue', lw=0.4, ls=':')
        ax2.set_ylabel('data − fit c (pixels)', color='tab:blue')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='lower left')
        if k == 1:
            ax.set_xlabel('relative azimuth psi (deg)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4, label=s)
               for s, c in SURFACE_COLOURS.items() if s in set(g.surface_inner) | set(g.surface_outer)]
    fig.legend(handles=handles, loc='upper right', ncol=len(handles), fontsize=8)
    fig.suptitle(f'Step 2: vignetting surface per psi, {side}-focal (0, {"+" if side == "extra" else "−"}4, '
                 f'{"+" if side == "extra" else "−"}4) mm. Black bar at top: Aaron\'s range psi '
                 f'{lo:.0f}-{hi:.0f} deg', fontsize=10, x=0.35)
    pdf.savefig(fig, dpi=90)
    plt.close(fig)


def range_table(vb, stack):
    rows = []
    for side, (lo, hi) in AARON_RANGES.items():
        g = vb[vb.side == side].sort_values('psi_bin').copy()
        for edge in ('inner', 'outer'):
            g[f'surface_{edge}'] = pv._mode_filter(g[f'surface_{edge}'].to_numpy())
        g = g[_in_range(g.psi_deg, lo, hi)]
        med = stack[side][0]
        mm = med[_in_range(med.psi_deg, lo, hi)]
        for edge in ('inner', 'outer'):
            for s, gs in g.groupby(f'surface_{edge}'):
                p = gs.psi_deg.to_numpy()
                rows.append(dict(side=side, psi_range=f'{lo:.0f}-{hi:.0f}', edge=edge, surface=s,
                                 n_deg=len(gs), psi_min=float(p.min()), psi_max=float(p.max())))
            gap = (g[f'r_{edge}_danish_pix'] - g[f'r_{edge}_batoid_pix'])[~g.spider]
            rows.append(dict(side=side, psi_range=f'{lo:.0f}-{hi:.0f}', edge=edge,
                             surface='(danish − Batoid, median pixels)', n_deg=len(g),
                             psi_min=float(gap.median()), psi_max=float(gap.min()),
                             data_minus_fit_c_median_pix=float(mm[edge].median())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 3

def step3_pages(pdf):
    fits = pd.read_parquet(pc.PASS4_RUN / 'fits.parquet')
    pages = fits[(fits.fit == 'c') & (fits.pass3_id != '') & (fits.field_r_deg > 1.4)
                 & (fits.error == '')].sort_values(['field_r_deg', 'pass3_id', 'side'])
    affected = pd.read_csv(RUN / 'step3_affected.csv')
    keys = set(zip(affected.pass3_id, affected.side))
    n = 0
    for _, r in pages.iterrows():
        if (r.pass3_id, r.side) not in keys:
            continue
        g = fits[(fits.sample_id == r.sample_id) & (fits.side == r.side)].set_index('fit')
        rep4.page_donut_images(pdf, pc.PASS4_RUN, g)
        n += 1
    return n


def step3_audit():
    """Which pass 4 step 3 pages had collapsed panels: replay the committed pass 4 layout
    (git ``615e7d0``) on every page and measure the panel widths."""
    import importlib.util
    import subprocess
    import tempfile
    import warnings
    src = subprocess.run(['git', 'show', '615e7d0:aos/giant_donut_pupil/code/pupil_pass4_report.py'],
                         capture_output=True, text=True, check=True,
                         cwd=pc.p3.STUDY_DIR).stdout.replace('from report_utils', '# ')
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('rep4_old', f.name)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    fits = pd.read_parquet(pc.PASS4_RUN / 'fits.parquet')
    pages = fits[(fits.fit == 'c') & (fits.pass3_id != '') & (fits.field_r_deg > 1.4)
                 & (fits.error == '')]
    rows = []
    for _, r in pages.iterrows():
        d = np.load(pc.PASS4_RUN / 'images' / f'{r.sample_id}_{r.side}.npz')
        data = np.where(d['masked'], np.nan, d['img'])
        g = fits[(fits.sample_id == r.sample_id) & (fits.side == r.side)].set_index('fit')
        width = []

        class Probe:
            def savefig(self, fig, dpi=None):
                fig.canvas.draw()
                width.append(min(a.get_position().width for a in fig.axes[:6]))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old.page_donut_images(Probe(), pc.PASS4_RUN, g)
        rows.append(dict(pass3_id=r.pass3_id, sample_id=r.sample_id, side=r.side,
                         sensor_class=r.sensor_class, min_panel_width_frac=width[0],
                         finite_pixels=int(np.isfinite(data).sum()),
                         stamp_p99p5_e=float(np.nanpercentile(data, 99.5)),
                         collapsed=bool(width[0] < 0.05)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 4 (6c)

def page_profiles(pdf, t, prof, pid):
    g = t[t.pass3_id == pid]
    fig, axes = plt.subplots(2, 8, figsize=(22, 7), sharex=True)
    r0 = g.iloc[0]
    for i, side in enumerate(('intra', 'extra')):
        s = g[g.side == side]
        if s.empty:
            for ax in axes[i]:
                empty_panel(ax, f'no {side}-focal side in the sample')
            continue
        used = set()
        for _, r in s.sort_values('psi_num').iterrows():
            if r.psi_num < 0:
                continue
            ax = axes[i, int(r.psi_num)]
            used.add(int(r.psi_num))
            key = f'{r.key}:{int(r.model_idx)}'
            have = False
            for name, col in (('data', 'k'), ('model', 'tab:green')):
                if f'{key}:{name}_off' in prof:
                    ax.plot(prof[f'{key}:{name}_off'], prof[f'{key}:{name}_flux'], '-', color=col,
                            lw=1.0, label='data' if name == 'data' else 'fit c')
                    have = True
            if not have:
                empty_panel(ax, r.missing or 'profile not saved')
            elif r.missing:
                ax.text(0.5, 0.05, 'MISSING: ' + r.missing, transform=ax.transAxes, ha='center',
                        fontsize=6, color='tab:red')
            ax.set_title(f'{side} psi# {int(r.psi_num)}, vane {int(r.vane_id)}\n'
                         f'depth {r.depth_data:.2f}/{r.depth_model:.2f}, FWHM '
                         f'{r.fwhm_pix_data:.1f}/{r.fwhm_pix_model:.1f} pix\n'
                         f'b {r.b_pix_data:+.1f}/{r.b_pix_model:+.1f} pix', fontsize=7,
                         color=VANE_COLOURS(int(r.vane_id)) if r.vane_id >= 0 else 'k')
            ax.tick_params(labelsize=6)
        for j in set(range(8)) - used:
            empty_panel(axes[i, j], 'shadow not numbered: no model or data azimuth at r_half')
        axes[i, 0].set_ylabel(f'{side}: flux (electrons per pixel)', fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('offset across the vane (pixels)', fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f'6c (pass 5, guided tracer). {pid} {r0.detector}, field radius {r0.field_r_deg:.3f} '
                 f'deg, OCS azimuth {r0.field_az_ocs_deg:.1f} deg: spider shadow profiles along '
                 'their chords; black data, green fit c. Labels data/model', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=70)
    plt.close(fig)


def step4_counts():
    o = pd.read_parquet(pc.PASS4_RUN / 'spiders.parquet')
    o['missing'] = o.missing.fillna('')
    prof4 = set(np.load(pc.PASS4_RUN / 'spider_profiles.npz').files)
    s = o[o.set == 'sample'].copy()
    s['data_profile_saved'] = [f'sample:{r.sample_id}:{r.side}:{int(r.model_idx)}:data_off' in prof4
                               for r in s.itertuples()]
    s['panel'] = np.where(s.psi_num < 0, 'blank, no title (psi_num -1, not plotted)',
                          np.where(~s.data_profile_saved, 'model profile only', 'data and model'))
    before = s.groupby(['missing', 'panel']).size().rename('n_shadows').reset_index()
    before['pages'] = [s[(s.missing == a) & (s.panel == b)].pass3_id.nunique()
                       for a, b in zip(before.missing, before.panel)]
    t = pd.read_parquet(RUN / 'spiders.parquet')
    a = t[(t.centre == 'fit_a') & t.in_sample]
    after = a.groupby('missing').size().rename('n_shadows').reset_index()
    after['pages'] = [a[a.missing == m].pass3_id.nunique() for m in after.missing]
    a5 = t[t.centre == 'fit_a']
    b4 = o[o.set == 'bands']
    totals = pd.DataFrame([
        dict(set='pass 4, bands (180 sides)', shadows=len(b4), found=int((b4.missing == '').sum())),
        dict(set='pass 4, sample (49 sides)', shadows=len(s), found=int((s.missing == '').sum())),
        dict(set='pass 5, all distinct (188 sides)', shadows=len(a5),
             found=int((a5.missing == '').sum())),
        dict(set='pass 5, sample (49 sides)', shadows=len(a), found=int((a.missing == '').sum()))])
    return before, after, totals, b4.groupby('missing').size()


# ---------------------------------------------------------------- step 6

def page_delta(pdf, rf, rs, dt, t, rt):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for side, col in SIDE_COLOURS.items():
        g = rf[rf.side == side]
        for cls, m in (('science', 'o'), ('SW0', 's'), ('SW1', 'D')):
            s = g[g.sensor_class == cls]
            axes[0, 0].errorbar(s.field_r_deg, s.delta_best_deg, yerr=s.delta_err_deg, fmt=m,
                                color=col, ms=4, lw=0.5, mfc='none' if cls != 'science' else col,
                                label=f'{side} {cls}')
            axes[0, 1].plot(s.field_az_ocs_deg, s.delta_best_deg, m, color=col, ms=4,
                            mfc='none' if cls != 'science' else col)
        r = rs.set_index('group').loc[side]
        for ax in axes[0]:
            ax.axhline(r.delta_median_deg, color=col, lw=1)
        e = rt.set_index('side').loc[side].efd_minus_rtp_deg
        for ax in axes[0]:
            ax.axhline(e, color=col, lw=1, ls='--')
        axes[1, 0].hist(g.delta_best_deg, bins=np.arange(-0.3, 0.6, 0.02), histtype='step',
                        color=col, label=f'{side}: median {r.delta_median_deg:+.3f}, nMAD '
                        f'{r.delta_nmad_deg:.3f} deg, n {int(r.n_sides)}; EFD − rtp {e:+.3f} deg')
    axes[0, 0].set_xlabel('field radius (deg)')
    axes[0, 1].set_xlabel('field azimuth, OCS (deg)')
    for ax in axes[0]:
        ax.set_ylabel('best spider-angle offset delta (deg)')
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].set_xlabel('best delta (deg)')
    axes[1, 0].set_ylabel('sides')
    axes[1, 0].legend(fontsize=7)
    # sum of squares against delta, median-normalised, a few sides
    m = dt.merge(t[(t.centre == 'fit_a') & (t.missing == '')][['key', 'model_idx', 'side',
                                                               'az_half_ocs_deg_data']],
                 on=['key', 'model_idx'])
    m['res'] = pc.wrap(m.az_half_ocs_deg_data - m.az_half_ocs_deg_model)
    for side, col in SIDE_COLOURS.items():
        g = m[m.side == side]
        rms = g.groupby(['key', 'delta_deg']).res.apply(lambda r: np.sqrt(np.mean(r ** 2)))
        med = rms.groupby('delta_deg').median()
        axes[1, 1].plot(med.index, med.values, '-', color=col, label=f'{side}: median over sides')
    axes[1, 1].set_xlabel('delta (deg): danish spider_angle = rtp + delta')
    axes[1, 1].set_ylabel('RMS over shadows of data − model azimuth at r_half (deg)')
    axes[1, 1].legend(fontsize=7)
    fig.suptitle('Step 6a: spider-angle offset delta per side. Solid lines: median delta; dashed: '
                 'EFD MTRotator actualPosition minus the visitInfo rtp the fits use', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_m1(pdf, a, m1):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    cols = (('az_half_ocs_deg_dmm', 'at rtp (as pass 4)'),
            ('dmm_best', 'at rtp + per-side best delta'),
            ('dmm_common', 'at rtp + one delta per side of focus'),
            ('dmm_trans', 'after per-side rotation + translation'))
    grid = np.linspace(0, 360, 361)
    for i, side in enumerate(('extra', 'intra')):
        g = a[a.side == side]
        for j, (c, lab) in enumerate(cols):
            ax = axes[i, j]
            ax.plot(g.az_half_ocs_deg_model, g[c], '.', ms=2, color=SIDE_COLOURS[side], alpha=0.4)
            f = m1[(m1.side == side) & (m1.quantity == c)].iloc[0]
            ax.plot(grid, f.common + f.amp * np.cos(np.deg2rad(grid - f.phase_deg)), 'k-', lw=1)
            ax.axhline(0, color='0.6', lw=0.5)
            ax.set_title(f'{side}, {lab}\ncommon {f.common:+.3f} deg, m = 1 amplitude {f.amp:.3f} '
                         f'deg at {f.phase_deg:.0f} deg; scatter nMAD {f.resid_nmad:.3f} deg',
                         fontsize=8)
            ax.set_xlim(0, 360)
            ax.set_ylim(-1.5, 1.2)
        axes[i, 0].set_ylabel('shadow azimuth, data − model (deg)')
    for ax in axes[1]:
        ax.set_xlabel('shadow azimuth, OCS image (deg)')
    fig.suptitle('Step 6a/b: data − model shadow azimuth at r_half against azimuth (Huber fit of '
                 'common + m = 1). A rotation removes only the common term; a translation of the '
                 'shadows removes the m = 1', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_translation(pdf, tf, ct):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for side, col in SIDE_COLOURS.items():
        for cen, m in (('fit_a', 'o'), ('outer', 's'), ('inner', '^')):
            g = tf[(tf.side == side) & (tf.centre == cen)]
            axes[0].plot(g.dx_pix, g.dy_pix, m, color=col, ms=3, alpha=0.5 if cen != 'fit_a' else 0.8,
                         mfc='none' if cen != 'fit_a' else col, label=f'{side}, {cen} centre')
        g = tf[(tf.side == side) & (tf.centre == 'fit_a')]
        axes[1].plot(g.field_r_deg, g.dy_pix, 'o', color=col, ms=3, label=side)
    axes[0].axhline(0, color='0.6', lw=0.5)
    axes[0].axvline(0, color='0.6', lw=0.5)
    axes[0].set_xlabel('shadow-pattern translation dx, OCS image (pixels)')
    axes[0].set_ylabel('dy (pixels)')
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel('field radius (deg)')
    axes[1].set_ylabel('dy, fit-a centre (pixels)')
    axes[1].legend(fontsize=7)
    x = np.arange(3)
    for k, side in enumerate(('extra', 'intra')):
        c = ct[ct.side == side].set_index('centre').loc[['fit_a', 'outer', 'inner']]
        axes[2].bar(x + 0.2 * k - 0.3, c.m1_amp_deg, 0.2, color=SIDE_COLOURS[side],
                    label=f'{side}: m = 1 amplitude (deg)')
        axes[2].bar(x + 0.2 * k + 0.1, c.inner_edge_dy_ocs_pix / 10, 0.2, color=SIDE_COLOURS[side],
                    alpha=0.4, label=f'{side}: inner-edge dy / 10 (pixels / 10)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['fit-a centre', 'outer-edge centre', 'inner-edge centre'])
    axes[2].legend(fontsize=7)
    axes[2].set_title('Centre test: spider m = 1 amplitude, and the data − model inner-edge circle '
                      'centre dy\nrelative to that centre', fontsize=8)
    fig.suptitle('Step 6b/c: per-side translation (dx, dy) of the shadow pattern relative to the '
                 'donut centre, Huber fit with a common rotation', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


# ---------------------------------------------------------------- step 7

def page_blur(pdf, bf, bs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for side, col in SIDE_COLOURS.items():
        g = bf[bf.side == side]
        axes[0, 0].plot(g.mag, g.fwhm_fit_arcsec, 'o', color=col, ms=4,
                        label=f'{side}: median {g.fwhm_fit_arcsec.median():.3f}, nMAD '
                        f'{nmad(g.fwhm_fit_arcsec.to_numpy()):.3f} arcsec, n {len(g)}')
        axes[0, 1].plot(g.chi2_dof_fixed, g.chi2_dof_free / g.chi2_dof_fixed, 'o', color=col, ms=4)
    axes[0, 0].set_xlabel('magnitude (AB)')
    axes[0, 0].set_ylabel('fitted blur FWHM (arcsec)')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('chi2/dof, blur 1.0 arcsec')
    axes[0, 1].set_ylabel('chi2/dof free blur over chi2/dof blur 1.0 (dimensionless)')
    s = bs[bs.missing == ''].copy()
    s['ratio'] = s.fwhm_pix_data / s.fwhm_pix_model
    for k, (q, lab) in enumerate((('ratio', 'shadow FWHM, data over model (dimensionless)'),
                                  ('depth_model', 'shadow depth (dimensionless)'))):
        ax = axes[1, k]
        pos = 0
        ticks = []
        for side in ('extra', 'intra'):
            for model in ('blur_1.0', 'blur_free'):
                v = s[(s.side == side) & (s.model == model)][q].dropna()
                ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False)
                ticks.append(f'{side}\n{model}')
                if q == 'depth_model':
                    dv = s[(s.side == side) & (s.model == model)].depth_data.dropna()
                    ax.plot(pos + 0.35, dv.median(), 'k*', ms=9)
                pos += 1
        ax.set_xticks(range(pos))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_ylabel(lab)
    axes[1, 1].set_title('boxes: model depth; star: data median depth', fontsize=8)
    axes[1, 0].axhline(1, color='0.6', lw=0.5)
    fig.suptitle('Step 7: fit c with the blur free (0.5-2.5 arcsec) on the central-band science '
                 'sides; shadow width and depth against each model', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def blur_table(bf, bs):
    s = bs[bs.missing == ''].copy()
    rows = []
    for (side, model), g in s.groupby(['side', 'model']):
        f = bf[bf.side == side]
        rows.append(dict(side=side, model=model, n_shadows=len(g),
                         fwhm_fit_arcsec_median=f.fwhm_fit_arcsec.median() if model == 'blur_free'
                         else 1.0,
                         fwhm_fit_arcsec_nmad=nmad(f.fwhm_fit_arcsec.to_numpy())
                         if model == 'blur_free' else 0.0,
                         width_data_pix=g.fwhm_pix_data.median(),
                         width_model_pix=g.fwhm_pix_model.median(),
                         width_ratio_median=(g.fwhm_pix_data / g.fwhm_pix_model).median(),
                         depth_data=g.depth_data.median(), depth_model=g.depth_model.median(),
                         depth_dmm=(g.depth_data - g.depth_model).median(),
                         chi2_dof_median=(f.chi2_dof_free if model == 'blur_free'
                                          else f.chi2_dof_fixed).median()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 5 gallery

def gallery_sides(t):
    o = pd.read_parquet(pc.PASS4_RUN / 'spiders.parquet')
    o['missing'] = o.missing.fillna('')
    S = pd.read_parquet(RUN / 'sides.parquet')
    # the pass 4 sample set keys its sides by the pass 3 id
    to_sid = dict(zip(S.pass3_id, S.sample_id))
    to_sid.update(zip(S.sample_id, S.sample_id))
    o['key'] = o.sample_id.map(to_sid) + ':' + o.side
    a = t[t.centre == 'fit_a']
    miss = set(o[o.missing != ''].key) | set(a[a.missing != ''].key)
    full = a.groupby('key').filter(lambda g: (g.missing == '').all() and g.key.iloc[0] not in miss)
    f = full.drop_duplicates('key')
    picks = []
    for cls, band, pick in (('science', 'central', 'min'), ('science', 'central', 'median'),
                            ('science', 'outer', 'max'), ('science', 'outer', 'median'),
                            ('science', 'outer', 'min'),
                            ('SW0', 'outer', 'any'), ('SW1', 'outer', 'any')):
        # no SW1 side has all 8 shadows found in both passes
        g = f[(f.sensor_class == cls) & ((f.band == band) | (cls != 'science'))]
        if g.empty:
            continue
        g = g.sort_values('field_r_deg')
        r = g.iloc[{'min': 0, 'max': -1, 'median': len(g) // 2, 'any': 0}[pick]]
        picks.append(r.sample_id)
    return miss, picks


def _chord_xy(line, r_edge, cx, cy):
    px, py, dx, dy = line[:4]
    if not np.isfinite(dx):
        return None
    d = np.array([dx, dy])
    p = np.array([px - cx, py - cy])
    if p @ d < 0:
        d = -d
    s = np.linspace(-r_edge, r_edge, 400)
    q = p[None, :] + s[:, None] * d[None, :]
    r = np.hypot(q[:, 0], q[:, 1])
    k = (r >= 0.62 * r_edge) & (r <= 0.98 * r_edge) & (q @ d > 0)
    return q[k, 0] + cx, q[k, 1] + cy


def gallery_page(pdf, S, t, geom, sid, reason):
    rows = S[S.sample_id == sid].sort_values('side')
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 7 * len(rows)), squeeze=False)
    for i, (_, row) in enumerate(rows.iterrows()):
        key = row.key
        if f'{key}:meta' not in geom:
            for ax in axes[i]:
                empty_panel(ax, 'side not traced in pass 5')
            continue
        cx, cy, r_edge, _ = geom[f'{key}:meta']
        d = pc.load_side(row)
        sh = t[(t.key == key) & (t.centre == 'fit_a')].set_index('model_idx')
        top = np.nanpercentile(np.where(d['masked'], np.nan, d['img']), 95)
        for j, (name, im) in enumerate((('data', d['img']), ('fit c model', d['model_c']))):
            ax = axes[i, j]
            ax.imshow(np.where(d['masked'], 0, im), origin='lower', cmap='gray', vmin=0,
                      vmax=1.2 * top)
            for k in range(pc.N_SHADOW):
                for who, col in (('data', 'tab:red'), ('model', 'cyan')):
                    xy = _chord_xy(geom[f'{key}:{k}:{who}_line'], r_edge, cx, cy)
                    if xy is not None:
                        ax.plot(*xy, '-', color=col, lw=0.8)
                    c = geom[f'{key}:{k}:{who}_centres']
                    ax.plot(c[:, 0], c[:, 1], 'x' if who == 'data' else '+', color=col, ms=4,
                            mew=0.7)
                r = sh.loc[k]
                az = r.az_half_ocs_deg_model - row.rtp_deg
                rl = 1.06 * r_edge
                txt = f'psi# {int(r.psi_num)}\nvane {int(r.vane_id)}' + (
                    f'\n{r.missing}' if r.missing else '')
                ax.text(cx + rl * np.cos(np.deg2rad(az)), cy + rl * np.sin(np.deg2rad(az)), txt,
                        fontsize=6, ha='center', va='center',
                        color='tab:red' if r.missing else 'yellow')
            rh = np.nanmedian(sh.r_half_pix_model)
            ph = np.linspace(0, 2 * np.pi, 361)
            ax.plot(cx + rh * np.cos(ph), cy + rh * np.sin(ph), ':', color='yellow', lw=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            nf = int((sh.missing == '').sum())
            ax.set_title(f'{row.label} {row.side}-focal {row.detector}, field r {row.field_r_deg:.3f} '
                         f'deg: {name}. {nf} of 8 shadows found\nchords: red data, cyan fit c; '
                         'x / + traced centres at 7 radii; dotted: r_half', fontsize=8)
    fig.suptitle(f'Step 5: spider trajectories, {rows.iloc[0].label} ({reason})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- driver

def main():
    S = pd.read_parquet(RUN / 'sides.parquet')
    t = pd.read_parquet(RUN / 'spiders.parquet')
    dt = pd.read_parquet(RUN / 'spider_delta.parquet')
    geom = dict(np.load(RUN / 'spider_geometry.npz'))
    prof = dict(np.load(RUN / 'spider_profiles.npz'))
    rt = pd.read_parquet(RUN / 'rtp_check.parquet')
    an = pa.run_all(RUN)
    vb = pd.read_parquet(RUN / 'vignetting_bins.parquet')
    vr = pd.read_parquet(RUN / 'vignetting_ranges.parquet')
    imgs = dict(np.load(RUN / 'vignetting_images.npz'))
    bf = pd.read_parquet(RUN / 'blur_fits.parquet')
    bs = pd.read_parquet(RUN / 'blur_spiders.parquet')

    audit = step3_audit()
    audit.to_csv(RUN / 'step3_audit.csv', index=False, float_format='%.4g')
    audit[audit.collapsed].to_csv(RUN / 'step3_affected.csv', index=False)
    before, after, totals, b4reasons = step4_counts()
    for name, df in (('step4_before', before), ('step4_after', after), ('step4_totals', totals)):
        pa._save(df, RUN, name)
    stack = psi_stack_pass4()
    rtab = range_table(vb, stack)
    pa._save(rtab, RUN, 'vignetting_aaron_ranges')
    btab = blur_table(bf, bs)
    pa._save(btab, RUN, 'blur_summary')
    pd.set_option('display.width', 200)
    with PdfPages(RUN / 'pupil_pass5.pdf') as pdf:
        text_page(pdf, 'Step 2: vignetting surface per psi range, per edge and side (Batoid v4.02 '
                  'full model, field 1.7 deg at 45 deg, science sensor). psi in deg',
                  vr.to_string(index=False), 7)
        text_page(pdf, "Step 2: Aaron's ranges, psi 330-30 deg intra and 160-200 deg extra: "
                  'surface per edge (n_deg: 1 deg bins; psi_min/max deg). Rows "(danish − '
                  'Batoid)": psi_min = median and psi_max = minimum of the danish v4.02g mask edge '
                  'minus the Batoid edge (pixels), and the pass 4 median data − fit c (pixels)',
                  rtab.round(2).to_string(index=False), 8)
        for side in ('intra', 'extra'):
            page_vignetting(pdf, vb, imgs, stack, side)
        text_page(pdf, 'Step 3: pass 4 step 3 image pages, audit of the old layout (panel width as '
                  'a fraction of the page; collapsed under 0.05)', audit.round(4).to_string(
                      index=False) + f'\n\nCollapsed: {int(audit.collapsed.sum())} of {len(audit)}. '
                  'Stamps all present, finite and with a positive 99.5th percentile, so the cause '
                  'is the layout: the fit text was placed as axes text below panel (0, 0), wider '
                  'than the axes, and tight_layout shrank every panel to zero width. Fixed in '
                  '`pupil_pass4_report.page_donut_images`: the text goes in reserved figure space.',
                  7)
        n3 = step3_pages(pdf)
        text_page(pdf, 'Step 4 (6c): spider shadows, pass 4 sample set per reason and what the 6c '
                  'panel showed (n_shadows; pages = 6c pages with at least one)',
                  before.to_string(index=False) + '\n\nPass 4, bands set, missing by reason:\n'
                  + b4reasons.to_string() + '\n\nAfter (pass 5 guided tracer, sample sides):\n'
                  + after.to_string(index=False) + '\n\nTotals:\n' + totals.to_string(index=False), 8)
        samp = t[(t.centre == 'fit_a') & t.in_sample]
        for pid in samp.drop_duplicates('pass3_id').sort_values('field_r_deg').pass3_id:
            page_profiles(pdf, samp, prof, pid)
        text_page(pdf, 'Step 6a: spider-angle offset delta (deg), danish spider_angle = rtp + delta, '
                  'per side; summaries', an['rs'].round(3).T.to_string() + '\n\nrtp check (deg):\n'
                  + rt.round(4).T.to_string() + '\n\nPer vane id (deg): data − model at rtp, at '
                  'rtp + best delta, and after rotation + translation\n'
                  + an['vr'].round(3).to_string(index=False), 7)
        page_delta(pdf, an['rf'], an['rs'], dt, t, rt)
        page_m1(pdf, an['a'], an['m1'])
        text_page(pdf, 'Step 6b/c: translation (OCS image pixels) and rotation (deg) of the shadow '
                  'pattern per side, by centre definition; m = 1 summary and centre test',
                  an['ts'].round(3).to_string(index=False) + '\n\n' + an['m1'].round(3).to_string(
                      index=False) + '\n\nCentre test (deg; edge offsets data − model in OCS image '
                  'pixels relative to the centre used):\n' + an['ct'].round(3).T.to_string(), 7)
        page_translation(pdf, an['tf'], an['ct'])
        text_page(pdf, 'Step 7: free-blur fit c, central-band science sides',
                  btab.round(3).T.to_string(), 8)
        page_blur(pdf, bf, bs)
    miss, picks = gallery_sides(t)
    donuts = sorted({k.split(':')[0] for k in miss})
    order = S.drop_duplicates('sample_id').set_index('sample_id').field_r_deg
    with PdfPages(RUN / 'gallery_spiders.pdf') as pdf:
        for sid in sorted(donuts, key=lambda s: order.get(s, 9)):
            gallery_page(pdf, S, t, geom, sid, 'a shadow missing in pass 4 or pass 5')
        for sid in picks:
            gallery_page(pdf, S, t, geom, sid, 'all 8 found in both passes')
    print('step 3 pages', n3, 'gallery donuts', len(donuts), '+', len(picks), flush=True)


if __name__ == '__main__':
    main()
