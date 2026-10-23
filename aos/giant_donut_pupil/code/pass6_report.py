"""Pupil pass 6 report: ``pupil_pass6.pdf``, sections 1-5 plus summary pages.

Reads only the pass 6 run directory (tables and the small sim model arrays) and the pass 5
summary tables, so it reruns without refitting. It also runs the step 4b silhouette fit to
the data (`pass6_silhouette.data_fit`, two Batoid traces) and writes its table, plus the
sim-against-data tables, as parquet and csv.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass6_report.py
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
import pass6_common as p6  # noqa: E402
import pass5_common as pc  # noqa: E402
import pass6_pipeline as pp  # noqa: E402
import pass5_analysis as pa  # noqa: E402
import pupil_pass4_report as rep4  # noqa: E402
from pass5_report import _chord_xy  # noqa: E402
from report_utils import empty_panel  # noqa: E402

from rubinwork.common.utils import nmad  # noqa: E402

RUN = p6.RUN
SIM = RUN / 'sims'
SIDE_COLOURS = rep4.SIDE_COLOURS
DATA_CFG = 'isr_phys'
SIM_CFGS = ('nominal_v402g', 'nominal_v402s', 'rot20_v402g', 'tea_y10_v402g')
PSI_RANGES = {'intra': (330.0, 30.0), 'extra': (160.0, 200.0)}
PSI_BAND_DEG = (1.6, 1.725)
ROT_DRAWN_DEG = 0.20
TEA_EXPECTED_PIX = 0.88


def text_page(pdf, title, text, fontsize=7):
    rep4.text_page(pdf, title, text, fontsize)


def _in_range(psi, lo, hi):
    return (psi >= lo) | (psi < hi) if lo > hi else (psi >= lo) & (psi < hi)


def err_median(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return 1.2533 * nmad(x) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def load(cfg):
    d = RUN / cfg if not cfg.startswith(SIM_CFGS) else SIM / cfg
    out = {n: pd.read_parquet(d / f'{n}.parquet') for n in
           ('fits', 'spiders', 'sides', 'centres', 'bins', 'translation_summary', 'centre_test')}
    for n in ('m1_summary', 'rotation_summary', 'rotation_fit', 'vane_residuals'):
        if (d / f'{n}.parquet').exists():
            out[n] = pd.read_parquet(d / f'{n}.parquet')
    return out


# ---------------------------------------------------------------- tables

def effect_row(name, T, keys=None):
    """Common translation, m = 1 amplitude, edge-circle shifts and ratio excess per side."""
    rows = []
    sp = T['spiders']
    f = T['fits']
    if keys is not None:
        sp, f = sp[sp.key.isin(keys)], f[f.key.isin(keys)]
    tf = pa.translation_fit(sp)
    ct, _ = pa.centre_test(sp, T['centres'][T['centres'].key.isin(sp.key)],
                           T['sides'][T['sides'].key.isin(sp.key)])
    ct = ct[ct.centre == 'fit_a'].set_index('side')
    for side in ('extra', 'intra'):
        g = tf[(tf.side == side)]
        c = ct.loc[side]
        h = f[f.side == side]
        rows.append(dict(config=name, side=side, n_sides=len(g),
                         rot_deg=np.median(g.rot_deg), dx_pix=np.median(g.dx_pix),
                         dx_err_pix=err_median(g.dx_pix), dy_pix=np.median(g.dy_pix),
                         dy_err_pix=err_median(g.dy_pix), m1_amp_deg=c.m1_amp_deg,
                         m1_phase_deg=c.m1_phase_deg, inner_dy_pix=c.inner_edge_dy_ocs_pix,
                         outer_dy_pix=c.outer_edge_dy_ocs_pix,
                         ratio_excess=np.median(h.ratio_excess),
                         chi2_dof_c=np.median(h.chi2_dof)))
    return rows


def per_key_m1(sp):
    a = pa.found(sp)
    rows = []
    for key, g in a.groupby('key'):
        if len(g) < pa.MIN_SHADOWS:
            continue
        m = pc.m1_fit(g.az_half_ocs_deg_dmm, g.az_half_ocs_deg_model)
        rows.append(dict(key=key, side=g.side.iloc[0], field_r_deg=g.field_r_deg.iloc[0],
                         sensor_class=g.sensor_class.iloc[0], m1_amp_deg=m['amp'],
                         m1_phase_deg=m['phase_deg'], common_deg=m['common']))
    return pd.DataFrame(rows)


def per_vane(sp, keys=None):
    a = pa.found(sp)
    if keys is not None:
        a = a[a.key.isin(keys)]
    return a.groupby(['side', 'vane_id']).agg(
        n=('key', 'count'), pupil_az_deg=('pupil_az_deg', 'median'),
        dmm_median_deg=('az_half_ocs_deg_dmm', 'median'),
        dmm_nmad_deg=('az_half_ocs_deg_dmm', nmad)).reset_index()


def tea_check(nom, tea):
    """TEA +10 mm: shadow shift in the image (tea − nominal data shadows), per side."""
    import statsmodels.api as sm
    k = ['key', 'model_idx']
    n = nom[nom.centre == 'fit_a'].set_index(k)
    r = tea[tea.centre == 'fit_a'].set_index(k)
    j = n.join(r, rsuffix='_t', how='inner').reset_index()
    j = j[(j.missing == '') & (j.missing_t == '')]
    rows = []
    for key, g in j.groupby('key'):
        az = np.deg2rad(g.az_half_ocs_deg_data.to_numpy())
        rh = g.r_half_pix_data.to_numpy()
        y = np.deg2rad(pc.wrap(g.az_half_ocs_deg_data_t - g.az_half_ocs_deg_data))
        X = np.column_stack([np.ones_like(az), -np.sin(az) / rh, np.cos(az) / rh])
        p = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit().params
        rows.append(dict(key=key, side=g.side.iloc[0], rot_deg=np.degrees(p[0]),
                         dx_about_center_pix=p[1], dy_about_center_pix=p[2]))
    return pd.DataFrame(rows)


def checks_table(T):
    """rot20 and tea_y10 checks against the nominal sims, per side."""
    nom, rot, tea = T['nominal_v402g'], T['rot20_v402g'], T['tea_y10_v402g']
    k = ['key', 'model_idx']
    n = nom['spiders'][nom['spiders'].centre == 'fit_a'].set_index(k)
    r = rot['spiders'][rot['spiders'].centre == 'fit_a'].set_index(k)
    j = n.join(r, rsuffix='_r', how='inner')
    j['shadow_shift_deg'] = pc.wrap(j.az_half_ocs_deg_data_r - j.az_half_ocs_deg_data)
    j['model_shift_deg'] = pc.wrap(j.az_half_ocs_deg_model_r - j.az_half_ocs_deg_model)
    tc = tea_check(nom['spiders'], tea['spiders'])
    cn = nom['centres'].set_index('key')
    ctea = tea['centres'].set_index('key')
    dcen = (ctea.cy_fit_a - cn.cy_fit_a)
    rf = rot['rotation_fit']
    rows = []
    for side in ('extra', 'intra'):
        g = j[j.side == side]
        d = rf[rf.side == side].delta_best_deg
        t = tc[tc.side == side]
        c = dcen[dcen.index.str.endswith(side)]
        # center motion is in CCS; rtp is about 0, so CCS y is OCS y to 0.01 deg
        img = t.set_index('key').dy_about_center_pix + c.reindex(t.key).to_numpy()
        rows.append(dict(side=side, rot_drawn_deg=ROT_DRAWN_DEG,
                         shadow_shift_in_sim_deg=np.median(g.shadow_shift_deg),
                         fit_c_model_shadow_shift_deg=np.median(g.model_shift_deg),
                         delta_median_deg=np.median(d), delta_err_deg=err_median(d),
                         delta_recovery=np.median(d) / ROT_DRAWN_DEG,
                         tea_dy_about_center_pix=np.median(t.dy_about_center_pix),
                         tea_center_dy_pix=np.median(c),
                         tea_dy_image_pix=np.median(img), tea_dy_image_err_pix=err_median(img),
                         tea_expected_pix=-TEA_EXPECTED_PIX if side == 'extra' else TEA_EXPECTED_PIX))
    return pd.DataFrame(rows)


def psi_stack(T, mask_label):
    """Science sides at 1.6-1.725 deg: data − model inner and outer edge per 1 deg psi bin."""
    b = T['bins'].merge(T['sides'][['key', 'field_az_ocs_deg', 'field_r_deg', 'sensor_class']],
                        on='key').merge(T['fits'][['key', 'fit_ok']], on='key')
    b = b[(b.sensor_class == 'science') & b.field_r_deg.between(*PSI_BAND_DEG) & b.fit_ok
          & ~b.spider & ~b.vignetted].copy()
    b['psi_deg'] = np.mod(b.az_ocs_deg - (b.field_az_ocs_deg + 180.0), 360.0)
    b['inner_dmm'] = b.r_inner_data_pix - b.r_inner_c_pix
    b['outer_dmm'] = b.r_outer_data_pix - b.r_outer_c_pix
    b['pbin'] = b.psi_deg.astype(int)
    st = b.groupby(['side', 'pbin']).agg(n=('key', 'count'), inner_dmm=('inner_dmm', 'median'),
                                         outer_dmm=('outer_dmm', 'median')).reset_index()
    st['psi_deg'] = st.pbin + 0.5
    st['mask'] = mask_label
    rows = []
    for side, (lo, hi) in PSI_RANGES.items():
        g = b[b.side == side]
        inr = _in_range(g.psi_deg, lo, hi)
        per = g[inr].groupby('key').inner_dmm.median()
        rows.append(dict(mask=mask_label, side=side, psi_range=f'{lo:.0f}-{hi:.0f}',
                         n_sides=len(per), inner_in_range_pix=np.median(per),
                         inner_in_range_err_pix=err_median(per),
                         inner_outside_pix=np.median(g[~inr].inner_dmm),
                         outer_in_range_pix=np.median(g[inr].outer_dmm)))
    return st, pd.DataFrame(rows)


# ---------------------------------------------------------------- pages

def page_effects(pdf, eff, title):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    qs = (('dy_pix', 'common translation dy, OCS image (pixels)'),
          ('m1_amp_deg', 'spider m = 1 amplitude (deg)'),
          ('inner_dy_pix', 'inner-edge circle dy, data − model (pixels)'),
          ('ratio_excess', 'inner/outer ratio excess (dimensionless)'))
    names = list(dict.fromkeys(eff.config))
    x = np.arange(len(names))
    for ax, (q, lab) in zip(axes, qs):
        for k, side in enumerate(('extra', 'intra')):
            g = eff[eff.side == side].set_index('config').reindex(names)
            ax.bar(x + 0.4 * k - 0.2, g[q], 0.4, color=SIDE_COLOURS[side], label=side)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_delta(pdf, rf, rs, corr):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for side, col in SIDE_COLOURS.items():
        g = rf[rf.side == side]
        r = rs.set_index('group').loc[side]
        for cls, m in (('science', 'o'), ('SW0', 's'), ('SW1', 'D')):
            s = g[g.sensor_class == cls]
            axes[0].errorbar(s.field_r_deg, s.delta_best_deg, yerr=s.delta_err_deg, fmt=m, color=col,
                             ms=3, lw=0.4, mfc='none' if cls != 'science' else col,
                             label=f'{side} {cls}')
            axes[1].plot(s.field_az_ocs_deg, s.delta_best_deg, m, color=col, ms=3,
                         mfc='none' if cls != 'science' else col)
        axes[2].hist(g.delta_best_deg, bins=np.arange(-0.4, 0.5, 0.02), histtype='step', color=col,
                     label=f'{side}: median {r.delta_median_deg:+.3f} deg, nMAD '
                     f'{r.delta_nmad_deg:.3f} deg, n {int(r.n_sides)}\n  corrected for the '
                     f'rot20 recovery: {corr[side]:+.3f} deg')
        for ax in axes[:2]:
            ax.axhline(r.delta_median_deg, color=col, lw=1)
    axes[0].set_xlabel('field radius (deg)')
    axes[1].set_xlabel('field azimuth, OCS (deg)')
    for ax in axes[:2]:
        ax.set_ylabel('best spider-angle offset delta (deg)')
    axes[0].legend(fontsize=7, ncol=2)
    axes[2].set_xlabel('best delta (deg)')
    axes[2].set_ylabel('sides')
    axes[2].legend(fontsize=7)
    fig.suptitle('Step 2: delta per side with rtp = physical_rotator_angle (danish spider_angle = '
                 'rtp + delta)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_sim_side(pdf, row, fits, sps):
    """One sim side: the sim, its danish fit c with v4.02g and with v4.02s, both residuals
    on one color scale, and the data stamp of the same side (not fitted here)."""
    key = row.key
    zs = {m: SIM / 'stamps' / f'{row.sample_id}_{row.side}_nominal_{m}_model.npz'
          for m in ('v4.02g', 'v4.02s')}
    fig, axes = plt.subplots(2, 3, figsize=(19, 15.5))
    if not all(z.exists() for z in zs.values()):
        for ax in axes.ravel():
            empty_panel(ax, f'no model arrays for {key}')
        pdf.savefig(fig, dpi=70)
        plt.close(fig)
        return
    d = {m: dict(np.load(z)) for m, z in zs.items()}
    data = np.load(RUN / 'isr_stamps' / f'{row.sample_id}_{row.side}.npz')
    msk = data['masked'].astype(bool)
    sim = d['v4.02g']['img']
    top = np.nanpercentile(np.where(msk, np.nan, sim), 95)
    ring = ~msk & (d['v4.02g']['model'] > 0.1 * top)
    noise = pp.noise_of(sim, msk)
    res = {m: sim - d[m]['model'] for m in d}
    lim = 5 * noise

    def gray(ax, im):
        ax.imshow(np.where(msk, 0, im), origin='lower', cmap='gray', vmin=0, vmax=1.2 * top)

    def chords(ax, m, who_list):
        cx, cy = d[m]['center']
        r_edge = float(d[m]['r_edge'])
        for k in range(pc.N_SHADOW):
            for who, col in who_list:
                xy = _chord_xy(d[m][f'line_{who}_{k}'], r_edge, cx, cy)
                if xy is not None:
                    ax.plot(*xy, '-', color=col, lw=0.8)

    gray(axes[0, 0], sim)
    chords(axes[0, 0], 'v4.02g', (('data', 'tab:red'),))
    axes[0, 0].set_title('Batoid v4.02 design simulation (Poisson noise); red: traced shadows',
                         fontsize=9)
    for col, m in ((1, 'v4.02g'), (2, 'v4.02s')):
        f = fits[m].loc[key]
        sh = sps[m][(sps[m].key == key) & (sps[m].centre == 'fit_a')]
        dmm = sh.az_half_ocs_deg_dmm.dropna()
        gray(axes[0, col], d[m]['model'])
        chords(axes[0, col], m, (('model', 'cyan'),))
        axes[0, col].set_title(f'danish fit c to the simulation, mask {m}\nchi2/dof {f.chi2_dof:.3f} '
                               f'(dof {int(f.dof)}); cyan: model shadows', fontsize=9)
        r = res[m]
        axes[1, col - 1].imshow(np.where(msk, 0, r), origin='lower', cmap='RdBu_r', vmin=-lim,
                                vmax=lim)
        axes[1, col - 1].set_title(
            f'simulation − its danish fit ({m}); color ±5 noise sigma (±{lim:.0f} e)\nRMS on the '
            f'ring {np.sqrt(np.mean(r[ring] ** 2)):.0f} e (noise {noise:.0f} e); shadow azimuth '
            f'sim − model median {np.median(dmm):+.3f} deg', fontsize=9)
    gray(axes[1, 2], data['img'])
    axes[1, 2].set_title('data stamp of the same side (ISR); not fitted on this page', fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'Step 3f: sim {row.label} {row.side}-focal {row.detector}, field r '
                 f'{row.field_r_deg:.3f} deg, az OCS {row.field_az_ocs_deg:.0f} deg; blur '
                 f'{p6.SIM_FWHM_ARCSEC[row.side]:.2f} arcsec (drawn and fitted). The sim has the '
                 'design wavefront only; danish fits it against the intrinsicTA reference.\n'
                 'Red in a residual: the sim has light the danish model lacks; blue: the model has '
                 'light the sim lacks', fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.03, hspace=0.16)
    pdf.savefig(fig, dpi=70)
    plt.close(fig)


def sim_chi2_table(F, S):
    t = S[['key', 'label', 'side', 'detector', 'field_r_deg']].copy()
    for m in F:
        t[f'chi2_dof_{m}'] = t.key.map(F[m].chi2_dof)
    t['dof'] = t.key.map(F['v4.02g'].dof)
    big = t.field_r_deg > 1.4
    head = ''.join(f'field {lab}: n {int(g.shape[0])}, median chi2/dof v4.02g '
                   f'{g["chi2_dof_v4.02g"].median():.3f}, v4.02s {g["chi2_dof_v4.02s"].median():.3f}\n'
                   for lab, g in (('<= 1.4 deg', t[~big]), ('> 1.4 deg', t[big])))
    return (head + '\nThe sims are Batoid v4.02 design images: chi2/dof near 1 means danish '
            'reproduces Batoid to the noise.\n\n'
            'What is left with v4.02s at large field (P1693, CR00SW0_0): stop-plane rays under the '
            'residual blobs, first clipper in the full Batoid v4.02 trace.\n'
            '  blue (model has light the sim lacks): TMASection (TMA truss sections), with M2, '
            'CameraSilhouette and a vane;\n'
            '  red (sim has light the model lacks): mostly M1Baffle1, with TopEndRing and '
            'MidLevelBaffle on the corner side.\n'
            'The danish mask has no TMASection circle and models each edge as a circle whose center '
            'moves only along the field direction.\n\n' + t.drop(columns='key').round(3).to_string(index=False))


def page_vanes_compare(pdf, vd, vs, vm):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    for ax, side in zip(axes, ('extra', 'intra')):
        for tab, lab, m, col in ((vd, 'data, all sides', 'o', SIDE_COLOURS[side]),
                                 (vm, 'data, the sim donuts', 's', 'k'),
                                 (vs, 'simulation', 'D', 'tab:gray')):
            g = tab[tab.side == side]
            ax.errorbar(g.vane_id + {'o': -0.15, 's': 0, 'D': 0.15}[m], g.dmm_median_deg,
                        yerr=g.dmm_nmad_deg / np.sqrt(g.n), fmt=m, color=col, ms=5, label=lab)
        g = vd[vd.side == side].set_index('vane_id')
        s = vs[vs.side == side].set_index('vane_id')
        ax.plot(g.index, g.dmm_median_deg - s.dmm_median_deg.reindex(g.index), 'x-',
                color='tab:red', label='data − simulation')
        ax.axhline(0, color='0.5', lw=0.5)
        ax.set_xlabel('vane id (sorted by pupil azimuth)')
        ax.set_title(f'{side}-focal', fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('shadow azimuth at r_half, data − model (deg)')
    fig.suptitle('Step 3d: per vane id, shadow azimuth data − danish model at rtp; error bars nMAD '
                 '/ sqrt(n)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_m1_field(pdf, md, ms):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    for ax, side in zip(axes, ('extra', 'intra')):
        g = md[md.side == side]
        s = ms[ms.side == side]
        ax.plot(g.field_r_deg, g.m1_amp_deg, 'o', ms=3, color=SIDE_COLOURS[side],
                label=f'data, n {len(g)}: median {np.median(g.m1_amp_deg):.3f} deg')
        ax.plot(s.field_r_deg, s.m1_amp_deg, 'D', ms=5, color='k',
                label=f'simulation, n {len(s)}: median {np.median(s.m1_amp_deg):.3f} deg')
        ax.set_xlabel('field radius (deg)')
        ax.set_title(f'{side}-focal', fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('spider m = 1 amplitude per side (deg)')
    fig.suptitle('Step 3f: m = 1 amplitude of the shadow azimuth data − model, per side, against '
                 'field radius', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_checks(pdf, T, chk):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    rf = T['rot20_v402g']['rotation_fit']
    for side, col in SIDE_COLOURS.items():
        c = chk.set_index('side').loc[side]
        axes[0].hist(rf[rf.side == side].delta_best_deg, bins=np.arange(0.1, 0.25, 0.005),
                     histtype='step', color=col,
                     label=f'{side}: median {c.delta_median_deg:.3f} deg, recovery '
                     f'{c.delta_recovery:.2f} (dimensionless; fitted over drawn)')
    axes[0].axvline(ROT_DRAWN_DEG, color='k', ls='--', label='drawn 0.20 deg')
    axes[0].set_xlabel('best delta (deg), sims drawn at rtp + 0.20 deg')
    axes[0].legend(fontsize=7)
    x = np.arange(2)
    for k, (q, lab) in enumerate((('tea_dy_about_center_pix', 'shadows about fit-a center'),
                                  ('tea_center_dy_pix', 'fit-a center'),
                                  ('tea_dy_image_pix', 'shadows in the image'),
                                  ('tea_expected_pix', 'expected (Batoid, on axis)'))):
        axes[1].bar(x + 0.2 * k - 0.3, chk[q], 0.2, label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(chk.side)
    axes[1].axhline(0, color='0.5', lw=0.5)
    axes[1].set_ylabel('dy, OCS image (pixels), TEA +10 mm y minus nominal')
    axes[1].legend(fontsize=7)
    for side, col in SIDE_COLOURS.items():
        c = chk.set_index('side').loc[side]
        axes[2].bar(['shadows in sim', 'fit c model shadows'],
                    [c.shadow_shift_in_sim_deg, c.fit_c_model_shadow_shift_deg], 0.35,
                    color=col, alpha=0.6, label=side)
    axes[2].set_ylabel('rot20 minus nominal, shadow azimuth (deg)')
    axes[2].legend(fontsize=7)
    axes[2].set_title('fit c moves its own shadows toward the drawn rotation, so delta sees only '
                      'the rest', fontsize=8)
    fig.suptitle('Step 3e: pipeline checks. Left: rotation drawn at rtp + 0.20 deg, fitted at rtp. '
                 'Middle: TEA +10 mm in y. Right: why delta recovers less than 0.20 deg',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_silhouette(pdf, sb):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    fields = sorted(sb.field_deg.unique())
    cmap = plt.get_cmap('viridis')
    for i, cam in enumerate((4.0, -4.0)):
        g0 = sb[(sb.cam_mm == cam) & (sb.det_mm == 0.0)]
        for j, m in enumerate(('v4.02g', 'v4.02s')):
            ax = axes[i, j]
            for k, f in enumerate(fields):
                g = g0[g0.field_deg == f].sort_values('psi_deg')
                y = np.where(g.sil_set, g[f'dmb_{m}_pix'], np.nan)
                ax.plot(g.psi_deg, y, '-', color=cmap(k / max(1, len(fields) - 1)), lw=0.8,
                        label=f'{f:.2f} deg')
            ax.axhline(0, color='0.5', lw=0.5)
            ax.set_title(f'camera {cam:+.0f} mm ({"extra" if cam > 0 else "intra"}), {m}: danish '
                         '− Batoid inner edge, silhouette-set bins', fontsize=9)
            ax.set_ylim(-6, 2)
    for ax in axes[1]:
        ax.set_xlabel('relative azimuth psi (deg)')
    for ax in axes[:, 0]:
        ax.set_ylabel('inner edge, danish − Batoid (pixels)')
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle('Step 4a: the CameraSilhouette circle refit (v4.02s, cubic in field 1.3-1.75 deg)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_psi_stack(pdf, stacks):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for ax, side in zip(axes, ('intra', 'extra')):
        lo, hi = PSI_RANGES[side]
        for st, ls in zip(stacks, ('-', '-')):
            g = st[st.side == side].sort_values('psi_deg')
            ax.plot(g.psi_deg, g.inner_dmm, ls, lw=1, label=f'inner, {g["mask"].iloc[0]}')
        g = stacks[-1][stacks[-1].side == side].sort_values('psi_deg')
        ax.plot(g.psi_deg, g.outer_dmm, ':', color='k', lw=0.8, label='outer, v4.02s')
        if lo > hi:
            ax.axvspan(lo, 360, color='0.85')
            ax.axvspan(0, hi, color='0.85')
        else:
            ax.axvspan(lo, hi, color='0.85')
        ax.axhline(0, color='0.5', lw=0.5)
        ax.set_ylim(-4, 8)
        ax.set_ylabel('data − fit c model (pixels)')
        ax.set_title(f'{side}-focal; grey: the silhouette range psi {lo:.0f}-{hi:.0f} deg',
                     fontsize=9)
        ax.legend(fontsize=7)
    axes[1].set_xlabel('relative azimuth psi (deg)')
    fig.suptitle('Step 4b: psi-stacked data − model edges, science sides at 1.6-1.725 deg, '
                 'non-spider, non-vignetted bins', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_vane_width(pdf, wd, ws, cd, cs):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for tab, chi, lab, m in ((wd, cd, 'data', 'o'), (ws, cs, 'sim', 'D')):
        for side, col in SIDE_COLOURS.items():
            g = tab[tab.side == side]
            axes[0, 0].errorbar(g.field_r_deg, g.width_best_mm, yerr=g.width_err_mm, fmt=m,
                                color=col, ms=4, mfc=col if lab == 'data' else 'none',
                                label=f'{lab} {side}: median {np.median(g.width_best_mm):.1f} mm, '
                                f'nMAD {nmad(g.width_best_mm):.1f} mm, n {len(g)}')
            if lab == 'data':
                axes[1, 0].plot(g.depth_model_cad, g.depth_data_cad, 'x', color=col,
                                label=f'{side}, model at CAD 50 mm')
                axes[1, 0].plot(g.depth_model_best, g.depth_data_best, 'o', color=col, ms=3,
                                label=f'{side}, model at best width')
            c = chi[chi.key.isin(g.key)]
            for key, h in c.groupby('key'):
                h = h.sort_values('width_mm')
                axes[0, 1].plot(h.width_mm, h.chi2_dof - h.chi2_dof.min(), color=col, lw=0.4,
                                alpha=0.6, ls='-' if lab == 'data' else '--')
    axes[0, 0].axhline(p6.CAD_VANE_WIDTH_M * 1e3, color='k', ls='--', lw=0.8)
    axes[0, 0].set_xlabel('field radius (deg)')
    axes[0, 0].set_ylabel('fitted vane width in the pupil (mm)')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_xlabel('vane width (mm)')
    axes[0, 1].set_ylabel('chi2/dof − minimum (dof 7.3e5-7.4e5)')
    axes[0, 1].set_ylim(0, 1.5)
    axes[0, 1].set_title('solid: data; dashed: sims', fontsize=8)
    lim = [0.2, 1.0]
    axes[1, 0].plot(lim, lim, 'k-', lw=0.5)
    axes[1, 0].set_xlabel('model shadow depth (dimensionless; fraction of the ring level)')
    axes[1, 0].set_ylabel('data shadow depth (dimensionless)')
    axes[1, 0].legend(fontsize=7)
    for side, col in SIDE_COLOURS.items():
        g = wd[wd.side == side]
        axes[1, 1].plot(g.fwhm_arcsec, g.width_best_mm, 'o', color=col, ms=4, label=side)
    axes[1, 1].set_xlabel('free-fit blur FWHM, pass 5 (arcsec)')
    axes[1, 1].set_ylabel('fitted vane width (mm)')
    axes[1, 1].legend(fontsize=7)
    fig.suptitle('Step 5: one free vane width (all 4 vanes) per side, central-band science sides, '
                 'blur fixed; sims as the control (drawn with the CAD 50 mm vanes)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


# ---------------------------------------------------------------- driver

HUYGENS_DIRS = ('huygens', 'huygens_plume2')


def huygens_tables():
    """Per-variant summaries of the Huygens runs that exist, plus the data on the same sides."""
    out = [pd.read_parquet(RUN / d / 'huygens_summary.parquet').assign(run=d)
           for d in HUYGENS_DIRS if (RUN / d / 'huygens_summary.parquet').exists()]
    return pd.concat(out, ignore_index=True) if out else None


def data_shadow_shape(keys):
    """Data shadow shape on the sim sides, from pass 5 (the pass 6 data runs kept no profiles,
    so their depth and FWHM columns are empty; the ISR does not change the shape)."""
    a = pd.read_parquet(p6.PASS5_RUN / 'spiders.parquet')
    a = a[(a.centre == 'fit_a') & (a.missing == '') & a.key.isin(keys)].copy()
    a['ew_pix'] = a.depth_data * a.fwhm_pix_data
    return a.groupby('side').agg(depth=('depth_data', 'median'), fwhm_pix=('fwhm_pix_data', 'median'),
                                 ew_pix=('ew_pix', 'median'))


def huygens_profiles(run_dir, variant):
    """Median shadow profile per side over the central band (|offset| <= 18 pixels)."""
    f = RUN / run_dir / variant / 'spider_profiles.npz'
    sp = RUN / run_dir / variant / 'spiders.parquet'
    return _median_profiles(f, sp, 'data')


def _median_profiles(npz, spiders, who):
    grid = np.arange(-18, 18.01, 0.5)
    if not npz.exists():
        return grid, {}
    p = np.load(npz)
    t = pd.read_parquet(spiders)
    t = t[(t.centre == 'fit_a') & (t.missing == '') & (t.sensor_class == 'science')]
    acc = {}
    for r in t.itertuples():
        k = f'{r.key}:{r.model_idx}'
        if f'{k}:{who}_off' not in p:
            continue
        o, fl = p[f'{k}:{who}_off'].astype(float), p[f'{k}:{who}_flux'].astype(float)
        ok = np.isfinite(fl)
        if ok.sum() < 40:
            continue
        prof = np.interp(grid, o[ok], fl[ok], left=np.nan, right=np.nan)
        acc.setdefault(r.side, []).append(prof / np.nanmedian(prof[np.abs(grid) > 15]))
    return grid, {s: np.nanmedian(v, axis=0) for s, v in acc.items()}


def page_huygens(pdf, H, dshape, dprof):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    H = H.copy()
    names = list(dict.fromkeys(H.variant))
    x = np.arange(len(names))
    for ax, (q, lab, dq) in zip(axes[0], (
            ('ew_pix', 'shadow equivalent width, depth x FWHM (pixels)', 'ew_pix'),
            ('edge_inner_pix', 'inner edge, sim − danish model, median (pixels)', None),
            ('dy_pix', 'shadow translation dy, OCS image (pixels)', None))):
        for k, side in enumerate(('extra', 'intra')):
            g = H[H.side == side].drop_duplicates('variant').set_index('variant').reindex(names)
            ax.bar(x + 0.4 * k - 0.2, g[q], 0.4, color=SIDE_COLOURS[side], label=side)
            if dq is not None and side in dshape.index:
                ax.axhline(dshape.loc[side, dq], color=SIDE_COLOURS[side], ls='--', lw=1,
                           label=f'data, {side}')
        ax.axhline(0, color='0.5', lw=0.5)
        if dq is not None and len(dshape):
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, max(hi, 1.1 * float(dshape[dq].max())))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7)
    show = [v for v in ('geometric', 'diffraction', 'plume_p0p1', 'plume_p0p15', 'plume_m0p1')
            if v in names]
    for ax, side in zip(axes[1, :2], ('extra', 'intra')):
        if side in dprof[1]:
            ax.plot(dprof[0], dprof[1][side], 'k-', lw=2, label='data (central band)')
        for v in show:
            rd = H[H.variant == v].run.iloc[0]
            grid, pr = huygens_profiles(rd, v)
            if side in pr:
                ax.plot(grid, pr[side], '-', lw=1, label=v)
        ax.set_xlabel('offset across the shadow (pixels)')
        ax.set_ylabel('flux / ring level (dimensionless)')
        ax.set_title(f'{side}-focal: median shadow profile, science sides', fontsize=9)
        ax.legend(fontsize=7)
    ax = axes[1, 2]
    for side, col in SIDE_COLOURS.items():
        g = H[H.side == side].drop_duplicates('variant').set_index('variant').reindex(names)
        ax.plot(x, g.chi2_dof, 'o-', color=col, label=side)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('fit c chi2/dof of danish (v4.02s) to the sim, median (dof 3.8e5-7.4e5)')
    ax.legend(fontsize=7)
    fig.suptitle('Huygens (wave-optics) giant donuts, 5 wavelengths 570-670 nm, 26 sim sides: '
                 'diffraction and vane phase strips (OPD in um; _yp/_ym: strip offset along stop y '
                 'in mm; _s: Gaussian sigma in mm). Dashed: data', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def silhouette_data_fit(stack_s):
    import pass6_silhouette as psil
    rows = []
    for side in ('intra', 'extra'):
        m = stack_s[stack_s.side == side][['psi_deg', 'inner_dmm']]
        rows.append(psil.data_fit(m, side))
    return pd.DataFrame(rows)


def main():
    pd.set_option('display.width', 220)
    T = {c: load(c) for c in ('isr_visitinfo', DATA_CFG, 'isr_phys_v402s') + SIM_CFGS}
    p5 = {n: pd.read_parquet(p6.PASS5_RUN / f'{n}.parquet')
          for n in ('translation_summary', 'centre_test', 'rotation_summary')}
    sim_keys = T['nominal_v402g']['fits'].key.unique()

    # step 1 and 2
    e1 = pd.DataFrame(effect_row('isr_visitinfo', T['isr_visitinfo'])
                      + effect_row(DATA_CFG, T[DATA_CFG]))
    p5t = p5['translation_summary']
    p5t = p5t[p5t.centre == 'fit_a'].set_index('side')
    p5c = p5['centre_test']
    p5c = p5c[p5c.centre == 'fit_a'].set_index('side')
    e1 = pd.concat([pd.DataFrame([dict(config='pass5', side=s, n_sides=p5t.loc[s].n_sides,
                                       rot_deg=p5t.loc[s].rot_deg_median,
                                       dx_pix=p5t.loc[s].dx_pix_median,
                                       dx_err_pix=p5t.loc[s].dx_pix_median_err,
                                       dy_pix=p5t.loc[s].dy_pix_median,
                                       dy_err_pix=p5t.loc[s].dy_pix_median_err,
                                       m1_amp_deg=p5c.loc[s].m1_amp_deg,
                                       m1_phase_deg=p5c.loc[s].m1_phase_deg,
                                       inner_dy_pix=p5c.loc[s].inner_edge_dy_ocs_pix,
                                       outer_dy_pix=p5c.loc[s].outer_edge_dy_ocs_pix)
                                  for s in ('extra', 'intra')]), e1], ignore_index=True)
    pa._save(e1, RUN, 'summary_isr_rtp')
    calibs = pd.read_csv(RUN / 'isr_calibs.csv')
    rot = pd.read_parquet(RUN / 'rotator.parquet')

    # step 3
    chk = checks_table(T)
    pa._save(chk, RUN, 'sim_checks')
    corr = {s: T[DATA_CFG]['rotation_summary'].set_index('group').loc[s].delta_median_deg
            / chk.set_index('side').loc[s].delta_recovery for s in ('extra', 'intra')}
    e3 = pd.DataFrame(effect_row('data, all sides', T[DATA_CFG])
                      + effect_row('data, sim donuts', T[DATA_CFG], sim_keys)
                      + sum((effect_row(f'sim {c}', T[c]) for c in SIM_CFGS[:2]), []))
    sub = []
    for side in ('extra', 'intra'):
        d = e3[(e3.config == 'data, sim donuts') & (e3.side == side)].iloc[0]
        s = e3[(e3.config == 'sim nominal_v402g') & (e3.side == side)].iloc[0]
        sub.append(dict(config='data - sim (sim donuts)', side=side, n_sides=d.n_sides,
                        **{q: d[q] - s[q] for q in ('rot_deg', 'dx_pix', 'dy_pix', 'm1_amp_deg',
                                                    'inner_dy_pix', 'outer_dy_pix',
                                                    'ratio_excess')}))
    e3 = pd.concat([e3, pd.DataFrame(sub)], ignore_index=True)
    pa._save(e3, RUN, 'summary_sims_vs_data')
    vd = per_vane(T[DATA_CFG]['spiders'])
    vm = per_vane(T[DATA_CFG]['spiders'], sim_keys)
    vs = per_vane(T['nominal_v402g']['spiders'])
    vt = vd.merge(vs, on=['side', 'vane_id'], suffixes=('_data', '_sim'))
    vt['dmm_data_minus_sim_deg'] = vt.dmm_median_deg_data - vt.dmm_median_deg_sim
    pa._save(vt, RUN, 'per_vane_sims_vs_data')
    md = per_key_m1(T[DATA_CFG]['spiders'])
    ms = per_key_m1(T['nominal_v402g']['spiders'])
    pa._save(pd.concat([md.assign(set='data'), ms.assign(set='sim')]), RUN, 'm1_per_side')

    # step 4
    sb = pd.read_parquet(RUN / 'silhouette_bins.parquet')
    ss = pd.read_parquet(RUN / 'silhouette_summary.parquet')
    st_g, r_g = psi_stack(T[DATA_CFG], 'v4.02g')
    st_s, r_s = psi_stack(T['isr_phys_v402s'], 'v4.02s')
    pa._save(pd.concat([st_g, st_s]), RUN, 'psi_stack_edges')
    r4 = pd.concat([r_g, r_s], ignore_index=True)
    pa._save(r4, RUN, 'silhouette_data_miss')
    dfit = silhouette_data_fit(st_s)
    pa._save(dfit, RUN, 'silhouette_data_fit')

    # step 5
    wd = pd.read_parquet(RUN / 'vane_width_data.parquet')
    ws = pd.read_parquet(RUN / 'vane_width_sims.parquet')
    cd = pd.read_parquet(RUN / 'vane_width_chi2_data.parquet')
    cs = pd.read_parquet(RUN / 'vane_width_chi2_sims.parquet')
    w = pd.concat([wd, ws]).groupby(['set', 'side']).agg(
        n=('key', 'count'), width_median_mm=('width_best_mm', 'median'),
        width_nmad_mm=('width_best_mm', nmad), width_err_median_mm=('width_best_mm', err_median),
        depth_data=('depth_data_best', 'median'), depth_model_best=('depth_model_best', 'median'),
        depth_model_cad=('depth_model_cad', 'median'), fwhm_data_pix=('fwhm_data_pix_cad', 'median'),
        fwhm_model_cad_pix=('fwhm_model_pix_cad', 'median'),
        chi2_dof_cad=('chi2_dof_cad', 'median'), chi2_dof_best=('chi2_dof_best', 'median'),
        dof=('dof', 'median')).reset_index()
    pa._save(w, RUN, 'vane_width_summary')

    rs_d = T[DATA_CFG]['rotation_summary']
    fmt = lambda df: df.round(3).to_string(index=False)  # noqa: E731
    with PdfPages(RUN / 'pupil_pass6.pdf') as pdf:
        text_page(pdf, 'Pupil pass 6 (run 20261023_pupil6): summary', summary_text(
            e1, rs_d, corr, chk, e3, r4, dfit, w), 8)
        text_page(pdf, 'Section 1: ISR with flat and CTI correction. Calibration collections',
                  calibs.to_string(index=False) + '\n\nIsrTaskLSST with the obs_lsst isrLSST '
                  'overrides; doBrighterFatter and doCorrectGains off (no gain_correction '
                  'calibration); config in isr_config.py. One sky constant per stamp, as before.\n\n'
                  'Translation (OCS image pixels, fit-a center, median and error on the median), '
                  'rotation (deg), m = 1 amplitude (deg), inner/outer edge circle dy data − model '
                  '(pixels), ratio excess (dimensionless), fit c chi2/dof (dof about 7.4e5):\n'
                  + fmt(e1), 7)
        page_effects(pdf, e1, 'Section 1-2: pass 5 against the ISR stamps (visitInfo rtp) and '
                     'the ISR stamps with physical_rotator_angle')
        text_page(pdf, 'Section 2: rtp = physical_rotator_angle (ConsDB cdb_lsstcam.'
                  'visit1_quicklook). Rotator angles (deg)', rot.round(4).to_string(index=False)
                  + '\n\ndelta (deg), danish spider_angle = rtp + delta:\n' + fmt(rs_d)
                  + '\n\nPass 5 (visitInfo rtp), for comparison:\n' + fmt(p5['rotation_summary'])
                  + '\n\nCorrected for the rot20 check recovery (section 3e): '
                  + ', '.join(f'{s} {corr[s]:+.3f} deg' for s in corr), 7)
        page_delta(pdf, T[DATA_CFG]['rotation_fit'], rs_d, corr)
        text_page(pdf, 'Section 3: simulated giant donuts (Batoid v4.02, design state, 30 million '
                  'rays, 620 nm). Sides and draws', T['nominal_v402g']['sides'][
                      ['label', 'side', 'detector', 'field_r_deg', 'field_az_ocs_deg', 'mag',
                       'rtp_deg']].round(3).to_string(index=False) + '\n\nSim − data effects '
                  '(units as section 1):\n' + fmt(e3), 7)
        page_effects(pdf, e3, 'Section 3d: simulation against data. The sims show no spider '
                     'm = 1, no translation, no edge-circle shift and no ratio excess')
        page_vanes_compare(pdf, vd, vs, vm)
        page_m1_field(pdf, md, ms)
        text_page(pdf, 'Section 3e: pipeline checks', fmt(chk.T.reset_index()) +
                  '\n\nrot20: the fit c model is fitted with the spider at rtp, and fit c pulls its '
                  'own shadows toward the drawn rotation (fit_c_model_shadow_shift_deg), so delta '
                  'recovers delta_recovery of the drawn rotation.\ntea_y10: the TEA central shade '
                  'moves with the vanes, so the fit-a center moves too; the shadow shift in the '
                  'image is the sum (tea_dy_image_pix). CCS y is OCS y to 0.01 deg here.', 8)
        page_checks(pdf, T, chk)
        F = {m: T[f'nominal_v402{m[-1]}']['fits'].set_index('key') for m in ('v4.02g', 'v4.02s')}
        SP = {m: T[f'nominal_v402{m[-1]}']['spiders'] for m in ('v4.02g', 'v4.02s')}
        S = T['nominal_v402g']['sides'].sort_values(['field_r_deg', 'side'])
        text_page(pdf, 'Section 3f: sim pages, fit c chi2/dof of the danish fit to each sim, '
                  'v4.02g and v4.02s masks', sim_chi2_table(F, S), 8)
        for _, row in S.iterrows():
            page_sim_side(pdf, row, F, SP)
        text_page(pdf, 'Section 4: camera silhouette. 4a: v4.02g and v4.02s against Batoid, '
                  'silhouette-set bins (pixels)', fmt(ss) + '\n\n4b: data − model inner edge in '
                  'the silhouette ranges, science 1.6-1.725 deg (pixels):\n' + fmt(r4)
                  + '\n\nSilhouette (outer_s) radius and center change that removes the v4.02s '
                  'miss (Huber fit; mm in the silhouette plane; center along the field direction):\n'
                  + fmt(dfit), 7)
        page_silhouette(pdf, sb)
        page_psi_stack(pdf, (st_g, st_s))
        text_page(pdf, 'Section 5: free vane width (mm in the pupil; CAD 50 mm). Depths '
                  'dimensionless (fraction of the ring level), FWHM pixels', fmt(w.T.reset_index()),
                  8)
        page_vane_width(pdf, wd, ws, cd, cs)
        H = huygens_tables()
        if H is not None:
            sim_keys_all = T['nominal_v402g']['fits'].key.unique()
            dshape = data_shadow_shape(sim_keys_all)
            dprof = _median_profiles(p6.PASS5_RUN / 'spider_profiles.npz',
                                     p6.PASS5_RUN / 'spiders.parquet', 'data')
            pa._save(H, RUN, 'huygens_summary_all')
            text_page(pdf, 'Section 6: Huygens giant donuts (Batoid v4.02 rays, NUFFT Huygens sum, '
                      'unit weights), fitted with danish v4.02s at the sim blur. Pixels; '
                      'ew = depth x FWHM; ratio_excess dimensionless',
                      H.drop(columns='run').round(3).to_string(index=False)
                      + '\n\nData on the same 26 sides (pass 5 shadows, central and outer band):\n'
                      + dshape.round(3).to_string(), 7)
            page_huygens(pdf, H, dshape, dprof)
        text_page(pdf, 'Summary: open causes of the remaining data − model mismatch',
                  OPEN_TEXT, 8)
    print('wrote', RUN / 'pupil_pass6.pdf', flush=True)


def summary_text(e1, rs, corr, chk, e3, r4, dfit, w):
    def g(df, cfg, side, q):
        return df[(df.config == cfg) & (df.side == side)].iloc[0][q]
    c = chk.set_index('side')
    r = rs.set_index('group')
    lines = []
    for side in ('extra', 'intra'):
        lines.append(
            f'{side}: ISR changes dy by {g(e1, "isr_visitinfo", side, "dy_pix") - g(e1, "pass5", side, "dy_pix"):+.3f} '
            f'pixels, m = 1 by {g(e1, "isr_visitinfo", side, "m1_amp_deg") - g(e1, "pass5", side, "m1_amp_deg"):+.3f} deg, '
            f'inner-edge dy by {g(e1, "isr_visitinfo", side, "inner_dy_pix") - g(e1, "pass5", side, "inner_dy_pix"):+.3f} pixels.')
    lines.append('')
    for side in ('extra', 'intra'):
        lines.append(f'{side}: delta {r.loc[side].delta_median_deg:+.3f} deg (nMAD '
                     f'{r.loc[side].delta_nmad_deg:.3f}, error {r.loc[side].delta_median_err_deg:.3f}, n '
                     f'{int(r.loc[side].n_sides)}); recovery {c.loc[side].delta_recovery:.2f} '
                     f'(dimensionless) -> corrected {corr[side]:+.3f} deg.')
    lines.append('')
    for side in ('extra', 'intra'):
        lines.append(f'{side} sims: m = 1 {g(e3, "sim nominal_v402g", side, "m1_amp_deg"):.3f} deg, '
                     f'dy {g(e3, "sim nominal_v402g", side, "dy_pix"):+.3f} pixels, inner dy '
                     f'{g(e3, "sim nominal_v402g", side, "inner_dy_pix"):+.3f} pixels, ratio excess '
                     f'{g(e3, "sim nominal_v402g", side, "ratio_excess"):+.4f}; data − sim m = 1 '
                     f'{g(e3, "data - sim (sim donuts)", side, "m1_amp_deg"):.3f} deg, dy '
                     f'{g(e3, "data - sim (sim donuts)", side, "dy_pix"):+.3f} pixels.')
        lines.append(f'   TEA +10 mm: shadows in the image {c.loc[side].tea_dy_image_pix:+.3f} pixels '
                     f'(expected {c.loc[side].tea_expected_pix:+.2f}); about the fit-a center '
                     f'{c.loc[side].tea_dy_about_center_pix:+.3f} pixels.')
    lines.append('')
    for _, q in r4.iterrows():
        lines.append(f'{q["mask"]} {q.side} psi {q.psi_range} deg: inner miss '
                     f'{q.inner_in_range_pix:+.2f} +- {q.inner_in_range_err_pix:.2f} pixels '
                     f'(n {int(q.n_sides)}).')
    for _, q in dfit.iterrows():
        lines.append(f'silhouette fit {q.side}: radius {q.d_radius_mm:+.2f} +- {q.d_radius_err_mm:.2f} '
                     f'mm, center {q.d_center_mm:+.2f} +- {q.d_center_err_mm:.2f} mm; rms '
                     f'{q.miss_rms_before_pix:.2f} -> {q.miss_rms_after_pix:.2f} pixels.')
    lines.append('')
    for _, q in w.iterrows():
        lines.append(f'vane width {q.set} {q.side}: {q.width_median_mm:.1f} mm (nMAD '
                     f'{q.width_nmad_mm:.1f}, n {int(q.n)}); depth data {q.depth_data:.2f}, model '
                     f'at best {q.depth_model_best:.2f}, at CAD {q.depth_model_cad:.2f}.')
    return '\n'.join(lines)


OPEN_TEXT = """Neither danish nor the geometric Batoid sims contain the effects below.

1. Fresnel diffraction (section 6, Batoid Huygens sums, 570-670 nm, 26 sides). The detector
   is 8.4 mm from focus: equivalent propagation distance F^2 / dz = 12.7 km, Fresnel scale
   89 mm in the pupil, 7.2 pixels. Diffraction moves the edges +0.31 / +0.26 pixels inner and
   -0.37 / -0.28 outer (extra / intra) and gives ratio excess +0.002 / +0.001: about half of
   the data's (+0.75 / +0.42, -0.51 / -0.46 pixels; +0.004 / +0.002), same signs and order.
   It widens the vane shadows equally on both sides of focus.

2. A phase strip along the vanes. Shadow equivalent width (depth x FWHM) in the data is
   10.0 pixels extra and 5.7 intra on the sim sides; an opaque vane gives the same on both
   sides (sims 5.1 / 4.9). A strip of +0.10 to +0.15 um of extra optical path (colder or
   denser air than ambient), Gaussian sigma 40-90 mm on the stop plane, gives 8.8-9.6 / 3.6-4.5
   pixels. The negative sign gives the reverse. About 0.1 K over 1 m of path.
   Tests: shadows against vane - air temperature; a giant pair at rtp +-90 deg.

3. Still open: the common translation of the shadows (dy -1.65 / -2.07 pixels OCS image) and
   the inner-edge shift (+2.1 / +1.8 pixels). Offset phase strips move the pattern only
   0.13-0.27 pixels. Not danish, TEA, coma, rotator, ISR or the detector: CCD readout and CTI
   are ruled out (translation the same in both amplifier halves and both vendors: extra dy
   -1.70 / -1.64 E2V and -1.75 / -1.57 ITL bottom / top, pixels); brighter-fatter would fill
   the shadows, and the data shadows are deeper than the model.
"""


if __name__ == '__main__':
    main()
