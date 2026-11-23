"""Figures and numbers for the BLOCK-T720 A–B–A bounce note.

Reads the ``bounce_aba`` product (22/12 batoid and RBR 50/50 MIW variants) for the
results, and reruns the A–B–A reduction through `rubinwork.bounce.aba` for what the
product does not hold: the per-visit DZ, the variogram, the 50/34 truncated recovery
and the per-leg MC draws. Writes PNG and PDF into ``figures/`` and the numbers the
note quotes into ``figures/numbers.json``.

    python make_figures.py
"""
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rubinwork import aos_state
from rubinwork.bounce import aba
from rubinwork.bounce import analysis as ba
from rubinwork.bounce import legs as bl
from rubinwork.common.utils import repo_root
from rubinwork.products import bounce_aba, catalog
from rubinwork.products.bounce_aba import reader

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / 'figures'
V22 = 'd13-batoid-22_12'
VRBR = 'd13-A_50_50_i_rbr_5rot-rbr_50_50'
NIGHT_COLOR = {20260418: 'C0', 20260419: 'C1', 20260513: 'C2', 20260709: 'C3',
               20260711: 'C4', 20260713: 'C5'}
SCHEME_COLOR = {'50/34': 'C3', 'RBR 50/50': 'C1', '22/12': 'C0'}
NUM = {}


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=130, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def xoff(legs):
    """Small horizontal offset so the three 40 deg nights do not overlap, deg."""
    off = np.zeros(len(legs))
    for b, g in legs.groupby('b_elev_deg'):
        if len(g) > 1:
            off[g.index] = np.linspace(-0.8, 0.8, len(g))
    return legs.b_elev_deg.to_numpy() + off


# ---------------------------------------------------------------- reduction inputs
def setup():
    vc = reader.variant_config(V22)
    cfg = bl.resolve_config('danish_1_3', None,
                            analysis_config=repo_root() / 'aos' / 'analysis_config.yaml')
    ft = ba.load_fit_table(catalog.path('fam_tables', 'danish_1_3') / 'fits.parquet', cfg,
                           min_detectors=int(vc['min_detectors']))
    bounce = next(x for x in cfg['bounces'] if x['name'] == vc['bounce'])
    visits, pairs, legs = aba.leg_tables(ft, bounce, cfg['fit_prefix'], list(vc['day_obs']))
    legs = legs.rename(columns={'b_value_deg': 'b_elev_deg'})
    return vc, cfg, ft, bounce, visits, pairs, legs


def recovery_50_34(cfg):
    """The default 50-DOF, 34-mode truncated recovery, as an `aba.Recovery`."""
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd
    kr = cfg['focal_k_range']
    svd = build_ofc_svd(list(cfg['pupil_j_range']), int(min(kr)), int(max(kr)),
                        int(cfg['n_keep']), n_dof=cfg['n_dof'],
                        ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
    inv_t = bl.rbr_module().invert_truncated
    P = np.array([inv_t(e, svd) for e in np.eye(len(svd.kj_grid))]).T
    se = aos_state.make_state_estimator(dof_set='all_50', n_modes=int(cfg['n_keep']))
    T_V = aos_state.vmodes_from_dofs(np.eye(50), se, n_modes=int(cfg['n_keep']))
    return aba.Recovery('50_34', svd, None, P, T_V, range(50))


# ---------------------------------------------------------------- figures
def fig_visits(ft, visits, legs):
    alt = np.degrees(np.asarray(ft['alt'], float))
    nights = list(dict.fromkeys(legs.night))
    fig, axs = plt.subplots(len(nights), 1, figsize=(8, 1.6 * len(nights)), sharex=True)
    rows = []
    for ax, n in zip(axs, nights):
        v = visits[visits.night == n].drop_duplicates('seq_num')
        for lab, g in visits[(visits.night == n) & (visits.setting == 'B')].groupby('leg'):
            ax.plot(g.t_min, alt[g.row], 's', color='C3', ms=5, label=f'B ({lab})')
        a = v[v.row.isin(visits[(visits.night == n) & (visits.setting == 'A')].row)]
        ax.plot(a.t_min, alt[a.row], 'o', color='C0', ms=5, label='A (Elev=70)')
        ax.plot(v.sort_values('t_min').t_min, alt[v.sort_values('t_min').row], '-',
                color='0.7', lw=0.8, zorder=0)
        ax.set_ylabel('elev (deg)')
        ax.set_ylim(25, 80)
        ax.text(0.01, 0.08, str(n), transform=ax.transAxes, fontsize=9)
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    for L in legs.itertuples():
        v = aba.leg_visits(visits, L.night, L.leg)
        p = visits[(visits.night == L.night) & (visits.leg == L.leg)]
        rows.append({'night': L.night, 'leg': L.leg, 'n_A': int((v.setting == 'A').sum()),
                     'n_B': int((v.setting == 'B').sum()), 'n_aba': len(aba.aba_matrix(v)),
                     'span_min': float(v.t_min.max() - v.t_min.min()),
                     'band': str(np.asarray(ft['band'])[v.row.iloc[0]]),
                     'blur_med_arcsec': float(np.median(
                         np.asarray(ft['median_blur_arcsec'], float)[v.row])),
                     'seq': f'{v.seq_num.min()}-{v.seq_num.max()}',
                     'pattern': ''.join(np.where(p.drop_duplicates('seq_num').setting == 'other',
                                                 '.', p.drop_duplicates('seq_num').setting))})
    axs[-1].set_xlabel('time since first bounce visit of the night (min)')
    fig.suptitle('BLOCK-T720 visits used: A = 70 deg reference, B = comparison elevation',
                 fontsize=10)
    save(fig, 'fig01_visits')
    NUM['legs'] = rows


def fig_aba_example(W, kj, visits, legs, night, leg, name):
    i14 = kj.index((1, 4))
    v = aba.leg_visits(visits, night, leg)
    M = aba.aba_matrix(v)
    y = W[v.row.to_numpy(), i14]
    t = v.t_min.to_numpy()
    s = v.setting.to_numpy()
    d = M @ y
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.4), width_ratios=[2, 1])
    a0.plot(t[s == 'A'], y[s == 'A'], 'o-', color='C0', label='A (70 deg)')
    a0.plot(t[s == 'B'], y[s == 'B'], 's', color='C3', label=f'B ({leg})')
    for row in M:
        ib = np.nonzero(row == 1)[0][0]
        yi = -(row[row < 0] @ y[row < 0])
        a0.plot([t[ib], t[ib]], [yi, y[ib]], color='C3', lw=1)
        a0.plot(t[ib], yi, 'x', color='C0', ms=6)
    a0.set_xlabel('time (min)')
    a0.set_ylabel('DZ(1,4) (µm)')
    a0.set_title(f'{night} {leg}: per-visit DZ(1,4); x = A interpolated to B time', fontsize=9)
    a0.legend(fontsize=8)
    a1.plot(np.arange(len(d)), d, 's', color='C3')
    a1.axhline(np.median(d), color='k', lw=1, label=f'median = {np.median(d):+.3f} µm')
    a1.set_xlabel('B visit index')
    a1.set_ylabel('B − interp(A) DZ(1,4) (µm)')
    a1.legend(fontsize=8)
    fig.tight_layout()
    save(fig, name)


def fig_noise(noise, red, kj):
    sw = noise['sigma_w']
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.4))
    k = np.array([a for a, _ in kj])
    for kk in sorted(set(k)):
        m = k == kk
        a0.plot(np.nonzero(m)[0], sw[m], 'o', ms=3, label=f'k = {kk}')
    a0.set_yscale('log')
    a0.set_xlabel('DZ term index (k-major, 21 pupil j per k)')
    a0.set_ylabel('white σ per visit (µm)')
    a0.set_title(f'White noise from {noise["n_pairs"]} B−A pairs, {noise["n_groups"]} legs',
                 fontsize=9)
    a0.legend(fontsize=7, ncol=3)
    i14 = kj.index((1, 4))
    tau = np.linspace(0.5, 130, 200)
    D = 2 * sw[i14] ** 2 + red['sigma_r'] ** 2 * (tau / 10.0) ** red['alpha']
    a1.errorbar(red['lag_min'], red['D'], red['D'] * np.sqrt(2 / red['n']), fmt='o',
                color='k', label='same-setting differences')
    a1.plot(tau, D, 'C3', label=(f'2σ_W² + σ_R² (τ/10 min)^α\nσ_W = {sw[i14]:.3f} µm, '
                                  f'σ_R = {red["sigma_r"]:.3f} µm, α = {red["alpha"]:.2f}'))
    a1.axhline(2 * sw[i14] ** 2, color='0.5', ls=':', lw=1)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.set_xlabel('lag τ (min)')
    a1.set_ylabel('DZ(1,4) structure function D(τ) (µm²)')
    a1.set_title(f'DZ(1,4) red noise: chi2/dof = {red["chi2"]:.2f}/{red["dof"]}', fontsize=9)
    a1.legend(fontsize=7)
    fig.tight_layout()
    save(fig, 'fig03_noise')
    NUM['noise'] = {'sigma_w_dz14_um': float(sw[i14]),
                    'sigma_w_max_um': float(sw.max()),
                    'sigma_w_max_term': f'DZ{kj[int(sw.argmax())]}',
                    'sigma_w_median_um': float(np.median(sw)),
                    'sigma_w_kge2_max_um': float(sw[k >= 2].max()),
                    'shrink': noise['shrink'], 'sigma_r_um': red['sigma_r'],
                    'sigma_r_err_um': red['sigma_r_err'], 'alpha': red['alpha'],
                    'alpha_err': red['alpha_err'], 'red_chi2': red['chi2'],
                    'red_dof': red['dof']}


def per_leg_points(ax, legs, val, err, **kw):
    x = xoff(legs)
    for i, L in enumerate(legs.itertuples()):
        ax.errorbar(x[i], val[i], err[i], fmt='o', ms=4, color=NIGHT_COLOR[L.night], **kw)
    ax.axhline(0, color='0.6', lw=0.6)


def night_legend(fig):
    h = [plt.Line2D([], [], marker='o', ls='', color=c, label=str(n))
         for n, c in NIGHT_COLOR.items()]
    fig.legend(handles=h, loc='upper center', ncol=6, fontsize=8, bbox_to_anchor=(0.5, 1.02))


def fig_dz(dz, legs, name, terms, title):
    nc = 4
    nr = int(np.ceil(len(terms) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(11, 2.3 * nr), sharex=True)
    for ax, t in zip(axs.flat, terms):
        g = dz[dz.name == t].set_index(['night', 'leg']).loc[list(zip(legs.night, legs.leg))]
        per_leg_points(ax, legs, g.delta.to_numpy(), g.err.to_numpy())
        ax.plot(xoff(legs), g.fit.to_numpy(), '_', color='k', ms=10, mew=1.5)
        ax.set_title(f'Δ{t} (µm)', fontsize=9)
    for ax in axs.flat[len(terms):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('B elevation (deg)')
    night_legend(fig)
    fig.suptitle(title, fontsize=10, y=1.05)
    fig.tight_layout()
    save(fig, name)


def fig_range(res, ranges, legs, labels):
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    out = {}
    for ax, (sch, (r, idx)) in zip(axs, res.items()):
        rat = np.abs(np.array([x['dof'] for x in r])) / ranges[idx][None, :]
        for i, L in enumerate(legs.itertuples()):
            ax.plot(np.array(idx) + (i - 3) * 0.08, rat[i], 'o', ms=3,
                    color=NIGHT_COLOR[L.night])
        ax.axhline(1, color='k', lw=1)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 50)
        n_over = int((rat > 1).sum())
        ax.set_title(f'{sch}: {n_over} of {rat.size} (leg, DOF) values over range; '
                     f'largest |ΔDOF|/range = {rat.max():.2f}', fontsize=9)
        ax.set_ylabel('|ΔDOF| / range')
        out[sch] = {'n_over': n_over, 'n': int(rat.size), 'max_ratio': float(rat.max()),
                    'max_dof': labels[idx[int(rat.max(0).argmax())]]}
    axs[-1].set_xticks(range(50))
    axs[-1].set_xticklabels(labels, rotation=90, fontsize=6)
    night_legend(fig)
    fig.tight_layout()
    save(fig, 'fig05_dof_over_range')
    NUM['range'] = out


def fig_hexapod(res, legs, rot):
    """M2 dz against camera dz per leg, with the MC cloud, for each scheme."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    out = {}
    for ax, (sch, (r, idx)) in zip(axs, res.items()):
        i2, ic = list(idx).index(0), list(idx).index(5)
        out[sch] = []
        for i, L in enumerate(legs.itertuples()):
            mc = r[i]['dof_mc']
            ax.plot(mc[::5, i2], mc[::5, ic], '.', ms=1, alpha=0.3,
                    color=NIGHT_COLOR[L.night])
            ax.plot(r[i]['dof'][i2], r[i]['dof'][ic], 'o', ms=6, mec='k',
                    color=NIGHT_COLOR[L.night])
            ax.annotate(f'{L.b_elev_deg:.0f}', (r[i]['dof'][i2], r[i]['dof'][ic]),
                        fontsize=7, xytext=(4, 4), textcoords='offset points')
            out[sch].append({'night': L.night, 'leg': L.leg,
                             'm2_dz_um': float(r[i]['dof'][i2]),
                             'cam_dz_um': float(r[i]['dof'][ic]),
                             'mc_r': float(np.corrcoef(mc[:, i2], mc[:, ic])[0, 1]),
                             'sum_um': float(r[i]['dof'][i2] + r[i]['dof'][ic]),
                             'sum_err_um': float((mc[:, i2] + mc[:, ic]).std(ddof=1))})
        lim = np.abs(np.array(ax.get_xlim() + ax.get_ylim())).max()
        ax.plot([-lim, lim], [lim, -lim], color='0.6', lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('ΔM2_dz (µm)')
        ax.set_ylabel('ΔCam_dz (µm)')
        med_r = np.median([o['mc_r'] for o in out[sch]])
        ax.set_title(f'{sch}: median MC r = {med_r:+.2f}', fontsize=9)
    night_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, 'fig06_hexapod_dz')
    NUM['hexapod_dz'] = out


def fig_power(res, legs):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.27
    out = {}
    lab = [f'{L.night}\n{L.leg}' for L in legs.itertuples()]
    for j, (sch, (r, _)) in enumerate(res.items()):
        f = np.array([(x['resid'] ** 2).sum() / (x['dz'] ** 2).sum() for x in r])
        tot = sum((x['resid'] ** 2).sum() for x in r) / sum((x['dz'] ** 2).sum() for x in r)
        ax.bar(np.arange(len(r)) + (j - 1) * w, f, w, color=SCHEME_COLOR[sch],
               label=f'{sch} (all legs: {tot:.3f})')
        out[sch] = {'per_leg': f.tolist(), 'all': float(tot)}
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel('ΔDZ power left after fit\n(Σ resid² / Σ ΔDZ², dimensionless)', fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'fig07_power_left')
    NUM['power_left'] = out


def fig_state(st, legs, kind, names, name, title):
    nc = 5 if kind == 'dof' else 4
    nr = int(np.ceil(len(names) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(12, 2.2 * nr), sharex=True)
    for ax, nm in zip(axs.flat, names):
        g = st[(st.kind == kind) & (st.name == nm)].set_index(['night', 'leg']).loc[
            list(zip(legs.night, legs.leg))]
        per_leg_points(ax, legs, g.delta.to_numpy(), g.err.to_numpy())
        u = g.unit.iloc[0]
        ax.set_title(f'Δ{nm}' + ('' if u == 'dimensionless' else f' ({u.replace("um", "µm")})'), fontsize=9)
    for ax in axs.flat[len(names):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('B elevation (deg)')
    night_legend(fig)
    fig.suptitle(title, fontsize=10, y=1.05)
    fig.tight_layout()
    save(fig, name)


def chi2_40(st, kind):
    """chi2 of the three 40 deg nights about their weighted mean, per quantity (dof 2)."""
    g = st[(st.kind == kind) & (st.leg == 'Elev=40')]
    out = {}
    for nm, h in g.groupby('name', sort=False):
        w = 1 / h.err ** 2
        m = (w * h.delta).sum() / w.sum()
        out[nm] = float((((h.delta - m) / h.err) ** 2).sum())
    return out


def main():
    FIG.mkdir(exist_ok=True)
    vc, cfg, ft, bounce, visits, pairs, legs = setup()
    NUM['n_visits_after_cut'] = len(ft)
    from lsst.ts.intrinsic.wavefront.ofc_svd import LABELS_50DOF, project_dz_table

    r22 = aba.make_recovery('22_12', cfg)
    r34 = recovery_50_34(cfg)
    rrb = aba.make_recovery('rbr_50_50', cfg)
    kj = list(r22.kj)
    W = project_dz_table(ft, cfg['fit_prefix'], r22.svd)[3]
    i14 = kj.index((1, 4))
    noise = aba.white_noise(W, pairs)
    red = aba.red_noise(W, visits, i14, noise['sigma_w'][i14], list(vc['lag_bins_min']),
                        float(vc['t0_min']))
    kw = dict(n_mc=int(vc['n_mc']), seed=int(vc['seed']), t0_min=float(vc['t0_min']))
    res = {'50/34': (aba.run(W, visits, legs, r34, noise, red, i14, **kw), r34.dof_index),
           'RBR 50/50': (aba.run(W, visits, legs, rrb, noise, red, i14, **kw), rrb.dof_index),
           '22/12': (aba.run(W, visits, legs, r22, noise, red, i14, **kw), r22.dof_index)}

    # The product must agree with this rerun for the 22/12 batoid variant.
    _, dz22, st22, _ = bounce_aba.load(V22)
    d = st22[st22.kind == 'dof']
    mine = np.concatenate([x['dof'] for x in res['22/12'][0]])
    assert np.allclose(d.delta.to_numpy(), mine, rtol=1e-9, atol=1e-12), 'product mismatch'

    ranges = bl.rbr_module().dof_range_vector(r34.svd)
    NUM['ranges'] = dict(zip(LABELS_50DOF, ranges.tolist()))

    fig_visits(ft, visits, legs)
    fig_aba_example(W, kj, visits, legs, 20260419, 'Elev=40', 'fig02a_aba_20260419')
    fig_aba_example(W, kj, visits, legs, 20260713, 'Elev=30', 'fig02b_aba_20260713')
    fig_noise(noise, red, kj)

    rms = dz22.groupby('name', sort=False).delta.apply(lambda x: np.sqrt((x ** 2).mean()))
    top = list(rms.sort_values(ascending=False).index[:16])
    NUM['dz_rms_top'] = {t: float(rms[t]) for t in top}
    fig_dz(dz22, legs, 'fig04_dz_top16', top,
           'A–B–A ΔDZ, the 16 terms with the largest RMS over legs (µm); '
           'black bar = 22/12 fit')
    fig_range({k: (v[0], v[1]) for k, v in res.items()}, ranges, legs, LABELS_50DOF)
    fig_hexapod(res, legs, None)
    fig_power(res, legs)

    hexn = [LABELS_50DOF[i] for i in r22.dof_index[:10]]
    bend = [LABELS_50DOF[i] for i in r22.dof_index[10:]]
    fig_state(st22, legs, 'dof', hexn, 'fig08_dof22_hexapods',
              '22/12 ΔDOF: M2 and camera hexapods (translations µm, tilts deg)')
    fig_state(st22, legs, 'dof', bend, 'fig09_dof22_bending',
              '22/12 ΔDOF: M1M3 bending modes B1_1–B1_7 and M2 B2_1–B2_5 (µm)')
    fig_state(st22, legs, 'vmode', [f'v{i}' for i in range(1, 13)], 'fig10_vmode22',
              '22/12 Δv-modes (dimensionless)')

    c = {'dof': chi2_40(st22, 'dof'), 'vmode': chi2_40(st22, 'vmode')}
    gdz = dz22[dz22.leg == 'Elev=40']
    cdz = {}
    for nm, h in gdz.groupby('name', sort=False):
        w = 1 / h.err ** 2
        m = (w * h.delta).sum() / w.sum()
        cdz[nm] = float((((h.delta - m) / h.err) ** 2).sum())
    c['dz'] = cdz
    NUM['chi2_40deg'] = {k: {'median': float(np.median(list(v.values()))),
                             'frac_over_9p21': float(np.mean(np.array(list(v.values())) > 9.21)),
                             'values': v} for k, v in c.items()}
    NUM['dof22'] = st22.to_dict(orient='records')
    (FIG / 'numbers.json').write_text(json.dumps(NUM, indent=1, default=str))
    print('wrote numbers.json')


if __name__ == '__main__':
    main()
