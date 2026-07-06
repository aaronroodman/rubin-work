#!/usr/bin/env python3
"""compare_fam_processings — compare two FAM reductions of the same data.

Pages (default --comparison all, in order):
  split-ocs   OCS intrinsic-split maps (default mi=pathA_50_34_i_5rot): per Zernike
              Z4..Z26 four panels — OCS map B, OCS map A, difference (2–98% scale),
              histogram of differences.
  dof         Average 50-DOF LUT, A vs B, grouped into the validation-style panels
              (hexapod translations µm / rotations arcsec / M1M3 / M2 bending).
  visit-rms   Per-visit robust nMAD of zk_deviation by Zernike (summary + paired scatter);
              nMAD is over ALL used donuts in a visit (whole focal plane).
  dz-ordinal  rotator∈[-rot-lim,+rot-lim]: final DZ c_{k,j} vs ordinal image # (mjd order),
              A & B overplotted — one page per focal k, 21 pupil-j panels.
  dz-scatter  all rotators: A-vs-B scatter of DZ c_{k,j} — one page per k, 21 j panels.
  fwhm-edge   rotator∈[-rot-lim,+rot-lim]: per-donut FWHM-equivalent (arcsec) at the FoV
              edge, A & B overplotted (page-16 style).  Needs the LSST stack (ts_wep);
              skipped with a note if unavailable.
  yield       per-visit donut count, blur, quality-pass fraction, coverage.

Mostly pure parquet; fwhm-edge additionally needs ts_wep (run on the RSP).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OLD_DEFAULT = 'fam_danish_1_0_wep17_3_0_bin2x'
NEW_DEFAULT = 'fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x'

# DOF grouping — mirrors intrinsic_build_plots.plot_dof_median_summary / ofc_svd.LABELS_50DOF
DOF_GROUPS = [
    ('Hexapod Translations', 'µm',     [0, 1, 2, 5, 6, 7]),   # M2 dz/dx/dy, Cam dz/dx/dy
    ('Hexapod Rotations',    'arcsec', [3, 4, 8, 9]),          # M2 rx/ry, Cam rx/ry
    ('M1M3 Bending Modes',   'µm',     list(range(10, 30))),   # B1_1..20
    ('M2 Bending Modes',     'µm',     list(range(30, 50))),   # B2_1..20
]


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size >= 3 else np.nan


def _noll(base):
    vt = pq.read_table(str(base / 'visits.parquet'), columns=['nollIndices']).to_pandas()
    return [int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]


def _coord_sys(ps, default='OCS'):
    for p in (Path('snake_config.yaml'), Path(__file__).resolve().parent.parent / 'snake_config.yaml'):
        if p.exists():
            import yaml
            cfg = yaml.safe_load(open(p)).get('param_sets', {})
            return cfg.get(ps, {}).get('coord_sys', default)
    return default


# ---------------------------------------------------------------- split-ocs
def compare_split_ocs(baseA, baseB, mi, zks, pdf, per_page=3):
    import matplotlib.pyplot as plt
    fa, fb = baseA / mi / 'intrinsic_split_maps.parquet', baseB / mi / 'intrinsic_split_maps.parquet'
    if not (fa.exists() and fb.exists()):
        print(f'[split-ocs] missing intrinsic_split_maps for A or B (mi={mi})'); return
    A, B = pd.read_parquet(fa), pd.read_parquet(fb)
    key = ['thx_deg', 'thy_deg']
    for d in (A, B):
        d[key] = d[key].round(5)
    M = A.merge(B, on=key, suffixes=('_A', '_B'))
    js = [j for j in zks if f'Z{j}_OCS_A' in M.columns]
    print(f'[split-ocs] mi={mi}; grid overlap {len(M)}/{len(A)}; Zernikes {js}')
    x, y = M['thx_deg'].values, M['thy_deg'].values
    for p0 in range(0, len(js), per_page):
        grp = js[p0:p0 + per_page]
        fig, axes = plt.subplots(len(grp), 4, figsize=(15, 3.4 * len(grp)),
                                 constrained_layout=True, squeeze=False)
        for r, j in enumerate(grp):
            a, b = M[f'Z{j}_OCS_A'].values, M[f'Z{j}_OCS_B'].values
            d = b - a
            vlo, vhi = np.nanpercentile(np.concatenate([a, b]), [2, 98])
            dlo, dhi = np.nanpercentile(d, [2, 98])
            for c, (vals, lo, hi, cmap, ttl) in enumerate([
                    (b, vlo, vhi, 'viridis', f'Z{j} OCS  B'),
                    (a, vlo, vhi, 'viridis', f'Z{j} OCS  A'),
                    (d, dlo, dhi, 'RdBu_r',  f'Z{j} OCS  B−A')]):
                ax = axes[r, c]
                sc = ax.scatter(x, y, c=vals, s=6, cmap=cmap, vmin=lo, vmax=hi)
                ax.set_aspect('equal'); ax.set_title(ttl, fontsize=8); ax.tick_params(labelsize=6)
                fig.colorbar(sc, ax=ax, shrink=0.8)
            ax = axes[r, 3]
            ax.hist(d[np.isfinite(d)], bins=50, color='slategray')
            ax.set_title(f'Z{j} (B−A) hist  rms={np.sqrt(np.nanmean(d**2)):.4f}', fontsize=8)
            ax.tick_params(labelsize=6); ax.set_xlabel('B−A [µm]', fontsize=7)
        fig.suptitle(f'OCS intrinsic split (mi={mi})  [B={baseB.name}, A={baseA.name}]', fontsize=10)
        pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- dof
def compare_dof(baseA, baseB, mi, pdf):
    import matplotlib.pyplot as plt
    fa, fb = baseA / mi / 'lut' / 'lut.parquet', baseB / mi / 'lut' / 'lut.parquet'
    if not (fa.exists() and fb.exists()):
        print(f'[dof] missing lut.parquet for A or B (mi={mi})'); return
    A = pd.read_parquet(fa).set_index('dof_index')
    B = pd.read_parquet(fb).set_index('dof_index')
    fig, axes = plt.subplots(len(DOF_GROUPS), 1, figsize=(13, 11), constrained_layout=True)
    for ax, (title, unit, idx) in zip(axes, DOF_GROUPS):
        idx = [i for i in idx if i in A.index and i in B.index]
        x = np.arange(len(idx))
        labels = [A.loc[i, 'dof_label'] for i in idx]
        ax.bar(x - 0.2, [A.loc[i, 'value'] for i in idx], 0.4, label=baseA.name)
        ax.bar(x + 0.2, [B.loc[i, 'value'] for i in idx], 0.4, label=baseB.name)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel(f'DOF value ({unit})'); ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Average DOF, A vs B (mi={mi})  [A={baseA.name}, B={baseB.name}]', fontsize=11)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- visit-rms
def _visit_rms_table(base, coord, zks):
    noll = _noll(base); col = f'zk_deviation_{coord}'
    df = pq.read_table(str(base / 'donuts.parquet'),
                       columns=['day_obs', 'seq_num', 'used', col]).to_pandas()
    if 'used' in df.columns:
        df = df[df['used'].astype(bool)]
    Z = np.stack(df[col].values).astype(float)
    jidx = {j: noll.index(j) for j in zks if j in noll}
    out = []
    for (d, s), g in df.groupby(['day_obs', 'seq_num']):
        zz = Z[g.index.values]; rec = dict(day_obs=int(d), seq_num=int(s))
        for j, i in jidx.items():
            rec[f'rms_Z{j}'] = nmad(zz[:, i])
        out.append(rec)
    return pd.DataFrame(out), list(jidx)


def compare_visit_rms(baseA, baseB, coord, zks, pdf):
    import matplotlib.pyplot as plt
    A, ja = _visit_rms_table(baseA, coord, zks)
    B, jb = _visit_rms_table(baseB, coord, zks)
    js = [j for j in zks if j in ja and j in jb]
    M = A.merge(B, on=['day_obs', 'seq_num'], suffixes=('_A', '_B'))
    print(f'[visit-rms] matched visits {len(M)}; coord={coord} (nMAD over all used donuts/visit)')
    S = pd.DataFrame([dict(Z=j, medA=np.nanmedian(M[f'rms_Z{j}_A']), medB=np.nanmedian(M[f'rms_Z{j}_B']),
                           med_ratio=np.nanmedian(M[f'rms_Z{j}_B'] / M[f'rms_Z{j}_A'])) for j in js])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(len(js))
    ax[0].bar(x - 0.2, S['medA'], 0.4, label=baseA.name); ax[0].bar(x + 0.2, S['medB'], 0.4, label=baseB.name)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f'Z{j}' for j in js])
    ax[0].set_ylabel(f'median per-visit nMAD [{coord}, µm]'); ax[0].legend(fontsize=8)
    ax[0].set_title('Per-visit donut-to-donut robust RMS')
    ax[1].axhline(1, color='k', lw=0.7)
    ax[1].bar(x, S['med_ratio'], color=['crimson' if r > 1 else 'steelblue' for r in S['med_ratio']])
    ax[1].set_xticks(x); ax[1].set_xticklabels([f'Z{j}' for j in js])
    ax[1].set_ylabel('median ratio B/A  (<1 = B quieter)'); ax[1].set_title('Robust-RMS ratio (paired)')
    fig.suptitle(f'Per-visit robust RMS by Zernike [{baseB.name} vs {baseA.name}]', fontsize=10)
    pdf.savefig(fig); plt.close(fig)
    n = len(js); fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6),
                                          constrained_layout=True, squeeze=False)
    for ax, j in zip(axes.ravel(), js):
        a, b = M[f'rms_Z{j}_A'], M[f'rms_Z{j}_B']; hi = np.nanpercentile(np.concatenate([a, b]), 99)
        ax.scatter(a, b, s=8, alpha=0.4); ax.plot([0, hi], [0, hi], 'k--', lw=0.7)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect('equal'); ax.set_title(f'Z{j}', fontsize=8)
        ax.set_xlabel('A', fontsize=7); ax.set_ylabel('B', fontsize=7); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle(f'Per-visit nMAD A vs B (each point = one visit), {coord}', fontsize=10)
    pdf.savefig(fig); plt.close(fig)
    print(S.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


# ---------------------------------------------------------------- DZ helpers (7a/7b)
def _dz_load(base, mi, prefix):
    pat = re.compile(rf'^{re.escape(prefix)}_z(\d+)_c(\d+)$')
    df = pd.read_parquet(base / mi / 'fits.parquet')
    cols = [c for c in df.columns if pat.match(c)]
    js = sorted({int(pat.match(c).group(1)) for c in cols})
    ks = sorted({int(pat.match(c).group(2)) for c in cols})
    return df, js, ks, pat


def _panels(n):
    nc = int(np.ceil(np.sqrt(n))); nr = int(np.ceil(n / nc)); return nr, nc


def compare_dz_ordinal(baseA, baseB, mi, prefix, rot_lim, pdf):
    import matplotlib.pyplot as plt
    A, jsa, ks, _ = _dz_load(baseA, mi, prefix); B, jsb, _, _ = _dz_load(baseB, mi, prefix)
    js = sorted(set(jsa) & set(jsb))
    cut = lambda d: d[np.abs(d['rotator_angle'].astype(float)) <= rot_lim]
    M = cut(A).merge(cut(B), on=['day_obs', 'seq_num'], suffixes=('_A', '_B')).sort_values('mjd_A')
    M = M.reset_index(drop=True); ordn = np.arange(len(M))
    print(f'[dz-ordinal] mi={mi}; rotator|<= {rot_lim}; matched visits {len(M)}; k={ks}')
    nr, nc = _panels(len(js))
    for k in ks:
        fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 2.0 * nr), constrained_layout=True, squeeze=False)
        for ax, j in zip(axes.ravel(), js):
            c = f'{prefix}_z{j}_c{k}'
            ax.plot(ordn, M[f'{c}_A'], '.-', ms=3, lw=0.6, label='A', color='steelblue')
            ax.plot(ordn, M[f'{c}_B'], '.-', ms=3, lw=0.6, label='B', color='crimson')
            ax.axhline(0, color='k', lw=0.4); ax.set_title(f'Z{j} c{k}', fontsize=7); ax.tick_params(labelsize=5)
        for ax in axes.ravel()[len(js):]:
            ax.axis('off')
        axes.ravel()[0].legend(fontsize=6)
        fig.suptitle(f'DZ c{k} vs ordinal image # (rotator∈[-{rot_lim:g},{rot_lim:g}], mjd order), '
                     f'mi={mi}  [A blue={baseA.name}, B red={baseB.name}]', fontsize=9)
        pdf.savefig(fig); plt.close(fig)


def compare_dz_scatter(baseA, baseB, mi, prefix, pdf):
    import matplotlib.pyplot as plt
    A, jsa, ks, _ = _dz_load(baseA, mi, prefix); B, jsb, _, _ = _dz_load(baseB, mi, prefix)
    js = sorted(set(jsa) & set(jsb))
    M = A.merge(B, on=['day_obs', 'seq_num'], suffixes=('_A', '_B'))
    print(f'[dz-scatter] mi={mi}; all rotators; matched visits {len(M)}; k={ks}')
    nr, nc = _panels(len(js))
    for k in ks:
        fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 2.2 * nr), constrained_layout=True, squeeze=False)
        for ax, j in zip(axes.ravel(), js):
            c = f'{prefix}_z{j}_c{k}'; a, b = M[f'{c}_A'], M[f'{c}_B']
            lo, hi = np.nanpercentile(np.concatenate([a, b]), [1, 99])
            ax.scatter(a, b, s=5, alpha=0.3); ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6)
            ax.set_title(f'Z{j} c{k}', fontsize=7); ax.tick_params(labelsize=5)
            ax.set_xlabel('A', fontsize=6); ax.set_ylabel('B', fontsize=6)
        for ax in axes.ravel()[len(js):]:
            ax.axis('off')
        fig.suptitle(f'DZ c{k}: A vs B per visit (all rotators), mi={mi}  '
                     f'[A={baseA.name}, B={baseB.name}]', fontsize=9)
        pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- fwhm-edge (7c)
def _fwhm_edge(base, mi, rot_lim, ei, eo, fwhm_fn):
    coord = _coord_sys(base.name)
    noll = _noll(base); zc = f'zk_{coord}'
    dn = pq.read_table(str(base / 'donuts.parquet'),
                       columns=['day_obs', 'seq_num', 'thx_CCS', 'thy_CCS', 'used', zc]).to_pandas()
    side = pq.read_table(str(base / mi / 'zk_intrinsic.parquet'), columns=['zk_intrinsic_MI']).to_pandas()
    r = np.degrees(np.hypot(dn['thx_CCS'].astype(float), dn['thy_CCS'].astype(float)))
    vis = pq.read_table(str(base / 'visits.parquet'),
                        columns=['day_obs', 'seq_num', 'rotator_angle']).to_pandas()
    dn = dn.merge(vis, on=['day_obs', 'seq_num'], how='left')
    mask = (dn['used'].astype(bool) if 'used' in dn else True) & (r >= ei) & (r <= eo) \
        & (np.abs(dn['rotator_angle'].astype(float)) <= rot_lim)
    mask = np.asarray(mask)
    resid = np.stack(dn.loc[mask, zc].values) - np.stack(side['zk_intrinsic_MI'].values)[mask]
    return np.asarray(fwhm_fn(resid, noll)), int(mask.sum()), coord


def compare_fwhm_edge(baseA, baseB, mi, rot_lim, ei, eo, pdf):
    import matplotlib.pyplot as plt
    try:
        from lsst.ts.intrinsic.wavefront.intrinsic_build_plots import donut_fwhm_from_zk
    except Exception as e:
        print(f'[fwhm-edge] cannot import donut_fwhm_from_zk ({e}); skip'); return
    if not np.isfinite(donut_fwhm_from_zk(np.zeros((2, 5)), [4, 5, 6, 7, 8])).any():
        print('[fwhm-edge] FWHM-equiv needs the LSST stack (ts_wep) — run on the RSP. skip'); return
    if not (baseA / mi / 'zk_intrinsic.parquet').exists() or not (baseB / mi / 'zk_intrinsic.parquet').exists():
        print(f'[fwhm-edge] missing zk_intrinsic sidecar for A or B (mi={mi}); skip'); return
    fwA, nA, cA = _fwhm_edge(baseA, mi, rot_lim, ei, eo, donut_fwhm_from_zk)
    fwB, nB, cB = _fwhm_edge(baseB, mi, rot_lim, ei, eo, donut_fwhm_from_zk)
    if not np.isfinite(fwA).any() or not np.isfinite(fwB).any():
        print('[fwhm-edge] FWHM all-NaN — needs the LSST stack (ts_wep); run on the RSP. skip'); return
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bins = np.linspace(0, np.nanpercentile(np.concatenate([fwA, fwB]), 98), 50)
    for fw, n, lab, col in [(fwA, nA, baseA.name, 'steelblue'), (fwB, nB, baseB.name, 'crimson')]:
        m = np.nanmedian(fw)
        ax.hist(fw[np.isfinite(fw)], bins, alpha=0.5, color=col, label=f'{lab}  (n={n}, med={m:.3f}")')
        ax.axvline(m, color=col, lw=1.5)
    ax.set_xlabel('per-donut FWHM-equivalent (arcsec)'); ax.set_ylabel('donut count'); ax.legend(fontsize=8)
    ax.set_title(f'FWHM-equiv at FoV edge ({ei:.2f}–{eo:.2f}°, rotator∈[-{rot_lim:g},{rot_lim:g}]), mi={mi}',
                 fontsize=10)
    pdf.savefig(fig); plt.close(fig)
    print(f'[fwhm-edge] median FWHM-equiv  A={np.nanmedian(fwA):.3f}"  B={np.nanmedian(fwB):.3f}"')


# ---------------------------------------------------------------- yield
def compare_yield(baseA, baseB, pdf):
    import matplotlib.pyplot as plt
    want = ['day_obs', 'seq_num', 'n_donuts', 'median_blur_arcsec', 'visit_quality_pass']

    def load(base):
        cols = [c for c in want if c in pq.read_schema(str(base / 'visits.parquet')).names]
        return pd.read_parquet(base / 'visits.parquet', columns=cols)
    A, B = load(baseA), load(baseB)
    kA, kB = set(zip(A.day_obs, A.seq_num)), set(zip(B.day_obs, B.seq_num))
    print(f'[yield] visits A={len(A)} B={len(B)} matched={len(kA & kB)} onlyA={len(kA - kB)} onlyB={len(kB - kA)}')
    for nm, df in (('A=' + baseA.name, A), ('B=' + baseB.name, B)):
        qp = df['visit_quality_pass'].mean() if 'visit_quality_pass' in df else np.nan
        print(f'  {nm}: median n_donuts={df["n_donuts"].median():.0f}  '
              f'median blur={df.get("median_blur_arcsec", pd.Series([np.nan])).median():.3f}"  qpass={qp:.3f}')
    M = A.merge(B, on=['day_obs', 'seq_num'], suffixes=('_A', '_B'))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    hi = np.nanpercentile(np.concatenate([M['n_donuts_A'], M['n_donuts_B']]), 99)
    ax[0].scatter(M['n_donuts_A'], M['n_donuts_B'], s=8, alpha=0.4); ax[0].plot([0, hi], [0, hi], 'k--', lw=0.7)
    ax[0].set_xlim(0, hi); ax[0].set_ylim(0, hi); ax[0].set_aspect('equal')
    ax[0].set_xlabel('n_donuts A'); ax[0].set_ylabel('n_donuts B'); ax[0].set_title('Per-visit donut yield')
    if 'median_blur_arcsec_A' in M:
        bins = np.linspace(0, np.nanpercentile(np.concatenate(
            [M['median_blur_arcsec_A'], M['median_blur_arcsec_B']]), 99), 30)
        ax[1].hist(M['median_blur_arcsec_A'], bins, alpha=0.5, label=baseA.name)
        ax[1].hist(M['median_blur_arcsec_B'], bins, alpha=0.5, label=baseB.name)
        ax[1].set_xlabel('median blur ["]'); ax[1].set_ylabel('visits'); ax[1].legend(fontsize=8)
    fig.suptitle(f'Yield & coverage [{baseB.name} vs {baseA.name}]', fontsize=10)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ps-a', default=OLD_DEFAULT); ap.add_argument('--ps-b', default=NEW_DEFAULT)
    ap.add_argument('--mi', default='pathA_50_34_i', help='mi config for dof + DZ comparisons')
    ap.add_argument('--split-mi', default='pathA_50_34_i_5rot', help='mi config for the OCS split page')
    ap.add_argument('--comparison', default='all',
                    choices=['all', 'split-ocs', 'dof', 'visit-rms', 'dz-ordinal', 'dz-scatter',
                             'fwhm-edge', 'yield'])
    ap.add_argument('--coord', default='CCS', choices=['CCS', 'OCS'], help='frame for visit-rms')
    ap.add_argument('--dz-prefix', default='z1toz6', help='fits.parquet DZ column prefix')
    ap.add_argument('--rot-lim', type=float, default=3.0, help='|rotator_angle| cut for dz-ordinal/fwhm-edge')
    ap.add_argument('--edge-inner', type=float, default=1.52); ap.add_argument('--edge-outer', type=float, default=1.75)
    ap.add_argument('--zernikes', default='4,5,6,7,8,9,10,11', help='Zernikes for visit-rms')
    ap.add_argument('--split-zernikes', default=','.join(str(j) for j in list(range(4, 20)) + list(range(22, 27))),
                    help='Zernikes for the OCS split page (default Z4..Z26)')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()
    zks = [int(x) for x in args.zernikes.split(',')]
    szks = [int(x) for x in args.split_zernikes.split(',')]
    root = Path(args.output_root); baseA, baseB = root / args.ps_a, root / args.ps_b
    for b in (baseA, baseB):
        if not b.exists():
            raise SystemExit(f'missing output dir: {b}')
    out = root / args.ps_b / 'plots' / f'compare_vs_{args.ps_a}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    do = args.comparison
    print(f'A (baseline) = {args.ps_a}\nB (new)      = {args.ps_b}\nmi={args.mi}  split-mi={args.split_mi}')
    with PdfPages(str(out)) as pdf:
        if do in ('all', 'split-ocs'):
            compare_split_ocs(baseA, baseB, args.split_mi, szks, pdf)
        if do in ('all', 'dof'):
            compare_dof(baseA, baseB, args.mi, pdf)
        if do in ('all', 'visit-rms'):
            compare_visit_rms(baseA, baseB, args.coord, zks, pdf)
        if do in ('all', 'dz-ordinal'):
            compare_dz_ordinal(baseA, baseB, args.mi, args.dz_prefix, args.rot_lim, pdf)
        if do in ('all', 'dz-scatter'):
            compare_dz_scatter(baseA, baseB, args.mi, args.dz_prefix, pdf)
        if do in ('all', 'fwhm-edge'):
            compare_fwhm_edge(baseA, baseB, args.mi, args.rot_lim, args.edge_inner, args.edge_outer, pdf)
        if do in ('all', 'yield'):
            compare_yield(baseA, baseB, pdf)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
