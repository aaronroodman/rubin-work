#!/usr/bin/env python3
"""compare_fam_processings — compare two FAM reductions of the same data.

Compares pipeline products of two param_sets (e.g. an old Danish-1.0 reduction vs a
new Danish-1.2 refitWCS reduction) at several levels:

  lut-maps   A: OCS/CCS intrinsic-split lookup-table maps  (<mi>/intrinsic_split_maps.parquet)
               -> per-(Zernike, frame) difference maps + RMS-of-difference summary.
  lut-dof    B: averaged 50/34-DOF LUT (lut.parquet) + double-Zernike LUT (lut_dz.parquet)
               -> per-DOF value & scatter_mad A-vs-B; per-(k,j) dz_raw/dz_fit/dz_resid.
  visit-rms  C: per-visit robust RMS (nMAD) of zk_deviation by Zernike (donuts.parquet),
               paired over the visits common to both reductions.
  all        : run all three.

Pure parquet (numpy/pandas/pyarrow/matplotlib) — runs anywhere once both outputs are synced.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OLD_DEFAULT = 'fam_danish_1_0_wep17_3_0_bin2x'
NEW_DEFAULT = 'fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x'


def nmad(x):
    """Robust scatter = 1.4826 * median(|x - median(x)|), NaN-safe."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _noll(base):
    vt = pq.read_table(str(base / 'visits.parquet'), columns=['nollIndices']).to_pandas()
    return [int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]


# ---------------------------------------------------------------- A: LUT maps
def compare_lut_maps(baseA, baseB, mi, zks, pdf):
    import matplotlib.pyplot as plt
    fa = baseA / mi / 'intrinsic_split_maps.parquet'
    fb = baseB / mi / 'intrinsic_split_maps.parquet'
    if not (fa.exists() and fb.exists()):
        print(f'[lut-maps] missing: {fa if not fa.exists() else fb}'); return
    A = pd.read_parquet(fa); B = pd.read_parquet(fb)
    key = ['thx_deg', 'thy_deg']
    for d in (A, B):
        d[key] = d[key].round(5)
    M = A.merge(B, on=key, suffixes=('_A', '_B'))
    print(f'[lut-maps] grid overlap {len(M)}/{len(A)} (A) {len(B)} (B)')
    rows = []
    for frame in ('OCS', 'CCS'):
        zj = [j for j in zks if f'Z{j}_{frame}_A' in M.columns]
        n = len(zj)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6),
                                 constrained_layout=True, squeeze=False)
        for ax, j in zip(axes.ravel(), zj):
            d = M[f'Z{j}_{frame}_B'] - M[f'Z{j}_{frame}_A']
            r = nmad(d); rows.append(dict(frame=frame, Z=j, rms_diff=np.sqrt(np.nanmean(d**2)), nmad_diff=r))
            v = np.nanpercentile(np.abs(d), 98) + 1e-9
            sc = ax.scatter(M['thx_deg'], M['thy_deg'], c=d, s=10, cmap='RdBu_r', vmin=-v, vmax=v)
            ax.set_title(f'Z{j} {frame}  Δrms={rows[-1]["rms_diff"]:.4f}', fontsize=8)
            ax.set_aspect('equal'); ax.tick_params(labelsize=6)
            fig.colorbar(sc, ax=ax, shrink=0.8)
        for ax in axes.ravel()[n:]:
            ax.axis('off')
        fig.suptitle(f'Intrinsic-split {frame}: (B−A) µm  [{baseB.name} − {baseA.name}], mi={mi}', fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return df


# ---------------------------------------------------------------- B: DOF / DZ LUT
def compare_lut_dof(baseA, baseB, mi, pdf):
    import matplotlib.pyplot as plt
    fa, fb = baseA / mi / 'lut' / 'lut.parquet', baseB / mi / 'lut' / 'lut.parquet'
    if not (fa.exists() and fb.exists()):
        print(f'[lut-dof] missing lut.parquet for A or B'); return
    A = pd.read_parquet(fa); B = pd.read_parquet(fb)
    M = A.merge(B, on=['dof_index', 'dof_label'], suffixes=('_A', '_B'))
    M['dval'] = M['value_B'] - M['value_A']
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(M))
    ax[0].bar(x - 0.2, M['value_A'], 0.4, label=baseA.name)
    ax[0].bar(x + 0.2, M['value_B'], 0.4, label=baseB.name)
    ax[0].set_xticks(x); ax[0].set_xticklabels(M['dof_label'], rotation=90, fontsize=5)
    ax[0].legend(fontsize=8); ax[0].set_ylabel('DOF value'); ax[0].set_title(f'Averaged-DOF LUT (mi={mi})')
    if 'scatter_mad_A' in M and 'scatter_mad_B' in M:
        ax[1].bar(x - 0.2, M['scatter_mad_A'], 0.4, label='A')
        ax[1].bar(x + 0.2, M['scatter_mad_B'], 0.4, label='B')
        ax[1].set_xticks(x); ax[1].set_xticklabels(M['dof_label'], rotation=90, fontsize=5)
        ax[1].set_ylabel('scatter_mad (visit spread)'); ax[1].legend(fontsize=8)
    pdf.savefig(fig); plt.close(fig)
    top = M.reindex(M['dval'].abs().sort_values(ascending=False).index).head(12)
    print('[lut-dof] largest DOF shifts (B−A):')
    print(top[['dof_label', 'value_A', 'value_B', 'dval']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    za, zb = baseA / mi / 'lut' / 'lut_dz.parquet', baseB / mi / 'lut' / 'lut_dz.parquet'
    if za.exists() and zb.exists():
        DA, DB = pd.read_parquet(za), pd.read_parquet(zb)
        D = DA.merge(DB, on=['k', 'j'], suffixes=('_A', '_B'))
        for c in ('dz_raw', 'dz_fit', 'dz_resid'):
            if f'{c}_A' in D:
                D[f'd_{c}'] = D[f'{c}_B'] - D[f'{c}_A']
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        lbl = [f'k{int(k)},Z{int(j)}' for k, j in zip(D['k'], D['j'])]
        ax.bar(np.arange(len(D)), D.get('d_dz_raw', pd.Series(np.zeros(len(D)))))
        ax.set_xticks(np.arange(len(D))); ax.set_xticklabels(lbl, rotation=90, fontsize=4)
        ax.set_ylabel('Δ dz_raw (B−A)'); ax.set_title(f'Double-Zernike LUT change (mi={mi})')
        pdf.savefig(fig); plt.close(fig)
        print(f'[lut-dz] max |Δdz_raw| = {D.get("d_dz_raw", pd.Series([np.nan])).abs().max():.4f} µm')


# ---------------------------------------------------------------- C: per-visit robust RMS
def _visit_rms_table(base, coord, zks):
    noll = _noll(base)
    col = f'zk_deviation_{coord}'
    df = pq.read_table(str(base / 'donuts.parquet'),
                       columns=['day_obs', 'seq_num', 'used', col]).to_pandas()
    if 'used' in df.columns:
        df = df[df['used'].astype(bool)]
    Z = np.stack(df[col].values).astype(float)
    jidx = {j: noll.index(j) for j in zks if j in noll}
    out = []
    for (d, s), g in df.groupby(['day_obs', 'seq_num']):
        zz = Z[g.index.values]
        rec = dict(day_obs=int(d), seq_num=int(s), n_donuts=len(g))
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
    print(f'[visit-rms] matched visits {len(M)} (A {len(A)}, B {len(B)}); coord={coord}')

    # summary: median robust-RMS over matched visits, A vs B
    summ = []
    for j in js:
        a, b = M[f'rms_Z{j}_A'], M[f'rms_Z{j}_B']
        summ.append(dict(Z=j, medA=np.nanmedian(a), medB=np.nanmedian(b),
                         med_ratio=np.nanmedian(b / a)))
    S = pd.DataFrame(summ)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(len(js))
    ax[0].bar(x - 0.2, S['medA'], 0.4, label=baseA.name)
    ax[0].bar(x + 0.2, S['medB'], 0.4, label=baseB.name)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f'Z{j}' for j in js])
    ax[0].set_ylabel(f'median per-visit nMAD [{coord}, µm]'); ax[0].legend(fontsize=8)
    ax[0].set_title('Per-visit donut-to-donut robust RMS')
    ax[1].axhline(1, color='k', lw=0.7)
    ax[1].bar(x, S['med_ratio'], color=['crimson' if r > 1 else 'steelblue' for r in S['med_ratio']])
    ax[1].set_xticks(x); ax[1].set_xticklabels([f'Z{j}' for j in js])
    ax[1].set_ylabel('median ratio B/A  (<1 = B quieter)'); ax[1].set_title('Robust-RMS ratio (paired)')
    fig.suptitle(f'Per-visit robust RMS by Zernike  [{baseB.name} vs {baseA.name}]', fontsize=10)
    pdf.savefig(fig); plt.close(fig)

    # paired scatter per Zernike
    n = len(js); fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6),
                                          constrained_layout=True, squeeze=False)
    for ax, j in zip(axes.ravel(), js):
        a, b = M[f'rms_Z{j}_A'], M[f'rms_Z{j}_B']
        hi = np.nanpercentile(np.concatenate([a, b]), 99)
        ax.scatter(a, b, s=8, alpha=0.4); ax.plot([0, hi], [0, hi], 'k--', lw=0.7)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect('equal')
        ax.set_title(f'Z{j}', fontsize=8); ax.tick_params(labelsize=6)
        ax.set_xlabel('A', fontsize=7); ax.set_ylabel('B', fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle(f'Per-visit nMAD A vs B (each point = one visit), {coord}', fontsize=10)
    pdf.savefig(fig); plt.close(fig)
    print(S.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ps-a', default=OLD_DEFAULT, help='baseline (old) param_set')
    ap.add_argument('--ps-b', default=NEW_DEFAULT, help='new param_set')
    ap.add_argument('--mi', default='pathA_50_34_i', help='measured-intrinsic config (for lut comparisons)')
    ap.add_argument('--comparison', default='all', choices=['all', 'lut-maps', 'lut-dof', 'visit-rms'])
    ap.add_argument('--coord', default='CCS', choices=['CCS', 'OCS'], help='frame for visit-rms')
    ap.add_argument('--zernikes', default='4,5,6,7,8,9,10,11')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()
    zks = [int(x) for x in args.zernikes.split(',')]
    root = Path(args.output_root)
    baseA, baseB = root / args.ps_a, root / args.ps_b
    for b in (baseA, baseB):
        if not b.exists():
            raise SystemExit(f'missing output dir: {b}')
    out = root / args.ps_b / 'plots' / f'compare_vs_{args.ps_a}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    print(f'A (baseline) = {args.ps_a}\nB (new)      = {args.ps_b}\nmi={args.mi}')
    with PdfPages(str(out)) as pdf:
        if args.comparison in ('all', 'lut-maps'):
            compare_lut_maps(baseA, baseB, args.mi, zks, pdf)
        if args.comparison in ('all', 'lut-dof'):
            compare_lut_dof(baseA, baseB, args.mi, pdf)
        if args.comparison in ('all', 'visit-rms'):
            compare_visit_rms(baseA, baseB, args.coord, zks, pdf)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
