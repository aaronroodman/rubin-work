#!/usr/bin/env python3
"""Correlate the uniform-focus Double Zernike term DZ(k=1, j=4) with TMA truss temperature.

DZ(k=1, j=4) is the focal-plane-uniform part of the Zernike Z4 (defocus) wavefront: the
k=1 focal Zernike is a constant over the field, so this single coefficient is the mean
defocus of the whole focal plane, in µm of wavefront. Truss temperature is known to
correlate with focus changes, and this script asks how much of DZ(1,4) it actually
explains -- separately for the drift between Full Array Mode (FAM) sequences and the large
swings seen within a single sequence.

Alongside it the script reconstructs v-mode 1 of the Optical Feedback Control (OFC) 22-DOF
/ 12-v-mode scheme from the commanded degrees of freedom, so the measured focus can be
compared against the commanded optical state. v-mode 1 is dominated by the two hexapod
dz axes (99.8% of its normalized weight), and a hexapod position is the look-up table
(LUT) baseline plus the accumulated Trim offset, so both terms are included:

    v1 = C_CamHexdz * (lut_dof5 + dof5) + C_M2Hexdz * (lut_dof0 + dof0)
         + C_M1M3B3 * dof12 + C_M2B5 * dof34 + C_M2B4 * dof33

The mirror-mode LUT is omitted deliberately: it has never been changed from the
mirror-laboratory values, and the three mirror bending terms carry only 0.063 of v-mode 1's
normalized weight in total.

Fits are Huber M-estimators (`statsmodels` RLM with `HuberT`), and both Pearson r and
Spearman rho are reported, per the repository's convention for AOS correlations.

Usage:
    python code/correlations/run_dz14_truss.py --param-set <ps>
    python code/correlations/run_dz14_truss.py --param-set <ps> --dz-prefix z1toz3
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # aos/code
from common.utils import nmad  # noqa: E402

# v-mode 1 of the OFC 22_12 scheme, as v-mode amplitude per unit DOF, from the ts_ofc
# StateEstimator (notebooks/smatrix_vmode/vmode_dof_ts_ofc.ipynb). Units 1/µm.
# The normalized weight of each term is given for context: the two hexapod dz axes carry
# 0.759 and 0.650, the three mirror bending modes only 0.033, 0.029 and 0.001.
V1_HEX = {5: -0.0008915,    # camera hexapod dz [1/µm], normalized weight -0.759
          0: -0.0009104}    # M2 hexapod dz     [1/µm], normalized weight -0.650
V1_BEND = {12: +0.1172,     # M1M3 bending mode 3 [1/µm], normalized weight +0.033
           34: +0.1142,     # M2 bending mode 5   [1/µm], normalized weight +0.029
           33: +0.001562}   # M2 bending mode 4   [1/µm], normalized weight +0.001

SEQ_GAP = 5          # a seq_num jump larger than this starts a new FAM sequence
MIN_SEQ_LEN = 5      # sequences shorter than this are excluded from within-sequence stats


def robust_line(x, y):
    """Huber-RLM line plus Pearson and Spearman statistics for one (x, y) pair.

    Parameters
    ----------
    x, y : `array_like`
        Paired samples in any units; NaN pairs are dropped. The returned slope carries
        units of y per unit of x.

    Returns
    -------
    stats : `dict` or `None`
        ``n``, ``intercept``, ``slope``, ``slope_err`` (Huber RLM), ``pearson_r``,
        ``pearson_p``, ``spearman_rho``, ``spearman_p``, and ``resid_nmad`` (robust
        scatter of the residual, in units of y). `None` if fewer than 10 finite pairs.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 10:
        return None
    x, y = x[m], y[m]
    X = sm.add_constant(x)
    r = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return dict(n=int(x.size), intercept=float(r.params[0]), slope=float(r.params[1]),
                slope_err=float(r.bse[1]), pearson_r=float(pr), pearson_p=float(pp),
                spearman_rho=float(sr), spearman_p=float(sp),
                resid_nmad=float(nmad(y - r.predict(X))))


def label_fam_sequences(df, gap=SEQ_GAP):
    """Integer FAM-sequence id per visit: a new night or a seq_num gap starts a sequence.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must be sorted by ``(day_obs, seq_num)``.
    gap : `int`, optional
        A ``seq_num`` increment larger than this begins a new sequence.

    Returns
    -------
    seq_id : `numpy.ndarray`
        Sequence id per row, counting from 1.
    """
    out = np.empty(len(df), dtype=int)
    cur, prev_day, prev_sn = 0, None, None
    for i, (day, sn) in enumerate(zip(df['day_obs'].to_numpy(),
                                      df['seq_num'].to_numpy())):
        if prev_day is None or day != prev_day or (sn - prev_sn) > gap:
            cur += 1
        out[i] = cur
        prev_day, prev_sn = day, sn
    return out


def load(param_set, dz_prefix):
    """Join the DZ fit, the truss temperature and the commanded DOF, one row per visit.

    Returns
    -------
    df : `pandas.DataFrame`
        Quality-passing FAM visits with ``dz14`` (µm of wavefront), ``truss`` and
        ``truss_dt`` (deg C), ``v1_lut_trim``, ``v1_trim`` and ``v1_lut``
        (dimensionless v-mode amplitudes), and ``fam_seq``.
    """
    base = pathlib.Path('output') / param_set
    dz_col = f'{dz_prefix}_z4_c1'
    fits = pd.read_parquet(base / 'fits.parquet', columns=[
        'day_obs', 'seq_num', dz_col, f'{dz_col}_err', f'{dz_prefix}_bad_fit',
        'visit_quality_pass', 'tma_truss_temp_pxpy', 'tma_truss_temp_mxmy',
        'alt', 'az', 'band'])
    dof_cols = ([f'dof{k}' for k in list(V1_HEX) + list(V1_BEND)]
                + [f'lut_dof{k}' for k in V1_HEX])
    vis = pd.read_parquet(base / 'visits.parquet',
                          columns=['day_obs', 'seq_num'] + dof_cols)
    df = fits.merge(vis, on=['day_obs', 'seq_num'], how='left', validate='one_to_one')

    df = df[(~df[f'{dz_prefix}_bad_fit'].astype(bool))
            & df['visit_quality_pass'].astype(bool)].copy()
    df = df.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)

    df['dz14'] = df[dz_col]
    df['truss'] = 0.5 * (df['tma_truss_temp_pxpy'] + df['tma_truss_temp_mxmy'])
    df['truss_dt'] = df['tma_truss_temp_pxpy'] - df['tma_truss_temp_mxmy']

    # v-mode 1 from the commanded DOF. The hexapod dz axes need LUT + Trim to be a
    # physical position; the mirror bending modes use Trim alone (see module docstring).
    df['v1_lut'] = sum(c * df[f'lut_dof{k}'] for k, c in V1_HEX.items())
    df['v1_trim'] = (sum(c * df[f'dof{k}'] for k, c in V1_HEX.items())
                     + sum(c * df[f'dof{k}'] for k, c in V1_BEND.items()))
    df['v1_lut_trim'] = df['v1_lut'] + df['v1_trim']
    df['fam_seq'] = label_fam_sequences(df)
    return df


def _scatter(ax, x, y, xlabel, ylabel, title=None, color='C0'):
    """Scatter with a Huber-RLM line, annotating slope, Pearson r and Spearman rho."""
    ax.plot(x, y, '.', ms=2.5, alpha=0.35, color=color)
    f = robust_line(x, y)
    if f is not None:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, f['intercept'] + f['slope'] * xs, 'r-', lw=1.5)
        head = f"{title + chr(10) if title else ''}"
        ax.set_title(f"{head}slope {f['slope']:+.4g} +/- {f['slope_err']:.2g}\n"
                     f"Pearson r {f['pearson_r']:+.3f}, "
                     f"Spearman rho {f['spearman_rho']:+.3f}, n={f['n']}", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    return f


def page_overview(df, pdf, rows):
    """DZ(1,4) and v-mode 1 against truss temperature, all visits pooled."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    specs = [
        ('truss', 'dz14', 'mean TMA truss temp [deg C]',
         'DZ(k=1,j=4) [um of wavefront]', 'measured focus vs truss temp'),
        ('truss', 'v1_lut_trim', 'mean TMA truss temp [deg C]',
         'v1 from LUT+Trim [dimensionless]', 'commanded v-mode 1 vs truss temp'),
        ('v1_lut_trim', 'dz14', 'v1 from LUT+Trim [dimensionless]',
         'DZ(k=1,j=4) [um of wavefront]', 'measured focus vs commanded v-mode 1'),
        ('truss_dt', 'dz14', 'truss pxpy - mxmy [deg C]',
         'DZ(k=1,j=4) [um of wavefront]', 'focus vs truss gradient'),
    ]
    for ax, (xc, yc, xl, yl, ti) in zip(axes.ravel(), specs):
        f = _scatter(ax, df[xc].to_numpy(float), df[yc].to_numpy(float), xl, yl, ti)
        if f:
            rows.append(dict(kind='pooled', x=xc, y=yc, **f))
    fig.suptitle('DZ(k=1,j=4) uniform focus, TMA truss temperature and OFC v-mode 1\n'
                 'all quality-passing FAM visits pooled', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def page_lut_vs_trim(df, pdf, rows):
    """Why the LUT term cannot be dropped: LUT, Trim and their sum against each other."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    specs = [
        ('v1_trim', 'v1_lut', 'v1 from Trim alone [dimensionless]',
         'v1 from LUT alone [dimensionless]', 'LUT vs Trim: are they anti-correlated?'),
        ('truss', 'v1_lut', 'mean TMA truss temp [deg C]',
         'v1 from LUT alone [dimensionless]', 'the LUT is an elevation/filter model'),
        ('truss', 'v1_trim', 'mean TMA truss temp [deg C]',
         'v1 from Trim alone [dimensionless]', 'Trim alone vs truss temp'),
        ('v1_trim', 'dz14', 'v1 from Trim alone [dimensionless]',
         'DZ(k=1,j=4) [um of wavefront]', 'focus vs Trim-only v1 (incomplete)'),
    ]
    for ax, (xc, yc, xl, yl, ti) in zip(axes.ravel(), specs):
        f = _scatter(ax, df[xc].to_numpy(float), df[yc].to_numpy(float), xl, yl, ti,
                     color='C2')
        if f:
            rows.append(dict(kind='lut_vs_trim', x=xc, y=yc, **f))
    for c, lbl in [('v1_lut', 'LUT alone'), ('v1_trim', 'Trim alone'),
                   ('v1_lut_trim', 'LUT+Trim')]:
        v = df[c].to_numpy(float)
        v = v[np.isfinite(v)]
        print(f"  v1 {lbl:10s}: n={v.size} median {np.median(v):+.4f} "
              f"std {v.std():.4f} nMAD {nmad(v):.4f} (dimensionless)")
    fig.suptitle('v-mode 1 from the hexapod LUT, the Trim, and their sum\n'
                 'a hexapod position is LUT + Trim; neither term alone is the position',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def page_within_between(df, pdf, rows):
    """Split the DZ(1,4) variation into within-FAM-sequence and between-sequence parts."""
    sizes = df.groupby('fam_seq').size()
    big = sizes[sizes >= MIN_SEQ_LEN].index
    sub = df[df['fam_seq'].isin(big)]
    g = sub.groupby('fam_seq')

    w_dz = g['dz14'].apply(lambda v: nmad(v.to_numpy(float)))
    w_tr = g['truss'].apply(lambda v: nmad(v.to_numpy(float))
                            if v.notna().sum() >= 3 else np.nan)
    w_v1 = g['v1_lut_trim'].apply(lambda v: nmad(v.to_numpy(float)))
    m_dz = g['dz14'].median()
    m_tr = g['truss'].median()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.hist(w_dz.dropna(), bins=30, color='C0', alpha=0.85)
    med_w = float(np.nanmedian(w_dz))
    ax.axvline(med_w, color='r', lw=1.5)
    ax.set_xlabel('within-sequence nMAD of DZ(k=1,j=4) [um of wavefront]', fontsize=8)
    ax.set_ylabel(f'FAM sequences (>= {MIN_SEQ_LEN} visits)', fontsize=8)
    ax.set_title(f'the swing within one FAM sequence\nmedian {med_w:.4f} um of wavefront, '
                 f'{len(big)} sequences', fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)

    f = _scatter(axes[0, 1], w_tr.to_numpy(float), w_dz.to_numpy(float),
                 'within-sequence nMAD of truss temp [deg C]',
                 'within-sequence nMAD of DZ(1,4) [um]',
                 'does a wobblier truss give a wobblier focus?')
    if f:
        rows.append(dict(kind='within_seq_nmad', x='truss_nmad', y='dz14_nmad', **f))

    f = _scatter(axes[1, 0], m_tr.to_numpy(float), m_dz.to_numpy(float),
                 'per-sequence median truss temp [deg C]',
                 'per-sequence median DZ(1,4) [um of wavefront]',
                 'BETWEEN sequences: medians only')
    if f:
        rows.append(dict(kind='between_seq_median', x='truss_median', y='dz14_median', **f))

    ax = axes[1, 1]
    labels = ['DZ(1,4)\n[um of wf]', 'truss temp\n[deg C]', 'v1 LUT+Trim\n[dimensionless]']
    within = [np.nanmedian(w_dz), np.nanmedian(w_tr), np.nanmedian(w_v1)]
    between = [nmad(m_dz.to_numpy(float)),
               nmad(m_tr.dropna().to_numpy(float)),
               nmad(sub.groupby('fam_seq')['v1_lut_trim'].median().to_numpy(float))]
    xs = np.arange(3)
    ax.bar(xs - 0.2, within, 0.4, label='within sequence (median nMAD)')
    ax.bar(xs + 0.2, between, 0.4, label='between sequences (nMAD of medians)')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('robust scatter, own units', fontsize=8)
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('where the variation lives', fontsize=8)
    ax.tick_params(labelsize=7)

    print(f"\n  FAM sequences: {len(sizes)} total, {len(big)} with >= {MIN_SEQ_LEN} visits")
    print(f"  WITHIN-sequence median nMAD:  DZ(1,4) {within[0]:.4f} um of wavefront, "
          f"truss {within[1]:.4f} deg C, v1 {within[2]:.4f} dimensionless")
    print(f"  BETWEEN-sequence nMAD of medians: DZ(1,4) {between[0]:.4f} um of wavefront, "
          f"truss {between[1]:.4f} deg C, v1 {between[2]:.4f} dimensionless")
    rows.append(dict(kind='variance_split', x='within_vs_between', y='dz14',
                     n=int(len(big)), dz_within_nmad_um=float(within[0]),
                     dz_between_nmad_um=float(between[0]),
                     truss_within_nmad_degC=float(within[1]),
                     truss_between_nmad_degC=float(between[1])))
    fig.suptitle('Is the DZ(k=1,j=4) swing during a FAM sequence thermal?\n'
                 'truss temperature is nearly constant within a sequence', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)
    return w_dz, w_tr, m_dz, m_tr


def page_sequences(df, pdf, n_show=6):
    """DZ(1,4), truss temperature and commanded v-mode 1 through the longest sequences."""
    sizes = df.groupby('fam_seq').size().sort_values(ascending=False)
    show = list(sizes.index[:n_show])
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for ax, sid in zip(axes.ravel(), show):
        s = df[df['fam_seq'] == sid]
        ax.plot(s['seq_num'], s['dz14'], 'o-', ms=3, lw=0.8, color='C0')
        ax.set_ylabel('DZ(1,4) [um of wf]', fontsize=7, color='C0')
        ax.tick_params(labelsize=6)
        ax2 = ax.twinx()
        ax2.plot(s['seq_num'], s['truss'], 's-', ms=2.5, lw=0.8, color='C3')
        ax2.set_ylabel('truss [deg C]', fontsize=7, color='C3')
        ax2.tick_params(labelsize=6)
        dz_sw = nmad(s['dz14'].to_numpy(float))
        tr = s['truss'].to_numpy(float)
        tr_sw = nmad(tr) if np.isfinite(tr).sum() >= 3 else np.nan
        ax.set_title(f"day_obs {int(s['day_obs'].iloc[0])}, {len(s)} visits: "
                     f"DZ nMAD {dz_sw:.3f} um, truss nMAD {tr_sw:.3f} deg C", fontsize=7)
        ax.set_xlabel('seq_num', fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('The longest FAM sequences: measured focus (blue) and truss temp (red)\n'
                 'large focus swings at essentially fixed temperature', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--dz-prefix', default='z1toz6', choices=['z1toz3', 'z1toz6'],
                    help='which focal-order DZ fit to read (default z1toz6)')
    ap.add_argument('--output-dir', default=None,
                    help='default output/<param_set>/correlations')
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir or
                       f'output/{args.param_set}/correlations')
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.param_set, args.dz_prefix)
    print(f"quality-passing FAM visits: {len(df)}")
    print(f"  with truss temperature:   {int(df['truss'].notna().sum())}")
    print(f"  with hexapod LUT:         {int(df['v1_lut'].notna().sum())}")
    for c, unit in [('dz14', 'um of wavefront'), ('truss', 'deg C'),
                    ('v1_lut_trim', 'dimensionless')]:
        v = df[c].to_numpy(float)
        v = v[np.isfinite(v)]
        print(f"  {c:12s} [{unit:15s}] n={v.size:5d} median {np.median(v):+.4f} "
              f"nMAD {nmad(v):.4f}")

    rows = []
    pdf_path = out / f'dz14_truss_{args.dz_prefix}.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        page_overview(df, pdf, rows)
        page_lut_vs_trim(df, pdf, rows)
        page_within_between(df, pdf, rows)
        page_sequences(df, pdf)
    print(f"\nSaved: {pdf_path}")

    sm_path = out / f'dz14_truss_{args.dz_prefix}_summary.parquet'
    pd.DataFrame(rows).to_parquet(sm_path)
    print(f"Saved: {sm_path}  ({len(rows)} fit rows)")


if __name__ == '__main__':
    main()
