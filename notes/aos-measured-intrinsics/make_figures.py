#!/usr/bin/env python3
"""Regenerate the figures for the "two epochs" note.

Reads the AOS pipeline outputs for param_set ``fam_danish_1_0_wep17_3_0_bin2x``
(raw DZ fits, MI-refit fits, and the per-rotator-bin measured-intrinsic grids)
and writes the PNGs into ./figures/.  Run from anywhere:

    /opt/local/bin/python3 make_figures.py

Paths are resolved relative to this file (notes/<slug>/ → repo root → aos/output).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
AOS_OUT = HERE.parent.parent / 'aos' / 'output'          # rubin-work/aos/output
PS = 'fam_danish_1_0_wep17_3_0_bin2x'
MI = 'pathA_50_34_i'
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
NIGHTS = [20260315, 20260316, 20260317, 20260324, 20260409]   # the 2-epoch window
FP_BASIS = 1.75                                                # DZ field normalization

# 9 rotator bins: (build dir, rotator centre, family)  A=March(0,±15,±60) B=Apr9(±30,±45)
ROT = [('rot_-65_-55', -60, 'A'), ('rot_-50_-40', -45, 'B'), ('rot_-35_-25', -30, 'B'),
       ('rot_-20_-10', -15, 'A'), ('rot_-3_3', 0, 'A'),      ('rot_10_20', 15, 'A'),
       ('rot_25_35', 30, 'B'),    ('rot_40_50', 45, 'B'),    ('rot_55_65', 60, 'A')]


def _save(fig, stem, dpi):
    """Write both PNG (Slack/Markdown) and vector PDF (lsstdoc LaTeX)."""
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{stem}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig); print(f'  wrote {stem}.{{png,pdf}}')


def _split(df):
    return df[df.day_obs != 20260409], df[df.day_obs == 20260409]


def _focal_basis(rho, th):
    """k=1..6 focal-plane (double-Zernike) Zernike basis at (rho, theta)."""
    return [np.ones_like(rho), 2 * rho * np.cos(th), 2 * rho * np.sin(th),
            np.sqrt(3) * (2 * rho ** 2 - 1),
            np.sqrt(6) * rho ** 2 * np.sin(2 * th),
            np.sqrt(6) * rho ** 2 * np.cos(2 * th)]


def fig_rotmaps():
    """One page per Zernike (Z5, Z7): the measured intrinsic for all 9 rotator
    bins, straight from each bin's intrinsic_grid.parquet."""
    base = AOS_OUT / PS / MI / 'build'

    def load_map(d, j):
        p = base / d / 'intrinsic_grid.parquet'
        if not p.exists():
            return None, None, None
        df = pd.read_parquet(p)
        noll = [int(x) for x in df['nollIndices'].iloc[0]]
        df['z'] = [v[noll.index(j)] for v in df['zk']]
        piv = df.pivot_table(index='thy_deg', columns='thx_deg', values='z')
        return piv.values, piv.columns.values, piv.index.values

    for j, name in [(5, 'Z5_astigmatism'), (7, 'Z7_coma')]:
        maps = [(c, fam, load_map(d, j)) for d, c, fam in ROT]
        finite = [m[0][np.isfinite(m[0])] for _, _, m in maps if m[0] is not None]
        if not finite:
            print(f'  (no rotbin grids for {name}; skipped)'); continue
        vlim = np.nanpercentile(np.abs(np.concatenate(finite)), 99)
        fig, axes = plt.subplots(3, 3, figsize=(11, 11), layout='constrained')
        for ax, (c, fam, (M, xs, ys)) in zip(axes.ravel(), maps):
            if M is None:
                ax.axis('off'); continue
            im = ax.imshow(M, origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                           cmap='RdBu_r', vmin=-vlim, vmax=vlim, interpolation='nearest')
            for rr in (1.5178, 1.725):
                ax.add_patch(Circle((0, 0), rr, fill=False, ec='k', lw=0.6, ls=':', alpha=0.5))
            ax.set_title(f'rot {c:+d}°   [{"Apr 9 (B)" if fam == "B" else "March (A)"}]',
                         fontsize=10, color='crimson' if fam == 'B' else 'steelblue', weight='bold')
            ax.set_aspect('equal'); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=7); fig.colorbar(im, ax=ax, shrink=0.72, label='μm')
        fig.suptitle(f'{name.replace("_", " ")} measured intrinsic by rotator angle  ({PS} / {MI})\n'
                     'blue = March epoch (A);  red = Apr-9 epoch (B) — the ±30/±45 maps are the second family',
                     fontsize=12.5, weight='bold')
        _save(fig, f'rotmaps_{name}', 115)


def fig_intrinsic_and_driver():
    """3-panel: Z5 & Z7 at the WFS radius vs azimuth by epoch + the z_gradient driver."""
    raw = pd.read_parquet(AOS_OUT / PS / 'fits.parquet'); raw = raw[raw.day_obs.isin(NIGHTS)]
    mi = pd.read_parquet(AOS_OUT / PS / MI / 'fits.parquet'); mi = mi[mi.day_obs.isin(NIGHTS)]
    R = 1.6; rho = R / FP_BASIS; th = np.linspace(0, 2 * np.pi, 73)
    B = np.stack(_focal_basis(np.full_like(th, rho), th), axis=1)
    rA, rB = _split(raw)

    def ring(df, j):
        C = df[[f'z1toz6_z{j}_c{k}' for k in range(1, 7)]].to_numpy(float); Z = C @ B.T
        med = np.nanmedian(Z, axis=0); return med, 1.4826 * np.nanmedian(np.abs(Z - med), axis=0)
    azdeg = np.degrees(th)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    for p, (j, nm) in enumerate([(5, 'Z5 astigmatism'), (7, 'Z7 coma')]):
        mA, sA = ring(rA, j); mB, sB = ring(rB, j)
        ax[p].fill_between(azdeg, mA - sA, mA + sA, color='steelblue', alpha=0.18)
        ax[p].fill_between(azdeg, mB - sB, mB + sB, color='crimson', alpha=0.18)
        ax[p].plot(azdeg, mA, '-', color='steelblue', lw=2.4, label=f'A: March 15–24  n={len(rA)}')
        ax[p].plot(azdeg, mB, '-', color='crimson', lw=2.4, label=f'B: Apr 9  n={len(rB)}')
        ax[p].axhline(0, color='k', lw=0.5, alpha=0.5); ax[p].grid(alpha=0.3)
        ax[p].set_xlim(0, 360); ax[p].set_xticks(range(0, 361, 90))
        ax[p].set_xlabel('focal-plane azimuth [deg]'); ax[p].set_ylabel(f'{nm.split()[0]} intrinsic [μm]')
        ax[p].set_title(f'{nm} at WFS radius ({R}°), by epoch', fontsize=11); ax[p].legend(fontsize=8)
    mA_, mB_ = _split(mi); gA = mA_['z_gradient'].dropna(); gB = mB_['z_gradient'].dropna()
    ax[2].hist(gA, bins=18, color='steelblue', alpha=0.6, label=f'A March  med={gA.median():+.2f}')
    ax[2].hist(gB, bins=18, color='crimson', alpha=0.6, label=f'B Apr 9  med={gB.median():+.2f}')
    ax[2].axvline(gA.median(), color='steelblue', lw=2); ax[2].axvline(gB.median(), color='crimson', lw=2)
    ax[2].set_xlabel('z_gradient (M1M3 axial thermal)'); ax[2].set_ylabel('visits')
    ax[2].set_title('The driver: M1M3 axial gradient\nflips +0.09 → −0.30', fontsize=11)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle('Measured intrinsic is non-stationary across two observing epochs — the "two families"',
                 fontsize=12.5, weight='bold')
    _save(fig, 'two_epochs_intrinsic', 130)


def fig_zgradient():
    """Standalone z_gradient (M1M3 axial thermal) histogram, March vs Apr-9."""
    mi = pd.read_parquet(AOS_OUT / PS / MI / 'fits.parquet'); mi = mi[mi.day_obs.isin(NIGHTS)]
    A, B = _split(mi); gA = A['z_gradient'].dropna(); gB = B['z_gradient'].dropna()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout='constrained')
    ax.hist(gA, bins=18, color='steelblue', alpha=0.6, label=f'A: March 15–24  (median {gA.median():+.2f})')
    ax.hist(gB, bins=18, color='crimson', alpha=0.6, label=f'B: Apr 9  (median {gB.median():+.2f})')
    ax.axvline(gA.median(), color='steelblue', lw=2); ax.axvline(gB.median(), color='crimson', lw=2)
    ax.set_xlabel('z_gradient  (M1M3 axial thermal gradient)'); ax.set_ylabel('visits')
    ax.set_title('M1M3 axial thermal gradient by epoch', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, 'two_epochs_zgradient', 150)


def fig_fieldmaps():
    """Reconstructed full-FoV Z5/Z7 intrinsic field maps per epoch + difference."""
    raw = pd.read_parquet(AOS_OUT / PS / 'fits.parquet'); raw = raw[raw.day_obs.isin(NIGHTS)]
    A, B = _split(raw); FOV = 1.8; n = 400
    g = np.linspace(-FOV, FOV, n); X, Y = np.meshgrid(g, g); Rr = np.hypot(X, Y); disk = Rr <= FOV
    basis = _focal_basis(Rr / FP_BASIS, np.arctan2(Y, X))

    def field(df, j):
        F = sum(df[f'z1toz6_z{j}_c{k}'].median() * b for k, b in zip(range(1, 7), basis))
        return np.where(disk, F, np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), layout='constrained')
    for row, (j, nm) in enumerate([(5, 'Z5 astigmatism'), (7, 'Z7 coma')]):
        FA, FB = field(A, j), field(B, j); D = FB - FA
        vl = np.nanpercentile(np.abs(np.concatenate([FA[disk], FB[disk]])), 99)
        dl = np.nanpercentile(np.abs(D[disk]), 99)
        for col, (F, ttl, v) in enumerate([(FA, f'A: March (n={len(A)})', vl),
                                           (FB, f'B: Apr 9 (n={len(B)})', vl), (D, 'B − A', dl)]):
            ax = axes[row][col]
            im = ax.imshow(F, origin='lower', extent=[-FOV, FOV, -FOV, FOV], cmap='RdBu_r',
                           vmin=-v, vmax=v, interpolation='nearest')
            for rr, ls in [(1.5178, ':'), (1.725, ':'), (FOV, '-')]:
                ax.add_patch(Circle((0, 0), rr, fill=False, ec='k', lw=0.7, ls=ls, alpha=0.5))
            ax.set_aspect('equal'); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=8); ax.set_title(f'{nm} — {ttl}', fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.78, label='μm')
    fig.suptitle('FAM measured-intrinsic field maps, two epochs — Z5 & Z7  (reconstructed low-order DZ field)',
                 fontsize=13, weight='bold')
    _save(fig, 'two_epochs_fieldmaps', 130)


if __name__ == '__main__':
    print(f'reading from {AOS_OUT}')
    fig_rotmaps()
    fig_intrinsic_and_driver()
    fig_zgradient()
    fig_fieldmaps()
    print('done.')
