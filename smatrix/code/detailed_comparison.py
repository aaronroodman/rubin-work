#!/usr/bin/env python
"""Detailed comparison plots: our DZ sensitivity matrix vs the OFC matrix.

Produces (../output/):
  smatrix_ofc_fieldconst_r.png  - field-constant (k=0) pupil-Zernike x DOF
                                   heatmaps: ours (sign-aligned), OFC, residual
  smatrix_ofc_residual_r.png    - per-DOF max|residual| after single global sign
  smatrix_ofc_examples_r.png    - full 31x29 DZ maps ours/OFC/diff for a few DOF

The matrix is (field_k=31, pupil_j=29, dof=50), units um(Zk)/(um or degree).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from compute_smatrix import CONFIG_TAG
from compare_ofc import DOF_LABELS, find_ofc_file, load_ofc


def snorm(vm):
    """Symmetric diverging norm, safe against vm==0."""
    vm = max(float(vm), 1e-12)
    return TwoSlopeNorm(vmin=-vm, vcenter=0.0, vmax=vm)

# pupil Noll labels (index 0 -> Noll 1)
PUPIL_NOLL = np.arange(1, 30)
DOF_TICKS = list(range(0, 50, 1))


def global_sign(ours, ofc):
    """Single scalar sign relating the two matrices (from well-agreeing DOF)."""
    num = np.sum(ours * ofc)
    return -1.0 if num < 0 else 1.0


def fig_fieldconst(ours_s, ofc, band, outdir):
    """Field-constant term = field Noll 1 (index 1); pupil-Zernike x DOF heatmaps."""
    a = ours_s[1]          # (29 pupil, 50 dof)
    b = ofc[1]
    r = a - b
    jlo = 3                # Noll 4 (defocus) and up
    mats = [a[jlo:], b[jlo:], r[jlo:]]
    titles = ["ours (sign-aligned)", "OFC", "residual (ours - OFC)"]
    vmax = np.nanmax(np.abs(np.concatenate([a[jlo:], b[jlo:]])))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, m, t in zip(axes, mats, titles):
        vm = vmax if t != titles[2] else np.nanmax(np.abs(r)) or vmax
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r",
                       norm=snorm(vm))
        ax.set_title(t)
        ax.set_yticks(range(len(PUPIL_NOLL) - jlo))
        ax.set_yticklabels([f"Z{n}" for n in PUPIL_NOLL[jlo:]], fontsize=6)
        ax.set_xticks(range(0, 50, 2))
        ax.set_xticklabels([DOF_LABELS[i] for i in range(0, 50, 2)],
                           rotation=90, fontsize=5)
        for xb in (9.5, 29.5):
            ax.axvline(xb, color="k", lw=0.6, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.8, label="um wavefront / DOF unit")
    fig.suptitle(f"Field-constant DZ term (field Noll 1): pupil Zernike sensitivity "
                 f"per DOF — band {band}", fontsize=13)
    out = outdir / f"smatrix_ofc_fieldconst_{band}.png"
    fig.savefig(out, dpi=130); print("wrote", out)


def fig_residual(ours_s, ofc, band, outdir):
    ndof = 50
    resid = np.array([np.max(np.abs(ours_s[:, :, d] - ofc[:, :, d])) for d in range(ndof)])
    scale = np.array([np.max(np.abs(ofc[:, :, d])) for d in range(ndof)])
    rel = resid / np.where(scale > 0, scale, np.nan)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    x = np.arange(ndof)
    ax0.bar(x, resid, color="slateblue")
    ax0.set_ylabel("max|resid| (µm)")
    ax0.set_title(f"Residual after single global sign (ours = {SIGN_STR}·OFC) — band {band}")
    ax1.bar(x, rel, color="teal")
    ax1.set_ylabel("max|resid| / max|OFC|")
    ax1.set_ylim(0, max(0.05, np.nanmax(rel) * 1.1))
    ax1.set_xticks(x); ax1.set_xticklabels(DOF_LABELS, rotation=90, fontsize=6)
    for ax in (ax0, ax1):
        for xb in (9.5, 29.5):
            ax.axvline(xb, color="k", lw=0.6, alpha=0.4)
    ax1.set_xlabel("DOF  |  0-9 rigid · 10-29 M1M3 B1-20 · 30-49 M2 B1-20")
    fig.tight_layout()
    out = outdir / f"smatrix_ofc_residual_{band}.png"
    fig.savefig(out, dpi=130); print("wrote", out)


def fig_examples(ours_s, ofc, band, outdir, dofs=(3, 10, 49)):
    n = len(dofs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.4 * n), constrained_layout=True)
    for row, d in enumerate(dofs):
        a, b = ours_s[:, :, d], ofc[:, :, d]
        r = a - b
        vmax = np.nanmax(np.abs(np.concatenate([a, b]))) or 1.0
        for col, (m, t) in enumerate(zip([a, b, r], ["ours", "OFC", "resid"])):
            ax = axes[row, col] if n > 1 else axes[col]
            vm = vmax if col < 2 else (np.nanmax(np.abs(r)) or vmax)
            im = ax.imshow(m, aspect="auto", cmap="RdBu_r",
                           norm=snorm(vm))
            ax.set_title(f"{DOF_LABELS[d]}: {t}", fontsize=9)
            ax.set_xlabel("pupil Noll idx"); ax.set_ylabel("field Noll idx")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Example DZ sensitivity maps (field_k x pupil_j) — band {band}",
                 fontsize=13)
    out = outdir / f"smatrix_ofc_examples_{band}.png"
    fig.savefig(out, dpi=130); print("wrote", out)


def main():
    global SIGN_STR
    p = argparse.ArgumentParser()
    p.add_argument("--band", default="r")
    p.add_argument("--ofc-file", default=None)
    args = p.parse_args()
    outdir = Path(__file__).resolve().parent.parent / "output"
    ours = np.load(outdir / f"smatrix_dz_31_29_50_{args.band}_{CONFIG_TAG}.npy")
    ofc = load_ofc(find_ofc_file(args.ofc_file))
    s = global_sign(ours, ofc)
    SIGN_STR = "-" if s < 0 else "+"
    ours_s = s * ours
    print(f"global sign: ours = {SIGN_STR}OFC")
    fig_fieldconst(ours_s, ofc, args.band, outdir)
    fig_residual(ours_s, ofc, args.band, outdir)
    fig_examples(ours_s, ofc, args.band, outdir)


if __name__ == "__main__":
    main()
