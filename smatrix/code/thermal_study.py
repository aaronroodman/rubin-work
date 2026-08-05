#!/usr/bin/env python
"""Thermal-gradient wavefront study: coupling, field maps, and MIW astig/coma.

Reads thermal_sensitivity.npz (7 thermal DOF DZ sensitivity, um WF per unit
gradient).  Produces ../output/thermal_study.pdf:

  p1  pupil-Zernike coupling heatmap (pupil Noll x thermal DOF, field-constant)
  p2  focal-plane maps of the dominant Zernike for each thermal DOF
  p3  astigmatism/coma (Z5-Z8) peak per gradient + gradient needed for MIW scale
"""
from pathlib import Path
import numpy as np
import galsim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent.parent / "output"
FR = 1.75
MIW = 0.05    # µm, representative MIW-scale contribution


def zmap(field_zk, X, Y, inside):
    m = galsim.zernike.Zernike(field_zk, R_outer=FR)(X, Y)
    m[~inside] = np.nan
    return m


def main():
    d = np.load(OUT / "thermal_sensitivity.npz", allow_pickle=True)
    sens = d["sens"]                      # (31,29,7)
    labels = list(d["labels"]); units = list(d["units"])
    ndof = sens.shape[2]
    t = np.linspace(-FR, FR, 120); X, Y = np.meshgrid(t, t); inside = np.hypot(X, Y) <= FR

    with PdfPages(OUT / "thermal_study.pdf") as pdf:
        # --- p1: coupling heatmap (field-constant pupil Noll x thermal DOF) ---
        pup = np.arange(4, 29)
        block = sens[1, 4:29, :]          # (npupil, ndof)
        vmax = np.nanmax(np.abs(block))
        fig, ax = plt.subplots(figsize=(11, 8))
        im = ax.imshow(block, aspect="auto", cmap="RdBu_r",
                       norm=SymLogNorm(linthresh=1e-3, vmin=-vmax, vmax=vmax))
        ax.set_xticks(range(ndof)); ax.set_xticklabels(
            [f"{l}\n({u})" for l, u in zip(labels, units)], fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(pup))); ax.set_yticklabels([f"Z{j}" for j in pup], fontsize=7)
        ax.set_ylabel("pupil Noll (field-constant term)")
        ax.set_title("Thermal-gradient -> pupil-Zernike coupling (µm WF per unit gradient, symlog)")
        for j in (4, 7, 8, 11, 22):
            ax.axhline(j - 4, color="k", lw=0.3, alpha=0.3)
        fig.colorbar(im, ax=ax, shrink=0.8, label="µm WF / unit gradient")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # --- p2: focal-plane map of the dominant (non piston/tilt) Zernike per DOF ---
        fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.ravel()
        for ax in axes:
            ax.axis("off")
        for k in range(ndof):
            fc = sens[1, :, k].copy(); fc[:4] = 0        # ignore piston/tilt
            j = int(np.argmax(np.abs(fc)))
            m = zmap(sens[:, j, k], X, Y, inside); vmax = max(np.nanmax(np.abs(m)), 1e-12)
            ax = axes[k]; ax.axis("on")
            im = ax.imshow(m, origin="lower", cmap="RdBu_r",
                           norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                           extent=[-FR, FR, -FR, FR])
            ax.set_title(f"{labels[k]}: Z{j}\npk {vmax:.3g} µm / {units[k]}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle("Focal-plane map of the dominant pupil Zernike per thermal gradient",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96)); pdf.savefig(fig); plt.close(fig)

        # --- p3: astig/coma (Z5-Z8) peak over field per gradient ---
        terms = [5, 6, 7, 8]
        peak = np.zeros((len(terms), ndof))
        for a, j in enumerate(terms):
            for k in range(ndof):
                peak[a, k] = np.nanmax(np.abs(zmap(sens[:, j, k], X, Y, inside)))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))
        xx = np.arange(ndof)
        for a, j in enumerate(terms):
            ax0.bar(xx + a * 0.2 - 0.3, peak[a], width=0.2, label=f"Z{j}")
        ax0.set_yscale("log"); ax0.set_xticks(xx)
        ax0.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax0.set_ylabel("peak |Zj| over field (µm WF / unit gradient)")
        ax0.set_title("Astigmatism/coma per thermal gradient"); ax0.legend()
        ax0.axhline(MIW, color="k", ls="--", lw=1)
        # gradient needed for MIW-scale (0.05 µm) of each term
        grad_needed = MIW / np.where(peak > 0, peak, np.nan)
        im = ax1.imshow(grad_needed, aspect="auto", cmap="viridis_r",
                        norm=SymLogNorm(linthresh=0.01))
        ax1.set_yticks(range(len(terms))); ax1.set_yticklabels([f"Z{j}" for j in terms])
        ax1.set_xticks(range(ndof)); ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax1.set_title(f"gradient needed for {MIW} µm of each term  (unit = C or C/m)")
        for (yy, xx2), v in np.ndenumerate(grad_needed):
            if np.isfinite(v):
                ax1.text(xx2, yy, f"{v:.2g}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax1, shrink=0.8, label="gradient (C or C/m)")
        fig.suptitle("Can thermal gradients produce MIW-scale astigmatism/coma?", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)

    print("wrote", OUT / "thermal_study.pdf")


if __name__ == "__main__":
    main()
