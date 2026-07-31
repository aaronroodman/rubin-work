#!/usr/bin/env python
"""Which mirror bending modes contribute to pupil astigmatism/coma (Z5-Z8)?

A pupil-Zernike slice through the DZ sensitivity matrix, FORCE-WEIGHTED by each
mode's allowed range.  For a chosen pupil Noll term j, each DOF's sensitivity
sens[:, j, dof] is a field-Zernike expansion giving the pupil-Zj wavefront (um)
vs focal-plane position.  Each mode is dialed to a fixed fraction of its
force-limited range r (default fraction 1.0), so the plotted quantity is the
peak |Zj| over the field for that realistic offset:

    contribution_j(mode) = peak_field |Zj sens per um| * range_fraction * r_mode

r comes from the geometric-normalization range weights (full_zemax, ZEMAX
force).  This answers: within its force budget, how much Zj can each mode add?
For every mirror bending mode we compute this contribution and:

  - top panel: the per-mode contribution landscape (all 153 M1M3 + 69 M2 modes),
    with the AOS-used 20 marked and the threshold line;
  - lower panels: focal-plane maps of Zj for the strongest contributors.

Motivation: the measured intrinsic wavefront (MIW) shows larger/different Z5-Z8
than the batoid model.  This shows whether small offsets in (possibly
higher-order) mirror modes could produce such contributions and with what
focal-plane pattern.  full_zemax basis; rigid-body DOF excluded.

Output: ../output/pupil_zernike_study.pdf  (one page per pupil Noll)
"""
import argparse
from pathlib import Path

import numpy as np
import galsim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent.parent / "output"
FR = 1.75
N_RIGID, N_M1M3, N_M2 = 10, 153, 69
AOS_M1M3 = set(range(19)) | {26}
AOS_M2 = set(range(17)) | {25, 26, 27}


def dof_label(idof):
    b = idof - N_RIGID
    if b < N_M1M3:
        return f"M1M3 B{b+1}", (b in AOS_M1M3)
    b -= N_M1M3
    return f"M2 B{b+1}", (b in AOS_M2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="min peak |Zj| over field (um WF) at range_fraction*range")
    ap.add_argument("--range-fraction", type=float, default=1.0,
                    help="fraction of each mode's allowed (force-limited) range")
    ap.add_argument("--pupil", type=int, nargs="+", default=list(range(4, 27)),
                    help="pupil Noll terms (default Z4-Z26, incl. Z20,Z21)")
    ap.add_argument("--n-maps", type=int, default=18, help="# top maps to show")
    args = ap.parse_args()

    sens = np.load(OUT / "smatrix_dz_31_29_232_r_bend_zemax.npy")
    # per-mode force-limited range (ZEMAX force), bending DOF only
    r_all = np.load(OUT / "normalization_all.npz")["full_zemax_r"]   # (232,)
    r_bend = r_all[N_RIGID:] * args.range_fraction                    # (222,)
    t = np.linspace(-FR, FR, 120); X, Y = np.meshgrid(t, t)
    inside = np.hypot(X, Y) <= FR
    bend = np.arange(N_RIGID, N_RIGID + N_M1M3 + N_M2)
    m1m3 = bend[:N_M1M3]; m2 = bend[N_M1M3:]

    def zmap(idof, j):
        m = galsim.zernike.Zernike(sens[:, j, idof], R_outer=FR)(X, Y)
        m[~inside] = np.nan
        return m

    with PdfPages(OUT / "pupil_zernike_study.pdf") as pdf:
        for j in args.pupil:
            peak_raw = np.array([np.nanmax(np.abs(zmap(i, j))) for i in bend])  # µm/µm
            peak = peak_raw * r_bend                       # µm WF at range_fraction*range
            sel = peak >= args.threshold
            n_aos = sum(dof_label(bend[k])[1] for k in np.where(sel)[0])
            n_ho = int(sel.sum()) - n_aos
            top = bend[np.argsort(-peak)[:args.n_maps]]
            print(f"Z{j}: {int(sel.sum())}/{len(bend)} modes >= {args.threshold} µm WF "
                  f"(rf={args.range_fraction}) ({n_aos} AOS-20, {n_ho} higher-order); "
                  f"top: {', '.join(dof_label(d)[0] for d in top[:6])}")

            nrow_maps = int(np.ceil(args.n_maps / 6))
            fig = plt.figure(figsize=(15, 3 + 2.4 * nrow_maps))
            gs = GridSpec(nrow_maps + 1, 6, figure=fig, height_ratios=[1.4] + [1] * nrow_maps)

            # --- landscape bar chart ---
            axb = fig.add_subplot(gs[0, :])
            axb.bar(np.arange(1, N_M1M3 + 1), peak[:N_M1M3], width=1.0,
                    color=["crimson" if (bend[k] - N_RIGID) in AOS_M1M3 else "steelblue"
                           for k in range(N_M1M3)], label="M1M3 (red=AOS-20)")
            axb.bar(np.arange(1, N_M2 + 1) + N_M1M3 + 5, peak[N_M1M3:], width=1.0,
                    color=["crimson" if (bend[N_M1M3 + k] - N_RIGID - N_M1M3) in AOS_M2
                           else "darkorange" for k in range(N_M2)],
                    label="M2 (red=AOS-20)")
            axb.axhline(args.threshold, color="k", ls="--", lw=1,
                        label=f"threshold {args.threshold} µm WF")
            axb.set_yscale("log")
            axb.set_ylim(1e-4, 0.5)
            axb.set_xlabel("mode number   (M1M3 B1-153  |  gap  |  M2 B1-69)")
            axb.set_ylabel(f"peak |Z{j}| over field (µm WF)\nat {args.range_fraction}×range")
            # legend just above the panel (outside the data) so it never obscures bars
            axb.legend(fontsize=8, ncol=4, loc="lower center",
                       bbox_to_anchor=(0.5, 1.0), frameon=False)

            # --- top-contributor focal-plane maps ---
            for k, idof in enumerate(top):
                ax = fig.add_subplot(gs[1 + k // 6, k % 6])
                m = zmap(idof, j) * r_bend[idof - N_RIGID]     # µm WF at range_fraction*range
                vmax = float(np.nanmax(np.abs(m)))
                vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1e-9
                im = ax.imshow(m, origin="lower", cmap="RdBu_r",
                               norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                               extent=[-FR, FR, -FR, FR])
                lab, aos = dof_label(idof)
                ax.set_title(f"{lab}{' *' if aos else ''}  {vmax*1e3:.0f}nm",
                             fontsize=8, color=("k" if aos else "crimson"))
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                fig.colorbar(im, ax=ax, shrink=0.7)

            fig.suptitle(f"Pupil Z{j} at {args.range_fraction}×(force-limited range) per "
                         f"mirror mode — {int(sel.sum())} modes >{args.threshold}µm "
                         f"({n_aos} AOS-20, {n_ho} higher-order); top {len(top)} mapped "
                         f"[full_zemax]", fontsize=12, y=0.997)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            pdf.savefig(fig); plt.close(fig)
    print("wrote", OUT / "pupil_zernike_study.pdf")


if __name__ == "__main__":
    main()
