#!/usr/bin/env python
"""Fit the MIW to the FULL focal-plane Zernike basis k=1..K and show how much of it
lives above the k<=6 the build actually fits.

Motivation.  The per-visit DZ fit uses only focal orders k=1..6, because that is all
the optical-state determination needs.  Every downstream subspace I have built --
including n_null, the "unreachable" part of w -- therefore lives entirely inside
k<=6 and is structurally blind to higher focal orders.  But the MIW plainly has
structure at higher k, and that is where the interesting behaviour is likely to be.
This script quantifies that: fit each MIW pupil Zernike map Z_j(theta) [µm] to focal
Zernikes k=1..K and report how much MIW POWER each order captures.

Reference marks on every plot:
  k = 6   the build's fit limit (field radial order n<=2: piston, tip/tilt, focus,
          two astigmatisms).  Everything above this is invisible to the current
          optical-state fit AND to n_null.
  k = 30  the highest focal order tabulated in the ts_ofc DZ sensitivity (its field
          axis has length 31 with index 0 unused).  Above this the sensitivity
          matrix has nothing to say at all, so no DOF-based subspace -- reachable or
          null -- is even defined there.

Uses the OCS columns (for Z4 the OCS piece, as intended: the CCS piece is the CCD
height / camera-moving term).  Fit is unweighted least squares over the finite grid
cells; the MIW grid is a per-cell donut median so cell weights vary somewhat.

  python code/analyze_miw_dz_full_k.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import galsim.zernike as gz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ZK = [z for z in range(4, 27) if z not in (20, 21)]      # 21 pupil Noll indices
K_FIT = 6        # what the per-visit build fits
K_OFC = 30       # highest focal order in the ts_ofc DZ sensitivity
# Noll k at which each focal radial order n starts: n(n+1)/2 + 1
N_START = {n: n * (n + 1) // 2 + 1 for n in range(0, 12)}


def noll_radial_order(k):
    n = 0
    while N_START.get(n + 1, 10 ** 9) <= k:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miw", default="output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
                    "pathA_50_34_i_5rot/intrinsic_split_maps.parquet")
    ap.add_argument("--kmax", type=int, default=45,
                    help="highest focal Noll index to fit (45 completes radial order n=8)")
    ap.add_argument("--fp-radius", type=float, default=1.8,
                    help="focal-plane normalization radius [deg] (fp_radius_basis)")
    ap.add_argument("--coord", default="OCS")
    ap.add_argument("--out", default="output/miw_dz_full_k.pdf")
    args = ap.parse_args()

    df = pd.read_parquet(args.miw)
    thx = df["thx_deg"].to_numpy(float)
    thy = df["thy_deg"].to_numpy(float)
    u, v = thx / args.fp_radius, thy / args.fp_radius
    K = args.kmax
    B = np.array([gz.Zernike([0] * k + [1]).evalCartesian(u, v)
                  for k in range(1, K + 1)])                 # (K, n_pts)
    print(f"MIW: {args.miw}\n  {len(df)} field cells, fp_radius {args.fp_radius} deg, "
          f"fitting focal Noll k=1..{K} ({args.coord} columns)")

    W = np.full((len(ZK), K), np.nan)        # coefficients [µm]
    rows = []
    maps = {}
    for i, j in enumerate(ZK):
        m = df[f"Z{j}_{args.coord}"].to_numpy(float)
        ok = np.isfinite(m)
        c, *_ = np.linalg.lstsq(B[:, ok].T, m[ok], rcond=None)
        W[i] = c
        tot = float(np.sum(m[ok] ** 2))                       # total MIW power [µm^2]
        # cumulative power captured as a function of k_max
        cum = np.array([float(np.sum((B[:kk, ok].T @ c[:kk]) ** 2)) for kk in range(1, K + 1)])
        # cross terms matter (basis not orthonormal on this masked grid), so measure
        # capture as 1 - residual power, which is the honest "how much is explained"
        capt = np.array([1.0 - np.sum((m[ok] - B[:kk, ok].T @ c[:kk]) ** 2) / tot
                         for kk in range(1, K + 1)])
        maps[j] = (m, ok, c, capt)
        rows.append(dict(j=j, rms_um=np.sqrt(np.mean(m[ok] ** 2)),
                         f_k6=capt[K_FIT - 1], f_k30=capt[min(K_OFC, K) - 1],
                         f_kmax=capt[-1]))
        print(f"  Z{j:<3d} MIW RMS {rows[-1]['rms_um']:.4f} µm | power captured: "
              f"k<=6 {capt[K_FIT-1]:6.3f}   k<=30 {capt[min(K_OFC,K)-1]:6.3f}   "
              f"k<={K} {capt[-1]:6.3f}")

    R = pd.DataFrame(rows)
    print("\n--- pooled over the 21 pupil Zernikes (power-weighted) ---")
    wgt = R["rms_um"].to_numpy() ** 2
    for lab, col in (("k<=6  (the build's fit)", "f_k6"),
                     (f"k<=30 (ts_ofc DZ limit)", "f_k30"),
                     (f"k<={K}", "f_kmax")):
        print(f"  fraction of total MIW POWER captured by {lab}: "
              f"{np.sum(wgt * R[col]) / wgt.sum():.3f}")
    print(f"  => MIW power ABOVE k=6 but within the ts_ofc range: "
          f"{(np.sum(wgt*R['f_k30'])-np.sum(wgt*R['f_k6']))/wgt.sum():.3f}")
    print(f"  => MIW power above k=30 (outside ts_ofc entirely): "
          f"{(np.sum(wgt*R['f_kmax'])-np.sum(wgt*R['f_k30']))/wgt.sum():.3f}")
    print(f"  => MIW power not captured even by k<={K}: "
          f"{1 - np.sum(wgt*R['f_kmax'])/wgt.sum():.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kk = np.arange(1, K + 1)
    with PdfPages(args.out) as pdf:
        # --- page 1: cumulative power captured vs k_max ---
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
        for j in ZK:
            lw = 2.0 if j in (5, 6, 7, 8) else 0.8
            ax[0].plot(kk, maps[j][3], lw=lw,
                       label=f"Z{j}" if j in (5, 6, 7, 8, 11) else None)
        pooled = (np.array([maps[j][3] for j in ZK]).T * wgt).sum(1) / wgt.sum()
        ax[0].plot(kk, pooled, "k-", lw=3, label="pooled (power-weighted)")
        for kx, lab, c in ((K_FIT, "k=6 build fit", "tab:red"),
                           (K_OFC, "k=30 ts_ofc limit", "tab:blue")):
            ax[0].axvline(kx, color=c, ls="--", lw=1.2, label=lab)
        ax[0].set_xlabel("focal-plane Zernike order $k_{max}$ (Noll)")
        ax[0].set_ylabel("fraction of MIW POWER captured (dimensionless)")
        ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax[0].set_title("cumulative capture of the MIW vs focal order")
        # incremental power per radial order n, pooled
        ords = np.array([noll_radial_order(k) for k in kk])
        inc = np.diff(np.concatenate([[0.0], pooled]))
        by_n = {n: inc[ords == n].sum() for n in sorted(set(ords))}
        ax[1].bar(list(by_n), list(by_n.values()), color="tab:purple")
        ax[1].set_xlabel("focal-plane radial order $n$")
        ax[1].set_ylabel("incremental fraction of MIW power (dimensionless)")
        ax[1].set_title("where the MIW power actually sits\n"
                        "(the build fits only $n\\leq2$, i.e. $k\\leq6$)")
        ax[1].grid(alpha=0.3, axis="y")
        fig.suptitle("MIW fitted to the full focal-plane Zernike basis "
                     f"(OCS, {len(df)} cells, fp_radius {args.fp_radius} deg)",
                     fontsize=12)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # --- page 2: the coefficient matrix w_kj [µm] ---
        fig, a = plt.subplots(figsize=(15, 7))
        vmax = np.nanpercentile(np.abs(W), 99)
        im = a.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      extent=[0.5, K + 0.5, len(ZK) - 0.5, -0.5])
        a.set_yticks(range(len(ZK))); a.set_yticklabels([f"Z{j}" for j in ZK], fontsize=8)
        a.set_xlabel("focal-plane Zernike Noll index $k$")
        a.set_ylabel("pupil Zernike")
        for kx, c in ((K_FIT + 0.5, "tab:red"), (K_OFC + 0.5, "tab:blue")):
            a.axvline(kx, color=c, lw=2)
        a.set_title("MIW Double-Zernike coefficients $w_{kj}$ [µm]   "
                    "(red line = k=6 build fit limit, blue = k=30 ts_ofc limit)")
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.02, label="$w_{kj}$ [µm]")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # --- page 3: per-Z_j power split ---
        fig, a = plt.subplots(figsize=(14, 5.4))
        x = np.arange(len(ZK))
        f6 = R["f_k6"].to_numpy(); f30 = R["f_k30"].to_numpy(); fm = R["f_kmax"].to_numpy()
        a.bar(x, f6, 0.7, label="$k\\leq6$ (fitted by the build)", color="tab:red")
        a.bar(x, f30 - f6, 0.7, bottom=f6, color="tab:orange",
              label="$7\\leq k\\leq30$ (in ts_ofc, NOT fitted)")
        a.bar(x, fm - f30, 0.7, bottom=f30, color="tab:green",
              label=f"$31\\leq k\\leq{K}$ (outside ts_ofc)")
        a.bar(x, 1 - fm, 0.7, bottom=fm, color="0.7", label="unfit residual")
        a.set_xticks(x); a.set_xticklabels([f"Z{j}" for j in ZK], fontsize=8)
        a.set_ylabel("fraction of that $Z_j$'s MIW POWER (dimensionless)")
        a.set_ylim(0, 1); a.legend(fontsize=8, ncol=4, loc="lower right")
        a.set_title("Where each pupil Zernike's MIW power lives in focal order")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # --- page 4: field maps, MIW vs k<=6 vs k=7..K vs residual ---
        for j in (5, 6, 7, 8):
            m, ok, c, _ = maps[j]
            lo = B[:K_FIT].T @ c[:K_FIT]
            hi = B[K_FIT:].T @ c[K_FIT:]
            res = m - (lo + hi)
            fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
            vmax = np.nanpercentile(np.abs(m[ok]), 99)
            for c_, (lab, val) in enumerate((("MIW", m), ("$k\\leq6$ part", lo),
                                             (f"$k=7..{K}$ part", hi),
                                             ("residual", res))):
                val = np.where(ok, val, np.nan)
                s = ax[c_].scatter(thx, thy, c=val, s=4, cmap="RdBu_r",
                                   vmin=-vmax, vmax=vmax)
                ax[c_].set_aspect("equal")
                ax[c_].set_xlabel("thx [deg]")
                if c_ == 0:
                    ax[c_].set_ylabel("thy [deg]")
                ax[c_].set_title(f"{lab}   RMS={np.sqrt(np.nanmean(val[ok]**2)):.4f} µm",
                                 fontsize=10)
                plt.colorbar(s, ax=ax[c_], fraction=0.046, label="µm")
            fig.suptitle(f"MIW Z{j} ({args.coord}) split by focal-plane Zernike order",
                         fontsize=12)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    outpq = os.path.splitext(args.out)[0] + "_coeffs.parquet"
    pd.DataFrame(W, index=[f"Z{j}" for j in ZK],
                 columns=[f"k{k}" for k in kk]).to_parquet(outpq)
    print(f"\nwrote {args.out}\nwrote {outpq}  (w_kj coefficients [µm], 21 x {K})")


if __name__ == "__main__":
    main()
