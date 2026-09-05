#!/usr/bin/env python
"""Is the MIW's high-field-order astigmatism the UN-SUBTRACTED k>6 wavefront?

The build recovers a DOF state from the k<=6 fit, then subtracts only the k<=6
part of that state's wavefront.  The k>6 part of the SAME state is predictable
from the full 31-field-order ts_ofc sensitivity.  Here we form it for the build's
own mean state and compare, map to map, with the MIW.
"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # aos/code (siblings)
from lsst.ts.ofc import OFCData
import galsim.zernike as gz
from lsst.ts.intrinsic.wavefront import ofc_svd as osv
from recompute_coadd_metrics import umode_reference  # noqa: E402

# aos/output, relative to this file — works on RSP, s3df batch, and the laptop alike.
_OUT = os.environ.get(
    "AOS_OUTPUT", str(pathlib.Path(__file__).resolve().parents[1] / "output"))

ZK = [z for z in range(4, 27) if z not in (20, 21)]
K_MIN, K_MAX, N_KEEP, K_ALL = 1, 6, 34, 30
FP_R = 1.8                                   # fp_radius_basis (mi_config)
Z_SHOW = [5, 6, 7, 8]
MIW = (f"{_OUT}/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
       "pathA_50_34_i_5rot/intrinsic_split_maps.parquet")
CO = f"{_OUT}/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/coadd_50_34"

S = np.nan_to_num(np.asarray(OFCData("lsst").sensitivity_matrix, float))[:, ZK, :]
svd = osv.build_ofc_svd(ZK, K_MIN, K_MAX, N_KEEP)
V, Sig, N = svd.V, svd.Sigma, svd.normalization_weights

# --- the build's own mean u-mode state -> DOF -> full 31-order wavefront ---
d = np.load(f"{CO}/block_grids.npz", allow_pickle=True)
mt = pd.read_parquet(f"{CO}/coadd_metrics_rebin3.parquet")
Um = np.asarray(d["umodes"], float)
u_ref, sig_b, _, src = umode_reference(
    Um, mt["build_used"].to_numpy(bool), np.asarray(d["n_visits"], float))
dof = N * (V[:, :N_KEEP] @ (u_ref / Sig[:N_KEEP]))          # recover_dof_per_visit
w_full = np.tensordot(S, dof, axes=([2], [0]))              # (31, 21) all field orders
print(f"build state from {src}")
print(f"  |w| k<=6 = {np.linalg.norm(w_full[K_MIN:K_MAX+1]):.4f} um   "
      f"|w| k>6 = {np.linalg.norm(w_full[K_MAX+1:]):.4f} um  "
      f"(ratio {np.linalg.norm(w_full[K_MAX+1:])/np.linalg.norm(w_full[K_MIN:K_MAX+1]):.3f})")

# --- evaluate the k>6 (never-subtracted) part on the MIW field grid ---
miw = pd.read_parquet(MIW)
thx, thy = miw["thx_deg"].to_numpy(float), miw["thy_deg"].to_numpy(float)
u, v = thx / FP_R, thy / FP_R
basis = np.array([gz.Zernike([0] * k + [1]).evalCartesian(u, v)
                  for k in range(1, K_ALL + 1)])              # (30, npts), k=1..30
hi = slice(K_MAX, K_ALL)                                      # rows for k=7..30
leak = {j: basis[hi].T @ w_full[K_MAX + 1:, i] for i, j in enumerate(ZK)}
lo6 = {j: basis[:K_MAX].T @ w_full[K_MIN:K_MAX + 1, i] for i, j in enumerate(ZK)}

print(f"\n{'Zj':>4} {'MIW rms':>9} {'k>6 leak rms':>13} {'frac':>7} "
      f"{'corr(leak,MIW)':>15} {'corr(k<=6,MIW)':>15}")
rows = []
for j in Z_SHOW:
    m = miw[f"Z{j}_OCS"].to_numpy(float)
    ok = np.isfinite(m)
    lk, l6 = leak[j], lo6[j]
    rm, rl = np.sqrt(np.nanmean(m[ok] ** 2)), np.sqrt(np.mean(lk[ok] ** 2))
    c1 = float(np.corrcoef(lk[ok], m[ok])[0, 1])
    c6 = float(np.corrcoef(l6[ok], m[ok])[0, 1])
    rows.append((j, m, lk, ok, rm, rl, c1))
    print(f"{'Z'+str(j):>4} {rm:9.4f} {rl:13.4f} {rl/rm:7.3f} {c1:+15.3f} {c6:+15.3f}")

with PdfPages(f"{_OUT}/miw_k_gt6_leakage.pdf") as pdf:
    fig, ax = plt.subplots(3, len(Z_SHOW), figsize=(4.1 * len(Z_SHOW), 11))
    for c, (j, m, lk, ok, rm, rl, c1) in enumerate(rows):
        for r, (lab, val) in enumerate((("MIW", m), ("un-subtracted k>6", lk),
                                        ("MIW - k>6", m - lk))):
            vmax = np.nanpercentile(np.abs(m[ok]), 99)
            s = ax[r][c].scatter(thx, thy, c=val, s=4, cmap="RdBu_r",
                                 vmin=-vmax, vmax=vmax)
            ax[r][c].set_aspect("equal")
            rr = np.sqrt(np.nanmean(np.asarray(val)[ok] ** 2))
            ax[r][c].set_title(f"Z{j} {lab}  rms={rr:.4f}"
                               + (f"  r={c1:+.2f}" if r == 1 else ""), fontsize=9)
            plt.colorbar(s, ax=ax[r][c], fraction=0.046)
    fig.suptitle("Is the MIW the un-subtracted k>6 wavefront of the build's own "
                 "optical state?\ntop: MIW (OCS).  middle: k>6 response of the "
                 "recovered DOF, never subtracted by the k<=6 build.  bottom: "
                 "difference", fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
print("\nwrote aos/output/miw_k_gt6_leakage.pdf")
