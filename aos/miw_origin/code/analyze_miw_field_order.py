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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))          # same-study siblings
from lsst.ts.ofc import OFCData
import galsim.zernike as gz
from lsst.ts.intrinsic.wavefront import ofc_svd as osv
from recompute_coadd_metrics import (          # noqa: E402
    COADD_REQUIRED, old_tree_coadd_dir, old_tree_miw_dir, resolve_product_dir,
    umode_reference)

from rubinwork.products import catalog  # noqa: E402

ZK = [z for z in range(4, 27) if z not in (20, 21)]
K_MIN, K_MAX, N_KEEP, K_ALL = 1, 6, 34, 30
FP_R = 1.8                                   # fp_radius_basis (mi_config)
Z_SHOW = [5, 6, 7, 8]

# This script has no argument parser -- it runs top to bottom -- so the two builds it
# reads are named by environment variable, defaulting to the production pair.  Before
# phase 3a both were `<aos/output>/<long param_set>/...` paths built from the long key, a
# layout the tree lost when dir_name arrived, so neither resolved.  Each is now resolved
# in the aos/Snakefile order: the catalog's build if complete, else the old aos/output
# tree, else a named error.  COADD_DIR and MIW_DIR override a resolution
# outright, the equivalent of the sibling scripts' --coadd-dir.
MIW_VARIANT = os.environ.get("COADD_MIW_VARIANT", "d12-A_50_34_i_5rot")
COADD_VARIANT = os.environ.get("COADD_VARIANT", "d12-A_50_34_i_5rot")
MIW_BUILD = os.environ.get("COADD_MIW_BUILD", catalog.CURRENT)
COADD_BUILD = os.environ.get("COADD_BUILD", catalog.CURRENT)
MIW_REQUIRED = ("intrinsic_split_maps.parquet",)

_miw_dir = os.environ.get("MIW_DIR") or resolve_product_dir(
    "miw", MIW_VARIANT, old_tree_miw_dir(MIW_VARIANT), MIW_REQUIRED, build=MIW_BUILD)
MIW = os.path.join(_miw_dir, "intrinsic_split_maps.parquet")
CO = os.environ.get("COADD_DIR") or resolve_product_dir(
    "coadds", COADD_VARIANT, old_tree_coadd_dir(COADD_VARIANT), COADD_REQUIRED,
    build=COADD_BUILD)

# The rebinned metrics are a STUDY output of recompute_coadd_metrics.py, not part of the
# coadd build, so they are not in CO.  COADD_STUDY_DIR says where that script wrote them.
STUDY = os.environ.get("COADD_STUDY_DIR", CO)

# Where this script's own plot goes: the study output@, relative to this file, which works
# on the RSP, on an s3df batch node and on the laptop alike.
_OUT = os.environ.get(
    "AOS_OUTPUT", str(pathlib.Path(__file__).resolve().parents[1] / "output"))

S = np.nan_to_num(np.asarray(OFCData("lsst").sensitivity_matrix, float))[:, ZK, :]
svd = osv.build_ofc_svd(ZK, K_MIN, K_MAX, N_KEEP)
V, Sig, N = svd.V, svd.Sigma, svd.normalization_weights

# --- the build's own mean u-mode state -> DOF -> full 31-order wavefront ---
_npz = f"{CO}/block_grids.npz"
_metrics = f"{STUDY}/coadd_metrics_rebin3.parquet"
d = np.load(_npz, allow_pickle=True)
mt = pd.read_parquet(_metrics)
Um = np.asarray(d["umodes"], float)
# The metrics table is masked by a boolean array sized from the npz, so the two must
# come from the SAME run.  The usual cause of a mismatch is a differing band selection:
# run_coadd_blocks_miw.py's --bands inherits mi_config.yaml `defaults: filter: [i]` when
# omitted, and 2025 FAM is mostly r-band, so an i-band run builds far fewer 2025 blocks
# than an all-band one (130 vs 216 blocks on the current chunk tables).
if len(mt) != len(Um):
    raise SystemExit(
        f"block count mismatch -- these two products are from different runs:\n"
        f"  {_npz}\n      umodes {Um.shape[0]} blocks\n"
        f"  {_metrics}\n      {len(mt)} rows\n"
        f"Regenerate both from one run (same --bands): rerun run_coadd_blocks_miw.sbatch, "
        f"then recompute_coadd_metrics.py on its block_grids.npz.")
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
print(f"\nwrote {_OUT}/miw_k_gt6_leakage.pdf")
