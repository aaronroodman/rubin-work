#!/usr/bin/env python
"""Recompute coadd-vs-MIW metrics on a COARSER (rebinned) grid and remake the
time-series metrics page -- WITHOUT overwriting any existing output.

Why: the native fine-grid Pearson r / residual RMS are noise-limited (few donuts
per cell + large single-donut wavefront scatter), so they understate the real
structural agreement.  Rebinning the saved block grids by --rebin before the
metric averages that noise down and reveals the true coadd-vs-MIW agreement.

Reads ONLY block_grids.npz (grids + miw + metadata already saved by
run_coadd_blocks_miw.py), so it needs no LSST stack -- runs on the laptop or RSP
with numpy/pandas/matplotlib.

Writes (new files; originals untouched):
  <coadd_dir>/coadd_metrics_rebin<f>.parquet     -- per-coadd rebinned metrics
  <coadd_dir>/coadd_timeseries_rebin<f>.pdf       -- the metrics-vs-ordinal page
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

Z_TERMS = [5, 6, 7, 8]


def coarsen(A, f):
    """Block-mean an (ny, nx) map by factor f (NaN-aware), truncating remainders."""
    ny, nx = (A.shape[0] // f) * f, (A.shape[1] // f) * f
    if ny == 0 or nx == 0:
        return A.copy()
    return np.nanmean(A[:ny, :nx].reshape(ny // f, f, nx // f, f), axis=(1, 3))


def coarsen_centers(c, f):
    n = (len(c) // f) * f
    return c[:n].reshape(-1, f).mean(1)


def pear(a, b, mask=None):
    a = a.ravel(); b = b.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok = ok & mask.ravel()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if int(ok.sum()) > 5 else np.nan


def block_metrics(cg, wg, cwfs):
    """Rebinned coadd (cg[z]) vs MIW (wg[z]) metrics: per-Z + combined resid RMS
    and Pearson r, over the full FP and the CWFS annulus."""
    m = {}
    diffs, rs, cwd, cwr = [], [], [], []
    for z in Z_TERMS:
        c, w = cg[z], wg[z]
        ok = np.isfinite(c) & np.isfinite(w)
        d = (c - w)[ok]
        m[f"resid_rms_z{z}"] = float(np.sqrt(np.mean(d ** 2))) if ok.sum() >= 5 else np.nan
        m[f"corr_z{z}"] = pear(c, w)
        if ok.sum() >= 5:
            diffs.append(d); rs.append(m[f"corr_z{z}"])
        okc = ok & cwfs
        if okc.sum() >= 5:
            cwd.append((c - w)[okc]); cwr.append(pear(c, w, mask=cwfs))
    m["resid_rms_combined"] = float(np.sqrt(np.mean(np.concatenate(diffs) ** 2))) if diffs else np.nan
    m["corr_combined"] = float(np.nanmean(rs)) if rs else np.nan
    m["cwfs_resid_rms_combined"] = float(np.sqrt(np.mean(np.concatenate(cwd) ** 2))) if cwd else np.nan
    m["cwfs_corr_combined"] = float(np.nanmean(cwr)) if cwr else np.nan
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-set", default="fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x")
    ap.add_argument("--mi-name", default="pathA_50_34_i_5rot",
                    help="(not used for the path; coadd dir is <ps>/<out_name>)")
    ap.add_argument("--out-name", default="coadd_50_34")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--npz", default=None, help="override path to block_grids.npz")
    ap.add_argument("--rebin", type=int, default=3)
    ap.add_argument("--cwfs-inner", type=float, default=1.59)
    ap.add_argument("--cwfs-outer", type=float, default=1.725)
    args = ap.parse_args()

    # coadd output dir = output/<param_set>/<out_name> (NOT nested under mi_name;
    # mi_name only selects the sidecar/build, not the output location)
    cdir = os.path.join(args.output_root, args.param_set, args.out_name)
    npz = args.npz or os.path.join(cdir, "block_grids.npz")
    d = np.load(npz, allow_pickle=True)
    grids, miw = d["grids"], d["miw"]        # (n, 4, ny, nx), term order = d["terms"]
    terms = [int(t) for t in d["terms"]]
    assert terms == Z_TERMS, f"unexpected term order {terms}"
    n, f = grids.shape[0], args.rebin

    # CWFS annulus mask on the COARSE grid
    cx = 0.5 * (d["xbins"][:-1] + d["xbins"][1:])
    cy = 0.5 * (d["ybins"][:-1] + d["ybins"][1:])
    cxc, cyc = coarsen_centers(cx, f), coarsen_centers(cy, f)
    rr = np.hypot(cyc[:, None], cxc[None, :])
    cwfs = (rr >= args.cwfs_inner) & (rr <= args.cwfs_outer)

    met = []
    for i in range(n):
        cg = {z: coarsen(grids[i, k], f) for k, z in enumerate(terms)}
        wg = {z: coarsen(miw[i, k], f) for k, z in enumerate(terms)}
        met.append(block_metrics(cg, wg, cwfs))

    day = np.asarray(d["day_obs"])
    outfam = ~np.asarray(d["in_family"]).astype(bool)
    ordn = np.arange(n)

    # ---- rebinned metrics parquet (new file) ----
    base = {k: np.asarray(d[k]) for k in d.files
            if np.asarray(d[k]).ndim == 1 and np.asarray(d[k]).shape[0] == n
            and k not in ("terms",)}
    dfm = pd.DataFrame(base)
    for key in met[0]:
        dfm[key] = [mm[key] for mm in met]
    outpq = os.path.join(cdir, f"coadd_metrics_rebin{f}.parquet")
    dfm.to_parquet(outpq, index=False)
    print(f"wrote {outpq}  ({n} coadds, rebin {f}x{f})")
    print(f"  median corr_combined: native N/A -> rebinned {np.nanmedian(dfm['corr_combined']):.3f}; "
          f"median resid_rms_combined {np.nanmedian(dfm['resid_rms_combined']):.4f} um")

    # ---- big time-series metrics page (new file) ----
    W = max(12.0, 0.30 * n + 3.0)
    xlim = (-0.6, n - 0.4)
    bnd = np.where(day[1:] != day[:-1])[0] + 0.5
    day_start = [0] + list(np.where(day[1:] != day[:-1])[0] + 1)

    def decorate(ax, first=False):
        for x in bnd:
            ax.axvline(x, color="0.5", lw=0.6, ls=":")
        for i in np.where(outfam)[0]:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.07)
        ax.set_xlim(*xlim); ax.grid(alpha=0.3)
        if first:
            for i in day_start:
                ax.text(i, 0.985, str(int(day[i])), transform=ax.get_xaxis_transform(),
                        rotation=90, fontsize=6, va="top", ha="left", color="0.35")

    outpdf = os.path.join(cdir, f"coadd_timeseries_rebin{f}.pdf")
    with PdfPages(outpdf) as pdf:
        fig, ax = plt.subplots(2, 1, figsize=(W, 6.8), sharex=True)
        for z in Z_TERMS:
            ax[0].plot(ordn, [mm[f"resid_rms_z{z}"] for mm in met], "o-", ms=3, label=f"Z{z}")
        ax[0].plot(ordn, [mm["resid_rms_combined"] for mm in met], "k-", lw=2, label="combined (full FP)")
        ax[0].plot(ordn, [mm["cwfs_resid_rms_combined"] for mm in met],
                   color="tab:red", ls="--", lw=2, label="combined (CWFS annulus)")
        ax[0].set_ylabel("resid RMS (um)"); ax[0].legend(fontsize=7, ncol=6)
        for z in Z_TERMS:
            ax[1].plot(ordn, [mm[f"corr_z{z}"] for mm in met], "o-", ms=3, label=f"Z{z}")
        ax[1].plot(ordn, [mm["corr_combined"] for mm in met], "k-", lw=2, label="mean (full FP)")
        ax[1].plot(ordn, [mm["cwfs_corr_combined"] for mm in met],
                   color="tab:red", ls="--", lw=2, label="mean (CWFS annulus)")
        ax[1].set_ylabel("spatial corr r"); ax[1].legend(fontsize=7, ncol=6)
        ax[1].set_xlabel("coadd ordinal (day_obs / seq order)")
        decorate(ax[0], first=True); decorate(ax[1], first=True)
        fig.suptitle(f"Coadd-vs-MIW comparison metrics vs ordinal  (rebin {f}x{f}; "
                     "vertical = day_obs; red bands = out-of-5rot-family)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {outpdf}")


if __name__ == "__main__":
    main()
