#!/usr/bin/env python
"""Recompute coadd-vs-MIW metrics on a COARSER (rebinned) grid and remake the
FULL time-series PDF -- WITHOUT overwriting any existing output.

Why rebin: the native fine-grid Pearson r / residual RMS are noise-limited (few
donuts per cell + large single-donut wavefront scatter), so they understate the
real structural agreement.  Rebinning the saved block grids by --rebin averages
that noise down and reveals the true coadd-vs-MIW agreement (r 0.54 -> 0.74 at
3x3 on a test block).

Reads ONLY block_grids.npz (grids + miw + telemetry already saved by
run_coadd_blocks_miw.py), so it is self-contained -- numpy/pandas/matplotlib/
scipy, no LSST stack -- and runs on the laptop from the synced output.

Pages (mirrors run_coadd_blocks_miw.render_timeseries, with rebinned metrics):
  1. comparison metrics (resid RMS + spatial r) vs ordinal
  2. telemetry <-> metric Spearman correlation bars
  3. scatter of the leading telemetry vs spatial r (Theil-Sen line)
  4. block telemetry (alt/az/rot/fwhm/z_gradient/band) vs ordinal
  5. u-mode amplitude heatmap vs ordinal
Light blue shading / lines mark the blocks used to BUILD the 5rot MIW
(a build rotation in_family, within the frozen 2026 build epoch).

Writes (new files; originals untouched):
  <coadd_dir>/coadd_metrics_rebin<f>.parquet
  <coadd_dir>/coadd_timeseries_rebin<f>.pdf
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, theilslopes

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

Z_TERMS = [5, 6, 7, 8]
THERMAL_VARS = ["cam_air_temp", "m2_air_temp", "m1m3_air_temp", "outside_temp",
                "m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t",
                "x_gradient", "y_gradient", "z_gradient", "radial_gradient",
                "tma_truss_temp_pxpy", "tma_truss_temp_mxmy"]
CORR_VARS = ["alt", "rot", "fwhm"] + THERMAL_VARS
SCATTER_VARS = ["z_gradient", "y_gradient", "dome_delta_t"]


# ---------- metric recompute on the rebinned grid ----------
def coarsen(A, f):
    ny, nx = (A.shape[0] // f) * f, (A.shape[1] // f) * f
    if ny == 0 or nx == 0:
        return A.copy()
    return np.nanmean(A[:ny, :nx].reshape(ny // f, f, nx // f, f), axis=(1, 3))


def coarsen_centers(c, f):
    n = (len(c) // f) * f
    return c[:n].reshape(-1, f).mean(1)


def pear(a, b, mask=None):
    a, b = a.ravel(), b.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok = ok & mask.ravel()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if int(ok.sum()) > 5 else np.nan


def block_metrics(cg, wg, cwfs):
    m = {}; diffs, rs, cwd, cwr = [], [], [], []
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


def _spear(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 5:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-set", default="fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x")
    ap.add_argument("--out-name", default="coadd_50_34")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--npz", default=None, help="override path to block_grids.npz")
    ap.add_argument("--rebin", type=int, default=3)
    ap.add_argument("--cwfs-inner", type=float, default=1.59)
    ap.add_argument("--cwfs-outer", type=float, default=1.725)
    ap.add_argument("--build-day-min", type=int, default=20260101)
    ap.add_argument("--build-day-max", type=int, default=20260513)
    args = ap.parse_args()

    cdir = os.path.join(args.output_root, args.param_set, args.out_name)
    npz = args.npz or os.path.join(cdir, "block_grids.npz")
    d = np.load(npz, allow_pickle=True)
    grids, miw = d["grids"], d["miw"]
    terms = [int(t) for t in d["terms"]]
    assert terms == Z_TERMS, f"unexpected term order {terms}"
    n, f = grids.shape[0], args.rebin
    tele = {k: np.asarray(d[k], float) for k in CORR_VARS if k in d.files}
    band = np.asarray(d["band"]).astype(str)
    Um = np.asarray(d["umodes"], float)
    day = np.asarray(d["day_obs"]).astype(int)
    infam = np.asarray(d["in_family"]).astype(bool)
    build_used = infam & (day >= args.build_day_min) & (day <= args.build_day_max)
    ordn = np.arange(n)

    # coarse CWFS annulus mask
    cx = 0.5 * (d["xbins"][:-1] + d["xbins"][1:])
    cy = 0.5 * (d["ybins"][:-1] + d["ybins"][1:])
    cxc, cyc = coarsen_centers(cx, f), coarsen_centers(cy, f)
    rr = np.hypot(cyc[:, None], cxc[None, :])
    cwfs = (rr >= args.cwfs_inner) & (rr <= args.cwfs_outer)

    met = [block_metrics({z: coarsen(grids[i, k], f) for k, z in enumerate(terms)},
                         {z: coarsen(miw[i, k], f) for k, z in enumerate(terms)},
                         cwfs) for i in range(n)]

    def M(key):
        return np.array([mm[key] for mm in met], float)

    # ---- rebinned metrics parquet (new file) ----
    base = {k: np.asarray(d[k]) for k in d.files
            if np.asarray(d[k]).ndim == 1 and np.asarray(d[k]).shape[0] == n and k != "terms"}
    dfm = pd.DataFrame(base)
    dfm["build_used"] = build_used
    for key in met[0]:
        dfm[key] = M(key)
    outpq = os.path.join(cdir, f"coadd_metrics_rebin{f}.parquet")
    dfm.to_parquet(outpq, index=False)
    print(f"wrote {outpq}  ({n} coadds, rebin {f}x{f}; build_used={int(build_used.sum())})")
    print(f"  median corr_combined (rebinned) = {np.nanmedian(M('corr_combined')):.3f}, "
          f"resid_rms_combined = {np.nanmedian(M('resid_rms_combined')):.4f} um")

    # ---- time-series PDF (new file) ----
    W = max(12.0, 0.30 * n + 3.0); xlim = (-0.6, n - 0.4)
    bnd = np.where(day[1:] != day[:-1])[0] + 0.5
    day_start = [0] + list(np.where(day[1:] != day[:-1])[0] + 1)

    def decorate(ax, first=False):
        for x in bnd:
            ax.axvline(x, color="0.5", lw=0.6, ls=":")
        for i in np.where(build_used)[0]:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:blue", alpha=0.10)
        ax.set_xlim(*xlim); ax.grid(alpha=0.3)
        if first:
            for i in day_start:
                ax.text(i, 0.985, str(int(day[i])), transform=ax.get_xaxis_transform(),
                        rotation=90, fontsize=6, va="top", ha="left", color="0.35")

    outpdf = os.path.join(cdir, f"coadd_timeseries_rebin{f}.pdf")
    with PdfPages(outpdf) as pdf:
        # 1. metrics vs ordinal
        fig, ax = plt.subplots(2, 1, figsize=(W, 6.8), sharex=True)
        for z in Z_TERMS:
            ax[0].plot(ordn, M(f"resid_rms_z{z}"), "o-", ms=3, label=f"Z{z}")
        ax[0].plot(ordn, M("resid_rms_combined"), "k-", lw=2, label="combined (full FP)")
        ax[0].plot(ordn, M("cwfs_resid_rms_combined"), color="tab:red", ls="--", lw=2,
                   label="combined (CWFS annulus)")
        ax[0].set_ylabel("resid RMS (um)"); ax[0].legend(fontsize=7, ncol=6)
        for z in Z_TERMS:
            ax[1].plot(ordn, M(f"corr_z{z}"), "o-", ms=3, label=f"Z{z}")
        ax[1].plot(ordn, M("corr_combined"), "k-", lw=2, label="mean (full FP)")
        ax[1].plot(ordn, M("cwfs_corr_combined"), color="tab:red", ls="--", lw=2,
                   label="mean (CWFS annulus)")
        ax[1].set_ylabel("spatial corr r"); ax[1].legend(fontsize=7, ncol=6)
        ax[1].set_xlabel("coadd ordinal (day_obs / seq order)")
        decorate(ax[0], first=True); decorate(ax[1], first=True)
        fig.suptitle(f"Coadd-vs-MIW comparison metrics vs ordinal  (rebin {f}x{f}; "
                     "blue bands = 5rot MIW build blocks)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2. telemetry <-> metric Spearman bars
        rms, cor = M("resid_rms_combined"), M("corr_combined")
        rows = [(tv, _spear(tele[tv], rms), _spear(tele[tv], cor))
                for tv in CORR_VARS if tv in tele]
        rows = [r for r in rows if np.isfinite(r[1]) or np.isfinite(r[2])]
        rows.sort(key=lambda r: -(abs(r[1]) if np.isfinite(r[1]) else 0.0))
        labels = [r[0] for r in rows]; xr = np.arange(len(labels))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(10, 0.45 * len(labels) + 3), 8.5))
        for a, vals, ylab, ttl in [
                (a1, [r[1] for r in rows], "rho vs resid_rms_combined", "telemetry vs deviation (residual RMS)"),
                (a2, [r[2] for r in rows], "rho vs corr_combined", "telemetry vs spatial agreement (mean r)")]:
            vals = np.array(vals, float)
            a.bar(xr, np.nan_to_num(vals), color=["tab:red" if v > 0 else "tab:blue" for v in vals])
            a.axhline(0, color="k", lw=0.5); a.set_ylim(-1, 1); a.grid(alpha=0.3)
            a.set_ylabel(ylab, fontsize=9); a.set_title(ttl, fontsize=10)
            a.set_xticks(xr); a.set_xticklabels(labels, rotation=90, fontsize=7)
        fig.suptitle(f"Telemetry <-> coadd-vs-MIW metric Spearman corr  (rebin {f}x{f}; "
                     f"n={n}; red rho>0, blue rho<0)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3. scatter: leading telemetry vs spatial r
        fig, axes = plt.subplots(1, len(SCATTER_VARS), figsize=(4.6 * len(SCATTER_VARS), 4.6),
                                 squeeze=False)
        for a, tv in zip(axes[0], SCATTER_VARS):
            x = tele.get(tv, np.full(n, np.nan)); y = cor
            ok = np.isfinite(x) & np.isfinite(y)
            a.scatter(x[ok & ~build_used], y[ok & ~build_used], s=20, c="0.55", alpha=0.6, label="other")
            a.scatter(x[ok & build_used], y[ok & build_used], s=26, c="tab:green", alpha=0.85, label="MIW build")
            rho = _spear(x, y)
            if np.isfinite(rho) and int(ok.sum()) >= 5:
                sl, ic, *_ = theilslopes(y[ok], x[ok])
                xx = np.array([np.nanmin(x[ok]), np.nanmax(x[ok])])
                a.plot(xx, ic + sl * xx, "k--", lw=1)
            a.set_xlabel(tv, fontsize=9); a.grid(alpha=0.3)
            a.set_title(f"{tv}  Spearman rho={rho:+.2f}", fontsize=9)
        axes[0][0].set_ylabel("spatial corr (combined Z5-Z8)", fontsize=9)
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"telemetry vs spatial r (rebin {f}x{f}; one point per coadd)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4. block telemetry vs ordinal (2 line panels/page + band strip)
        def g(k):  # az etc. are in the npz even if not in CORR_VARS/tele
            return np.asarray(d[k], float) if k in d.files else np.full(n, np.nan)
        tlist = [("elevation (deg)", g("alt")), ("azimuth (deg)", g("az")),
                 ("camera rotator (deg)", g("rot")), ("donut blur FWHM (\")", g("fwhm")),
                 ("M1M3 z_gradient", g("z_gradient"))]
        for p0 in range(0, len(tlist), 2):
            chunk = tlist[p0:p0 + 2]
            fig, axs = plt.subplots(len(chunk), 1, figsize=(W, 4.2 * len(chunk) + 0.4),
                                    sharex=True, squeeze=False)
            for a, (lab, vals) in zip(axs[:, 0], chunk):
                a.plot(ordn, vals, "o-", ms=3, color="C0"); a.set_ylabel(lab, fontsize=9)
                decorate(a, first=(a is axs[0, 0]))
            axs[-1, 0].set_xlabel("coadd ordinal (day_obs / seq order)")
            fig.suptitle("Block telemetry vs coadd ordinal", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # band strip
        fig, a = plt.subplots(figsize=(W, 3.2))
        ub = sorted(set(band)); code = {b: i for i, b in enumerate(ub)}
        a.scatter(ordn, [code[b] for b in band], c=[code[b] for b in band],
                  cmap="tab10", s=20)
        a.set_yticks(range(len(ub))); a.set_yticklabels(ub); a.set_ylabel("filter band")
        a.set_xlabel("coadd ordinal (day_obs / seq order)"); decorate(a, first=True)
        fig.suptitle("Filter band vs coadd ordinal", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 5. u-mode heatmap
        nkeep = Um.shape[1]
        fig, a = plt.subplots(figsize=(W, max(6.0, 0.20 * nkeep + 2.5)))
        vlim = float(np.nanpercentile(np.abs(Um), 98)) or 1.0
        im = a.imshow(Um.T, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                      origin="lower", extent=[-0.5, n - 0.5, 0.5, nkeep + 0.5])
        for x in bnd:
            a.axvline(x, color="0.3", lw=0.6, ls=":")
        for i in np.where(build_used)[0]:
            a.axvline(i, color="tab:blue", lw=0.5, alpha=0.3)
        a.set_xlabel("coadd ordinal (day_obs / seq order)"); a.set_ylabel("u-mode index")
        a.set_yticks(list(range(1, nkeep + 1, 2)))
        a.set_title(f"u-mode amplitude (um) vs coadd ordinal  (rebin {f}x{f}; "
                    "blue lines = 5rot MIW build blocks)")
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.02, label="u-mode amplitude (um)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"wrote {outpdf}  (5 page groups, rebin {f}x{f})")


if __name__ == "__main__":
    main()
