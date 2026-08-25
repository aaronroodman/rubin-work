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
  6. ML prediction of spatial r from ENVIRONMENTAL telemetry only -- geometry +
     thermal, no fwhm/band/donut-counts (grouped leave-night-out CV, vs an
     S/N-only baseline; excludes the MIW-build blocks)
Light blue shading / lines mark the blocks used to BUILD the 5rot MIW.  This is
the ACTUAL build selection (from mi_config.yaml, verified against the build
fits.parquet): i-band, program BLOCK-T614_triplets, elevation 65-75 deg, the 5
in-family rotator windows, day_obs <= 20260513 -- 16 blocks on 4 nights
(2026-03-15..03-24) at elevation ~70.  `program` is not in block_grids.npz, so
it is read from the sibling blocks_summary.parquet.

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

# ML predictors: ENVIRONMENTAL factors only -- telescope geometry + thermal
# telemetry.  fwhm (donut blur), band, and the donut counts are deliberately
# EXCLUDED from the predictive model: they describe/bias the donuts themselves,
# so a correlation with r through them would be a measurement artifact, not an
# environmental driver.  n_donuts/n_visits are kept ONLY as an S/N contrast baseline.
GEOM_VARS = ["alt", "rot", "az_sin", "az_cos"]
ENV_FEATS = GEOM_VARS + THERMAL_VARS
SN_FEATS = ["n_donuts", "n_visits"]


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


def _block_programs(cdir, blocks, n):
    """Per-block science_program (BLOCK- prefix stripped) from blocks_summary.parquet.

    block_grids.npz does not carry program, so the build-selection needs this
    sidecar.  Returns "" for any block not found (or if the file is absent)."""
    p = os.path.join(cdir, "blocks_summary.parquet")
    if not os.path.exists(p):
        print(f"  WARNING: {p} missing -> cannot apply the program cut for build_used")
        return np.array([""] * n)
    bs = (pd.read_parquet(p, columns=["block", "program"])
          .drop_duplicates("block").set_index("block"))
    return np.array([str(bs.loc[b, "program"]).replace("BLOCK-", "")
                     if b in bs.index else "" for b in blocks])


def _block_camera_means(cdir, visits_path, blocks, min_cov=10):
    """Per-block mean of the camera-body cam_<field> columns, aggregated from the
    combined visits.parquet (option-1 path: no coadd rebuild).  A block's cam
    value is the mean over its visits (blocks_summary day_obs, seq_min..seq_max).

    Excludes the pre-existing ESS cols cam_air_temp / cam_m1m3_delta_t (already in
    THERMAL_VARS) and cam_n_samp.  Returns ({col: array aligned to blocks}, [cols])
    keeping only columns with >= min_cov finite blocks."""
    bpath = os.path.join(cdir, "blocks_summary.parquet")
    if not (visits_path and os.path.exists(visits_path) and os.path.exists(bpath)):
        return {}, []
    v = pd.read_parquet(visits_path)
    camcols = [c for c in v.columns if c.startswith("cam_")
               and c not in THERMAL_VARS and c != "cam_n_samp"]
    if not camcols:
        return {}, []
    v = v[["day_obs", "seq_num"] + camcols].copy()
    v["day_obs"] = v["day_obs"].astype(int)
    v["seq_num"] = v["seq_num"].astype(int)
    bs = (pd.read_parquet(bpath, columns=["block", "day_obs", "seq_min", "seq_max"])
          .drop_duplicates("block").set_index("block"))
    means = {c: [] for c in camcols}
    for b in blocks:
        if b in bs.index:
            r = bs.loc[b]
            sub = v[(v.day_obs == int(r.day_obs)) & (v.seq_num >= int(r.seq_min))
                    & (v.seq_num <= int(r.seq_max))]
        else:
            sub = v.iloc[0:0]
        for c in camcols:
            x = sub[c].to_numpy(float) if len(sub) else np.array([])
            means[c].append(float(np.nanmean(x)) if x.size and np.isfinite(x).any()
                            else np.nan)
    out = {c: np.array(means[c], float) for c in camcols}
    keep = [c for c in camcols if np.isfinite(out[c]).sum() >= min_cov]
    return {c: out[c] for c in keep}, keep


# ---------- ML: predict spatial r from telemetry (excludes build blocks) ----------
def render_ml(pdf, Fdf, r, groups, build_used, f, env_cols, sn_cols):
    """Predict the per-block combined spatial r from ENVIRONMENTAL telemetry
    (env_cols: telescope geometry + thermal), on the NON-build blocks only (the
    16 MIW-build blocks are the self-comparison / a curated self-consistent set,
    so they are excluded from train AND test).

    fwhm/band/donut-counts are NOT in env_cols: they describe or bias the donuts,
    so predicting r through them would be a measurement artifact, not an
    environmental driver.  sn_cols (n_donuts/n_visits) is kept only as an S/N
    contrast baseline.

    Target is Fisher-z(r)=arctanh(r) (linearises the bounded metric).  Two feature
    sets (environmental vs S/N-only) x two models (RidgeCV,
    HistGradientBoosting) are scored with GroupKFold by day_obs (leave-night-out;
    blocks from one night share conditions).  The increment environmental - S/N is
    the test of whether the observing environment predicts agreement beyond S/N."""
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import GroupKFold, cross_val_predict, GroupShuffleSplit
        from sklearn.inspection import permutation_importance, PartialDependenceDisplay
        from sklearn.metrics import r2_score, mean_squared_error
    except Exception as e:  # sklearn not installed -> a single note page
        fig, a = plt.subplots(figsize=(10, 3)); a.axis("off")
        a.text(0.5, 0.5, f"ML pages skipped: scikit-learn unavailable ({e})",
               ha="center", va="center", fontsize=11)
        pdf.savefig(fig); plt.close(fig); return None

    used = [c for c in (list(env_cols) + list(sn_cols)) if c in Fdf.columns]
    y = np.asarray(r, float)
    keep = (~build_used) & np.isfinite(y)
    F = Fdf.loc[keep, used].copy(); yv = y[keep]; grp = np.asarray(groups)[keep]
    F = F.loc[:, F.notna().any(axis=0)]                 # drop all-NaN columns
    rowok = F.notna().all(axis=1).values                # then rows with any NaN
    F = F[rowok]; yv = yv[rowok]; grp = grp[rowok]
    F = F.loc[:, F.nunique() > 1]                        # drop constant columns
    env = [c for c in env_cols if c in F.columns]        # environmental predictors
    sn = [c for c in sn_cols if c in F.columns]          # S/N contrast baseline
    n_used, n_nights = len(yv), len(np.unique(grp))
    if n_used < 25 or n_nights < 4 or not env:
        fig, a = plt.subplots(figsize=(10, 3)); a.axis("off")
        a.text(0.5, 0.5, f"ML pages skipped: too little usable data "
               f"(n={n_used}, nights={n_nights}, env feats={len(env)})",
               ha="center", va="center", fontsize=11)
        pdf.savefig(fig); plt.close(fig)
        return None

    yz = np.arctanh(np.clip(yv, -0.999, 0.999))          # Fisher-z target
    k = int(min(5, n_nights))
    gkf = GroupKFold(n_splits=k)
    ridge = lambda: make_pipeline(StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-3, 3, 25)))
    gbt = lambda: HistGradientBoostingRegressor(
        max_depth=3, max_iter=400, learning_rate=0.05, l2_regularization=1.0,
        min_samples_leaf=8, random_state=0)

    def cv(cols, mk):
        Xc = F[cols].values.astype(float)
        pred = cross_val_predict(mk(), Xc, yz, cv=gkf, groups=grp)
        return (r2_score(yz, pred),
                float(np.sqrt(mean_squared_error(yv, np.tanh(pred)))), pred)

    res = {}
    sets = [("environmental", env)] + ([("S/N-only", sn)] if sn else [])
    for fname, cols in sets:
        for mname, mk in [("Ridge", ridge), ("GBT", gbt)]:
            res[(fname, mname)] = cv(cols, mk)
    print("  ML grouped-CV (leave-night-out) R^2 on Fisher-z(r), "
          f"non-build n={n_used}, nights={n_nights} (predictors: environmental only):")
    for kk, (r2, rmse, _) in res.items():
        print(f"    {kk[0]:13s} {kk[1]:5s}: R2={r2:+.3f}  RMSE(r)={rmse:.3f}")

    # ---- page A: CV R^2 bars (environmental vs S/N, Ridge vs GBT) ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    combos = [c for c in [("environmental", "GBT"), ("environmental", "Ridge"),
                          ("S/N-only", "GBT"), ("S/N-only", "Ridge")] if c in res]
    labs = [f"{c[0]}\n{c[1]}" for c in combos]
    colmap = {"environmental": "tab:green", "S/N-only": "0.5"}
    cols_bar = [colmap[c[0]] if c[1] == "GBT" else
                ("tab:olive" if c[0] == "environmental" else "0.7") for c in combos]
    xp = np.arange(len(combos))
    a1.bar(xp, [res[c][0] for c in combos], color=cols_bar)
    a1.axhline(0, color="k", lw=0.5); a1.set_xticks(xp); a1.set_xticklabels(labs, fontsize=8)
    a1.set_ylabel("grouped-CV R$^2$  (Fisher-z r)"); a1.grid(alpha=0.3, axis="y")
    a1.set_title("predictability of spatial r")
    a2.bar(xp, [res[c][1] for c in combos], color=cols_bar)
    a2.set_xticks(xp); a2.set_xticklabels(labs, fontsize=8)
    a2.set_ylabel("CV RMSE in r units"); a2.grid(alpha=0.3, axis="y")
    a2.set_title("residual scatter")
    env_r2 = res[("environmental", "GBT")][0]
    sub = f"GBT environmental-model R$^2$={env_r2:+.2f}"
    if ("S/N-only", "GBT") in res:
        sub += f"; adds {env_r2 - res[('S/N-only','GBT')][0]:+.2f} over S/N-only (n_donuts/n_visits)"
    fig.suptitle(f"ML: predict coadd-vs-MIW spatial r from ENVIRONMENTAL telemetry "
                 f"(geometry+thermal; no fwhm/band)\nnon-build blocks; n={n_used}, "
                 f"{n_nights} nights.  {sub}", fontsize=10)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- page B: permutation importance of the environmental model (GBT) ----
    Xenv = F[env].values.astype(float)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    (tr, te), = gss.split(Xenv, yz, grp)
    mdl = gbt().fit(Xenv[tr], yz[tr])
    pi = permutation_importance(mdl, Xenv[te], yz[te], n_repeats=30, random_state=0)
    order = np.argsort(pi.importances_mean)
    fig, a = plt.subplots(figsize=(9, max(4.5, 0.32 * len(env) + 1.5)))
    a.barh(np.arange(len(env)), pi.importances_mean[order],
           xerr=pi.importances_std[order], color="tab:blue")
    a.set_yticks(np.arange(len(env))); a.set_yticklabels([env[i] for i in order], fontsize=8)
    a.axvline(0, color="k", lw=0.5); a.grid(alpha=0.3, axis="x")
    a.set_xlabel("permutation importance (drop in R$^2$ on held-out nights)")
    a.set_title(f"Environmental-model GBT feature importance for spatial r  "
                f"(train {len(tr)} / test {len(te)} blocks, grouped by night)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- page C: OOF predicted-vs-actual + partial dependence of top drivers ----
    feat = env; Xall = Xenv
    _, _, predz = res[("environmental", "GBT")]
    pred_r = np.tanh(predz)
    top = [feat[i] for i in order[::-1][:4]]
    fig = plt.figure(figsize=(12, 7.6))
    axp = fig.add_subplot(2, 3, 1)
    axp.scatter(yv, pred_r, s=22, c="tab:green", alpha=0.7)
    lim = [min(yv.min(), pred_r.min()), max(yv.max(), pred_r.max())]
    axp.plot(lim, lim, "k--", lw=1); axp.set_xlabel("actual r"); axp.set_ylabel("CV-predicted r")
    axp.set_title(f"GBT out-of-fold  (R$^2$={res[('environmental','GBT')][0]:+.2f})", fontsize=9)
    axp.grid(alpha=0.3)
    full_fit = gbt().fit(Xall, yz)
    for j, tv in enumerate(top):
        ax = fig.add_subplot(2, 3, j + 2)
        try:
            PartialDependenceDisplay.from_estimator(
                full_fit, Xall, [feat.index(tv)], feature_names=feat, ax=ax)
            ax.set_title(f"partial dep: {tv}", fontsize=9)
        except Exception:
            ax.axis("off"); ax.text(0.5, 0.5, f"PD failed: {tv}", ha="center")
    fig.suptitle("GBT prediction of spatial r: out-of-fold accuracy + partial "
                 "dependence of the leading drivers (Fisher-z target)", fontsize=11)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-set", default="fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x")
    ap.add_argument("--out-name", default="coadd_50_34")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--npz", default=None, help="override path to block_grids.npz")
    ap.add_argument("--rebin", type=int, default=3)
    ap.add_argument("--cwfs-inner", type=float, default=1.59)
    ap.add_argument("--cwfs-outer", type=float, default=1.725)
    # MIW 5rot build selection (mi_config.yaml defaults + entry) -- see docstring
    ap.add_argument("--build-bands", nargs="+", default=["i"])
    ap.add_argument("--build-programs", nargs="+", default=["T614_triplets"])
    ap.add_argument("--build-alt-min", type=float, default=65.0)
    ap.add_argument("--build-alt-max", type=float, default=75.0)
    ap.add_argument("--build-day-max", type=int, default=20260513)
    ap.add_argument("--no-ml", action="store_true", help="skip the ML prediction pages")
    ap.add_argument("--visits", default=None,
                    help="combined visits.parquet carrying camera-body cam_<field> "
                    "columns (from run_backfill_camera_telemetry --merge); default: "
                    "auto-detect <output_root>/<param_set>/visits.parquet")
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
    alt = np.asarray(d["alt"], float)
    infam = np.asarray(d["in_family"]).astype(bool)  # == the 5 rotator-window membership
    # ACTUAL MIW-build selection: program is not in the npz -> read blocks_summary.parquet.
    prog = _block_programs(cdir, np.asarray(d["block"]), n)
    progset = {p.replace("BLOCK-", "") for p in args.build_programs}
    build_used = (np.isin(band, args.build_bands)
                  & np.array([p in progset for p in prog])
                  & (alt >= args.build_alt_min) & (alt <= args.build_alt_max)
                  & infam & (day <= args.build_day_max))
    ordn = np.arange(n)

    # option-1: camera-body temps aggregated per block from the merged visits.parquet
    # (no coadd rebuild); added to the Spearman bars, scatter, and environmental ML.
    vpath = args.visits or os.path.join(args.output_root, args.param_set, "visits.parquet")
    cam_means, cam_vars = _block_camera_means(cdir, vpath, np.asarray(d["block"]))
    for c in cam_vars:
        tele[c] = cam_means[c]
    if cam_vars:
        print(f"  merged {len(cam_vars)} camera-body temp col(s) per block from "
              f"{vpath}: {cam_vars[:5]}{'...' if len(cam_vars) > 5 else ''}")
    else:
        print(f"  (no camera-body cam_* cols in {vpath}; run "
              f"run_backfill_camera_telemetry.py --merge on the RSP to add them)")

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
                for tv in CORR_VARS + cam_vars if tv in tele]
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

        # 3. scatter: leading telemetry vs spatial r (+ camera-body avg if present)
        scatter_vars = SCATTER_VARS + (["cam_AverageTemp"] if "cam_AverageTemp" in tele else [])
        fig, axes = plt.subplots(1, len(scatter_vars), figsize=(4.6 * len(scatter_vars), 4.6),
                                 squeeze=False)
        for a, tv in zip(axes[0], scatter_vars):
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

        # 6. ML: predict spatial r from ENVIRONMENTAL telemetry only (geometry +
        #    thermal); fwhm/band/donut-counts are excluded from the predictors so
        #    the model can't cheat through donut-quality proxies.  Build blocks
        #    are excluded from train and test.
        if not args.no_ml:
            az = np.asarray(d["az"], float) if "az" in d.files else np.full(n, np.nan)
            ndon = np.asarray(d["n_donuts"], float) if "n_donuts" in d.files else np.full(n, np.nan)
            nvis = np.asarray(d["n_visits"], float) if "n_visits" in d.files else np.full(n, np.nan)
            feat = {"alt": tele.get("alt", alt), "rot": tele.get("rot", np.full(n, np.nan)),
                    "az_sin": np.sin(np.deg2rad(az)), "az_cos": np.cos(np.deg2rad(az)),
                    "n_donuts": ndon, "n_visits": nvis}
            for tv in THERMAL_VARS:
                feat[tv] = tele.get(tv, np.full(n, np.nan))
            for cv in cam_vars:                       # camera-body temps (per block)
                feat[cv] = tele[cv]
            Fdf = pd.DataFrame(feat)   # NOTE: no fwhm, no band -> environmental only
            render_ml(pdf, Fdf, M("corr_combined"), day, build_used, f,
                      ENV_FEATS + cam_vars, SN_FEATS)

    print(f"wrote {outpdf}  ({'6' if not args.no_ml else '5'} page groups, rebin {f}x{f})")


if __name__ == "__main__":
    main()
