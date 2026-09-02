#!/usr/bin/env python
"""Recompute coadd-vs-MIW metrics on one or more rebinned grids and render the
SINGLE combined time-series PDF -- WITHOUT overwriting any existing output.

Why rebin: the native fine-grid Pearson r / residual RMS are noise-limited (few
donuts per cell + large single-donut wavefront scatter), so they understate the
real structural agreement.  Rebinning the saved block grids averages that noise
down and reveals the true coadd-vs-MIW agreement (r 0.54 -> 0.74 at 3x3 on a
test block).  `--rebin 1 3` renders the metric time series at EACH factor on its
OWN page (native and rebinned are not overplotted -- one metric set per page, so
the noise floor is documented rather than hidden); `--analysis-rebin` (default =
the coarsest) picks which metric set feeds every downstream correlation/ML page.

This is the single renderer for the coadd time series: it supersedes
run_coadd_blocks_miw.render_timeseries (same page sequence, native grid, no ML),
so that script only needs to produce the DATA (block_grids.npz, coadd_metrics)
and the per-block map PDF.

Reads ONLY block_grids.npz (grids + miw + telemetry + u-modes already saved by
run_coadd_blocks_miw.py), so it is self-contained -- numpy/pandas/matplotlib/
scipy, no LSST stack -- and runs on the laptop from the synced output.

Pages:
  1..k  comparison metrics (resid RMS + spatial r) vs ordinal, ONE PAGE PER
        --rebin factor
  T1    telemetry <-> metric Spearman bars (incl. the u-mode scalar distances)
  T2    scatter of the leading telemetry vs spatial r (Theil-Sen line)
  T3    block telemetry (alt/az/rot/fwhm/z_gradient/band) vs ordinal
  U1    u-mode amplitude heatmap vs ordinal (raw, um of wavefront)
  U2    du = u - <u>_build heatmap: displacement from the state the MIW was
        calibrated at (the quantity the residual actually responds to)
  U3    per-mode displacement scales: build vs non-build |du| against the
        reference's SEM, and du in units of the build envelope sigma_build
  U4    per-mode du <-> metric Spearman bars with a permutation null band
        (34 modes x 2 metrics -> multiple comparisons need the yardstick)
  U5    scatter of the leading du drivers + du_norm/du_maha vs spatial r
  M     ML prediction of spatial r: ENVIRONMENTAL telemetry (geometry+thermal,
        no fwhm/band/donut-counts) vs u-mode-only vs both, grouped
        leave-night-out CV against an S/N-only baseline; excludes build blocks
Light blue shading / lines mark the blocks used to BUILD the 5rot MIW.  This is
the ACTUAL build selection (from mi_config.yaml, verified against the build
fits.parquet): i-band, program BLOCK-T614_triplets, elevation 65-75 deg, the 5
in-family rotator windows, day_obs <= 20260513 -- 16 blocks on 4 nights
(2026-03-15..03-24) at elevation ~70.  `program` is not in block_grids.npz, so
it is read from the sibling blocks_summary.parquet.

Writes (new files; originals untouched):
  <coadd_dir>/coadd_metrics_rebin<f>.parquet   (one per --rebin factor)
  <coadd_dir>/coadd_timeseries.pdf             (--out-pdf to override)
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
# telemetry shown as scatter (one panel each) vs the spatial r on page 3; any
# entry not present for this run (e.g. the camera gradient when visits carries no
# cam_* columns) is silently skipped.
SCATTER_VARS = ["z_gradient", "y_gradient", "dome_delta_t", "cam_AverageTemp_dAmb"]

# ML predictors: ENVIRONMENTAL factors only -- telescope geometry + thermal
# telemetry.  fwhm (donut blur), band, and the donut counts are deliberately
# EXCLUDED from the predictive model: they describe/bias the donuts themselves,
# so a correlation with r through them would be a measurement artifact, not an
# environmental driver.  n_donuts/n_visits are kept ONLY as an S/N contrast baseline.
GEOM_VARS = ["alt", "rot", "az_sin", "az_cos"]
ENV_FEATS = GEOM_VARS + THERMAL_VARS
SN_FEATS = ["n_donuts", "n_visits"]

# ---- u-mode (Tier-1) features ----
# u-modes are the amplitudes of the OFC-controllable modes REMOVED from each
# coadd to leave its intrinsic remnant (um of wavefront, block median over the
# per-visit raw DZ fits).  The MIW reference is itself a remnant built with the
# same 50/34 removal, so it carries the build state's own residual: for a
# fractional gain error eps_i in mode i's removal, with M_i that mode's Z5-Z8
# field pattern,
#     coadd_b - MIW = sum_i eps_i * (u_b,i - <u_i>_build) * M_i
# i.e. the residual responds to the DISPLACEMENT from the state the MIW was
# calibrated at, not to the raw amplitude.  Hence du_i = u_i - <u_i>_build.
#
# Which du_i can CAUSE residual, though, is set by field-order leakage, not by
# amplitude.  The build fits only k=1..6 while the ts_ofc DZ sensitivity carries 31
# field orders, so a mode's k>6 wavefront is never fitted and never subtracted -- it
# lands in the MIW.  Since the k<=6 and k>6 parts share one DOF amplitude, a_m is a
# linear proxy for that un-removed content, weighted by
#   rho_m = |k>6| / |k<=6|   (calibration/miw/umode_k_leakage_50_34.npy)
# which runs 0.000-0.002 for modes 1-7 but 0.13-0.47 for modes 28-34.  For the
# leading modes the software removal is essentially exact, so du_1..10 CANNOT be
# causal.  du_leak = ||diag(rho) du|| is therefore the physically motivated scalar.
# (NOT a sensitivity-matrix "gain error": the removal is pure software -- the build
# subtracts exactly the R[P w] it just fitted -- so dS does not enter.)
N_UMODE_LEAD = 10                 # leading modes carried as individual du_i features
UMODE_SCALARS = ["du_norm", "du_norm_lead", "du_norm_tail", "du_maha", "du_leak"]
LEAKAGE_NPY = "calibration/miw/umode_k_leakage_50_34.npy"


def umode_reference(Um, build_used, n_visits, sidecar=None):
    """``<u_i>_build`` -- the u-mode state the MIW was calibrated at -- plus its
    per-mode spread and standard error.

    PREFERRED source is `sidecar`: a length-n_keep vector written RSP-side by
    projecting the MIW build's OWN per-visit DZ fits
    (pathA_.._5rot/fits.parquet) onto the SAME U_eff, i.e. exactly the visits and
    basis the build used, with the build's own central statistic (median -- the
    MIW map cell is a per-donut median, so strictly the effective centre is a
    donut-weighted median that can drift slightly cell to cell; that variation is
    second order and ignored here).

    FALLBACK (laptop-only, approximate) is the n_visits-weighted mean over the
    `build_used` blocks.  Those block u-modes are themselves per-block medians, so
    this is a median-of-medians -- close but not identical; fine for a first look,
    not for quoting a gain error.

    Returns (u_ref, sigma_build, sem, source_str)."""
    B = np.asarray(Um, float)[build_used]
    w = np.asarray(n_visits, float)[build_used]
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    nb = int(B.shape[0])
    if nb == 0:
        nk = Um.shape[1]
        return np.zeros(nk), np.full(nk, np.nan), np.full(nk, np.nan), "NO BUILD BLOCKS"
    wsum = w.sum()
    mean_w = np.nansum(B * w[:, None], axis=0) / wsum
    var_w = np.nansum(w[:, None] * (B - mean_w) ** 2, axis=0) / wsum
    sigma = np.sqrt(var_w)
    sem = sigma / np.sqrt(max(nb, 1))
    if sidecar is not None:
        ref = np.asarray(sidecar, float)
        if ref.shape != mean_w.shape:
            raise ValueError(f"u-mode reference sidecar has shape {ref.shape}, "
                             f"expected {mean_w.shape}")
        return ref, sigma, sem, "build fits.parquet projection (exact)"
    return mean_w, sigma, sem, f"n_visits-weighted mean of {nb} build blocks (approx)"


def load_leakage(path=LEAKAGE_NPY, n_keep=None):
    """Per-mode field-order leakage rho_m = |k>6|/|k<=6|, or None if unavailable.

    Regenerate with scratch kleak.py: map each kept mode's DOF direction N.v_m
    through the FULL 31-field-order ts_ofc DZ sensitivity and compare the fitted
    (k<=6) and un-fitted (k>6) wavefront norms."""
    for p in (path, os.path.join(os.path.dirname(__file__), "..", path)):
        if p and os.path.exists(p):
            rho = np.load(p).astype(float).ravel()
            if n_keep is not None and rho.size != n_keep:
                print(f"  (leakage vector has {rho.size} modes, need {n_keep}; ignored)")
                return None
            return rho
    return None


def umode_features(Um, u_ref, sigma_build, n_lead=N_UMODE_LEAD, floor_pct=25.0,
                   leakage=None):
    """du = u - <u>_build, plus scalar distances from the MIW build state.

    du_maha uses a DIAGONAL metric: with only ~16 build blocks and 34 modes the
    full build covariance is rank-deficient, so each mode is normalized by its own
    build scatter, floored at the `floor_pct` percentile so near-degenerate tail
    modes (sigma -> 0) cannot dominate the distance."""
    dU = np.asarray(Um, float) - np.asarray(u_ref, float)[None, :]
    nk = dU.shape[1]
    nl = int(min(n_lead, nk))
    sg = np.asarray(sigma_build, float)
    good = np.isfinite(sg) & (sg > 0)
    flo = float(np.percentile(sg[good], floor_pct)) if good.any() else 1.0
    sg = np.maximum(np.nan_to_num(sg, nan=flo), flo if flo > 0 else 1.0)
    feat = {f"du_{i + 1}": dU[:, i] for i in range(nl)}
    feat["du_norm"] = np.sqrt(np.nansum(dU ** 2, axis=1))
    feat["du_norm_lead"] = np.sqrt(np.nansum(dU[:, :nl] ** 2, axis=1))
    feat["du_norm_tail"] = (np.sqrt(np.nansum(dU[:, nl:] ** 2, axis=1))
                            if nk > nl else np.zeros(dU.shape[0]))
    feat["du_maha"] = np.sqrt(np.nanmean((dU / sg[None, :]) ** 2, axis=1))
    # leakage-weighted displacement: only the un-removed k>6 part of a mode can
    # cause MIW residual, so weight each du_i by that mode's leakage ratio
    feat["du_leak"] = (np.sqrt(np.nansum((dU * np.asarray(leakage, float)[None, :]) ** 2,
                                         axis=1))
                       if leakage is not None else np.full(dU.shape[0], np.nan))
    return dU, feat


def _spear_null(n_eff, n_perm=2000, q=95.0, seed=0):
    """|rho| at the q-th percentile of the permutation null for n_eff samples.

    Spearman depends only on ranks, so the null distribution is a function of
    n_eff alone -- one band serves all 34 per-mode bars, which is the yardstick
    the multiple comparisons need."""
    if n_eff < 8:
        return np.nan
    rng = np.random.default_rng(seed)
    y = np.arange(n_eff, dtype=float)
    return float(np.percentile(
        [abs(_spear(rng.permutation(y), y)) for _ in range(n_perm)], q))


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


def _block_camera_means(cdir, visits_path, blocks, min_cov=10,
                        ref="cam_AmbAirtemp", diff=True, keep_only=("AverageTemp",)):
    """Per-block camera-body temperatures aggregated from the combined
    visits.parquet (option-1 path: no coadd rebuild).  A block's value is the
    mean over its visits (blocks_summary day_obs, seq_min..seq_max).

    With ``diff`` (default), returns the GRADIENT of each camera line relative to
    the camera ambient-air sensor: ``cam_<field>_dAmb = mean(cam_<field>) -
    mean(<ref>)`` per block (an optical effect is more plausibly driven by a
    temperature difference than an absolute temperature).  The reference column
    (default cam_AmbAirtemp) and the raw lines are then dropped.  With
    ``diff=False`` the raw per-block means are returned instead.

    ``keep_only`` restricts to these field basenames (default just AverageTemp --
    the ~25 utiltrunk sensors are highly collinear, so one aggregate is the clean
    representative and the full set only dilutes the ML; None keeps them all).

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
    # gradient relative to the camera ambient-air sensor
    if diff:
        if ref not in out:
            print(f"  WARNING: camera ref {ref!r} not in visits -> using raw camera "
                  f"temps (add --camera-raw to silence)")
        else:
            rv = out[ref]
            out = {f"{c}_dAmb": out[c] - rv for c in camcols if c != ref}
    if keep_only is not None:
        kset = set(keep_only)
        def _basename(col):
            c = col[4:] if col.startswith("cam_") else col   # strip cam_
            return c[:-5] if c.endswith("_dAmb") else c       # strip _dAmb
        out = {c: v for c, v in out.items() if _basename(c) in kset}
    keep = [c for c in out if np.isfinite(out[c]).sum() >= min_cov]
    return {c: out[c] for c in keep}, keep


# ---------- ML: predict spatial r from telemetry (excludes build blocks) ----------
def render_ml(pdf, Fdf, r, groups, build_used, f, env_cols, sn_cols, umode_cols=()):
    """Predict the per-block combined spatial r from ENVIRONMENTAL telemetry
    (env_cols: telescope geometry + thermal), on the NON-build blocks only (the
    16 MIW-build blocks are the self-comparison / a curated self-consistent set,
    so they are excluded from train AND test).

    fwhm/band/donut-counts are NOT in env_cols: they describe or bias the donuts,
    so predicting r through them would be a measurement artifact, not an
    environmental driver.  sn_cols (n_donuts/n_visits) is kept only as an S/N
    contrast baseline.

    umode_cols (the du_i / du_norm / du_maha displacements from the MIW build
    state) are scored as their OWN feature set and jointly with the environmental
    one, so the increment reads directly: if u-modes add little over the thermal
    telemetry they are a proxy for the thermal state; if they add a lot, the mode
    REMOVAL itself is implicated rather than the environment.

    Target is Fisher-z(r)=arctanh(r) (linearises the bounded metric).  Each
    feature set x two models (RidgeCV, HistGradientBoosting) is scored with
    GroupKFold by day_obs (leave-night-out; blocks from one night share
    conditions).  The increment environmental - S/N is the test of whether the
    observing environment predicts agreement beyond S/N."""
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

    used = [c for c in (list(env_cols) + list(sn_cols) + list(umode_cols))
            if c in Fdf.columns]
    y = np.asarray(r, float)
    keep = (~build_used) & np.isfinite(y)
    F = Fdf.loc[keep, used].copy(); yv = y[keep]; grp = np.asarray(groups)[keep]
    F = F.loc[:, F.notna().any(axis=0)]                 # drop all-NaN columns
    rowok = F.notna().all(axis=1).values                # then rows with any NaN
    F = F[rowok]; yv = yv[rowok]; grp = grp[rowok]
    F = F.loc[:, F.nunique() > 1]                        # drop constant columns
    env = [c for c in env_cols if c in F.columns]        # environmental predictors
    sn = [c for c in sn_cols if c in F.columns]          # S/N contrast baseline
    um = [c for c in umode_cols if c in F.columns]       # u-mode displacements
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
    sets = ([("environmental", env)]
            + ([("u-mode only", um)] if um else [])
            + ([("env+u-mode", env + um)] if um else [])
            + ([("S/N-only", sn)] if sn else []))
    for fname, cols in sets:
        for mname, mk in [("Ridge", ridge), ("GBT", gbt)]:
            res[(fname, mname)] = cv(cols, mk)
    print("  ML grouped-CV (leave-night-out) R^2 on Fisher-z(r), "
          f"non-build n={n_used}, nights={n_nights} "
          f"(environmental={len(env)}, u-mode={len(um)} predictors):")
    for kk, (r2, rmse, _) in res.items():
        print(f"    {kk[0]:13s} {kk[1]:5s}: R2={r2:+.3f}  RMSE(r)={rmse:.3f}")

    # ---- page A: CV R^2 bars (environmental vs S/N, Ridge vs GBT) ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    order_sets = ["environmental", "u-mode only", "env+u-mode", "S/N-only"]
    combos = [(s, m) for s in order_sets for m in ("GBT", "Ridge") if (s, m) in res]
    labs = [f"{c[0]}\n{c[1]}" for c in combos]
    colmap = {"environmental": ("tab:green", "tab:olive"),
              "u-mode only": ("tab:purple", "plum"),
              "env+u-mode": ("tab:blue", "lightsteelblue"),
              "S/N-only": ("0.5", "0.7")}
    cols_bar = [colmap[c[0]][0 if c[1] == "GBT" else 1] for c in combos]
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
    if ("env+u-mode", "GBT") in res:
        sub += (f"; u-modes add {res[('env+u-mode','GBT')][0] - env_r2:+.2f} "
                f"on top of environmental (u-mode alone "
                f"{res[('u-mode only','GBT')][0]:+.2f})")
    fig.suptitle(f"ML: predict coadd-vs-MIW spatial r -- ENVIRONMENTAL telemetry "
                 f"(geometry+thermal; no fwhm/band) vs u-mode displacement from the "
                 f"MIW build state\nnon-build blocks; n={n_used}, "
                 f"{n_nights} nights.  {sub}", fontsize=10)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- page B: permutation importance of the RICHEST model (GBT) ----
    # (env+u-mode when u-modes are available, so environmental and u-mode features
    # compete for the same variance and a thermal proxy shows up as such)
    best_set = "env+u-mode" if ("env+u-mode", "GBT") in res else "environmental"
    feat = dict(sets)[best_set]
    Xenv = F[feat].values.astype(float)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    (tr, te), = gss.split(Xenv, yz, grp)
    mdl = gbt().fit(Xenv[tr], yz[tr])
    pi = permutation_importance(mdl, Xenv[te], yz[te], n_repeats=30, random_state=0)
    order = np.argsort(pi.importances_mean)
    umset = set(um)
    fig, a = plt.subplots(figsize=(9, max(4.5, 0.32 * len(feat) + 1.5)))
    a.barh(np.arange(len(feat)), pi.importances_mean[order],
           xerr=pi.importances_std[order],
           color=["tab:purple" if feat[i] in umset else "tab:blue" for i in order])
    a.set_yticks(np.arange(len(feat)))
    a.set_yticklabels([feat[i] for i in order], fontsize=8)
    a.axvline(0, color="k", lw=0.5); a.grid(alpha=0.3, axis="x")
    a.set_xlabel("permutation importance (drop in R$^2$ on held-out nights)")
    a.set_title(f"{best_set} GBT feature importance for spatial r  "
                f"(train {len(tr)} / test {len(te)} blocks, grouped by night; "
                f"purple = u-mode)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- page C: OOF predicted-vs-actual + partial dependence of top drivers ----
    Xall = Xenv
    _, _, predz = res[(best_set, "GBT")]
    pred_r = np.tanh(predz)
    top = [feat[i] for i in order[::-1][:4]]
    fig = plt.figure(figsize=(12, 7.6))
    axp = fig.add_subplot(2, 3, 1)
    axp.scatter(yv, pred_r, s=22, c="tab:green", alpha=0.7)
    lim = [min(yv.min(), pred_r.min()), max(yv.max(), pred_r.max())]
    axp.plot(lim, lim, "k--", lw=1); axp.set_xlabel("actual r"); axp.set_ylabel("CV-predicted r")
    axp.set_title(f"GBT out-of-fold, {best_set}  (R$^2$={res[(best_set,'GBT')][0]:+.2f})",
                  fontsize=9)
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
    ap.add_argument("--rebin", type=int, nargs="+", default=[1, 3],
                    help="rebin factor(s); the metric time series is rendered on a "
                    "SEPARATE PAGE for each (default: 1 3 = native + 3x3)")
    ap.add_argument("--analysis-rebin", type=int, default=None,
                    help="which rebin factor's metrics feed the correlation/u-mode/ML "
                    "pages (default: the coarsest = least noise-limited)")
    ap.add_argument("--out-pdf", default=None,
                    help="output PDF path (default: <coadd_dir>/coadd_timeseries.pdf)")
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
    ap.add_argument("--camera-ref", default="cam_AmbAirtemp",
                    help="camera ambient reference for the temperature-gradient "
                    "features cam_<field>_dAmb (default: cam_AmbAirtemp)")
    ap.add_argument("--camera-raw", action="store_true",
                    help="use raw camera temps instead of the _dAmb gradients")
    ap.add_argument("--camera-all", action="store_true",
                    help="use all ~23 camera sensors (default: just AverageTemp -- "
                    "the sensors are collinear, so one aggregate is the clean choice)")
    ap.add_argument("--umode-ref", default=None,
                    help="EXACT <u_i>_build reference: .npy/.parquet with the "
                    "n_keep-vector obtained RSP-side by projecting the MIW build's own "
                    "per-visit DZ fits (pathA_.._5rot/fits.parquet) onto the same "
                    "U_eff.  Default: the n_visits-weighted mean over the build blocks "
                    "in the npz (median-of-medians, approximate)")
    ap.add_argument("--umode-lead", type=int, default=N_UMODE_LEAD,
                    help="number of leading u-modes carried as individual du_i features")
    ap.add_argument("--leakage", default=LEAKAGE_NPY,
                    help="per-mode k>6 field-order leakage ratios (.npy) used to build "
                    "du_leak; only modes whose wavefront extends beyond the fitted "
                    "k<=6 basis can cause MIW residual")
    ap.add_argument("--no-umode", action="store_true",
                    help="skip the u-mode displacement pages and features")
    args = ap.parse_args()

    cdir = os.path.join(args.output_root, args.param_set, args.out_name)
    npz = args.npz or os.path.join(cdir, "block_grids.npz")
    d = np.load(npz, allow_pickle=True)
    grids, miw = d["grids"], d["miw"]
    terms = [int(t) for t in d["terms"]]
    assert terms == Z_TERMS, f"unexpected term order {terms}"
    n = grids.shape[0]
    rebins = sorted({int(x) for x in args.rebin if int(x) >= 1})
    f = int(args.analysis_rebin or max(rebins))     # metrics feeding every later page
    if f not in rebins:
        rebins = sorted(set(rebins) | {f})
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
    cam_means, cam_vars = _block_camera_means(
        cdir, vpath, np.asarray(d["block"]),
        ref=args.camera_ref, diff=not args.camera_raw,
        keep_only=None if args.camera_all else ("AverageTemp",))
    for c in cam_vars:
        tele[c] = cam_means[c]
    if cam_vars:
        mode = "raw" if args.camera_raw else f"gradient vs {args.camera_ref}"
        print(f"  merged {len(cam_vars)} camera-body temp col(s) per block ({mode}) "
              f"from {vpath}: {cam_vars[:5]}{'...' if len(cam_vars) > 5 else ''}")
    else:
        print(f"  (no camera-body cam_* cols in {vpath}; run "
              f"run_backfill_camera_telemetry.py --merge on the RSP to add them)")

    # ---- metrics at every requested rebin factor (each gets its own PDF page) ----
    cx = 0.5 * (d["xbins"][:-1] + d["xbins"][1:])
    cy = 0.5 * (d["ybins"][:-1] + d["ybins"][1:])

    def metrics_at(fac):
        cxc, cyc = coarsen_centers(cx, fac), coarsen_centers(cy, fac)
        rr = np.hypot(cyc[:, None], cxc[None, :])          # coarse CWFS annulus mask
        cwfs = (rr >= args.cwfs_inner) & (rr <= args.cwfs_outer)
        return [block_metrics(
            {z: coarsen(grids[i, k], fac) for k, z in enumerate(terms)},
            {z: coarsen(miw[i, k], fac) for k, z in enumerate(terms)},
            cwfs) for i in range(n)]

    METS = {fac: metrics_at(fac) for fac in rebins}

    def Mf(fac, key):
        return np.array([mm[key] for mm in METS[fac]], float)

    def M(key):                                            # the analysis rebin
        return Mf(f, key)

    # ---- one metrics parquet per rebin factor (new files) ----
    base = {k: np.asarray(d[k]) for k in d.files
            if np.asarray(d[k]).ndim == 1 and np.asarray(d[k]).shape[0] == n and k != "terms"}
    for fac in rebins:
        dfm = pd.DataFrame(base)
        dfm["build_used"] = build_used
        for key in METS[fac][0]:
            dfm[key] = Mf(fac, key)
        outpq = os.path.join(cdir, f"coadd_metrics_rebin{fac}.parquet")
        dfm.to_parquet(outpq, index=False)
        print(f"wrote {outpq}  ({n} coadds, rebin {fac}x{fac}"
              f"{' [analysis]' if fac == f else ''}; "
              f"build_used={int(build_used.sum())})")
        print(f"  median corr_combined = {np.nanmedian(Mf(fac, 'corr_combined')):.3f}, "
              f"resid_rms_combined = {np.nanmedian(Mf(fac, 'resid_rms_combined')):.4f} um")

    # ---- u-mode displacement from the MIW build state (Tier 1) ----
    nvis = np.asarray(d["n_visits"], float) if "n_visits" in d.files else np.ones(n)
    dU = ufeat = None
    umode_cols = []
    if not args.no_umode:
        sidecar = None
        if args.umode_ref:
            sidecar = (np.load(args.umode_ref) if args.umode_ref.endswith(".npy")
                       else pd.read_parquet(args.umode_ref).to_numpy(float).ravel())
        u_ref, sig_b, sem_b, uref_src = umode_reference(Um, build_used, nvis, sidecar)
        leak = load_leakage(args.leakage, n_keep=Um.shape[1])
        dU, ufeat = umode_features(Um, u_ref, sig_b, n_lead=args.umode_lead,
                                   leakage=leak)
        if leak is None:
            print("  (no u-mode leakage vector -> du_leak unavailable)")
        else:
            print(f"  u-mode k>6 leakage: median {np.median(leak):.3f}, "
                  f"modes 1-7 <= {leak[:7].max():.3f}, modes 28-34 "
                  f"{leak[27:].min():.2f}-{leak[27:].max():.2f} "
                  "(only leaky modes can be causal)")
        umode_cols = [c for c in ufeat if c.startswith("du_")]
        print(f"  u-mode reference <u_i>_build from {uref_src}: "
              f"{Um.shape[1]} modes, |u_ref| = {np.linalg.norm(u_ref):.3f} um")
        # Scale check: the weighted mean of the build blocks' du is IDENTICALLY zero
        # (that is how u_ref is defined), so there is no test in the centre -- the
        # useful numbers are the per-block spread in units of the build envelope
        # sigma_build, and how far outside that envelope the other blocks sit.
        sgf = np.maximum(sig_b, np.nanpercentile(sig_b[sig_b > 0], 25))
        zb = np.abs(dU[build_used] / sgf) if build_used.any() else np.zeros((0, Um.shape[1]))
        zo = np.abs(dU[~build_used] / sgf)
        print(f"  build-block |du|/sigma_build: max {np.nanmax(zb):.1f}, "
              f"95th pct {np.nanpercentile(zb, 95):.1f} (~1 by construction)")
        print(f"  non-build blocks outside the build envelope (any leading mode "
              f">2 sigma): {100 * np.mean((zo[:, :args.umode_lead] > 2).any(1)):.0f}%")
        print(f"  reference resolution floor SEM/sigma_build = "
              f"{1 / np.sqrt(max(int(build_used.sum()), 1)):.2f}")

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

    outpdf = args.out_pdf or os.path.join(cdir, "coadd_timeseries.pdf")
    with PdfPages(outpdf) as pdf:
        # 1..k. metrics vs ordinal -- ONE PAGE PER REBIN FACTOR (native and rebinned
        #       are deliberately NOT overplotted: 10 curves on one axis is unreadable,
        #       and a page each makes the noise floor a direct visual comparison)
        for fac in rebins:
            fig, ax = plt.subplots(2, 1, figsize=(W, 6.8), sharex=True)
            for z in Z_TERMS:
                ax[0].plot(ordn, Mf(fac, f"resid_rms_z{z}"), "o-", ms=3, label=f"Z{z}")
            ax[0].plot(ordn, Mf(fac, "resid_rms_combined"), "k-", lw=2,
                       label="combined (full FP)")
            ax[0].plot(ordn, Mf(fac, "cwfs_resid_rms_combined"), color="tab:red",
                       ls="--", lw=2, label="combined (CWFS annulus)")
            ax[0].set_ylabel("resid RMS (um)"); ax[0].legend(fontsize=7, ncol=6)
            for z in Z_TERMS:
                ax[1].plot(ordn, Mf(fac, f"corr_z{z}"), "o-", ms=3, label=f"Z{z}")
            ax[1].plot(ordn, Mf(fac, "corr_combined"), "k-", lw=2, label="mean (full FP)")
            ax[1].plot(ordn, Mf(fac, "cwfs_corr_combined"), color="tab:red", ls="--",
                       lw=2, label="mean (CWFS annulus)")
            ax[1].set_ylabel("spatial corr r"); ax[1].legend(fontsize=7, ncol=6)
            ax[1].set_xlabel("coadd ordinal (day_obs / seq order)")
            decorate(ax[0], first=True); decorate(ax[1], first=True)
            grid_lab = ("native grid" if fac == 1 else f"rebin {fac}x{fac}")
            fig.suptitle(f"Coadd-vs-MIW comparison metrics vs ordinal -- {grid_lab}"
                         f"{'  [drives the analysis pages]' if fac == f else ''}   "
                         f"median r={np.nanmedian(Mf(fac, 'corr_combined')):.3f}  "
                         "(blue bands = 5rot MIW build blocks)", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2. telemetry <-> metric Spearman bars
        rms, cor = M("resid_rms_combined"), M("corr_combined")
        # the few u-mode SCALARS (distance from the MIW build state) belong with the
        # telemetry bars; the 34 per-mode du_i get their own page with a null band
        for c in (UMODE_SCALARS if ufeat is not None else []):
            tele[c] = ufeat[c]
        bar_vars = CORR_VARS + cam_vars + (UMODE_SCALARS if ufeat is not None else [])
        rows = [(tv, _spear(tele[tv], rms), _spear(tele[tv], cor))
                for tv in bar_vars if tv in tele]
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

        # 3. scatter: leading telemetry vs spatial r (skip any SCATTER_VARS entry,
        #    e.g. the camera gradient, that is absent for this run)
        scatter_vars = [v for v in SCATTER_VARS if v in tele]
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
        a.set_title("u-mode amplitude (um) vs coadd ordinal -- RAW (amplitude removed "
                    "from each coadd; blue lines = 5rot MIW build blocks)")
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.02, label="u-mode amplitude (um)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if ufeat is not None:
            # U2. du heatmap -- displacement from the state the MIW was calibrated at.
            #     This, not the raw u, is what the residual responds to:
            #     coadd_b - MIW = sum_i eps_i (u_b,i - <u_i>_build) M_i
            fig, a = plt.subplots(figsize=(W, max(6.0, 0.20 * nkeep + 2.5)))
            vlim = float(np.nanpercentile(np.abs(dU), 98)) or 1.0
            im = a.imshow(dU.T, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                          origin="lower", extent=[-0.5, n - 0.5, 0.5, nkeep + 0.5])
            for x in bnd:
                a.axvline(x, color="0.3", lw=0.6, ls=":")
            for i in np.where(build_used)[0]:
                a.axvline(i, color="tab:green", lw=0.6, alpha=0.5)
            a.set_xlabel("coadd ordinal (day_obs / seq order)")
            a.set_ylabel("u-mode index")
            a.set_yticks(list(range(1, nkeep + 1, 2)))
            a.set_title("du = u - <u>_build : u-mode displacement from the MIW build "
                        f"state ({uref_src})\ngreen lines = build blocks (must read ~0)")
            fig.colorbar(im, ax=a, fraction=0.03, pad=0.02, label="du (um)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # U3. per-mode displacement scales.  NOTE the build blocks' weighted mean
            #     du is IDENTICALLY zero -- that is the definition of u_ref, not a
            #     test -- so the informative quantities are (a) how far the non-build
            #     blocks sit outside the build envelope sigma_build, and (b) the
            #     reference's own resolution floor SEM = sigma_build/sqrt(n_build),
            #     below which a du is indistinguishable from the calibration state.
            mi = np.arange(1, nkeep + 1)
            fig, (b1, b2) = plt.subplots(2, 1, figsize=(max(11, 0.30 * nkeep + 5), 8.4))
            rms_b = np.sqrt(np.nanmean(dU[build_used] ** 2, axis=0)) if build_used.any() \
                else np.full(nkeep, np.nan)
            rms_o = np.sqrt(np.nanmean(dU[~build_used] ** 2, axis=0))
            b1.bar(mi - 0.2, rms_o, 0.4, color="tab:orange", label="non-build blocks")
            b1.bar(mi + 0.2, rms_b, 0.4, color="tab:green", label="build blocks")
            b1.plot(mi, sem_b, "k^", ms=5, label="SEM of the reference")
            b1.set_yscale("log"); b1.set_ylabel("RMS |du| (um)")
            b1.set_xticks(mi); b1.set_xticklabels(mi, fontsize=7)
            b1.legend(fontsize=8); b1.grid(alpha=0.3, axis="y")
            b1.set_title("per-mode displacement scale: how far the non-build blocks "
                         "sit from the MIW build state, vs the build scatter and the "
                         "reference's own SEM")
            sgf = np.maximum(sig_b, np.nanpercentile(sig_b[sig_b > 0], 25))
            zo = dU[~build_used] / sgf
            zb = dU[build_used] / sgf if build_used.any() else np.zeros((0, nkeep))
            b2.boxplot([zo[:, i] for i in range(nkeep)], positions=mi - 0.17,
                       widths=0.3, showfliers=True,
                       flierprops=dict(ms=2, mfc="tab:orange", mec="tab:orange"),
                       medianprops=dict(color="tab:orange"))
            if zb.size:
                b2.boxplot([zb[:, i] for i in range(nkeep)], positions=mi + 0.17,
                           widths=0.3, showfliers=False,
                           medianprops=dict(color="tab:green"))
            b2.axhline(0, color="k", lw=0.8)
            for s in (-1, 1):
                b2.axhline(s, color="tab:green", lw=0.8, ls=":")
            floor = 1 / np.sqrt(max(int(build_used.sum()), 1))
            b2.axhspan(-floor, floor, color="0.6", alpha=0.25)
            b2.set_ylabel("du / sigma_build"); b2.set_xlabel("u-mode index")
            b2.set_xticks(mi); b2.set_xticklabels(mi, fontsize=7); b2.grid(alpha=0.3)
            b2.set_title("du in units of the BUILD ENVELOPE (orange = non-build, green "
                         "= build).  Build blocks straddle +/-1 BY CONSTRUCTION (u_ref "
                         "is their weighted mean) -- this is a scale check, not a test. "
                         f"Grey band = reference floor SEM/sigma = {floor:.2f}; "
                         "non-build boxes reaching well outside +/-1 are the blocks "
                         "observed away from the calibration state")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # U4. per-mode du <-> metric Spearman, with a permutation null band.
            #     34 modes x 2 metrics is a real multiple-comparison problem, so the
            #     band (|rho| exceeded 5% of the time under random rank permutation)
            #     is the yardstick for calling any single bar significant.
            ok_n = int(np.isfinite(cor).sum())
            null95 = _spear_null(ok_n)
            rho_rms = np.array([_spear(dU[:, i], rms) for i in range(nkeep)])
            rho_cor = np.array([_spear(dU[:, i], cor) for i in range(nkeep)])
            fig, (c1, c2) = plt.subplots(2, 1, figsize=(max(11, 0.30 * nkeep + 5), 8.4))
            for a_, vals, ylab, ttl in [
                    (c1, rho_rms, "rho vs resid_rms_combined",
                     "du_i vs deviation (residual RMS)"),
                    (c2, rho_cor, "rho vs corr_combined",
                     "du_i vs spatial agreement (mean r)")]:
                sig = np.abs(vals) > (null95 if np.isfinite(null95) else np.inf)
                a_.bar(mi, np.nan_to_num(vals),
                       color=["tab:red" if s else "0.7" for s in sig])
                if np.isfinite(null95):
                    for s in (-null95, null95):
                        a_.axhline(s, color="k", lw=0.9, ls="--")
                a_.axhline(0, color="k", lw=0.5); a_.set_ylim(-1, 1)
                a_.set_ylabel(ylab, fontsize=9); a_.set_title(ttl, fontsize=10)
                a_.set_xticks(mi); a_.set_xticklabels(mi, fontsize=7)
                a_.grid(alpha=0.3, axis="y")
            c2.set_xlabel("u-mode index")
            fig.suptitle("Per-mode u-mode DISPLACEMENT du_i vs the coadd-vs-MIW "
                         f"metrics (rebin {f}x{f}, n={n})\ndashed = permutation null "
                         f"|rho|_95 = {null95:.2f} for n={ok_n}; red bars exceed it "
                         "(coloured only, NOT FDR-corrected across 34 modes)",
                         fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # U5. scatter of the strongest du drivers + the scalar distances vs r
            top_i = [int(i) for i in np.argsort(-np.abs(np.nan_to_num(rho_cor)))[:4]]
            sc_items = ([(f"du_{i + 1}", dU[:, i]) for i in top_i]
                        + [(s, ufeat[s]) for s in UMODE_SCALARS])
            ncol = 4
            nrow = int(np.ceil(len(sc_items) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.3 * nrow),
                                     squeeze=False)
            for a_, (lab, x) in zip(axes.ravel(), sc_items):
                y = cor
                okx = np.isfinite(x) & np.isfinite(y)
                a_.scatter(x[okx & ~build_used], y[okx & ~build_used], s=20,
                           c="0.55", alpha=0.6, label="other")
                a_.scatter(x[okx & build_used], y[okx & build_used], s=26,
                           c="tab:green", alpha=0.85, label="MIW build")
                rho = _spear(x, y)
                if np.isfinite(rho) and int(okx.sum()) >= 5:
                    sl, ic, *_ = theilslopes(y[okx], x[okx])
                    xx = np.array([np.nanmin(x[okx]), np.nanmax(x[okx])])
                    a_.plot(xx, ic + sl * xx, "k--", lw=1)
                a_.axvline(0, color="0.8", lw=0.5)
                a_.set_xlabel(f"{lab} (um)", fontsize=9); a_.grid(alpha=0.3)
                a_.set_title(f"{lab}   rho={rho:+.2f}", fontsize=9)
            for a_ in axes.ravel()[len(sc_items):]:
                a_.axis("off")
            axes[0][0].set_ylabel("spatial corr (combined Z5-Z8)", fontsize=9)
            axes[0][0].legend(fontsize=7)
            fig.suptitle("Leading u-mode displacements and the scalar distances from "
                         "the MIW build state vs spatial r (Theil-Sen line; build "
                         "blocks green, and they sit at du~0 by construction)",
                         fontsize=11)
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
            for uc in umode_cols:                     # du_i + scalar distances
                feat[uc] = ufeat[uc]
            Fdf = pd.DataFrame(feat)   # NOTE: no fwhm, no band -> environmental only
            render_ml(pdf, Fdf, M("corr_combined"), day, build_used, f,
                      ENV_FEATS + cam_vars, SN_FEATS, umode_cols=umode_cols)

    print(f"wrote {outpdf}  (metric pages: rebin {rebins}, analysis rebin {f}; "
          f"u-modes {'on' if ufeat is not None else 'off'}, "
          f"ML {'off' if args.no_ml else 'on'})")


if __name__ == "__main__":
    main()
