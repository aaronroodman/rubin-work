"""Gradient-boosted trees for the intra - extra CWFS blur, scored against the Huber model.

Every model is scored out of fold with whole nights held out (`GroupKFold` on
``day_obs``), since visits within a night are correlated and a random split would let a
model memorize each night. The score is the robust scatter (nMAD) of the held-out
residuals, on the visits that have every thermal quantity, so all models see the same
visits.

Models:

- ``huber3``: the 3-quantity Huber model of `blur_thermal`, refit in each fold;
- ``gbt3``: boosted trees on the same 3 quantities;
- ``gbt_thermal``: boosted trees on every thermal quantity;
- ``gbt_all``: boosted trees on thermal, wind and pointing. Missing values are allowed,
  so the TMA top-ring sonics, which start on 2026-02-16, enter without cutting visits.

Inputs are telemetry only: no quantity measured from the donuts themselves.

Run from this study's directory, after the two fetch scripts::

    python code/blur_gbt.py
"""

import pathlib
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GroupKFold

from rubinwork.common import study_run
from rubinwork.common.utils import nmad, setup_plotting
from rubinwork.forward_selection import huber_fit
from rubinwork.products import cwfs_science

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from blur_thermal import (MIN_DONUTS, MIN_RAFTS, TARGET, TARGET_LABEL,  # noqa: E402
                          THERMAL_FEATURES, VARIANT, WIND_FEATURES, add_thermal, add_wind,
                          per_visit)

N_FOLDS = 5
HUBER3 = ["cam_body_minus_m1m3_air_c", "m1m3_r2_coeff_c", "cam_minus_m1m3_air_c"]
POINTING = {"alt": "TMA elevation (deg)", "az": "TMA azimuth (deg)"}
ALL_FEATURES = {**THERMAL_FEATURES, **WIND_FEATURES, **POINTING}
GBT_PARAMS = dict(loss="absolute_error", learning_rate=0.05, max_iter=600,
                  max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=False, random_state=0)
N_PERM = 3
N_PDP = 8


def raft_noise_floor():
    """Expected nMAD of the per-visit blur difference from raft-to-raft scatter alone.

    The per-visit value is the median over rafts, so its error is about
    1.2533 sigma_raft / sqrt(n_rafts), with sigma_raft the pooled robust scatter of each
    raft about its visit median (corrected for the median's own use of that raft).
    """
    ccd = cwfs_science.load_table("ccd_visits", VARIANT)
    ccd = ccd[ccd["n_quality"] >= MIN_DONUTS]
    p = ccd.pivot_table(index=["visit", "raft"], columns="wfs_offset",
                        values="blur_median").dropna()
    raft = (p[-1] - p[0]).rename("diff").reset_index()
    n = raft.groupby("visit")["diff"].transform("size")
    raft = raft[n >= MIN_RAFTS].assign(n=n[n >= MIN_RAFTS])
    dev = raft["diff"] - raft.groupby("visit")["diff"].transform("median")
    out = {}
    for k in (2, 3, 4):
        sel = raft["n"] == k
        sigma = nmad(dev[sel]) * np.sqrt(k / (k - 1))
        out[k] = dict(n_visits=int(sel.sum() // k), sigma_raft=float(sigma),
                      floor=float(1.2533 * sigma / np.sqrt(k)))
    return out


def gbt():
    return HistGradientBoostingRegressor(**GBT_PARAMS)


def oof_predictions(d, folds):
    """Out-of-fold predictions of every model, on the rows of `d`."""
    y = d[TARGET].to_numpy()
    pred = {m: np.full(len(d), np.nan) for m in ("huber3", "gbt3", "gbt_thermal", "gbt_all")}
    sets = {"gbt3": HUBER3, "gbt_thermal": list(THERMAL_FEATURES),
            "gbt_all": list(ALL_FEATURES)}
    for k, (tr, te) in enumerate(folds):
        fit = huber_fit(y[tr], d[HUBER3].to_numpy()[tr])
        pred["huber3"][te] = fit.predict(np.column_stack(
            [np.ones(len(te)), d[HUBER3].to_numpy()[te]]))
        for m, cols in sets.items():
            pred[m][te] = gbt().fit(d[cols].to_numpy()[tr], y[tr]).predict(
                d[cols].to_numpy()[te])
        print(f"fold {k + 1}/{len(folds)} done", flush=True)
    return pred


def permutation_nmad(d, folds, cols, rng, labels=ALL_FEATURES):
    """Rise in held-out residual nMAD when each input is shuffled, in arcsec."""
    y = d[TARGET].to_numpy()
    X = d[cols].to_numpy()
    rise = np.zeros((len(folds), len(cols)))
    for k, (tr, te) in enumerate(folds):
        model = gbt().fit(X[tr], y[tr])
        te = rng.choice(te, size=min(len(te), 8000), replace=False)
        base = nmad(y[te] - model.predict(X[te]))
        for j in range(len(cols)):
            vals = []
            for _ in range(N_PERM):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                vals.append(nmad(y[te] - model.predict(Xp)))
            rise[k, j] = np.mean(vals) - base
    return pd.DataFrame({"column": cols, "feature": [labels[c] for c in cols],
                         "rise_mean": rise.mean(0), "rise_std": rise.std(0)}
                        ).sort_values("rise_mean", ascending=False).reset_index(drop=True)


def scores(d, pred, folds):
    y = d[TARGET].to_numpy()
    rows = []
    for m, p in pred.items():
        per_fold = [nmad(y[te] - p[te]) for _, te in folds]
        r = y - p
        rows.append(dict(model=m, nmad=nmad(r), nmad_fold_std=float(np.std(per_fold)),
                         r=stats.pearsonr(p, y)[0], rho=stats.spearmanr(p, y)[0],
                         n=len(y)))
    return pd.DataFrame(rows)


def page_scores(pdf, sc, floor, n_start, title):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(sc))
    ax.bar(x, sc["nmad"], yerr=sc["nmad_fold_std"], color="C0", capsize=4)
    for xi, v in zip(x, sc["nmad"]):
        ax.text(xi, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
    ax.axhline(n_start, color="gray", ls="--", label=f"no model: {n_start:.4f} arcsec")
    ax.axhline(floor, color="C3", ls=":", label=f"raft-scatter floor: {floor:.4f} arcsec")
    ax.set_xticks(x, sc["model"])
    ax.set_ylabel("held-out residual nMAD (arcsec)")
    ax.set_title(f"{title}; n={sc['n'].iloc[0]} visits; {N_FOLDS} folds by night; "
                 "error bar std over folds")
    ax.legend()
    pdf.savefig(fig)
    return fig


def page_pred_obs(pdf, d, pred, models):
    y = d[TARGET].to_numpy()
    lim = np.percentile(y, [0.5, 99.5])
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    for ax, m in zip(axes, models):
        ax.hexbin(pred[m], y, gridsize=60, bins="log", mincnt=1, extent=(*lim, *lim))
        ax.plot(lim, lim, "r-", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{m} held-out prediction (arcsec)")
        ax.set_ylabel(f"observed {TARGET_LABEL}")
        ax.set_title(f"{m}: residual nMAD {nmad(y - pred[m]):.4f} arcsec\n"
                     f"r={stats.pearsonr(pred[m], y)[0]:+.3f}, "
                     f"rho={stats.spearmanr(pred[m], y)[0]:+.3f} (dimensionless)")
    fig.suptitle("Held-out prediction against observation; colour log count per hexagon")
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def page_importance(pdf, imp):
    t = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 11))
    y = np.arange(len(t))
    ax.barh(y, t["rise_mean"], xerr=t["rise_std"], color="C2")
    ax.set_yticks(y, t["feature"], fontsize=8)
    ax.set_xlabel("rise in held-out residual nMAD when the input is shuffled (arcsec)")
    ax.set_title(f"gbt_all permutation importance, mean over {N_FOLDS} night folds "
                 f"({N_PERM} shuffles, up to 8000 visits each)")
    pdf.savefig(fig)
    plt.close(fig)


def page_partial_dependence(pdf, d, imp):
    cols = list(ALL_FEATURES)
    model = gbt().fit(d[cols].to_numpy(), d[TARGET].to_numpy())
    top = list(imp["column"][:N_PDP])
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, c in zip(axes.flat, top):
        j = cols.index(c)
        pdp = partial_dependence(model, d[cols].to_numpy(), [j],
                                 percentiles=(0.01, 0.99), grid_resolution=60)
        ax.plot(pdp["grid_values"][0], pdp["average"][0], "C0-")
        ax.set_xlabel(ALL_FEATURES[c], fontsize=8.5)
        ax.set_ylabel("partial dependence (arcsec)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("gbt_all partial dependence, top inputs by permutation importance "
                 "(fit on all visits; 1st to 99th percentile of each input)")
    pdf.savefig(fig)
    plt.close(fig)


def page_residual_nightly(pdf, d, pred, models):
    fig, ax = plt.subplots(figsize=(14, 6))
    for k, m in enumerate(models):
        r = pd.Series(d[TARGET].to_numpy() - pred[m], index=d.index)
        g = r.groupby(d["day_obs"])
        night = pd.DataFrame({"median": g.median(), "n": g.size()})
        night = night[night["n"] >= 20]
        date = pd.to_datetime(night.index.astype(str), format="%Y%m%d")
        ax.plot(date, night["median"], "o", ms=3, color=f"C{k}",
                label=f"{m}: nMAD of nightly medians {nmad(night['median']):.4f} arcsec")
    ax.axhline(0, color="gray", lw=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("day_obs")
    ax.set_ylabel("nightly median held-out residual (arcsec)")
    ax.set_title("Nightly median of the held-out residual, nights with >= 20 visits")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)


def page_histograms(pdf, d, pred, models, out_dir):
    y = d[TARGET].to_numpy()
    lo, hi = np.percentile(y - np.median(y), [0.2, 99.8])
    bins = np.linspace(lo, hi, 121)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(y - np.median(y), bins=bins, histtype="step", color="gray", lw=1.5,
            label=f"no model: nMAD {nmad(y):.4f} arcsec")
    for k, m in enumerate(models):
        r = y - pred[m]
        ax.hist(r, bins=bins, histtype="step", color=f"C{k}", lw=1.5,
                label=f"{m}: nMAD {nmad(r):.4f} arcsec")
    ax.set_xlabel("held-out residual, or blur difference minus its median (arcsec)")
    ax.set_ylabel("visits per bin")
    ax.set_title(f"Intra - extra blur residuals, n={len(y)} visits")
    ax.legend()
    pdf.savefig(fig)
    fig.savefig(out_dir / "gbt_residual_histograms.png", dpi=110)
    plt.close(fig)


def main():
    setup_plotting()
    run = study_run("aos/cwfs_blur_intra_extra", "blur_gbt")
    rng = np.random.default_rng(0)
    full = add_wind(add_thermal(per_visit()))
    floor = raft_noise_floor()
    print("raft-scatter floor per n_rafts:", floor)
    n_by = full["n_rafts"].value_counts()
    floor_mix = float(sum(floor[k]["floor"] * n_by.get(k, 0) for k in floor) / n_by.sum())

    d = full.dropna(subset=[TARGET] + list(THERMAL_FEATURES)).reset_index(drop=True)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(d, groups=d["day_obs"]))
    pred = oof_predictions(d, folds)
    sc = scores(d, pred, folds)
    sc.to_csv(run.path / "gbt_scores.csv", index=False)

    # gbt_all on every visit, missing inputs allowed, to see the full-sample score.
    fa = full.dropna(subset=[TARGET]).reset_index(drop=True)
    folds_a = list(GroupKFold(n_splits=N_FOLDS).split(fa, groups=fa["day_obs"]))
    ya = fa[TARGET].to_numpy()
    pa = np.full(len(fa), np.nan)
    for tr, te in folds_a:
        X = fa[list(ALL_FEATURES)].to_numpy()
        pa[te] = gbt().fit(X[tr], ya[tr]).predict(X[te])
    all_visits = dict(n=len(fa), nmad_start=nmad(ya), nmad_gbt_all=nmad(ya - pa))

    imp = permutation_nmad(d, folds, list(ALL_FEATURES), rng)
    imp.to_csv(run.path / "gbt_permutation_importance.csv", index=False)

    models = ["huber3", "gbt_thermal", "gbt_all"]
    with PdfPages(run.path / "blur_gbt.pdf") as pdf:
        fig = page_scores(pdf, sc, floor_mix, nmad(d[TARGET]),
                          "Intra - extra blur, held-out residual nMAD by model")
        fig.savefig(run.path / "gbt_scores.png", dpi=110)
        plt.close(fig)
        page_pred_obs(pdf, d, pred, models)
        page_histograms(pdf, d, pred, models, run.path)
        page_residual_nightly(pdf, d, pred, models)
        page_importance(pdf, imp)
        page_partial_dependence(pdf, d, imp)

    pd.set_option("display.width", 200)
    print(sc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"raft-scatter floor (visit mix) {floor_mix:.4f} arcsec")
    print("all visits, missing allowed:", all_visits)
    print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"written to {run.path}")


if __name__ == "__main__":
    main()
