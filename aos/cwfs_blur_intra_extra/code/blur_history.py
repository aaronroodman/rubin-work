"""Thermal history and the 2026-04-06 camera-region cover against the intra - extra blur.

Two questions:

1. Do rates of change, time since twilight and the thermal state at twilight
   (`fetch_history`) improve the prediction? Scored as in `blur_gbt`: 5 folds by whole
   night, held-out residual nMAD.
2. Did the neoprene cover over the camera utility trunk, camera hexapod and rotator,
   installed on 2026-04-06 in place of a thin rubber one, change the blur difference,
   before or after the telemetry correction? Visits with ``day_obs`` >= `COVER_DAY_OBS`
   count as after.

Run from this study's directory, after the three fetch scripts::

    python code/blur_history.py
"""

import pathlib
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import GroupKFold

from rubinwork.common import study_run
from rubinwork.common.utils import nmad, setup_plotting
from rubinwork.forward_selection import huber_fit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from blur_gbt import ALL_FEATURES, HUBER3, N_FOLDS, gbt, permutation_nmad  # noqa: E402
from blur_thermal import (TARGET, TARGET_LABEL, THERMAL_FEATURES, add_thermal,  # noqa: E402
                          add_wind, pages_scatter, page_ranking, per_visit)
from fetch_history import CACHE as HISTORY_CACHE  # noqa: E402
from fetch_history import SENSORS  # noqa: E402

COVER_DAY_OBS = 20260406
SENSOR_LABEL = {"cam_air": "camera air", "m2_air": "M2 air", "m1m3_air": "M1M3 air",
                "outside": "outside air", "truss": "mean truss", "cam_body": "camera body"}
HISTORY_FEATURES = {"hours_after_twilight": "time after 0 deg twilight (h)"}
for _s in SENSORS:
    _n = SENSOR_LABEL[_s]
    HISTORY_FEATURES |= {
        f"{_s}_rate30": f"{_n} rate, previous 30 min (deg C/h)",
        f"{_s}_rate60": f"{_n} rate, previous 60 min (deg C/h)",
        f"{_s}_twi": f"{_n} at twilight (deg C)",
        f"{_s}_since_twi": f"{_n} change since twilight (deg C)",
    }
HISTORY_FEATURES |= {
    "cam_body_minus_m1m3_air_twi": "camera body - M1M3 air at twilight (deg C)",
}
FEATURES_H = {**ALL_FEATURES, **HISTORY_FEATURES}


def load():
    d = add_wind(add_thermal(per_visit()))
    d = d.merge(pd.read_parquet(HISTORY_CACHE), on="visit", how="left")
    d["cam_body_minus_m1m3_air_twi"] = d["cam_body_twi"] - d["m1m3_air_twi"]
    d["after_cover"] = d["day_obs"] >= COVER_DAY_OBS
    return d


def oof(d, cols, folds):
    y = d[TARGET].to_numpy()
    X = d[cols].to_numpy()
    p = np.full(len(d), np.nan)
    for tr, te in folds:
        p[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
    return p


def oof_huber(d, folds):
    y = d[TARGET].to_numpy()
    X = d[HUBER3].to_numpy()
    p = np.full(len(d), np.nan)
    for tr, te in folds:
        f = huber_fit(y[tr], X[tr])
        p[te] = f.params[0] + X[te] @ f.params[1:]
    return p


def page_scores(pdf, rows):
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(t))
    ax.bar(x, t["nmad"], yerr=t["fold_std"], capsize=4, color="C0")
    for xi, v in zip(x, t["nmad"]):
        ax.text(xi, v + 0.002, f"{v:.4f}", ha="center")
    ax.set_xticks(x, t["model"])
    ax.set_ylabel("held-out residual nMAD (arcsec)")
    ax.set_title(f"Adding thermal history; n={t['n'].iloc[0]} visits, {N_FOLDS} folds by "
                 "night; error bar std over folds")
    pdf.savefig(fig)
    plt.close(fig)
    return t


def page_cover_time(pdf, d, series):
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 3.4 * len(series)), sharex=True)
    for ax, (label, r) in zip(axes, series.items()):
        g = pd.Series(r, index=d.index).groupby(d["day_obs"])
        night = pd.DataFrame({"median": g.median(), "n": g.size()})
        night = night[night["n"] >= 20]
        date = pd.to_datetime(night.index.astype(str), format="%Y%m%d")
        after = night.index >= COVER_DAY_OBS
        ax.plot(date[~after], night["median"][~after], "o", ms=3, color="C0",
                label="before cover")
        ax.plot(date[after], night["median"][after], "o", ms=3, color="C3",
                label="after cover")
        ax.axvline(pd.Timestamp(str(COVER_DAY_OBS)), color="k", ls="--")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("nightly median (arcsec)")
        ax.set_title(label, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("day_obs")
    fig.suptitle(f"Nightly medians, before and after the {COVER_DAY_OBS} neoprene cover "
                 "(nights with >= 20 visits)")
    pdf.savefig(fig)
    return fig


def cover_rows(d, series):
    rows = []
    a = d["after_cover"].to_numpy()
    for label, r in series.items():
        ok = np.isfinite(r)
        for name, m in (("before", ~a & ok), ("after", a & ok)):
            g = pd.Series(r[m]).groupby(d["day_obs"].to_numpy()[m]).median()
            rows.append(dict(quantity=label, period=name, n_visits=int(m.sum()),
                             n_nights=int(g.size), median=float(np.median(r[m])),
                             nmad=float(nmad(r[m])),
                             night_median_err=float(1.2533 * nmad(g) / np.sqrt(g.size))))
    return pd.DataFrame(rows)


def page_cover_hist(pdf, d, series, table, out_dir):
    a = d["after_cover"].to_numpy()
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 3.4 * len(series)))
    for ax, (label, r) in zip(axes, series.items()):
        lo, hi = np.nanpercentile(r, [0.3, 99.7])
        bins = np.linspace(lo, hi, 101)
        for name, m, c in (("before", ~a, "C0"), ("after", a, "C3")):
            row = table[(table["quantity"] == label) & (table["period"] == name)].iloc[0]
            ax.hist(r[m & np.isfinite(r)], bins=bins, density=True, histtype="step",
                    color=c, lw=1.5,
                    label=f"{name}: median {row['median']:+.4f} +- "
                          f"{row['night_median_err']:.4f} arcsec, nMAD {row['nmad']:.4f} "
                          f"arcsec, n={row['n_visits']} visits / {row['n_nights']} nights")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("visit density (1/arcsec)")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("arcsec")
    fig.suptitle(f"Before and after the {COVER_DAY_OBS} cover; median error from the "
                 "night-to-night scatter of nightly medians")
    pdf.savefig(fig)
    fig.savefig(out_dir / "cover_histograms.png", dpi=110)
    plt.close(fig)


def page_cover_slopes(pdf, d):
    """Huber 3-quantity fit separately before and after the cover."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    rows = []
    for name, m, c in (("before", ~d["after_cover"], "C0"), ("after", d["after_cover"], "C3")):
        sub = d[m].dropna(subset=[TARGET] + HUBER3)
        f = huber_fit(sub[TARGET].to_numpy(), sub[HUBER3].to_numpy())
        rows.append(dict(period=name, n=len(sub), intercept=f.params[0],
                         intercept_err=f.bse[0],
                         **{f"slope_{h}": f.params[i + 1] for i, h in enumerate(HUBER3)},
                         **{f"err_{h}": f.bse[i + 1] for i, h in enumerate(HUBER3)}))
        for i, (ax, h) in enumerate(zip(axes, HUBER3)):
            x = sub[h].to_numpy()
            lo, hi = np.percentile(x, [1, 99])
            bins = np.linspace(lo, hi, 25)
            idx = np.digitize(x, bins)
            # Partial residual: the blur difference minus the other two terms.
            others = sub[TARGET].to_numpy() - f.params[0] - sum(
                f.params[k + 1] * sub[HUBER3[k]].to_numpy() for k in range(3) if k != i)
            med = [np.median(others[idx == k]) if (idx == k).sum() > 30 else np.nan
                   for k in range(1, len(bins))]
            ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "o", color=c, ms=4)
            ax.plot([lo, hi], f.params[i + 1] * np.array([lo, hi]), "-", color=c,
                    label=f"{name}: slope {f.params[i + 1]:+.4f} +- {f.bse[i + 1]:.4f} "
                          f"arcsec/deg C, n={len(sub)} visits")
            ax.set_xlabel(THERMAL_FEATURES[h])
            ax.set_ylabel("partial residual, binned median (arcsec)")
            ax.legend(fontsize=8)
    fig.suptitle(f"Huber 3-quantity model fitted separately before and after {COVER_DAY_OBS}")
    pdf.savefig(fig)
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    setup_plotting()
    run = study_run("aos/cwfs_blur_intra_extra", "blur_history")
    rng = np.random.default_rng(0)
    full = load()
    d = full.dropna(subset=[TARGET] + list(THERMAL_FEATURES)).reset_index(drop=True)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(d, groups=d["day_obs"]))
    y = d[TARGET].to_numpy()

    p_h3 = oof_huber(d, folds)
    p_all = oof(d, list(ALL_FEATURES), folds)
    p_hist = oof(d, list(FEATURES_H), folds)
    d["after_cover_f"] = d["after_cover"].astype(float)
    p_cover = oof(d, list(FEATURES_H) + ["after_cover_f"], folds)
    rows = []
    for name, p in (("huber3", p_h3), ("gbt_all", p_all), ("gbt_all+history", p_hist),
                    ("+cover flag", p_cover)):
        rows.append(dict(model=name, nmad=nmad(y - p), n=len(y),
                         fold_std=float(np.std([nmad(y[te] - p[te]) for _, te in folds]))))
        print(name, f"{rows[-1]['nmad']:.4f}", flush=True)

    # Correlations of the history inputs with the residual of the Huber model fit to all.
    f3 = huber_fit(full.dropna(subset=[TARGET] + HUBER3)[TARGET].to_numpy(),
                   full.dropna(subset=[TARGET] + HUBER3)[HUBER3].to_numpy())
    full["huber3_resid"] = full[TARGET] - f3.params[0] - full[HUBER3].to_numpy() @ f3.params[1:]
    resid_label = "residual after Huber 3-quantity model (arcsec)"

    series = {
        f"{TARGET_LABEL}": y,
        "residual after Huber 3-quantity model (held-out)": y - p_h3,
        "residual after boosted trees, thermal + wind + pointing (held-out)": y - p_all,
        "residual after boosted trees, + thermal history (held-out)": y - p_hist,
    }
    cover = cover_rows(d, series)
    cover.to_csv(run.path / "cover_before_after.csv", index=False)

    with PdfPages(run.path / "blur_history.pdf") as pdf:
        scores = page_scores(pdf, rows)
        scores.to_csv(run.path / "history_scores.csv", index=False)
        hist = pages_scatter(pdf, full, HISTORY_FEATURES, "Intra - extra blur vs history",
                             sort=True)
        hist.to_csv(run.path / "history_correlations.csv", index=False)
        page_ranking(pdf, hist, "Thermal-history quantities")
        hres = pages_scatter(pdf, full, HISTORY_FEATURES, "Huber residual vs history",
                             sort=True, target="huber3_resid", ylabel=resid_label)
        hres.to_csv(run.path / "history_vs_huber_residual.csv", index=False)
        page_ranking(pdf, hres, "Thermal-history quantities against the Huber residual",
                     resid_label)
        imp = permutation_nmad(d, folds, list(FEATURES_H), rng, FEATURES_H)
        imp.to_csv(run.path / "history_permutation_importance.csv", index=False)
        page_importance_h(pdf, imp)

        fig = page_cover_time(pdf, d, series)
        fig.savefig(run.path / "cover_nightly.png", dpi=110)
        plt.close(fig)
        page_cover_hist(pdf, d, series, cover, run.path)
        slopes = page_cover_slopes(pdf, full)
        slopes.to_csv(run.path / "cover_huber_slopes.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["feature", "n", "slope", "r", "rho", "rho_within_night"]
    fmt = lambda x: f"{x:+.4g}"  # noqa: E731
    print(scores.to_string(index=False))
    print(hist[cols].head(15).to_string(index=False, float_format=fmt))
    print(hres[cols].head(15).to_string(index=False, float_format=fmt))
    print(imp.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(cover.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(slopes.T.to_string())
    print(f"written to {run.path}")


def page_importance_h(pdf, imp):
    t = imp.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 11))
    yy = np.arange(len(t))
    ax.barh(yy, t["rise_mean"], xerr=t["rise_std"], color="C2")
    ax.set_yticks(yy, t["feature"], fontsize=8)
    ax.set_xlabel("rise in held-out residual nMAD when the input is shuffled (arcsec)")
    ax.set_title("gbt_all+history permutation importance, top 30, mean over night folds")
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
