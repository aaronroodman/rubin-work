"""Intra - extra CWFS blur per science visit: time series, thermal correlations, forward
selection, and correlations with mean blur, pointing and the symmetric Zernikes.

Per visit, each corner raft whose two halves both have at least `MIN_DONUTS` quality-cut
donuts gives ``blur_median(SW1) - blur_median(SW0)``; the visit value is the median over
those rafts, kept if at least `MIN_RAFTS` contribute.

Run from this study's directory, after ``code/fetch_thermal.py``::

    python code/blur_thermal.py
"""

import pathlib
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lsst.utils.plotting import get_multiband_plot_colors
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from rubinwork.common import study_run
from rubinwork.common.utils import nmad, setup_plotting
from rubinwork.forward_selection import huber_fit, select_predictors
from rubinwork.products import cwfs_science

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fetch_thermal import CACHE  # noqa: E402

VARIANT = "d13-blitz_rso1071"
MIN_DONUTS = 3
MIN_RAFTS = 2
MIN_NIGHT_VISITS = 20

THERMAL_FEATURES = {
    "m1m3_glass_minus_air_c": "M1M3 glass - M1M3 air (deg C)",
    "m1_glass_minus_air_c": "M1 glass - M1M3 air (deg C)",
    "m3_glass_minus_air_c": "M3 glass - M1M3 air (deg C)",
    "m1_minus_m3_glass_c": "M1 glass - M3 glass (deg C)",
    "m1m3_mean_temp_c": "M1M3 glass temperature (deg C)",
    "truss_minus_m1m3_air_c": "truss - M1M3 air (deg C)",
    "truss_minus_m2_air_c": "truss - M2 air (deg C)",
    "m2_minus_m1m3_air_c": "M2 air - M1M3 air (deg C)",
    "cam_minus_m1m3_air_c": "camera air - M1M3 air (deg C)",
    "cam_AverageTemp": "camera body average temperature (deg C)",
    "cam_body_minus_m1m3_air_c": "camera body - M1M3 air (deg C)",
    "m2_air_minus_outside_c": "M2 air - outside (deg C)",
    "m1m3_x_gradient_c_per_m": "M1M3 x gradient (deg C/m)",
    "m1m3_y_gradient_c_per_m": "M1M3 y gradient (deg C/m)",
    "m1m3_z_gradient_c_per_m": "M1M3 z gradient (deg C/m)",
    "m1m3_radial_gradient_c_per_m": "M1M3 radial gradient (deg C/m)",
    "m1m3_r2_coeff_c": "M1M3 quadratic-in-radius (deg C)",
    "m1m3_std_temp_c": "M1M3 thermocouple scatter (deg C)",
    "m1m3_range_temp_c": "M1M3 thermocouple range (deg C)",
    "truss_temp_mean_c": "mean truss temperature (deg C)",
    "m1m3_air_temp": "M1M3 air temperature (deg C)",
    "outside_temp": "outside air temperature (deg C)",
}

OTHER_FEATURES = {
    "blur_mean": "mean of intra and extra blur (arcsec)",
    "alt": "TMA elevation (deg)",
    "az": "TMA azimuth (deg)",
    "z4_mean": "Z4 deviation, mean of corners (um)",
    "z11_mean": "Z11 deviation, mean of corners (um)",
    "z22_mean": "Z22 deviation, mean of corners (um)",
}

TARGET = "blur_diff"
TARGET_LABEL = "intra - extra blur (arcsec)"


def per_visit():
    """One row per visit: the blur difference and mean, pointing, and the Zernike means."""
    ccd = cwfs_science.load_table("ccd_visits", VARIANT)
    ccd = ccd[ccd["n_quality"] >= MIN_DONUTS].copy()
    for j in (4, 11, 22):
        ccd[f"dev_z{j}"] = ccd[f"zk_OCS_median_z{j}"] - ccd[f"zk_intrinsic_OCS_median_z{j}"]
    p = ccd.pivot_table(index=["visit", "raft"], columns="wfs_offset",
                        values="blur_median").dropna()
    raft = pd.DataFrame({"diff": p[-1] - p[0], "mean": 0.5 * (p[-1] + p[0])}).reset_index()
    g = raft.groupby("visit")
    v = pd.DataFrame({TARGET: g["diff"].median(), "blur_mean": g["mean"].median(),
                      "n_rafts": g.size()})
    v = v[v["n_rafts"] >= MIN_RAFTS]
    # The Zernike means use the same rafts as the blur difference.
    keep = ccd.merge(raft[["visit", "raft"]], on=["visit", "raft"])
    zk = keep.groupby("visit")[["dev_z4", "dev_z11", "dev_z22"]].mean()
    zk.columns = ["z4_mean", "z11_mean", "z22_mean"]
    info = (ccd.groupby("visit")[["day_obs", "seq_num", "band", "mjd", "alt", "az",
                                  "science_program", "observation_reason"]].first())
    v = v.join(zk).join(info).reset_index().sort_values("visit").reset_index(drop=True)
    v["ordinal"] = np.arange(len(v))
    v["time"] = pd.to_datetime(v["mjd"] - 40587, unit="D")
    return v


def add_thermal(v):
    t = pd.read_parquet(CACHE).drop(columns=["day_obs", "seq_num"])
    d = v.merge(t, on="visit", how="left")
    d["truss_minus_m1m3_air_c"] = d["truss_temp_mean_c"] - d["m1m3_air_temp"]
    d["truss_minus_m2_air_c"] = d["truss_temp_mean_c"] - d["m2_air_temp"]
    d["m2_minus_m1m3_air_c"] = d["m2_air_temp"] - d["m1m3_air_temp"]
    d["cam_minus_m1m3_air_c"] = d["cam_air_temp"] - d["m1m3_air_temp"]
    d["m2_air_minus_outside_c"] = d["m2_air_temp"] - d["outside_temp"]
    d["m1m3_glass_minus_air_c"] = d["m1m3_mean_temp_c"] - d["m1m3_air_temp"]
    d["m1_glass_minus_air_c"] = d["m1_mean_temp_c"] - d["m1m3_air_temp"]
    d["m3_glass_minus_air_c"] = d["m3_mean_temp_c"] - d["m1m3_air_temp"]
    d["m1_minus_m3_glass_c"] = d["m1_mean_temp_c"] - d["m3_mean_temp_c"]
    d["cam_body_minus_m1m3_air_c"] = d["cam_AverageTemp"] - d["m1m3_air_temp"]
    return d


def line_stats(x, y):
    """Huber slope and intercept, Pearson r, Spearman rho and n, on finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    fit = huber_fit(y, x)
    return dict(n=int(ok.sum()), slope=float(fit.params[1]), slope_err=float(fit.bse[1]),
                intercept=float(fit.params[0]), r=float(stats.pearsonr(x, y)[0]),
                rho=float(stats.spearmanr(x, y)[0]))


def within_night_rho(d, col):
    sub = d[np.isfinite(d[col]) & np.isfinite(d[TARGET])]
    sub = sub[sub.groupby("day_obs")[col].transform("size") >= MIN_NIGHT_VISITS]
    x = sub[col] - sub.groupby("day_obs")[col].transform("mean")
    y = sub[TARGET] - sub.groupby("day_obs")[TARGET].transform("mean")
    return float(stats.spearmanr(x, y)[0]), int(sub["day_obs"].nunique())


def scatter_panel(ax, x, y, xlabel, s, ylabel=TARGET_LABEL):
    ok = np.isfinite(x) & np.isfinite(y)
    lo, hi = np.nanpercentile(x[ok], [0.5, 99.5])
    ylo, yhi = np.nanpercentile(y[ok], [0.5, 99.5])
    ax.hexbin(x[ok], y[ok], gridsize=60, bins="log", cmap="viridis", mincnt=1,
              extent=(lo, hi, ylo, yhi))
    xx = np.array([lo, hi])
    ax.plot(xx, s["intercept"] + s["slope"] * xx, "r-", lw=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"r={s['r']:+.3f}  rho={s['rho']:+.3f}  n={s['n']}\n"
                 f"Huber slope {s['slope']:+.4g} +- {s['slope_err']:.2g} arcsec/unit",
                 fontsize=8.5)
    ax.tick_params(labelsize=8)


def page_ordinal(pdf, v):
    colors = get_multiband_plot_colors()
    fig, ax = plt.subplots(figsize=(14, 7))
    for band in "ugrizy":
        b = v[v["band"] == band]
        ax.scatter(b["ordinal"], b[TARGET], s=1.5, alpha=0.4, color=colors[band],
                   rasterized=True, label=f"{band} (n={len(b)} visits)")
    month = v["time"].dt.to_period("M")
    starts = v.groupby(month)["ordinal"].min()
    for per, o in starts.items():
        ax.axvline(o, color="k", lw=0.6, alpha=0.5)
        ax.text(o, 1.0, str(per), transform=ax.get_xaxis_transform(), fontsize=8,
                rotation=90, va="top", ha="right")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(np.nanpercentile(v[TARGET], [0.1, 99.9]))
    ax.set_xlabel("visit ordinal (dimensionless, time order)")
    ax.set_ylabel(TARGET_LABEL)
    ax.set_title(f"CWFS intra - extra blur per science visit: median "
                 f"{v[TARGET].median():+.4f} arcsec, nMAD {nmad(v[TARGET]):.4f} arcsec, "
                 f"n={len(v)} visits; vertical lines mark month starts")
    ax.legend(markerscale=6, loc="lower left", fontsize=9)
    pdf.savefig(fig, dpi=150)
    return fig


def page_nightly(pdf, v):
    n = v.groupby("day_obs")
    night = pd.DataFrame({"median": n[TARGET].median(), "nmad": n[TARGET].agg(nmad),
                          "n": n.size()})
    night["err"] = 1.2533 * night["nmad"] / np.sqrt(night["n"])
    night["date"] = pd.to_datetime(night.index.astype(str), format="%Y%m%d")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  height_ratios=(3, 1))
    ax.errorbar(night["date"], night["median"], yerr=night["err"], fmt="o", ms=3,
                lw=0.8, color="C0")
    ax.axhline(v[TARGET].median(), color="gray", lw=0.8, ls="--")
    ax.set_ylabel(f"nightly median {TARGET_LABEL}")
    ax.set_title(f"Nightly median intra - extra blur, n={len(night)} nights; "
                 "error bar 1.2533 nMAD / sqrt(n) (arcsec)")
    ax2.bar(night["date"], night["n"], width=0.8, color="C1")
    ax2.set_ylabel("visits per night")
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.set_xlabel("day_obs")
    pdf.savefig(fig)
    return fig, night


def pages_scatter(pdf, d, features, title, per_page=12):
    rows = []
    names = list(features)
    for c in names:
        s = line_stats(d[c].to_numpy(float), d[TARGET].to_numpy(float))
        s["rho_within_night"], s["n_nights"] = within_night_rho(d, c)
        rows.append(dict(column=c, feature=features[c], **s))
    for start in range(0, len(names), per_page):
        chunk = names[start:start + per_page]
        fig, axes = plt.subplots(3, 4, figsize=(17, 12))
        for ax, c, s in zip(axes.flat, chunk, rows[start:start + per_page]):
            scatter_panel(ax, d[c].to_numpy(float), d[TARGET].to_numpy(float),
                          features[c], s)
        for ax in axes.flat[len(chunk):]:
            ax.set_visible(False)
        fig.suptitle(f"{title} ({start // per_page + 1}); r Pearson, rho Spearman "
                     "(dimensionless); colour log count per hexagon; red Huber line")
        pdf.savefig(fig, dpi=120)
        plt.close(fig)
    return pd.DataFrame(rows)


def page_ranking(pdf, table):
    t = table.reindex(table["rho"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(12, 9))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["rho"], height=0.4, label="Spearman rho, all visits")
    ax.barh(y + 0.2, t["rho_within_night"], height=0.4,
            label="Spearman rho, each night's mean removed")
    ax.set_yticks(y, t["feature"], fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman rho with intra - extra blur (dimensionless)")
    ax.set_title(f"Thermal quantities ranked by |rho| (n up to {t['n'].max()} visits; "
                 f"within-night over nights with >= {MIN_NIGHT_VISITS} visits)")
    ax.legend(loc="lower right")
    pdf.savefig(fig)
    plt.close(fig)


def page_selection(pdf, d, steps, fit, frame):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flat
    y = frame[TARGET].to_numpy()
    chosen = []
    for i, row in steps.iterrows():
        chosen.append(row["variable"])
        f = huber_fit(y, frame[chosen].to_numpy())
        ax = axes[i]
        lim = np.percentile(y, [0.5, 99.5])
        ax.hexbin(f.fittedvalues, y, gridsize=50, bins="log", mincnt=1, cmap="viridis",
                  extent=(*lim, *lim))
        ax.plot(lim, lim, "r-", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("predicted (arcsec)", fontsize=9)
        ax.set_ylabel("observed (arcsec)", fontsize=9)
        ax.set_title(f"step {i + 1}: + {THERMAL_FEATURES[row['variable']]}\n"
                     f"residual nMAD {row['nmad_after']:.4f} arcsec, gain "
                     f"{row['gain_frac']:.3f}, VIF {row['vif']:.2f}", fontsize=8.5)
    for ax in axes[len(steps):]:
        ax.axis("off")
    ax = axes[len(steps)]
    lines = [f"Greedy forward selection, Huber, n={len(frame)} visits",
             f"start nMAD {nmad(y - np.median(y)):.4f} arcsec", ""]
    for i, row in steps.iterrows():
        j = i + 1
        lines.append(f"{j}. {THERMAL_FEATURES[row['variable']]}")
        lines.append(f"   slope {fit.params[j]:+.4g} +- {fit.bse[j]:.2g} arcsec/unit")
        lines.append(f"   rho own {row['rho_own']:+.3f}, rho vs residual "
                     f"{row['rho_partial']:+.3f}")
    lines.append(f"intercept {fit.params[0]:+.4f} arcsec")
    pred = fit.fittedvalues
    lines.append(f"pred vs obs: r={stats.pearsonr(pred, y)[0]:+.3f}, "
                 f"rho={stats.spearmanr(pred, y)[0]:+.3f}")
    lines += ["", "gain: fractional drop in residual nMAD", "(dimensionless, of amplitude)",
              "VIF: variance inflation factor", "(dimensionless)"]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9,
            transform=ax.transAxes)
    fig.suptitle("Forward selection of thermal predictors of intra - extra blur")
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def main():
    setup_plotting()
    run = study_run("aos/cwfs_blur_intra_extra", "blur_thermal")
    v = per_visit()
    d = add_thermal(v)
    d.to_parquet(run.path / "blur_diff_per_visit.parquet", index=False)
    print(f"{len(d)} visits, median {TARGET_LABEL} {d[TARGET].median():+.4f}, "
          f"nMAD {nmad(d[TARGET]):.4f} arcsec")

    with PdfPages(run.path / "blur_diff_science.pdf") as pdf:
        fig = page_ordinal(pdf, d)
        fig.savefig(run.path / "blur_diff_vs_ordinal.png", dpi=110)
        plt.close(fig)
        fig, night = page_nightly(pdf, d)
        fig.savefig(run.path / "blur_diff_nightly.png", dpi=110)
        plt.close(fig)
        night.to_csv(run.path / "blur_diff_nightly.csv")

        thermal = pages_scatter(pdf, d, THERMAL_FEATURES, "Intra - extra blur vs thermal")
        thermal = thermal.reindex(thermal["rho"].abs().sort_values(ascending=False).index)
        thermal.to_csv(run.path / "thermal_correlations.csv", index=False)
        page_ranking(pdf, thermal)

        steps, ranking, fit = select_predictors(d[TARGET], d[list(THERMAL_FEATURES)],
                                                n_max=6, min_gain_frac=0.01)
        frame = pd.concat([d[TARGET], d[list(THERMAL_FEATURES)]], axis=1).dropna()
        steps.to_csv(run.path / "forward_selection_steps.csv", index=False)
        page_selection(pdf, d, steps, fit, frame)

        other = pages_scatter(pdf, d, OTHER_FEATURES,
                              "Intra - extra blur vs mean blur, pointing, Zernikes")
        other.to_csv(run.path / "other_correlations.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["feature", "n", "slope", "slope_err", "r", "rho", "rho_within_night"]
    print(thermal[cols].to_string(index=False, float_format=lambda x: f"{x:+.4g}"))
    print(steps.to_string(index=False, float_format=lambda x: f"{x:+.4g}"))
    print("final fit params", fit.params, "bse", fit.bse)
    print(other[cols].to_string(index=False, float_format=lambda x: f"{x:+.4g}"))
    print(f"written to {run.path}")


if __name__ == "__main__":
    main()
