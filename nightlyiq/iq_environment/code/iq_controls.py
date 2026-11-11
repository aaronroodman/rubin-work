"""Airflow and blur-difference correlations with wind speed and elevation controlled,
and the top-ring sonic axis-2 correlation split by pointing relative to the wind.

Two controls on every result: partial Spearman rho (ranks regressed on the ranks of
wind speed and elevation) and a Huber fit, target ~ input + wind speed + airmass. Every
result is given over all visits and within night (night medians subtracted from every
variable). One PDF and CSV tables in the run directory.

Usage::

    python nightlyiq/iq_environment/code/iq_controls.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import iq_correlations as iqc  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

TARGETS = ["fwhm", "fwhm_z500", "aos_fwhm", "donut_blur_fwhm", "psf_e_azel_p50", "e_coh"]
TMA = iqc.TMA_SONIC
INPUTS = ([f"sonic{i}_mean_2_ms" for i in TMA] + [f"sonic{i}_abs2_ms" for i in TMA]
          + [f"sonic{i}_cross_ms" for i in TMA] + [f"sonic{i}_rms_2_ms" for i in TMA]
          + ["blur_diff"])
CONTROLS = ["wind_speed_ms", "altitude_deg"]
# Pointing relative to the wind: |into_wind_deg|, 0 deg = pointing into the wind.
DIR_BINS = [(0, 45, "into wind (0-45 deg)"), (45, 135, "cross wind (45-135 deg)"),
            (135, 180, "down wind (135-180 deg)")]
MIN_BIN = 300
COLORS = ["#56B4E9", "#0072B2", "#000000"]


def within_night(d, cols):
    """Night medians subtracted, nights with >= MIN_NIGHT visits."""
    d = d[d.groupby("day_obs")["visit_id"].transform("size") >= iqc.MIN_NIGHT].copy()
    d[cols] = d[cols] - d.groupby("day_obs")[cols].transform("median")
    return d


def partial_spearman(s, t, x, controls):
    """Spearman rho of t and x after regressing both ranks on the control ranks."""
    r = s[[t, x] + controls].rank()
    a = sm.add_constant(r[controls].to_numpy())
    rt = r[t].to_numpy() - a @ np.linalg.lstsq(a, r[t].to_numpy(), rcond=None)[0]
    rx = r[x].to_numpy() - a @ np.linalg.lstsq(a, r[x].to_numpy(), rcond=None)[0]
    return stats.pearsonr(rt, rx)[0]


def huber(s, t, x):
    """Huber RLM t ~ x + wind speed + airmass; slope of x and its standard error."""
    X = sm.add_constant(s[[x, "wind_speed_ms", "airmass"]].to_numpy())
    fit = sm.RLM(s[t].to_numpy(), X, M=sm.robust.norms.HuberT()).fit()
    return fit.params[1], fit.bse[1]


def control_table(d, mode):
    """One row per (target, input): raw and partial Spearman, Huber slope."""
    rows = []
    for t in TARGETS:
        for x in INPUTS:
            s = d[["day_obs", t, x] + CONTROLS + ["airmass"]].dropna()
            if len(s) < 500:
                continue
            rows.append(dict(mode=mode, target=t, input=x, n=len(s),
                             rho=stats.spearmanr(s[t], s[x])[0],
                             rho_partial=partial_spearman(s, t, x, CONTROLS),
                             huber_slope=(h := huber(s, t, x))[0], huber_se=h[1]))
    return pd.DataFrame(rows)


def split_table(d, mode):
    """Spearman rho of target against sonic axis 2, by wind direction and speed tercile."""
    edges = d.wind_speed_ms.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    d = d.assign(abs_into=d.into_wind_deg.abs(),
                 wbin=pd.cut(d.wind_speed_ms, edges, labels=False, include_lowest=True))
    rows = []
    for t in TARGETS:
        for i in TMA:
            x = f"sonic{i}_mean_2_ms"
            for lo, hi, dlab in DIR_BINS:
                for w in range(3):
                    s = d[(d.abs_into >= lo) & (d.abs_into < hi + (hi == 180))
                          & (d.wbin == w)][[t, x]].dropna()
                    rows.append(dict(mode=mode, target=t, sonic=i, direction=dlab,
                                     wind_bin=w, n=len(s),
                                     rho=stats.spearmanr(s[t], s[x])[0]
                                     if len(s) >= MIN_BIN else np.nan,
                                     axis2_median=s[x].median()))
    return pd.DataFrame(rows), edges


def page_text(pdf, title, lines):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.89, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def page_raw_vs_partial(pdf, tab, mode):
    """Raw against partial Spearman rho, one panel per target."""
    sub = tab[tab["mode"] == mode]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, TARGETS):
        s = sub[sub.target == t]
        y = np.arange(len(s))
        ax.plot(s.rho, y, "o", color="#0072B2", ms=5, label="raw")
        ax.plot(s.rho_partial, y, "s", color="#E69F00", ms=5, label="wind speed, elevation controlled")
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yticks(y, [iqc.INPUTS[x] for x in s.input], fontsize=6)
        ax.set_title(iqc.TARGETS[t], fontsize=9)
        ax.set_xlabel("Spearman rho (dimensionless)")
        ax.grid(axis="y", color="0.92")
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Spearman rho, raw and partial, {mode}")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_huber(pdf, tab):
    """Huber slopes for the airflow inputs, all visits against within night."""
    air = [x for x in INPUTS if x != "blur_diff"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharey=True)
    for ax, t in zip(axes.flat, TARGETS):
        for k, (mode, c, off) in enumerate((("all visits", "#0072B2", -0.15),
                                            ("within night", "#D55E00", 0.15))):
            s = tab[(tab["mode"] == mode) & (tab.target == t)].set_index("input").reindex(air)
            y = np.arange(len(air)) + off
            ax.errorbar(s.huber_slope, y, xerr=s.huber_se, fmt="o", color=c, ms=4,
                        lw=1, label=mode)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yticks(np.arange(len(air)), [iqc.INPUTS[x] for x in air], fontsize=6)
        unit = "arcsec" if "fwhm" in t else "dimensionless"
        ax.set_xlabel(f"Huber slope ({unit} per m/s)", fontsize=8)
        ax.set_title(iqc.TARGETS[t], fontsize=9)
        ax.grid(axis="y", color="0.92")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Huber fit target ~ airflow input + wind speed + airmass (slope, 1 sigma)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_split(pdf, sp, edges, mode, targets=("fwhm_z500", "donut_blur_fwhm", "aos_fwhm")):
    sub = sp[sp["mode"] == mode]
    fig, axes = plt.subplots(len(targets), len(TMA), figsize=(14, 8.5), sharex=True,
                             sharey=True)
    for r, t in enumerate(targets):
        for c, i in enumerate(TMA):
            ax = axes[r, c]
            for w in range(3):
                s = sub[(sub.target == t) & (sub.sonic == i) & (sub.wind_bin == w)]
                ax.plot(range(3), s.rho, "o-", color=COLORS[w], ms=5,
                        label=f"wind {edges[w]:.1f}-{edges[w + 1]:.1f} m/s")
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(3), ["into", "cross", "down"])
            if r == 0:
                ax.set_title(f"sonic {i} axis 2", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"rho, {iqc.TARGETS[t]}", fontsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Spearman rho of IQ against sonic axis 2, by pointing relative to the wind "
                 f"and wind-speed tercile, {mode} (bins with n >= {MIN_BIN})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_split_axis2(pdf, sp, edges):
    """Median axis-2 flow per bin: what the flow does when the dome turns."""
    sub = sp[(sp["mode"] == "all visits") & (sp.target == "fwhm_z500")]
    fig, axes = plt.subplots(1, len(TMA), figsize=(14, 4), sharey=True)
    for ax, i in zip(axes, TMA):
        for w in range(3):
            s = sub[(sub.sonic == i) & (sub.wind_bin == w)]
            ax.plot(range(3), s.axis2_median, "o-", color=COLORS[w], ms=5,
                    label=f"wind {edges[w]:.1f}-{edges[w + 1]:.1f} m/s")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(3), ["into", "cross", "down"])
        ax.set_title(f"sonic {i}", fontsize=9)
        ax.set_ylabel("median axis-2 speed (m/s)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Median sonic axis-2 flow by pointing relative to the wind (all visits)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fmt_rows(tab, mode, k=8):
    s = tab[tab["mode"] == mode].copy()
    s = s.reindex(s.rho_partial.abs().sort_values(ascending=False).index)[:k]
    return [f"  {r.target:16s} {r.input:20s} rho {r.rho:+.3f}  partial {r.rho_partial:+.3f}  "
            f"Huber {r.huber_slope:+.4f} +/- {r.huber_se:.4f}  n {r.n}" for r in s.itertuples()]


def main():
    run = study_run("nightlyiq/iq_environment", "iq_controls")
    d = iqc.load()
    d["airmass"] = 1.0 / np.sin(np.radians(d.altitude_deg))
    cols = TARGETS + INPUTS + CONTROLS + ["airmass"]
    dw = within_night(d, cols)
    tab = pd.concat([control_table(d, "all visits"), control_table(dw, "within night")])
    tab.to_csv(run.path / "controls.csv", index=False)
    # Within night the direction and wind-speed bins stay absolute; only the IQ and the
    # axis-2 flow are night-subtracted.
    dws = within_night(d, TARGETS + [f"sonic{i}_mean_2_ms" for i in TMA])
    sp_a, edges = split_table(d, "all visits")
    sp_w, _ = split_table(dws, "within night")
    sp = pd.concat([sp_a, sp_w])
    sp.to_csv(run.path / "wind_direction_split.csv", index=False)

    lines = [f"{len(d)} science visits, {d.day_obs.nunique()} nights (sample as in iq_correlations)",
             "Partial Spearman: ranks of target and input regressed on ranks of wind speed and "
             "elevation.",
             "Huber: RLM(target ~ input + wind speed + airmass, HuberT); slope in target units "
             "per input unit.",
             f"Within night: night medians subtracted from every variable, nights with >= "
             f"{iqc.MIN_NIGHT} visits.",
             f"Wind-speed terciles (m/s): {edges[0]:.2f}, {edges[1]:.2f}, {edges[2]:.2f}, "
             f"{edges[3]:.2f}",
             "", "Largest |partial rho|, all visits (rho, Huber slope dimensionless or "
             "arcsec per input unit):"]
    lines += fmt_rows(tab, "all visits")
    lines += ["", "Largest |partial rho|, within night:"] + fmt_rows(tab, "within night")
    out = run.path / "iq_controls.pdf"
    with PdfPages(out) as pdf:
        page_text(pdf, "IQ against airflow and blur difference, wind and elevation controlled",
                  lines)
        page_raw_vs_partial(pdf, tab, "all visits")
        page_raw_vs_partial(pdf, tab, "within night")
        page_huber(pdf, tab)
        page_split_axis2(pdf, sp, edges)
        page_split(pdf, sp, edges, "all visits")
        page_split(pdf, sp, edges, "within night")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
