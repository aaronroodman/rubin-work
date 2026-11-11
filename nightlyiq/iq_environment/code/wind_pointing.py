"""Image quality against pointing relative to the wind, and whether axis-2 airflow carries it.

Pointing relative to the wind is |into_wind_deg| (0 deg = pointing into the wind). For
each IQ quantity: binned medians against it by wind-speed tercile; and Huber fits
q ~ into + cross + wind speed + airmass, down wind the baseline, per wind tercile, before
and after adding the top-ring sonic axis-2 flow. The drop in the into-wind coefficient
is the part the measured airflow carries. Every fit over all visits and within night.

Usage::

    python nightlyiq/iq_environment/code/wind_pointing.py
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ellipticity_drivers as ed  # noqa: E402
import iq_correlations as iqc  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

Q = {"fwhm_z500": "PSF FWHM, zenith 500 nm (arcsec)",
     "fwhm": "PSF FWHM (arcsec)",
     "donut_blur_fwhm": "donut blur FWHM (arcsec)",
     "aos_fwhm": "AOS FWHM (arcsec)",
     "dev_z4": "corner Z4 deviation, OCS (um)",
     "e1_azel_res": "e1 Az/El, pointing-model residual",
     "e2_azel_res": "e2 Az/El, pointing-model residual",
     "e_coh_res": "hypot(e1, e2 residuals)",
     "psf_e_azel_p50_res": "|e| p50, pointing-model residual",
     "psf_e_azel_p90_res": "|e| p90, pointing-model residual"}
AXIS2 = [f"sonic{i}_mean_2_ms" for i in iqc.TMA_SONIC]
ABS2 = [f"sonic{i}_abs2_ms" for i in iqc.TMA_SONIC]
ANG_BINS = np.arange(0, 181, 22.5)
COLORS = ["#56B4E9", "#0072B2", "#000000"]


def load():
    d = ed.load()
    ed.pointing_model(d)
    d["airmass"] = 1.0 / np.sin(np.radians(d.altitude_deg))
    d["abs_into"] = d.into_wind_deg.abs()
    d["into"] = (d.abs_into < 45).astype(float)
    d["cross"] = ((d.abs_into >= 45) & (d.abs_into < 135)).astype(float)
    edges = d.wind_speed_ms.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    d["wbin"] = pd.cut(d.wind_speed_ms, edges, labels=False, include_lowest=True)
    return d, edges


def within(d):
    cols = list(Q) + AXIS2 + ABS2 + ["wind_speed_ms", "airmass"]
    d = d[d.groupby("day_obs")["visit_id"].transform("size") >= iqc.MIN_NIGHT].copy()
    d[cols] = d[cols] - d.groupby("day_obs")[cols].transform("median")
    return d


def fit(s, q, extra):
    xs = ["into", "cross", "wind_speed_ms", "airmass"] + extra
    s = s[[q] + xs].dropna()
    X = sm.add_constant(s[xs].to_numpy(), has_constant="add")
    f = sm.RLM(s[q].to_numpy(), X, M=sm.robust.norms.HuberT()).fit()
    return f.params[1], f.bse[1], len(s)


def table(d, mode):
    rows = []
    for q in Q:
        for w in (0, 1, 2, "all"):
            s = d if w == "all" else d[d.wbin == w]
            # Same visits for the comparison: those with every axis-2 sonic.
            s = s.dropna(subset=AXIS2 + ABS2)
            b0, e0, n = fit(s, q, [])
            b1, e1, _ = fit(s, q, AXIS2 + ABS2)
            rows.append(dict(mode=mode, quantity=q, wind_bin=w, n=n, into=b0, into_se=e0,
                             into_with_axis2=b1, into_with_axis2_se=e1,
                             carried_frac=1 - b1 / b0 if b0 != 0 else np.nan))
    return pd.DataFrame(rows)


def page_binned(pdf, d, edges, qs, mode):
    fig, axes = plt.subplots(2, (len(qs) + 1) // 2, figsize=(17, 8.5))
    c = 0.5 * (ANG_BINS[1:] + ANG_BINS[:-1])
    for ax, q in zip(axes.flat, qs):
        for w in range(3):
            s = d[d.wbin == w]
            idx = np.digitize(s.abs_into, ANG_BINS)
            med = [s[q][idx == k].median() if (idx == k).sum() > 100 else np.nan
                   for k in range(1, len(ANG_BINS))]
            ax.plot(c, med, "o-", ms=4, color=COLORS[w],
                    label=f"wind {edges[w]:.1f}-{edges[w + 1]:.1f} m/s")
        ax.set_xlabel("|pointing azimuth - wind direction| (deg, 0 into wind)", fontsize=7)
        ax.set_ylabel(f"median {Q[q]}", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(qs):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Image quality against pointing relative to the wind, {mode}")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_coeffs(pdf, tab, edges):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    labels = [f"{edges[w]:.1f}-{edges[w + 1]:.1f}" for w in range(3)] + ["all"]
    for ax, q in zip(axes.flat, Q):
        for mode, c, off in (("all visits", "#0072B2", -0.12), ("within night", "#D55E00", 0.12)):
            s = tab[(tab["mode"] == mode) & (tab.quantity == q)]
            x = np.arange(len(s))
            ax.errorbar(x + off - 0.05, s.into, yerr=s.into_se, fmt="o", color=c, ms=4,
                        label=f"{mode}")
            ax.errorbar(x + off + 0.05, s.into_with_axis2, yerr=s.into_with_axis2_se, fmt="s",
                        mfc="white", color=c, ms=4, label=f"{mode}, + axis 2")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(4), labels, fontsize=7)
        ax.set_xlabel("wind speed (m/s)", fontsize=7)
        ax.set_title(Q[q], fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Into-wind minus down-wind, Huber q ~ into + cross + wind speed + airmass "
                 "(filled), and with the 4 sonic axis-2 and |axis-2| flows added (open)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_axis2(pdf, d, edges):
    """Does pointing into the wind change the axis-2 flow?"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    c = 0.5 * (ANG_BINS[1:] + ANG_BINS[:-1])
    for ax, x in zip(axes, AXIS2):
        for w in range(3):
            s = d[d.wbin == w]
            idx = np.digitize(s.abs_into, ANG_BINS)
            med = [s[x][idx == k].median() if (idx == k).sum() > 100 else np.nan
                   for k in range(1, len(ANG_BINS))]
            ax.plot(c, med, "o-", ms=4, color=COLORS[w],
                    label=f"wind {edges[w]:.1f}-{edges[w + 1]:.1f} m/s")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(iqc.AIRFLOW[x], fontsize=8)
        ax.set_xlabel("|pointing - wind| (deg)", fontsize=7)
    axes[0].set_ylabel("median axis-2 speed (m/s)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Sonic axis-2 flow against pointing relative to the wind (all visits)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    run = study_run("nightlyiq/iq_environment", "wind_pointing")
    d, edges = load()
    dw = within(d)
    tab = pd.concat([table(d, "all visits"), table(dw, "within night")])
    tab.to_csv(run.path / "into_wind_fits.csv", index=False)
    lines = [f"{len(d)} science visits; wind-speed terciles {edges[0]:.2f}, {edges[1]:.2f}, "
             f"{edges[2]:.2f}, {edges[3]:.2f} m/s (weather tower)",
             "into = |pointing - wind| < 45 deg, cross 45-135 deg, down wind > 135 deg baseline.",
             "Huber q ~ into + cross + wind speed + airmass; then + 4 sonic axis-2 and 4 |axis-2|.",
             "Both on visits with every top-ring sonic (from 20260120). carried = 1 - "
             "with/without (dimensionless).", "",
             f"{'quantity':20s} {'mode':13s} {'wind':>5s} {'n':>6s} {'into - down':>16s} "
             f"{'+ axis 2':>16s} {'carried':>8s}"]
    for r in tab[tab.wind_bin.isin([2, "all"])].itertuples():
        lines.append(f"{r.quantity:20s} {r.mode:13s} {str(r.wind_bin):>5s} {r.n:6d} "
                     f"{r.into:+.4f} +/- {r.into_se:.4f} {r.into_with_axis2:+.4f} +/- "
                     f"{r.into_with_axis2_se:.4f} {r.carried_frac:+8.2f}")
    lines += ["", "Units: FWHM terms arcsec, Z4 um of wavefront, ellipticity dimensionless.",
              "wind = tercile index (2 = strongest) or all."]
    out = run.path / "wind_pointing.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.95, "Image quality and pointing into the wind", fontsize=15,
                 weight="bold", va="top")
        fig.text(0.05, 0.89, "\n".join(lines), fontsize=7, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        page_binned(pdf, d, edges, ["fwhm_z500", "fwhm", "donut_blur_fwhm", "aos_fwhm",
                                    "dev_z4"], "all visits")
        page_binned(pdf, d, edges, ["e1_azel_res", "e2_azel_res", "e_coh_res",
                                    "psf_e_azel_p50_res", "psf_e_azel_p90_res"], "all visits")
        page_binned(pdf, dw, edges, ["fwhm_z500", "fwhm", "donut_blur_fwhm", "aos_fwhm",
                                     "dev_z4"], "within night")
        page_coeffs(pdf, tab, edges)
        page_axis2(pdf, d, edges)
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
