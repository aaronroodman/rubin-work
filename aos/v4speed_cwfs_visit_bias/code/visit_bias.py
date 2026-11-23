#!/usr/bin/env python
"""CWFS wavefront bias over simulated visits, as the MIW combines the corners.

Reads ``sim_donut_visit_fits`` variants (Danish fits of ``sim_donut_visits``: 8 donuts
per corner half-chip inside 1.725 deg, perfectly aligned optics). Per visit and CCD the
median OPD over its donuts; per corner the mean of its extra (SW0) and intra (SW1) CCD;
the 4-corner mean of those. Each is averaged over visits and compared with the same
statistic of the batoid intrinsic at the same donuts. Bias = OPD statistic minus
intrinsic statistic (µm of wavefront, CCS; rotator 0 deg so CCS = OCS). Errors are the
standard error over visits.

Usage::

    python aos/v4speed_cwfs_visit_bias/code/visit_bias.py
    python aos/v4speed_cwfs_visit_bias/code/visit_bias.py --variants v4_speed_cwfs8_vk_dense_prod
"""
import argparse

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rubinwork.common import study_run
from rubinwork.products import catalog, sim_donut_visit_fits

VARIANTS = ("v4_speed_cwfs8_vk_dense_prod", "v4_speed_cwfs8_vk_dense_prod_nonoise",
            "v4_speed_cwfs8_vk_blur35_dense_prod",
            "v4_speed_cwfs8_vk_blur35_dense_prod_nonoise")
CORNERS = ("R00", "R04", "R40", "R44")
PAIRS = [(4, 11), (5, 13), (6, 12), (7, 17), (8, 16)]
SHOW = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17]


def noll_of(df):
    return sorted(int(c[2:-7]) for c in df.columns
                  if c.startswith("dz") and c.endswith("_CCS_um"))


def per_visit(df, noll):
    """Per visit: CCD medians, corner means and 4-corner mean of OPD, intrinsic and
    their difference (µm). Returns ``{level: DataFrame indexed by (visit, unit)}``."""
    d = df.copy()
    for j in noll:
        d[f"opd{j}"] = d[f"z{j}_intr_CCS_um"] + d[f"dz{j}_CCS_um"]
        d[f"intr{j}"] = d[f"z{j}_intr_CCS_um"]
    cols = [f"{p}{j}" for p in ("opd", "intr") for j in noll]
    ccd = d.groupby(["visit", "ccd"])[cols].median()
    for j in noll:
        ccd[f"bias{j}"] = ccd[f"opd{j}"] - ccd[f"intr{j}"]
    c = ccd.reset_index()
    c["corner"] = c.ccd.str[:3]
    corner = c.groupby(["visit", "corner"])[list(ccd.columns)].mean()
    four = corner.groupby("visit").mean()
    return {"ccd": ccd, "corner": corner, "four": four}


def mean_se(t, level, noll):
    """Mean over visits and its standard error of the bias, per unit (µm)."""
    b = t[[f"bias{j}" for j in noll]]
    b.columns = noll
    if level == "four":
        return pd.DataFrame([b.mean()], index=["4-corner"]), \
            pd.DataFrame([b.std() / np.sqrt(len(b))], index=["4-corner"])
    g = b.groupby(level=1)
    return g.mean(), g.std() / np.sqrt(g.size().to_numpy()[:, None])


def summary_page(pdf, results):
    lines = ["CWFS bias over visits: OPD statistic minus intrinsic statistic at the same "
             "donuts (um of wavefront, CCS),",
             "mean over visits +- standard error; CCD = median over its 8 donuts; corner = "
             "mean of extra SW0 and intra SW1; 4-corner = mean of corners", ""]
    for v, (df, noll, pv) in results.items():
        m4, s4 = mean_se(pv["four"], "four", noll)
        mc, _ = mean_se(pv["corner"], "corner", noll)
        mcc, _ = mean_se(pv["ccd"], "ccd", noll)
        lines.append(f"{v}: {df.visit.nunique()} visits, {len(df)} donuts; fit success "
                     f"{df.success.mean():.3f} (fraction), nfev at max "
                     f"{(df.nfev >= 30).mean():.3f} (fraction), median FWHM "
                     f"{df.fwhm_arcsec.median():.3f} arcsec, median chi2/dof "
                     f"{df.chi2_dof.median():.3f} (dof = binned pixels - parameters)")
        lines.append("  Noll  4-corner mean +- SE   corners R00 R04 R40 R44         "
                     "RMS over CCDs   RMS over corners")
        for j in noll:
            lines.append(f"  Z{j:<4d} {m4[j].iloc[0]:+.4f} +- {s4[j].iloc[0]:.4f}   "
                         + " ".join(f"{mc.loc[c, j]:+.4f}" for c in CORNERS)
                         + f"   {np.sqrt((mcc[j] ** 2).mean()):.4f}"
                         f"         {np.sqrt((mc[j] ** 2).mean()):.4f}")
        lines.append("")
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", size=5.6)
    pdf.savefig(fig)
    plt.close(fig)


def bias_page(pdf, v, noll, pv):
    fig, axs = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    x = np.arange(len(noll))
    mcc, scc = mean_se(pv["ccd"], "ccd", noll)
    mc, sc = mean_se(pv["corner"], "corner", noll)
    m4, s4 = mean_se(pv["four"], "four", noll)
    for k, c in enumerate(CORNERS):
        for half, mk in (("SW0", "^"), ("SW1", "v")):
            u = f"{c}_{half}"
            axs[0].errorbar(x + (k - 1.5) * 0.12, mcc.loc[u], scc.loc[u], fmt=mk, ms=3,
                            lw=0.6, color=f"C{k}",
                            label=f"{u} ({'extra' if half == 'SW0' else 'intra'})")
        axs[1].errorbar(x + (k - 1.5) * 0.12, mc.loc[c], sc.loc[c], fmt="o", ms=4, lw=0.6,
                        color=f"C{k}", label=f"{c} corner mean")
    axs[1].bar(x, m4.iloc[0], 0.6, yerr=s4.iloc[0], color="0.8", zorder=0,
               label="4-corner mean")
    axs[0].set_title(f"{v}: per-CCD median OPD minus median intrinsic, mean over visits "
                     "+- SE", size=9)
    axs[1].set_title("corner mean (extra+intra)/2 and 4-corner mean", size=9)
    for ax in axs:
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel("bias (um of wavefront, CCS)", size=8)
        ax.legend(fontsize=6, ncol=4)
    axs[1].set_xticks(x, [str(j) for j in noll])
    axs[1].set_xlabel("Noll index")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pair_page(pdf, v, noll, pv, df):
    fig, axs = plt.subplots(3, len(PAIRS), figsize=(11, 7.5))
    mcc, _ = mean_se(pv["ccd"], "ccd", noll)
    corner = pv["corner"]
    for col, (a, b) in enumerate(PAIRS):
        ax = axs[0, col]
        for k, c in enumerate(CORNERS):
            for half, mk in (("SW0", "^"), ("SW1", "v")):
                u = f"{c}_{half}"
                ax.plot(mcc.loc[u, a], mcc.loc[u, b], mk, color=f"C{k}", ms=5)
        ax.set_title(f"CCD mean bias, Z{b} vs Z{a}", size=7)
        ax = axs[1, col]
        x, y = corner[f"bias{a}"], corner[f"bias{b}"]
        ax.plot(x, y, ".", ms=2)
        ax.set_title(f"corner mean per visit: r {np.corrcoef(x, y)[0, 1]:+.2f}, "
                     f"rho {scipy.stats.spearmanr(x, y)[0]:+.2f}, n {len(x)}", size=6)
        ax = axs[2, col]
        x, y = df[f"dz{a}_CCS_um"], df[f"dz{b}_CCS_um"]
        ax.plot(x, y, ".", ms=1, alpha=0.3)
        ax.set_title(f"per donut: r {np.corrcoef(x, y)[0, 1]:+.2f}, "
                     f"rho {scipy.stats.spearmanr(x, y)[0]:+.2f}, n {len(x)}", size=6)
        for r in axs[:, col]:
            r.set_xlabel(f"Z{a} (um)", size=6)
            r.set_ylabel(f"Z{b} (um)", size=6)
            r.tick_params(labelsize=5)
            r.axhline(0, color="k", lw=0.3)
            r.axvline(0, color="k", lw=0.3)
    fig.suptitle(f"{v}: 1st against 2nd aberration bias (OPD - intrinsic); top: CCD means "
                 "over visits (^ extra, v intra; colour = corner)", size=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def dependence_page(pdf, v, df):
    """Per-donut deviation against magnitude and against the seeing major-axis angle."""
    fig, axs = plt.subplots(2, 6, figsize=(11, 6))
    bins = np.linspace(df.mag.min(), df.mag.max(), 9)
    two_b = np.deg2rad(2 * df.seeing_beta_deg)
    for col, j in enumerate((5, 13, 6, 12, 7, 17)):
        c = f"dz{j}_CCS_um"
        ax = axs[0, col]
        for s, colour in (("extra", "C0"), ("intra", "C1")):
            g = df[df.side == s]
            idx = np.digitize(g.mag, bins)
            med = [g[c][idx == i].median() for i in range(1, len(bins))]
            ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "o-", ms=3, color=colour, label=s)
        ax.set_xlabel("star r mag (AB)", size=7)
        ax.set_ylabel(f"median dz{j} (um)", size=7)
        ax.legend(fontsize=6)
        ax = axs[1, col]
        X = np.c_[np.ones(len(df)), np.cos(two_b), np.sin(two_b)]
        co = np.linalg.lstsq(X, df[c], rcond=None)[0]
        ax.plot(df.seeing_beta_deg, df[c], ".", ms=1, alpha=0.3)
        bb = np.linspace(0, 180, 181)
        ax.plot(bb, co[0] + co[1] * np.cos(np.deg2rad(2 * bb)) + co[2] * np.sin(np.deg2rad(2 * bb)),
                "r-")
        ax.set_title(f"cos2b {co[1]:+.4f}, sin2b {co[2]:+.4f} um", size=6)
        ax.set_xlabel("seeing major axis (deg, CCS)", size=7)
        ax.set_ylabel(f"dz{j} (um)", size=7)
        for a in axs[:, col]:
            a.tick_params(labelsize=5)
    fig.suptitle(f"{v}: per-donut Zernike deviation against star magnitude (median per "
                 "bin) and against the seeing kernel angle", size=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--label", default="visit_bias")
    a = ap.parse_args()
    variants = a.variants or [v for v in VARIANTS if _has_current(v)]
    run = study_run("aos/v4speed_cwfs_visit_bias", a.label)
    results = {}
    for v in variants:
        df = sim_donut_visit_fits.load(v)
        noll = noll_of(df)
        results[v] = (df, noll, per_visit(df, noll))
    with PdfPages(run.path / "visit_bias.pdf") as pdf:
        summary_page(pdf, results)
        for v, (df, noll, pv) in results.items():
            bias_page(pdf, v, noll, pv)
            pair_page(pdf, v, noll, pv, df)
            dependence_page(pdf, v, df)
    rows = []
    for v, (df, noll, pv) in results.items():
        for level in ("ccd", "corner", "four"):
            m, s = mean_se(pv[level], level, noll)
            for unit in m.index:
                rows.append({"variant": v, "level": level, "unit": unit,
                             **{f"bias{j}_um": m.loc[unit, j] for j in noll},
                             **{f"se{j}_um": s.loc[unit, j] for j in noll}})
    pd.DataFrame(rows).to_parquet(run.path / "visit_bias.parquet")
    print(f"wrote {run.path / 'visit_bias.pdf'}")


def _has_current(variant):
    try:
        catalog.manifest("sim_donut_visit_fits", variant)
        return True
    except catalog.ProductNotFound:
        return False


if __name__ == "__main__":
    main()
