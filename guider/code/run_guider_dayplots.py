#!/usr/bin/env python3
"""One multi-page per-day QA/summary PDF from a per-day moments parquet.

Pages: (1) validation QA vs seqNum (sensors tracked, median FWHM, Alt/Az
jitter, 2nd-order additivity residual); (2) detector counts (guiders per visit
+ per-guider presence); (3) static vs dynamic 2nd-order shape (Q1, Q2);
(4) static vs dynamic 3rd-order coma & trefoil.

    python run_guider_dayplots.py <night>/guider_moments_<dayObs>.parquet \
        --output <night>/guider_plots_<dayObs>.pdf
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Per-day guider_moments parquet.")
    p.add_argument("--output", required=True, help="Output PDF path.")
    return p.parse_args(argv)


def yx(ax, x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return
    lo = float(np.nanmin([x[m].min(), y[m].min()]))
    hi = float(np.nanmax([x[m].max(), y[m].max()]))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")


def page_validation(pdf, df, dayObs):
    g = df.groupby("seqNum")
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(g.size().index, g.size().values, ".", ms=4)
    ax[0, 0].set(title="sensors tracked / exposure", xlabel="seqNum", ylabel="n")
    fw = g["fwhm_med"].median()
    ax[0, 1].plot(fw.index, fw.values, ".", ms=4, color="C1")
    ax[0, 1].set(title="median FWHM", xlabel="seqNum", ylabel="arcsec")
    jt = g["jitter_altaz"].median()
    ax[1, 0].plot(jt.index, jt.values, ".", ms=4, color="C2")
    ax[1, 0].set(title="Alt/Az jitter", xlabel="seqNum", ylabel="arcsec")
    frac = (df["resid_T"] / df["T_coadd"]).replace([np.inf, -np.inf], np.nan).dropna()
    lim = np.nanpercentile(np.abs(frac), 99) if len(frac) else 1.0
    ax[1, 1].hist(frac, bins=np.linspace(-lim, lim, 41), color="C3", alpha=0.8)
    ax[1, 1].set(title="2nd-order additivity residual (T)", xlabel="resid/coadd")
    fig.suptitle(f"Guider validation  {dayObs}  ({df.expId.nunique()} exp, {len(df)} rows)",
                 fontweight="bold")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_detector_counts(pdf, df, dayObs):
    """Guiders-with-moments per visit (recovery) + per-guider presence."""
    perVisit = df.groupby("expId")["detector"].nunique()
    nVisits = len(perVisit)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # (a) distribution of #guiders per visit
    counts = perVisit.value_counts().reindex(range(1, 9), fill_value=0)
    ax[0].bar(counts.index, counts.values, color="C0")
    ax[0].set(title=f"guiders with moments / visit  (mean {perVisit.mean():.2f})",
              xlabel="# guiders", ylabel="# visits", xticks=range(1, 9))
    for n, c in counts.items():
        if c:
            ax[0].text(n, c, str(int(c)), ha="center", va="bottom", fontsize=8)

    # (b) per-guider presence fraction across visits (spot chronic dropouts)
    pres = (df.groupby("detector")["expId"].nunique() / nVisits).sort_index()
    ax[1].barh(pres.index, pres.values, color="C2")
    ax[1].set(title="fraction of visits each guider is present",
              xlabel="fraction", xlim=(0, 1.02))
    ax[1].axvline(1.0, color="k", lw=0.8, ls="--")
    for i, (name, v) in enumerate(pres.items()):
        ax[1].text(v, i, f" {v:.2f}", va="center", fontsize=8)

    partial = int((perVisit < 8).sum())
    fig.suptitle(f"Guider detector counts  {dayObs}  "
                 f"({nVisits} visits, {partial} partial <8)", fontweight="bold")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_static_dynamic(pdf, df, pairs, title, dayObs):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5))
    axes = np.atleast_1d(axes)
    for ax, (xs, ys, lab) in zip(axes, pairs):
        if xs not in df or ys not in df:
            ax.set_visible(False); continue
        x, y = df[xs].to_numpy(), df[ys].to_numpy()
        ax.scatter(x, y, s=8, alpha=0.3, color="C4")
        yx(ax, x, y)
        ax.set(xlabel=f"{lab} static", ylabel=f"{lab} dynamic"); ax.grid(alpha=0.3)
    fig.suptitle(f"{title}  {dayObs}", fontweight="bold")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main(argv=None):
    args = parseArgs(argv)
    df = pd.read_parquet(args.input)
    if df.empty:
        raise SystemExit(f"{args.input} is empty; nothing to plot.")
    dayObs = int(df["dayObs"].iloc[0]) if "dayObs" in df else "?"

    with PdfPages(args.output) as pdf:
        page_validation(pdf, df, dayObs)
        page_detector_counts(pdf, df, dayObs)
        page_static_dynamic(pdf, df,
                            [("Q1_stamp", "Q1_motion", "Q1"), ("Q2_stamp", "Q2_motion", "Q2")],
                            "2nd-order static vs dynamic", dayObs)
        page_static_dynamic(pdf, df,
                            [("coma1_static", "coma1_dyn", "coma1"),
                             ("coma2_static", "coma2_dyn", "coma2"),
                             ("tref1_static", "tref1_dyn", "tref1"),
                             ("tref2_static", "tref2_dyn", "tref2")],
                            "3rd-order static vs dynamic", dayObs)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
