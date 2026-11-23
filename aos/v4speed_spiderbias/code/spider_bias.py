#!/usr/bin/env python
"""Wavefront bias from omitting the spider in the Danish model.

Compares two ``sim_donut_fits`` variants of the same noise-free ``sim_donuts`` stamps,
dense Noll set 4-19, 22-26, binning 2:

- ``v4_speed_dense_prod``: no spider, ts_wep production tolerances (the current default);
- ``v4_speed_dense``: spider modelled at 0 deg, tight tolerances.

Both variants share the intrinsic, so the difference in Zernike deviation is the
difference in recovered wavefront (OPD), ``delta_j = OPD_j(no spider) - OPD_j(spider)``,
µm. Even part ``(extra + intra) / 2`` is what an extra/intra average (the MIW) keeps;
odd part ``(extra - intra) / 2`` cancels in it. FAM and giant have the same field points
on both sides; CWFS sides are different half-chips, so CWFS is summarised per CCD
(median over points inside 1.725 deg) and per corner (mean of its two CCDs).

Usage::

    python aos/v4speed_spiderbias/code/spider_bias.py
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
from rubinwork.products import sim_donut_fits

NOSPIDER, SPIDER = "v4_speed_dense_prod", "v4_speed_dense"
KEYS = ["sim_variant", "band", "config", "side", "point_id"]
FIELD_LIMIT_DEG = 1.725
PAIRS = [(4, 11), (5, 13), (6, 12), (7, 17), (8, 16)]
CORNERS = ("R00", "R04", "R40", "R44")


def load():
    """Per-donut table of both variants' deviations and their difference (µm)."""
    a, b = sim_donut_fits.load(NOSPIDER), sim_donut_fits.load(SPIDER)
    noll = sorted({int(c[2:-7]) for c in a.columns
                   if c.startswith("dz") and c.endswith("_CCS_um") and "true" not in c})
    cols = sim_donut_fits.dz_columns(noll, "CCS")
    keep = KEYS + ["thx_CCS_deg", "thy_CCS_deg", "field_r_deg"]
    m = a[keep + cols + ["l1"]].merge(b[KEYS + cols + ["l1"]], on=KEYS,
                                      suffixes=("_ns", "_sp"))
    for j, c in zip(noll, cols):
        m[f"d{j}"] = m[f"{c}_ns"] - m[f"{c}_sp"]
        m[f"ns{j}"], m[f"sp{j}"] = m[f"{c}_ns"], m[f"{c}_sp"]
    return m, noll


def even_odd(m, config, prefix):
    """Per-point even and odd parts of ``<prefix><j>`` for FAM or giant."""
    g = m[m.config == config]
    cols = [c for c in g.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    e = g[g.side == "extra"].set_index(["sim_variant", "point_id"])
    i = g[g.side == "intra"].set_index(["sim_variant", "point_id"])
    e, i = e.align(i, join="inner", axis=0)
    out = e[["thx_CCS_deg", "thy_CCS_deg", "field_r_deg"]].copy()
    for c in cols:
        out[f"even{c[len(prefix):]}"] = (e[c] + i[c]) / 2
        out[f"odd{c[len(prefix):]}"] = (e[c] - i[c]) / 2
    return out


def cwfs_tables(m, noll, prefix):
    """Per-CCD medians (points inside 1.725 deg), per-corner and 4-corner means (µm)."""
    g = m[(m.config == "cwfs") & (m.field_r_deg <= FIELD_LIMIT_DEG)].copy()
    g["ccd"] = g.point_id.str[:7]
    cols = [f"{prefix}{j}" for j in noll]
    ccd = g.groupby("ccd")[cols].median()
    ccd.columns = noll
    corner = pd.DataFrame({c: ccd.loc[[f"{c}_SW0", f"{c}_SW1"]].mean() for c in CORNERS}).T
    return ccd, corner, corner.mean()


def rms(x):
    return float(np.sqrt(np.nanmean(np.square(x))))


def summary_page(pdf, m, noll, eo):
    lines = [f"sim_donut_fits {NOSPIDER} (no spider, production tolerances) minus "
             f"{SPIDER} (spider at 0 deg, tight tolerances)",
             "delta = OPD(no spider) - OPD(spider), um of wavefront, CCS; same stamps, "
             "noise-free, 1e6 e- per donut; field r <= 1.725 deg", ""]
    lines.append("RMS over field points of delta, um: per side / even (extra+intra)/2 / "
                 "odd (extra-intra)/2")
    hdr = "Noll  " + "".join(f"{c:>26s}" for c in ("fam", "giant"))
    lines.append(hdr)
    for j in noll:
        row = f"Z{j:<4d}"
        for c in ("fam", "giant"):
            g = m[(m.config == c) & (m.field_r_deg <= FIELD_LIMIT_DEG)]
            t = eo[c][eo[c].field_r_deg <= FIELD_LIMIT_DEG]
            row += f"   {rms(g[f'd{j}']):6.3f} / {rms(t[f'even{j}']):6.3f} / " \
                   f"{rms(t[f'odd{j}']):6.3f}"
        lines.append(row)
    _, corner, mean4 = cwfs_tables(m, noll, "d")
    lines += ["", "CWFS 4-corner mean of the corner (extra+intra) means of per-CCD medians, "
              "delta, um:",
              "  " + "  ".join(f"Z{j} {mean4[j]:+.3f}" for j in noll)]
    lines += ["", "l1 (fraction of flux misplaced, dimensionless), median inside 1.725 deg: "
              + ", ".join(f"{c} {m[(m.config == c) & (m.field_r_deg <= FIELD_LIMIT_DEG)].l1_ns.median():.4f}"
                          f" no spider / {m[(m.config == c) & (m.field_r_deg <= FIELD_LIMIT_DEG)].l1_sp.median():.4f}"
                          " spider" for c in ("fam", "cwfs", "giant"))]
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", size=6.5)
    pdf.savefig(fig)
    plt.close(fig)


def rms_bars(pdf, m, noll, eo):
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    x = np.arange(len(noll))
    for ax, c in zip(axs, ("fam", "cwfs", "giant")):
        g = m[(m.config == c) & (m.field_r_deg <= FIELD_LIMIT_DEG)]
        sets = [(f"{s} side", [rms(g[g.side == s][f"d{j}"]) for j in noll])
                for s in ("extra", "intra")]
        if c in eo:
            t = eo[c][eo[c].field_r_deg <= FIELD_LIMIT_DEG]
            sets.append(("even (extra+intra)/2", [rms(t[f"even{j}"]) for j in noll]))
        else:
            _, corner, _ = cwfs_tables(m, noll, "d")
            sets.append(("corner mean of CCD medians", [rms(corner[j]) for j in noll]))
        w = 0.8 / len(sets)
        for k, (lab, v) in enumerate(sets):
            ax.bar(x + (k - (len(sets) - 1) / 2) * w, v, w, label=lab)
        ax.set_ylabel("RMS delta (um)", size=8)
        ax.set_title(f"{c}: RMS over field (r <= 1.725 deg) of OPD(no spider) - "
                     "OPD(spider)", size=9)
        ax.legend(fontsize=7)
    axs[-1].set_xticks(x, [str(j) for j in noll])
    axs[-1].set_xlabel("Noll index")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def field_maps(pdf, t, noll, config, part, label):
    fig, axs = plt.subplots(4, 6, figsize=(11, 8.5))
    for ax, j in zip(axs.flat, noll):
        v = t[f"{part}{j}"]
        lim = max(np.abs(v).max(), 1e-3)
        sc = ax.scatter(t.thx_CCS_deg, t.thy_CCS_deg, c=v, s=4, cmap="RdBu_r",
                        vmin=-lim, vmax=lim)
        ax.set_aspect("equal")
        ax.set_title(f"Z{j}: RMS {rms(v):.3f} um", size=7)
        ax.tick_params(labelsize=5)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)
    for ax in axs.flat[len(noll):]:
        ax.axis("off")
    fig.suptitle(f"{config}: {label} of OPD(no spider) - OPD(spider), um, against field "
                 "angle CCS x, y (deg)", size=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def cwfs_page(pdf, m, noll):
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    x = np.arange(len(noll))
    for ax, (pref, lab) in zip(axs, (("ns", "no spider: OPD - intrinsic"),
                                     ("sp", "spider: OPD - intrinsic"),
                                     ("d", "OPD(no spider) - OPD(spider)"))):
        ccd, corner, mean4 = cwfs_tables(m, noll, pref)
        for k, c in enumerate(CORNERS):
            ax.plot(x + (k - 1.5) * 0.12, ccd.loc[f"{c}_SW0"], "^", ms=3, color=f"C{k}")
            ax.plot(x + (k - 1.5) * 0.12, ccd.loc[f"{c}_SW1"], "v", ms=3, color=f"C{k}")
            ax.plot(x + (k - 1.5) * 0.12, corner.loc[c], "o", ms=5, mfc="none",
                    color=f"C{k}", label=f"{c}: SW0 extra ^, SW1 intra v, mean o")
        ax.bar(x, mean4, 0.6, color="0.8", zorder=0, label="4-corner mean")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel("um", size=8)
        ax.set_title(f"CWFS, per-CCD median inside 1.725 deg: {lab}", size=9)
        ax.legend(fontsize=6, ncol=5)
    axs[-1].set_xticks(x, [str(j) for j in noll])
    axs[-1].set_xlabel("Noll index")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pair_page(pdf, m, eo):
    fig, axs = plt.subplots(3, len(PAIRS), figsize=(11, 7.5))
    sets = [("fam even", eo["fam"], "even"), ("fam odd", eo["fam"], "odd"),
            ("giant even", eo["giant"], "even")]
    for row, (lab, t, part) in zip(axs, sets):
        t = t[t.field_r_deg <= FIELD_LIMIT_DEG]
        for ax, (a, b) in zip(row, PAIRS):
            x, y = t[f"{part}{a}"], t[f"{part}{b}"]
            ax.plot(x, y, ".", ms=2)
            r = np.corrcoef(x, y)[0, 1]
            rho = scipy.stats.spearmanr(x, y)[0]
            ax.set_title(f"{lab}: Pearson r {r:+.2f}, Spearman rho {rho:+.2f}, "
                         f"n {len(t)}", size=6)
            ax.set_xlabel(f"delta Z{a} (um)", size=7)
            ax.set_ylabel(f"delta Z{b} (um)", size=7)
            ax.tick_params(labelsize=5)
    fig.suptitle("1st against 2nd aberration of OPD(no spider) - OPD(spider), field r <= "
                 "1.725 deg", size=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="spider_bias")
    a = ap.parse_args()
    run = study_run("aos/v4speed_spiderbias", a.label)
    m, noll = load()
    eo = {c: even_odd(m, c, "d") for c in ("fam", "giant")}
    with PdfPages(run.path / "spider_bias.pdf") as pdf:
        summary_page(pdf, m, noll, eo)
        rms_bars(pdf, m, noll, eo)
        cwfs_page(pdf, m, noll)
        pair_page(pdf, m, eo)
        for c in ("fam", "giant"):
            for part, lab in (("even", "even part (extra+intra)/2"),
                              ("odd", "odd part (extra-intra)/2")):
                field_maps(pdf, eo[c], noll, c, part, lab)
    m.to_parquet(run.path / "spider_bias.parquet")
    ccd, corner, mean4 = cwfs_tables(m, noll, "d")
    print("CWFS 4-corner mean delta (um):", mean4.round(3).to_dict())
    print(f"wrote {run.path / 'spider_bias.pdf'}")


if __name__ == "__main__":
    main()
