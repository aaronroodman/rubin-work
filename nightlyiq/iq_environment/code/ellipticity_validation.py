"""Validation pages for the value_added `psf_shape` ellipticity columns.

Checks coverage, internal consistency (percentile ordering, mean |e| against |mean e|,
the two-point functions against the coherent ellipticity), dependence on rotator angle
and azimuth, and the night-by-night trend. One PDF in the run directory.

Usage::

    python nightlyiq/iq_environment/code/ellipticity_validation.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy import stats

from rubinwork.common import study_run
from rubinwork.common.utils import nmad
from rubinwork.products.value_added import reader

VA_BUILD = "snapshots/20261105"
# Fewer CCDs than this means a partial readout; full focal plane is 189 science CCDs.
MIN_CCD = 150
PCTS = ("p10", "p25", "p50", "p75", "p90")
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "black": "#000000"}


def load():
    """Science-like visits with a psf_shape summary, plus derived quantities."""
    con = reader.open_db(reader.catalog_db_path("v1", VA_BUILD))
    try:
        cols = [r[0] for r in con.execute("DESCRIBE visit_telemetry").fetchall()]
        psf = [c for c in cols if c.startswith("psf_")]
        d = con.execute(f"SELECT visit_id, day_obs, seq_num, obs_start, azimuth_deg, "
                        f"{', '.join(psf)} FROM visit_telemetry "
                        f"WHERE psf_n_ccd IS NOT NULL").df()
    finally:
        con.close()
    n_all = len(d)
    # LUT compensation was off before seq 398 on 20260424 (image_quality_trending cut).
    d = d[~((d.day_obs == 20260424) & (d.seq_num < 398))]
    d = d[d.psf_n_ccd >= MIN_CCD].copy()
    d["e_coh"] = np.hypot(d.psf_e1_azel_mean, d.psf_e2_azel_mean)
    d["coh_frac"] = d.e_coh / d.psf_e_azel_mean
    for r in ("r2", "r6"):
        d[f"xi_sum_{r}"] = d[f"psf_xi_e1e1_{r}"] + d[f"psf_xi_e2e2_{r}"]
    d["date"] = pd.to_datetime(d.day_obs.astype(str), format="%Y%m%d")
    return d, n_all


def corr(x, y):
    """Pearson r and Spearman rho over finite pairs, with n."""
    m = np.isfinite(x) & np.isfinite(y)
    return stats.pearsonr(x[m], y[m])[0], stats.spearmanr(x[m], y[m])[0], int(m.sum())


def hexbin(ax, x, y, xlabel, ylabel, extent=None, diag=False):
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    n_out = 0
    if extent is not None:
        inside = (x >= extent[0]) & (x <= extent[1]) & (y >= extent[2]) & (y <= extent[3])
        n_out = int((m & ~inside).sum())
        m &= inside
    hb = ax.hexbin(x[m], y[m], gridsize=80, norm=LogNorm(), cmap="viridis",
                   mincnt=1, extent=extent)
    if diag:
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color=OI["verm"], lw=1, ls="--", label="y = x")
        ax.legend(loc="upper left")
    r, rho, n = corr(x, y)
    ax.set_title(f"Pearson r {r:+.3f}, Spearman rho {rho:+.3f}, n {n} "
                 f"({n_out} outside the axes)", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(hb, ax=ax, label="visits per cell")


def binned(ax, x, y, bins, color, label):
    """Median and 25-75 percentile band of y in bins of x."""
    idx = np.digitize(x, bins)
    c = 0.5 * (bins[1:] + bins[:-1])
    med = np.array([np.nanmedian(y[idx == i]) if (idx == i).sum() > 20 else np.nan
                    for i in range(1, len(bins))])
    q25 = np.array([np.nanpercentile(y[idx == i], 25) if (idx == i).sum() > 20 else np.nan
                    for i in range(1, len(bins))])
    q75 = np.array([np.nanpercentile(y[idx == i], 75) if (idx == i).sum() > 20 else np.nan
                    for i in range(1, len(bins))])
    ax.fill_between(c, q25, q75, color=color, alpha=0.2, lw=0)
    ax.plot(c, med, color=color, lw=2, label=label)


def checks(d):
    """Internal consistency counts, as text lines."""
    out = []
    for q in ("e", "e1", "e2"):
        v = d[[f"psf_{q}_azel_{p}" for p in PCTS]].to_numpy()
        ok = np.isfinite(v).all(axis=1)
        bad = (np.diff(v[ok], axis=1) < 0).any(axis=1).sum()
        out.append(f"{q} percentiles out of order (p10 <= ... <= p90): {bad} of {ok.sum()} visits")
    m = d.coh_frac.notna()
    out.append(f"|mean e| > mean |e| (impossible): {(d.coh_frac[m] > 1 + 1e-9).sum()} "
               f"of {m.sum()} visits")
    for r in ("r2", "r6"):
        x = d[f"xi_sum_{r}"]
        out.append(f"xi_e1e1 + xi_e2e2 at {r} < 0: {(x < 0).sum()} of {x.notna().sum()} visits "
                   f"(fraction {(x < 0).mean():.4f}, dimensionless)")
    out.append(f"psf_e_azel_mean NaN with psf_n_ccd >= {MIN_CCD}: "
               f"{d.psf_e_azel_mean.isna().sum()} visits")
    return out


def page_summary(pdf, d, n_all, lines):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, "value_added psf_shape: ellipticity validation",
             fontsize=16, weight="bold", va="top")
    s = [f"value_added v1@{VA_BUILD}; visits with psf_shape: {n_all}",
         f"Kept: psf_n_ccd >= {MIN_CCD} (of 189 science CCDs), 20260424 seq < 398 dropped:",
         f"    {len(d)} visits, {d.day_obs.nunique()} nights, day_obs {d.day_obs.min()} to "
         f"{d.day_obs.max()}",
         "No science-program cut yet (needs ConsDB observation_reason); engineering visits "
         "are included.",
         "Frame: RubinTV Az/El (+x increasing azimuth, +y increasing elevation). "
         "e1_OCS = e1_AzEl, e2_OCS = -e2_AzEl.",
         "",
         "Medians over visits (all dimensionless):"]
    for c, lab in [("psf_e_azel_p10", "|e| p10 over CCDs"), ("psf_e_azel_p50", "|e| p50 over CCDs"),
                   ("psf_e_azel_p90", "|e| p90 over CCDs"), ("psf_e_azel_mean", "mean |e| over CCDs"),
                   ("e_coh", "|mean e| (coherent)"), ("coh_frac", "|mean e| / mean |e|"),
                   ("xi_sum_r2", "xi_e1e1 + xi_e2e2, 0.35-0.59 deg"),
                   ("xi_sum_r6", "xi_e1e1 + xi_e2e2, 1.29-1.52 deg")]:
        s.append(f"    {lab}: median {d[c].median():.4f}, nMAD {nmad(d[c].dropna()):.4f}")
    s += ["", "Consistency checks:"] + ["    " + x for x in lines]
    fig.text(0.05, 0.88, "\n".join(s), fontsize=10, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def page_coverage(pdf, d_all):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].hist(d_all.psf_n_ccd, bins=np.arange(-0.5, 190.5, 1), color=OI["blue"])
    ax[0].axvline(MIN_CCD - 0.5, color=OI["verm"], ls="--", label=f"cut {MIN_CCD}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("psf_n_ccd (science CCDs with a PSF)")
    ax[0].set_ylabel("visits")
    ax[0].legend()
    n = d_all.groupby("date").size()
    ax[1].plot(n.index, n.values, ".", color=OI["blue"], ms=4)
    ax[1].set_xlabel("day_obs")
    ax[1].set_ylabel("visits with psf_shape per night")
    fig.suptitle("Coverage")
    fig.autofmt_xdate()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_magnitude(pdf, d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, 0.25, 101)
    for p, c in zip(("p10", "p50", "p90"), (OI["sky"], OI["blue"], OI["black"])):
        ax[0].hist(d[f"psf_e_azel_{p}"], bins=bins, histtype="step", lw=2, color=c,
                   label=f"|e| {p} over CCDs")
    ax[0].hist(d.e_coh, bins=bins, histtype="step", lw=2, color=OI["orange"],
               label="|mean e| (coherent)")
    ax[0].set_xlabel("ellipticity (dimensionless)")
    ax[0].set_ylabel("visits")
    ax[0].legend()
    hexbin(ax[1], d.psf_e_azel_p50, d.psf_e_azel_mean, "|e| p50 over CCDs",
           "mean |e| over CCDs", extent=(0, 0.2, 0, 0.2), diag=True)
    fig.suptitle("Ellipticity magnitude per visit (dimensionless)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_spread(pdf, d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    hexbin(ax[0], d.psf_e_azel_p50, d.psf_e_azel_p10, "|e| p50 over CCDs",
           "|e| p10 over CCDs", extent=(0, 0.2, 0, 0.2), diag=True)
    hexbin(ax[1], d.psf_e_azel_p50, d.psf_e_azel_p90, "|e| p50 over CCDs",
           "|e| p90 over CCDs", extent=(0, 0.2, 0, 0.3), diag=True)
    fig.suptitle("Low and high |e| against the median (dimensionless)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_coherence(pdf, d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    hexbin(ax[0], d.psf_e_azel_mean, d.e_coh, "mean |e| over CCDs",
           "|mean e| over CCDs (coherent)", extent=(0, 0.2, 0, 0.2), diag=True)
    ax[1].hist(d.coh_frac, bins=np.linspace(0, 1, 101), color=OI["blue"])
    ax[1].set_xlabel("|mean e| / mean |e| (dimensionless; coherent over total)")
    ax[1].set_ylabel("visits")
    fig.suptitle("How much of the ellipticity is field-coherent")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_components(pdf, d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    hexbin(ax[0], d.psf_e1_azel_mean, d.psf_e2_azel_mean, "mean e1, Az/El",
           "mean e2, Az/El", extent=(-0.12, 0.12, -0.12, 0.12))
    ax[0].set_aspect("equal")
    hexbin(ax[1], d.psf_e1_azel_mean, d.psf_e1_azel_p50, "mean e1, Az/El",
           "e1 p50 over CCDs, Az/El", extent=(-0.12, 0.12, -0.12, 0.12), diag=True)
    fig.suptitle("Ellipticity components per visit (dimensionless, Az/El frame)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_pointing(pdf, d):
    """A camera-fixed pattern shows as a 2*rot sinusoid in the Az/El frame."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    rbins = np.linspace(np.nanpercentile(d.psf_rot_deg, 0.5),
                        np.nanpercentile(d.psf_rot_deg, 99.5), 31)
    abins = np.linspace(0, 360, 37)
    for a, x, bins, xl in ((ax[0], d.psf_rot_deg, rbins, "psf_rot_deg (deg, rotation applied)"),
                           (ax[1], d.azimuth_deg % 360, abins, "TMA azimuth (deg)")):
        binned(a, x.to_numpy(), d.psf_e1_azel_mean.to_numpy(), bins, OI["blue"], "mean e1")
        binned(a, x.to_numpy(), d.psf_e2_azel_mean.to_numpy(), bins, OI["orange"], "mean e2")
        binned(a, x.to_numpy(), d.e_coh.to_numpy(), bins, OI["black"], "|mean e|")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel(xl)
        a.set_ylabel("ellipticity, Az/El (dimensionless)")
        a.legend()
    fig.suptitle("Coherent ellipticity against rotator angle and azimuth "
                 "(median, 25-75 percentile band)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_nightly(pdf, d):
    g = d.groupby("date")
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for c, col, lab in ((OI["sky"], "psf_e_azel_p10", "|e| p10"),
                        (OI["blue"], "psf_e_azel_p50", "|e| p50"),
                        (OI["black"], "psf_e_azel_p90", "|e| p90"),
                        (OI["orange"], "e_coh", "|mean e|")):
        ax[0].plot(g[col].median().index, g[col].median().values, ".", color=c, ms=4, label=lab)
    ax[0].set_ylabel("nightly median (dimensionless)")
    ax[0].legend(ncol=4)
    for c, col, lab in ((OI["blue"], "psf_e1_azel_mean", "mean e1"),
                        (OI["orange"], "psf_e2_azel_mean", "mean e2")):
        ax[1].plot(g[col].median().index, g[col].median().values, ".", color=c, ms=4, label=lab)
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_ylabel("nightly median, Az/El (dimensionless)")
    ax[1].legend(ncol=2)
    fig.suptitle("Night by night")
    fig.autofmt_xdate()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_xi(pdf, d):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    hexbin(ax[0, 0], d.xi_sum_r2, d.xi_sum_r6, "xi_e1e1 + xi_e2e2, 0.35-0.59 deg",
           "xi_e1e1 + xi_e2e2, 1.29-1.52 deg", extent=(-0.001, 0.01, -0.001, 0.01), diag=True)
    hexbin(ax[0, 1], d.e_coh ** 2, d.xi_sum_r6, "|mean e|^2", "xi_e1e1 + xi_e2e2, 1.29-1.52 deg",
           extent=(0, 0.01, -0.001, 0.01), diag=True)
    hexbin(ax[1, 0], d.psf_xi_e1e1_r2, d.psf_xi_e2e2_r2, "xi_e1e1, 0.35-0.59 deg",
           "xi_e2e2, 0.35-0.59 deg", extent=(-0.001, 0.008, -0.001, 0.008), diag=True)
    ax[1, 1].hist(d.psf_xi_e1e2_r2, bins=np.linspace(-0.004, 0.004, 101), histtype="step",
                  lw=2, color=OI["blue"], label="0.35-0.59 deg")
    ax[1, 1].hist(d.psf_xi_e1e2_r6, bins=np.linspace(-0.004, 0.004, 101), histtype="step",
                  lw=2, color=OI["orange"], label="1.29-1.52 deg")
    ax[1, 1].set_xlabel("xi_e1e2 (dimensionless)")
    ax[1, 1].set_ylabel("visits")
    ax[1, 1].legend()
    fig.suptitle("Raw two-point functions (not mean-subtracted; dimensionless)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_corr_matrix(pdf, d):
    cols = {"psf_e_azel_p10": "|e| p10", "psf_e_azel_p50": "|e| p50",
            "psf_e_azel_p90": "|e| p90", "psf_e_azel_mean": "mean |e|", "e_coh": "|mean e|",
            "coh_frac": "coh frac", "xi_sum_r2": "xi sum r2", "xi_sum_r6": "xi sum r6",
            "psf_e1_azel_mean": "mean e1", "psf_e2_azel_mean": "mean e2"}
    sub = d[list(cols)].dropna()
    rho = sub.corr(method="spearman").to_numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), list(cols.values()), rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), list(cols.values()))
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{rho[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(rho[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Spearman rho (dimensionless)")
    ax.set_title(f"Spearman rho among ellipticity summaries, n {len(sub)} visits")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    run = study_run("nightlyiq/iq_environment", "ellipticity_validation")
    d, n_all = load()
    con = reader.open_db(reader.catalog_db_path("v1", VA_BUILD))
    d_all = con.execute("SELECT day_obs, psf_n_ccd FROM visit_telemetry "
                        "WHERE psf_n_ccd IS NOT NULL").df()
    con.close()
    d_all["date"] = pd.to_datetime(d_all.day_obs.astype(str), format="%Y%m%d")
    lines = checks(d)
    out = run.path / "ellipticity_validation.pdf"
    with PdfPages(out) as pdf:
        page_summary(pdf, d, n_all, lines)
        page_coverage(pdf, d_all)
        page_magnitude(pdf, d)
        page_spread(pdf, d)
        page_coherence(pdf, d)
        page_components(pdf, d)
        page_pointing(pdf, d)
        page_nightly(pdf, d)
        page_xi(pdf, d)
        page_corr_matrix(pdf, d)
    print("\n".join(lines))
    print(f"{len(d)} visits; wrote {out}")


if __name__ == "__main__":
    main()
