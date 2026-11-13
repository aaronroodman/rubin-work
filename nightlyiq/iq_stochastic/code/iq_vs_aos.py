"""PSF FWHM and median ellipticity against AOS_FWHM, per band, as 2D histograms.

Science visits from ConsDB ``visit1_quicklook``: ``psf_sigma_median`` (pixels) and
``aos_fwhm`` (arcsec, the PSF FWHM the AOS wavefront residual alone would give). The
median ellipticity is ``psf_e_azel_p50`` from value_added, the median |e| over CCDs
(dimensionless); ConsDB ``psf_ellipticity_median`` is empty. Each band gets a
Huber robust linear fit. The ConsDB pull is cached in ``output/cache/``; ``--refetch``
replaces it.

Usage::

    python nightlyiq/iq_stochastic/code/iq_vs_aos.py [--refetch]
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import statsmodels.api as sm
from scipy import stats

from rubinwork.common import study_run
from rubinwork.common.telemetry_clients import make_consdb_client
from rubinwork.common.utils import nmad
from rubinwork.products.value_added import reader

STUDY = "nightlyiq/iq_stochastic"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "output" / "cache" / "consdb_iq.parquet"
SIG2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
PIXEL_SCALE = 0.2  # arcsec per pixel
VA_BUILD = "snapshots/20261105"
MIN_CCD = 150  # partial readouts bias the ellipticity summary
BANDS = ("u", "g", "r", "i", "z", "y")
ENG_REASON = (r"aos|stability|focus|check|position|align|manual|dark|bias|flat|^test$"
              r"|_test|block-|filter_change_close_loop")
# Ranges that drop failed measurements; both the fit and the histogram use them.
CUTS = {"aos_fwhm": (0.0, 2.0), "fwhm": (0.3, 3.0), "e_med": (0.0, 0.3)}
LABELS = {"aos_fwhm": "AOS_FWHM (arcsec)", "fwhm": "PSF FWHM (arcsec)",
          "e_med": "median ellipticity (dimensionless)"}
BINS = {"aos_fwhm": np.linspace(0.0, 1.5, 76), "fwhm": np.linspace(0.4, 2.8, 97),
        "e_med": np.linspace(0.0, 0.25, 101)}


def fetch():
    """Every LSSTCam science visit with an AOS_FWHM, one row per visit."""
    q = ("SELECT e.exposure_id AS visit_id, e.day_obs, e.seq_num, e.band, "
         "e.observation_reason, q.psf_sigma_median, q.aos_fwhm "
         "FROM cdb_lsstcam.exposure e "
         "JOIN cdb_lsstcam.visit1_quicklook q ON q.visit_id = e.exposure_id "
         "WHERE e.img_type = 'science' AND q.aos_fwhm IS NOT NULL")
    d = make_consdb_client("auto").query(q).to_pandas()
    for c in ("psf_sigma_median", "aos_fwhm"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load(refetch=False):
    if refetch or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        fetch().to_parquet(CACHE, index=False)
    d = pd.read_parquet(CACHE)
    con = reader.open_db(reader.catalog_db_path("v1", VA_BUILD))
    try:
        va = con.execute("SELECT visit_id, psf_e_azel_p50, psf_n_ccd FROM visit_telemetry").df()
    finally:
        con.close()
    va.loc[~(va.psf_n_ccd >= MIN_CCD), "psf_e_azel_p50"] = np.nan
    d = d.merge(va, on="visit_id", how="left")
    d = d[d.band.isin(BANDS)
          & ~d.observation_reason.str.contains(ENG_REASON, case=False, regex=True, na=False)]
    d = d.assign(fwhm=d.psf_sigma_median * SIG2FWHM * PIXEL_SCALE,
                 e_med=d.psf_e_azel_p50)
    return d


def fit(x, y):
    """Huber fit of y on x, with correlations and the robust residual scatter."""
    rlm = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    b0, b1 = rlm.params
    e0, e1 = rlm.bse
    return dict(n=len(x), intercept=b0, intercept_err=e0, slope=b1, slope_err=e1,
                pearson=stats.pearsonr(x, y)[0], spearman=stats.spearmanr(x, y)[0],
                nmad_resid=nmad(y - (b0 + b1 * x)), nmad_y=nmad(y))


def band_grid(d, ycol, results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    xb, yb = BINS["aos_fwhm"], BINS[ycol]
    for ax, b in zip(axes.flat, BANDS):
        s = d[d.band == b]
        lo_x, hi_x = CUTS["aos_fwhm"]
        lo_y, hi_y = CUTS[ycol]
        s = s[s.aos_fwhm.between(lo_x, hi_x) & s[ycol].between(lo_y, hi_y)]
        if len(s) < 20:
            ax.text(0.5, 0.5, f"{b}: {len(s)} visits", transform=ax.transAxes, ha="center")
            continue
        h = ax.hist2d(s.aos_fwhm, s[ycol], bins=[xb, yb], norm=LogNorm(), cmap="viridis",
                      cmin=1)
        r = fit(s.aos_fwhm.to_numpy(), s[ycol].to_numpy())
        results.append(dict(band=b, y=ycol, **r))
        xx = np.array([xb[0], xb[-1]])
        ax.plot(xx, r["intercept"] + r["slope"] * xx, color="#D55E00", lw=1.8,
                label="Huber fit")
        if ycol == "fwhm":
            ax.plot(xx, xx, color="0.4", lw=1, ls="--", label="y = x")
        unit = "arcsec" if ycol == "fwhm" else ""
        slope_unit = "arcsec/arcsec" if ycol == "fwhm" else "1/arcsec"
        ax.text(0.03, 0.97,
                f"band {b}, n = {r['n']} visits\n"
                f"slope = {r['slope']:.3f} ± {r['slope_err']:.3f} {slope_unit}\n"
                f"intercept = {r['intercept']:.3f} {unit}\n"
                f"Pearson r = {r['pearson']:.2f}, Spearman rho = {r['spearman']:.2f}\n"
                f"nMAD resid = {r['nmad_resid']:.3f} {unit}",
                transform=ax.transAxes, va="top", fontsize=7.5, color="black",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        fig.colorbar(h[3], ax=ax, label="visits per bin")
    for ax in axes[1]:
        ax.set_xlabel(LABELS["aos_fwhm"])
    for ax in axes[:, 0]:
        ax.set_ylabel(LABELS[ycol])
    axes.flat[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{LABELS[ycol]} vs AOS_FWHM, LSSTCam science visits, by band")
    fig.tight_layout()
    return fig


def page_summary(pdf, d, res):
    lines = [
        "nightlyiq/iq_stochastic: PSF FWHM and median ellipticity vs AOS_FWHM",
        "",
        f"ConsDB visit1_quicklook, LSSTCam img_type science with aos_fwhm, engineering",
        f"observation_reason excluded: {len(d)} visits, {d.day_obs.nunique()} nights, "
        f"day_obs {d.day_obs.min()} to {d.day_obs.max()}",
        "PSF FWHM = psf_sigma_median x 2.355 x 0.2 arcsec/pixel",
        f"Median ellipticity = value_added v1@{VA_BUILD} psf_e_azel_p50 (median |e| over CCDs), "
        f"psf_n_ccd >= {MIN_CCD}; {d.e_med.notna().sum()} visits have it",
        f"Cuts: AOS_FWHM {CUTS['aos_fwhm']} arcsec, FWHM {CUTS['fwhm']} arcsec, "
        f"ellipticity {CUTS['e_med']} (dimensionless)",
        "Fit: Huber RLM, y = intercept + slope x AOS_FWHM. nMAD resid = robust scatter of",
        "the fit residuals; nMAD y = robust scatter of y alone (same units as y).",
        "",
        f"{'band':4s} {'y':6s} {'n':>6s} {'slope':>15s} {'intercept':>10s} {'Pearson r':>9s} "
        f"{'Spearman':>9s} {'nMAD resid':>10s} {'nMAD y':>8s}",
        "     units: FWHM slope arcsec/arcsec, intercept and nMAD arcsec; ellipticity slope "
        "1/arcsec, intercept and nMAD dimensionless",
    ]
    for r in res.itertuples():
        lines.append(f"{r.band:4s} {r.y:6s} {r.n:6d} {r.slope:7.3f}±{r.slope_err:.3f}  "
                     f"{r.intercept:10.3f} {r.pearson:9.2f} {r.spearman:9.2f} "
                     f"{r.nmad_resid:10.3f} {r.nmad_y:8.3f}")
    fig = plt.figure(figsize=(13, 8.5))
    fig.text(0.04, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    d = load(args.refetch)
    run = study_run(STUDY, "iq_vs_aos")
    results = []
    figs = [band_grid(d, ycol, results) for ycol in ("fwhm", "e_med")]
    res = pd.DataFrame(results)
    res.to_csv(run.path / "huber_fits.csv", index=False)
    out = run.path / "iq_vs_aos.pdf"
    with PdfPages(out) as pdf:
        page_summary(pdf, d, res)
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
    print(res.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
