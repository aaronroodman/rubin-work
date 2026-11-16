"""PSF FWHM and median ellipticity against AOS_FWHM, per band, as 2D histograms.

Science visits from ConsDB ``visit1_quicklook``: ``psf_sigma_median`` (pixels) and
``aos_fwhm`` (arcsec, the PSF FWHM the AOS wavefront residual alone would give). The
median ellipticity is ``psf_e_azel_p50`` from value_added, the median |e| over CCDs
(dimensionless); ConsDB ``psf_ellipticity_median`` is empty. Each band's 2D histogram
carries the median and nMAD of y in AOS_FWHM bins. The ConsDB pull is cached in
``output/cache/`` under the date in ``FETCH_DATE``; change that date to refetch.

Usage::

    python nightlyiq/iq_stochastic/code/iq_vs_aos.py
"""

import datetime
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy import stats

from rubinwork.common import study_run
from rubinwork.common.telemetry_clients import make_consdb_client
from rubinwork.common.utils import nmad
from rubinwork.products.value_added import reader

STUDY = "nightlyiq/iq_stochastic"
FETCH_DATE = "20261116"
CACHE = (pathlib.Path(__file__).resolve().parents[1] / "output" / "cache"
         / f"consdb_iq_{FETCH_DATE}.parquet")
SIG2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
PIXEL_SCALE = 0.2  # arcsec per pixel
VA_BUILD = "snapshots/20261105"
MIN_CCD = 150  # partial readouts bias the ellipticity summary
BANDS = ("u", "g", "r", "i", "z", "y")
ENG_REASON = (r"aos|stability|focus|check|position|align|manual|dark|bias|flat|^test$"
              r"|_test|block-|filter_change_close_loop")
# Survey-scheduler (FBS) visits taken for AOS stability; ordinary science images.
KEEP_REASONS = ("fbs_driven_aos_stability_test",)
# Ranges that drop failed measurements; statistics and histograms both use them.
CUTS = {"aos_fwhm": (0.0, 1.0), "fwhm": (0.3, 3.0), "e_med": (0.0, 0.3)}
LABELS = {"aos_fwhm": "AOS_FWHM (arcsec)", "fwhm": "PSF FWHM (arcsec)",
          "e_med": "median ellipticity (dimensionless)"}
BINS = {"aos_fwhm": np.linspace(0.0, 1.0, 51), "fwhm": np.linspace(0.4, 2.8, 97),
        "e_med": np.linspace(0.0, 0.25, 101)}
MED_EDGES = np.linspace(0.0, 1.0, 21)  # arcsec of AOS_FWHM
MIN_BIN = 250  # visits a bin needs before its median is drawn
# Every band has >= MIN_BIN visits in each bin here, so the rises compare like for like.
RISE_RANGE = (0.30, 0.60)  # arcsec of AOS_FWHM
IQ_COLS = ("psf_sigma_median", "aos_fwhm", "donut_blur_fwhm", "guider_psf_fwhm",
           "seeing_zenith_500nm_median")


def fetch():
    """Every LSSTCam science visit with an AOS_FWHM, one row per visit."""
    q = ("SELECT e.exposure_id AS visit_id, e.day_obs, e.seq_num, e.band, "
         "e.observation_reason, e.airmass, " + ", ".join(f"q.{c}" for c in IQ_COLS) + " "
         "FROM cdb_lsstcam.exposure e "
         "JOIN cdb_lsstcam.visit1_quicklook q ON q.visit_id = e.exposure_id "
         "WHERE e.img_type = 'science' AND q.aos_fwhm IS NOT NULL")
    d = make_consdb_client("auto").query(q).to_pandas()
    for c in IQ_COLS + ("airmass",):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, q


def load():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        d, q = fetch()
        d.to_parquet(CACHE, index=False)
        CACHE.with_suffix(".json").write_text(json.dumps(
            dict(fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 query=q, rows=len(d)), indent=1))
    d = pd.read_parquet(CACHE)
    con = reader.open_db(reader.catalog_db_path("v1", VA_BUILD))
    try:
        va = con.execute("SELECT visit_id, psf_e_azel_p50, psf_n_ccd FROM visit_telemetry").df()
    finally:
        con.close()
    va.loc[~(va.psf_n_ccd >= MIN_CCD), "psf_e_azel_p50"] = np.nan
    d = d.merge(va, on="visit_id", how="left")
    eng = (d.observation_reason.str.contains(ENG_REASON, case=False, regex=True, na=False)
           & ~d.observation_reason.isin(KEEP_REASONS))
    d = d[d.band.isin(BANDS) & ~eng]
    d = d.assign(fwhm=d.psf_sigma_median * SIG2FWHM * PIXEL_SCALE,
                 e_med=d.psf_e_azel_p50)
    return d


def binned_median(x, y):
    """Median and nMAD of y in AOS_FWHM bins holding at least MIN_BIN visits."""
    idx = np.digitize(x, MED_EDGES) - 1
    rows = []
    for k in range(len(MED_EDGES) - 1):
        yk = y[idx == k]
        if len(yk) >= MIN_BIN:
            rows.append(dict(x_lo=MED_EDGES[k], x_hi=MED_EDGES[k + 1], n=len(yk),
                             median=np.median(yk), nmad=nmad(yk)))
    return pd.DataFrame(rows)


def band_grid(d, ycol, stats_rows, med_rows):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    xb, yb = BINS["aos_fwhm"], BINS[ycol]
    unit = " arcsec" if ycol == "fwhm" else ""
    for ax, b in zip(axes.flat, BANDS):
        s = d[d.band == b]
        s = s[s.aos_fwhm.between(*CUTS["aos_fwhm"]) & s[ycol].between(*CUTS[ycol])]
        if len(s) < 20:
            ax.text(0.5, 0.5, f"{b}: {len(s)} visits", transform=ax.transAxes, ha="center")
            continue
        x, y = s.aos_fwhm.to_numpy(), s[ycol].to_numpy()
        h = ax.hist2d(x, y, bins=[xb, yb], norm=LogNorm(), cmap="viridis", cmin=1)
        m = binned_median(x, y)
        med_rows.append(m.assign(band=b, y=ycol))
        ax.errorbar(0.5 * (m.x_lo + m.x_hi), m["median"], yerr=m["nmad"], fmt="o",
                    color="#D55E00", ms=4, capsize=2, lw=1.2,
                    label=f"median ± nMAD, bins of >= {MIN_BIN} visits")
        lo, hi = RISE_RANGE
        win = m[(m.x_lo >= lo - 1e-9) & (m.x_hi <= hi + 1e-9)]
        full = len(win) == round((hi - lo) / (MED_EDGES[1] - MED_EDGES[0]))
        r = dict(band=b, y=ycol, n=len(s), pearson=stats.pearsonr(x, y)[0],
                 spearman=stats.spearmanr(x, y)[0], median_y=np.median(y), nmad_y=nmad(y),
                 rise=win["median"].iloc[-1] - win["median"].iloc[0] if full else np.nan,
                 bin_nmad_min=win.nmad.min(), bin_nmad_max=win.nmad.max())
        stats_rows.append(r)
        ax.text(0.03, 0.97,
                f"band {b}, n = {r['n']} visits\n"
                f"Pearson r = {r['pearson']:.2f}, Spearman rho = {r['spearman']:.2f}\n"
                f"nMAD = {r['nmad_y']:.3f}{unit}",
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
    fetched = json.loads(CACHE.with_suffix(".json").read_text())["fetched_utc"][:16]
    lines = [
        "nightlyiq/iq_stochastic: PSF FWHM and median ellipticity vs AOS_FWHM",
        "",
        f"ConsDB visit1_quicklook fetched {fetched}: LSSTCam img_type science with aos_fwhm,",
        f"engineering observation_reason excluded except {', '.join(KEEP_REASONS)}:",
        f"{len(d)} visits, {d.day_obs.nunique()} nights, day_obs {d.day_obs.min()} to "
        f"{d.day_obs.max()}, before the cuts below (per-band n after cuts is in the table)",
        "PSF FWHM = psf_sigma_median x 2.355 x 0.2 arcsec/pixel",
        f"Median ellipticity = value_added v1@{VA_BUILD} psf_e_azel_p50 (median |e| over CCDs), "
        f"psf_n_ccd >= {MIN_CCD}; {d.e_med.notna().sum()} visits have it",
        f"Cuts: AOS_FWHM {CUTS['aos_fwhm']} arcsec, FWHM {CUTS['fwhm']} arcsec, "
        f"ellipticity {CUTS['e_med']} (dimensionless)",
        f"Points: median of y in {MED_EDGES[1] - MED_EDGES[0]:.2f} arcsec AOS_FWHM bins, "
        f"error bar nMAD (1.4826 x MAD) of y; only bins with >= {MIN_BIN} visits.",
        f"rise = median y in the last minus the first bin of AOS_FWHM {RISE_RANGE} arcsec;",
        "bin nMAD = range of the per-bin nMAD of y over the same bins.",
        "",
        f"{'band':4s} {'y':6s} {'n':>6s} {'Pearson r':>9s} {'Spearman':>9s} {'median y':>9s} "
        f"{'nMAD y':>8s} {'rise':>7s} {'bin nMAD':>11s}",
        "     units: median, nMAD and rise arcsec for fwhm, dimensionless for e_med",
    ]
    for r in res.itertuples():
        lines.append(f"{r.band:4s} {r.y:6s} {r.n:6d} {r.pearson:9.2f} {r.spearman:9.2f} "
                     f"{r.median_y:9.3f} {r.nmad_y:8.3f} {r.rise:7.3f} "
                     f"{r.bin_nmad_min:5.3f}-{r.bin_nmad_max:5.3f}")
    fig = plt.figure(figsize=(13, 8.5))
    fig.text(0.04, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    d = load()
    run = study_run(STUDY, "binned_median")
    stats_rows, med_rows = [], []
    figs = [band_grid(d, ycol, stats_rows, med_rows) for ycol in ("fwhm", "e_med")]
    res = pd.DataFrame(stats_rows)
    res.to_csv(run.path / "band_stats.csv", index=False)
    pd.concat(med_rows, ignore_index=True).to_csv(run.path / "binned_medians.csv", index=False)
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
