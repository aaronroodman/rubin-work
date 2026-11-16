"""PSF FWHM and median ellipticity against AOS_FWHM and the donut blur, per band.

Science visits from ConsDB ``visit1_quicklook``: ``psf_sigma_median`` (pixels) and
``aos_fwhm`` (arcsec, the PSF FWHM the AOS wavefront residual alone would give). The
median ellipticity is ``psf_e_azel_p50`` from value_added, the median |e| over CCDs
(dimensionless); ConsDB ``psf_ellipticity_median`` is empty. ``donut_blur_fwhm``
(arcsec) is the atmospheric blur fitted to the corner-sensor donuts of the same exposure,
the per-visit seeing. Each band's 2D histogram carries the median and nMAD of y in bins
of x. The ConsDB pull is cached in
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
CUTS = {"aos_fwhm": (0.0, 1.0), "fwhm": (0.3, 3.0), "e_med": (0.0, 0.3), "blur": (0.1, 3.0)}
LABELS = {"aos_fwhm": "AOS_FWHM (arcsec)", "fwhm": "PSF FWHM (arcsec)",
          "e_med": "median ellipticity (dimensionless)", "blur": "donut blur FWHM (arcsec)",
          "excess2": "excess over seeing, FWHM² - blur² (arcsec²)",
          "leftover": "FWHM - sqrt(blur² + AOS_FWHM²) (arcsec)"}
BINS = {"aos_fwhm": np.linspace(0.0, 1.0, 51), "fwhm": np.linspace(0.4, 2.8, 97),
        "e_med": np.linspace(0.0, 0.25, 101), "blur": np.linspace(0.4, 2.4, 81),
        "excess2": np.linspace(-1.0, 2.0, 121), "leftover": np.linspace(-0.6, 0.6, 121)}
# Median bins, arcsec of x.
MED_EDGES = {"aos_fwhm": np.linspace(0.0, 1.0, 21), "blur": np.linspace(0.4, 2.4, 21)}
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
                 e_med=d.psf_e_azel_p50, blur=d.donut_blur_fwhm)
    # Squared, not a signed root: the root's infinite slope at zero empties the bins there.
    d["excess2"] = d.fwhm ** 2 - d.blur ** 2
    d["leftover"] = d.fwhm - np.hypot(d.blur, d.aos_fwhm)
    return d


def cut(d, cols):
    """Rows with every column in `cols` that has a cut inside its range."""
    keep = np.ones(len(d), bool)
    for c in cols:
        lo, hi = CUTS.get(c, (-np.inf, np.inf))
        keep &= d[c].between(lo, hi).to_numpy()
    return d[keep]


def binned_median(x, y, edges):
    """Median and nMAD of y in bins of x holding at least MIN_BIN visits."""
    idx = np.digitize(x, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        yk = y[idx == k]
        if len(yk) >= MIN_BIN:
            rows.append(dict(x_lo=edges[k], x_hi=edges[k + 1], n=len(yk),
                             median=np.median(yk), nmad=nmad(yk)))
    return pd.DataFrame(rows, columns=["x_lo", "x_hi", "n", "median", "nmad"])


def corr(x, y):
    return stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0]


def band_grid(d, xcol, ycol, med_rows, ref=None, extra=None, needs=()):
    """2x3 per-band 2D histograms of y against x with binned medians.

    ``ref`` is ``(label, f)``, a reference curve y = f(x). ``extra(s)`` returns more lines
    for the panel text; ``needs`` are columns cut on top
    of x and y, so a page's sample is the same as the page it is compared with.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    xb, yb = BINS[xcol], BINS[ycol]
    for ax, b in zip(axes.flat, BANDS):
        s = cut(d[d.band == b], (xcol, ycol) + tuple(needs))
        if len(s) < 20:
            ax.text(0.5, 0.5, f"{b}: {len(s)} visits", transform=ax.transAxes, ha="center")
            continue
        x, y = s[xcol].to_numpy(), s[ycol].to_numpy()
        h = ax.hist2d(x, y, bins=[xb, yb], norm=LogNorm(), cmap="viridis", cmin=1)
        m = binned_median(x, y, MED_EDGES[xcol])
        med_rows.append(m.assign(band=b, x=xcol, y=ycol))
        ax.errorbar(0.5 * (m.x_lo + m.x_hi), m["median"], yerr=m["nmad"], fmt="o",
                    color="#D55E00", ms=4, capsize=2, lw=1.2,
                    label=f"median ± nMAD, bins of >= {MIN_BIN} visits")
        if ref:
            xx = np.linspace(xb[0], xb[-1], 200)
            ax.plot(xx, ref[1](xx), color="0.3", lw=1, ls="--", label=ref[0])
        pr, sr = corr(x, y)
        lines = [f"band {b}, n = {len(s)} visits",
                 f"Pearson r = {pr:.2f}, Spearman rho = {sr:.2f}"]
        if extra:
            lines += extra(s)
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                fontsize=7.5, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        fig.colorbar(h[3], ax=ax, label="visits per bin")
    for ax in axes[1]:
        ax.set_xlabel(LABELS[xcol])
    for ax in axes[:, 0]:
        ax.set_ylabel(LABELS[ycol])
    axes.flat[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{LABELS[ycol]} vs {LABELS[xcol]}, LSSTCam science visits, by band")
    fig.tight_layout()
    return fig


def aos_stats(d, ycol):
    """Per-band correlations and rise of y over RISE_RANGE of AOS_FWHM."""
    rows = []
    edges = MED_EDGES["aos_fwhm"]
    lo, hi = RISE_RANGE
    for b in BANDS:
        s = cut(d[d.band == b], ("aos_fwhm", ycol))
        x, y = s.aos_fwhm.to_numpy(), s[ycol].to_numpy()
        m = binned_median(x, y, edges)
        win = m[(m.x_lo >= lo - 1e-9) & (m.x_hi <= hi + 1e-9)]
        full = len(win) == round((hi - lo) / (edges[1] - edges[0]))
        pr, sr = corr(x, y)
        rows.append(dict(band=b, y=ycol, n=len(s), pearson=pr, spearman=sr,
                         median_y=np.median(y), nmad_y=nmad(y),
                         rise=win["median"].iloc[-1] - win["median"].iloc[0] if full else np.nan,
                         bin_nmad_min=win.nmad.min(), bin_nmad_max=win.nmad.max()))
    return rows


def blur_stats(d):
    """Per-band seeing numbers on the sample with FWHM, blur and AOS_FWHM all cut."""
    rows = []
    for b in BANDS:
        s = cut(d[d.band == b], ("aos_fwhm", "fwhm", "blur"))
        diff = s.fwhm - s.blur
        rows.append(dict(
            band=b, n=len(s),
            spearman_blur_fwhm=corr(s.blur, s.fwhm)[1],
            spearman_aos_blur=corr(s.aos_fwhm, s.blur)[1],
            spearman_aos_excess=corr(s.aos_fwhm, s.excess2)[1],
            median_fwhm_minus_blur=diff.median(), nmad_fwhm_minus_blur=nmad(diff),
            nmad_fwhm=nmad(s.fwhm), median_leftover=s.leftover.median(),
            nmad_leftover=nmad(s.leftover)))
    return pd.DataFrame(rows)


def page_leftover(bs, d):
    """D: distribution of what neither seeing nor AOS_FWHM accounts for."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True)
    for ax, b, r in zip(axes.flat, BANDS, bs.itertuples()):
        s = cut(d[d.band == b], ("aos_fwhm", "fwhm", "blur"))
        ax.hist(s.leftover, bins=BINS["leftover"], color="#0072B2", histtype="stepfilled",
                alpha=0.8)
        ax.axvline(0, color="0.3", lw=1, ls="--")
        ax.text(0.03, 0.97,
                f"band {b}, n = {r.n} visits\nmedian = {r.median_leftover:+.3f} arcsec\n"
                f"nMAD = {r.nmad_leftover:.3f} arcsec\n(nMAD of FWHM = {r.nmad_fwhm:.3f} arcsec)",
                transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_ylabel("visits per bin")
    for ax in axes[1]:
        ax.set_xlabel(LABELS["leftover"])
    fig.suptitle("FWHM left after seeing and AOS_FWHM added in quadrature, by band")
    fig.tight_layout()
    return fig


def page_terciles(d, ycol):
    """E: binned median of y against AOS_FWHM, one line per donut-blur tercile."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    colors = ("#0072B2", "#E69F00", "#D55E00")
    edges = MED_EDGES["aos_fwhm"]
    for ax, b in zip(axes.flat, BANDS):
        s = cut(d[d.band == b], ("aos_fwhm", "blur", ycol))
        if len(s) < 20:
            continue
        t = np.quantile(s.blur, [0.0, 1 / 3, 2 / 3, 1.0])
        for k, color in enumerate(colors):
            sk = s[s.blur.between(t[k], t[k + 1])]
            m = binned_median(sk.aos_fwhm.to_numpy(), sk[ycol].to_numpy(), edges)
            ax.errorbar(0.5 * (m.x_lo + m.x_hi), m["median"], yerr=m["nmad"], fmt="o-",
                        color=color, ms=3.5, lw=1.2, capsize=2, alpha=0.9,
                        label=f"blur {t[k]:.2f}-{t[k + 1]:.2f} arcsec, n = {len(sk)}")
        ax.text(0.03, 0.97, f"band {b}", transform=ax.transAxes, va="top", fontsize=8)
        ax.legend(loc="lower right", fontsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel(LABELS["aos_fwhm"])
    for ax in axes[:, 0]:
        ax.set_ylabel(LABELS[ycol])
    fig.suptitle(f"{LABELS[ycol]} vs AOS_FWHM by donut-blur tercile: median ± nMAD, "
                 f"bins of >= {MIN_BIN} visits")
    fig.tight_layout()
    return fig


def page_text(pdf, lines, size=8.0):
    fig = plt.figure(figsize=(13, 8.5))
    fig.text(0.04, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=size)
    pdf.savefig(fig)
    plt.close(fig)


def summary_lines(d, res, bs):
    fetched = json.loads(CACHE.with_suffix(".json").read_text())["fetched_utc"][:16]
    lo, hi = RISE_RANGE
    lines = [
        "nightlyiq/iq_stochastic: PSF FWHM and median ellipticity vs AOS_FWHM and donut blur",
        "",
        f"ConsDB visit1_quicklook fetched {fetched}: LSSTCam img_type science with aos_fwhm,",
        f"engineering observation_reason excluded except {', '.join(KEEP_REASONS)}:",
        f"{len(d)} visits, {d.day_obs.nunique()} nights, day_obs {d.day_obs.min()} to "
        f"{d.day_obs.max()}, before the cuts below (per-band n after cuts is in the tables)",
        "PSF FWHM = psf_sigma_median x 2.355 x 0.2 arcsec/pixel; donut blur = donut_blur_fwhm",
        f"Median ellipticity = value_added v1@{VA_BUILD} psf_e_azel_p50 (median |e| over CCDs), "
        f"psf_n_ccd >= {MIN_CCD}; {d.e_med.notna().sum()} visits have it",
        "Cuts: " + ", ".join(f"{c} {v}" for c, v in CUTS.items())
        + " (arcsec; e_med dimensionless)",
        f"Points: median of y in bins of x (0.05 arcsec of AOS_FWHM, 0.10 arcsec of blur), "
        f"error bar nMAD (1.4826 x MAD); bins with >= {MIN_BIN} visits only.",
        "",
        f"1. y vs AOS_FWHM. rise = median y in the last minus the first 0.05 arcsec bin of "
        f"AOS_FWHM {lo:.2f}-{hi:.2f} arcsec;",
        "   bin nMAD = range of the per-bin nMAD of y over the same bins.",
        f"{'band':4s} {'y':6s} {'n':>6s} {'Pearson r':>9s} {'Spearman':>9s} {'median y':>9s} "
        f"{'nMAD y':>8s} {'rise':>7s} {'bin nMAD':>11s}",
        "     units: median, nMAD and rise arcsec for fwhm, dimensionless for e_med",
    ]
    for r in res.itertuples():
        lines.append(f"{r.band:4s} {r.y:6s} {r.n:6d} {r.pearson:9.2f} {r.spearman:9.2f} "
                     f"{r.median_y:9.3f} {r.nmad_y:8.3f} {r.rise:7.3f} "
                     f"{r.bin_nmad_min:5.3f}-{r.bin_nmad_max:5.3f}")
    lines += [
        "",
        "2. Seeing. Sample with AOS_FWHM, FWHM and blur all inside their cuts. Spearman rho of "
        "(blur, FWHM), (AOS_FWHM, blur),",
        "   (AOS_FWHM, excess); excess = FWHM² - blur²; leftover = "
        "FWHM - sqrt(blur² + AOS_FWHM²). F-b, F and left in arcsec.",
        f"{'band':4s} {'n':>6s} {'rho b,F':>8s} {'rho A,b':>8s} {'rho A,ex':>8s} "
        f"{'med F-b':>8s} {'nMAD F-b':>8s} {'nMAD F':>7s} {'med left':>8s} {'nMAD left':>9s}",
    ]
    for r in bs.itertuples():
        lines.append(f"{r.band:4s} {r.n:6d} {r.spearman_blur_fwhm:8.2f} "
                     f"{r.spearman_aos_blur:8.2f} {r.spearman_aos_excess:8.2f} "
                     f"{r.median_fwhm_minus_blur:8.3f} {r.nmad_fwhm_minus_blur:8.3f} "
                     f"{r.nmad_fwhm:7.3f} {r.median_leftover:+8.3f} {r.nmad_leftover:9.3f}")
    lines += [
        "",
        "Pages: FWHM vs AOS_FWHM; median ellipticity vs AOS_FWHM; A FWHM vs blur; "
        "B blur vs AOS_FWHM;",
        "C FWHM² - blur² vs AOS_FWHM (y = AOS_FWHM²: AOS_FWHM accounts for the excess in "
        "quadrature); D leftover histograms;",
        "E FWHM and median ellipticity vs AOS_FWHM by blur tercile (terciles per band).",
    ]
    return lines


def main():
    d = load()
    run = study_run(STUDY, "aos_and_blur")
    med = []
    three = ("aos_fwhm", "fwhm", "blur")
    diff_text = lambda s: [f"FWHM - blur: median {np.median(s.fwhm - s.blur):.3f}, "
                           f"nMAD {nmad(s.fwhm - s.blur):.3f} arcsec"]
    figs = [
        band_grid(d, "aos_fwhm", "fwhm", med,
                  extra=lambda s: [f"nMAD = {nmad(s.fwhm):.3f} arcsec"]),
        band_grid(d, "aos_fwhm", "e_med", med,
                  extra=lambda s: [f"nMAD = {nmad(s.e_med):.3f}"]),
        band_grid(d, "blur", "fwhm", med, ref=("y = x", lambda x: x), extra=diff_text,
                  needs=three),
        band_grid(d, "aos_fwhm", "blur", med, needs=three),
        band_grid(d, "aos_fwhm", "excess2", med, ref=("y = AOS_FWHM²", np.square),
                  needs=three),
    ]
    res = pd.DataFrame(aos_stats(d, "fwhm") + aos_stats(d, "e_med"))
    bs = blur_stats(d)
    figs += [page_leftover(bs, d), page_terciles(d, "fwhm"), page_terciles(d, "e_med")]
    res.to_csv(run.path / "aos_stats.csv", index=False)
    bs.to_csv(run.path / "blur_stats.csv", index=False)
    pd.concat(med, ignore_index=True).to_csv(run.path / "binned_medians.csv", index=False)
    out = run.path / "iq_stochastic.pdf"
    with PdfPages(out) as pdf:
        page_text(pdf, summary_lines(d, res, bs))
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
    print(res.round(3).to_string(index=False))
    print(bs.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
