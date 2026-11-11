"""Science-visit ellipticity against guider image motion and guider PSF shape.

Image motion predicts e1 = (Var_az - Var_el) / T and e2 = 2 Cov(az, el) / T in the Az/El
frame, T = 2 sigma^2 from the science psf_sigma_median. The guider stamp shapes
``e1_altaz``, ``e2_altaz`` are rotated by summit_utils into Alt/Az with a convention not
yet checked against psf_shape, so all four component pairs are correlated.
Also: which environment inputs drive the guider motion and the guider shape.

Usage::

    python nightlyiq/iq_environment/code/guider_ellipticity.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import iq_correlations as iqc  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402

GUIDER_CACHE = (pathlib.Path(__file__).resolve().parents[1] / "output" / "cache"
                / "guider_per_visit.parquet")
GUIDER_VARIANT = "v4"
SCI = {"e1_azel": "science mean e1, Az/El", "e2_azel": "science mean e2, Az/El",
       "e_coh": "science |mean e|", "psf_e_azel_p50": "science |e| p50"}
GD = {"mot_e1": "motion e1, Az/El", "mot_e2": "motion e2, Az/El",
      "mot_e": "motion |e|", "gd_e1_altaz": "guider stamp e1_altaz",
      "gd_e2_altaz": "guider stamp e2_altaz", "gd_e": "guider stamp |e|",
      "gd_jitter_arcsec": "guider motion RMS, sqrt(Var_az + Var_el) (arcsec)",
      "gd_fwhm": "guider stamp FWHM (arcsec)"}
DRIVERS = {"wind_speed_ms": "tower wind speed (m/s)",
           **{f"sonic{i}_mean_2_ms": f"sonic {i} axis 2 (m/s)" for i in iqc.TMA_SONIC},
           **{f"sonic{i}_rms_2_ms": f"sonic {i} axis-2 RMS (m/s)" for i in iqc.TMA_SONIC},
           "altitude_deg": "elevation (deg)", "abs_into_wind_deg": "|pointing - wind| (deg)",
           "cam_body_minus_m1m3_air_c": "camera body - M1M3 air (deg C)",
           "m1m3_glass_minus_air_c": "M1M3 glass - M1M3 air (deg C)",
           "blur_diff": "intra - extra blur (arcsec)"}


def load():
    d = iqc.load().merge(pd.read_parquet(GUIDER_CACHE), on="visit_id", how="inner")
    d = d[d.psf_n_ccd >= 150].copy()
    sigma = d.psf_sigma_median * iqc.PIXEL_SCALE
    t = 2 * sigma ** 2
    d["mot_e1"] = (d.gd_var_az - d.gd_var_el) / t
    d["mot_e2"] = 2 * d.gd_cov_azel / t
    d["mot_e"] = np.hypot(d.mot_e1, d.mot_e2)
    d["gd_e"] = np.hypot(d.gd_e1_altaz, d.gd_e2_altaz)
    d["gd_jitter_arcsec"] = np.sqrt(d.gd_var_az + d.gd_var_el)
    d["e1_azel"] = d.psf_e1_azel_mean
    d["e2_azel"] = d.psf_e2_azel_mean
    d["abs_into_wind_deg"] = d.into_wind_deg.abs()
    return d


def corr(s, a, b):
    s = s[[a, b]].dropna()
    return (stats.pearsonr(s[a], s[b])[0], stats.spearmanr(s[a], s[b])[0], len(s))


def within(d, cols):
    d = d.copy()
    d[cols] = d[cols] - d.groupby("day_obs")[cols].transform("median")
    return d


def scatter(ax, s, x, y, xl, yl, diag=True):
    s = s[[x, y]].dropna()
    lo, hi = s.quantile(0.005), s.quantile(0.995)
    s = s[(s[x] >= lo[x]) & (s[x] <= hi[x]) & (s[y] >= lo[y]) & (s[y] <= hi[y])]
    ax.plot(s[x], s[y], ".", ms=2, alpha=0.3, color="#0072B2")
    iqc.binned_median(ax, s[x].to_numpy(), s[y].to_numpy(), "#D55E00")
    if diag:
        lim = np.array([min(s[x].min(), s[y].min()), max(s[x].max(), s[y].max())])
        ax.plot(lim, lim, "--", color="0.5", lw=1, label="y = x")
        ax.legend(fontsize=7)
    r, rho, n = corr(s, x, y)
    ax.set_title(f"Pearson r {r:+.3f}, Spearman rho {rho:+.3f}, n {n}", fontsize=8)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)


def page_text(pdf, title, lines):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.89, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def page_components(pdf, d, gx1, gx2, label, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for r, sx in enumerate(("e1_azel", "e2_azel")):
        for c, g in enumerate((gx1, gx2)):
            scatter(axes[r, c], d, g, sx, f"{GD[g]} (dimensionless)",
                    f"{SCI[sx]} (dimensionless)", diag=(r == c))
    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def matrix(d, rows, cols):
    out = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for a in rows:
        for b in cols:
            out.loc[a, b] = corr(d, a, b)[1]
    return out


def page_matrix(pdf, m, title, rlab, clab):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(m.to_numpy(float), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_yticks(range(len(m.index)), [rlab[x] for x in m.index], fontsize=7)
    ax.set_xticks(range(len(m.columns)), [clab[x] for x in m.columns], rotation=50,
                  ha="right", fontsize=7)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.iat[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(m.iat[i, j]) > 0.4 else "black")
    plt.colorbar(im, ax=ax, label="Spearman rho (dimensionless)", fraction=0.025)
    ax.set_title(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    run = study_run("nightlyiq/iq_environment", "guider_ellipticity")
    d = load()
    cols = list(SCI) + list(GD) + list(DRIVERS)
    dw = within(d, cols)
    lines = [f"guider_moments {GUIDER_VARIANT}: {len(d)} science visits with psf_n_ccd >= 150 "
             f"on {d.day_obs.nunique()} nights ({', '.join(map(str, sorted(d.day_obs.unique())))})",
             "Motion: per stamp, median over stars (>= 3) of daz (x cos alt) and dalt; "
             "variances over the exposure, drift kept.",
             "Motion e: e1 = (Var_az - Var_el)/T, e2 = 2 Cov/T, T = 2 sigma^2 from the "
             "science psf_sigma_median.",
             "",
             "Medians over visits:",
             f"  guider motion RMS sqrt(Var_az + Var_el): {d.gd_jitter_arcsec.median():.4f} arcsec",
             f"  motion |e|: {d.mot_e.median():.4f}   science |mean e|: {d.e_coh.median():.4f}  "
             f"(dimensionless)",
             f"  guider stamp |median e|: {d.gd_e.median():.4f} (dimensionless)", "",
             "Spearman rho, guider against science components (all visits | within night):"]
    for g in ("mot_e1", "mot_e2", "gd_e1_altaz", "gd_e2_altaz"):
        for sx in ("e1_azel", "e2_azel"):
            a, w = corr(d, g, sx), corr(dw, g, sx)
            lines.append(f"  {g:12s} vs {sx}: {a[1]:+.3f} | {w[1]:+.3f}   (Pearson {a[0]:+.3f} "
                         f"| {w[0]:+.3f}, n {a[2]})")
    for g, sx in (("mot_e", "e_coh"), ("gd_e", "e_coh"), ("gd_jitter_arcsec", "e_coh"),
                  ("gd_jitter_arcsec", "psf_e_azel_p50")):
        a, w = corr(d, g, sx), corr(dw, g, sx)
        lines.append(f"  {g:16s} vs {sx}: {a[1]:+.3f} | {w[1]:+.3f}")
    m_a = matrix(d, list(GD), list(DRIVERS))
    m_w = matrix(dw, list(GD), list(DRIVERS))
    m_a.to_csv(run.path / "guider_vs_drivers_all.csv")
    m_w.to_csv(run.path / "guider_vs_drivers_within.csv")
    out = run.path / "guider_ellipticity.pdf"
    with PdfPages(out) as pdf:
        page_text(pdf, "Science ellipticity against guider motion and guider shape", lines)
        page_components(pdf, d, "gd_e1_altaz", "gd_e2_altaz", "guider stamp",
                        "Science coherent ellipticity against guider stamp shape, all visits")
        page_components(pdf, dw, "gd_e1_altaz", "gd_e2_altaz", "guider stamp",
                        "Same, within night (night medians subtracted)")
        page_components(pdf, d, "mot_e1", "mot_e2", "motion",
                        "Science coherent ellipticity against guider-motion prediction, all visits")
        page_matrix(pdf, m_a, "Guider quantities against environment, Spearman rho, all visits",
                    GD, DRIVERS)
        page_matrix(pdf, m_w, "Guider quantities against environment, Spearman rho, within night",
                    GD, DRIVERS)
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
