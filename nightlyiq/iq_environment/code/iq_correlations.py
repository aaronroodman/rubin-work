"""Science-visit image quality against thermal, airflow and the intra - extra donut blur.

Spearman rho of each IQ target against each input, over all visits and within night
(night medians subtracted from both), then hexbin pages for the airflow and blur inputs.
One PDF in the run directory.

Usage::

    python nightlyiq/iq_environment/code/iq_correlations.py
"""

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
from rubinwork.common.utils import nmad
from rubinwork.products import cwfs_science
from rubinwork.products.value_added import reader

VA_BUILD = "snapshots/20261105"
CWFS_VARIANT = "d13-blitz_rso1071"
CONSDB_CACHE = (pathlib.Path(__file__).resolve().parents[1] / "output" / "cache"
                / "consdb_per_visit.parquet")
SIG2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
PIXEL_SCALE = 0.2  # arcsec per pixel
# Effective wavelengths [nm] for the zenith, 500 nm FWHM scaling.
BAND_NM = {"u": 368.0, "g": 482.0, "r": 622.0, "i": 755.0, "z": 869.0, "y": 971.0}
ENG_REASON = (r"aos|stability|focus|check|position|align|manual|dark|bias|flat|^test$"
              r"|_test|block-|filter_change_close_loop")
MIN_DONUTS = 3
MIN_RAFTS = 2
MIN_NIGHT = 20  # visits a night needs to enter the within-night correlation
TMA_SONIC = (123, 124, 125, 126)
OI = {"blue": "#0072B2", "orange": "#E69F00", "verm": "#D55E00", "black": "#000000"}

TARGETS = {
    "fwhm": "PSF FWHM (arcsec)",
    "fwhm_z500": "PSF FWHM, zenith 500 nm (arcsec)",
    "aos_fwhm": "AOS FWHM (arcsec)",
    "donut_blur_fwhm": "donut blur FWHM (arcsec)",
    "psf_e_azel_p50": "|e| p50 over CCDs (dimensionless)",
    "psf_e_azel_p90": "|e| p90 over CCDs (dimensionless)",
    "e_coh": "|mean e| (dimensionless)",
    "coh_frac": "|mean e| / mean |e| (dimensionless)",
}

THERMAL = {
    "m1m3_glass_minus_air_c": "M1M3 glass - M1M3 air (deg C)",
    "m1_glass_minus_air_c": "M1 glass - M1M3 air (deg C)",
    "m3_glass_minus_air_c": "M3 glass - M1M3 air (deg C)",
    "m1_minus_m3_glass_c": "M1 glass - M3 glass (deg C)",
    "m1m3_mean_temp_c": "M1M3 glass temperature (deg C)",
    "truss_minus_m1m3_air_c": "truss - M1M3 air (deg C)",
    "truss_minus_m2_air_c": "truss - M2 air (deg C)",
    "m2_minus_m1m3_air_c": "M2 air - M1M3 air (deg C)",
    "cam_minus_m1m3_air_c": "camera air - M1M3 air (deg C)",
    "cam_AverageTemp": "camera body average temperature (deg C)",
    "cam_body_minus_m1m3_air_c": "camera body - M1M3 air (deg C)",
    "cam_body_minus_truss_c": "camera body - mean truss (deg C)",
    "m2_air_minus_outside_c": "M2 air - outside (deg C)",
    "m1m3_x_gradient_c_per_m": "M1M3 x gradient (deg C/m)",
    "m1m3_y_gradient_c_per_m": "M1M3 y gradient (deg C/m)",
    "m1m3_z_gradient_c_per_m": "M1M3 z gradient (deg C/m)",
    "m1m3_radial_gradient_c_per_m": "M1M3 radial gradient (deg C/m)",
    "m1m3_r2_coeff_c": "M1M3 quadratic-in-radius (deg C)",
    "m1m3_std_temp_c": "M1M3 thermocouple scatter (deg C)",
    "m1m3_range_temp_c": "M1M3 thermocouple range (deg C)",
    "truss_temp_mean_c": "mean truss temperature (deg C)",
    "m1m3_air_temp": "M1M3 air temperature (deg C)",
    "outside_temp": "outside air temperature (deg C)",
}
AIRFLOW = {
    **{f"sonic{i}_mean_2_ms": f"sonic {i} axis 2 (m/s)" for i in TMA_SONIC},
    **{f"sonic{i}_abs2_ms": f"|sonic {i} axis 2| (m/s)" for i in TMA_SONIC},
    **{f"sonic{i}_cross_ms": f"sonic {i} cross-axis, hypot(ax 0, ax 1) (m/s)" for i in TMA_SONIC},
    **{f"sonic{i}_rms_2_ms": f"sonic {i} axis 2 RMS (m/s)" for i in TMA_SONIC},
    "wind_speed_ms": "weather tower wind speed (m/s)",
}
OTHER = {
    "blur_diff": "intra - extra donut blur (arcsec)",
    "altitude_deg": "elevation (deg)",
}
INPUTS = {**THERMAL, **AIRFLOW, **OTHER}


def blur_diff():
    """Intra - extra CWFS blur per visit, as in aos/cwfs_blur_intra_extra."""
    ccd = cwfs_science.load_table("ccd_visits", CWFS_VARIANT)
    ccd = ccd[ccd["n_quality"] >= MIN_DONUTS]
    p = ccd.pivot_table(index=["visit", "raft"], columns="wfs_offset",
                        values="blur_median").dropna()
    raft = pd.DataFrame({"diff": p[-1] - p[0]}).reset_index()
    g = raft.groupby("visit")["diff"]
    v = pd.DataFrame({"blur_diff": g.median(), "n_rafts": g.size()})
    v = v[v.n_rafts >= MIN_RAFTS].reset_index().rename(columns={"visit": "visit_id"})
    return v[["visit_id", "blur_diff"]]


def load():
    """Science visits with targets and inputs."""
    c = pd.read_parquet(CONSDB_CACHE)
    c = c[(c.img_type.str.lower() == "science")
          & ~c.observation_reason.str.contains(ENG_REASON, case=False, regex=True, na=False)
          & ~((c.day_obs == 20260424) & (c.seq_num < 398))]
    con = reader.open_db(reader.catalog_db_path("v1", VA_BUILD))
    try:
        va_cols = (["visit_id", "cam_AverageTemp", "wind_speed_ms", "into_wind_deg", "psf_n_ccd",
                    "m1m3_x_gradient_c_per_m", "m1m3_y_gradient_c_per_m",
                    "m1m3_z_gradient_c_per_m", "m1m3_radial_gradient_c_per_m"]
                   + [c_ for c_ in [r[0] for r in con.execute(
                       "DESCRIBE visit_telemetry").fetchall()]
                      if c_.startswith(("psf_e", "psf_xi"))
                      or any(c_.startswith(f"sonic{i}_") for i in TMA_SONIC)])
        va = con.execute(f"SELECT {', '.join(va_cols)} FROM visit_telemetry").df()
        r2 = con.execute("SELECT visit_id, m1m3_r2_coeff_c, m1m3_mean_temp_c, "
                         "m1m3_std_temp_c, m1m3_range_temp_c, m1_mean_temp_c, "
                         "m3_mean_temp_c FROM m1m3_thermal_r2").df()
    finally:
        con.close()
    d = c.merge(va, on="visit_id", how="left").merge(r2, on="visit_id", how="left")
    d = d.merge(blur_diff(), on="visit_id", how="left")

    d["fwhm"] = d.psf_sigma_median * SIG2FWHM * PIXEL_SCALE
    airmass = 1.0 / np.sin(np.radians(d.altitude_deg))
    lam = d.band.map(BAND_NM)
    d["fwhm_z500"] = d.fwhm * airmass ** -0.6 * (lam / 500.0) ** 0.2
    # Partial readouts give a biased ellipticity summary.
    bad_e = ~(d.psf_n_ccd >= 150)
    d["e_coh"] = np.hypot(d.psf_e1_azel_mean, d.psf_e2_azel_mean)
    d["coh_frac"] = d.e_coh / d.psf_e_azel_mean
    for col in ("psf_e_azel_p50", "psf_e_azel_p90", "e_coh", "coh_frac"):
        d.loc[bad_e, col] = np.nan

    d["truss_minus_m1m3_air_c"] = d.truss_temp_mean_c - d.m1m3_air_temp
    d["truss_minus_m2_air_c"] = d.truss_temp_mean_c - d.m2_air_temp
    d["m2_minus_m1m3_air_c"] = d.m2_air_temp - d.m1m3_air_temp
    d["cam_minus_m1m3_air_c"] = d.cam_air_temp - d.m1m3_air_temp
    d["m2_air_minus_outside_c"] = d.m2_air_temp - d.outside_temp
    d["m1m3_glass_minus_air_c"] = d.m1m3_mean_temp_c - d.m1m3_air_temp
    d["m1_glass_minus_air_c"] = d.m1_mean_temp_c - d.m1m3_air_temp
    d["m3_glass_minus_air_c"] = d.m3_mean_temp_c - d.m1m3_air_temp
    d["m1_minus_m3_glass_c"] = d.m1_mean_temp_c - d.m3_mean_temp_c
    d["cam_body_minus_m1m3_air_c"] = d.cam_AverageTemp - d.m1m3_air_temp
    d["cam_body_minus_truss_c"] = d.cam_AverageTemp - d.truss_temp_mean_c
    for i in TMA_SONIC:
        d[f"sonic{i}_abs2_ms"] = d[f"sonic{i}_mean_2_ms"].abs()
        d[f"sonic{i}_cross_ms"] = np.hypot(d[f"sonic{i}_mean_0_ms"], d[f"sonic{i}_mean_1_ms"])
    return d.reset_index(drop=True)


def rho_matrix(d, within):
    """Spearman rho and n, targets x inputs; within-night subtracts night medians."""
    rho = pd.DataFrame(index=list(TARGETS), columns=list(INPUTS), dtype=float)
    n = rho.copy()
    for t in TARGETS:
        for x in INPUTS:
            s = d[["day_obs", t, x]].dropna()
            if within:
                s = s[s.groupby("day_obs")[t].transform("size") >= MIN_NIGHT]
                s[[t, x]] = s[[t, x]] - s.groupby("day_obs")[[t, x]].transform("median")
            if len(s) > 100:
                rho.loc[t, x] = stats.spearmanr(s[t], s[x])[0]
            n.loc[t, x] = len(s)
    return rho, n


def page_heatmap(pdf, rho, n, title):
    fig, ax = plt.subplots(figsize=(17, 6.5))
    im = ax.imshow(rho.to_numpy(float), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_yticks(range(len(TARGETS)), list(TARGETS.values()), fontsize=8)
    ax.set_xticks(range(len(INPUTS)), list(INPUTS.values()), rotation=60, ha="right",
                  fontsize=6.5)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(v) > 0.4 else "black")
    plt.colorbar(im, ax=ax, label="Spearman rho (dimensionless)", fraction=0.02)
    nmin, nmax = int(np.nanmin(n.to_numpy(float))), int(np.nanmax(n.to_numpy(float)))
    ax.set_title(f"{title}; n per cell {nmin} to {nmax} visits")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def binned_median(ax, x, y, color):
    q = np.nanpercentile(x, [1, 99])
    bins = np.linspace(q[0], q[1], 21)
    idx = np.digitize(x, bins)
    c = 0.5 * (bins[1:] + bins[:-1])
    med = [np.nanmedian(y[idx == i]) if (idx == i).sum() > 30 else np.nan
           for i in range(1, len(bins))]
    ax.plot(c, med, "o-", color=color, ms=4, lw=1.5, label="binned median")


def page_scatter(pdf, d, inputs, targets, title):
    """One row per input, one column per target; hexbin plus binned median."""
    fig, axes = plt.subplots(len(inputs), len(targets),
                             figsize=(3.2 * len(targets), 2.6 * len(inputs)), squeeze=False)
    for i, x in enumerate(inputs):
        for j, t in enumerate(targets):
            ax = axes[i, j]
            s = d[[x, t]].dropna()
            lo = s.quantile(0.005)
            hi = s.quantile(0.995)
            s = s[(s[x] >= lo[x]) & (s[x] <= hi[x]) & (s[t] >= lo[t]) & (s[t] <= hi[t])]
            if len(s) < 100:
                ax.set_visible(False)
                continue
            ax.hexbin(s[x], s[t], gridsize=50, norm=LogNorm(), cmap="viridis", mincnt=1)
            binned_median(ax, s[x].to_numpy(), s[t].to_numpy(), OI["verm"])
            r = stats.pearsonr(s[x], s[t])[0]
            rho = stats.spearmanr(s[x], s[t])[0]
            ax.set_title(f"r {r:+.2f}, rho {rho:+.2f}, n {len(s)}", fontsize=7)
            ax.tick_params(labelsize=6)
            if i == len(inputs) - 1:
                ax.set_xlabel(INPUTS[x], fontsize=7)
            else:
                ax.set_xlabel(INPUTS[x], fontsize=6)
            if j == 0:
                ax.set_ylabel(TARGETS[t], fontsize=7)
    fig.suptitle(f"{title} (central 99% in each axis; Pearson r, Spearman rho)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_summary(pdf, d, rho_w, n_w, lines):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, "Science-visit image quality against thermal, airflow and blur",
             fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.88, "\n".join(lines), fontsize=9, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def top_lines(rho, n, k=4):
    out = []
    for t in TARGETS:
        r = rho.loc[t].dropna()
        best = r.reindex(r.abs().sort_values(ascending=False).index)[:k]
        out.append(f"  {t}: " + "; ".join(f"{x} {v:+.2f} (n {int(n.loc[t, x])})"
                                           for x, v in best.items()))
    return out


def main():
    run = study_run("nightlyiq/iq_environment", "iq_correlations")
    d = load()
    rho_a, n_a = rho_matrix(d, within=False)
    rho_w, n_w = rho_matrix(d, within=True)
    rho_a.to_csv(run.path / "spearman_all.csv")
    rho_w.to_csv(run.path / "spearman_within_night.csv")
    lines = [f"value_added v1@{VA_BUILD}; cwfs_science {CWFS_VARIANT}; ConsDB fetched 2026-11-11",
             f"Science visits (img_type science, engineering observation_reason excluded, "
             f"20260424 seq < 398 dropped):",
             f"    {len(d)} visits, {d.day_obs.nunique()} nights, day_obs {d.day_obs.min()} "
             f"to {d.day_obs.max()}",
             f"FWHM = psf_sigma_median x 2.355 x 0.2 arcsec; zenith 500 nm: x airmass^-0.6 "
             f"x (lambda_eff/500 nm)^0.2",
             f"Ellipticity summaries need psf_n_ccd >= 150. Within-night: night medians "
             f"subtracted, nights with >= {MIN_NIGHT} visits.",
             "",
             "Medians over visits:"]
    for t, lab in TARGETS.items():
        lines.append(f"    {lab}: {d[t].median():.4f}, nMAD {nmad(d[t].dropna()):.4f}, "
                     f"n {d[t].notna().sum()}")
    lines += ["", "Largest |Spearman rho|, within night (dimensionless):"] + top_lines(rho_w, n_w)
    lines += ["", "Largest |Spearman rho|, all visits (dimensionless):"] + top_lines(rho_a, n_a)
    out = run.path / "iq_correlations.pdf"
    with PdfPages(out) as pdf:
        page_summary(pdf, d, rho_w, n_w, lines)
        page_heatmap(pdf, rho_w, n_w, "Spearman rho, within night")
        page_heatmap(pdf, rho_a, n_a, "Spearman rho, all visits")
        tg = ["fwhm_z500", "aos_fwhm", "donut_blur_fwhm", "psf_e_azel_p50", "e_coh"]
        page_scatter(pdf, d, [f"sonic{i}_mean_2_ms" for i in TMA_SONIC], tg,
                     "Top-ring sonic axis 2")
        page_scatter(pdf, d, [f"sonic{i}_cross_ms" for i in TMA_SONIC], tg,
                     "Top-ring sonic cross-axis speed")
        page_scatter(pdf, d, ["blur_diff", "cam_body_minus_m1m3_air_c",
                              "m1m3_glass_minus_air_c", "truss_minus_m1m3_air_c"], tg,
                     "Blur difference and leading thermal differences")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
