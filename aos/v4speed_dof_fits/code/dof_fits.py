#!/usr/bin/env python
"""Danish fits of single-DOF perturbed donuts against the expected Zernike deviation.

Reads ``sim_donut_fits`` DOF-scan variants (``v4_speed_dof_p2_sparse``,
``v4_speed_dof_p2_dense_prod``) and their unperturbed counterparts (``v4_speed_sparse``,
``v4_speed_dense_prod``, the same settings on the same field points). Per donut, over the
variant's fitted Noll terms (quadrature sums, µm of wavefront):

- ``true``: the expected Zernike deviation, perturbed in-focus OPD minus the nominal
  intrinsic;
- ``err``: fitted deviation minus expected;
- ``err_delta``: (fitted deviation minus the unperturbed fit's) minus expected, the error
  on the DOF's signal with the fit's own model bias removed.

Writes ``dof_summary.pdf`` (medians per DOF and configuration), ``dof_summary.parquet``
and, per variant, ``dof_examples_<variant>.pdf``: one page per DOF, a FAM, a CWFS and a
giant donut with data, model and data - model per side, and the fitted deviation per
term (points, photon-noise 1-sigma at the variant's flux and sky) against the expected
(bars).

Usage::

    python aos/v4speed_dof_fits/code/dof_fits.py
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rubinwork.common import study_run
from rubinwork.donut_sim import fit, optics
from rubinwork.products import batoid_intrinsic, catalog, sim_donut_fits, sim_donuts

PAIRS = {"v4_speed_dof_p2_sparse": "v4_speed_sparse",
         "v4_speed_dof_p2_dense_prod": "v4_speed_dense_prod"}
CONFIGS = ("fam", "cwfs", "giant")
SIDES = ("extra", "intra")
FIELD_LIMIT_DEG = 1.725
EXAMPLE_R_DEG = 1.2              # FAM and giant example: nearest point, azimuth near 45 deg
RIGID = ["M2 dz", "M2 dx", "M2 dy", "M2 Rx", "M2 Ry",
         "Cam dz", "Cam dx", "Cam dy", "Cam Rx", "Cam Ry"]
DOF_NAMES = RIGID + [f"M1M3 B{i}" for i in range(1, 21)] + [f"M2 B{i}" for i in range(1, 21)]
DOF_UNITS = ["um", "um", "um", "deg", "deg", "um", "um", "um", "deg", "deg"] + ["um"] * 40


def dof_label(d, amp):
    return f"{DOF_NAMES[d]} {amp:+.4g} {DOF_UNITS[d]}"


def per_donut(variant, nominal):
    """One row per DOF donut: quadrature-sum metrics over the fitted terms (µm)."""
    cfg = sim_donut_fits.variant_config(variant)
    df = sim_donut_fits.load(variant)
    nom = sim_donut_fits.load(nominal)
    nom = nom[nom.sim_variant == "v4_speed_mono"].set_index(["config", "side", "point_id"])
    rows = []
    for config, g in df.groupby("config"):
        noll = cfg["noll"][config]
        fitc = [f"dz{j}_CCS_um" for j in noll]
        truec = [f"dz{j}_true_CCS_um" for j in noll]
        other = [f"dz{j}_true_CCS_um" for j in range(4, 79) if j not in noll]
        n = nom.loc[[(config, s, p) for s, p in zip(g.side, g.point_id)]]
        f, t = g[fitc].to_numpy(), g[truec].to_numpy()
        f0 = n[fitc].to_numpy()
        rows.append(g[["config", "side", "point_id", "dof", "dof_amplitude", "field_r_deg",
                       "l1", "fwhm_arcsec", "chi2_dof"]].assign(
            l1_nominal=n.l1.to_numpy(), fwhm_nominal=n.fwhm_arcsec.to_numpy(),
            true_um=np.sqrt((t ** 2).sum(1)),
            err_um=np.sqrt(((f - t) ** 2).sum(1)),
            err_delta_um=np.sqrt(((f - f0 - t) ** 2).sum(1)),
            bias_nominal_um=np.sqrt((f0 ** 2).sum(1)),
            true_unfitted_um=np.sqrt((g[other].to_numpy() ** 2).sum(1)),
            gain=((f - f0) * t).sum(1) / np.maximum((t ** 2).sum(1), 1e-30)))
    out = pd.concat(rows, ignore_index=True)
    return out[out.field_r_deg <= FIELD_LIMIT_DEG]


def summary_pages(pdf, tabs):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, "v4speed_dof_fits: Danish fits of single-DOF perturbed donuts",
             size=13, weight="bold", va="top")
    lines = [
        "Sims: sim_donuts v4_speed_mono_dof_p2, the v4_speed_mono field points traced once per",
        "DOF of the 22-DOF set, each DOF alone at +2 steps of the smatrix/sparse_fit scan",
        "(about 0.04 um RMS in-focus Noll 4-78 change). Noise-free, 1e6 electrons per donut,",
        "fit variance sky 1000 electrons per 10 um pixel plus the model.",
        "Fits against the NOMINAL intrinsicTA, as production: v4_speed_dof_p2_sparse (d13-sparse",
        "set, giant fit c, spider on, bin 2, converged) and v4_speed_dof_p2_dense_prod (Noll 4-19,",
        "22-26, no spider, bin 2, donut_viz tolerances).",
        "",
        "Per donut, quadrature sums over the fitted Noll terms (um of wavefront, CCS):",
        "  expected = Zernike deviation, perturbed in-focus OPD - nominal intrinsic;",
        "  error = fitted deviation - expected;",
        "  error of the change = (fitted - unperturbed fit at the same point) - expected:",
        "    the fit's own model bias (v4_speed_sparse, v4_speed_dense_prod) removed;",
        "  gain = (fitted change . expected) / |expected|^2 (dimensionless; 1 is full recovery).",
        "Medians over donuts within 1.725 deg field radius, both sides.",
    ]
    fig.text(0.05, 0.88, "\n".join(lines), size=9, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)
    for variant, d in tabs.items():
        med = d.groupby(["config", "dof"]).median(numeric_only=True)
        fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for ax, config in zip(axs, CONFIGS):
            m = med.loc[config]
            x = np.arange(len(m))
            ax.bar(x, m.true_um, 0.7, color="0.8", label="expected deviation")
            ax.plot(x, m.err_um, "o", ms=4, label="error (fitted - expected)")
            ax.plot(x, m.err_delta_um, "s", ms=4, mfc="none",
                    label="error of the change (unperturbed fit removed)")
            ax.plot(x, m.true_unfitted_um, "x", ms=4, color="tab:red",
                    label="expected deviation in unfitted Noll 4-78")
            ax.set_yscale("log")
            ax.set_ylabel("quad. sum over fitted terms\n(um of wavefront)", size=8)
            ax.set_title(f"{variant}, {config}: medians per DOF", size=9)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=6, ncol=2)
        amp = d.groupby("dof").dof_amplitude.first()
        axs[-1].set_xticks(x, [dof_label(i, a) for i, a in amp.items()], rotation=60,
                           ha="right", size=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for ax, config in zip(axs, CONFIGS):
            m = med.loc[config]
            x = np.arange(len(m))
            ax.plot(x, m.l1, "o", ms=4, label="l1 with the DOF")
            ax.axhline(m.l1_nominal.median(), color="k", lw=0.8, ls="--",
                       label="l1 unperturbed")
            ax2 = ax.twinx()
            ax2.plot(x, m.gain, "^", ms=4, color="tab:green", label="gain")
            ax2.axhline(1, color="tab:green", lw=0.5)
            ax2.set_ylabel("gain (dimensionless)", size=8, color="tab:green")
            ax.set_ylabel("l1, fraction of flux misplaced", size=8)
            ax.set_title(f"{variant}, {config}: fit quality and gain per DOF (medians)", size=9)
            ax.legend(fontsize=6, loc="upper left")
            ax2.legend(fontsize=6, loc="upper right")
        axs[-1].set_xticks(x, [dof_label(i, a) for i, a in amp.items()], rotation=60,
                           ha="right", size=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


class Context:
    """Danish factories, intrinsicTA and intrinsic for the Jacobian errors."""

    def __init__(self, cfg):
        self.cfg, self._fac = cfg, {}
        sv = cfg["sims"][0].split("@")[0]
        self.sim_variant = sv
        self.ivar = catalog.manifest("sim_donuts", sv, cfg["sims"][0].split("@")[1])[
            "inputs"]["batoid_intrinsic"].split("@")[0]
        iman = catalog.manifest("batoid_intrinsic", self.ivar)
        self.icfg = iman["config"]
        cols = batoid_intrinsic.noll_columns(self.icfg["jmax"], "CCS")
        ta = batoid_intrinsic.load(self.ivar, iman["build"], kind="ta")
        self.ta = ta.set_index(["band", "config", "side", "point_id"])[cols]
        idx = sim_donuts.load(sv, cfg["sims"][0].split("@")[1])
        self.npix = idx.groupby("config").npix.first().to_dict()

    def factory(self, config):
        if config not in self._fac:
            instr = optics.instrument(self.icfg["model"], self.icfg["obscuration"],
                                      self.icfg["configs"][config])
            b = self.cfg["binning"]
            self._fac[config] = (fit.factory(instr, b, self.cfg["spider_angle_deg"]),
                                 3600 * np.rad2deg(instr.pixelSize * b / instr.focalLength))
        return self._fac[config]

    def sigma(self, r, data):
        """Photon-noise 1-sigma of the fitted deviations (µm)."""
        noll = self.cfg["noll"][r.config]
        fac, pix_arcsec = self.factory(r.config)
        z_ref = np.zeros(self.icfg["jmax"] + 1)
        z_ref[batoid_intrinsic.reader.JMIN:] = 1e-6 * self.ta.loc[
            (r.band, r.config, r.side, r.point_id)].to_numpy(float)
        c0 = fit.centre_offset_px(int(self.npix[r.config]), self.cfg["binning"])
        params = dict(flux=r.flux_ratio * data.sum(), dx=(r.dx_px + c0) * pix_arcsec,
                      dy=(r.dy_px + c0) * pix_arcsec, fwhm=r.fwhm_arcsec,
                      z_fit_m=1e-6 * np.array([r[f"dz{j}_CCS_um"] for j in noll]))
        sig = fit.param_errors(fac, data, z_ref,
                               *np.deg2rad([r.thx_CCS_deg, r.thy_CCS_deg]), noll,
                               self.cfg["sky_e_per_px"] / self.cfg["binning"] ** 2, params)
        return 1e6 * sig["z_fit_m"]


def _az(df):
    return np.rad2deg(np.arctan2(df.thy_CCS_deg, df.thx_CCS_deg)) % 360


def example_points(df):
    """``{config: {side: point_id}}`` for the example pages (same for every DOF)."""
    g = df[df.dof == df.dof.iloc[0]]
    out = {}
    for config in CONFIGS:
        h = g[g.config == config]
        if config == "cwfs":
            out[config] = {}
            for s in SIDES:
                hs = h[(h.side == s) & (h.field_r_deg <= FIELD_LIMIT_DEG)].sort_values(
                    "field_r_deg")
                out[config][s] = hs.iloc[len(hs) // 2].point_id
        else:
            he = h[h.side == "extra"]
            daz = np.abs((_az(he) - 45 + 180) % 360 - 180)
            p = he.loc[(np.abs(he.field_r_deg - EXAMPLE_R_DEG) + 1e-3 * daz).idxmin()].point_id
            out[config] = {s: p for s in SIDES}
    return out


def example_pages(path, variant):
    cfg = sim_donut_fits.variant_config(variant)
    df = sim_donut_fits.load(variant)
    ctx = Context(cfg)
    pts = example_points(df)
    with PdfPages(path) as pdf:
        for d, gd in df.groupby("dof"):
            fig = plt.figure(figsize=(19, 9))
            gs = fig.add_gridspec(3, 7, width_ratios=[1] * 6 + [2.4])
            for i, config in enumerate(CONFIGS):
                noll = cfg["noll"][config]
                bars = {}
                for k, s in enumerate(SIDES):
                    pid = pts[config][s]
                    r = gd[(gd.config == config) & (gd.side == s)
                           & (gd.point_id == pid)].iloc[0]
                    data, model, _ = sim_donut_fits.stamp(variant, ctx.sim_variant, r.band,
                                                          config, s, pid, dof=d)
                    res = data - model
                    lim, vmax = np.abs(res).max(), data.max()
                    ax = [fig.add_subplot(gs[i, 3 * k + j]) for j in range(3)]
                    ax[0].imshow(data, origin="lower", vmin=0, vmax=vmax)
                    ax[0].set_title(f"{config} {s} {pid}: data\nfield r "
                                    f"{r.field_r_deg:.3f} deg CCS", size=7)
                    ax[1].imshow(model, origin="lower", vmin=0, vmax=vmax)
                    ax[1].set_title(f"model: FWHM {r.fwhm_arcsec:.3f} arcsec\n"
                                    f"chi2/dof {r.chi2_dof:.4f}", size=7)
                    im = ax[2].imshow(res, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
                    ax[2].set_title(f"data - model, +-{100 * lim / vmax:.1f}% of data peak\n"
                                    f"l1 {r.l1:.4f}", size=7)
                    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02).ax.tick_params(
                        labelsize=5)
                    for a in ax:
                        a.set_xticks([])
                        a.set_yticks([])
                    bars[s] = (pid,
                               np.array([r[f"dz{j}_true_CCS_um"] for j in noll]),
                               np.array([r[f"dz{j}_CCS_um"] for j in noll]),
                               ctx.sigma(r, data))
                ax = fig.add_subplot(gs[i, 6])
                x = np.arange(len(noll))
                same = bars["extra"][0] == bars["intra"][0]
                if same:
                    ax.bar(x, bars["extra"][1], 0.7, color="0.8", label="expected deviation")
                for k, (s, c) in enumerate(zip(SIDES, ("tab:blue", "tab:orange"))):
                    pid, true, fz, sig = bars[s]
                    if not same:
                        ax.bar(x + (k - 0.5) * 0.4, true, 0.38, color=("0.8", "0.65")[k],
                               label=f"expected {s} ({pid})")
                    ax.errorbar(x + (k - 0.5) * 0.2, fz, yerr=sig, fmt="o", ms=2.5, lw=0.8,
                                color=c, label=f"fitted {s}: |fitted - expected| "
                                f"{np.sqrt(((fz - true) ** 2).sum()):.3f} um")
                ax.axhline(0, color="k", lw=0.4)
                ax.set_xticks(x, [str(j) for j in noll], size=6)
                ax.tick_params(axis="y", labelsize=6)
                ax.set_ylabel("Zernike deviation\n(um of wavefront, CCS)", size=7)
                ax.legend(fontsize=5.5)
            spider = ("off" if cfg["spider_angle_deg"] is None
                      else f"{cfg['spider_angle_deg']:.0f} deg")
            fig.suptitle(
                f"{variant}: {dof_label(d, gd.dof_amplitude.iloc[0])} (ts_ofc convention); "
                f"binning {cfg['binning']}, spider {spider}; fit against the nominal "
                "intrinsicTA\nright: Zernike deviation per fitted term, fitted (points, "
                "1-sigma photon noise at 1e6 e-, sky 1000 e- per 10 um pixel) against "
                "expected = perturbed OPD - nominal intrinsic (bars); images electrons "
                "per binned pixel, row = CCS y", size=8)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {variant} DOF {d} page", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="dof_fits")
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()
    run = study_run("aos/v4speed_dof_fits", a.label)
    tabs = {v: per_donut(v, n) for v, n in PAIRS.items()}
    pd.concat([t.assign(variant=v) for v, t in tabs.items()]).to_parquet(
        run.path / "dof_summary.parquet", index=False)
    with PdfPages(run.path / "dof_summary.pdf") as pdf:
        summary_pages(pdf, tabs)
    for v, t in tabs.items():
        med = t.groupby(["config", "dof"])[["true_um", "err_um", "err_delta_um", "gain",
                                             "l1", "l1_nominal"]].median()
        print(v)
        print(med.round(4).to_string())
    if not a.no_examples:
        for v in PAIRS:
            example_pages(run.path / f"dof_examples_{v}.pdf", v)
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
