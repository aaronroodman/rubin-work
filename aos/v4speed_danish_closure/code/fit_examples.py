#!/usr/bin/env python
"""Data, model and data - model images for a sample of ``sim_donut_fits`` donuts.

For each variant and defocus configuration, one PDF page: four field points, each side's
binned stamp, best-fit Danish model and residual, and a panel of the fitted wavefront
(OPD) per fitted Noll term against the batoid intrinsic (``batoid_intrinsic`` kind
``gq_infocus``), with photon-noise 1-sigma errors from the fit Jacobian at the variant's
flux and sky (``rubinwork.donut_sim.fit.param_errors``). FAM and giant use the same points on
both sides, the ones nearest field radius 0, 0.9, 1.4 and 1.725 deg at azimuth near
45 deg (CCS); CWFS takes each side's innermost and median point, its outermost within
1.725 deg and its outermost.

Usage::

    python aos/v4speed_danish_closure/code/fit_examples.py
    python aos/v4speed_danish_closure/code/fit_examples.py --variants v4_speed_sparse
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rubinwork.common import study_run
from rubinwork.donut_sim import fit, optics
from rubinwork.products import batoid_intrinsic, catalog, sim_donut_fits, sim_donuts

VARIANTS = ("v4_speed_sparse", "v4_speed_dense", "v4_speed_sparse_prod",
            "v4_speed_sparse_bin1")
CONFIGS = ("fam", "cwfs", "giant")
SIDES = ("extra", "intra")
TARGET_R_DEG = (0.0, 0.9, 1.4, 1.725)
TARGET_AZ_DEG = 45.0
FIELD_LIMIT_DEG = 1.725


def _az(df):
    return np.rad2deg(np.arctan2(df.thy_CCS_deg, df.thx_CCS_deg)) % 360


def examples(df, config):
    """``[(side, sim_variant, point_id), ...]`` rows of the page, four per side."""
    g = df[df.config == config]
    out = {s: [] for s in SIDES}
    if config == "cwfs":
        for s in SIDES:
            h = g[g.side == s].sort_values("field_r_deg")
            inside = h[h.field_r_deg <= FIELD_LIMIT_DEG]
            picks = [inside.iloc[0], inside.iloc[len(inside) // 2], inside.iloc[-1], h.iloc[-1]]
            out[s] = [(s, p.sim_variant, p.point_id) for p in picks]
        return out
    h = g[g.side == "extra"]
    dist_az = np.abs((_az(h) - TARGET_AZ_DEG + 180) % 360 - 180)
    for r in TARGET_R_DEG:
        # Field radius first; azimuth breaks ties (0.01 deg of radius per 10 deg of az).
        p = h.loc[(np.abs(h.field_r_deg - r) + 1e-3 * dist_az).idxmin()]
        for s in SIDES:
            out[s].append((s, p.sim_variant, p.point_id))
    return out


class Context:
    """Danish factory, intrinsicTA and intrinsic of the sims' ``batoid_intrinsic``."""

    def __init__(self, cfg):
        self.cfg, self._fac = cfg, {}
        ivars = {catalog.manifest("sim_donuts", sv)["inputs"]["batoid_intrinsic"].split("@")[0]
                 for sv in cfg["sims"]}
        (self.ivar,) = ivars
        iman = catalog.manifest("batoid_intrinsic", self.ivar)
        self.icfg = iman["config"]
        cols = batoid_intrinsic.noll_columns(self.icfg["jmax"], "CCS")
        ta = batoid_intrinsic.load(self.ivar, iman["build"], kind="ta")
        self.ta = ta.set_index(["band", "config", "side", "point_id"])[cols]
        infoc = batoid_intrinsic.load(self.ivar, iman["build"], kind="gq_infocus")
        self.intr = infoc.set_index(["band", "point_id"])[cols]
        self.npix = {(sv, c): int(n) for sv in cfg["sims"]
                     for c, n in sim_donuts.load(sv).groupby("config").npix.first().items()}

    def factory(self, config):
        if config not in self._fac:
            instr = optics.instrument(self.icfg["model"], self.icfg["obscuration"],
                                      self.icfg["configs"][config])
            b = self.cfg["binning"]
            self._fac[config] = (fit.factory(instr, b, self.cfg["spider_angle_deg"]),
                                 3600 * np.rad2deg(instr.pixelSize * b / instr.focalLength))
        return self._fac[config]

    def wavefront(self, r, data):
        """OPD, its 1-sigma error and the intrinsic for the fitted terms (µm)."""
        noll = self.cfg["noll"][r.config]
        fac, pix_arcsec = self.factory(r.config)
        z_ref = np.zeros(self.icfg["jmax"] + 1)
        z_ref[batoid_intrinsic.reader.JMIN:] = 1e-6 * self.ta.loc[
            (r.band, r.config, r.side, r.point_id)].to_numpy(float)
        dz = np.array([r[f"dz{j}_CCS_um"] for j in noll])
        c0 = fit.centre_offset_px(self.npix[r.sim_variant, r.config], self.cfg["binning"])
        params = dict(flux=r.flux_ratio * data.sum(), dx=(r.dx_px + c0) * pix_arcsec,
                      dy=(r.dy_px + c0) * pix_arcsec, fwhm=r.fwhm_arcsec, z_fit_m=1e-6 * dz)
        sig = fit.param_errors(fac, data, z_ref, *np.deg2rad([r.thx_CCS_deg, r.thy_CCS_deg]),
                               noll, self.cfg["sky_e_per_px"] / self.cfg["binning"] ** 2,
                               params)
        intr = self.intr.loc[(r.band, r.point_id), [f"z{j}_CCS" for j in noll]].to_numpy(float)
        return np.asarray(noll), intr + dz, 1e6 * sig["z_fit_m"], intr


def page(pdf, variant, config, df, cfg, ctx):
    picks = examples(df, config)
    rows = len(picks["extra"])
    fig = plt.figure(figsize=(19, 2.6 * rows + 0.8))
    gs = fig.add_gridspec(rows, 7, width_ratios=[1] * 6 + [2.4])
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(7)] for i in range(rows)])
    dzc = [c for c in df.columns if c.endswith("_CCS_um")]
    for i in range(rows):
        wf = {}
        for k, s in enumerate(SIDES):
            _, sv, pid = picks[s][i]
            r = df[(df.config == config) & (df.side == s) & (df.sim_variant == sv)
                   & (df.point_id == pid)].iloc[0]
            data, model, _ = sim_donut_fits.stamp(variant, sv, r.band, config, s, pid)
            res = data - model
            lim = np.abs(res).max()
            vmax = data.max()
            dz = np.sqrt(np.nansum(r[dzc].to_numpy(float) ** 2))
            a = axs[i, 3 * k:3 * k + 3]
            a[0].imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            a[0].set_title(f"{s} {pid}: data\nfield r {r.field_r_deg:.3f} deg, "
                           f"az {_az(r):.0f} deg CCS", size=7)
            a[1].imshow(model, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            a[1].set_title(f"model: FWHM {r.fwhm_arcsec:.3f} arcsec\n"
                           f"|OPD - intrinsic| {dz:.3f} um (quad. sum)", size=7)
            im = a[2].imshow(res, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            a[2].set_title(f"data - model, +-{100 * lim / vmax:.1f}% of data peak\n"
                           f"l1 {r.l1:.4f}, chi2/dof {r.chi2_dof:.4f}", size=7)
            fig.colorbar(im, ax=a[2], fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)
            for ax in a:
                ax.set_xticks([])
                ax.set_yticks([])
            wf[s] = (pid, *ctx.wavefront(r, data), dz)
        bar_panel(axs[i, 6], wf)
    spider = ("off" if cfg["spider_angle_deg"] is None
              else f"{cfg['spider_angle_deg']:.0f} deg")
    fig.suptitle(
        f"sim_donut_fits {variant}, {config}: binning {cfg['binning']}, spider {spider}, "
        f"Noll {cfg['noll'][config]}\nimages in electrons per binned pixel "
        f"({cfg['flux_e']:.0e} e- per donut, noise-free; chi2/dof at sky "
        f"{cfg['sky_e_per_px']:.0f} e- per 10 um pixel); row = CCS y, column = CCS x; "
        "l1 = fraction of flux misplaced\nright: OPD (fitted wavefront) per fitted term "
        "against the batoid intrinsic (bars); error bars 1-sigma photon noise at that flux "
        "and sky, from the fit Jacobian", size=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def bar_panel(ax, wf):
    """Intrinsic as bars, OPD per side as points with 1-sigma photon-noise errors."""
    noll = wf["extra"][1]
    x = np.arange(len(noll))
    same = wf["extra"][0] == wf["intra"][0]
    if same:
        ax.bar(x, wf["extra"][4], 0.7, color="0.8", label="intrinsic")
    for k, (s, colour) in enumerate(zip(SIDES, ("tab:blue", "tab:orange"))):
        pid, _, opd, sig, intr, dz = wf[s]
        if not same:
            ax.bar(x + (k - 0.5) * 0.4, intr, 0.38, color=("0.8", "0.65")[k],
                   label=f"intrinsic {s} ({pid})")
        ax.errorbar(x + (k - 0.5) * 0.2, opd, yerr=sig, fmt="o", ms=2.5, lw=0.8,
                    color=colour,
                    label=f"OPD {s}: |OPD - intrinsic| {dz:.3f} um, median 1-sigma "
                          f"{np.median(sig):.4f} um")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(x, [str(j) for j in noll], size=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Noll index", size=7)
    ax.set_ylabel("Zernike (um of wavefront, CCS)", size=7)
    ax.legend(fontsize=5.5, loc="best")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="default: every variant with a current build")
    ap.add_argument("--label", default="fit_examples")
    a = ap.parse_args()
    variants = a.variants or [v for v in VARIANTS if _has_current(v)]
    run = study_run("aos/v4speed_danish_closure", a.label)
    with PdfPages(run.path / "fit_examples.pdf") as pdf:
        for v in variants:
            df = sim_donut_fits.load(v)
            cfg = sim_donut_fits.variant_config(v)
            ctx = Context(cfg)
            for config in CONFIGS:
                page(pdf, v, config, df, cfg, ctx)
            print(f"{v} done", flush=True)
    print(f"wrote {run.path / 'fit_examples.pdf'}")


def _has_current(variant):
    try:
        catalog.manifest("sim_donut_fits", variant)
        return True
    except catalog.ProductNotFound:
        return False


if __name__ == "__main__":
    main()
