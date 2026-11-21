#!/usr/bin/env python
"""Danish model at the batoid wavefront against batoid-traced donuts: no wavefront fit.

For every ``sim_donuts`` stamp, Danish draws the donut from the ``batoid_intrinsic``
defocused transverse-aberration (TA) Zernikes (CCS, Noll 1-78, as ts_wep) with the v4_speed
obscuration file, at the sim's blur (1.0 arcsec) and centred on the chief ray. The
wavefront is never fitted, so the residual is the obscuration file against the batoid
pupil, plus whatever the Danish forward model itself leaves out.

Per stamp and Danish ``spider_angle`` (None and 0 deg):

- ``l1_nofit``: half the summed absolute difference of the two images, each normalised
  to unit sum; the fraction of the flux in the wrong place, dimensionless;
- ``pow_nofit``: residual power over image power, sum (sim - model)² / sum sim²,
  dimensionless (power, not amplitude);

- ``model_sum_ratio``: Danish model sum over its ``flux`` parameter, dimensionless (the
  example pages scale the no-fit model to the sim's sum);

and for ``spider_angle`` 0 deg a nuisance-only fit (flux, dx, dy, fwhm free, wavefront
fixed): ``l1_fit``, ``pow_fit``, ``dx_px``, ``dy_px`` (fitted shift beyond Danish's
half-pixel centring convention, pixels, CCS x and y), ``fwhm_fit_arcsec``.

The no-fit models and the fit start are drawn at -0.5 px in x and y: Danish centres an
odd stamp on the corner of the middle pixel, the sims on its centre. Danish's ``dx`` and
``dy`` are in arcsec.

Usage::

    python aos/v4speed_mask_vs_batoid/code/mask_vs_batoid.py
    python aos/v4speed_mask_vs_batoid/code/mask_vs_batoid.py --limit 20 --label smoke
"""
import argparse
import time

import numpy as np
import pandas as pd
import danish
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rubinwork.common import study_run
from rubinwork.donut_sim import optics
from rubinwork.products import batoid_intrinsic, catalog, sim_donuts

SIM_VARIANT = "v4_speed_mono"
SPIDER_ANGLES = (None, 0.0)       # deg; None leaves the obscuration file's spider unrotated
FWHM_MIN = 0.2                    # arcsec; guard against the huge-kernel step (donut_leakage)
FLUX = 1e5                        # sims are rescaled to this sum for the fit
DANISH_CENTRE_PX = 0.5            # Danish centres an odd stamp at +0.5 px in x and y (the
                                  # corner of the middle pixel); the sims, on the middle
                                  # pixel's centre. Measured 2026-11-21, danish 1.4.0.
CONFIG_ORDER = ("fam", "cwfs", "giant")
FIELD_LIMIT_DEG = 1.725           # field radius the AOS fits are restricted to
EXAMPLE_FROM_DEG = {"fam": 1.4, "giant": 1.4, "cwfs": None}   # None: the innermost point
EXAMPLE_COUNT = {"fam": 4, "giant": 4, "cwfs": 3}


def icfg_focal(icfg, config):
    """Focal length (m) of the configuration's ts_wep instrument."""
    key = (icfg["model"], config)
    if key not in _FOCAL:
        _FOCAL[key] = optics.instrument(icfg["model"], icfg["obscuration"],
                                        icfg["configs"][config]).focalLength
    return _FOCAL[key]


_FOCAL = {}


def factory(instr, spider_angle):
    return danish.DonutFactory(
        R_outer=instr.radius, R_inner=instr.radius * instr.obscuration,
        mask_params=instr.maskParams, focal_length=instr.focalLength,
        pixel_scale=instr.pixelSize, spider_angle=spider_angle)


def metrics(sim, mod):
    a, b = sim / sim.sum(), mod / mod.sum()
    return 0.5 * np.abs(a - b).sum(), ((a - b) ** 2).sum() / (a ** 2).sum()


def nuisance_fit(model, img, fwhm0, d0):
    """Flux, dx, dy (arcsec) and fwhm free from (sum, `d0`, `d0`, `fwhm0`); the
    wavefront fixed at the reference."""
    def chi(x, *a):
        if model.unpack_params(x)["fwhm"] < FWHM_MIN:
            return np.full(img.size, 1e10)
        return model.chi(x, *a)
    x0 = model.pack_params(flux=float(img.sum()), dx=d0, dy=d0, fwhm=fwhm0, bkg=[],
                           z_fit=[])
    r = least_squares(chi, jac=model.jac, x0=x0, args=(img, 1.0), xtol=1e-10, ftol=1e-10)
    p = model.unpack_params(r.x)
    best = model.model(flux=p["flux"], dx=p["dx"], dy=p["dy"], fwhm=p["fwhm"], z_fit=[])
    return p, best


def run(limit=None):
    index = sim_donuts.load(SIM_VARIANT)
    sman = catalog.manifest("sim_donuts", SIM_VARIANT)
    # The sims' intrinsic variant at its current build: the field points are the same in
    # every build, and the current one carries the TA tilts z2, z3.
    ivar = sman["inputs"]["batoid_intrinsic"].split("@")[0]
    iman = catalog.manifest("batoid_intrinsic", ivar)
    icfg = iman["config"]
    ta = batoid_intrinsic.load(ivar, iman["build"], kind="ta")
    cols = batoid_intrinsic.noll_columns(icfg["jmax"], "CCS")
    ta = ta.set_index(["band", "config", "side", "point_id"])
    if limit:
        index = index.iloc[::max(1, len(index) // limit)]
    index = index.assign(field_r_deg=np.hypot(index.thx_CCS_deg, index.thy_CCS_deg))
    wanted = set(examples(index))

    facs = {}
    rows, keep = [], {}
    t0 = time.time()
    for n, r in enumerate(index.itertuples()):
        key = (r.band, r.config)
        if key not in facs:
            instr = optics.instrument(icfg["model"], icfg["obscuration"],
                                      icfg["configs"][r.config])
            facs[key] = {sa: factory(instr, sa) for sa in SPIDER_ANGLES}
        sim, _ = sim_donuts.stamp(SIM_VARIANT, r.band, r.config, r.side, r.point_id)
        sim = sim * FLUX / sim.sum()
        z = np.zeros(icfg["jmax"] + 1)
        z[batoid_intrinsic.reader.JMIN:] = ta.loc[(r.band, r.config, r.side, r.point_id), cols].to_numpy(float) * 1e-6
        th = np.deg2rad([r.thx_CCS_deg, r.thy_CCS_deg])
        pix_arcsec = 3600 * np.rad2deg(r.pixel_m / icfg_focal(icfg, r.config))
        d0 = -DANISH_CENTRE_PX * pix_arcsec
        row = dict(band=r.band, config=r.config, side=r.side, point_id=r.point_id,
                   thx_CCS_deg=r.thx_CCS_deg, thy_CCS_deg=r.thy_CCS_deg,
                   field_r_deg=r.field_r_deg)
        for sa in SPIDER_ANGLES:
            m = danish.SingleDonutModel(facs[key][sa], z_ref=z, z_terms=(), thx=th[0],
                                        thy=th[1], npix=int(r.npix))
            mod = m.model(flux=FLUX, dx=d0, dy=d0, fwhm=r.fwhm_arcsec, z_fit=[])
            tag = "none" if sa is None else "0"
            row[f"l1_nofit_sp{tag}"], row[f"pow_nofit_sp{tag}"] = metrics(sim, mod)
            row[f"model_sum_ratio_sp{tag}"] = mod.sum() / FLUX
            mod = mod * FLUX / mod.sum()
            if sa == 0.0:
                p, best = nuisance_fit(m, sim, r.fwhm_arcsec, d0)
                row["l1_fit"], row["pow_fit"] = metrics(sim, best)
                row.update(dx_px=p["dx"] / pix_arcsec + DANISH_CENTRE_PX,
                           dy_px=p["dy"] / pix_arcsec + DANISH_CENTRE_PX,
                           fwhm_fit_arcsec=p["fwhm"],
                           flux_ratio=p["flux"] / FLUX)
                if (r.config, r.side, r.point_id) in wanted:
                    keep[(r.config, r.side, r.point_id)] = (sim, mod, best)
        rows.append(row)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(index)} done, {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows), keep, sman


def examples(df):
    """Per configuration and side: points spread in field radius up to `FIELD_LIMIT_DEG`.

    FAM and giant: 4 from 1.4 deg; CWFS: 3 from its innermost point. Each target radius
    takes the nearest unused point inside the limit.
    """
    out = []
    for (c, s), g in df.groupby(["config", "side"]):
        g = g[g.field_r_deg <= FIELD_LIMIT_DEG]
        lo = EXAMPLE_FROM_DEG[c] if EXAMPLE_FROM_DEG[c] is not None else g.field_r_deg.min()
        used = []
        for t in np.linspace(lo, FIELD_LIMIT_DEG, EXAMPLE_COUNT[c]):
            free = g[~g.point_id.isin(used)]
            used.append(free.iloc[int(np.argmin(np.abs(free.field_r_deg - t)))].point_id)
        for pid in g[g.point_id.isin(used)].sort_values("field_r_deg").point_id:
            out.append((c, s, pid))
    return out


def summary_table(df):
    df = df.assign(l1_fit_in=df.l1_fit.where(df.field_r_deg <= FIELD_LIMIT_DEG))
    agg = df.groupby(["config", "side"]).agg(
        n=("point_id", "size"),
        l1_nofit_spnone=("l1_nofit_spnone", "median"), l1_nofit_sp0=("l1_nofit_sp0", "median"),
        l1_fit=("l1_fit", "median"), l1_fit_max=("l1_fit", "max"),
        l1_fit_max_in=("l1_fit_in", "max"),
        pow_fit=("pow_fit", "median"),
        dx_px=("dx_px", "median"), dy_px=("dy_px", "median"),
        fwhm_fit=("fwhm_fit_arcsec", "median"))
    return agg.reindex(sorted(agg.index, key=lambda k: (CONFIG_ORDER.index(k[0]), k[1])))


def plots(df, keep, path, sman):
    agg = summary_table(df)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.95, "Danish v4_speed obscuration file vs batoid-traced donuts",
                 size=14, weight="bold")
        fig.text(0.04, 0.91, f"sim_donuts {SIM_VARIANT}@{sman['build']}; "
                 f"TA from batoid_intrinsic current; r band, rotator 0 deg, "
                 "Kolmogorov 1.0 arcsec FWHM", size=9)
        fig.text(0.04, 0.86, "Metrics, with $s_i$ the batoid stamp and $m_i$ the Danish "
                 "model in pixel $i$, each normalised to unit sum "
                 "($\\tilde s_i = s_i / \\sum_k s_k$, $\\tilde m_i = m_i / \\sum_k m_k$):",
                 size=9.5)
        fig.text(0.08, 0.80, r"$l_1 = \frac{1}{2}\sum_i |\tilde s_i - \tilde m_i|$"
                 "    fraction of the flux in the wrong place (dimensionless, 0 to 1)",
                 size=11)
        fig.text(0.08, 0.745, r"$\mathrm{pow} = \sum_i (\tilde s_i - \tilde m_i)^2 "
                 r"\, / \, \sum_i \tilde s_i^2$"
                 "    residual power over image power (dimensionless; power, not "
                 "amplitude)", size=11)
        lines = ["Medians per configuration and side, all points; 'max in' is over field "
                 f"radius <= {FIELD_LIMIT_DEG} deg.",
                 "nofit: Danish at the TA wavefront (Noll 1-78), on the chief ray, fwhm 1.0 "
                 "arcsec; fit: flux, dx, dy, fwhm free, wavefront fixed (spider_angle 0 deg).",
                 "dx, dy: fitted shift beyond Danish's half-pixel centring (px, CCS).", "",
                 f"{'config':6s} {'side':5s} {'n':>4s} {'l1 nofit':>9s} {'l1 nofit':>9s} "
                 f"{'l1 fit':>7s} {'l1 fit':>7s} {'l1 fit':>7s} {'pow fit':>8s} {'dx':>6s} "
                 f"{'dy':>6s} {'fwhm':>6s}",
                 f"{'':6s} {'':5s} {'':>4s} {'sp None':>9s} {'sp 0':>9s} {'median':>7s} "
                 f"{'max':>7s} {'max in':>7s} {'median':>8s} {'(px)':>6s} {'(px)':>6s} "
                 f"{'(as)':>6s}"]
        for (c, s), a in agg.iterrows():
            lines.append(f"{c:6s} {s:5s} {int(a.n):4d} {a.l1_nofit_spnone:9.4f} "
                         f"{a.l1_nofit_sp0:9.4f} {a.l1_fit:7.4f} {a.l1_fit_max:7.4f} "
                         f"{a.l1_fit_max_in:7.4f} {a.pow_fit:8.5f} {a.dx_px:6.2f} "
                         f"{a.dy_px:6.2f} {a.fwhm_fit:6.3f}")
        lines += ["", "Blur: both images use a galsim Kolmogorov of fixed FWHM, no "
                  "wavelength scaling. The sims convolve it (no pixel) on a",
                  "3x oversampled ray deposit and sum to 10 um pixels; Danish draws it with "
                  "galsim's default pixel convolution.",
                  "Wavelength: the sims trace at 619.4 nm (r), which enters through the "
                  "glass indices; the TA reference is batoid at 619.4 nm.",
                  "Danish's model is geometric given Zernikes in m, so it has no "
                  "wavelength of its own (bandpass_filter None)."]
        fig.text(0.04, 0.69, "\n".join(lines), family="monospace", size=8, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axs = plt.subplots(2, 3, figsize=(11, 8))
        for ax, col, lab in zip(axs.flat,
                                ["l1_nofit_spnone", "l1_nofit_sp0", "l1_fit", "pow_fit",
                                 "fwhm_fit_arcsec", "flux_ratio"],
                                ["l1, no fit, spider None", "l1, no fit, spider 0 deg",
                                 "l1, nuisance fit", "residual power / image power, fit",
                                 "fitted FWHM (arcsec)", "fitted flux / sim flux"]):
            for (c, s), g in df.groupby(["config", "side"]):
                ax.plot(g.field_r_deg, g[col], ".", ms=3, label=f"{c} {s}")
            ax.axvline(FIELD_LIMIT_DEG, color="k", lw=0.8, ls="--")
            ax.set_xlabel("field radius (deg)")
            ax.set_title(lab, size=9)
        axs[0, 0].legend(fontsize=7)
        axs[0, 0].text(FIELD_LIMIT_DEG, axs[0, 0].get_ylim()[1], f" {FIELD_LIMIT_DEG} deg",
                       size=7, va="top", ha="right", rotation=90)
        fig.suptitle("Agreement against field radius (dimensionless unless stated)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for cfg in CONFIG_ORDER:
            g = df[df.config == cfg]
            if g.empty:
                continue
            fig, axs = plt.subplots(1, 3, figsize=(11, 4.2))
            for ax, s in zip(axs[:2], ("extra", "intra")):
                h = g[g.side == s]
                sc = ax.scatter(h.thx_CCS_deg, h.thy_CCS_deg, c=h.l1_fit, s=14, cmap="viridis")
                plt.colorbar(sc, ax=ax, label="l1 after nuisance fit (dimensionless)")
                ax.set_aspect("equal")
                ax.set_xlabel("thx CCS (deg)")
                ax.set_ylabel("thy CCS (deg)")
                ax.set_title(f"{cfg} {s}", size=9)
            q = axs[2].quiver(g.thx_CCS_deg, g.thy_CCS_deg, g.dx_px, g.dy_px, angles="xy")
            axs[2].quiverkey(q, 0.8, 1.03, 1.0, "1 px", labelpos="E")
            axs[2].set_aspect("equal")
            axs[2].set_title(f"{cfg}: fitted centroid shift (px, CCS)", size=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for c, s, pid in examples(df):
            sim, mod, best = keep[(c, s, pid)]
            r = df[(df.config == c) & (df.side == s) & (df.point_id == pid)].iloc[0]
            fig, axs = plt.subplots(1, 4, figsize=(11, 3.4))
            vmax = sim.max()
            for ax, im, t in zip(axs, [sim, mod, sim - mod, sim - best],
                                 ["batoid", "Danish, no fit (scaled to sim sum)",
                                  "batoid - Danish, no fit",
                                  "batoid - Danish, nuisance fit"]):
                lim = (0, vmax) if not t.startswith("batoid -") else (-0.2 * vmax,
                                                                      0.2 * vmax)
                ax.imshow(im, origin="lower", vmin=lim[0], vmax=lim[1],
                          cmap="gray" if lim[0] == 0 else "RdBu_r")
                ax.set_title(t, size=8)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{c} {s} {pid}: field ({r.thx_CCS_deg:.2f}, {r.thy_CCS_deg:.2f}) "
                         f"deg CCS; l1 no fit {r.l1_nofit_sp0:.4f}, after fit {r.l1_fit:.4f} "
                         "(dimensionless); residuals at ±20% of the sim peak", size=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="about this many stamps")
    ap.add_argument("--label", default="ta_model")
    a = ap.parse_args()
    run_ = study_run("aos/v4speed_mask_vs_batoid", a.label)
    df, keep, sman = run(a.limit)
    df.to_parquet(run_.path / "metrics.parquet", index=False)
    plots(df, keep, run_.path / "mask_vs_batoid.pdf", sman)
    print(summary_table(df).round(4).to_string())
    print(f"wrote {run_.path}")


if __name__ == "__main__":
    main()
