#!/usr/bin/env python
"""Simulated against real PSF for one or more visits: guider centroid motion and the
science-CCD FWHM and ellipticity across the field.

Sim: `atmo_sim_moments` (guider tables and ``ccd_psf``) and `atmo_sim` (truth).
Data: `guider_moments` (guider tables) and ConsDB ``ccdvisit1_quicklook`` (science CCDs).

The simulated guide stamps are drawn in each sensor's real ROI and frame, so sim and
data guider moments are in the same DVCS frame and compare 1:1, Q1 and Q2 included.
DVCS axes lie along the cameraGeom FOCAL_PLANE axes for all 8 guiders, so the motion
ellipticity e_motion = |Q| / T and its position angle are in the FOCAL_PLANE frame and
can be drawn as whiskers at the guider positions. The
science-CCD moments are in each CCD's pixel frame on both sides; for the focal-plane
maps they are rotated into the cameraGeom FOCAL_PLANE frame with each CCD's
PIXELS-to-FOCAL_PLANE Jacobian.

    python guider/guider_atmo_sim/code/compare_visit.py --day-obs 20260706 --seq-num 688
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.products import atmo_sim, atmo_sim_moments, guider_moments  # noqa: E402

SIM_COLOR, DATA_COLOR = "#2a78d6", "#eb6834"
FWHM_PER_SIGMA = 2.3548
MOTION = ("Mxx_motion", "Myy_motion", "Mxy_motion", "T_motion", "Q1_motion", "Q2_motion")


def guider_table(day_obs, seq_num, gm_variant):
    """Sim and data guider moments side by side, one row per guide sensor."""
    sim = atmo_sim_moments.load_table("moments", day_obs, seq_num)
    data = guider_moments.load_table("moments", day_obs, variant=gm_variant)
    data = data[data.seqNum == seq_num]
    cols = ["detector", "T_stamp", *MOTION]
    out = sim[cols + ["xfp", "yfp"]].merge(data[cols], on="detector", suffixes=("_sim", "_data"))
    for side in ("sim", "data"):
        q1, q2 = out[f"Q1_motion_{side}"], out[f"Q2_motion_{side}"]
        out[f"e_motion_{side}"] = np.hypot(q1, q2) / out[f"T_motion_{side}"]
        out[f"pa_motion_{side}"] = 0.5 * np.degrees(np.arctan2(q2, q1))
    out.insert(0, "seq_num", seq_num)
    return out


def ccd_table(day_obs, seq_num):
    """Sim and ConsDB science-CCD moments, one row per CCD, CCD pixel frame (pix**2)."""
    from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
    from lsst.obs.lsst import LsstCam

    from rubinwork.common.psf_moments_consdb import PSFMomentsTable

    sim = atmo_sim_moments.load_table("ccd_psf", day_obs, seq_num)
    truth = atmo_sim.load_truth(day_obs, seq_num)
    truth = truth[truth.kind == "science"].set_index("detector")
    dets = sorted(sim.detector)
    cdb = PSFMomentsTable(day_obs, detectors=dets, seq_min=seq_num, seq_max=seq_num).fetch()
    camera = LsstCam.getCamera()
    rows = []
    for r in sim.itertuples():
        d = r.detector
        det = camera[d]
        scale = np.sqrt(abs(np.linalg.det(np.reshape(truth.loc[d, "jac"], (2, 2)))))
        c = det.getBBox().getCenter()
        fp = det.getTransform(PIXELS, FOCAL_PLANE).getMapping().applyForward(
            np.array([[c.x, c.x + 1, c.x], [c.y, c.y, c.y + 1]]))
        jac = np.column_stack([fp[:, 1] - fp[:, 0], fp[:, 2] - fp[:, 0]])
        jac /= np.sqrt(abs(np.linalg.det(jac)))
        row = dict(seq_num=seq_num, detector=d, x_fp=fp[0, 0], y_fp=fp[1, 0], scale=scale)
        sides = {"sim": (r.psf_ixx, r.psf_iyy, r.psf_ixy)}
        key = f"psf_ixx_{d}"
        if len(cdb) and key in cdb.colnames:
            sides["data"] = tuple(float(cdb[0][f"psf_{m}_{d}"]) for m in ("ixx", "iyy", "ixy"))
        for side, (ixx, iyy, ixy) in sides.items():
            m = jac @ np.array([[ixx, ixy], [ixy, iyy]]) @ jac.T
            T = ixx + iyy
            sigma = (ixx * iyy - ixy**2) ** 0.25
            row.update({f"fwhm_{side}": FWHM_PER_SIGMA * sigma * scale,
                        f"e1_{side}": (m[0, 0] - m[1, 1]) / T,
                        f"e2_{side}": 2 * m[0, 1] / T})
        rows.append(row)
    df = pd.DataFrame(rows)
    for side in ("sim", "data"):
        if f"e1_{side}" in df:
            df[f"e_{side}"] = np.hypot(df[f"e1_{side}"], df[f"e2_{side}"])
    return df


def _whiskers(ax, df, side, color, scale_mm):
    e, phi = df[f"e_{side}"], 0.5 * np.arctan2(df[f"e2_{side}"], df[f"e1_{side}"])
    dx, dy = e * np.cos(phi) * scale_mm, e * np.sin(phi) * scale_mm
    ax.quiver(df.x_fp, df.y_fp, dx, dy, color=color, angles="xy", scale_units="xy",
              scale=1, headwidth=0, headlength=0, headaxislength=0, pivot="middle",
              width=0.004)
    ax.set_aspect("equal")
    ax.set_xlabel("focal plane x (mm)")
    ax.set_ylabel("focal plane y (mm)")


def plot_visit(pdf, g, c, atm, day_obs, seq_num):
    """Pages for one visit: summary, guider motion, science-CCD maps."""
    fig = plt.figure(figsize=(8.5, 11))
    lines = [f"day_obs {day_obs} seq {seq_num}: atmo_sim v1 against data",
             "",
             f"band {atm['band']}, alt {atm['alt_deg']:.1f} deg, rotator {atm['rot_deg']:.2f} deg,"
             f" ERA5 step {atm['era5_time'][:16]} UTC",
             f"r0_500 {atm['r0_500_m']:.3f} m, L0 {atm['L0_m']:.1f} m, target guider FWHM "
             f"{atm['target_fwhm_asec']:.3f} arcsec",
             f"layer distance (km): {', '.join(f'{h:.1f}' for h in atm['altitude_km'])}",
             f"layer weight (fraction of Cn2): {', '.join(f'{w:.3f}' for w in atm['weights'])}",
             f"layer wind (m/s): {', '.join(f'{v:.1f}' for v in atm['speed_mps'])}",
             f"screens {atm['screen_size_m']:.0f} m built, {atm['screen_need_m']:.0f} m needed",
             "",
             "Guider, median over sensors (arcsec^2):"]
    for k in ("T_stamp", "T_motion", "Mxx_motion", "Myy_motion"):
        s, d = g[f"{k}_sim"].median(), g[f"{k}_data"].median()
        lines.append(f"  {k}: sim {s:.4f}, data {d:.4f}, sim/data {s / d:.2f}")
    lines += ["", "Guider motion ellipticity |Q|/T (dimensionless), position angle (deg,",
              "FOCAL_PLANE frame, from +x toward +y):",
              "  sensor     e sim  e data   PA sim  PA data"]
    for r in g.itertuples():
        lines.append(f"  {r.detector}  {r.e_motion_sim:6.3f}  {r.e_motion_data:6.3f}  "
                     f"{r.pa_motion_sim:7.1f}  {r.pa_motion_data:7.1f}")
    lines += ["", "Science CCDs, median over CCDs:"]
    for k, u in (("fwhm", "arcsec"), ("e", "dimensionless")):
        s = c[f"{k}_sim"].median()
        d = c[f"{k}_data"].median() if f"{k}_data" in c else np.nan
        lines.append(f"  {k} ({u}): sim {s:.3f}, ConsDB {d:.3f}")
    fig.text(0.07, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    x = np.arange(len(g))
    for ax, k in zip(axs.flat, ("T_motion", "Mxx_motion", "Myy_motion", "Mxy_motion")):
        ax.bar(x - 0.2, g[f"{k}_sim"], 0.38, color=SIM_COLOR, label="sim")
        ax.bar(x + 0.2, g[f"{k}_data"], 0.38, color=DATA_COLOR, label="data")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(x, g.detector, rotation=45, fontsize=8)
        ax.set_ylabel(f"{k} (arcsec$^2$)")
        ax.grid(axis="y", color="0.9")
    axs[0, 0].legend(frameon=False)
    fig.suptitle(f"{day_obs} seq {seq_num}: guider centroid-motion second moments, "
                 "DVCS frame (= FOCAL_PLANE axes)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.8))
    for ax, side, col in zip(axs, ("sim", "data"), (SIM_COLOR, DATA_COLOR)):
        e, phi = g[f"e_motion_{side}"], np.radians(g[f"pa_motion_{side}"])
        ax.quiver(g.xfp, g.yfp, e * np.cos(phi) * 200, e * np.sin(phi) * 200, color=col,
                  angles="xy", scale_units="xy", scale=1, headwidth=0, headlength=0,
                  headaxislength=0, pivot="middle", width=0.006)
        for r in g.itertuples():
            ax.annotate(f"{getattr(r, f'e_motion_{side}'):.2f}", (r.xfp, r.yfp),
                        textcoords="offset points", xytext=(0, -16), ha="center",
                        fontsize=8, color="0.3")
        ax.set_xlim(-330, 330)
        ax.set_ylim(-330, 330)
        ax.set_aspect("equal")
        ax.set_xlabel("focal plane x (mm)")
        ax.set_ylabel("focal plane y (mm)")
        ax.set_title(f"motion ellipticity |Q|/T, {side} (0.5 = 100 mm)")
    fig.suptitle(f"{day_obs} seq {seq_num}: guider centroid-motion ellipticity, "
                 "FOCAL_PLANE frame")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    if "fwhm_data" not in c:
        return
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    lo, hi = np.nanpercentile(np.r_[c.fwhm_sim, c.fwhm_data], [1, 99])
    for ax, side in zip(axs[0], ("sim", "data")):
        sc = ax.scatter(c.x_fp, c.y_fp, c=c[f"fwhm_{side}"], s=60, marker="s",
                        cmap="Blues", vmin=lo, vmax=hi)
        ax.set_aspect("equal")
        ax.set_title(f"FWHM, {'sim (atmosphere only)' if side == 'sim' else 'ConsDB'}")
        fig.colorbar(sc, ax=ax, label="FWHM (arcsec)")
    for ax, side, col in zip(axs[1], ("sim", "data"), (SIM_COLOR, DATA_COLOR)):
        _whiskers(ax, c, side, col, scale_mm=400)
        ax.set_title(f"ellipticity, {'sim' if side == 'sim' else 'ConsDB'} "
                     "(FOCAL_PLANE frame; 0.05 = 20 mm)")
    fig.suptitle(f"{day_obs} seq {seq_num}: science-CCD PSF")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, nargs="+", required=True)
    p.add_argument("--guider-moments-variant", default="v4")
    p.add_argument("--label", default=None)
    args = p.parse_args(argv)
    label = args.label or f"visits_{args.day_obs}_" + "_".join(map(str, args.seq_num))
    run = study_run("guider/guider_atmo_sim", label)
    gs, cs = [], []
    with PdfPages(run.path / f"{label}.pdf") as pdf:
        for seq in args.seq_num:
            g = guider_table(args.day_obs, seq, args.guider_moments_variant)
            c = ccd_table(args.day_obs, seq)
            plot_visit(pdf, g, c, atmo_sim.load_atmosphere(args.day_obs, seq),
                       args.day_obs, seq)
            gs.append(g)
            cs.append(c)
    g, c = pd.concat(gs), pd.concat(cs)
    g.to_csv(run.path / "guider_motion.csv", index=False)
    c.to_csv(run.path / "ccd_psf.csv", index=False)
    pd.set_option("display.width", 160)
    print(g[["seq_num", "detector", "T_stamp_sim", "T_stamp_data", "T_motion_sim",
             "T_motion_data", "e_motion_sim", "e_motion_data", "pa_motion_sim",
             "pa_motion_data"]].round(3).to_string(index=False))
    for k in ("T_stamp", "T_motion", "Mxx_motion", "Myy_motion"):
        print(f"{k}: median sim/data {np.median(g[f'{k}_sim'] / g[f'{k}_data']):.2f} "
              "(dimensionless)")
    print(f"science CCDs: median FWHM sim {c.fwhm_sim.median():.3f} arcsec, ConsDB "
          f"{c.fwhm_data.median():.3f} arcsec; median e sim {c.e_sim.median():.3f}, "
          f"ConsDB {c.e_data.median():.3f}")
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
