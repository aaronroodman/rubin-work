#!/usr/bin/env python
"""Per-element edge offsets, Danish obscuration file against batoid, on the stop plane.

For FAM and giant, both sides, at the ``batoid_intrinsic`` edge-grid radii (1.400 to
1.725 deg) and a few azimuths, and for CWFS at its 6 x 3 points on each corner half-chip
(each on its own side): a 512 x 512 stop-plane ray grid is traced through the
defocused v4_speed batoid model and tested against Danish's mask
(``rubinwork.donut_sim.pupil.compare``, spider_angle 0 deg). Each mismatched ray goes to
the nearest Danish edge. Per edge, ``offset_mm`` = (batoid-only area - Danish-only area)
/ edge length: positive where Danish masks more than batoid (mm, stop plane).

Writes ``pupil_edges.parquet`` and ``pupil_edges.pdf`` to the run directory.

Usage::

    python aos/v4speed_mask_vs_batoid/code/pupil_edges.py
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rubinwork.common import study_run
from rubinwork.donut_sim import optics, pupil
from rubinwork.products import batoid_intrinsic, catalog

CONFIGS = ("fam", "giant")         # on edge-grid radii x AZIMUTHS_DEG
ALL_CONFIGS = ("fam", "cwfs", "giant")
FIELD_LIMIT_DEG = 1.725
CWFS_BIN_DEG = 0.02                # CWFS points are binned in field radius for the medians
AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0)
NX = 512
MIN_AREA_M2 = 1e-3      # edges with less mismatched area than this are not plotted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="pupil_edges")
    a = ap.parse_args()
    run = study_run("aos/v4speed_mask_vs_batoid", a.label)
    iman = catalog.manifest("batoid_intrinsic", "v4_speed")
    icfg = iman["config"]
    e = icfg["grids"]["edge"]
    radii = np.arange(e["r_min_deg"], e["r_max_deg"] + 0.5 * e["dr_deg"], e["dr_deg"])
    ta = batoid_intrinsic.load("v4_speed", iman["build"], kind="ta", config="cwfs", band="r")
    rows, totals = [], []
    for config in ALL_CONFIGS:
        instr = optics.instrument(icfg["model"], icfg["obscuration"], icfg["configs"][config])
        base = optics.rotate_telescope(instr.getBatoidModel("r"), icfg["rotator_angle_deg"])
        for side in optics.SIDES:
            optic = optics.defocused(instr, base, side)
            if config == "cwfs":
                pts = [(row.point_id, row.thx_CCS_deg, row.thy_CCS_deg)
                       for row in ta[ta.side == side].itertuples()]
            else:
                pts = [(None, r * np.cos(np.deg2rad(az)), r * np.sin(np.deg2rad(az)))
                       for az in AZIMUTHS_DEG for r in radii]
            for pid, x, y in pts:
                tot, per = pupil.compare(optic, instr.maskParams, instr.radius, x, y,
                                         instr.wavelength["r"], nx=NX, spider_angle=0.0)
                r = float(np.hypot(x, y))
                key = dict(config=config, side=side, point_id=pid,
                           az_deg=float(np.rad2deg(np.arctan2(y, x))) % 360,
                           field_r_deg=r if config == "cwfs" else round(r, 3))
                totals.append({**key, **tot})
                rows += [{**key, **p} for p in per]
            print(f"{config} {side} done", flush=True)
    per = pd.DataFrame(rows)
    tot = pd.DataFrame(totals)
    tot["mismatch_frac"] = (tot.area_batoid_only_m2 + tot.area_danish_only_m2) / tot.area_batoid_m2
    per.to_parquet(run.path / "pupil_edges.parquet", index=False)
    tot.to_parquet(run.path / "pupil_totals.parquet", index=False)

    per["element"] = per.edge.str.replace(r"\.\d+$", "", regex=True)
    big = per.groupby("element").apply(
        lambda g: (g.batoid_only_m2 + g.danish_only_m2).max(), include_groups=False)
    shown = sorted(big[big > MIN_AREA_M2].index)
    per["rbin"] = np.where(per.config == "cwfs",
                           (per.field_r_deg / CWFS_BIN_DEG).round() * CWFS_BIN_DEG,
                           per.field_r_deg)
    with PdfPages(run.path / "pupil_edges.pdf") as pdf:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
        for ax, config in zip(axs, ALL_CONFIGS):
            g0 = tot[tot.config == config]
            if config == "cwfs":
                for side, g in g0.groupby("side"):
                    ax.plot(g.field_r_deg, 100 * g.mismatch_frac, ".", ms=4,
                            label=f"{side} ({'SW0' if side == 'extra' else 'SW1'})")
            else:
                for (side, az), g in g0.groupby(["side", "az_deg"]):
                    ax.plot(g.field_r_deg, 100 * g.mismatch_frac, ".-", ms=3,
                            label=f"{side} az {az:.0f} deg")
            ax.axvline(FIELD_LIMIT_DEG, color="k", lw=0.8, ls="--")
            ax.set_xlabel("field radius (deg)")
            ax.set_title(f"{config}: batoid vs Danish pupil, stop plane", size=9)
            ax.legend(fontsize=6)
        axs[0].set_ylabel("mismatched pupil area / batoid clear area (%)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        for config in ALL_CONFIGS:
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
            for ax, side in zip(axs, optics.SIDES):
                g = per[(per.config == config) & (per.side == side)
                        & per.element.isin(shown) & (per.length_m > 0.1)]
                for el, h in g.groupby("element"):
                    m = h.groupby("rbin").offset_mm.median()
                    ax.plot(m.index, m.values, ".-", ms=4, label=el)
                ax.axhline(0, color="k", lw=0.5)
                ax.axvline(FIELD_LIMIT_DEG, color="k", lw=0.8, ls="--")
                ax.set_xlabel("field radius (deg)")
                over = ("points in 0.02 deg radius bins" if config == "cwfs" else "azimuth")
                ax.set_title(f"{config} {side}: median edge offset over {over}", size=9)
            axs[0].set_ylabel("edge offset (mm, stop plane; + Danish masks more)")
            axs[1].legend(fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    s = tot.groupby(["config", "side"]).mismatch_frac.agg(["median", "max"])
    print((100 * s).round(3).rename(columns=lambda c: f"{c} (%)").to_string())
    big_off = per[per.length_m > 0.1].groupby(["config", "side", "element"]).offset_mm.agg(
        lambda x: x.abs().max())
    print("max |edge offset| (mm) per element, edges longer than 0.1 m:")
    print(big_off[big_off > 1].round(1).to_string())
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
