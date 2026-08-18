#!/usr/bin/env python
"""Side-by-side focal-plane Z5-Z8 maps: batoid camera-gravity vs the measured
MIW (OCS), before and after the 50/34 AOS correction.

For each pupil Zernike Z5,Z6,Z7,Z8 (astig/coma), one row of focal-plane maps:

  RAW figure       : [MIW OCS] [batoid design] [design+gravity el70] [el45] [el30]
  AFTER-50/34 fig  : same, but each batoid map has its 50/34-controllable part
                     removed (compared to the measured MIW, which is already the
                     post-correction residual).

The 50/34 correction removes the projection of the wavefront onto the top-34
v-modes of the geom-normalized OFC double-Zernike sensitivity SVD.  It REUSES
the aos/USDF code ``lsst.ts.intrinsic.wavefront.ofc_svd.build_ofc_svd`` (needs
the LSST stack).  Note the OFC DZ correction spans field Noll k<=6 (field radial
order <=2), so a high-field-order (n=3-5) MIW pattern is uncorrectable by
construction and survives the 50/34 subtraction.

Runs fully on the USDF RSP.  On a laptop (no LSST stack) the RAW figure is
produced and the AFTER-50/34 figure is skipped with a message.

Output: ../output/camera_gravity/camera_gravity_maps_{raw,corr}_{rb tag}.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

import camera_gravity as cg

OUT = Path(__file__).resolve().parents[1] / "output" / "camera_gravity"
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_MIW = (Path(__file__).resolve().parents[1] / "output" /
               "fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x" /
               "pathA_50_34_i_5rot" / "intrinsic_split_maps.parquet")

# Zernikes kept for the OFC wavefront correction: Z4..Z26 excl Z20,Z21
ZK_NOLL = [z for z in range(4, 27) if z not in (20, 21)]
NOLLS = (5, 6, 7, 8)


def load_miw(path):
    m = pd.read_parquet(path)
    good = np.isfinite(m["Z5_OCS"].values)
    return m[good].reset_index(drop=True)


def design_and_gravity_dz(band, elevations, include_rb):
    """Return dz_design and {el: dz_total(design+gravity)} in WAVES."""
    b = cg.base_builder(band)
    wl = cg.WL[band] * 1e-9
    dz_design = cg.C.double_zernike(b.build(), cg.FIELD_RAD, wl)
    dz_tot = {}
    for el in elevations:
        dchg, _ = cg.gravity_dz(b, el, band, 0.0, include_rb, dz_design)
        dz_tot[el] = dz_design + dchg
    return dz_design, dz_tot


def controllable_removed(dz_waves, band):
    """Return dz with its 50/34-controllable part removed (WAVES).

    Reuses lsst.ts.intrinsic.wavefront.ofc_svd.build_ofc_svd (LSST stack).
    """
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd
    svd = build_ofc_svd(list(ZK_NOLL), k_min=1, k_max=6, n_keep=34, n_dof=None)
    wl_um = cg.WL[band] * 1e-9 * 1e6
    # extract W (um) on the svd kj grid; k is field Noll, j is pupil Noll
    W = np.array([[dz_waves[k, j] * wl_um for (k, j) in svd.kj_grid]])
    A = svd.project_amplitudes(W)               # (1, 34)
    W_ctrl = (A @ svd.U_eff.T)[0]               # controllable part (um)
    dz_res = dz_waves.copy()
    for (k, j), w_full, w_ctrl in zip(svd.kj_grid, W[0], W_ctrl):
        dz_res[k, j] = (w_full - w_ctrl) / wl_um  # back to waves
    return dz_res


def _panel(ax, thx, thy, v, title, vmax):
    sc = ax.scatter(thx, thy, c=v, s=6, cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return sc


def make_figure(miw, cols, band, title, out_pdf):
    """cols = list of (label, dz_or_None); None => the measured MIW column."""
    thx, thy = miw["thx_deg"].values, miw["thy_deg"].values
    with PdfPages(str(out_pdf)) as pdf:
        fig, axes = plt.subplots(len(NOLLS), len(cols),
                                 figsize=(2.5 * len(cols), 2.5 * len(NOLLS)))
        for ri, j in enumerate(NOLLS):
            # common colour scale per row from the MIW amplitude
            vmax = np.nanpercentile(np.abs(miw[f"Z{j}_OCS"].values), 98) or 0.05
            for ci, (lab, dz) in enumerate(cols):
                ax = axes[ri, ci]
                if dz is None:
                    v = miw[f"Z{j}_OCS"].values
                else:
                    v = cg.dz_to_maps(dz, thx, thy, band, nolls=(j,))[j]
                sc = _panel(ax, thx, thy, v, f"{lab}\nZ{j} rms={np.nanstd(v):.3f}", vmax)
                if ci == 0:
                    ax.set_ylabel(f"Z{j}", fontsize=10)
            fig.colorbar(sc, ax=axes[ri, :].tolist(), shrink=0.6, pad=0.01)
        fig.suptitle(title, fontsize=11)
        pdf.savefig(fig); plt.close(fig)
    print("wrote", out_pdf)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="i")
    ap.add_argument("--elevations", type=float, nargs="+", default=[70, 45, 30])
    ap.add_argument("--miw", default=str(DEFAULT_MIW))
    ap.add_argument("--rb", action="store_true", help="include RB rigid-body sag")
    ap.add_argument("--rings", type=int, default=11)
    ap.add_argument("--spokes", type=int, default=35)
    ap.add_argument("--nx", type=int, default=255)
    args = ap.parse_args()
    cg.C.RINGS, cg.C.SPOKES, cg.C.NX = args.rings, args.spokes, args.nx
    tag = "rb" if args.rb else "surf"

    miw = load_miw(args.miw)
    dz_design, dz_tot = design_and_gravity_dz(args.band, args.elevations, args.rb)

    # RAW figure
    cols = [("MIW OCS (measured)", None), ("batoid design", dz_design)]
    cols += [(f"design+gravity el{int(e)}", dz_tot[e]) for e in args.elevations]
    make_figure(miw, cols, args.band,
                f"Z5-Z8 focal-plane maps — RAW (before 50/34), {tag}, band {args.band}",
                OUT / f"camera_gravity_maps_raw_{tag}.pdf")

    # AFTER-50/34 figure (needs the LSST stack)
    try:
        dz_design_c = controllable_removed(dz_design, args.band)
        dz_tot_c = {e: controllable_removed(dz_tot[e], args.band) for e in args.elevations}
    except ModuleNotFoundError as e:
        print(f"[skip] 50/34 correction needs the LSST stack "
              f"(lsst.ts.intrinsic): {e}.  Run on the USDF RSP.")
        return
    cols = [("MIW OCS (measured)", None), ("design (50/34-corr)", dz_design_c)]
    cols += [(f"design+grav el{int(e)} (50/34-corr)", dz_tot_c[e]) for e in args.elevations]
    make_figure(miw, cols, args.band,
                f"Z5-Z8 maps — AFTER 50/34 correction, {tag}, band {args.band}",
                OUT / f"camera_gravity_maps_corr_{tag}.pdf")


if __name__ == "__main__":
    main()
