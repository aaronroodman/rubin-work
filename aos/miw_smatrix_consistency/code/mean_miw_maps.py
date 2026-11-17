"""Bin-mean MIW maps beside the Batoid intrinsic, one page per Zernike.

For each ``miw`` build, three means over rotator bins: all nine bins, the five 5rot bins,
and the other four (night 20260409). Each is shown on the field cells present in all
nine bins, beside the Batoid intrinsic (``zk_tabulated``) on the same cells. MIW in µm
of wavefront, OCS.

Writes ``mean_miw_maps.pdf`` and ``mean_minus_batoid.csv`` into the run directory.

Usage::

    python aos/miw_smatrix_consistency/code/mean_miw_maps.py \\
        --miw fsm-sparse_no22=d13s-A_50_34_i_fsm_sparse_no22@20261116b \\
        --miw fsm-dense21=d13-A_50_34_i_fsm_dense21@20261116b
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from compare_miw import ROT5, load_bins  # noqa: E402

MEANS = ["all 9 bins", "5rot bins", "other 4 bins"]


def bin_means(t):
    """Per cell and j: the three bin-means and Batoid, on cells present in all 9 bins."""
    n_bins = t.rot_bin.nunique()
    t = t[t.groupby(["thx", "thy", "j"]).rot_bin.transform("nunique") == n_bins]
    sel = {"all 9 bins": t, "5rot bins": t[t.rot_bin.isin(ROT5)],
           "other 4 bins": t[~t.rot_bin.isin(ROT5)]}
    out = pd.concat({k: s.groupby(["j", "thx", "thy"]).zk.mean() for k, s in sel.items()},
                    axis=1)
    out["Batoid"] = t.groupby(["j", "thx", "thy"]).batoid.mean()
    return out.reset_index()


def to_image(c, col, xs, ys):
    img = np.full((len(ys), len(xs)), np.nan)
    img[np.searchsorted(ys, c.thy), np.searchsorted(xs, c.thx)] = c[col]
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--miw", action="append", required=True,
                    metavar="LABEL=VARIANT@BUILD")
    ap.add_argument("--label", default="mean_miw_maps")
    ap.add_argument("--min-donuts", type=int, default=1,
                    help="drop grid cells with fewer donuts (count)")
    args = ap.parse_args()
    run = study_run("aos/miw_smatrix_consistency", args.label)

    means = {}
    for spec in args.miw:
        lab, _, vb = spec.partition("=")
        v, _, b = vb.partition("@")
        means[lab] = bin_means(load_bins(v, b, args.min_donuts))
    labs = list(means)
    allc = pd.concat(means.values())
    xs, ys = np.unique(allc.thx), np.unique(allc.thy)
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    js = sorted(set(allc.j))

    rms = []
    for lab, m in means.items():
        for j, g in m.groupby("j"):
            rms.append(dict(miw=lab, j=j, n_cells=len(g),
                            rms_batoid_um=np.sqrt(np.mean(g.Batoid ** 2)),
                            **{f"rms_{k.split()[0]}_minus_batoid_um":
                               np.sqrt(np.mean((g[k] - g.Batoid) ** 2)) for k in MEANS}))
    rms = pd.DataFrame(rms)
    rms.to_csv(run.path / "mean_minus_batoid.csv", index=False)

    pdf_path = run.path / "mean_miw_maps.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(len(labs), 1, figsize=(11, 4 * len(labs)), squeeze=False)
        for ax, lab in zip(axes[:, 0], labs):
            r = rms[rms.miw == lab].set_index("j").reindex(js)
            x = np.arange(len(js))
            for k, dx in zip(MEANS, (-0.2, 0, 0.2)):
                ax.bar(x + dx, r[f"rms_{k.split()[0]}_minus_batoid_um"], 0.2,
                       label=f"mean over {k} - Batoid")
            ax.plot(x, r.rms_batoid_um, "k_", ms=14, label="RMS of Batoid itself")
            ax.set_xticks(x, [f"Z{j}" for j in js])
            ax.set_yscale("log")
            ax.set_ylabel("RMS over field [µm]")
            ax.set_title(f"{lab}: bin-mean MIW against Batoid, OCS, "
                         "cells present in all 9 bins")
            ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        cols = MEANS + ["Batoid"]
        for j in js:
            fig, axes = plt.subplots(len(labs), 4, figsize=(15, 3.8 * len(labs)),
                                     squeeze=False)
            vals = np.concatenate([means[lab].loc[means[lab].j == j, cols].to_numpy().ravel()
                                   for lab in labs])
            lo, hi = np.nanpercentile(vals, [1, 99])
            for lab, row in zip(labs, axes):
                c = means[lab][means[lab].j == j]
                for ax, col in zip(row, cols):
                    ax.set_aspect("equal")
                    ax.set_title(f"{lab}\n{col if col == 'Batoid' else 'mean over ' + col}",
                                 fontsize=9)
                    ax.set_xlabel("thx OCS [deg]")
                    ax.set_ylabel("thy OCS [deg]")
                    if c.empty:
                        ax.text(0.5, 0.5, f"Z{j} not fitted", ha="center",
                                transform=ax.transAxes)
                        continue
                    im = ax.imshow(to_image(c, col, xs, ys), origin="lower", extent=ext,
                                   vmin=lo, vmax=hi, cmap="RdBu_r")
                    ax.text(0.02, 0.02, f"RMS {np.sqrt(np.mean(c[col] ** 2)):.3f} µm",
                            transform=ax.transAxes, fontsize=8)
            fig.colorbar(im, ax=axes, shrink=0.8, label=f"Z{j} [µm]")
            fig.suptitle(f"Z{j}: bin-mean MIW and Batoid intrinsic, OCS")
            pdf.savefig(fig)
            plt.close(fig)
    print(pdf_path)


if __name__ == "__main__":
    main()
