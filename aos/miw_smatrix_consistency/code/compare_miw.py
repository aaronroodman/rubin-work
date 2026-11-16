"""Consistency of the nine per-rotator-bin MIWs, for MIWs built on different S-matrices.

For each ``miw`` build: the OCS/CCS split residual per Zernike (``intrinsic_split_rms``),
the scatter of the nine bin grids about their mean on common field cells, the
bin-averaged grid against the Batoid intrinsic (``zk_tabulated``), and the recovered
DoF per visit against the DoF ranges. MIW in µm of wavefront, OCS.

Writes ``miw_smatrix_consistency.pdf``, ``split_rms.csv``, ``bin_scatter.csv``,
``batoid_diff.csv`` and ``dof_range.csv`` into the run directory.

Usage::

    python aos/miw_smatrix_consistency/code/compare_miw.py \\
        --miw v13-dense=d13-A_50_34_i@20261025 \\
        --miw fsm-dense21=d13-A_50_34_i_fsm_dense21@20261116 \\
        --miw fsm-sparse_no22=d13s-A_50_34_i_fsm_sparse_no22@20261116
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from lsst.ts.intrinsic.wavefront import ofc_svd  # noqa: E402
from rubinwork.common import study_run  # noqa: E402
from rubinwork.products import catalog  # noqa: E402
from rubinwork.products.miw import reader as miw  # noqa: E402

# Rotator bins the _5rot variants keep; the other four are the night 20260409.
ROT5 = {"rot_-65_-55", "rot_-20_-10", "rot_-3_3", "rot_10_20", "rot_55_65"}


def load_bins(variant, build, min_donuts=5):
    """Per-bin grids as one long table: bin, cell (rounded thx, thy), j, zk, batoid.

    Cells with fewer than ``min_donuts`` donuts are dropped: a 1-donut cell can be off by
    over 1000 µm and dominates any RMS over cells.
    """
    rows = []
    for p in sorted((catalog.path("miw", variant, build) / "build").glob(
            "rot_*/intrinsic_grid.parquet")):
        g = pd.read_parquet(p)
        g = g[g.n_donuts >= min_donuts]
        js = np.asarray(g.nollIndices.iloc[0], int)
        zk, tab = np.stack(g.zk.to_numpy()), np.stack(g.zk_tabulated.to_numpy())
        for i, j in enumerate(js):
            rows.append(pd.DataFrame(dict(
                rot_bin=p.parent.name, thx=g.thx_deg.round(4), thy=g.thy_deg.round(4),
                j=j, zk=zk[:, i], batoid=tab[:, i], n_donuts=g.n_donuts)))
    return pd.concat(rows, ignore_index=True)


def bin_scatter(t, bins=None):
    """Per j: RMS over common cells of the std over bins, and of the bin-mean."""
    if bins is not None:
        t = t[t.rot_bin.isin(bins)]
    nb = t.rot_bin.nunique()
    out = []
    for j, g in t.groupby("j"):
        w = g.pivot_table(index=["thx", "thy"], columns="rot_bin", values="zk").dropna()
        out.append(dict(j=j, n_bins=nb, n_cells=len(w),
                        scatter_rms_um=np.sqrt(np.mean(w.std(axis=1, ddof=1) ** 2)),
                        mean_rms_um=np.sqrt(np.mean(w.mean(axis=1) ** 2))))
    return pd.DataFrame(out)


def batoid_diff(t):
    """Per j: RMS over cells of (bin-mean MIW - Batoid) and of Batoid itself."""
    out = []
    for j, g in t.groupby("j"):
        c = g.groupby(["thx", "thy"]).agg(zk=("zk", "mean"), batoid=("batoid", "mean"))
        out.append(dict(j=j, n_cells=len(c),
                        rms_diff_um=np.sqrt(np.mean((c.zk - c.batoid) ** 2)),
                        rms_miw_um=np.sqrt(np.mean(c.zk ** 2)),
                        rms_batoid_um=np.sqrt(np.mean(c.batoid ** 2))))
    return pd.DataFrame(out)


def dof_range(variant, build, ranges, labels):
    """Per DoF: median and 95th percentile over visits of |DoF| / range."""
    d = pd.concat([pd.read_parquet(p) for p in sorted(
        (catalog.path("miw", variant, build) / "build").glob("rot_*/dz_fits.parquet"))])
    x = np.abs(d[[f"dof_{n}" for n in labels]].to_numpy(float)) / ranges[None, :]
    return pd.DataFrame(dict(dof=labels, median_frac=np.nanmedian(x, axis=0),
                             p95_frac=np.nanpercentile(x, 95, axis=0),
                             frac_visits_over_range=np.nanmean(x > 1, axis=0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--miw", action="append", required=True,
                    metavar="LABEL=VARIANT@BUILD")
    ap.add_argument("--label", default="smatrix_consistency")
    ap.add_argument("--min-donuts", type=int, default=5,
                    help="drop grid cells with fewer donuts (count)")
    args = ap.parse_args()
    run = study_run("aos/miw_smatrix_consistency", args.label)
    miws = []
    for spec in args.miw:
        lab, _, vb = spec.partition("=")
        v, _, b = vb.partition("@")
        miws.append((lab, v, b or catalog.CURRENT))

    est = ofc_svd.build_ofc_svd(list(range(4, 12)), 1, 1, 8).estimator
    labels, _ = est.dof_labels()
    ranges = np.asarray(est.dof_ranges(), float)

    split, scat, bat, dofs = [], [], [], []
    for lab, v, b in miws:
        r = pd.read_parquet(catalog.path("miw", v, b) / "intrinsic_split_rms.parquet")
        split.append(r.assign(miw=lab))
        t = load_bins(v, b, args.min_donuts)
        scat.append(pd.concat([bin_scatter(t).assign(bins="all 9"),
                               bin_scatter(t, ROT5).assign(bins="5rot")]).assign(miw=lab))
        bat.append(batoid_diff(t).assign(miw=lab))
        dofs.append(dof_range(v, b, ranges, labels).assign(miw=lab))
    split, scat, bat, dofs = (pd.concat(x, ignore_index=True)
                              for x in (split, scat, bat, dofs))
    for name, df in [("split_rms", split), ("bin_scatter", scat),
                     ("batoid_diff", bat), ("dof_range", dofs)]:
        df.to_csv(run.path / f"{name}.csv", index=False)

    labs = [m[0] for m in miws]
    js = sorted(set.intersection(*(set(split[split.miw == lab].j) for lab in labs)))
    pdf_path = run.path / "miw_smatrix_consistency.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
        x = np.arange(len(js))
        w = 0.8 / len(labs)
        for i, lab in enumerate(labs):
            s = split[split.miw == lab].set_index("j").reindex(js)
            axes[0].bar(x + (i - (len(labs) - 1) / 2) * w, s.residRMS, w, label=lab)
            for ax, sel in [(axes[1], "all 9"), (axes[2], "5rot")]:
                c = scat[(scat.miw == lab) & (scat.bins == sel)].set_index("j").reindex(js)
                ax.bar(x + (i - (len(labs) - 1) / 2) * w, c.scatter_rms_um, w, label=lab)
        axes[0].set_ylabel("OCS/CCS split residual RMS [µm]")
        axes[1].set_ylabel("bin-to-bin scatter, 9 bins [µm]")
        axes[2].set_ylabel("bin-to-bin scatter, 5rot bins [µm]")
        axes[2].set_xticks(x, [f"Z{j}" for j in js])
        for ax in axes:
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=8)
        fig.suptitle("Consistency of the per-rotator-bin MIWs (common Noll j). Bin-to-bin "
                     "scatter: RMS over common field cells of the std over bins, OCS; it "
                     "includes the real rotation of the camera-fixed part.", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(15, 6))
        for i, lab in enumerate(labs):
            s = bat[bat.miw == lab].set_index("j").reindex(js)
            ax.bar(x + (i - (len(labs) - 1) / 2) * w, s.rms_diff_um, w, label=lab)
        ax.plot(x, bat[bat.miw == labs[0]].set_index("j").reindex(js).rms_batoid_um,
                "k_", ms=20, label="RMS of Batoid itself")
        ax.set_xticks(x, [f"Z{j}" for j in js])
        ax.set_ylabel("RMS over field of (bin-mean MIW - Batoid) [µm]")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        ax.set_title("MIW against the Batoid intrinsic, OCS")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(17, 6))
        xd = np.arange(len(labels))
        for i, lab in enumerate(labs):
            s = dofs[dofs.miw == lab]
            ax.scatter(xd + (i - 1) * 0.2, s.p95_frac, s=12, label=f"{lab}: 95th pct")
            ax.scatter(xd + (i - 1) * 0.2, s.median_frac, s=12, marker="_",
                       color=f"C{i}")
        ax.axhline(1, color="k", ls=":")
        ax.set_yscale("log")
        ax.set_xticks(xd, labels, rotation=90, fontsize=7)
        ax.set_ylabel("|DoF| / DoF range over visits (dimensionless); dot 95th pct, "
                      "dash median")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Recovered DoF per visit against the DoF ranges")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(split.pivot(index="j", columns="miw", values="residRMS").round(4).to_string())
    print(scat.pivot_table(index="j", columns=["bins", "miw"],
                           values="scatter_rms_um").round(4).to_string())
    print(bat.pivot(index="j", columns="miw", values="rms_diff_um").round(4).to_string())
    print(dofs.groupby("miw")[["p95_frac", "frac_visits_over_range"]].max().round(3))
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
