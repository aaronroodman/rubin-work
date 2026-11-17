"""OCS (telescope) term of each MIW's OCS/CCS split beside Batoid, one page per Zernike.

The maps are the "O telescope (OCS)" panels of each build's ``intrinsic_split.pdf``, drawn
the same way (filled contours on the split's polar grid) from
``intrinsic_split_decomp.parquet``, on one shared colour scale per page. Batoid is the
build grids' ``zk_tabulated`` (rot_-3_3 bin, all cells, first build fitting each j) interpolated onto the same polar
grid. RMS is over the split's window, field radius 0.1 to 1.6 deg. µm of wavefront, OCS.

Writes ``ocs_split_maps.pdf`` and ``ocs_minus_batoid.csv`` into the run directory.

Usage::

    python aos/miw_smatrix_consistency/code/ocs_split_maps.py \\
        --miw sparse=d13s-A_50_34_i_fsm_sparse_no22@20261116b \\
        --miw sparse-rbr=d13s-A_50_50_i_rbr_fsm_sparse_no22@20261116
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.products import catalog  # noqa: E402

R_LIM = (0.1, 1.6)   # deg; the split's RMS window (mi_config r_lim_lo, r_lim_hi)


def load_ocs(variant, build):
    """OCS term per j on the polar grid: dict j -> (n_r, n_az) array [µm], and X, Y [deg]."""
    t = Table.read(catalog.path("miw", variant, build) / "intrinsic_split_decomp.parquet")
    n_r, n_az = t.meta["n_r"], t.meta["n_az"]
    X = np.reshape(t.meta["X"], (n_r, n_az))
    Y = np.reshape(t.meta["Y"], (n_r, n_az))
    ocs = {int(r["j"]): np.reshape(r["O_im"] if r["part"] else r["O_re"], (n_r, n_az))
           for r in t}
    return ocs, X, Y


def load_batoid(variant, build, X, Y):
    """Batoid intrinsic per j, interpolated from the rot_-3_3 grid onto (X, Y) [µm]."""
    g = pd.read_parquet(catalog.path("miw", variant, build) / "build" / "rot_-3_3"
                        / "intrinsic_grid.parquet")
    tab = np.stack(g.zk_tabulated.to_numpy())
    pts = np.column_stack([g.thx_deg, g.thy_deg])
    return {int(j): griddata(pts, tab[:, i], (X, Y), method="linear")
            for i, j in enumerate(g.nollIndices.iloc[0])}


def rms(field, R):
    m = (R >= R_LIM[0]) & (R <= R_LIM[1])
    return float(np.sqrt(np.nanmean(field[m, :] ** 2)))


def plot_field(ax, X, Y, vals, title, vlim):
    """As _plot_field in run_intrinsic_split.py: rasterized filled contours, 21 levels."""
    fin = np.isfinite(vals)
    im = ax.tricontourf(X[fin], Y[fin], vals[fin], levels=np.linspace(-vlim, vlim, 21),
                        cmap="RdBu_r", extend="both")
    im.set_rasterized(True)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("thx OCS [deg]", fontsize=7)
    ax.set_ylabel("thy OCS [deg]", fontsize=7)
    ax.tick_params(labelsize=6)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--miw", action="append", required=True,
                    metavar="LABEL=VARIANT@BUILD")
    ap.add_argument("--label", default="ocs_split_maps")
    args = ap.parse_args()
    run = study_run("aos/miw_smatrix_consistency", args.label)

    ocs, specs = {}, {}
    for spec in args.miw:
        lab, _, vb = spec.partition("=")
        specs[lab] = vb.partition("@")[::2]
        ocs[lab], X, Y = load_ocs(*specs[lab])
    batoid = {}
    for v, b in specs.values():
        batoid = {**load_batoid(v, b, X, Y), **batoid}
    R = np.hypot(X[:, 0], Y[:, 0])
    labs = list(ocs)
    js = sorted(set().union(*ocs.values()))

    rows = [dict(miw=lab, j=j, rms_ocs_um=rms(o[j], R), rms_batoid_um=rms(batoid[j], R),
                 rms_ocs_minus_batoid_um=rms(o[j] - batoid[j], R))
            for lab, o in ocs.items() for j in js if j in o]
    tab = pd.DataFrame(rows)
    tab.to_csv(run.path / "ocs_minus_batoid.csv", index=False)

    pdf_path = run.path / "ocs_split_maps.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(14, 5.5), layout="constrained")
        x = np.arange(len(js))
        w = 0.8 / len(labs)
        for k, lab in enumerate(labs):
            r = tab[tab.miw == lab].set_index("j").reindex(js)
            ax.bar(x - 0.4 + w * (k + 0.5), r.rms_ocs_minus_batoid_um, w,
                   label=f"{lab}: OCS term - Batoid")
        ax.plot(x, [rms(batoid[j], R) for j in js], "k_", ms=14,
                label="RMS of Batoid itself")
        ax.set_xticks(x, [f"Z{j}" for j in js])
        ax.set_yscale("log")
        ax.set_ylabel("RMS over field radius 0.1 to 1.6 deg [µm]")
        ax.set_title("OCS term of the OCS/CCS split against Batoid")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        m = (R >= R_LIM[0]) & (R <= R_LIM[1])
        for j in js:
            fields = [(lab, ocs[lab].get(j)) for lab in labs] + [("Batoid", batoid[j])]
            vlim = max(np.nanpercentile(np.abs(np.concatenate(
                [f[m, :].ravel() for _, f in fields if f is not None])), 98), 1e-6)
            ncols = 3
            nrows = -(-len(fields) // ncols)
            fig, axs = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.5 * nrows),
                                    layout="constrained", squeeze=False)
            for ax in axs.ravel()[len(fields):]:
                ax.set_visible(False)
            im = None
            for ax, (lab, f) in zip(axs.ravel(), fields):
                if f is None:
                    ax.text(0.5, 0.5, f"Z{j} not fitted", ha="center",
                            transform=ax.transAxes)
                    ax.set_title(lab, fontsize=9)
                    continue
                if lab == "Batoid":
                    title = f"Batoid intrinsic\nRMS={rms(f, R):.4f} µm"
                else:
                    title = (f"{lab}: O telescope (OCS)\nRMS={rms(f, R):.4f} µm, "
                             f"RMS(O - Batoid)={rms(f - batoid[j], R):.4f} µm")
                im = plot_field(ax, X, Y, f, title, vlim)
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label=f"Z{j} [µm]")
            fig.suptitle(f"Z{j}: OCS term of the OCS/CCS split, and Batoid", fontsize=13)
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
    print(pdf_path)


if __name__ == "__main__":
    main()
