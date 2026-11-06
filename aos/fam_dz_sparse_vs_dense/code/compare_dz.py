"""Per-visit DZ(k,j) of the sparse Danish 1.3 fit against the dense fit.

Joins the ``fits`` tables of two ``fam_tables`` builds on (day_obs, seq_num) and, for
every pupil Noll index j fitted in both and every focal order k of the k=1..6 and
k=1..3 fits, compares the coefficients visit by visit. Coefficients are in µm of
wavefront, OCS.

Writes one PDF (summary pages, then one page per j) and ``dz_stats.csv`` into the run
directory.

Usage::

    python aos/fam_dz_sparse_vs_dense/code/compare_dz.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402

from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES  # noqa: E402
from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402
from rubinwork.products.fam_tables import reader  # noqa: E402

FITS = {"z1toz6": 6, "z1toz3": 3}
KEYS = ["day_obs", "seq_num"]


def noll_set(fits):
    """Pupil Noll indices present in a fits table, from its column names."""
    return sorted({int(c.split("_")[1][1:]) for c in fits.columns
                   if c.startswith("z1toz6_z") and c.endswith("_c1")})


def huber_line(x, y):
    """Huber fit y = a + b x; returns (a, b) in the units of y and y/x."""
    res = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    return res.params[0], res.params[1]


def compare(sparse, dense, nolls):
    """One stats row per (fit, j, k) over visits good in both builds."""
    rows = []
    for fit, kmax in FITS.items():
        good = ~(sparse[f"{fit}_bad_fit"].to_numpy(bool)
                 | dense[f"{fit}_bad_fit"].to_numpy(bool))
        for j in nolls:
            for k in range(1, kmax + 1):
                c = f"{fit}_z{j}_c{k}"
                x, y = dense[c].to_numpy(float), sparse[c].to_numpy(float)
                m = good & np.isfinite(x) & np.isfinite(y)
                d = y[m] - x[m]
                a, b = huber_line(x[m], y[m])
                rows.append(dict(
                    fit=fit, j=j, k=k, n=int(m.sum()),
                    median_diff_um=np.median(d), nmad_diff_um=nmad(d),
                    nmad_dense_um=nmad(x[m]), nmad_sparse_um=nmad(y[m]),
                    huber_offset_um=a, huber_slope=b,
                    pearson_r=stats.pearsonr(x[m], y[m])[0],
                    spearman_rho=stats.spearmanr(x[m], y[m])[0]))
    return pd.DataFrame(rows)


def summary_page(pdf, st, fit, nolls, title):
    """Grids over (j, k) of the median and nMAD of sparse minus dense, and of rho."""
    kmax = FITS[fit]
    s = st[st.fit == fit]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for ax, col, lab, cmap, sym in [
            (axes[0], "median_diff_um", "median(sparse - dense) [µm]", "RdBu_r", True),
            (axes[1], "nmad_diff_um", "nMAD(sparse - dense) [µm]", "viridis", False),
            (axes[2], "spearman_rho", "Spearman rho (dimensionless)", "viridis", False)]:
        g = s.pivot(index="j", columns="k", values=col).loc[nolls]
        v = np.nanmax(np.abs(g.to_numpy())) if sym else None
        im = ax.imshow(g.to_numpy(), cmap=cmap, aspect="auto",
                       vmin=-v if sym else None, vmax=v if sym else None)
        for (r, cc), val in np.ndenumerate(g.to_numpy()):
            ax.text(cc, r, f"{val:.3f}", ha="center", va="center", fontsize=7,
                    color="k" if sym else "w")
        ax.set_xticks(range(kmax), [f"k={k}" for k in range(1, kmax + 1)])
        ax.set_yticks(range(len(nolls)), [f"Z{j} {NOLL_NAMES.get(j, '')}" for j in nolls])
        ax.set_title(lab)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def j_page(pdf, sparse, dense, st, j):
    """Scatter of sparse against dense for every k of both fits, at one j."""
    fig, axes = plt.subplots(2, 6, figsize=(24, 9))
    for row, (fit, kmax) in enumerate(FITS.items()):
        bad = (sparse[f"{fit}_bad_fit"].to_numpy(bool)
               | dense[f"{fit}_bad_fit"].to_numpy(bool))
        for k in range(1, 7):
            ax = axes[row, k - 1]
            if k > kmax:
                ax.axis("off")
                continue
            c = f"{fit}_z{j}_c{k}"
            x, y = dense[c].to_numpy(float), sparse[c].to_numpy(float)
            ax.scatter(x[~bad], y[~bad], s=4, alpha=0.5, color="C0")
            ax.scatter(x[bad], y[bad], s=12, facecolors="none", edgecolors="C3",
                       label="bad fit (either)")
            r = st[(st.fit == fit) & (st.j == j) & (st.k == k)].iloc[0]
            lo, hi = np.nanpercentile(np.r_[x[~bad], y[~bad]], [0.5, 99.5])
            pad = 0.1 * (hi - lo)
            lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, "k:", lw=1)
            ax.plot(lim, r.huber_offset_um + r.huber_slope * np.array(lim), "C1", lw=1)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_title(f"{fit}  DZ(k={k}, Z{j})", fontsize=10)
            ax.set_xlabel("dense danish_1_3 [µm]")
            ax.set_ylabel("sparse d13-sparse [µm]")
            ax.text(0.03, 0.97,
                    f"n={r.n}\nslope={r.huber_slope:.3f}, offset={r.huber_offset_um:+.4f} µm"
                    f"\nr={r.pearson_r:.3f}, rho={r.spearman_rho:.3f}"
                    f"\nmed diff={r.median_diff_um:+.4f} µm\nnMAD diff={r.nmad_diff_um:.4f} µm",
                    transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(fc="w", alpha=0.8, lw=0))
            if bad.any() and k == 1:
                ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(f"Z{j} {NOLL_NAMES.get(j, '')}: per-visit DZ, sparse against dense, OCS, "
                 "µm of wavefront.  Orange: Huber line; dotted: equality. "
                 "Top: k=1..6 fit; bottom: k=1..3 fit.", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sparse", default="d13-sparse@20261106")
    ap.add_argument("--dense", default="danish_1_3@20261028b")
    ap.add_argument("--label", default="sparse_vs_dense")
    args = ap.parse_args()
    run = study_run("aos/fam_dz_sparse_vs_dense", args.label)

    sv, sb = args.sparse.split("@")
    dv, db = args.dense.split("@")
    fs, fd = reader.load_table("fits", sv, sb), reader.load_table("fits", dv, db)
    nolls = sorted(set(noll_set(fs)) & set(noll_set(fd)))
    both = fs.merge(fd, on=KEYS, suffixes=("", "__dense"))
    sparse = both[[c for c in both.columns if not c.endswith("__dense")]]
    dense = both[KEYS + [c for c in both.columns if c.endswith("__dense")]]
    dense.columns = [c.removesuffix("__dense") for c in dense.columns]
    print(f"{len(fs)} sparse visits, {len(fd)} dense, {len(both)} in both; "
          f"common Noll j: {nolls}")

    st = compare(sparse, dense, nolls)
    st.to_csv(run.path / "dz_stats.csv", index=False)
    pdf_path = run.path / "dz_sparse_vs_dense.pdf"
    with PdfPages(pdf_path) as pdf:
        for fit in FITS:
            summary_page(pdf, st, fit, nolls,
                         f"{args.sparse} (sparse) minus {args.dense} (dense), {fit} fit, "
                         f"OCS; {len(both)} common visits, n >= "
                         f"{int(st[st.fit == fit].n.min())} good in both")
        for j in nolls:
            j_page(pdf, sparse, dense, st, j)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
