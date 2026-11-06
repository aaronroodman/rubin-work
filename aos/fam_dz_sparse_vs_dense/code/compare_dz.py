"""Per-visit DZ(k,j) of the sparse Danish 1.3 fit against the dense fit.

Joins the ``fits`` tables of two ``fam_tables`` builds on (day_obs, seq_num) and, for
every pupil Noll index j fitted in both and every focal order k of the k=1..6 and
k=1..3 fits, compares the coefficients visit by visit. Coefficients are in µm of
wavefront, OCS.

Also compares each sparse DZ(k,j) with the sum of the dense terms of the same
azimuthal order and parity (``FAMILIES``), e.g. sparse Z7 against dense Z7 + Z17, and
with the Huber-fitted linear combination of those terms.

Writes one PDF (summary pages, one page per j, one page per family) and
``dz_stats.csv``, ``dz_family_stats.csv`` and ``dz_family_fit.csv`` into the run
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

# Sparse term -> dense terms of the same azimuthal order m and sin/cos parity, in
# increasing radial order n: the higher-n terms the sparse fit omits, which it is
# expected to fold into the lower-n term it does fit.
FAMILIES = {
    5: [5, 13, 23], 6: [6, 12, 24],
    7: [7, 17], 8: [8, 16],
    9: [9, 19], 10: [10, 18],
    11: [11, 22],
    14: [14, 26], 15: [15, 25],
}


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


def family_compare(sparse, dense, fit="z1toz6"):
    """Sparse DZ(k,j) against partial sums of its dense family, plus a Huber
    regression on the family terms.

    Returns
    -------
    sums : `pandas.DataFrame`
        One row per (j, k, number of terms summed).
    combo : `pandas.DataFrame`
        One row per (j, k): Huber coefficients of sparse on the dense family terms
        (dimensionless, sparse over dense) with formal errors, the offset (µm), and
        statistics of sparse against the fitted combination.
    """
    good = ~(sparse[f"{fit}_bad_fit"].to_numpy(bool)
             | dense[f"{fit}_bad_fit"].to_numpy(bool))
    rows, combo = [], []
    for j, fam in FAMILIES.items():
        for k in range(1, FITS[fit] + 1):
            y = sparse[f"{fit}_z{j}_c{k}"].to_numpy(float)
            X = np.column_stack([dense[f"{fit}_z{t}_c{k}"].to_numpy(float) for t in fam])
            m = good & np.isfinite(y) & np.isfinite(X).all(axis=1)
            reg = sm.RLM(y[m], sm.add_constant(X[m]), M=sm.robust.norms.HuberT()).fit()
            pred = reg.fittedvalues
            c = dict(fit=fit, j=j, k=k, terms="+".join(f"Z{t}" for t in fam),
                     n=int(m.sum()), offset_um=reg.params[0],
                     offset_err_um=reg.bse[0],
                     nmad_resid_um=nmad(y[m] - pred),
                     nmad_unit_sum_um=nmad(y[m] - X[m].sum(axis=1)),
                     pearson_r=stats.pearsonr(pred, y[m])[0],
                     spearman_rho=stats.spearmanr(pred, y[m])[0])
            for i, t in enumerate(fam):
                c[f"coef_{i + 1}"], c[f"coef_{i + 1}_err"] = reg.params[i + 1], reg.bse[i + 1]
            combo.append(c)
            for nt in range(1, len(fam) + 1):
                x = X[m, :nt].sum(axis=1)
                d = y[m] - x
                a, b = huber_line(x, y[m])
                rows.append(dict(
                    fit=fit, j=j, k=k, terms="+".join(f"Z{t}" for t in fam[:nt]),
                    n_terms=nt, n=int(m.sum()),
                    median_diff_um=np.median(d), nmad_diff_um=nmad(d),
                    huber_offset_um=a, huber_slope=b,
                    pearson_r=stats.pearsonr(x, y[m])[0],
                    spearman_rho=stats.spearmanr(x, y[m])[0],
                    regression=" ".join(f"Z{t}:{c:.3f}" for t, c in
                                        zip(fam, reg.params[1:])),
                    regression_offset_um=reg.params[0]))
    return pd.DataFrame(rows), pd.DataFrame(combo)


def family_summary_page(pdf, fs, title):
    """Per (j, k): nMAD of sparse minus the dense sum, against the number of terms."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    labels = [f"Z{j} {NOLL_NAMES[j]}" for j in FAMILIES]
    for ax, col, lab in [(axes[0], "nmad_diff_um", "nMAD(sparse - dense sum) [µm]"),
                         (axes[1], "spearman_rho", "Spearman rho (dimensionless)")]:
        for nt, mk in [(1, "o"), (2, "s"), (3, "^")]:
            s = fs[fs.n_terms == nt]
            for k in range(1, 7):
                sk = s[s.k == k].set_index("j").reindex(list(FAMILIES))
                ax.scatter(np.arange(len(FAMILIES)) + (k - 3.5) * 0.1, sk[col],
                           marker=mk, color=f"C{k - 1}", s=25,
                           label=f"k={k}" if nt == 1 else None)
        ax.set_xticks(range(len(FAMILIES)), labels, rotation=45, ha="right")
        ax.set_ylabel(lab)
        if col == "nmad_diff_um":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="color: focal k; marker: o lowest-n term only, "
                   "s +2nd, ^ +3rd", title_fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def coef_summary_page(pdf, fc, title):
    """Huber coefficients of sparse Z_j on each dense family term, against k."""
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex=True)
    ks = np.arange(1, 7)
    for ax, (j, fam) in zip(axes.flat, FAMILIES.items()):
        s = fc[fc.j == j].set_index("k").reindex(ks)
        for i, t in enumerate(fam):
            ax.errorbar(ks + (i - 1) * 0.08, s[f"coef_{i + 1}"], s[f"coef_{i + 1}_err"],
                        fmt="os^"[i], color=f"C{i}", ms=5, capsize=2,
                        label=f"dense Z{t} {NOLL_NAMES[t]}")
        ax.axhline(1, color="k", ls=":", lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"sparse Z{j} {NOLL_NAMES[j]}", fontsize=10)
        ax.set_ylabel("coefficient (dimensionless)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("focal-plane order k")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def family_page(pdf, sparse, dense, fs, fc, j, fit="z1toz6"):
    """Rows: sparse DZ(k,j) against dense Z_j, Z_j + 2nd, (+ 3rd), and against the
    Huber-fitted combination of the family; columns: k."""
    fam = FAMILIES[j]
    bad = (sparse[f"{fit}_bad_fit"].to_numpy(bool)
           | dense[f"{fit}_bad_fit"].to_numpy(bool))
    nr = len(fam) + 1
    fig, axes = plt.subplots(nr, 6, figsize=(24, 4.3 * nr), squeeze=False)
    for k in range(1, 7):
        y = sparse[f"{fit}_z{j}_c{k}"].to_numpy(float)
        for nt in range(1, len(fam) + 1):
            ax = axes[nt - 1, k - 1]
            x = sum(dense[f"{fit}_z{t}_c{k}"].to_numpy(float) for t in fam[:nt])
            r = fs[(fs.j == j) & (fs.k == k) & (fs.n_terms == nt)].iloc[0]
            ax.scatter(x[~bad], y[~bad], s=4, alpha=0.5, color="C0")
            if bad.any():
                ax.scatter(x[bad], y[bad], s=12, facecolors="none", edgecolors="C3")
            lo, hi = np.nanpercentile(np.r_[x[~bad], y[~bad]], [0.5, 99.5])
            lim = (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            ax.plot(lim, lim, "k:", lw=1)
            ax.plot(lim, r.huber_offset_um + r.huber_slope * np.array(lim), "C1", lw=1)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_title(f"DZ(k={k}): sparse Z{j} vs dense {r.terms}", fontsize=9)
            ax.set_xlabel(f"dense {r.terms} [µm]", fontsize=8)
            ax.set_ylabel(f"sparse Z{j} [µm]", fontsize=8)
            txt = (f"n={r.n}\nslope={r.huber_slope:.3f}, offset={r.huber_offset_um:+.4f} µm"
                   f"\nr={r.pearson_r:.3f}, rho={r.spearman_rho:.3f}"
                   f"\nmed diff={r.median_diff_um:+.4f} µm\nnMAD diff={r.nmad_diff_um:.4f} µm")
            if nt == len(fam):
                txt += f"\nHuber regression coeffs:\n{r.regression}"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(fc="w", alpha=0.8, lw=0))
        # Last row: the optimal (Huber) linear combination, offset included.
        c = fc[(fc.j == j) & (fc.k == k)].iloc[0]
        x = c.offset_um + sum(c[f"coef_{i + 1}"] * dense[f"{fit}_z{t}_c{k}"].to_numpy(float)
                              for i, t in enumerate(fam))
        ax = axes[-1, k - 1]
        ax.scatter(x[~bad], y[~bad], s=4, alpha=0.5, color="C2")
        if bad.any():
            ax.scatter(x[bad], y[bad], s=12, facecolors="none", edgecolors="C3")
        lo, hi = np.nanpercentile(np.r_[x[~bad], y[~bad]], [0.5, 99.5])
        lim = (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.plot(lim, lim, "k:", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        combo = " + ".join(f"{c[f'coef_{i + 1}']:.2f} Z{t}" for i, t in enumerate(fam))
        ax.set_title(f"DZ(k={k}): sparse Z{j} vs fitted dense combination", fontsize=9)
        ax.set_xlabel(f"{c.offset_um:+.3f} µm + {combo} [µm]", fontsize=8)
        ax.set_ylabel(f"sparse Z{j} [µm]", fontsize=8)
        lines = [f"n={c.n}"]
        lines += [f"coef Z{t} = {c[f'coef_{i + 1}']:.3f} ± {c[f'coef_{i + 1}_err']:.3f}"
                  for i, t in enumerate(fam)]
        lines += [f"offset = {c.offset_um:+.4f} ± {c.offset_err_um:.4f} µm",
                  f"r={c.pearson_r:.3f}, rho={c.spearman_rho:.3f}",
                  f"nMAD resid={c.nmad_resid_um:.4f} µm",
                  f"(unit sum: {c.nmad_unit_sum_um:.4f} µm)"]
        txt = "\n".join(lines)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(fc="w", alpha=0.8, lw=0))
    fig.suptitle(f"Sparse Z{j} {NOLL_NAMES[j]} against its dense family "
                 f"{' + '.join(f'Z{t} {NOLL_NAMES[t]}' for t in fam)}, {fit} fit, per visit, "
                 "OCS, µm of wavefront. Orange: Huber line; dotted: equality. Bottom row (green): "
                 "Huber multi-regression combination; coefficients dimensionless (sparse "
                 "over dense term), errors formal.", fontsize=11)
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
    both = fs.merge(fd.rename(columns={c: c + "__dense" for c in fd.columns
                                       if c not in KEYS}), on=KEYS)
    sparse = both[[c for c in both.columns if not c.endswith("__dense")]]
    dense = both[KEYS + [c for c in both.columns if c.endswith("__dense")]]
    dense.columns = [c.removesuffix("__dense") for c in dense.columns]
    print(f"{len(fs)} sparse visits, {len(fd)} dense, {len(both)} in both; "
          f"common Noll j: {nolls}")

    st = compare(sparse, dense, nolls)
    st.to_csv(run.path / "dz_stats.csv", index=False)
    fs, fc = family_compare(sparse, dense)
    fs.to_csv(run.path / "dz_family_stats.csv", index=False)
    fc.to_csv(run.path / "dz_family_fit.csv", index=False)
    pdf_path = run.path / "dz_sparse_vs_dense.pdf"
    with PdfPages(pdf_path) as pdf:
        for fit in FITS:
            summary_page(pdf, st, fit, nolls,
                         f"{args.sparse} (sparse) minus {args.dense} (dense), {fit} fit, "
                         f"OCS; {len(both)} common visits, n >= "
                         f"{int(st[st.fit == fit].n.min())} good in both")
        family_summary_page(pdf, fs, f"Sparse Z_j against the sum of its dense family "
                            f"(same m, higher n), z1toz6 fit, OCS; n = {fs.n.min()} visits")
        coef_summary_page(pdf, fc, "Huber coefficients of sparse Z_j on its dense family "
                          "terms (dimensionless, sparse over dense; formal errors), "
                          f"z1toz6 fit, OCS; n = {fc.n.min()} visits. Dotted: 1")
        for j in nolls:
            j_page(pdf, sparse, dense, st, j)
        for j in FAMILIES:
            family_page(pdf, sparse, dense, fs, fc, j)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
