"""Is dense DZ(1,17) an optical state the AOS degrees of freedom (DoF) can make?

Uses the per-visit DZ split of a ``miw`` build (``build/rot_*/dz_fits.parquet``): the raw
dense DZ fit (``dz``), its correctable part, the projection onto the build's 50 v-modes
(``dz_corr`` = S @ DoF), and the residual (``dz_resid``). Also builds the same DZ
sensitivity matrix S (k=1..6, Noll 4-19 and 22-26, 50 DoF) to rank which DoF move
DZ(1,17), and per unit of their range. DZ in µm of wavefront, OCS.

Then the test that does not use Z17 at all: recover the DoF per visit from every DZ term
except Z17 (truncated SVD of the normalized S, 20, 30 or 50 modes) and predict DZ(1,17)
from that state. If the measured DZ(1,17) follows, it is the optical state seen in the
other terms (DZ(1,7), DZ(4,7), DZ(2,5), ...), not a fitting artifact.

Writes ``dz17_correctable.pdf``, ``correctable_fraction.csv``, ``s_row_1_17.csv`` and
``loo_1_17.csv``.

Usage::

    python aos/fam_dz_sparse_vs_dense/code/dz17_correctable.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402

from lsst.ts.intrinsic.wavefront import ofc_svd  # noqa: E402
from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402
from rubinwork.products import catalog  # noqa: E402

IZS = list(range(4, 20)) + list(range(22, 27))
# Fitted sparse term -> its omitted dense partners.
PAIRS = {5: [13, 23], 6: [12, 24], 7: [17], 8: [16], 9: [19], 10: [18], 11: [22],
         14: [26], 15: [25]}


def load_split(variant, build):
    """Per-visit raw, correctable and residual DZ over all rotator bins of a build."""
    root = catalog.path("miw", variant, build) / "build"
    parts = [pd.read_parquet(p).assign(rot_bin=p.parent.name)
             for p in sorted(root.glob("rot_*/dz_fits.parquet"))]
    d = pd.concat(parts, ignore_index=True)
    return d[~d.bad_fit.astype(bool)].drop_duplicates(["day_obs", "seq_num"])


def correctable_table(d):
    """Per (k, j): scatter over visits of raw, correctable and residual DZ."""
    rows = []
    for k in range(1, 7):
        for j in IZS:
            raw, cor, res = (d[f"{p}_z{j}_c{k}"].to_numpy(float)
                             for p in ("dz", "dz_corr", "dz_resid"))
            m = np.isfinite(raw) & np.isfinite(cor)
            rows.append(dict(
                k=k, j=j, n=int(m.sum()),
                mean_raw_um=raw[m].mean(), mean_corr_um=cor[m].mean(),
                mean_resid_um=res[m].mean(),
                std_raw_um=raw[m].std(), std_corr_um=cor[m].std(),
                std_resid_um=res[m].std(), nmad_raw_um=nmad(raw[m]),
                nmad_resid_um=nmad(res[m]),
                corr_var_frac=1 - np.var(res[m]) / np.var(raw[m]),
                pearson_raw_corr=stats.pearsonr(raw[m], cor[m])[0],
                spearman_raw_corr=stats.spearmanr(raw[m], cor[m])[0]))
    return pd.DataFrame(rows)


def predict_left_out(S, N, kj, W, drop, n_keep, target=(1, 17)):
    """DZ(target) from the DoF recovered without the ``drop`` rows.

    Least squares in normalized DoF (S times the normalization weights N), truncated
    to the ``n_keep`` largest singular values. Returns µm of wavefront per visit.
    """
    Sn = S * N[None, :]
    keep = [i for i, t in enumerate(kj) if t not in drop]
    U, sv, Vt = np.linalg.svd(Sn[keep], full_matrices=False)
    x = (np.nan_to_num(W[:, keep]) @ U[:, :n_keep]) / sv[:n_keep] @ Vt[:n_keep]
    return x @ Sn[kj.index(target)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--miw", default="d13-A_50_50_i_rbr@20261027")
    ap.add_argument("--label", default="dz17_correctable")
    args = ap.parse_args()
    run = study_run("aos/fam_dz_sparse_vs_dense", args.label)
    mv, mb = args.miw.split("@")

    svd = ofc_svd.build_ofc_svd(IZS, 1, 6, 50)
    est = svd.estimator
    kj = [tuple(x) for x in svd.kj_grid]
    S = est.forward_operator()                      # (n_kj, 50), µm per DoF unit
    labels, units = est.dof_labels()
    rng = np.asarray(est.dof_ranges(), float)       # DoF range, DoF units

    d = load_split(mv, mb)
    ct = correctable_table(d)
    ct.to_csv(run.path / "correctable_fraction.csv", index=False)

    # Per-visit DZ(1,17) from each DoF: S[(1,17), dof] * dof.
    dof = d[[f"dof_{n}" for n in labels]].to_numpy(float)
    i17, i7 = kj.index((1, 17)), kj.index((1, 7))
    contrib17 = dof * S[i17][None, :]
    srow = pd.DataFrame(dict(
        dof=labels, unit=units, range=rng, S_1_17=S[i17], S_1_7=S[i7],
        dz_1_17_at_range_um=S[i17] * rng, dz_1_7_at_range_um=S[i7] * rng,
        std_dof=dof.std(axis=0), std_contrib_1_17_um=contrib17.std(axis=0)))
    srow.to_csv(run.path / "s_row_1_17.csv", index=False)

    # Least-squares DoF making a unit pure DZ(1,17), and what it leaves.
    e = np.zeros(len(kj))
    e[i17] = 1.0
    x, *_ = np.linalg.lstsq(S, e, rcond=None)
    made = S @ x
    pure = dict(achieved=made[i17], leak_rms=np.sqrt(np.sum(np.delete(made, i17) ** 2)),
                max_range_frac=np.max(np.abs(x / rng)),
                worst_dof=labels[int(np.argmax(np.abs(x / rng)))])

    W = np.column_stack([d[f"dz_z{j}_c{k}"].to_numpy(float) for k, j in kj])
    y17 = W[:, i17]
    drop17 = [(k, 17) for k in range(1, 7)]
    loo = {nk: predict_left_out(S, svd.normalization_weights, kj, W, drop17, nk)
           for nk in (20, 30, 50)}
    loo_rows = [dict(n_keep=nk, n=len(y17), mean_meas_um=y17.mean(), mean_pred_um=p.mean(),
                     std_meas_um=y17.std(), std_pred_um=p.std(),
                     std_resid_um=(y17 - p).std(), nmad_resid_um=nmad(y17 - p),
                     pearson_r=stats.pearsonr(p, y17)[0],
                     spearman_rho=stats.spearmanr(p, y17)[0]) for nk, p in loo.items()]
    pd.DataFrame(loo_rows).to_csv(run.path / "loo_1_17.csv", index=False)

    pdf_path = run.path / "dz17_correctable.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: per-visit raw vs correctable, (1,17) and (1,7).
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        for row, j in zip(axes, (17, 7)):
            raw, cor, res = (d[f"{p}_z{j}_c1"].to_numpy(float)
                             for p in ("dz", "dz_corr", "dz_resid"))
            r = ct[(ct.k == 1) & (ct.j == j)].iloc[0]
            row[0].scatter(raw, cor, s=5, alpha=0.5)
            lim = [np.nanmin(np.r_[raw, cor]), np.nanmax(np.r_[raw, cor])]
            row[0].plot(lim, lim, "k:")
            row[0].set_xlabel(f"raw dense DZ(1,{j}) [µm]")
            row[0].set_ylabel(f"correctable S·DoF DZ(1,{j}) [µm]")
            row[0].set_title(f"r={r.pearson_raw_corr:.3f}, rho={r.spearman_raw_corr:.3f}, "
                             f"n={r.n}", fontsize=10)
            row[1].scatter(raw, res, s=5, alpha=0.5, color="C1")
            row[1].set_xlabel(f"raw dense DZ(1,{j}) [µm]")
            row[1].set_ylabel(f"uncorrectable residual DZ(1,{j}) [µm]")
            row[1].set_title(f"std raw {r.std_raw_um:.4f}, corr {r.std_corr_um:.4f}, "
                             f"resid {r.std_resid_um:.4f} µm", fontsize=10)
            cj = dof * S[kj.index((1, j))][None, :]
            top = np.argsort(cj.std(axis=0))[::-1][:12]
            row[2].barh([labels[t] for t in top][::-1], cj.std(axis=0)[top][::-1])
            row[2].set_xlabel(f"std over visits of S·DoF contribution to DZ(1,{j}) [µm]")
            for ax in row:
                ax.grid(alpha=0.3)
        fig.suptitle(f"miw {args.miw}: per-visit dense DZ split into correctable (50 "
                     f"v-modes) and residual; k=1, OCS; {len(d)} visits, "
                     f"{d.rot_bin.nunique()} rotator bins", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: DZ(1,17) against other terms, and predicted without Z17.
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
        for ax, (k, j) in zip(axes[:3], [(1, 7), (4, 7), (2, 5)]):
            x = W[:, kj.index((k, j))]
            ax.scatter(x, y17, s=5, alpha=0.5)
            ax.set_xlabel(f"dense DZ({k},{j}) [µm]")
            ax.set_ylabel("dense DZ(1,17) [µm]")
            ax.set_title(f"r={stats.pearsonr(x, y17)[0]:+.3f}, "
                         f"rho={stats.spearmanr(x, y17)[0]:+.3f}, n={len(y17)}", fontsize=10)
            ax.grid(alpha=0.3)
        ax = axes[3]
        p, r = loo[30], loo_rows[1]
        ax.scatter(p, y17, s=5, alpha=0.5, color="C2")
        lim = [min(p.min(), y17.min()), max(p.max(), y17.max())]
        ax.plot(lim, lim, "k:")
        ax.set_xlabel("DZ(1,17) predicted from DoF fitted without Z17 [µm]")
        ax.set_ylabel("measured dense DZ(1,17) [µm]")
        ax.set_title(f"30 modes: r={r['pearson_r']:.3f}, rho={r['spearman_rho']:.3f}; "
                     f"mean pred {r['mean_pred_um']:+.3f}, meas {r['mean_meas_um']:+.3f} µm;\n"
                     f"std meas {r['std_meas_um']:.4f}, resid {r['std_resid_um']:.4f} µm",
                     fontsize=9)
        ax.grid(alpha=0.3)
        fig.suptitle(f"Dense DZ(1,17) and the state seen in the other DZ terms; miw "
                     f"{args.miw} visits, OCS", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: S row (1,17) and (1,7), per unit DoF range.
        fig, axes = plt.subplots(2, 1, figsize=(17, 10), sharex=True)
        xs = np.arange(len(labels))
        for ax, col, lab in [(axes[0], "dz_1_17_at_range_um", "DZ(1,17)"),
                             (axes[1], "dz_1_7_at_range_um", "DZ(1,7)")]:
            ax.bar(xs, srow[col])
            ax.set_ylabel(f"{lab} at full DoF range [µm]")
            ax.grid(alpha=0.3)
        axes[1].set_xticks(xs, labels, rotation=90, fontsize=8)
        fig.suptitle("Sensitivity S(k=1, j) x DoF range, per DoF (S at rotator 0 deg, "
                     f"OCS). Unit pure DZ(1,17) by least squares: achieved "
                     f"{pure['achieved']:.3f}, leakage rms {pure['leak_rms']:.3f} "
                     f"(dimensionless, per µm), largest DoF use "
                     f"{pure['max_range_frac']:.2f} of range ({pure['worst_dof']})",
                     fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: correctable variance fraction for every (k, j).
        fig, ax = plt.subplots(figsize=(17, 7))
        piv = ct.pivot(index="k", columns="j", values="corr_var_frac")
        im = ax.imshow(piv.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for (a, b), v in np.ndenumerate(piv.to_numpy()):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if v < 0.5 else "k")
        ax.set_xticks(range(len(piv.columns)), [f"Z{j}" for j in piv.columns])
        ax.set_yticks(range(6), [f"k={k}" for k in piv.index])
        fig.colorbar(im, label="1 - var(resid)/var(raw) (dimensionless, over visits)")
        ax.set_title("Fraction of per-visit DZ variance the 50 DoF reproduce")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(pd.DataFrame(loo_rows).round(4).to_string(index=False))
    print(pure)
    print(srow.reindex(srow.dz_1_17_at_range_um.abs().sort_values(ascending=False)
                       .index).head(10).round(4).to_string(index=False))
    print(ct[ct.k == 1].round(4).to_string(index=False))
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
