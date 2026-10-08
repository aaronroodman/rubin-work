"""Compare three optical-state recovery schemes behind the same MIW build.

The Measured Intrinsic Wavefront (MIW) build recovers an optical state per
visit and subtracts the wavefront that state reproduces, so the recovery scheme
sets what the MIW keeps.  Three schemes are compared here, all built on the
same param_set, the same rotator bins and the same wavefronts:

====================  ==========================================================
scheme                recovery
====================  ==========================================================
22/12                 the canonical reduced set -- 10 rigid body + M1M3 bending
                      1-7 + M2 bending 1-5 -- truncated to 12 v-modes.  No
                      range barrier: the reduced basis is expected to keep every
                      degree of freedom (DOF) inside its allowed range `r_j`.
50/34 no RBR          all 50 DOF truncated to 34 v-modes, truncation the only
                      regularizer.  Reaches the largest correction, but leaves
                      bending modes tens of times outside `r_j`.
50/50 with RBR        all 50 DOF, every v-mode retained, with the Range-Bounded
                      Recovery (RBR) barrier as the only regularizer.
====================  ==========================================================

Three comparisons are drawn, each over the visits common to all three schemes:

1. **MIW per pupil Noll term, side by side.**  One page per term, the three
   schemes' field maps on a shared colour scale so the maps are comparable by
   eye, plus the inferred full width at half maximum (FWHM) in arcsec.
2. **Recovered DOF against ordinal visit number**, as the dimensionless ratio
   `d_j / r_j`, with the range boundary drawn.  This is where a scheme that
   cannot be realized on the telescope shows itself, and where the 22/12
   in-range expectation is checked rather than assumed.
3. **Residual DZ(k, j) after correction against ordinal visit number** -- what
   each scheme fails to correct, in um of wavefront.  A scheme correcting more
   leaves less here.

Ordinal visit number is the index into the visits sorted by `(day_obs,
seq_num)`, not a visit identifier; it exists so a long run plots legibly.

`r_j`, the solvers and the forward map all come from `lsst.ts.ofc`, which owns
RBR.  The per-arm loader and estimator are imported from
`compare_rbr_arms.py` rather than duplicated; that script does the pairwise
unconstrained-against-RBR comparison in more depth, including the achieved
residual and the per-v-mode shift.

Usage
-----
    python code/miw/compare_schemes.py \
        --out-dir output/miw/danish_1_3_v1000_schemes \
        --label "Danish 1.3 blitz, v1000 pupil model"
"""

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # headless: this runs over ssh
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # aos/code

import aos_state  # noqa: E402
from miw.compare_rbr_arms import build_estimator, load_arm  # noqa: E402

# The five in-family rotator bins, matching the 5rot split entries.
ROT_5 = ("rot_-65_-55", "rot_-20_-10", "rot_-3_3", "rot_10_20", "rot_55_65")

# One entry per scheme: the key, the directory under output/miw/, the DOF index
# set and the retained v-mode count.  Order is the plotting order.
SCHEMES = (
    ("22/12", "danish_1_3_v1000_A_22_12_i", aos_state.DOF22, 12),
    ("50/34 no RBR", "danish_1_3_v1000_A_50_34_i", 50, 34),
    ("50/50 RBR", "danish_1_3_v1000_A_50_50_i_rbr", 50, 50),
)

COLORS = {"22/12": "#2ca02c", "50/34 no RBR": "#1f77b4", "50/50 RBR": "#d62728"}


def load_all(miw_root, rotator_select, ofc_normalization_yaml=None):
    """Load every scheme's per-visit DOF and DZ wavefronts.

    Returns
    -------
    arms : `dict`
        Scheme key -> ``(estimator, DataFrame)``, the frame as `load_arm`
        returns it.
    """
    arms = {}
    for key, dirname, n_dof, n_keep in SCHEMES:
        build_dir = pathlib.Path(miw_root) / dirname / "build"
        if not build_dir.is_dir():
            raise RuntimeError(f"{key}: no build directory at {build_dir}")
        estimator = build_estimator(
            n_dof=n_dof, n_keep=n_keep,
            ofc_normalization_yaml=ofc_normalization_yaml)
        arms[key] = (estimator, load_arm(build_dir, estimator,
                                         rotator_select=rotator_select))
    return arms


def shared_visits(arms):
    """The `(day_obs, seq_num)` visits present in every scheme, sorted.

    Sorting is what makes the ordinal visit number well defined and the same
    across schemes, so the per-visit plots are aligned.
    """
    key = ["day_obs", "seq_num"]
    shared = None
    for _, df in arms.values():
        visits = df[key].drop_duplicates()
        shared = visits if shared is None else pd.merge(shared, visits, on=key)
    return shared.sort_values(key).reset_index(drop=True)


def aligned(df, visits):
    """One row per entry of `visits`, in that order."""
    return pd.merge(visits, df, on=["day_obs", "seq_num"], how="left")


def dof_ratio_table(arms, visits):
    """Per-visit `d_j / r_j` for every scheme, long form.

    Returns
    -------
    `pandas.DataFrame`
        Columns ``scheme``, ``visit_ordinal``, ``dof_label``, ``ratio``
        (dimensionless, recovered amplitude over allowed range) and
        ``amplitude`` in each DOF's own unit.
    """
    rows = []
    for key, (estimator, df) in arms.items():
        labels, _ = estimator.dof_labels()
        ranges = estimator.dof_ranges()
        use = aligned(df, visits)
        for label, r_j in zip(labels, ranges):
            values = np.asarray(use[label], dtype=float)
            rows.append(pd.DataFrame({
                "scheme": key,
                "visit_ordinal": np.arange(len(use)),
                "dof_label": label,
                "amplitude": values,
                "ratio": values / r_j,
            }))
    return pd.concat(rows, ignore_index=True)


def dof_range_summary(dof_long):
    """Largest |d_j|/r_j per scheme and per DOF, worst DOF first."""
    grouped = (dof_long.assign(abs_ratio=lambda d: d["ratio"].abs())
               .groupby(["scheme", "dof_label"], sort=False)["abs_ratio"]
               .agg(max_abs_ratio="max", median_abs_ratio="median")
               .reset_index())
    return grouped.sort_values("max_abs_ratio", ascending=False)


def residual_dz_table(arms, visits, estimator_ref):
    """Per-visit residual DZ amplitude left after each scheme's correction.

    The residual is ``dz_raw - dz_sub`` summed in quadrature over the
    ``(k, j)`` grid, in um of wavefront -- the wavefront the scheme did not
    correct.  Also returned per pupil Noll term at focal order ``k = 1``, the
    term the MIW maps show.

    Returns
    -------
    total : `pandas.DataFrame`
        Columns ``scheme``, ``visit_ordinal``, ``residual_rms_um``.
    per_term : `pandas.DataFrame`
        Columns ``scheme``, ``visit_ordinal``, ``noll``, ``residual_um``.
    """
    kj = estimator_ref.kj_grid
    k1_cols = [i for i, (k, _) in enumerate(kj) if k == 1]
    k1_noll = [j for (k, j) in kj if k == 1]

    totals, terms = [], []
    for key, (_, df) in arms.items():
        use = aligned(df, visits)
        raw = np.stack(use["dz_raw"].values)
        sub = np.stack(use["dz_sub"].values)
        residual = raw - sub
        totals.append(pd.DataFrame({
            "scheme": key,
            "visit_ordinal": np.arange(len(use)),
            "residual_rms_um": np.sqrt((residual ** 2).sum(axis=1)),
        }))
        for col, j in zip(k1_cols, k1_noll):
            terms.append(pd.DataFrame({
                "scheme": key,
                "visit_ordinal": np.arange(len(use)),
                "noll": j,
                "residual_um": residual[:, col],
            }))
    return (pd.concat(totals, ignore_index=True),
            pd.concat(terms, ignore_index=True))


def miw_grids(miw_root, rotator_select):
    """Each scheme's MIW field map, averaged over the selected rotator bins.

    The build writes one grid per rotator bin, and the bins do NOT share a row
    set: each keeps only the field cells with enough donuts, so the point count
    varies (3842 to 3879 over the nine bins of `danish_1_3_v1000`).  Averaging
    therefore keys on the field position rather than the row index, and keeps
    the positions common to every selected bin -- the same positions in every
    scheme, since the schemes differ only in the recovery.

    Returns
    -------
    grids : `dict`
        Scheme key -> ``(thx_deg, thy_deg, zk, noll)``, ``zk`` being
        ``(n_field_points, n_noll)`` in um of wavefront.
    """
    grids = {}
    for key, dirname, _, _ in SCHEMES:
        build_dir = pathlib.Path(miw_root) / dirname / "build"
        per_bin, noll = {}, None
        for sub in sorted(build_dir.glob("rot_*")):
            if rotator_select and sub.name not in rotator_select:
                continue
            path = sub / "intrinsic_grid.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if noll is None:
                noll = np.asarray(df["nollIndices"].iloc[0], dtype=int)
            # Round the field angle to bin the same physical cell together
            # across bins; the grid is on 73x73 so this is far below one cell.
            position = list(zip(np.round(np.asarray(df["thx_deg"], float), 6),
                                np.round(np.asarray(df["thy_deg"], float), 6)))
            per_bin[sub.name] = dict(zip(position, np.stack(df["zk"].values)))
        if not per_bin:
            raise RuntimeError(f"{key}: no intrinsic_grid.parquet under {build_dir}")

        shared = set.intersection(*(set(b) for b in per_bin.values()))
        if not shared:
            raise RuntimeError(f"{key}: the selected rotator bins share no "
                               "field positions")
        ordered = sorted(shared)
        zk = np.stack([
            np.nanmean(np.stack([per_bin[b][pos] for b in per_bin]), axis=0)
            for pos in ordered])
        thx = np.array([p[0] for p in ordered])
        thy = np.array([p[1] for p in ordered])
        grids[key] = (thx, thy, zk, noll)

    # The schemes must end on the same field positions, or the side-by-side
    # maps and any difference between them would compare unlike points.
    reference = next(iter(grids))
    for key, (thx, thy, _, _) in grids.items():
        if not (np.array_equal(thx, grids[reference][0])
                and np.array_equal(thy, grids[reference][1])):
            raise RuntimeError(
                f"{key} and {reference} resolved different field positions; "
                "the schemes must be built on the same rotator bins")
    return grids


def _field_map(ax, thx, thy, values, title, vlim):
    """One MIW field map, OCS field angle in deg."""
    im = ax.scatter(thx, thy, c=values, s=7, cmap="RdBu_r",
                    vmin=-vlim, vmax=vlim, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("thx (deg, OCS)", fontsize=7)
    ax.set_ylabel("thy (deg, OCS)", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MIW (um of wavefront)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return im


def plot_miw_side_by_side(pdf, grids, label):
    """One page per pupil Noll term: the three schemes on a shared scale.

    The shared scale is what makes the three comparable; a per-panel scale
    would hide a scheme leaving several times more wavefront behind.
    """
    keys = [k for k, _, _, _ in SCHEMES if k in grids]
    noll = grids[keys[0]][3]
    for term_i, j in enumerate(noll):
        values = {k: grids[k][2][:, term_i] for k in keys}
        finite = np.concatenate([v[np.isfinite(v)] for v in values.values()])
        if finite.size == 0:
            continue
        vlim = float(np.nanpercentile(np.abs(finite), 99.0)) or 1.0

        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.0))
        for ax, key in zip(np.atleast_1d(axes), keys):
            thx, thy, _, _ = grids[key]
            rms = float(np.sqrt(np.nanmean(values[key] ** 2)))
            _field_map(ax, thx, thy, values[key],
                       f"{key}\nRMS over field {rms:.4f} um of wavefront", vlim)
        fig.suptitle(f"{label}\nMIW, pupil Noll Z{j}"
                     f" (shared scale +-{vlim:.4f} um of wavefront)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        pdf.savefig(fig)
        plt.close(fig)


def plot_miw_fwhm(pdf, grids, label):
    """Inferred FWHM in arcsec per scheme, on a shared scale."""
    from lsst.ts.wep.utils import convertZernikesToPsfWidth

    keys = [k for k, _, _, _ in SCHEMES if k in grids]
    fwhm = {}
    for key in keys:
        thx, thy, zk, noll = grids[key]
        good = np.isfinite(zk).all(axis=1)
        widths = np.full(len(zk), np.nan)
        widths[good] = np.sqrt(
            (np.stack([convertZernikesToPsfWidth(z, jmin=int(noll[0]))
                       for z in zk[good]]) ** 2).sum(axis=1))
        fwhm[key] = widths

    finite = np.concatenate([v[np.isfinite(v)] for v in fwhm.values()])
    vmax = float(np.nanpercentile(finite, 99.0))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.0))
    for ax, key in zip(np.atleast_1d(axes), keys):
        thx, thy, _, _ = grids[key]
        im = ax.scatter(thx, thy, c=fwhm[key], s=7, cmap="viridis",
                        vmin=0.0, vmax=vmax, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"{key}\nmedian {np.nanmedian(fwhm[key]):.4f} arcsec",
                     fontsize=8)
        ax.set_xlabel("thx (deg, OCS)", fontsize=7)
        ax.set_ylabel("thy (deg, OCS)", fontsize=7)
        ax.tick_params(labelsize=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("inferred FWHM (arcsec)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"{label}\nMIW inferred FWHM, shared scale 0 to "
                 f"{vmax:.4f} arcsec", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    pdf.savefig(fig)
    plt.close(fig)


def plot_dof_vs_visit(pdf, dof_long, label, per_page=6):
    """Recovered `d_j / r_j` against ordinal visit number, per DOF.

    The range boundary at ratio 1 is drawn so an out-of-range scheme is
    visible without reading the axis.  DOF are ordered by the largest excess
    any scheme reaches, so the interesting ones come first.
    """
    order = (dof_long.assign(a=lambda d: d["ratio"].abs())
             .groupby("dof_label", sort=False)["a"].max()
             .sort_values(ascending=False).index.tolist())

    for start in range(0, len(order), per_page):
        chunk = order[start:start + per_page]
        n_rows = int(np.ceil(len(chunk) / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=(11.0, 2.3 * n_rows),
                                 squeeze=False)
        for ax, dof_label in zip(axes.ravel(), chunk):
            sub = dof_long[dof_long["dof_label"] == dof_label]
            for key, _, _, _ in SCHEMES:
                one = sub[sub["scheme"] == key]
                if one.empty:
                    continue
                ax.plot(one["visit_ordinal"], one["ratio"], ".",
                        ms=2.5, color=COLORS[key], label=key)
            for sign in (-1.0, 1.0):
                ax.axhline(sign, color="k", lw=0.7, ls="--")
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.set_title(f"{dof_label}", fontsize=8)
            ax.set_xlabel("ordinal visit number", fontsize=7)
            ax.set_ylabel("$d_j / r_j$ (dimensionless)", fontsize=7)
            ax.tick_params(labelsize=6)
            # Symmetric log keeps both a ratio of 0.01 and one of 50 legible;
            # the unconstrained arm spans that whole span across DOF.
            ax.set_yscale("symlog", linthresh=1.0)
        for ax in axes.ravel()[len(chunk):]:
            ax.set_visible(False)
        axes[0, 0].legend(fontsize=6, markerscale=3, loc="best")
        fig.suptitle(f"{label}\nRecovered DOF against allowed range;"
                     " dashed line is the range boundary", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)


def plot_residual_vs_visit(pdf, resid_total, resid_terms, label, per_page=6):
    """Residual DZ after correction against ordinal visit number.

    One page of the total over the ``(k, j)`` grid, then one panel per pupil
    Noll term at focal order ``k = 1``.
    """
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    for key, _, _, _ in SCHEMES:
        one = resid_total[resid_total["scheme"] == key]
        if one.empty:
            continue
        ax.plot(one["visit_ordinal"], one["residual_rms_um"], ".",
                ms=3, color=COLORS[key],
                label=f"{key} (median {one['residual_rms_um'].median():.4f} um)")
    ax.set_xlabel("ordinal visit number", fontsize=8)
    ax.set_ylabel("residual DZ, quadrature sum over (k, j)\n(um of wavefront)",
                  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title(f"{label}\nWavefront left uncorrected per visit", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    terms = sorted(resid_terms["noll"].unique())
    for start in range(0, len(terms), per_page):
        chunk = terms[start:start + per_page]
        n_rows = int(np.ceil(len(chunk) / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=(11.0, 2.3 * n_rows),
                                 squeeze=False)
        for ax, j in zip(axes.ravel(), chunk):
            sub = resid_terms[resid_terms["noll"] == j]
            for key, _, _, _ in SCHEMES:
                one = sub[sub["scheme"] == key]
                if one.empty:
                    continue
                ax.plot(one["visit_ordinal"], one["residual_um"], ".",
                        ms=2.5, color=COLORS[key], label=key)
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.set_title(f"residual DZ(k=1, j={j})", fontsize=8)
            ax.set_xlabel("ordinal visit number", fontsize=7)
            ax.set_ylabel("um of wavefront", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(chunk):]:
            ax.set_visible(False)
        axes[0, 0].legend(fontsize=6, markerscale=3, loc="best")
        fig.suptitle(f"{label}\nResidual DZ after correction,"
                     " focal order k = 1", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)


def report(label, visits, dof_summary, resid_total, grids):
    """Print the numbers the plots show."""
    print(f"\n{label}")
    print(f"Visits common to all three schemes: {len(visits)}")

    print("\nLargest |d_j|/r_j per scheme (dimensionless, recovered amplitude"
          " over allowed range):")
    for key, _, _, _ in SCHEMES:
        sub = dof_summary[dof_summary["scheme"] == key]
        if sub.empty:
            continue
        worst = sub.iloc[0]
        n_over = int((sub["max_abs_ratio"] > 1.0).sum())
        print(f"  {key:>14}: worst {worst['max_abs_ratio']:.3f} on "
              f"{worst['dof_label']}; {n_over} of {len(sub)} DOF exceed range")

    print("\nResidual DZ left uncorrected, quadrature sum over (k, j),"
          " um of wavefront:")
    for key, _, _, _ in SCHEMES:
        one = resid_total[resid_total["scheme"] == key]
        if one.empty:
            continue
        print(f"  {key:>14}: median {one['residual_rms_um'].median():.4f},"
              f" mean {one['residual_rms_um'].mean():.4f}")

    print("\nMIW RMS over the field, quadrature sum over pupil Noll terms,"
          " um of wavefront:")
    for key, _, _, _ in SCHEMES:
        if key not in grids:
            continue
        zk = grids[key][2]
        good = np.isfinite(zk).all(axis=1)
        print(f"  {key:>14}: "
              f"{float(np.sqrt((zk[good] ** 2).sum(axis=1).mean())):.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--miw-root", default="output/miw",
                    help="directory holding the per-scheme MIW directories")
    ap.add_argument("--out-dir", required=True,
                    help="where the PDF and the tables are written")
    ap.add_argument("--label", default="MIW scheme comparison",
                    help="title text identifying the build")
    ap.add_argument("--rotator-select", default="5rot",
                    choices=("5rot", "all"),
                    help="'5rot' uses the five in-family bins; 'all' uses nine")
    ap.add_argument("--ofc-normalization-yaml", default=None,
                    help="normalization weights; default is the build's own")
    args = ap.parse_args()

    rotator_select = ROT_5 if args.rotator_select == "5rot" else None
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = load_all(args.miw_root, rotator_select, args.ofc_normalization_yaml)
    visits = shared_visits(arms)
    if visits.empty:
        raise RuntimeError("No visits are common to all three schemes.")

    dof_long = dof_ratio_table(arms, visits)
    dof_summary = dof_range_summary(dof_long)
    # The (k, j) grid is the same for every scheme -- it is set by k_min/k_max
    # and the pupil Noll list, not by the DOF set -- so any estimator supplies it.
    reference = arms[SCHEMES[1][0]][0]
    resid_total, resid_terms = residual_dz_table(arms, visits, reference)
    grids = miw_grids(args.miw_root, rotator_select)

    dof_long.to_parquet(out_dir / "scheme_dof_per_visit.parquet", index=False)
    dof_summary.to_parquet(out_dir / "scheme_dof_summary.parquet", index=False)
    resid_total.to_parquet(out_dir / "scheme_residual_per_visit.parquet",
                           index=False)
    resid_terms.to_parquet(out_dir / "scheme_residual_per_term.parquet",
                           index=False)

    report(args.label, visits, dof_summary, resid_total, grids)

    out_pdf = out_dir / "scheme_comparison.pdf"
    with PdfPages(out_pdf) as pdf:
        plot_miw_fwhm(pdf, grids, args.label)
        plot_miw_side_by_side(pdf, grids, args.label)
        plot_dof_vs_visit(pdf, dof_long, args.label)
        plot_residual_vs_visit(pdf, resid_total, resid_terms, args.label)
    print(f"\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
