"""Compare an unconstrained MIW build against its Range-Bounded Recovery arm.

The Measured Intrinsic Wavefront (MIW) build recovers a 50-degree-of-freedom
(DOF) optical state per visit and subtracts the wavefront that state
reproduces.  The unconstrained build regularizes that recovery by truncation
alone, so nothing stops a recovered amplitude exceeding the stroke the hexapods
and mirrors can physically reach.  Range-Bounded Recovery (RBR) adds a
superlinear penalty on leaving the allowed range `r_j`; `check_dof_ranges.py`
measures the unconstrained excess, and this script compares the two arms once
both have been built.

Four things are reported, per the step B plan:

1. The recovered DOF against `r_j`, per arm -- the quantity RBR acts on.
2. The achieved residual `|| dW - S d ||` through the same rank-limited
   operator both recoveries used.  A subspace projection would instead credit
   the constrained arm with wavefront it does not reproduce, so this is the
   honest comparison metric.
3. The MIW itself, as inferred full width at half maximum (FWHM) in arcsec via
   ts_wep `convertZernikesToPsfWidth`, plus each pupil Zernike term's share of
   the difference power.
4. How the subtracted correction changes, per DOF and per v-mode.

Both arms must be built on the same param_set and the same rotator bins, so the
comparison runs visit by visit on the common `(day_obs, seq_num)` set.

**Check both arms converged before reading the MIW difference.**  The build
iterates, feeding the MIW grid back onto the donuts each pass, and RBR
converges more slowly than the unconstrained recovery: on the `rot_-3_3` bin at
`n_iter` 3 the unconstrained arm settled to 8.92e-04 um of wavefront between
iterations while the RBR arm was still moving by 5.42e-03 um, above the
1.0e-03 um tolerance the build warns at.  An unconverged arm's MIW is an
iterate, not a result.  The build prints `MI convergence (RMS Δ between iters)`
and warns when the last step exceeds tolerance; read that line in each arm's
log.

`r_j` comes from `lsst.ts.ofc.OFCData.dof_ranges`, and the solvers and the
forward map from `lsst.ts.ofc`, which owns RBR.  Nothing is re-derived here.

With `--plots` the same comparison is drawn to a PDF: the MIW field maps per
arm and their difference for the terms carrying the difference power, the
per-DOF amplitude against `r_j`, and the per-v-mode coefficient shift.  The
per-arm build PDFs that `intrinsic_build_plots.py` already writes show each arm
on its own; these show the two against each other.

Usage
-----
    python code/miw/compare_rbr_arms.py \
        --unconstrained output/miw/danish_1_3_v1000_A_50_34_i/build \
        --rbr output/miw/danish_1_3_v1000_A_50_34_i_rbr/build \
        --label "Danish 1.3 blitz, v1000 pupil model" \
        --out-dir output/miw/danish_1_3_v1000_rbr_arms \
        --plots
"""

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # headless: this runs under Snakemake and over ssh
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # aos/code

from miw_io import JS_DEFAULT  # noqa: E402

# The five in-family rotator bins the `_5rot` entries decompose, in the
# `rot_<lo>_<hi>` directory-name form the build writes.
ROT_5 = ("rot_-65_-55", "rot_-20_-10", "rot_-3_3", "rot_10_20", "rot_55_65")

# The MIW build's own SVD configuration, from mi_config.yaml.
N_DOF = 50
N_KEEP = 34
K_MIN = 1
K_MAX = 6


def build_estimator(js=JS_DEFAULT, n_dof=N_DOF, n_keep=N_KEEP,
                    k_min=K_MIN, k_max=K_MAX, ofc_normalization_yaml=None):
    """The build's own DZ state estimator, for `r_j` and the solvers.

    Returns
    -------
    estimator : `lsst.ts.ofc.DoubleZernikeStateEstimator`
        Configured as the build configures it, including the normalization
        weights the build uses (which are not ts_ofc's configured default).
    """
    from lsst.ts.intrinsic.wavefront import ofc_svd as osv
    from lsst.ts.ofc import DoubleZernikeStateEstimator, OFCData

    weights = osv.load_normalization_weights(
        ofc_normalization_yaml, osv.DEFAULT_NORM_YAML)
    return DoubleZernikeStateEstimator(
        OFCData("lsst"), js, k_min, k_max, n_keep, n_dof=n_dof,
        normalization_weights=weights)


def load_arm(build_dir, estimator, rotator_select=None):
    """Per-visit DOF, raw DZ wavefront and subtracted DZ wavefront for one arm.

    Parameters
    ----------
    build_dir : `str` or `pathlib.Path`
        The arm's ``build/`` directory, holding ``rot_*/dz_fits.parquet``.
    estimator : `lsst.ts.ofc.DoubleZernikeStateEstimator`
        Supplies the DOF labels and the ``(k, j)`` row order.
    rotator_select : `tuple` of `str`, optional
        Rotator-bin directory names to keep.  `None` keeps every bin present.

    Returns
    -------
    `pandas.DataFrame`
        One row per visit, with ``day_obs``, ``seq_num``, ``rot_bin``, the DOF
        columns, and the stacked wavefront matrices as object columns
        ``dz_raw`` and ``dz_sub``, both in um of wavefront.
    """
    build_dir = pathlib.Path(build_dir)
    labels, _ = estimator.dof_labels()
    kj = estimator.kj_grid
    rows = []
    for sub in sorted(build_dir.glob("rot_*")):
        if rotator_select and sub.name not in rotator_select:
            continue
        fits = sub / "dz_fits.parquet"
        if not fits.exists():
            continue
        df = pd.read_parquet(fits)
        # The table stores the subtracted (projected) fit as `dz_corr_*` and
        # what it left behind as `dz_resid_* = raw - corr`, so the raw fitted
        # wavefront both arms inverted is their sum.  There is no `dz_raw_*`.
        sub_wf = _stack(df, "dz_corr", kj)
        raw = sub_wf + _stack(df, "dz_resid", kj)
        for i in range(len(df)):
            row = {"day_obs": int(df["day_obs"].iloc[i]),
                   "seq_num": int(df["seq_num"].iloc[i]),
                   "rot_bin": sub.name,
                   "dz_raw": raw[i], "dz_sub": sub_wf[i]}
            for label in labels:
                row[label] = float(df[f"dof_{label}"].iloc[i])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No dz_fits.parquet found under {build_dir}")
    return pd.DataFrame(rows)


def _stack(df, prefix, kj):
    """Pack per-visit DZ columns into ``(n_visits, n_kj)`` in `kj` row order."""
    out = np.full((len(df), len(kj)), np.nan)
    for col_i, (k, j) in enumerate(kj):
        col = f"{prefix}_z{j}_c{k}"
        if col in df.columns:
            out[:, col_i] = np.asarray(df[col], dtype=float)
    return np.nan_to_num(out)


def common_visits(arm_a, arm_b):
    """Restrict both arms to their shared visits, in a common row order."""
    key = ["day_obs", "seq_num"]
    shared = pd.merge(arm_a[key], arm_b[key], on=key, how="inner")
    a = arm_a.merge(shared, on=key).sort_values(key).reset_index(drop=True)
    b = arm_b.merge(shared, on=key).sort_values(key).reset_index(drop=True)
    return a, b


def dof_range_table(arm_a, arm_b, estimator):
    """Per-DOF comparison of the recovered amplitude against `r_j`.

    Returns
    -------
    `pandas.DataFrame`
        One row per DOF: the median and maximum ``|d_j| / r_j``
        (dimensionless, recovered amplitude over allowed range) and the
        fraction of visits outside range, for each arm.
    """
    labels, units = estimator.dof_labels()
    ranges = estimator.dof_ranges()
    mat_a = np.column_stack([arm_a[label].values for label in labels])
    mat_b = np.column_stack([arm_b[label].values for label in labels])
    ratio_a = np.abs(mat_a) / ranges
    ratio_b = np.abs(mat_b) / ranges
    return pd.DataFrame({
        "dof": labels,
        "unit": units,
        "range_r_j": ranges,
        "median_ratio_unconstrained": np.median(ratio_a, axis=0),
        "median_ratio_rbr": np.median(ratio_b, axis=0),
        "max_ratio_unconstrained": ratio_a.max(axis=0),
        "max_ratio_rbr": ratio_b.max(axis=0),
        "frac_outside_unconstrained": (ratio_a > 1.0).mean(axis=0),
        "frac_outside_rbr": (ratio_b > 1.0).mean(axis=0),
    })


def residual_table(arm_a, arm_b, estimator):
    """Per-visit achieved residual for both arms.

    Each arm's residual is evaluated against **its own** raw fitted DZ
    wavefront.  The two are not the same vector: the build iterates, feeding
    the MIW grid back onto the donuts each pass, so the arms' fitted
    wavefronts diverge after the first iteration (measured at up to 0.1174 um
    of wavefront on the ``rot_-3_3`` bin).  Each number therefore answers "how
    much of what this arm measured does its own recovered state explain",
    which is the comparable question; a single shared reference would mix in
    the iteration's divergence.
    """
    from lsst.ts.ofc import achieved_residual

    labels, _ = estimator.dof_labels()
    sigma = estimator.Sigma[estimator.keep_idx]
    v_retained = estimator.V[:, estimator.keep_idx]
    weights = estimator.normalization_weights
    mat_a = np.column_stack([arm_a[label].values for label in labels])
    mat_b = np.column_stack([arm_b[label].values for label in labels])

    rows = []
    for i in range(len(arm_a)):
        rows.append({
            "day_obs": int(arm_a["day_obs"].iloc[i]),
            "seq_num": int(arm_a["seq_num"].iloc[i]),
            "residual_unconstrained": achieved_residual(
                arm_a["dz_raw"].iloc[i], mat_a[i],
                estimator.U_eff, sigma, v_retained, weights),
            "residual_rbr": achieved_residual(
                arm_b["dz_raw"].iloc[i], mat_b[i],
                estimator.U_eff, sigma, v_retained, weights),
            "raw_rms_unconstrained": float(
                np.sqrt(np.mean(arm_a["dz_raw"].iloc[i] ** 2))),
            "raw_rms_rbr": float(np.sqrt(np.mean(arm_b["dz_raw"].iloc[i] ** 2))),
            "subtracted_rms_unconstrained": float(
                np.sqrt(np.mean(arm_a["dz_sub"].iloc[i] ** 2))),
            "subtracted_rms_rbr": float(
                np.sqrt(np.mean(arm_b["dz_sub"].iloc[i] ** 2))),
        })
    return pd.DataFrame(rows)


def vmode_table(arm_a, arm_b, estimator):
    """Per-v-mode comparison of the two arms' recovered states.

    The states are projected back onto the v-mode basis, so the comparison
    says *which* modes RBR moved rather than only which DOF.
    """
    labels, _ = estimator.dof_labels()
    mat_a = np.column_stack([arm_a[label].values for label in labels])
    mat_b = np.column_stack([arm_b[label].values for label in labels])
    v_retained = estimator.V[:, estimator.keep_idx]
    weights = estimator.normalization_weights
    coeff_a = (mat_a / weights) @ v_retained
    coeff_b = (mat_b / weights) @ v_retained
    return pd.DataFrame({
        "vmode": np.arange(1, coeff_a.shape[1] + 1),
        "sigma": estimator.Sigma[estimator.keep_idx],
        "median_abs_unconstrained": np.median(np.abs(coeff_a), axis=0),
        "median_abs_rbr": np.median(np.abs(coeff_b), axis=0),
        "median_abs_change": np.median(np.abs(coeff_b - coeff_a), axis=0),
    })


def miw_table(grid_a, grid_b):
    """The two arms' MIW grids as inferred FWHM, and the difference per term.

    Parameters
    ----------
    grid_a, grid_b : `str` or `pathlib.Path`
        ``intrinsic_grid.parquet`` for the unconstrained and RBR arms.

    Returns
    -------
    summary : `dict`
        Field-averaged quantities, um of wavefront and arcsec.
    per_term : `pandas.DataFrame`
        Per pupil Noll term: the RMS over the field in each arm, and the term's
        share of the difference power (dimensionless).
    """
    from lsst.ts.wep.utils import convertZernikesToPsfWidth

    df_a = pd.read_parquet(grid_a)
    df_b = pd.read_parquet(grid_b)
    noll = np.asarray(df_a["nollIndices"].iloc[0], dtype=int)
    zk_a = np.stack(df_a["zk"].values)
    zk_b = np.stack(df_b["zk"].values)
    if not np.array_equal(np.asarray(df_a["thx_deg"]), np.asarray(df_b["thx_deg"])):
        raise RuntimeError("The two arms' field grids differ; cannot difference.")

    good = np.isfinite(zk_a).all(axis=1) & np.isfinite(zk_b).all(axis=1)
    zk_a, zk_b = zk_a[good], zk_b[good]
    diff = zk_b - zk_a

    fwhm_a = np.sqrt(
        (np.stack([convertZernikesToPsfWidth(z, jmin=int(noll[0])) for z in zk_a]) ** 2
         ).sum(axis=1))
    fwhm_b = np.sqrt(
        (np.stack([convertZernikesToPsfWidth(z, jmin=int(noll[0])) for z in zk_b]) ** 2
         ).sum(axis=1))

    power = (diff ** 2).sum(axis=0)
    summary = {
        "n_field_points": int(good.sum()),
        "miw_rms_unconstrained_um": float(np.sqrt((zk_a ** 2).sum(axis=1).mean())),
        "miw_rms_rbr_um": float(np.sqrt((zk_b ** 2).sum(axis=1).mean())),
        "miw_difference_rms_um": float(np.sqrt((diff ** 2).sum(axis=1).mean())),
        "fwhm_median_unconstrained_arcsec": float(np.median(fwhm_a)),
        "fwhm_median_rbr_arcsec": float(np.median(fwhm_b)),
    }
    per_term = pd.DataFrame({
        "noll": noll,
        "rms_unconstrained_um": np.sqrt((zk_a ** 2).mean(axis=0)),
        "rms_rbr_um": np.sqrt((zk_b ** 2).mean(axis=0)),
        "share_of_difference_power": power / power.sum(),
    }).sort_values("share_of_difference_power", ascending=False)
    return summary, per_term


def _field_map(ax, thx, thy, values, title, clabel, cmap="RdBu_r",
               vlim=None):
    """One field map, OCS field angle in deg, as a scatter of grid points."""
    if vlim is None:
        vlim = float(np.nanpercentile(np.abs(values), 99.0)) or 1.0
    im = ax.scatter(thx, thy, c=values, s=7, cmap=cmap,
                    vmin=-vlim, vmax=vlim, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("thx (deg, OCS)", fontsize=7)
    ax.set_ylabel("thy (deg, OCS)", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(clabel, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def plot_miw_arms(pdf, grid_a, grid_b, miw_terms, label, n_terms=6):
    """MIW field maps for both arms and their difference.

    One page of inferred FWHM, then one page per pupil Noll term carrying the
    most difference power: unconstrained, RBR, and RBR minus unconstrained on a
    shared symmetric colour scale so the difference is read against the signal.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open PDF to append pages to.
    grid_a, grid_b : `str` or `pathlib.Path`
        ``intrinsic_grid.parquet`` for the unconstrained and RBR arms.
    miw_terms : `pandas.DataFrame`
        Per-term table from `miw_table`, used to order the term pages.
    label : `str`
        Build label for the page titles.
    n_terms : `int`
        How many pupil terms to draw, highest difference power first.
    """
    from lsst.ts.wep.utils import convertZernikesToPsfWidth

    df_a = pd.read_parquet(grid_a)
    df_b = pd.read_parquet(grid_b)
    noll = np.asarray(df_a["nollIndices"].iloc[0], dtype=int)
    zk_a = np.stack(df_a["zk"].values)
    zk_b = np.stack(df_b["zk"].values)
    thx = np.asarray(df_a["thx_deg"], dtype=float)
    thy = np.asarray(df_a["thy_deg"], dtype=float)

    good = np.isfinite(zk_a).all(axis=1) & np.isfinite(zk_b).all(axis=1)
    zk_a, zk_b, thx, thy = zk_a[good], zk_b[good], thx[good], thy[good]

    fwhm_a = np.sqrt((np.stack(
        [convertZernikesToPsfWidth(z, jmin=int(noll[0])) for z in zk_a]) ** 2
    ).sum(axis=1))
    fwhm_b = np.sqrt((np.stack(
        [convertZernikesToPsfWidth(z, jmin=int(noll[0])) for z in zk_b]) ** 2
    ).sum(axis=1))

    # FWHM is positive, so it reads better on a sequential scale than on the
    # diverging one the signed Zernike maps use.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    fwhm_hi = float(np.nanpercentile(np.r_[fwhm_a, fwhm_b], 99.0))
    for ax, vals, name in ((axes[0], fwhm_a, "unconstrained"),
                           (axes[1], fwhm_b, "RBR")):
        im = ax.scatter(thx, thy, c=vals, s=7, cmap="viridis",
                        vmin=0.0, vmax=fwhm_hi, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"MIW inferred FWHM, {name}\nmedian "
                     f"{np.median(vals):.4f} arcsec", fontsize=8)
        ax.set_xlabel("thx (deg, OCS)", fontsize=7)
        ax.set_ylabel("thy (deg, OCS)", fontsize=7)
        ax.tick_params(labelsize=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("inferred FWHM (arcsec)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    _field_map(axes[2], thx, thy, fwhm_b - fwhm_a,
               "RBR minus unconstrained\nmedian "
               f"{np.median(fwhm_b - fwhm_a):+.4f} arcsec",
               "d(inferred FWHM) (arcsec)")
    fig.suptitle(f"MIW inferred FWHM, both arms — {label}", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    order = [int(j) for j in miw_terms["noll"].head(n_terms)]
    for j in order:
        col = int(np.where(noll == j)[0][0])
        a, b = zk_a[:, col], zk_b[:, col]
        # Shared scale across all three panels: the difference has to be read
        # against the size of the signal, not auto-scaled to fill its own panel.
        vlim = float(np.nanpercentile(np.abs(np.r_[a, b]), 99.0)) or 1.0
        share = float(miw_terms.loc[miw_terms["noll"] == j,
                                    "share_of_difference_power"].iloc[0])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
        _field_map(axes[0], thx, thy, a,
                   f"Z{j} unconstrained\nRMS {np.sqrt((a ** 2).mean()):.4f} um",
                   "Zernike coefficient (um of wavefront)", vlim=vlim)
        _field_map(axes[1], thx, thy, b,
                   f"Z{j} RBR\nRMS {np.sqrt((b ** 2).mean()):.4f} um",
                   "Zernike coefficient (um of wavefront)", vlim=vlim)
        _field_map(axes[2], thx, thy, b - a,
                   f"Z{j} RBR minus unconstrained\nRMS "
                   f"{np.sqrt(((b - a) ** 2).mean()):.4f} um",
                   "Zernike difference (um of wavefront)", vlim=vlim)
        fig.suptitle(f"MIW pupil Noll Z{j} — {share:.4f} of the difference "
                     f"power (dimensionless) — {label}", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_dof_ranges(pdf, dof_df, label):
    """Per-DOF recovered amplitude against `r_j`, both arms.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open PDF to append the page to.
    dof_df : `pandas.DataFrame`
        Table from `dof_range_table`.
    label : `str`
        Build label for the page title.
    """
    df = dof_df
    med_a, med_b = "median_ratio_unconstrained", "median_ratio_rbr"
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    ax = axes[0]
    ax.semilogy(x, np.maximum(df[med_a], 1e-6), "o-", ms=3, lw=0.8,
                color="tab:red", label="unconstrained")
    ax.semilogy(x, np.maximum(df[med_b], 1e-6), "s-", ms=3, lw=0.8,
                color="tab:blue", label="RBR")
    ax.axhline(1.0, color="k", ls="--", lw=0.8,
               label="allowed range r_j (|d_j|/r_j = 1)")
    ax.set_ylabel("median |d_j| / r_j\n(dimensionless)", fontsize=8)
    ax.set_title("Recovered DOF amplitude against the allowed range, "
                 "median over visits", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ratio = np.asarray(df[med_b]) / np.maximum(np.asarray(df[med_a]), 1e-12)
    ax.semilogy(x, np.maximum(ratio, 1e-4), "o-", ms=3, lw=0.8, color="k")
    ax.axhline(1.0, color="tab:gray", ls="--", lw=0.8)
    ax.set_ylabel("RBR / unconstrained\n(dimensionless)", fontsize=8)
    ax.set_xlabel("degree of freedom", fontsize=8)
    ax.set_title("Factor by which RBR pulls each DOF in "
                 "(below 1 means pulled in)", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["dof"], rotation=90, fontsize=5)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Recovered optical state against range — {label}",
                 fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_vmode_shift(pdf, vmode_df, label):
    """Per-v-mode coefficient shift against the mode's singular value.

    RBR acts on the poorly-conditioned directions, so the shift is expected to
    rise as sigma falls -- that is the mechanism, drawn.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open PDF to append the page to.
    vmode_df : `pandas.DataFrame`
        Table from `vmode_table`.
    label : `str`
        Build label for the page title.
    """
    df = vmode_df
    sig_col, shift_col = "sigma", "median_abs_change"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.asarray(df["vmode"])
    ax = axes[0]
    ax.semilogy(x, np.maximum(np.abs(df[shift_col]), 1e-6), "o-", ms=3,
                lw=0.8, color="tab:purple")
    ax.set_xlabel("retained v-mode index", fontsize=8)
    ax.set_ylabel("median |RBR - unconstrained|\n(v-mode coefficient, "
                  "normalized DOF units)", fontsize=8)
    ax.set_title("v-mode coefficient shift, RBR minus unconstrained",
                 fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(labelsize=6)

    ax = axes[1]
    ax.loglog(np.maximum(df[sig_col], 1e-12),
              np.maximum(np.abs(df[shift_col]), 1e-6), "o", ms=4,
              color="tab:purple")
    ax.set_xlabel("singular value sigma (um of wavefront per DOF unit)",
                  fontsize=8)
    ax.set_ylabel("median |RBR - unconstrained|\n(v-mode coefficient, "
                  "normalized DOF units)", fontsize=8)
    ax.set_title("The shift concentrates at small sigma:\nthe "
                 "poorly-conditioned directions", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(labelsize=6)

    fig.suptitle(f"Where RBR moves the recovery — {label}", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def write_plots(out_pdf, label, dof_df, vmode_df, miw_terms,
                grid_a=None, grid_b=None):
    """Write the arm-comparison PDF.

    Parameters
    ----------
    out_pdf : `str` or `pathlib.Path`
        Destination PDF.
    label : `str`
        Build label for the page titles.
    dof_df, vmode_df : `pandas.DataFrame`
        Tables from `dof_range_table` and `vmode_table`.
    miw_terms : `pandas.DataFrame` or `None`
        Per-term table from `miw_table`; the MIW pages are skipped without it.
    grid_a, grid_b : `str` or `pathlib.Path`, optional
        The two arms' ``intrinsic_grid.parquet``, for the field maps.
    """
    out_pdf = pathlib.Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        if miw_terms is not None and grid_a and grid_b:
            plot_miw_arms(pdf, grid_a, grid_b, miw_terms, label)
        plot_dof_ranges(pdf, dof_df, label)
        plot_vmode_shift(pdf, vmode_df, label)
    print(f"  wrote plots to {out_pdf}")


def report(label, n_visits, dof_df, resid_df, vmode_df, miw_summary, miw_terms):
    """Print the comparison."""
    print(f"\nRBR arm comparison: {label}")
    print(f"  visits common to both arms: {n_visits}")

    print("\n  Recovered DOF against the allowed range r_j (dimensionless,")
    print("  |d_j| / r_j, median over visits), six worst unconstrained DOF:")
    worst = dof_df.sort_values("median_ratio_unconstrained", ascending=False).head(6)
    for _, row in worst.iterrows():
        print(f"    {row['dof']:<8s} unconstrained {row['median_ratio_unconstrained']:8.2f}"
              f"  ->  RBR {row['median_ratio_rbr']:6.3f}"
              f"   (r_j = {row['range_r_j']:.5g} {row['unit']})")
    print(f"    worst over all DOF and visits: "
          f"unconstrained {dof_df['max_ratio_unconstrained'].max():.2f}, "
          f"RBR {dof_df['max_ratio_rbr'].max():.3f}")
    print(f"    fraction of (visit, DOF) pairs outside range: "
          f"unconstrained {dof_df['frac_outside_unconstrained'].mean():.4f}, "
          f"RBR {dof_df['frac_outside_rbr'].mean():.4f}")

    res_u = resid_df["residual_unconstrained"].median()
    res_r = resid_df["residual_rbr"].median()
    print("\n  Achieved residual || dW - S d || (um of wavefront, RMS over the")
    print("  (k, j) grid, median over visits, each arm against its own fit):")
    print(f"    unconstrained {res_u:.5f}, RBR {res_r:.5f}")
    frac_u = (resid_df["residual_unconstrained"]
              / resid_df["raw_rms_unconstrained"]).median()
    frac_r = (resid_df["residual_rbr"] / resid_df["raw_rms_rbr"]).median()
    print(f"    as a fraction of each arm's own fitted wavefront: "
          f"unconstrained {frac_u:.4f}, RBR {frac_r:.4f} "
          f"(dimensionless, amplitude)")
    print(f"    the fitted wavefronts themselves differ, median RMS "
          f"{resid_df['raw_rms_unconstrained'].median():.5f} um against "
          f"{resid_df['raw_rms_rbr'].median():.5f} um: the build iterates, so "
          f"the arms diverge after iteration 1")
    sub_u = resid_df["subtracted_rms_unconstrained"].median()
    sub_r = resid_df["subtracted_rms_rbr"].median()
    print(f"    subtracted wavefront RMS: unconstrained {sub_u:.5f} um, "
          f"RBR {sub_r:.5f} um; ratio {sub_r / sub_u:.4f} "
          f"(dimensionless, amplitude)")

    print("\n  v-modes moved most by the constraint (median |change| in the")
    print("  v-mode coefficient, dimensionless):")
    for _, row in vmode_df.sort_values(
            "median_abs_change", ascending=False).head(5).iterrows():
        print(f"    v{int(row['vmode']):<3d} sigma = {row['sigma']:.4g}  "
              f"median |change| = {row['median_abs_change']:.4g}")

    if miw_summary is not None:
        print("\n  The MIW itself:")
        print(f"    field points: {miw_summary['n_field_points']}")
        print(f"    MIW RMS over the field: unconstrained "
              f"{miw_summary['miw_rms_unconstrained_um']:.4f} um of wavefront, "
              f"RBR {miw_summary['miw_rms_rbr_um']:.4f} um")
        print(f"    difference RMS = {miw_summary['miw_difference_rms_um']:.4f} "
              f"um of wavefront")
        print(f"    inferred FWHM (median over field): unconstrained "
              f"{miw_summary['fwhm_median_unconstrained_arcsec']:.4f} arcsec, "
              f"RBR {miw_summary['fwhm_median_rbr_arcsec']:.4f} arcsec")
        print("    share of the difference power by pupil Noll term "
              "(dimensionless):")
        for _, row in miw_terms.head(6).iterrows():
            print(f"      Noll {int(row['noll']):2d}: "
                  f"{row['share_of_difference_power']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unconstrained", required=True,
                    help="build/ directory of the unconstrained arm")
    ap.add_argument("--rbr", required=True,
                    help="build/ directory of the RBR arm")
    ap.add_argument("--label", default="MIW build",
                    help="label for the printed report")
    ap.add_argument("--out-dir", default=None,
                    help="write the comparison tables here (default: no files)")
    ap.add_argument("--rotator-select", default="all",
                    choices=("all", "5rot"),
                    help="'5rot' restricts to the five in-family rotator bins")
    ap.add_argument("--grid-unconstrained", default=None,
                    help="intrinsic_grid.parquet of the unconstrained arm, for "
                         "the MIW comparison (default: skip it)")
    ap.add_argument("--grid-rbr", default=None,
                    help="intrinsic_grid.parquet of the RBR arm")
    ap.add_argument("--ofc-normalization-yaml", default=None,
                    help="override the build's normalization weights file")
    ap.add_argument("--plots", action="store_true",
                    help="also write rbr_arms_plots.pdf under --out-dir")
    args = ap.parse_args()
    if args.plots and not args.out_dir:
        ap.error("--plots needs --out-dir to write the PDF into")

    rot = ROT_5 if args.rotator_select == "5rot" else None
    estimator = build_estimator(ofc_normalization_yaml=args.ofc_normalization_yaml)

    arm_a = load_arm(args.unconstrained, estimator, rotator_select=rot)
    arm_b = load_arm(args.rbr, estimator, rotator_select=rot)
    arm_a, arm_b = common_visits(arm_a, arm_b)
    if len(arm_a) == 0:
        raise RuntimeError("The two arms share no visits.")

    dof_df = dof_range_table(arm_a, arm_b, estimator)
    resid_df = residual_table(arm_a, arm_b, estimator)
    vmode_df = vmode_table(arm_a, arm_b, estimator)

    miw_summary = miw_terms = None
    if args.grid_unconstrained and args.grid_rbr:
        miw_summary, miw_terms = miw_table(args.grid_unconstrained, args.grid_rbr)

    report(args.label, len(arm_a), dof_df, resid_df, vmode_df,
           miw_summary, miw_terms)

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dof_df.to_parquet(out_dir / "rbr_arms_dof.parquet")
        resid_df.to_parquet(out_dir / "rbr_arms_residual.parquet")
        vmode_df.to_parquet(out_dir / "rbr_arms_vmode.parquet")
        if miw_terms is not None:
            miw_terms.to_parquet(out_dir / "rbr_arms_miw_terms.parquet")
            pd.DataFrame([miw_summary]).to_parquet(
                out_dir / "rbr_arms_miw_summary.parquet")
        print(f"\n  wrote tables to {out_dir}")
        if args.plots:
            write_plots(out_dir / "rbr_arms_plots.pdf", args.label,
                        dof_df, vmode_df, miw_terms,
                        grid_a=args.grid_unconstrained,
                        grid_b=args.grid_rbr)


if __name__ == "__main__":
    main()
