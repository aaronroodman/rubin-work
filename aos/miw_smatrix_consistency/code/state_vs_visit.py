"""Per-visit DZ, v-modes and DoF from each MIW build, against ordinal visit number.

For each MIW build, the visits of the nine rotator bins are put in rotator-angle order
(time order within a bin) and numbered. One page per quantity, one panel per MIW:

- DZ(k, j) before the optical-state correction (``dz_``, the DZ fit to the donut Zernikes
  minus the MIW) and after (``dz_resid_``, minus the correctable part), µm;
- v-mode amplitude of the DZ before and after, from the build's own SVD (fitted S-matrix,
  n_keep). For a 50/34 build the "after" v-modes 1 to 34 are zero by construction;
- each of the 50 DoF of the solved state (``dof_``; range-bounded for RBR builds), with
  the DoF range dashed.

Thin vertical lines separate the FAM sets of one rotator bin; the label is the bin's
mean camera rotator angle, deg.

Writes ``state_vs_visit.pdf`` and ``rms_before_after.csv`` into the run directory.

Usage::

    python aos/miw_smatrix_consistency/code/state_vs_visit.py \\
        --miw sparse=d13s-A_50_34_i_fsm_sparse_no22@20261116b \\
        --miw sparse-rbr=d13s-A_50_50_i_rbr_fsm_sparse_no22@20261116
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from astropy.table import Table  # noqa: E402
from lsst.ts.intrinsic.wavefront import ofc_svd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.products import catalog  # noqa: E402

DZ_KJ = ([(k, j) for k in (1, 2, 3) for j in range(4, 12)]
         + [(k, j) for k in (4, 5, 6) for j in (5, 6, 7, 8)])
N_VMODES = 34


def dof_unit(name):
    """Hexapod tilts are deg; hexapod pistons, decenters and bending modes µm."""
    return "deg" if name.endswith(("_rx", "_ry")) else "µm"


def ofc_config_dir(variant, build):
    """The OFC config (fitted S-matrix) the build used, from its rot_-3_3 build log."""
    log = catalog.path("miw", variant, build) / "_work" / "logs" / "build_rot_-3_3.log"
    m = re.search(r"OFC config: (\S+)", log.read_text())
    return None if m is None or m.group(1) == "ts_config_mttcs" else m.group(1)


def load_case(variant, build):
    """Per-visit rows of all bins in plotting order, the SVD, and per-bin markers."""
    base = catalog.path("miw", variant, build)
    vis = pd.read_parquet(base / "fits.parquet", columns=["day_obs", "seq_num",
                                                          "rotator_angle"])
    parts = []
    for p in (base / "build").glob("rot_*/dz_fits.parquet"):
        d = pd.read_parquet(p).merge(vis, on=["day_obs", "seq_num"], how="left")
        parts.append(d.assign(rot_bin=p.parent.name,
                              rot_mean=d.rotator_angle.mean()))
    d = (pd.concat(parts, ignore_index=True)
         .sort_values(["rot_mean", "day_obs", "seq_num"], ignore_index=True))

    cfg = yaml.safe_load((base / "build" / "rot_-3_3" / "mi_config.yaml").read_text())
    b = cfg["build"]
    js = sorted({int(m.group(1)) for c in d.columns
                 if (m := re.fullmatch(r"dz_z(\d+)_c1", c))})
    svd = ofc_svd.build_ofc_svd(js, b.get("k_min", 1), b.get("k_max", 6), cfg["n_keep"],
                                n_dof=cfg["n_dof"],
                                ofc_normalization_yaml=b.get("ofc_normalization_yaml"),
                                ofc_config_dir=ofc_config_dir(variant, build))
    t = Table.from_pandas(d)
    v_before = ofc_svd.project_dz_table(t, "dz", svd)[0]
    v_after = ofc_svd.project_dz_table(t, "dz_resid", svd)[0]
    stored = d[[f"vmode_{i + 1}" for i in range(v_before.shape[1])]].to_numpy(float)
    dev = np.nanmax(np.abs(v_before - stored))
    print(f"{variant}@{build}: {len(d)} visits, n_keep {svd.n_keep_eff}, "
          f"max |recomputed - stored v-mode| = {dev:.2e}")
    edges = np.flatnonzero(np.r_[True, d.rot_bin.values[1:] != d.rot_bin.values[:-1]])
    marks = [(i, d.rot_mean.iloc[i]) for i in edges]
    return d, v_before, v_after, svd, marks


def mark_bins(ax, marks):
    for i, rot in marks:
        ax.axvline(i - 0.5, color="0.4", lw=0.5)
        ax.text(i, 1.0, f"rot {rot:+.0f}°", transform=ax.get_xaxis_transform(),
                fontsize=6, va="bottom", ha="left")


def page(pdf, cases, title, series, ylabel, hlines=None):
    """One panel per case; ``series(case)`` gives [(label, y)] or None if absent."""
    fig, axs = plt.subplots(len(cases), 1, figsize=(11, 2.6 * len(cases)),
                            layout="constrained", squeeze=False)
    for ax, (lab, c) in zip(axs[:, 0], cases.items()):
        s = series(c)
        ax.set_title(lab, fontsize=9, loc="left", pad=10)
        if s is None:
            ax.text(0.5, 0.5, "not in this build", ha="center", transform=ax.transAxes)
        else:
            x = np.arange(len(c[0]))
            for (name, y), col in zip(s, ("C0", "C3")):
                y = np.asarray(y, float)
                ax.plot(x, y, ".", ms=3, color=col,
                        label=f"{name}, RMS {np.sqrt(np.nanmean(y ** 2)):.3g}")
            if hlines is not None:
                lim = ax.get_ylim()      # scale to the data; range lines show if in view
                for h in hlines:
                    ax.axhline(h, color="k", ls="--", lw=0.6)
                ax.set_ylim(lim)
            ax.legend(fontsize=7, loc="upper right", markerscale=3)
        mark_bins(ax, c[4])
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
    axs[-1, 0].set_xlabel("ordinal visit number (rotator-bin order, then time)")
    fig.suptitle(title, fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--miw", action="append", required=True,
                    metavar="LABEL=VARIANT@BUILD")
    ap.add_argument("--label", default="state_vs_visit")
    args = ap.parse_args()
    run = study_run("aos/miw_smatrix_consistency", args.label)

    cases = {}
    for spec in args.miw:
        lab, _, vb = spec.partition("=")
        cases[lab] = load_case(*vb.partition("@")[::2])
    est = next(iter(cases.values()))[3].estimator
    dof_names, _ = est.dof_labels()
    ranges = np.asarray(est.dof_ranges(), float)

    def rms(y):
        return float(np.sqrt(np.nanmean(np.asarray(y, float) ** 2)))

    rows = []
    for lab, (d, *_rest) in cases.items():
        for k, j in DZ_KJ:
            if f"dz_z{j}_c{k}" in d:
                rows.append(dict(miw=lab, k=k, j=j, rms_before_um=rms(d[f"dz_z{j}_c{k}"]),
                                 rms_after_um=rms(d[f"dz_resid_z{j}_c{k}"])))
    tab = pd.DataFrame(rows)
    tab.to_csv(run.path / "rms_before_after.csv", index=False)

    pdf_path = run.path / "state_vs_visit.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axs = plt.subplots(len(cases), 1, figsize=(11, 2.6 * len(cases)),
                                layout="constrained", squeeze=False)
        for ax, lab in zip(axs[:, 0], cases):
            r = tab[tab.miw == lab]
            x = np.arange(len(r))
            ax.bar(x - 0.2, r.rms_before_um, 0.4, label="before correction")
            ax.bar(x + 0.2, r.rms_after_um, 0.4, color="C3", label="after correction")
            ax.set_xticks(x, [f"({k},{j})" for k, j in zip(r.k, r.j)], rotation=90,
                          fontsize=6)
            ax.set_yscale("log")
            ax.set_ylabel("RMS over visits [µm]", fontsize=8)
            ax.set_title(lab, fontsize=9, loc="left")
            ax.legend(fontsize=7)
        axs[-1, 0].set_xlabel("DZ (k, j)")
        fig.suptitle("DZ RMS over visits before and after the optical-state correction")
        pdf.savefig(fig)
        plt.close(fig)

        for k, j in DZ_KJ:
            page(pdf, cases, f"DZ(k={k}, j={j}) per visit",
                 lambda c: None if f"dz_z{j}_c{k}" not in c[0] else
                 [("before correction", c[0][f"dz_z{j}_c{k}"]),
                  ("after correction", c[0][f"dz_resid_z{j}_c{k}"])],
                 f"DZ({k},{j}) [µm]")
        for i in range(N_VMODES):
            page(pdf, cases, f"v-mode {i + 1} per visit (each build's own SVD)",
                 lambda c: [("before correction", c[1][:, i]),
                            ("after correction", c[2][:, i])],
                 f"v-mode {i + 1} amplitude\n[normalized DoF units]")
        for n, name in enumerate(dof_names):
            page(pdf, cases, f"DoF {n} {name}: solved optical state per visit "
                 f"(DoF range ± {ranges[n]:.4g} {dof_unit(name)}, dashed if in view)",
                 lambda c: [("solved state", c[0][f"dof_{name}"])],
                 f"{name} [{dof_unit(name)}]", hlines=(-ranges[n], ranges[n]))
    print(pdf_path)


if __name__ == "__main__":
    main()
