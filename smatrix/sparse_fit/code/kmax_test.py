"""Field model of the brute-force responses: fit FAM points with field Zernikes to kmax,
predict held-out points.

Per fit set and donut side, each response dz_j/dDOF_i (µm per DOF unit) is fit over
the FAM points (camera shift) with galsim field Zernikes 1..kmax on R_outer 1.75 deg.
Test sets:

- ``fam_cv``: FAM points, 5 folds by spoke (azimuth index mod 5), same geometry;
- ``sci``: the 9 science check points (camera shift);
- ``cwfs``: the 8 half-chip centres and the 144 CWFS grid points (detector shift, field
  radius to 1.82 deg, so partly extrapolated).

Errors per (point, DOF), vector norms over the fitted terms: ``err_nm`` =
|predicted - true| x DOF range (nm of Zernike at the full DOF range); ``rel`` =
|predicted - true| / RMS over FAM of |true| (dimensionless). ``fam_in`` is the in-sample
FAM residual scaled by sqrt(n / (n - kmax)): if it matches ``fam_cv``, the residual is
white (slope noise), not missing field structure. Output ``kmax_test.npz`` and
``kmax_test.pdf`` in a run directory.

Usage::

    python kmax_test.py --responses 20261111_responses --label kmax_test
"""
import argparse
import os

import galsim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from refit_images import OUT_ROOT
from rubinwork.common import study_run
from rubinwork.smatrix import normalization_weights as NW

KMAXES = (30, 45, 66, 91)
FIELD_RADIUS_DEG = 1.75
FIT_SETS = ("prod", "prod_z22", "dense21")
TESTS = ("fam_in", "fam_cv", "sci", "cwfs")
GROUPS = {"rigid body": range(0, 10), "M1M3 B1-B10": range(10, 20),
          "M1M3 B11-B20": range(20, 30), "M2 B1-B10": range(30, 40),
          "M2 B11-B20": range(40, 50)}
_MTTCS = os.environ.get("TS_CONFIG_MTTCS_DIR",
                        "/sdf/group/rubin/u/roodman/LSST/packages/ts_config_mttcs")
V13 = os.path.join(_MTTCS, "MTAOS/v13/ofc")


def dof_range():
    """ts_ofc v13 range per DOF (µm or deg): hexapod stroke; bending from IM forces."""
    f1 = np.abs(np.array(yaml.safe_load(open(f"{V13}/M1M3/M1M3_1um_156_force.yaml")))[:, 3:])
    f2 = np.abs(np.array(yaml.safe_load(open(f"{V13}/M2/M2_1um_72_force.yaml")))[:, 3:])
    return NW.compute_range_weights(f1, f2, list(range(20)), list(range(20)))


def basis(kmax, thx, thy):
    return galsim.zernike.zernikeBasis(kmax, thx, thy, R_outer=FIELD_RADIUS_DEG)[1:].T


def predict(kmax, xf, yf, sf, xt, yt):
    """Least-squares field fit of sf (n_fam, m) and evaluation at (xt, yt)."""
    c, *_ = np.linalg.lstsq(basis(kmax, xf, yf), sf, rcond=None)
    return basis(kmax, xt, yt) @ c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", default="20261111_responses")
    p.add_argument("--label", default="kmax_test")
    args = p.parse_args()
    d = np.load(f"{OUT_ROOT}/{args.responses}/responses.npz")
    rng = dof_range()
    name, kind, side = d["name"], d["kind"], d["side"]
    thx, thy = d["thx"], d["thy"]
    r = np.hypot(thx, thy)
    spoke = np.array([int(n.split("_a")[1][:2]) if k == "fam" else -1
                      for n, k in zip(name, kind)])
    out, summ = {}, []
    for fs in FIT_SETS:
        s = d[f"{fs}__slope"]                                    # (n, 50, nz)
        n, ndof, nz = s.shape
        for kmax in KMAXES:
            err = np.full((n, ndof), np.nan)
            ein = np.full((n, ndof), np.nan)
            tag = np.full(n, "", dtype=object)
            for sd in ("intra", "extra"):
                fam = (kind == "fam") & (side == sd)
                flat = s.reshape(n, -1)
                for fold in range(5):
                    tr, te = fam & (spoke % 5 != fold), fam & (spoke % 5 == fold)
                    pr = predict(kmax, thx[tr], thy[tr], flat[tr], thx[te], thy[te])
                    err[te] = np.linalg.norm((pr - flat[te]).reshape(-1, ndof, nz), axis=2)
                    tag[te] = "fam_cv"
                pr = predict(kmax, thx[fam], thy[fam], flat[fam], thx[fam], thy[fam])
                ein[fam] = np.linalg.norm((pr - flat[fam]).reshape(-1, ndof, nz), axis=2) \
                    * np.sqrt(fam.sum() / (fam.sum() - kmax))
                te = (kind != "fam") & (side == sd)
                pr = predict(kmax, thx[fam], thy[fam], flat[fam], thx[te], thy[te])
                err[te] = np.linalg.norm((pr - flat[te]).reshape(-1, ndof, nz), axis=2)
                tag[te & (kind == "cwfs")] = "cwfs"
                tag[te & (kind == "check") & np.char.startswith(name.astype(str), "R")] = "cwfs"
                tag[te & np.char.startswith(name.astype(str), "sci")] = "sci"
            norm = np.sqrt(np.mean(np.sum(s[kind == "fam"] ** 2, 2), 0))
            out[f"{fs}__k{kmax}__err_nm"] = err * rng[None, :] * 1e3
            out[f"{fs}__k{kmax}__rel"] = rel = err / norm
            out[f"{fs}__k{kmax}__rel_in"] = rel_in = ein / norm
            for t in TESTS:
                e = rel_in[kind == "fam"] if t == "fam_in" else rel[tag == t]
                summ.append((fs, kmax, t, [np.median(e[:, list(g)]) for g in GROUPS.values()]))
    out.update(name=name, tag=tag.astype(str), r=r, range=rng)
    run = study_run("smatrix/sparse_fit", args.label)
    np.savez(f"{run.path}/kmax_test.npz", **out)
    lines = ["median relative error |pred - true| / RMS_FAM |true| (dimensionless)",
             "fit set   kmax  test    " + "".join(f"{g:>14s}" for g in GROUPS)]
    for fs, k, t, meds in summ:
        lines.append(f"{fs:9s} {k:4d}  {t:7s}" + "".join(f"{m:14.4f}" for m in meds))
    print("\n".join(lines))
    with PdfPages(f"{run.path}/kmax_test.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.97, "Field-model error of the brute-force responses (FAM fit, "
                 "field Zernikes to kmax, R_outer 1.75 deg)", va="top", fontsize=10)
        fig.text(0.05, 0.92, "\n".join(lines), va="top", family="monospace", fontsize=6.5)
        pdf.savefig(fig)
        plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
        for ax, fs in zip(axs, FIT_SETS):
            for t, c in zip(TESTS, ("C3", "C0", "C1", "C2")):
                rows = [x for x in summ if x[0] == fs and x[2] == t]
                ax.plot([x[1] for x in rows], [np.median(x[3]) for x in rows], "o-", c=c,
                        label=t)
            ax.set(title=fs, xlabel="kmax (field Zernike terms)", yscale="log")
        axs[0].set_ylabel("median over DOF groups of median relative error")
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
        for ax, fs in zip(axs, FIT_SETS):
            e = np.nanmedian(out[f"{fs}__k66__rel"], axis=1)
            for t, c in zip(TESTS[1:], ("C0", "C1", "C2")):
                m = tag == t
                ax.scatter(r[m], e[m], s=4, c=c, label=t)
            ax.set(title=f"{fs}, kmax 66", xlabel="field radius (deg, OCS)", yscale="log")
        axs[0].set_ylabel("median over DOF of relative error")
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
