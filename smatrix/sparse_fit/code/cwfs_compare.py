"""CWFS responses per side and per corner against a single side-averaged Smatrix.

Fitted responses (µm per DOF unit) at the CWFS grid points (detector shift, extra-focal
on SW0, intra-focal on SW1) are compared with:

- ``own``: the FAM field model of the same side (kmax 30);
- ``single``: the side-averaged FAM field model, (intra + extra) / 2, as one Smatrix;
- ``dense``: the in-focus Zernike rows ``A_dense`` at the point (no leakage).

``corner``: per corner, the mean of the SW0 and SW1 half-chip means of the fitted
responses against the mean of ``single`` over the same points. ``fam_split``: |intra -
extra| / 2 at the FAM points, the side-dependent part. Errors are |difference| / RMS
over FAM of |response|, norms over the fitted terms (dimensionless); only points with
field radius <= ``R_CUT`` (deg), the ts_wep CWFS donut cut. Output ``cwfs_compare.npz``
and ``cwfs_compare.pdf``.

Usage::

    python cwfs_compare.py --label cwfs_compare
"""
import argparse
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from kmax_test import GROUPS, predict
from refit_images import OUT_ROOT
from rubinwork.common import study_run

R_CUT = 1.725   # deg
KMAX = 30
FIT_SETS = ("prod", "prod_z22", "dense21")


def load_dense(names):
    """A_dense (n, 50, 75) for the named configurations from the brute-force JSONs."""
    idx = {n: i for i, n in enumerate(names)}
    out = np.full((len(names), 50, 75), np.nan)
    for f in glob.glob(f"{OUT_ROOT}/2026111[01]_bf_*dof*/leak_*.json"):
        name = f.split("leak_")[1][:-5]
        if name not in idx:
            continue
        d = json.load(open(f))
        out[idx[name], d["dofs"]] = np.array(d["A_dense"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", default="20261111_responses")
    p.add_argument("--label", default="cwfs_compare")
    args = p.parse_args()
    d = np.load(f"{OUT_ROOT}/{args.responses}/responses.npz")
    name, kind, side = d["name"].astype(str), d["kind"], d["side"]
    thx, thy = d["thx"], d["thy"]
    r = np.hypot(thx, thy)
    cw = (kind == "cwfs") & (r <= R_CUT)
    corner = np.array([n[:3] for n in name])
    dense = load_dense(list(name[cw]))
    fam_i, fam_e = (kind == "fam") & (side == "intra"), (kind == "fam") & (side == "extra")
    assert np.allclose(thx[fam_i], thx[fam_e]) and np.allclose(thy[fam_i], thy[fam_e])
    out, lines = dict(name=name[cw], r=r[cw]), []
    hdr = "fit set   test      " + "".join(f"{g:>14s}" for g in GROUPS)
    lines += ["median relative error |difference| / RMS_FAM |response| (dimensionless), "
              f"CWFS points r <= {R_CUT} deg (n = {cw.sum()}), FAM model kmax {KMAX}", hdr]
    for fs in FIT_SETS:
        s = d[f"{fs}__slope"]
        n, ndof, nz = s.shape
        flat = s.reshape(n, -1)
        norm = np.sqrt(np.mean(np.sum(s[kind == "fam"] ** 2, 2), 0))
        avg = (flat[fam_i] + flat[fam_e]) / 2
        single = predict(KMAX, thx[fam_i], thy[fam_i], avg, thx[cw], thy[cw]).reshape(-1, ndof, nz)
        own = np.full_like(single, np.nan)
        for sd, fm in (("intra", fam_i), ("extra", fam_e)):
            m = side[cw] == sd
            own[m] = predict(KMAX, thx[fm], thy[fm], flat[fm], thx[cw][m],
                             thy[cw][m]).reshape(-1, ndof, nz)
        a = s[cw]
        jdx = d[f"{fs}__noll"] - 4
        res = dict(own=np.linalg.norm(a - own, axis=2) / norm,
                   single=np.linalg.norm(a - single, axis=2) / norm,
                   dense=np.linalg.norm(a - dense[:, :, jdx], axis=2) / norm,
                   fam_split=np.linalg.norm(s[fam_i] - s[fam_e], axis=2) / 2 / norm)
        cor = []
        for c in np.unique(corner[cw]):
            mc = corner[cw] == c
            half = [mc & (side[cw] == sd) for sd in ("intra", "extra")]
            fit_c = np.mean([a[h].mean(0) for h in half], 0)
            sgl_c = np.mean([single[h].mean(0) for h in half], 0)
            cor.append(np.linalg.norm(fit_c - sgl_c, axis=1) / norm)
        res["corner"] = np.array(cor)
        for t, e in res.items():
            out[f"{fs}__{t}"] = e
            lines.append(f"{fs:9s} {t:9s}" + "".join(
                f"{np.median(e[:, list(g)]):14.4f}" for g in GROUPS.values()))
    print("\n".join(lines))
    run = study_run("smatrix/sparse_fit", args.label)
    np.savez(f"{run.path}/cwfs_compare.npz", **out)
    with PdfPages(f"{run.path}/cwfs_compare.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=6.5)
        pdf.savefig(fig)
        plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
        for ax, fs in zip(axs, FIT_SETS):
            for t in ("own", "single", "dense", "corner"):
                ax.plot(np.arange(50), np.median(out[f"{fs}__{t}"], 0), ".-", label=t)
            ax.set(title=fs, xlabel="DOF index", yscale="log")
        axs[0].set_ylabel("median relative error (dimensionless)")
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
