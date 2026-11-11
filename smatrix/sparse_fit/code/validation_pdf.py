"""Validation PDF: donut fits, Double Zernike (DZ) against DOF amplitude, and the
v-mode SVD of the sparse Smatrix for each fit set.

1. Gallery: six ray-traced donuts (different DOF and field points) at +2 steps,
   refit with each fit set; data, model, residual, and the DOF signal (perturbed minus
   nominal) in data and model.
2. DZ against DOF: per DOF the four largest DZ slopes. Points: field fit (field
   Zernikes 1..``KMAX``, R_outer 1.75 deg) of the side-averaged per-point change at
   k = -2..+2 steps, each point rescaled to the FAM-mean step, so the abscissa is
   k x mean step (DOF units). Lines: the Smatrix slope (solid) and a quadratic through
   the five points (dashed).
3. SVD: the Smatrix (31 field x 29 pupil x 50 DOF, µm per DOF unit) replaces
   ``ofc_data.sensitivity_matrix`` in the ts_ofc v13 `StateEstimator`
   (``aos_state.make_state_estimator``, all 50 DOF), so normalization and SVD are
   OFC's. Compared with the v13 in-focus Smatrix restricted to the same pupil terms.

Usage::

    python validation_pdf.py --label validation
"""
import argparse
import glob
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm

import donut_leakage as dl
from kmax_test import basis, predict
from refit_images import FWHM_TOL, OUT_ROOT, _one, _setup
from rubinwork import aos_state
from rubinwork.common import study_run

KMAX = 30
FIT_SETS = ("prod", "prod_z22", "dense21")
RB = ["M2 dz", "M2 dx", "M2 dy", "M2 rx", "M2 ry", "Cam dz", "Cam dx", "Cam dy",
      "Cam rx", "Cam ry"]
DOF_NAMES = RB + [f"M1M3 B{i}" for i in range(1, 21)] + [f"M2 B{i}" for i in range(1, 21)]
DOF_UNIT = ["µm"] * 3 + ["deg"] * 2 + ["µm"] * 3 + ["deg"] * 2 + ["µm"] * 40
# (configuration, DOF): centre, FAM ring 5 where M1M3 B15 fits are lost, corner
# centres both sides, field edge, a science point.
GALLERY = [("fam_r00_a00_intra", 0), ("fam_r05_a14_extra", 24), ("R04_SW0", 6),
           ("R40_SW1", 12), ("fam_r10_a05_extra", 33), ("sci_mp_intra", 45)]


# --------------------------------------------------------------------------- gallery
def _donut_file(name, idof):
    for f in sorted(glob.glob(f"{OUT_ROOT}/2026111[01]_bf_*dof*/donuts_{name}.fits")):
        with fits.open(f) as h:
            if idof in list(h["DOFS"].data["idof"]):
                return f
    raise FileNotFoundError(f"{name} DOF {idof}")


def gallery_pages(pdf):
    dl._init(list(range(50)), {})
    for name, idof in GALLERY:
        with fits.open(_donut_file(name, idof)) as h:
            ks = [int(k) for k in h[0].header["STEPS_K"].split()]
            a = list(h["DOFS"].data["idof"]).index(idof)
            step = h["DOFS"].data["step"][a]
            nom = h["NOMINAL"].data.astype(float)
            img = h["STEPS"].data[a, ks.index(2)].astype(float)
        th, fac, refs = _setup(name)
        rows = []
        for fs in FIT_SETS:
            idx = dl.FIT_SETS[fs]
            f0 = _one(fac, refs[fs], idx, th, nom)
            c = _one(fac, refs[fs], idx, th, img)
            w = _one(fac, refs[fs], idx, th, img, x0=f0["x"])
            c = w if w["cost"] < c["cost"] else c
            m = dl.danish.SingleDonutModel(fac, z_ref=refs[fs], z_terms=idx, thx=th[0],
                                           thy=th[1], npix=dl.NPIX)
            p0, p = m.unpack_params(f0["x"]), m.unpack_params(c["x"])
            mod0 = m.model(p0["flux"], p0["dx"], p0["dy"], p0["fwhm"], p0["z_fit"])
            mod = m.model(p["flux"], p["dx"], p["dy"], p["fwhm"], p["z_fit"])
            rows.append((fs, mod, mod0, c, f0))
        _gallery_page(pdf, name, idof, step, nom, img, rows, th)
        print(f"gallery {name} DOF {idof}", flush=True)


def _gallery_page(pdf, name, idof, step, nom, img, rows, th):
    fig, axs = plt.subplots(4, 4, figsize=(10, 10.5))
    sig = img - nom
    res_lim = max(np.abs(img - r[1]).max() for r in rows)
    sig_lim = np.abs(sig).max()
    peak = img.max()

    def show(ax, a, lim=None, title=""):
        kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim) if lim else dict(cmap="viridis")
        im = ax.imshow(a, origin="lower", **kw)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

    show(axs[0, 0], nom, title="data, nominal (flux / pixel)")
    show(axs[0, 1], img, title="data, +2 steps (flux / pixel)")
    show(axs[0, 2], sig, sig_lim, "DOF signal: data(+2) - data(0)")
    axs[0, 3].axis("off")
    axs[0, 3].text(0, 0.9, f"{name}\nfield ({np.rad2deg(th[0]):+.3f}, "
                   f"{np.rad2deg(th[1]):+.3f}) deg OCS\nDOF {idof} {DOF_NAMES[idof]}\n"
                   f"+2 steps = {2 * step:.4g} {DOF_UNIT[idof]}\npeak {peak:.0f} flux / "
                   f"pixel\n\nresidual colour scale +-{res_lim:.1f}\nsignal colour "
                   f"scale +-{sig_lim:.1f}\n(flux / pixel)", va="top", fontsize=7)
    for i, (fs, mod, mod0, c, f0) in enumerate(rows, start=1):
        lost = abs(c["fwhm"] - f0["fwhm"]) > FWHM_TOL
        rms = np.sqrt(np.mean((img - mod) ** 2)) / peak
        show(axs[i, 0], mod, title=f"{fs} model, +2 steps")
        show(axs[i, 1], img - mod, res_lim,
             f"data - model; RMS / peak {rms:.4f}\nfwhm {c['fwhm']:.3f} arcsec "
             f"(nominal {f0['fwhm']:.3f}){' LOST' if lost else ''}")
        show(axs[i, 2], mod - mod0, sig_lim, "model signal: model(+2) - model(0)")
        show(axs[i, 3], sig - (mod - mod0), sig_lim,
             f"signal residual; cost / nominal cost {c['cost'] / f0['cost']:.2f}")
    fig.suptitle("Ray-traced donut and Danish fits (noise-free; RMS / peak is "
                 "dimensionless, residual RMS over peak data pixel)", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------- Smatrix (DZ)
def smatrix(resp, fs):
    """Side-averaged FAM field fit of the slopes: (KMAX+1, 29, 50) µm per DOF unit."""
    kind, side = resp["kind"], resp["side"]
    fi, fe = (kind == "fam") & (side == "intra"), (kind == "fam") & (side == "extra")
    s = resp[f"{fs}__slope"]
    avg = ((s[fi] + s[fe]) / 2).reshape(fi.sum(), -1)
    c, *_ = np.linalg.lstsq(basis(KMAX, resp["thx"][fi], resp["thy"][fi]), avg, rcond=None)
    out = np.zeros((KMAX + 1, 29, 50))
    out[1:, resp[f"{fs}__noll"]] = c.reshape(KMAX, 50, -1).transpose(0, 2, 1)
    return out


def dz_points(scan, fs):
    """DZ (5, 50, KMAX, n_z) µm and errors at h = k x mean step; mean step (50,)."""
    side = scan["side"]
    fi, fe = side == "intra", side == "extra"
    assert np.allclose(scan["thx"][fi], scan["thx"][fe])
    step = scan["step"]
    mstep = np.mean(step, 0)
    dz = scan[f"{fs}__dz"] * (mstep / step)[:, :, None, None]
    avg = (dz[fi] + dz[fe]) / 2                                   # (n, 50, 5, nz)
    B = basis(KMAX, scan["thx"][fi], scan["thy"][fi])
    nk, nz = avg.shape[2], avg.shape[3]
    val = np.full((nk, 50, KMAX, nz), np.nan)
    err = np.full_like(val, np.nan)
    for i in range(50):
        for k in range(nk):
            y = avg[:, i, k]
            ok = np.all(np.isfinite(y), 1)
            c, res, *_ = np.linalg.lstsq(B[ok], y[ok], rcond=None)
            val[k, i] = c
            sig2 = np.sum((B[ok] @ c - y[ok]) ** 2, 0) / max(ok.sum() - KMAX, 1)
            err[k, i] = np.sqrt(np.outer(np.diag(np.linalg.inv(B[ok].T @ B[ok])), sig2))
    return val, err, mstep


def quad_ratio(y, h):
    """Quadratic through the points; |quadratic| / |linear| change at h[-1]."""
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.zeros(3), np.nan
    q = np.polyfit(h[ok], y[ok], 2)
    lin = abs(q[1]) * h[-1]
    return q, abs(q[0]) * h[-1] ** 2 / lin if lin else np.nan


def top4(S, noll, i):
    sl = S[1:, noll, i]
    return sl, [np.unravel_index(t, sl.shape)
                for t in np.argsort(np.abs(sl).ravel())[::-1][:4]]


def dz_ratios(S, val, mstep, k, noll):
    h = k[:, None] * mstep[None, :]
    sl_t = [top4(S, noll, i) for i in range(50)]
    return np.array([quad_ratio(val[:, i, kf, jz], h[:, i])[1]
                     for i in range(50) for kf, jz in sl_t[i][1]])


def dz_pages(pdf, S, val, err, mstep, k, noll, fs):
    h = k[:, None] * mstep[None, :]
    for p0 in range(0, 50, 5):
        fig, axs = plt.subplots(5, 4, figsize=(10, 11))
        for r, i in enumerate(range(p0, p0 + 5)):
            sl, top = top4(S, noll, i)
            for c, (kf, jz) in enumerate(top):
                ax = axs[r, c]
                y, e = val[:, i, kf, jz], err[:, i, kf, jz]
                ax.errorbar(h[:, i], y, e, fmt="o", ms=3, c="C0")
                hh = np.linspace(h[0, i], h[-1, i], 50)
                ax.plot(hh, sl[kf, jz] * hh, "k-", lw=0.8)
                q, ratio = quad_ratio(y, h[:, i])
                ax.plot(hh, np.polyval(q, hh), "C1--", lw=0.8)
                ax.set_title(f"{DOF_NAMES[i]}: DZ(field {kf + 1}, Z{noll[jz]})\n"
                             f"slope {sl[kf, jz]:.3g} µm / {DOF_UNIT[i]}; quad / lin "
                             f"{ratio:.3f}", fontsize=6.5)
                ax.tick_params(labelsize=6)
                if r == 4 or i == 49:
                    ax.set_xlabel(f"DOF amplitude ({DOF_UNIT[i]})", fontsize=6)
                if c == 0:
                    ax.set_ylabel("DZ change (µm)", fontsize=6)
        fig.suptitle(f"{fs}: DZ change against DOF amplitude, four largest DZ slopes per "
                     f"DOF. Points: FAM field fit (kmax {KMAX}) at k = -2..+2 steps; "
                     "solid: Smatrix slope; dashed: quadratic.\nquad / lin: quadratic "
                     "over linear change at +2 steps (dimensionless)", fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)


# ------------------------------------------------------------------------------- SVD
def svd(S_full, n_modes=50):
    se = aos_state.make_state_estimator(dof_set="all_50")
    se.ofc_data.sensitivity_matrix = S_full
    se.refresh_from_ofc_data()
    se.truncate_index = n_modes
    return se


def _rows(noll):
    """Flattened (31 x 29) row indices for field 1..KMAX, pupil noll; j-major order."""
    return np.array([kf * 29 + j for j in noll for kf in range(1, KMAX + 1)])


def svd_all(mats, nolls):
    ref = aos_state.make_state_estimator(dof_set="all_50").ofc_data.sensitivity_matrix
    ses, refs = {}, {}
    for fs in FIT_SETS:
        ses[fs] = svd(mats[fs])
        r = np.zeros_like(ref)
        r[1:KMAX + 1, nolls[fs]] = ref[1:KMAX + 1, nolls[fs]]
        refs[fs] = svd(r)
    se_full = svd(ref.copy())
    return ses, refs, se_full


def svd_pages(pdf, ses, refs, se_full, nolls):
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, fs in enumerate(FIT_SETS):
        ax.semilogy(np.arange(1, 51), ses[fs].S, "o-", ms=3, c=f"C{c}",
                    label=f"{fs} (Danish-fit responses)")
        ax.semilogy(np.arange(1, 51), refs[fs].S, "--", c=f"C{c}",
                    label=f"ts_ofc v13 in-focus, {fs} pupil terms")
    ax.semilogy(np.arange(1, 51), se_full.S, "k:", label="ts_ofc v13 in-focus, Z4-Z28")
    ax.set(xlabel="v-mode index", ylabel="singular value (normalized Smatrix; µm of DZ "
           "per unit normalized DOF)", title="Singular values of S N, N = v13 "
           "range0.5_fwhm-0.15 normalization, field k 1-30")
    ax.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    for fs in FIT_SETS:
        se, rf = ses[fs], refs[fs]
        rows = _rows(nolls[fs])
        A = (se.ofc_data.sensitivity_matrix.reshape(-1, 50) @ se.normalization_matrix)[rows]
        fig, axs = plt.subplots(2, 2, figsize=(11, 10))

        def img(ax, a, title, xl, yl, sym=True, log=None):
            lim = np.abs(a).max()
            if log:
                kw = dict(norm=SymLogNorm(log * lim, vmin=-lim, vmax=lim), cmap="RdBu_r")
                title += f"\nsymmetric log scale, linear within +-{log * lim:.2g}"
            else:
                kw = dict(cmap="RdBu_r" if sym else "viridis",
                          vmin=-lim if sym else 0, vmax=lim)
            im = ax.imshow(a, aspect="auto", origin="lower", interpolation="nearest", **kw)
            ax.set(title=title, xlabel=xl, ylabel=yl)
            ax.title.set_fontsize(8)
            fig.colorbar(im, ax=ax, fraction=0.046)
            return ax

        def dofaxis(ax, which):
            for x in (9.5, 29.5):
                (ax.axhline if which == "y" else ax.axvline)(x, c="k", lw=0.5)

        def dzaxis(ax):
            for b in range(1, len(nolls[fs])):
                ax.axhline(b * KMAX - 0.5, c="k", lw=0.3)
            ax.set_yticks(np.arange(len(nolls[fs])) * KMAX + KMAX / 2)
            ax.set_yticklabels([f"Z{j}" for j in nolls[fs]], fontsize=6)

        ax = img(axs[0, 0], se.Vh.T, "Vh: DOF vs v-mode (dimensionless, unit rows)",
                 "v-mode index", "DOF index (0-9 RB, 10-29 M1M3, 30-49 M2)")
        dofaxis(ax, "y")
        ax = img(axs[0, 1], se.U[rows], "U: DZ vs v-mode (dimensionless, unit columns)",
                 "v-mode index", "DZ (pupil Zernike; field k 1-30 within)",
                 log=1e-2)
        dzaxis(ax)
        ax = img(axs[1, 0], A, "S N: DZ vs DOF (µm of DZ per unit normalized DOF)",
                 "DOF index", "DZ (pupil Zernike; field k 1-30 within)", log=1e-3)
        dzaxis(ax)
        dofaxis(ax, "x")
        ov = np.abs(se.Vh @ rf.Vh.T)
        ax = img(axs[1, 1], ov, "|Vh . Vh_ref^T|: v-mode overlap with ts_ofc v13 "
                 "in-focus,\nsame pupil terms (dimensionless)", "ts_ofc v-mode index",
                 f"{fs} v-mode index", sym=False)
        fig.suptitle(f"{fs}: SVD of the normalized sparse Smatrix (ts_ofc StateEstimator)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", default="20261111_responses")
    p.add_argument("--scan", default="20261111_dz_scan")
    p.add_argument("--no-gallery", action="store_true")
    p.add_argument("--label", default="validation")
    args = p.parse_args()
    resp = np.load(f"{OUT_ROOT}/{args.responses}/responses.npz")
    scan = np.load(f"{OUT_ROOT}/{args.scan}/dz_scan.npz")
    mats, nolls, pts, out, summary = {}, {}, {}, {}, []
    for fs in FIT_SETS:
        nolls[fs] = resp[f"{fs}__noll"]
        mats[fs] = smatrix(resp, fs)
        pts[fs] = dz_points(scan, fs)
        q = dz_ratios(mats[fs], pts[fs][0], pts[fs][2], scan["k"], nolls[fs])
        out[f"{fs}__smatrix"], out[f"{fs}__dz"], out[f"{fs}__dz_err"] = \
            mats[fs], pts[fs][0], pts[fs][1]
        summary.append(f"{fs}: quad / lin at +2 steps over the 200 plotted DZ: median "
                       f"{np.nanmedian(q):.4f}, max {np.nanmax(q):.3f} (dimensionless)")
    ses, refs, se_full = svd_all(mats, nolls)
    for fs in FIT_SETS:
        ovm = np.abs(ses[fs].Vh @ refs[fs].Vh.T)
        ov = np.array([np.linalg.norm(ovm[i, max(i - 1, 0):i + 2]) for i in range(50)])
        out[f"{fs}__S"], out[f"{fs}__Vh"], out[f"{fs}__U"] = ses[fs].S, ses[fs].Vh, ses[fs].U
        out[f"{fs}__S_v13"] = refs[fs].S
        summary.append(f"{fs}: singular value 1 {ses[fs].S[0]:.2f} (v13 same pupil terms "
                       f"{refs[fs].S[0]:.2f}; µm per unit normalized DOF); s50 / s1 "
                       f"{ses[fs].S[-1] / ses[fs].S[0]:.2e} (dimensionless); v-mode overlap "
                       "with v13 modes i-1..i+1 (norm, dimensionless), modes 1-12: " + " ".join(f"{x:.2f}" for x in ov[:12]))
    out["v13_S"] = se_full.S
    print("\n".join(summary), flush=True)
    run = study_run("smatrix/sparse_fit", args.label)
    np.savez(f"{run.path}/validation.npz", **out)
    with PdfPages(f"{run.path}/sparse_fit_validation.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.04, 0.97, VALIDATION_TEXT + "\n\nResults\n" + "\n".join(
            textwrap.fill(s, 92, initial_indent="- ", subsequent_indent="  ")
            for s in summary), va="top", family="monospace", fontsize=6.8)
        pdf.savefig(fig)
        plt.close(fig)
        if not args.no_gallery:
            gallery_pages(pdf)
        for fs in FIT_SETS:
            val, err, mstep = pts[fs]
            dz_pages(pdf, mats[fs], val, err, mstep, scan["k"], nolls[fs], fs)
        svd_pages(pdf, ses, refs, se_full, nolls)
    print(f"wrote {run.path}")


VALIDATION_TEXT = """sparse_fit validation: from ray-traced donuts to the sparse Smatrix and its v-modes

Process
- Batoid ray-traces noise-free donuts (1.5 mm defocus, Kolmogorov FWHM 1.0 arcsec,
  binned x2, 81 x 81 pixels) at 940 field configurations, for the nominal telescope
  and each of 50 DOF at k = -2, -1, +1, +2 steps (step: RMS in-focus Zernike change
  0.01 µm per field point).
- Danish fits each donut with three Zernike sets: prod (Noll 4-11, 14, 15, 20, 21,
  27, 28), prod_z22 (prod + 22), dense21 (Noll 4-26 but 20, 21); other terms fixed
  to the batoid intrinsic. Lost fits (fwhm more than 0.2 arcsec off nominal) dropped.
- Per point, DOF and fit set the slope is a quadratic through the five amplitudes.
- The Smatrix: side-averaged FAM slopes fit with field Zernikes 1-30 (R_outer
  1.75 deg), giving DZ(field k, pupil j) per DOF in µm per DOF unit (µm, deg).
- SVD: this Smatrix inside the ts_ofc v13 StateEstimator (all 50 DOF,
  range0.5_fwhm-0.15 normalization).

Pages
1. this summary
2-7. donut gallery: data, model, residual and DOF signal per fit set
then per fit set 10 pages: DZ against DOF amplitude, four largest DZ per DOF
then singular values, and per fit set Vh, U, S N and the overlap with ts_ofc v-modes

Notes
- v-mode signs are as the SVD returns them; compare |overlap|, not signs.
- Near-degenerate singular-value pairs mix within the pair, so pairs show as 2 x 2
  blocks in the overlap.
- The ts_ofc reference is the in-focus OPD Smatrix: it has no Danish leakage of
  unfitted terms, which is large for high bending modes."""


if __name__ == "__main__":
    main()
