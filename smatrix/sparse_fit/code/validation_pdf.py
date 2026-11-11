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

import donut_leakage as dl
from kmax_test import basis, dof_range
from refit_images import FWHM_TOL, OUT_ROOT, _one, _setup
from rubinwork import aos_state
from rubinwork.common import study_run

KMAX = 30
KSHOW = 6                        # field terms shown in the SVD matrix pages
FIT_SETS = ("prod", "prod_z22", "dense21")
RB = ["M2 dz", "M2 dx", "M2 dy", "M2 rx", "M2 ry", "Cam dz", "Cam dx", "Cam dy",
      "Cam rx", "Cam ry"]
DOF_NAMES = RB + [f"M1M3 B{i}" for i in range(1, 21)] + [f"M2 B{i}" for i in range(1, 21)]
DOF_UNIT = ["µm"] * 3 + ["deg"] * 2 + ["µm"] * 3 + ["deg"] * 2 + ["µm"] * 40
# (configuration, DOF): centre, FAM ring 5 where M1M3 B15 fits are lost, corner
# centres both sides, field edge, a science point.
GALLERY_RING = "fam_r04"          # FAM ring at field radius 1.06 deg


# --------------------------------------------------------------------------- gallery
def _ofc_image(h, idof, k):
    """Donut at k steps in the ts_ofc sign convention, and the step (DOF units)."""
    ks = [int(x) for x in h[0].header["STEPS_K"].split()]
    a = list(h["DOFS"].data["idof"]).index(idof)
    sgn = dl.stored_sign(h[0].header)[idof]
    return h["STEPS"].data[a, ks.index(int(k * sgn))].astype(float), h["DOFS"].data["step"][a]


def gallery_points():
    """Per DOF, the GALLERY_RING donut file with the largest sum of squares of
    image(+2) - image(-2)."""
    best = {}
    for f in sorted(glob.glob(f"{OUT_ROOT}/2026111[01]_bf_*dof*/donuts_{GALLERY_RING}_*.fits")):
        with fits.open(f) as h:
            for idof in h["DOFS"].data["idof"]:
                d = _ofc_image(h, idof, 2)[0] - _ofc_image(h, idof, -2)[0]
                ss = float(np.sum(d ** 2))
                if ss > best.get(idof, (0.0, None))[0]:
                    best[idof] = (ss, f)
    return {i: best[i][1] for i in range(50)}


def _fit_pair(fac, ref, idx, th, nom, imgs):
    """Fits of the nominal and of each image (cold, and warm from the nominal if better)."""
    f0 = _one(fac, ref, idx, th, nom)
    fits_ = []
    for img in imgs:
        c = _one(fac, ref, idx, th, img)
        if abs(c["fwhm"] - f0["fwhm"]) > FWHM_TOL:
            w = _one(fac, ref, idx, th, img, x0=f0["x"])
            c = w if w["cost"] < c["cost"] else c
        fits_.append(c)
    return f0, fits_


def gallery_fits(path):
    """Fit the per-DOF gallery donuts with every fit set; save models to ``path``."""
    dl._init(list(range(50)), {})
    files = gallery_points()
    out = dict(name=[], thx=[], thy=[], step=[], img=[], model=[], fwhm=[], fwhm0=[],
               cost=[], cost0=[])
    for idof in range(50):
        with fits.open(files[idof]) as h:
            name = h[0].header["CONFIG"]
            nom = h["NOMINAL"].data.astype(float)
            ip, step = _ofc_image(h, idof, 2)
            im = _ofc_image(h, idof, -2)[0]
        th, fac, refs = _setup(name)
        mods, fw, fw0, co, co0 = [], [], [], [], []
        for fs in FIT_SETS:
            idx = dl.FIT_SETS[fs]
            f0, (cp, cm) = _fit_pair(fac, refs[fs], idx, th, nom, (ip, im))
            m = dl.danish.SingleDonutModel(fac, z_ref=refs[fs], z_terms=idx, thx=th[0],
                                           thy=th[1], npix=dl.NPIX)
            row = []
            for c in (cp, cm):
                p = m.unpack_params(c["x"])
                row.append(m.model(p["flux"], p["dx"], p["dy"], p["fwhm"], p["z_fit"]))
            mods.append(row)
            fw.append([cp["fwhm"], cm["fwhm"]]), fw0.append(f0["fwhm"])
            co.append([cp["cost"], cm["cost"]]), co0.append(f0["cost"])
        for key, v in (("name", name), ("thx", np.rad2deg(th[0])), ("thy", np.rad2deg(th[1])),
                       ("step", step), ("img", [ip, im]), ("model", mods), ("fwhm", fw),
                       ("fwhm0", fw0), ("cost", co), ("cost0", co0)):
            out[key].append(v)
        print(f"gallery DOF {idof} {name}", flush=True)
    out = {k: np.array(v) for k, v in out.items()}
    np.savez(path, **out)
    return out


def gallery_pages(pdf, g):
    for idof in range(50):
        _gallery_page(pdf, idof, {k: v[idof] for k, v in g.items()})


def _gallery_page(pdf, idof, g):
    fig, axs = plt.subplots(4, 4, figsize=(10, 10.5))
    ip, im = g["img"]
    diff = ip - im
    res_lim = np.abs(g["img"][None, None] - g["model"]).max()
    peak = ip.max()

    def show(ax, a, lim=None, title=""):
        kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim) if lim else dict(cmap="viridis")
        img = ax.imshow(a, origin="lower", **kw)
        ax.set_title(title, fontsize=6.5)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

    show(axs[0, 0], ip, title="data, +2 steps (flux / pixel)")
    show(axs[0, 1], im, title="data, -2 steps (flux / pixel)")
    show(axs[0, 2], diff, np.abs(diff).max(), "data(+2) - data(-2) (flux / pixel)")
    axs[0, 3].axis("off")
    axs[0, 3].text(0, 0.95, f"peak {peak:.0f} flux / pixel\nresidual colour scale "
                   f"+-{res_lim:.2f} flux / pixel\n(all fit sets, both signs)\n\nRMS / "
                   "peak: residual RMS over\nthe peak data pixel\n(dimensionless)",
                   va="top", fontsize=7)
    for r, fs in enumerate(FIT_SETS, start=1):
        for c, (lab, j) in enumerate((("+2", 0), ("-2", 1))):
            mod = g["model"][r - 1, j]
            res = g["img"][j] - mod
            lost = abs(g["fwhm"][r - 1, j] - g["fwhm0"][r - 1]) > FWHM_TOL
            show(axs[r, 2 * c], mod, title=f"{fs} model, {lab} steps")
            show(axs[r, 2 * c + 1], res, res_lim,
                 f"data - model, {lab}; RMS / peak {np.sqrt(np.mean(res ** 2)) / peak:.4f}"
                 f"\nfwhm {g['fwhm'][r - 1, j]:.3f} arcsec (nominal "
                 f"{g['fwhm0'][r - 1]:.3f}){' LOST' if lost else ''}\ncost / nominal "
                 f"cost {g['cost'][r - 1, j] / g['cost0'][r - 1]:.2f}")
    r = np.hypot(g["thx"], g["thy"])
    az = np.rad2deg(np.arctan2(g["thy"], g["thx"]))
    fig.suptitle(f"DOF {idof} {DOF_NAMES[idof]}: +-2 steps = +-{2 * g['step']:.4g} "
                 f"{DOF_UNIT[idof]} (ts_ofc sign)\n{g['name']}: field ({g['thx']:+.3f}, "
                 f"{g['thy']:+.3f}) deg OCS, radius {r:.3f} deg, azimuth {az:+.0f} deg; the "
                 f"{GALLERY_RING} point with the largest image(+2) - image(-2)\n"
                 "Ray-traced noise-free donut and Danish fits", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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


def quad_fit(y, e, h):
    """Weighted fit y = a h + b h^2 through the nominal (k = 0, exactly zero).

    Returns a, b, sigma_a, sigma_b (µm per DOF unit, per DOF unit squared)."""
    ok = np.isfinite(y) & (h != 0)
    if ok.sum() < 3:
        return (np.nan,) * 4
    X = np.c_[h[ok], h[ok] ** 2] / e[ok, None]
    c, *_ = np.linalg.lstsq(X, y[ok] / e[ok], rcond=None)
    cov = np.linalg.inv(X.T @ X)
    return c[0], c[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


NL_METRICS = {
    "term": "quadratic over linear change of the same DZ term at +2 steps (measured)",
    "impact": "quadratic change of the DZ term over the DOF's total linear DZ change "
              "(norm over all DZ), at the ts_ofc v13 DOF range (extrapolated)",
}


def nonlinear_terms(S, val, err, mstep, k, noll, rng, metric, n=10, nsig=5.0):
    """Leading non-linear DZ terms with the quadratic significant at ``nsig`` sigma.

    ``term``: |b| h / |a| at h = +2 steps, also requiring the slope at ``nsig`` sigma so
    the ratio is defined. ``impact``: |b| r^2 / (||a_DOF|| r) at the DOF range r, where
    ||a_DOF|| is the norm of the DOF's Smatrix slopes over all DZ. Returns (value, DOF,
    field index, pupil index) tuples, largest first."""
    h = k[:, None] * mstep[None, :]
    cand = []
    for i in range(50):
        anorm = np.linalg.norm(S[1:, noll, i])
        for kf in range(KMAX):
            for jz in range(len(noll)):
                a, b, sa, sb = quad_fit(val[:, i, kf, jz], err[:, i, kf, jz], h[:, i])
                if not (np.isfinite(sb) and abs(b) > nsig * sb):
                    continue
                if metric == "term":
                    if abs(a) > nsig * sa:
                        cand.append((abs(b) * h[-1, i] / abs(a), i, kf, jz))
                else:
                    cand.append((abs(b) * rng[i] / anorm, i, kf, jz))
    return sorted(cand, reverse=True)[:n]


def _dz_panel(ax, i, kf, jz, S, val, err, h, noll, ref, rng=None):
    sl = S[1:, noll, i]
    y, e = val[:, i, kf, jz], err[:, i, kf, jz]
    ax.errorbar(h[:, i], y, e, fmt="o", ms=3, c="C0")
    hh = np.linspace(h[0, i], h[-1, i], 50)
    ax.plot(hh, sl[kf, jz] * hh, "k-", lw=0.8)
    q, ratio = quad_ratio(y, h[:, i])
    ax.plot(hh, np.polyval(q, hh), "C1--", lw=0.8)
    title = (f"{DOF_NAMES[i]}: DZ(field {kf + 1}, Z{noll[jz]})\nslope {sl[kf, jz]:.3g} "
             f"µm / {DOF_UNIT[i]}; quad / lin {ratio:.3f}")
    if ref is not None:
        r = ref[kf + 1, noll[jz], i]
        ax.plot(hh, r * hh, "C2:", lw=1.2)
        title += f"\nv13 slope {r:.3g} (ratio Danish / v13 {sl[kf, jz] / r:.2f})"
    if rng is not None:
        a, b, _, sb = quad_fit(y, e, h[:, i])
        title += (f" ({abs(sl[kf, jz]) / np.abs(sl).max():.2g} of the DOF's largest)"
                  f"\nquad at +2 steps {abs(b) * h[-1, i] ** 2:.2g} µm ({abs(b) / sb:.0f} "
                  f"sigma); impact at range {abs(b) * rng[i] / np.linalg.norm(sl):.3g}"
                  f"\n(range = {rng[i] / h[-1, i]:.3g} x the +2 step)")
    ax.set_title(title, fontsize=6)
    ax.tick_params(labelsize=6)


def nonlinear_page(pdf, S, val, err, mstep, k, noll, fs, ref, rng, metric):
    h = k[:, None] * mstep[None, :]
    terms = nonlinear_terms(S, val, err, mstep, k, noll, rng, metric)
    fig, axs = plt.subplots(5, 2, figsize=(8.5, 11))
    for ax, (_, i, kf, jz) in zip(axs.ravel(), terms):
        _dz_panel(ax, i, kf, jz, S, val, err, h, noll, ref, rng)
        ax.set_xlabel(f"DOF amplitude ({DOF_UNIT[i]})", fontsize=6)
        ax.set_ylabel("DZ change (µm)", fontsize=6)
    fig.suptitle(textwrap.fill(
        f"{fs}: the 10 most non-linear DZ terms, quadratic significant at 5 sigma, "
        f"ranked by {NL_METRICS[metric]} (dimensionless). 'impact at range': quadratic "
        "change over the DOF's total linear DZ change at the v13 range, extrapolated by the "
        "quadratic from +-2 steps (for rigid body the range is up to 400 x the +2 step)"
        + ("; green dotted: v13 in-focus slope" if ref is not None else ""), 110),
        fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)
    return terms


def dz_pages(pdf, S, val, err, mstep, k, noll, fs, ref=None):
    h = k[:, None] * mstep[None, :]
    for p0 in range(0, 50, 5):
        fig, axs = plt.subplots(5, 4, figsize=(10, 11))
        for r, i in enumerate(range(p0, p0 + 5)):
            sl, top = top4(S, noll, i)
            for c, (kf, jz) in enumerate(top):
                ax = axs[r, c]
                _dz_panel(ax, i, kf, jz, S, val, err, h, noll, ref)
                if r == 4 or i == 49:
                    ax.set_xlabel(f"DOF amplitude ({DOF_UNIT[i]})", fontsize=6)
                if c == 0:
                    ax.set_ylabel("DZ change (µm)", fontsize=6)
        fig.suptitle(f"{fs}: DZ change against DOF amplitude, four largest DZ slopes per "
                     f"DOF. Points: FAM field fit (kmax {KMAX}) at k = -2..+2 steps; "
                     "solid: Smatrix slope; dashed: quadratic"
                     + (";\ngreen dotted: ts_ofc v13 in-focus (OPD) Smatrix slope" if ref
                        is not None else "")
                     + ".\nquad / lin: quadratic over linear change at +2 steps "
                     "(dimensionless)", fontsize=7.5)
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


def _rows(noll, kmax=KMAX):
    """Flattened (31 x 29) row indices for field 1..kmax, pupil noll; j-major order."""
    return np.array([kf * 29 + j for j in noll for kf in range(1, kmax + 1)])


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
        rows = _rows(nolls[fs], KSHOW)
        lab = [f"Z{j} k{kf}" for j in nolls[fs] for kf in range(1, KSHOW + 1)]
        A = (se.ofc_data.sensitivity_matrix.reshape(-1, 50) @ se.normalization_matrix)[rows]
        ov = np.abs(se.Vh @ rf.Vh.T)
        pages = [
            (se.Vh.T, "Vh: DOF vs v-mode (dimensionless, unit rows)", "v-mode index",
             DOF_NAMES, "sym"),
            (se.U[rows], f"U: DZ vs v-mode, field k 1-{KSHOW} of 1-{KMAX} (dimensionless; "
             "columns unit norm over all rows)", "v-mode index", lab, "sym"),
            (A, f"S N: DZ vs DOF, field k 1-{KSHOW} of 1-{KMAX} (µm of DZ per unit "
             "normalized DOF)", "DOF", lab, "sym"),
            (ov, "|Vh . Vh_v13^T|: v-mode overlap with ts_ofc v13 in-focus, same pupil "
             "terms (dimensionless)", "ts_ofc v13 v-mode index", None, "pos"),
        ]
        for a, title, xl, ylab, mode in pages:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            lim = np.percentile(np.abs(a), 99.5) if mode == "sym" else 1.0
            x_vm, y_vm = xl != "DOF", ylab is None
            ext = ((0.5, 50.5) if x_vm else (-0.5, a.shape[1] - 0.5)) + \
                ((0.5, 50.5) if y_vm else (-0.5, a.shape[0] - 0.5))
            im = ax.imshow(a, aspect="auto", origin="lower", interpolation="nearest",
                           cmap="RdBu_r" if mode == "sym" else "viridis",
                           vmin=-lim if mode == "sym" else 0, vmax=lim, extent=ext)
            # v-modes numbered from 1, with a guide line every 10 modes.
            for on, line, ticks in ((x_vm, ax.axvline, ax.set_xticks),
                                    (y_vm, ax.axhline, ax.set_yticks)):
                if on:
                    ticks([1, 10, 20, 30, 40, 50])
                    for v in (10.5, 20.5, 30.5, 40.5):
                        line(v, c="0.4", lw=0.5, ls="--")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                         extend="both" if mode == "sym" else "neither")
            if mode == "sym":
                title += f"\ncolour scale clipped at the 99.5th percentile of |value|, {lim:.3g}"
            ax.set_title(f"{fs}: {title}", fontsize=8)
            if xl == "DOF":
                ax.set_xticks(range(50))
                ax.set_xticklabels(DOF_NAMES, rotation=90, fontsize=5)
                for x in (9.5, 29.5):
                    ax.axvline(x, c="k", lw=0.6)
            else:
                ax.set_xlabel(xl + " (from 1)")
            if ylab is not None:
                ax.set_yticks(range(len(ylab)))
                ax.set_yticklabels(ylab, fontsize=4.5 if len(ylab) > 60 else 6)
            else:
                ax.set_ylabel(f"{fs} v-mode index (from 1)")
            if ylab is DOF_NAMES:
                for y in (9.5, 29.5):
                    ax.axhline(y, c="k", lw=0.6)
            elif ylab is not None:
                for y in range(1, len(nolls[fs])):
                    ax.axhline(y * KSHOW - 0.5, c="k", lw=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", default="20261111_responses_ofcsign")
    p.add_argument("--scan", default="20261111_dz_scan_ofcsign")
    p.add_argument("--no-gallery", action="store_true")
    p.add_argument("--gallery-run", default=None,
                   help="reuse gallery.npz from this run instead of refitting")
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
    v13 = se_full.ofc_data.sensitivity_matrix
    rng = dof_range()
    for fs in FIT_SETS:
        val, err, mstep = pts[fs]
        for metric, label in (("term", "quad / lin of the term at +2 steps"),
                              ("impact", "impact at the v13 range")):
            terms = nonlinear_terms(mats[fs], val, err, mstep, scan["k"], nolls[fs], rng,
                                    metric)
            summary.append(f"{fs}: most non-linear DZ, {label} (dimensionless): "
                           + "; ".join(f"{DOF_NAMES[i]} DZ({kf + 1},Z{nolls[fs][jz]}) "
                                       f"{r:.3g}" for r, i, kf, jz in terms[:5]))
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
    for fs in FIT_SETS:
        n = nolls[fs]
        cos = [np.dot(mats[fs][1:, n, i].ravel(), v13[1:KMAX + 1, n, i].ravel())
               / np.linalg.norm(mats[fs][1:, n, i]) / np.linalg.norm(v13[1:KMAX + 1, n, i])
               for i in range(10)]
        summary.append(f"{fs}: rigid-body Smatrix columns against v13, cosine "
                       f"(dimensionless): min {min(cos):.4f}, max {max(cos):.4f}")
    print("\n".join(summary), flush=True)
    run = study_run("smatrix/sparse_fit", args.label)
    np.savez(f"{run.path}/validation.npz", **out)
    gal = None
    if args.gallery_run:
        gal = dict(np.load(f"{OUT_ROOT}/{args.gallery_run}/gallery.npz"))
    elif not args.no_gallery:
        gal = gallery_fits(f"{run.path}/gallery.npz")
    with PdfPages(f"{run.path}/sparse_fit_validation.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.04, 0.97, VALIDATION_TEXT + "\n\nResults\n" + "\n".join(
            textwrap.fill(s, 92, initial_indent="- ", subsequent_indent="  ")
            for s in summary), va="top", family="monospace", fontsize=6.8)
        pdf.savefig(fig)
        plt.close(fig)
        if gal is not None:
            gallery_pages(pdf, gal)
        for fs in FIT_SETS:
            val, err, mstep = pts[fs]
            ref = v13 if fs == "dense21" else None
            dz_pages(pdf, mats[fs], val, err, mstep, scan["k"], nolls[fs], fs, ref)
            for metric in NL_METRICS:
                nonlinear_page(pdf, mats[fs], val, err, mstep, scan["k"], nolls[fs], fs,
                               ref, rng, metric)
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
then 50 gallery pages, one per DOF: donuts at +2 and -2 steps and their
  difference, and per fit set the model and data - model at +-2 steps, at the
  FAM ring-4 point (field radius 1.06 deg) where image(+2) - image(-2) is largest
then per fit set 10 pages: DZ against DOF amplitude, four largest DZ per DOF
  (dense21 also with the ts_ofc v13 in-focus slope), and 2 pages: the 10 most
  non-linear DZ terms by the term's own quad / lin at +2 steps, and by impact
  (quadratic change over the DOF's total linear DZ change) at the v13 DOF range
then singular values, and per fit set 4 pages: Vh, U and S N (field k 1-6 shown),
  and the overlap with the ts_ofc v-modes

Notes
- DOF signs are the ts_ofc convention: M2 dy, M2 Ry, Cam dy and Cam Ry flipped
  from batoid_rubin ZCS (smatrix/docs/conventions.md). The donuts were made in ZCS;
  the scans are symmetric, so their amplitudes are negated on reading.
- v-modes are numbered from 1.
- v-mode signs are as the SVD returns them; compare |overlap|, not signs.
- Near-degenerate singular-value pairs mix within the pair, so pairs show as 2 x 2
  blocks in the overlap.
- The ts_ofc reference is the in-focus OPD Smatrix: it has no Danish leakage of
  unfitted terms, which is large for high bending modes."""


if __name__ == "__main__":
    main()
