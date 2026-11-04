"""Charge diffusion from cosmic-ray muon track widths, after Fisher-Levine 2015.

Reads the per-visit files written by ``find_cosmics.py``. Two stages:

1. Measure (slow, cached): for each track, refine the line, find the ends at half
   the median charge per unit length, measure straightness and off-line charge,
   cut the track into sections and fit a Gaussian width to each.
2. Select and fit (fast): apply the cuts, orient each track so its wide end is the
   back surface (longest drift), and fit per track

       sigma^2 = sigma_0^2 + k^2 * D * dz

   with a Huber robust linear fit, where sigma is the width in um, D = 100 um the
   thickness, dz the drift distance in um and k the dimensionless diffusion
   coefficient. For a uniform field, k = sqrt(2 kT / (e V)) (Fisher-Levine eq.
   3.3).

Writes one PDF (summary, results, then a validation plot for every quantity cut on,
then a gallery) and the measurement tables into a new run directory. Every
validation page has a REB0 panel and a REB1 panel side by side, one colour per CCD.

Run from the repo root::

    python camera/cosmic_diffusion/code/analyze_diffusion.py \
        --find-run 20261103_find_20261102_v2

Change cuts without remeasuring::

    python camera/cosmic_diffusion/code/analyze_diffusion.py \
        --find-run 20261103_find_20261102_v2 \
        --reuse 20261103_diffusion --max-d0-rms 0.3
"""

import argparse
import glob
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from astropy.io import fits  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cosmic_tools import (PIXEL_PITCH_UM, THICKNESS_UM, find_ends, nmad,  # noqa: E402
                          project, theory_k, track_sections)

STUDY = "camera/cosmic_diffusion"
SIGMA0_UM = PIXEL_PITCH_UM / np.sqrt(12)
T_K = 173.15
CCDS = {0: ["R21_S00", "R21_S01", "R21_S02"], 1: ["R21_S10", "R21_S11", "R21_S12"]}
CCD_COLOR = {"R21_S00": "C3", "R21_S01": "C1", "R21_S02": "C5",
             "R21_S10": "C0", "R21_S11": "C2", "R21_S12": "C9"}
STATUS_ORDER = ["short", "near_edge", "crooked", "delta_ray", "few_sections",
                "ambiguous", "ok"]


def huber(x, y):
    """Huber robust fit of y = a + b x; returns (a, b, a_err, b_err)."""
    res = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    return (*res.params, *res.bse)


def huber_sig0(dz, s2, sig0_um, prior_um):
    """Huber fit of s2 = sigma0^2 + k^2 D dz with sigma0 constrained.

    Parameters
    ----------
    dz : `numpy.ndarray`
        Drift distance, um.
    s2 : `numpy.ndarray`
        Squared section width, um^2.
    sig0_um : `float`
        Expected sigma0, um.
    prior_um : `float`
        Gaussian prior width on sigma0, um; 0 fixes sigma0.

    Returns
    -------
    a, b, b_err : `float`
        Fitted sigma0^2 (um^2), k^2 D (um) and its error (um).
    """
    a0 = sig0_um ** 2
    if prior_um <= 0:
        res = sm.RLM(s2 - a0, dz[:, None], M=sm.robust.norms.HuberT()).fit()
        return a0, res.params[0], res.bse[0]
    # Iteratively reweighted least squares: Huber weights on the data rows, a fixed
    # Gaussian weight on the one prior row, so the prior is not itself downweighted.
    xmat = sm.add_constant(dz)
    w_prior = 1.0 / (2 * sig0_um * prior_um) ** 2
    beta = np.linalg.lstsq(xmat, s2, rcond=None)[0]
    huber_norm = sm.robust.norms.HuberT()
    for _ in range(50):
        r = s2 - xmat @ beta
        scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
        w = huber_norm.weights(r / scale) / scale ** 2
        amat = (xmat * w[:, None]).T @ xmat
        amat[0, 0] += w_prior
        rhs = (xmat * w[:, None]).T @ s2
        rhs[0] += w_prior * a0
        new = np.linalg.solve(amat, rhs)
        if np.allclose(new, beta, rtol=1e-6, atol=1e-8):
            beta = new
            break
        beta = new
    cov = np.linalg.inv(amat)
    return beta[0], beta[1], np.sqrt(cov[1, 1])


def signed_k(b):
    """k = sign(b) sqrt(|b| / D) from b = k^2 D in um; dimensionless."""
    return np.sign(b) * np.sqrt(np.abs(b) / THICKNESS_UM)


# ---------------------------------------------------------------- measurement

def refine_line(stamp, fit, tmin, tmax, noise):
    """Refit the line through 2 px section centres.

    Returns
    -------
    fit : `dict`
        Refined line.
    d0_rms, d0_maxdev : `float`
        Rms and largest absolute residual of the section centres about the
        refined line, pixels; the straightness measures.
    """
    secs = track_sections(stamp, np.zeros(stamp.shape, bool), fit, tmin, tmax, noise,
                          sec_len=2.0, max_sections=25, n_iter=1)
    if len(secs) < 4:
        return fit, np.nan, np.nan
    t = np.array([tmin + s["frac"] * (tmax - tmin) for s in secs])
    d0 = np.array([s["d0"] for s in secs])
    a, b = np.polyfit(t, d0, 1)
    resid = d0 - (a * t + b)
    ux, uy = fit["ux"], fit["uy"]
    nx, ny = -uy, ux
    norm = np.hypot(1, a)
    # Shift the centre to the fitted line at t = 0, then rotate.
    new = dict(fit, xc=fit["xc"] + b * nx, yc=fit["yc"] + b * ny,
               ux=(ux + a * nx) / norm, uy=(uy + a * ny) / norm)
    return new, float(np.sqrt(np.mean(resid ** 2))), float(np.abs(resid).max())


def measure_track(hdu, exposure, args):
    """All per-track and per-section quantities for one stamp; no cuts applied."""
    h = hdu.header
    stamp = hdu.data.astype(float)
    noise = h["NOISE_E"]
    fit = dict(xc=h["XC"] - h["X0"], yc=h["YC"] - h["Y0"], ux=h["UX"], uy=h["UY"])
    fit, d0_rms, d0_max = refine_line(stamp, fit, h["TMIN"], h["TMAX"], noise)
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    t, d = project(xx.ravel(), yy.ravel(), fit)
    v = stamp.ravel()
    ok = np.isfinite(v)
    on = ok & (v > args.low_sigma * noise)
    t3a, t3b = t[on].min(), t[on].max()
    ta, tb = find_ends(stamp, fit, t3a, t3b)
    secs = track_sections(stamp, np.zeros(stamp.shape, bool), fit, ta, tb, noise,
                          sec_len=args.sec_len, max_sections=args.max_sections)
    sig_px = np.array([s["sigma"] for s in secs]) if secs else np.array([0.5])
    # Off-line charge, measured beyond the track's own widest part so that
    # diffusion at low bias is not mistaken for a delta ray.
    d_cut = 1.0 + 3.0 * np.percentile(sig_px, 90)
    region = on & (t >= ta - 2) & (t <= tb + 2)
    tot = v[region].sum()
    halo = v[region & (np.abs(d) > d_cut)].sum() / tot if tot > 0 else np.nan
    # Fisher-Levine style: weight each pixel exponentially in its distance.
    expw = np.log10((v[region] * np.exp(np.abs(d[region]))).sum() / tot) \
        if tot > 0 else np.nan
    row = {k: h[k] for k in ["DETNAME", "REB", "BSSVBS", "CCDTEMP", "LENGTH", "ANGLE",
                             "PEAKADU", "DEDX", "ENDDIST", "CROSSAMP", "SUM_E"]}
    row.update(exposure=exposure, extname=hdu.name, t_a=ta, t_b=tb,
               xc=fit["xc"], yc=fit["yc"], ux=fit["ux"], uy=fit["uy"],
               length_half=tb - ta, length_3sig=t3b - t3a, d0_rms=d0_rms,
               d0_maxdev=d0_max, halo_frac=halo, halo_dcut=d_cut, expw_log10=expw,
               n_sec=len(secs), noise_e=noise)
    sec = pd.DataFrame(secs)
    if not sec.empty:
        sec["exposure"] = exposure
        sec["extname"] = hdu.name
        sec["DETNAME"] = h["DETNAME"]
        sec["REB"] = h["REB"]
    return row, sec


def measure_all(files, args):
    rows, secs = [], []
    t0 = time.time()
    for i, f in enumerate(files):
        with fits.open(f) as hl:
            exposure = hl[0].header["EXPOSURE"]
            for hdu in hl[1:]:
                if hdu.header["LENGTH"] < args.min_length - 3:
                    continue
                r, s = measure_track(hdu, exposure, args)
                r["file"] = f
                rows.append(r)
                secs.append(s)
        if i % 50 == 0:
            print(f"measured {i + 1}/{len(files)} visits, {len(rows)} tracks, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows), pd.concat(secs, ignore_index=True)


# ------------------------------------------------------------ selection, fits

def select(tracks, secs, args):
    """Apply cuts, orient tracks and fit each one. Returns new frames."""
    tracks = tracks.copy()
    secs = secs.copy()
    secs["sigma_um"] = secs.sigma * PIXEL_PITCH_UM
    secs["sigma_err_um"] = secs.sigma_err * PIXEL_PITCH_UM
    secs["good"] = (secs.chi2_dof < args.max_chi2) & (secs.sigma_err < args.max_sig_err)
    key = ["exposure", "extname"]
    ngood = secs[secs.good].groupby(key).size().rename("n_good")
    tracks = tracks.drop(columns=[c for c in ["n_good"] if c in tracks]).merge(
        ngood, left_on=key, right_index=True, how="left")
    tracks["n_good"] = tracks.n_good.fillna(0).astype(int)

    fits_out = {}
    for (exp, ext), g in secs[secs.good].groupby(key):
        if len(g) < 3:
            continue
        f, s = g.frac.values, g.sigma_um.values
        try:
            _, b, _, be = huber(f, s)
            flip = b < 0
            dz = (1 - f if flip else f) * THICKNESS_UM
            a2, b2, a2e, b2e = huber(dz, s ** 2)
            _, bf, bfe = huber_sig0(dz, s ** 2, SIGMA0_UM, 0.0)
            ap, bp, bpe = huber_sig0(dz, s ** 2, SIGMA0_UM, args.sig0_prior)
        except (ValueError, np.linalg.LinAlgError):
            continue
        n3 = max(len(g) // 3, 1)
        order = np.argsort(dz)
        fits_out[(exp, ext)] = dict(
            slope_um=abs(b), slope_z=abs(b) / be if be > 0 else np.inf, flip=flip,
            sig_front_um=np.median(s[order[:n3]]), sig_back_um=np.median(s[order[-n3:]]),
            sig0sq_um2=a2, sig0sq_err_um2=a2e, k2D_um=b2, k2D_err_um=b2e,
            k2D_fix_um=bf, k2D_prior_um=bp, sig0sq_prior_um2=ap)
    fdf = pd.DataFrame.from_dict(fits_out, orient="index")
    fdf.index.names = key
    tracks = tracks.drop(columns=[c for c in fdf.columns if c in tracks]).merge(
        fdf, left_on=key, right_index=True, how="left")
    tracks["k_signed"] = signed_k(tracks.k2D_um)
    tracks["k_fix"] = signed_k(tracks.k2D_fix_um)
    tracks["k_prior"] = signed_k(tracks.k2D_prior_um)
    tracks["sig0_prior_um"] = np.sign(tracks.sig0sq_prior_um2) * np.sqrt(
        np.abs(tracks.sig0sq_prior_um2))
    tracks["sig0_signed_um"] = np.sign(tracks.sig0sq_um2) * np.sqrt(
        np.abs(tracks.sig0sq_um2))

    status = np.full(len(tracks), "ok", dtype=object)
    conds = [("ambiguous", ~(tracks.slope_z >= args.min_slope_z)),
             ("few_sections", tracks.n_good < args.min_good_sec),
             ("delta_ray", tracks.halo_frac > args.max_halo),
             ("crooked", ~(tracks.d0_rms <= args.max_d0_rms)),
             ("near_edge", tracks.ENDDIST < args.min_end_dist),
             ("short", tracks.length_half < args.min_length)]
    # Later entries win, so a track is labelled by the first cut it fails in
    # STATUS_ORDER.
    for name, c in conds:
        status[c.values] = name
    tracks["status"] = status

    flip = secs.merge(tracks[key + ["flip", "status"]], on=key, how="left")
    secs["status"] = flip.status.values
    secs["dz_um"] = np.where(flip.flip.fillna(False).values, 1 - secs.frac.values,
                             secs.frac.values) * THICKNESS_UM
    return tracks, secs


def summarize(tracks, secs, args):
    """Per CCD and per REB: median per-track k and sigma0, and pooled fits.

    Three fits of sigma^2 = sigma0^2 + k^2 D dz: sigma0 free, sigma0 fixed at
    pitch / sqrt(12), and sigma0 with a Gaussian prior of width
    ``args.sig0_prior`` um about that value.
    """
    rows = []
    ok = tracks[tracks.status == "ok"]
    s_ok = secs[(secs.status == "ok") & secs.good]
    groups = [(f"REB{r}", ok.REB == r, s_ok.REB == r) for r in (0, 1)]
    groups += [(n, ok.DETNAME == n, s_ok.DETNAME == n) for n in CCDS[0] + CCDS[1]]
    for name, mt, ms in groups:
        g, s = ok[mt], s_ok[ms]
        if len(g) < 2:
            continue
        n = len(g)
        k_med = np.median(g.k_signed)
        row = dict(group=name, volts=g.BSSVBS.median(), n_trk=n, n_sec=len(s),
                   k_med=k_med, k_med_err=1.2533 * nmad(g.k_signed) / np.sqrt(n),
                   k_nmad=nmad(g.k_signed),
                   sig0_med=np.median(g.sig0_signed_um),
                   sig0_med_err=1.2533 * nmad(g.sig0_signed_um) / np.sqrt(n),
                   k_fix_med=np.median(g.k_fix),
                   k_fix_err=1.2533 * nmad(g.k_fix) / np.sqrt(n),
                   k_prior_med=np.median(g.k_prior),
                   k_prior_err=1.2533 * nmad(g.k_prior) / np.sqrt(n),
                   sig0_prior_med=np.median(g.sig0_prior_um),
                   k_th=theory_k(g.BSSVBS.median(), T_K))
        dz, s2 = s.dz_um.values, s.sigma_um.values ** 2
        a, b, ae, be = huber(dz, s2)
        _, bf, _ = huber_sig0(dz, s2, SIGMA0_UM, 0.0)
        ap, bp, _ = huber_sig0(dz, s2, SIGMA0_UM, args.sig0_prior)
        row.update(k_pool=signed_k(b), sig0_pool=np.sign(a) * np.sqrt(abs(a)),
                   k_pool_fix=signed_k(bf), k_pool_prior=signed_k(bp),
                   sig0_pool_prior=np.sign(ap) * np.sqrt(abs(ap)))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plotting

def two_panels(title, nrows=1):
    """Figure with REB0 (left) and REB1 (right) columns."""
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.2 * nrows + 0.6), squeeze=False)
    fig.suptitle(title, fontsize=11)
    return fig, axes


def label_cols(axes, df):
    for reb in (0, 1):
        v = df[df.REB == reb].BSSVBS.median() if "BSSVBS" in df else np.nan
        axes[0, reb].set_title(f"REB{reb} ({v:.1f} V)", fontsize=9)


def hist_ccd(ax, df, reb, col, bins, logy=False, extra=None):
    for n in CCDS[reb]:
        v = df.loc[df.DETNAME == n, col].dropna().values
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", lw=1.3,
                color=CCD_COLOR[n], label=f"{n}, n = {len(v)}"
                + (extra(v) if extra else ""))
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7)


def cut_line(ax, x, text):
    ax.axvline(x, color="k", ls="--", lw=1)
    ax.text(x, 0.97, f" {text}", transform=ax.get_xaxis_transform(), fontsize=7,
            va="top")


def page_hist(pdf, specs, df, title):
    """One page, one row per spec: (col, bins, xlabel, cuts, logx, logy)."""
    fig, axes = two_panels(title, nrows=len(specs))
    for i, (col, bins, xlabel, cuts, logx, logy) in enumerate(specs):
        for reb in (0, 1):
            ax = axes[i, reb]
            hist_ccd(ax, df, reb, col, bins, logy=logy)
            for x, txt in cuts:
                cut_line(ax, x, txt)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("count per bin", fontsize=8)
    label_cols(axes, df)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_summary(pdf, tracks, summ, nvis, args):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.965, "cosmic_diffusion: charge diffusion from muon tracks",
             fontsize=13, weight="bold")
    counts = tracks.groupby(["REB", "status"]).size().unstack(0).reindex(
        STATUS_ORDER).fillna(0).astype(int)
    lines = [f"find runs: {', '.join(args.find_run)};  visits: {nvis};  tracks "
             f"measured: {len(tracks)}", "",
             "Track status (first cut failed), count per REB:",
             f"{'status':<14}{'REB0':>8}{'REB1':>8}"]
    for st, r in counts.iterrows():
        lines.append(f"{st:<14}{r.get(0, 0):>8}{r.get(1, 0):>8}")
    lines += ["", f"Cuts: length_half >= {args.min_length} px; end-to-edge/defect "
              f">= {args.min_end_dist} px; d0 rms <= {args.max_d0_rms} px;",
              "  off-line charge fraction <= {0}; good sections >= {1} "
              "(chi2/dof < {2}, sigma err < {3} px);".format(
                  args.max_halo, args.min_good_sec, args.max_chi2, args.max_sig_err),
              f"  orientation |slope| / err >= {args.min_slope_z}.", "",
              "Per-track Huber fit sigma^2 = sigma0^2 + k^2 D dz, D = 100 um, with "
              "sigma0 free, with a",
              f"Gaussian prior sigma0 = {SIGMA0_UM:.2f} +- {args.sig0_prior:.2f} um, "
              f"and with sigma0 fixed at {SIGMA0_UM:.2f} um.",
              "Columns are medians over ok tracks; error = 1.2533 nMAD / sqrt(n).",
              "Pooled: one Huber fit to all good sections of the group.", "",
              "k (dimensionless):",
              f"{'group':<9}{'V':>7}{'n_trk':>7}{'free':>17}{'prior':>17}"
              f"{'fixed':>8}{'pool free':>10}{'pool prior':>11}{'theory':>8}"]
    for _, r in summ.iterrows():
        lines.append(f"{r.group:<9}{r.volts:>7.2f}{r.n_trk:>7d}"
                     f"{r.k_med:>9.4f}+-{r.k_med_err:<6.4f}"
                     f"{r.k_prior_med:>9.4f}+-{r.k_prior_err:<6.4f}"
                     f"{r.k_fix_med:>8.4f}{r.k_pool:>10.4f}{r.k_pool_prior:>11.4f}"
                     f"{r.k_th:>8.4f}")
    lines += ["", "sigma0 (um):",
              f"{'group':<9}{'free median':>17}{'prior median':>14}{'pool free':>11}"
              f"{'pool prior':>11}"]
    for _, r in summ.iterrows():
        lines.append(f"{r.group:<9}{r.sig0_med:>9.2f}+-{r.sig0_med_err:<6.2f}"
                     f"{r.sig0_prior_med:>14.2f}{r.sig0_pool:>11.2f}"
                     f"{r.sig0_pool_prior:>11.2f}")
    lines += ["", "k is dimensionless; k theory = sqrt(2kT/(eV)) at T = 173.15 K "
              "(Fisher-Levine eq. 3.3).", "V is BSSVBS, the back-bias voltage from "
              "the raw header, in volts.",
              f"Expected sigma0 for a track along a row: 10 um / sqrt(12) = "
              f"{SIGMA0_UM:.2f} um.", "",
              "Pages: results (sigma vs dz per CCD, k and sigma0), dE/dx, finder cut "
              "validation,", "analysis cut validation, gallery."]
    fig.text(0.05, 0.93, "\n".join(lines), family="monospace", fontsize=7.3, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_sigma_dz(pdf, secs, summ):
    """Raw section width against inferred drift distance, one panel per CCD."""
    s = secs[(secs.status == "ok") & secs.good]
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), sharey=True)
    fig.suptitle("Section width against inferred drift distance, ok tracks "
                 "(cf. Fisher-Levine fig. 5); curves from median per-track fits",
                 fontsize=11)
    dzb = np.linspace(0, 100, 21)
    cen = 0.5 * (dzb[1:] + dzb[:-1])
    zz = np.linspace(0, 100, 200)
    for reb in (0, 1):
        for j, n in enumerate(CCDS[reb]):
            ax = axes[reb, j]
            g = s[s.DETNAME == n]
            if g.empty:
                continue
            if len(g) > 2000:
                ax.hist2d(g.dz_um, g.sigma_um, bins=[dzb, np.linspace(0, 14, 57)],
                          cmap="Greys", cmin=1)
            else:
                ax.plot(g.dz_um, g.sigma_um, ".", color="0.5", ms=3)
            idx = np.digitize(g.dz_um, dzb) - 1
            med = np.array([np.median(g.sigma_um[idx == i]) if (idx == i).sum() >= 3
                            else np.nan for i in range(len(cen))])
            ax.plot(cen, med, "o", color=CCD_COLOR[n], mec="k", ms=4,
                    label="binned median")
            r = summ[summ.group == n]
            if not r.empty:
                r = r.iloc[0]
                ax.plot(zz, np.sqrt(np.clip(np.sign(r.sig0_med) * r.sig0_med ** 2
                                            + r.k_med ** 2 * THICKNESS_UM * zz, 0,
                                            None)),
                        ":", color=CCD_COLOR[n],
                        label=f"sigma0 free: k = {r.k_med:.4f}, "
                              f"sigma0 = {r.sig0_med:.2f} um")
                ax.plot(zz, np.sqrt(np.clip(np.sign(r.sig0_prior_med)
                                            * r.sig0_prior_med ** 2
                                            + r.k_prior_med ** 2 * THICKNESS_UM * zz,
                                            0, None)),
                        "-", color=CCD_COLOR[n],
                        label=f"sigma0 prior: k = {r.k_prior_med:.4f}, "
                              f"sigma0 = {r.sig0_prior_med:.2f} um")
                ax.plot(zz, np.sqrt(SIGMA0_UM ** 2 + r.k_th ** 2 * THICKNESS_UM * zz),
                        "k--", lw=1, label=f"uniform field: k = {r.k_th:.4f}, "
                                           f"sigma0 = {SIGMA0_UM:.2f} um")
                ax.set_title(f"{n}, {r.volts:.1f} V, {r.n_trk} tracks", fontsize=9)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 14)
            ax.legend(fontsize=6, loc="upper left")
            ax.set_xlabel("drift distance dz (um), 0 = gates, 100 = back", fontsize=8)
            if j == 0:
                ax.set_ylabel("section width sigma (um)", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_k_sig0(pdf, tracks, summ, sig0_prior):
    ok = tracks[tracks.status == "ok"]
    fig, axes = two_panels("Per-track fit: sigma0 against k, ok tracks", nrows=3)
    for reb in (0, 1):
        ax = axes[0, reb]
        for n in CCDS[reb]:
            g = ok[ok.DETNAME == n]
            ax.plot(g.k_signed, g.sig0_signed_um, ".", ms=3 if len(ok) > 2000 else 6,
                    alpha=0.4 if len(ok) > 2000 else 0.9,
                    color=CCD_COLOR[n], label=f"{n}, n = {len(g)}")
            r = summ[summ.group == n]
            if not r.empty:
                ax.plot(r.k_med, r.sig0_med, "*", ms=14, color=CCD_COLOR[n], mec="k")
        kth = theory_k(ok[ok.REB == reb].BSSVBS.median(), T_K) if len(ok) else np.nan
        ax.axvline(kth, color="k", ls="--", lw=1)
        ax.axhline(SIGMA0_UM, color="k", ls=":", lw=1)
        ax.set_xlim(-0.05, 0.2)
        ax.set_ylim(-4, 6)
        ax.set_xlabel("k = sign(b) sqrt(|b| / D) (dimensionless)", fontsize=8)
        ax.set_ylabel("sigma0 = sign(a) sqrt(|a|) (um)", fontsize=8)
        ax.legend(fontsize=7, markerscale=4)
        ax.text(0.98, 0.03, "stars: median per CCD; dashed: uniform-field k;\n"
                "dotted: 10 um / sqrt(12)", transform=ax.transAxes, ha="right",
                fontsize=7)
        ax = axes[1, reb]
        hist_ccd(ax, ok, reb, "k_signed", np.linspace(-0.05, 0.2, 51),
                 extra=lambda v: f", median {np.median(v):.4f}" if len(v) else "")
        ax.axvline(kth, color="k", ls="--", lw=1)
        ax.set_xlabel("per-track k, sigma0 free (dimensionless)", fontsize=8)
        ax.set_ylabel("tracks per bin", fontsize=8)
        ax = axes[2, reb]
        hist_ccd(ax, ok, reb, "k_prior", np.linspace(-0.05, 0.2, 51),
                 extra=lambda v: f", median {np.median(v):.4f}" if len(v) else "")
        ax.axvline(kth, color="k", ls="--", lw=1)
        ax.set_xlabel(f"per-track k, sigma0 prior {SIGMA0_UM:.2f} +- "
                      f"{sig0_prior:.2f} um (dimensionless)", fontsize=8)
        ax.set_ylabel("tracks per bin", fontsize=8)
    label_cols(axes, ok)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_k_summary(pdf, summ, sig0_prior):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    p = summ[summ.group.str.startswith("R21")].reset_index(drop=True)
    ax = axes[0]
    for i, r in p.iterrows():
        ax.errorbar(i - 0.12, r.k_med, r.k_med_err, fmt="o", mfc="none",
                    color=CCD_COLOR[r.group], capsize=3)
        ax.errorbar(i + 0.12, r.k_prior_med, r.k_prior_err, fmt="o",
                    color=CCD_COLOR[r.group], capsize=3)
        ax.plot(i + 0.24, r.k_fix_med, "s", ms=4, color=CCD_COLOR[r.group])
        ax.plot([i - 0.3, i + 0.3], [r.k_th] * 2, "k--", lw=1)
    ax.set_xticks(p.index)
    ax.set_xticklabels([f"{g}\n{v:.1f} V" for g, v in zip(p.group, p.volts)],
                       fontsize=7)
    ax.set_ylabel("k (dimensionless)")
    ax.set_title("median per-track k: open o sigma0 free, filled o sigma0 prior "
                 f"{SIGMA0_UM:.2f} +- {sig0_prior:.2f} um,\nsquare sigma0 fixed at "
                 f"{SIGMA0_UM:.2f} um; -- uniform field", fontsize=8)
    ax.set_ylim(0, None)
    ax = axes[1]
    vv = np.linspace(5, 60, 200)
    ax.plot(vv, theory_k(vv, T_K), "k--", label="sqrt(2kT/eV), T = 173.15 K")
    for _, r in p.iterrows():
        ax.errorbar(r.volts, r.k_prior_med, r.k_prior_err, fmt="o",
                    color=CCD_COLOR[r.group], label=r.group)
    ax.set_xlabel("back-bias voltage BSSVBS (V)")
    ax.set_ylabel("median per-track k, sigma0 prior (dimensionless)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, None)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_amp_sigma_ccd(pdf, secs, summ):
    """Gaussian amplitude against sigma per CCD, good sections of ok tracks."""
    s = secs[(secs.status == "ok") & secs.good]
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), sharex=True, sharey=True)
    fig.suptitle("Section Gaussian amplitude against sigma, good sections of ok "
                 "tracks", fontsize=11)
    xb = np.logspace(np.log10(50), np.log10(2e4), 80)
    yb = np.linspace(0, 1.6, 81)
    for reb in (0, 1):
        for j, n in enumerate(CCDS[reb]):
            ax = axes[reb, j]
            g = s[s.DETNAME == n]
            h = ax.hist2d(g.amp, g.sigma, bins=[xb, yb], cmap="viridis", cmin=1,
                          norm=matplotlib.colors.LogNorm())
            fig.colorbar(h[3], ax=ax, label="sections per bin")
            ax.set_xscale("log")
            r = summ[summ.group == n]
            v = r.volts.iloc[0] if len(r) else np.nan
            ax.set_title(f"{n}, {v:.1f} V, {len(g)} sections", fontsize=9)
            if reb == 1:
                ax.set_xlabel("Gaussian amplitude (electrons)", fontsize=8)
            if j == 0:
                ax.set_ylabel("Gaussian sigma (pixels)", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def langau_pdf(x, mpv, eta, sigma):
    """Landau (most probable value mpv, scale eta) convolved with a Gaussian."""
    grid = np.linspace(x.min() - 10 * sigma, x.max() + 10 * sigma, 4000)
    dx = grid[1] - grid[0]
    # scipy's landau mode sits at loc - 0.22278 * scale.
    land = stats.landau.pdf(grid, loc=mpv + 0.22278 * eta, scale=eta)
    kern = stats.norm.pdf(np.arange(-5 * sigma, 5 * sigma + dx, dx), scale=sigma)
    return np.interp(x, grid, np.convolve(land, kern, mode="same") * dx)


def fit_dedx(vals, bins):
    hist, edges = np.histogram(vals, bins=bins)
    c = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]

    def model(x, n, mpv, eta, sg):
        return n * w * langau_pdf(x, mpv, eta, sg)
    try:
        popt, pcov = curve_fit(model, c, hist, p0=[len(vals), c[np.argmax(hist)], 5, 5],
                               sigma=np.sqrt(np.maximum(hist, 1)),
                               bounds=([0, 0, 0.5, 0.5], [np.inf, 500, 100, 100]))
        return popt, np.sqrt(np.diag(pcov)), model
    except (RuntimeError, ValueError):
        return None, None, model


def page_dedx(pdf, tracks):
    ok = tracks[tracks.status == "ok"].copy()
    path = np.hypot(ok.length_half * PIXEL_PITCH_UM, THICKNESS_UM)
    ok["dedx"] = ok.SUM_E / path
    fig, axes = two_panels("Validation: dE/dx of ok tracks, charge over path "
                           "sqrt((10 um L)^2 + (100 um)^2)")
    bins = np.linspace(0, 250, 51)
    for reb in (0, 1):
        ax = axes[0, reb]
        g = ok[ok.REB == reb]
        hist_ccd(ax, g, reb, "dedx", bins,
                 extra=lambda v: f", median {np.median(v):.1f}" if len(v) else "")
        popt, perr, model = fit_dedx(g.dedx.values[g.dedx.values < 250], bins)
        if popt is not None:
            xx = np.linspace(0, 250, 500)
            ax.plot(xx, model(xx, *popt) / 3, "k-", lw=1,
                    label=f"Landau*Gauss / 3, MPV {popt[1]:.1f} +- {perr[1]:.1f} e/um")
            ax.legend(fontsize=7)
        ax.set_xlabel("dE/dx (electrons per um of path in silicon)", fontsize=8)
        ax.set_ylabel("tracks per bin", fontsize=8)
    label_cols(axes, ok)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def finder_pages(pdf, src, args):
    """Validation of every finder cut, on all detected sources."""
    src = src.copy()
    src["log_peak"] = src.peak_adu
    page_hist(pdf, [
        ("npix", np.logspace(np.log10(3), 4, 50), "pixels above 3 sigma per source",
         [(args.f_min_npix, "min 5")], True, True),
        ("n_hi", np.logspace(0, 4, 50), "pixels above 8 sigma per source",
         [(1, "need >= 1")], True, True)],
        src, "Finder validation: source size, all detected sources")
    page_hist(pdf, [
        ("length_px", np.logspace(0, 3, 50), "projected length L (pixels)",
         [(args.f_min_length, "finder min"), (args.min_length, "analysis min")],
         True, True),
        ("perp_rms_px", np.linspace(0, 4, 81),
         "charge-weighted rms across the track, perp_rms (pixels)",
         [(args.f_max_perp, "finder max")], False, True)],
        src, "Finder validation: length and width, all detected sources")
    fig, axes = two_panels("Finder validation: peak, and length against perp_rms",
                           nrows=2)
    for reb in (0, 1):
        hist_ccd(axes[0, reb], src, reb, "peak_adu", np.logspace(1, 5.2, 60),
                 logy=True)
        axes[0, reb].set_xscale("log")
        cut_line(axes[0, reb], args.f_max_peak, "max peak")
        axes[0, reb].set_xlabel("peak pixel (ADU)", fontsize=8)
        ax = axes[1, reb]
        for n in CCDS[reb]:
            g = src[src.DETNAME == n]
            hot = g.hot_col | g.columnar
            ax.plot(g.length_px[~hot], g.perp_rms_px[~hot], ".", ms=1.5, alpha=0.4,
                    color=CCD_COLOR[n], label=n)
            ax.plot(g.length_px[hot], g.perp_rms_px[hot], "kx", ms=3)
        ax.plot([], [], "kx", label="hot column or columnar (rejected)")
        ax.axhline(args.f_max_perp, color="k", ls="--", lw=1)
        ax.axvline(args.f_min_length, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(1, 1000)
        ax.set_ylim(0, 4)
        ax.set_xlabel("projected length L (pixels)", fontsize=8)
        ax.set_ylabel("perp_rms (pixels)", fontsize=8)
        ax.legend(fontsize=7, markerscale=5)
    label_cols(axes, src)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def analysis_pages(pdf, tracks, secs, args):
    """Validation of every analysis cut."""
    t = tracks
    page_hist(pdf, [
        ("length_half", np.linspace(0, 120, 61),
         "track length between half-charge ends (pixels)",
         [(args.min_length, "min")], False, True),
        ("ENDDIST", np.linspace(0, 60, 61),
         "track end to nearest detector edge or defect (pixels)",
         [(args.min_end_dist, "min")], False, True)],
        t, "Analysis validation: length and ends, all measured tracks")
    fig, axes = two_panels("Analysis validation: half-charge length against "
                           "3-sigma-pixel length", nrows=1)
    for reb in (0, 1):
        ax = axes[0, reb]
        for n in CCDS[reb]:
            g = t[t.DETNAME == n]
            ax.plot(g.length_3sig, g.length_half - g.length_3sig, ".", ms=2,
                    alpha=0.4, color=CCD_COLOR[n], label=n)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("length spanned by pixels > 3 sigma (pixels)", fontsize=8)
        ax.set_ylabel("half-charge length minus that (pixels)", fontsize=8)
        ax.set_xlim(0, 120)
        ax.set_ylim(-4, 6)
        ax.legend(fontsize=7, markerscale=5)
    label_cols(axes, t)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    page_hist(pdf, [
        ("d0_rms", np.linspace(0, 1.0, 51),
         "rms of 2 px section centres about the line (pixels)",
         [(args.max_d0_rms, "max")], False, True),
        ("d0_maxdev", np.linspace(0, 2.0, 51),
         "largest section-centre deviation (pixels), not cut on", [], False, True)],
        t, "Analysis validation: straightness")
    page_hist(pdf, [
        ("halo_frac", np.linspace(0, 0.3, 61),
         "fraction of track charge beyond 1 px + 3 x (90th percentile sigma)",
         [(args.max_halo, "max")], False, True),
        ("expw_log10", np.linspace(-0.5, 3, 71),
         "log10 sum(q exp(|d| / 1 px)) / sum(q), Fisher-Levine style, not cut on",
         [], False, True)],
        t, "Analysis validation: delta rays (off-line charge)")
    page_hist(pdf, [
        ("n_sec", np.arange(-0.5, args.max_sections + 1.5), "sections per track",
         [], False, False),
        ("n_good", np.arange(-0.5, args.max_sections + 1.5),
         "good sections per track", [(args.min_good_sec - 0.5, "min")], False,
         False)],
        t, "Analysis validation: sections per track")

    fig, axes = two_panels("Analysis validation: section Gaussian fits, all "
                           "sections", nrows=4)
    for reb in (0, 1):
        s = secs[secs.REB == reb]
        ax = axes[0, reb]
        for n in CCDS[reb]:
            g = s[s.DETNAME == n]
            ax.plot(g.amp[g.good], g.sigma[g.good], ".", ms=1, alpha=0.3,
                    color=CCD_COLOR[n], label=f"{n} good")
        ax.plot(s.amp[~s.good], s.sigma[~s.good], "kx", ms=2, alpha=0.5,
                label="failed chi2 or error cut")
        ax.set_xscale("log")
        ax.set_xlim(10, 3e4)
        ax.set_ylim(0, 3.1)
        ax.set_xlabel("Gaussian amplitude (electrons)", fontsize=8)
        ax.set_ylabel("Gaussian sigma (pixels)", fontsize=8)
        ax.legend(fontsize=7, markerscale=6)
        ax = axes[1, reb]
        hist_ccd(ax, s, reb, "chi2_dof", np.logspace(-1.5, 2, 60), logy=True)
        ax.set_xscale("log")
        cut_line(ax, args.max_chi2, "max")
        ax.set_xlabel("section fit chi2/dof, 10% model error included "
                      "(dof = pixels - 3)", fontsize=8)
        ax = axes[2, reb]
        if "chi2_dof_nomod" in s:
            ax.hist2d(s.amp, np.clip(s.chi2_dof, 0.03, 99),
                      bins=[np.logspace(1.5, 4.5, 60), np.logspace(-1.5, 2, 60)],
                      cmap="Greys", cmin=1, norm=matplotlib.colors.LogNorm())
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axhline(args.max_chi2, color="r", ls="--", lw=1)
            ax.set_xlabel("Gaussian amplitude (electrons)", fontsize=8)
            ax.set_ylabel("chi2/dof, model error included", fontsize=8)
        ax = axes[3, reb]
        hist_ccd(ax, s, reb, "sigma_err", np.logspace(-3, 0.5, 60), logy=True)
        ax.set_xscale("log")
        cut_line(ax, args.max_sig_err, "max")
        ax.set_xlabel("section sigma error (pixels)", fontsize=8)
    label_cols(axes, secs)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    has = t[t.slope_z.notna()]
    fig, axes = two_panels("Analysis validation: orientation (front against back "
                           "width), tracks with >= 3 good sections", nrows=2)
    for reb in (0, 1):
        g = has[has.REB == reb]
        ax = axes[0, reb]
        amb = g.slope_z < args.min_slope_z
        for n in CCDS[reb]:
            m = (g.DETNAME == n) & ~amb
            ax.plot(g.sig_front_um[m], g.sig_back_um[m], ".", ms=2, alpha=0.4,
                    color=CCD_COLOR[n], label=f"{n}, oriented")
        ax.plot(g.sig_front_um[amb], g.sig_back_um[amb], "kx", ms=3,
                label=f"ambiguous, n = {amb.sum()}")
        ax.plot([0, 14], [0, 14], "k-", lw=0.5)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 14)
        ax.set_xlabel("median sigma, narrow third (um)", fontsize=8)
        ax.set_ylabel("median sigma, wide third (um)", fontsize=8)
        ax.legend(fontsize=7, markerscale=4)
        ax = axes[1, reb]
        hist_ccd(ax, g, reb, "slope_z", np.linspace(0, 20, 81), logy=True)
        cut_line(ax, args.min_slope_z, "min")
        ax.set_xlabel("|width slope| / its error along the track (dimensionless)",
                      fontsize=8)
    label_cols(axes, has)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = two_panels("Validation of the depth mapping: charge per unit length "
                           "against dz, ok tracks (flat if the ends are right)")
    s = secs[(secs.status == "ok") & secs.good]
    dzb = np.linspace(0, 100, 11)
    cen = 0.5 * (dzb[1:] + dzb[:-1])
    for reb in (0, 1):
        ax = axes[0, reb]
        for n in CCDS[reb]:
            g = s[s.DETNAME == n].merge(t[["exposure", "extname", "length_half"]],
                                        on=["exposure", "extname"])
            # Normalise each section by its track's median so tracks of
            # different angle and Landau draw can be stacked.
            g["q_rel"] = g.charge_per_px / g.groupby(
                ["exposure", "extname"]).charge_per_px.transform("median")
            idx = np.digitize(g.dz_um, dzb) - 1
            med = [np.median(g.q_rel[idx == i]) if (idx == i).sum() >= 5 else np.nan
                   for i in range(10)]
            ax.plot(cen, med, "o-", color=CCD_COLOR[n], label=n)
        ax.axhline(1, color="k", lw=0.5)
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel("drift distance dz (um)", fontsize=8)
        ax.set_ylabel("section charge per px / track median (dimensionless)",
                      fontsize=8)
        ax.legend(fontsize=7)
    label_cols(axes, t)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_gallery(pdf, tracks, secs, per_page=6, ok_pages=2, seed=3):
    """Gallery: the longest ok tracks, ``ok_pages`` pages per CCD, then a random
    sample of each rejection class per REB."""
    rng = np.random.default_rng(seed)
    chunks = []
    for n in CCDS[0] + CCDS[1]:
        g = tracks[(tracks.DETNAME == n) & (tracks.status == "ok")]
        g = g.sort_values("length_half", ascending=False).head(per_page * ok_pages)
        chunks += [g.iloc[i:i + per_page] for i in range(0, len(g), per_page)]
    for reb in (0, 1):
        g = tracks[tracks.REB == reb]
        pick = []
        for st in STATUS_ORDER[:-1]:
            h = g[g.status == st]
            if len(h):
                pick.append(h.iloc[rng.choice(len(h), min(3, len(h)), replace=False)])
        if pick:
            rej = pd.concat(pick)
            chunks += [rej.iloc[i:i + per_page] for i in range(0, len(rej), per_page)]
    for chunk in chunks:
        fig, axes = plt.subplots(per_page, 2, figsize=(8.5, 11),
                                 gridspec_kw=dict(width_ratios=[1.3, 1]))
        for (ax_im, ax_w), (_, r) in zip(axes, chunk.iterrows()):
            with fits.open(r.file) as hl:
                img = hl[r.extname].data.astype(float)
            im = ax_im.imshow(img, origin="lower", cmap="magma", vmin=-20,
                              vmax=np.nanpercentile(img, 99.5),
                              interpolation="nearest")
            fig.colorbar(im, ax=ax_im, fraction=0.04, label="electrons")
            # Front (gates, dz = 0) is the narrow end; flip means it is t_b.
            flip = bool(r.flip) if pd.notna(r.get("flip", np.nan)) else False
            t_front, t_back = (r.t_b, r.t_a) if flip else (r.t_a, r.t_b)
            ax_im.plot(r.xc + t_front * r.ux, r.yc + t_front * r.uy, "o", ms=7,
                       mfc="none", mec="lime", mew=1.5, label="front, dz = 0")
            ax_im.plot(r.xc + t_back * r.ux, r.yc + t_back * r.uy, "s", ms=7,
                       mfc="none", mec="cyan", mew=1.5, label="back, dz = 100 um")
            ax_im.set_title(f"{r.exposure} {r.extname} {r.BSSVBS:.1f} V "
                            f"L = {r.length_half:.0f} px [{r.status}]", fontsize=7)
            ax_im.tick_params(labelsize=6)
            g = secs[(secs.extname == r.extname) & (secs.exposure == r.exposure)]
            if not g.empty:
                ax_w.errorbar(g.dz_um[g.good], g.sigma_um[g.good],
                              g.sigma_err_um[g.good], fmt="o", ms=3,
                              label="section used in fit")
                ax_w.plot(g.dz_um[~g.good], g.sigma_um[~g.good], "x", color="0.5",
                          label="section failed chi2/dof or sigma error cut")
                if np.isfinite(r.get("k2D_um", np.nan)):
                    zz = np.linspace(0, 100, 100)
                    ax_w.plot(zz, np.sqrt(np.clip(r.sig0sq_um2 + r.k2D_um * zz, 0,
                                                  None)), "r:", lw=1,
                              label="fit, sigma0 free")
                    ax_w.plot(zz, np.sqrt(np.clip(r.sig0sq_prior_um2
                                                  + r.k2D_prior_um * zz, 0, None)),
                              "r-", lw=1, label="fit, sigma0 prior")
                    ax_w.text(0.03, 0.85, f"k free = {r.k_signed:.3f}, "
                              f"k prior = {r.k_prior:.3f}, "
                              f"d0 rms = {r.d0_rms:.2f} px, halo = {r.halo_frac:.3f}",
                              transform=ax_w.transAxes, fontsize=6.5)
            ax_w.set_ylim(0, 14)
            ax_w.set_xlim(0, 100)
            ax_w.tick_params(labelsize=6)
            ax_w.set_xlabel("dz (um)", fontsize=7)
            ax_w.set_ylabel("sigma (um)", fontsize=7)
        for ax_im, ax_w in axes[len(chunk):]:
            ax_im.axis("off")
            ax_w.axis("off")
        h, lab = axes[0][0].get_legend_handles_labels()
        for ax_w in axes[:, 1]:
            hw, lw_ = ax_w.get_legend_handles_labels()
            if len(hw) == 4:
                h, lab = h + hw, lab + lw_
                break
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.legend(h, lab, loc="upper center", ncol=3, fontsize=7,
                   title="track ends at half charge per length; image markers:")
        pdf.savefig(fig)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--find-run", nargs="+", required=True,
                   help="run directory names of find_cosmics.py in this study's output")
    p.add_argument("--label", default="diffusion")
    p.add_argument("--reuse", help="earlier run directory to take measurements from")
    p.add_argument("--low-sigma", type=float, default=3.0)
    p.add_argument("--sec-len", type=float, default=3.0, help="pixels")
    p.add_argument("--max-sections", type=int, default=10)
    p.add_argument("--min-length", type=float, default=15.0, help="pixels")
    p.add_argument("--min-end-dist", type=float, default=3.0, help="pixels")
    p.add_argument("--max-d0-rms", type=float, default=0.25, help="pixels")
    p.add_argument("--max-halo", type=float, default=0.02,
                   help="off-line charge fraction, dimensionless")
    p.add_argument("--max-chi2", type=float, default=5.0, help="chi2/dof per section")
    p.add_argument("--max-sig-err", type=float, default=0.3, help="pixels")
    p.add_argument("--min-good-sec", type=int, default=4)
    p.add_argument("--min-slope-z", type=float, default=1.0,
                   help="|width slope| / error, dimensionless")
    p.add_argument("--sig0-prior", type=float, default=0.5,
                   help="Gaussian prior width on sigma0 about 10 um / sqrt(12), um")
    # Finder cuts, drawn on the finder validation pages only.
    p.add_argument("--f-min-npix", type=float, default=5)
    p.add_argument("--f-min-length", type=float, default=10)
    p.add_argument("--f-max-perp", type=float, default=2.0)
    p.add_argument("--f-max-peak", type=float, default=50000)
    args = p.parse_args()

    from rubinwork.common import study_run
    from rubinwork.common.runs import data_root
    base = data_root() / "studies" / STUDY
    run = study_run(STUDY, args.label)
    files = []
    for fr in args.find_run:
        files += sorted(glob.glob(str(base / fr / "visits" / "cosmics_*.fits")))
    src = pd.concat([pd.read_parquet(f.replace("cosmics_", "tracks_")
                                     .replace(".fits", ".parquet")) for f in files],
                    ignore_index=True)
    if args.reuse:
        tracks = pd.read_parquet(base / args.reuse / "measured_tracks.parquet")
        secs = pd.read_parquet(base / args.reuse / "measured_sections.parquet")
    else:
        tracks, secs = measure_all(files, args)
    tracks.to_parquet(run.path / "measured_tracks.parquet")
    secs.to_parquet(run.path / "measured_sections.parquet")

    tracks, secs = select(tracks, secs, args)
    summ = summarize(tracks, secs, args)
    tracks.to_parquet(run.path / "tracks.parquet")
    summ.to_csv(run.path / "summary.csv", index=False)
    print(tracks.groupby(["status", "REB"]).size().unstack().to_string())
    print(summ.round(4).to_string())

    with PdfPages(run.path / "cosmic_diffusion.pdf") as pdf:
        page_summary(pdf, tracks, summ, len(files), args)
        page_sigma_dz(pdf, secs, summ)
        page_k_sig0(pdf, tracks, summ, args.sig0_prior)
        page_k_summary(pdf, summ, args.sig0_prior)
        page_amp_sigma_ccd(pdf, secs, summ)
        page_dedx(pdf, tracks)
        finder_pages(pdf, src, args)
        analysis_pages(pdf, tracks, secs, args)
        page_gallery(pdf, tracks, secs)
    print(f"wrote {run.path / 'cosmic_diffusion.pdf'}")


if __name__ == "__main__":
    main()
