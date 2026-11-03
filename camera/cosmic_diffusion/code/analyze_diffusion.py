"""Charge diffusion from cosmic-ray muon track widths, after Fisher-Levine 2015.

Reads the per-visit files written by ``find_cosmics.py``. For each long straight
track: refine the line, cut it into sections, fit a Gaussian width to each, orient
the track so the wide end is the back surface (longest drift), and fit

    sigma^2 = sigma_0^2 + k^2 * D * dz

with a Huber robust linear fit, where sigma is the width in um, D = 100 um the
thickness, dz the drift distance in um and k the dimensionless diffusion
coefficient. For a uniform field, k = sqrt(2 kT / (e V)) (Fisher-Levine eq. 3.3).

Writes one PDF (summary, dE/dx, pooled width against depth, k against voltage, the
gallery) and ``sections.parquet`` and ``tracks.parquet`` into a new run directory.

Run from the repo root::

    python camera/cosmic_diffusion/code/analyze_diffusion.py \
        --find-run 20261103_find_20261102
"""

import argparse
import glob
import pathlib
import sys

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
from cosmic_tools import (PIXEL_PITCH_UM, THICKNESS_UM, line_fit, nmad,  # noqa: E402
                          project, theory_k, track_sections)

STUDY = "camera/cosmic_diffusion"
SIGMA0_UM = PIXEL_PITCH_UM / np.sqrt(12)


def huber(x, y):
    """Huber robust fit of y = a + b x; returns (a, b, a_err, b_err, scale)."""
    res = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    return (*res.params, *res.bse, res.scale)


def refine_line(stamp, fit, tmin, tmax, noise):
    """Refit the line through the per-section Gaussian centres."""
    secs = track_sections(stamp, np.zeros(stamp.shape, bool), fit, tmin, tmax, noise,
                          sec_len=2.0, max_sections=40)
    if len(secs) < 4:
        return fit
    t = np.array([tmin + s["frac"] * (tmax - tmin) for s in secs])
    d0 = np.array([s["d0"] for s in secs])
    a, b = np.polyfit(t, d0, 1)
    ux, uy = fit["ux"], fit["uy"]
    nx, ny = -uy, ux
    xc = fit["xc"] + b * nx
    yc = fit["yc"] + b * ny
    norm = np.hypot(1, a)
    ux2, uy2 = (ux + a * nx) / norm, (uy + a * ny) / norm
    return dict(fit, xc=xc, yc=yc, ux=ux2, uy=uy2)


def analyze_track(hdu, args):
    """Sections and per-track fit for one stamp.

    Returns
    -------
    track : `dict`
    sections : `pandas.DataFrame`
    """
    h = hdu.header
    stamp = hdu.data.astype(float)
    fit = dict(xc=h["XC"] - h["X0"], yc=h["YC"] - h["Y0"], ux=h["UX"], uy=h["UY"])
    noise = h["NOISE_E"]
    fit = refine_line(stamp, fit, h["TMIN"], h["TMAX"], noise)
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    t, d = project(xx.ravel(), yy.ravel(), fit)
    v = stamp.ravel()
    ok = np.isfinite(v)
    on = ok & (v > args.low_sigma * noise)
    tmin, tmax = t[on].min(), t[on].max()
    # Delta rays put charge far from the line; Fisher-Levine reject such tracks.
    near = ok & (t >= tmin) & (t <= tmax)
    sig = near & (v > args.low_sigma * noise)
    halo = v[sig & (np.abs(d) > 2.5)].sum() / v[sig].sum()
    secs = track_sections(stamp, np.zeros(stamp.shape, bool), fit, tmin, tmax, noise,
                          sec_len=args.sec_len, max_sections=args.max_sections)
    meta = {k: h[k] for k in ["DETNAME", "RAFT", "SENSOR", "REB", "BSSVBS", "CCDTEMP",
                              "LENGTH", "ANGLE", "PEAKADU", "DEDX", "EDGE",
                              "CROSSAMP"]}
    meta.update(extname=hdu.name, exposure=args._exposure, halo_frac=halo,
                length_fit=tmax - tmin)
    sec = pd.DataFrame(secs)
    if sec.empty:
        return dict(meta, status="no_sections"), sec
    sec["sigma_um"] = sec["sigma"] * PIXEL_PITCH_UM
    sec["sigma_err_um"] = sec["sigma_err"] * PIXEL_PITCH_UM
    good = (sec["chi2_dof"] < args.max_chi2) & (sec["sigma_err"] < 0.3)
    sec["good"] = good
    status = "ok"
    if meta["EDGE"]:
        status = "edge"
    elif halo > args.max_halo:
        status = "delta_ray"
    elif good.sum() < 3:
        status = "few_sections"
    out = dict(meta, n_sec=int(good.sum()))
    if status == "ok":
        f = sec.loc[good, "frac"].values
        s = sec.loc[good, "sigma_um"].values
        a, b, ae, be, _ = huber(f, s)
        out.update(slope_um=b, slope_err_um=be)
        if abs(b) < be:
            status = "ambiguous"
        # Wide end is the back surface: longest drift.
        dz = (f if b > 0 else 1 - f) * THICKNESS_UM
        sec.loc[good, "dz_um"] = dz
        sec.loc[~good, "dz_um"] = ((sec.loc[~good, "frac"] if b > 0
                                    else 1 - sec.loc[~good, "frac"]) * THICKNESS_UM)
        a2, b2, a2e, b2e, _ = huber(dz, s ** 2)
        out.update(sig0sq_um2=a2, sig0sq_err_um2=a2e, k2D_um=b2, k2D_err_um=b2e,
                   k=np.sqrt(b2 / THICKNESS_UM) if b2 > 0 else np.nan,
                   sigma_back_um=np.sqrt(max(a2 + b2 * THICKNESS_UM, 0)))
    out["status"] = status
    for c in ["DETNAME", "REB", "BSSVBS", "extname", "exposure"]:
        sec[c] = out.get(c, meta.get(c))
    sec["status"] = status
    return out, sec


def langau_pdf(x, mpv, eta, sigma):
    """Landau (location mpv, scale eta) convolved with a Gaussian, unit area."""
    grid = np.linspace(x.min() - 10 * sigma, x.max() + 10 * sigma, 4000)
    dx = grid[1] - grid[0]
    # scipy's landau mode sits at loc - 0.22278 * scale.
    land = stats.landau.pdf(grid, loc=mpv + 0.22278 * eta, scale=eta)
    kern = stats.norm.pdf(np.arange(-5 * sigma, 5 * sigma + dx, dx), scale=sigma)
    conv = np.convolve(land, kern, mode="same") * dx
    return np.interp(x, grid, conv)


def fit_dedx(vals, bins):
    """Binned Landau-Gauss fit; returns (popt, perr, hist, centres)."""
    hist, edges = np.histogram(vals, bins=bins)
    c = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    model = lambda x, n, mpv, eta, sg: n * w * langau_pdf(x, mpv, eta, sg)  # noqa: E731
    p0 = [len(vals), c[np.argmax(hist)], 5.0, 5.0]
    try:
        popt, pcov = curve_fit(model, c, hist, p0=p0, sigma=np.sqrt(np.maximum(hist, 1)),
                               bounds=([0, 0, 0.5, 0.5], [np.inf, 500, 100, 100]))
        return popt, np.sqrt(np.diag(pcov)), hist, c, model
    except RuntimeError:
        return None, None, hist, c, model


def page_summary(pdf, tracks, pooled, nvis, find_runs):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.96, "cosmic_diffusion: charge diffusion from muon tracks",
             fontsize=14, weight="bold")
    lines = [f"find runs: {', '.join(find_runs)}", f"visits: {nvis}",
             f"tracks read: {len(tracks)}; status counts: "
             + ", ".join(f"{k} {v}" for k, v in tracks.status.value_counts().items()),
             "", "Pooled Huber fit, sigma^2 = sigma0^2 + k^2 D dz, D = 100 um,"
             " T = 173.15 K:", ""]
    hdr = f"{'group':<14}{'V (V)':>8}{'n_trk':>7}{'n_sec':>7}{'sigma0 (um)':>13}" \
          f"{'k (dimensionless)':>19}{'k_boot':>8}{'k_theory':>10}{'k/k_th':>8}"
    lines.append(hdr)
    for _, r in pooled.iterrows():
        lines.append(f"{r.group:<14}{r.volts:>8.2f}{r.n_trk:>7d}{r.n_sec:>7d}"
                     f"{r.sigma0:>8.2f}+-{r.sigma0_err:<4.2f}"
                     f"{r.k:>11.4f}+-{r.k_err:<6.4f}{r.k_boot:>8.4f}{r.k_th:>10.4f}"
                     f"{r.k / r.k_th:>8.2f}")
    lines += ["", "k_theory = sqrt(2 kT/(e V)), Fisher-Levine eq. 3.3, uniform field.",
              "sigma0 expected = pitch/sqrt(12) = 2.89 um.",
              "k/k_th is dimensionless: measured k over the uniform-field value.",
              "k_boot: bootstrap error on k over whole tracks; the formal error "
              "ignores correlation within a track."]
    fig.text(0.05, 0.92, "\n".join(lines), family="monospace", fontsize=7.5, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_dedx(pdf, all_src):
    sel = all_src[all_src.long & all_src.straight & ~all_src.hot_col & ~all_src.columnar
                  & ~all_src.edge]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
    for ax, (lab, df) in zip(axes, [("all R21", sel),
                                    ("by REB", None)]):
        if df is not None:
            vals = df.dedx_e_per_um.values
            bins = np.linspace(0, 250, 51)
            popt, perr, hist, c, model = fit_dedx(vals[vals < 250], bins)
            ax.stairs(hist, bins, color="k", label=f"tracks, n = {len(vals)}")
            if popt is not None:
                xx = np.linspace(0, 250, 500)
                ax.plot(xx, model(xx, *popt), "r-",
                        label=f"Landau*Gauss, MPV = {popt[1]:.1f} +- {perr[1]:.1f} e/um")
                ax.axvline(popt[1], color="r", ls=":")
            ax.set_title(f"dE/dx, long straight tracks, {lab}")
        else:
            bins = np.linspace(0, 250, 26)
            for reb, g in sel.groupby("REB"):
                v = g.dedx_e_per_um.values
                ax.hist(v, bins=bins, histtype="step", lw=1.5,
                        label=f"REB{reb} ({g.BSSVBS.median():.1f} V), n = {len(v)}, "
                              f"median = {np.median(v):.1f} e/um")
            ax.set_title("dE/dx by REB")
        ax.set_xlabel("dE/dx (electrons per um of path in silicon)")
        ax.set_ylabel("tracks per bin")
        ax.legend(fontsize=8)
    axes[0].text(0.98, 0.6, "path = sqrt((L * 10 um)^2 + (100 um)^2)\n"
                 "expected MPV for 100 um Si: about 70 to 80 e/um",
                 transform=axes[0].transAxes, ha="right", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pooled_fits(tracks, secs):
    rows = []
    ok = tracks[tracks.status == "ok"]
    s = secs[(secs.status == "ok") & secs.good]
    groups = [("REB0", s.REB == 0), ("REB1", s.REB == 1)]
    groups += [(n, s.DETNAME == n) for n in sorted(s.DETNAME.unique())]
    for name, m in groups:
        g = s[m]
        if len(g) < 5:
            continue
        a, b, ae, be, _ = huber(g.dz_um.values, g.sigma_um.values ** 2)
        volts = g.BSSVBS.median()
        k = np.sqrt(b / THICKNESS_UM) if b > 0 else np.nan
        rows.append(dict(group=name, volts=volts,
                         n_trk=int(ok.extname.isin(g.extname).sum()
                                   if name.startswith("REB") else
                                   (ok.DETNAME == name).sum()),
                         n_sec=len(g), sig0sq=a, sig0sq_err=ae, k2D=b, k2D_err=be,
                         sigma0=np.sqrt(max(a, 0)),
                         sigma0_err=ae / (2 * np.sqrt(a)) if a > 0 else np.nan,
                         k=k, k_err=be / (2 * THICKNESS_UM * k) if k > 0 else np.nan,
                         k_th=theory_k(volts, 173.15),
                         k_boot=bootstrap_k(g)))
    return pd.DataFrame(rows)


def page_pooled(pdf, secs, pooled):
    s = secs[(secs.status == "ok") & secs.good]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
    dzb = np.linspace(0, 100, 11)
    for reb, col in [(0, "C3"), (1, "C0")]:
        g = s[s.REB == reb]
        if g.empty:
            continue
        p = pooled[pooled.group == f"REB{reb}"].iloc[0]
        ax = axes[0]
        ax.plot(g.dz_um, g.sigma_um, ".", color=col, alpha=0.25, ms=3)
        idx = np.digitize(g.dz_um, dzb) - 1
        med = [np.median(g.sigma_um[idx == i]) if (idx == i).sum() >= 3 else np.nan
               for i in range(10)]
        cen = 0.5 * (dzb[1:] + dzb[:-1])
        ax.plot(cen, med, "o", color=col, mec="k",
                label=f"REB{reb}, {p.volts:.1f} V: binned median")
        zz = np.linspace(0, 100, 200)
        ax.plot(zz, np.sqrt(p.sig0sq + p.k2D * zz), "-", color=col,
                label=f"fit: sigma0 = {p.sigma0:.2f} um, k = {p.k:.4f}")
        ax.plot(zz, np.sqrt(SIGMA0_UM ** 2 + p.k_th ** 2 * THICKNESS_UM * zz), "--",
                color=col, label=f"uniform-field theory, k = {p.k_th:.4f}")
        ax2 = axes[1]
        diff = np.sqrt(np.clip(np.array(med) ** 2 - SIGMA0_UM ** 2, 0, None))
        ax2.plot(cen, diff, "o", color=col, mec="k", label=f"REB{reb}, {p.volts:.1f} V")
        ax2.plot(zz, p.k * np.sqrt(THICKNESS_UM * zz), "-", color=col)
        ax2.plot(zz, p.k_th * np.sqrt(THICKNESS_UM * zz), "--", color=col)
    axes[0].set_ylabel("track width sigma (um)")
    axes[0].set_title("Section width against drift distance (cf. Fisher-Levine fig. 5)")
    axes[1].set_ylabel("sigma_diff = sqrt(sigma^2 - (2.89 um)^2) (um)")
    axes[1].set_title("Diffusion part; solid: fit, dashed: k = sqrt(2kT/eV)")
    for ax in axes:
        ax.set_xlabel("drift distance dz (um), 0 = front (gates), 100 = back")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, None)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def bootstrap_k(g, n_boot=200, seed=2):
    """Bootstrap error on k, resampling whole tracks, dimensionless."""
    rng = np.random.default_rng(seed)
    keys = g.exposure.astype(str) + g.extname
    ids = keys.unique()
    if ids.size < 5:
        return np.nan
    groups = {i: g[keys == i] for i in ids}
    ks = []
    for _ in range(n_boot):
        sample = pd.concat([groups[i] for i in rng.choice(ids, ids.size)])
        _, b, _, _, _ = huber(sample.dz_um.values, sample.sigma_um.values ** 2)
        ks.append(np.sqrt(b / THICKNESS_UM) if b > 0 else np.nan)
    return np.nanstd(ks)


def page_per_ccd(pdf, secs, pooled):
    """Width against drift distance for each sensor, and k per sensor."""
    s = secs[(secs.status == "ok") & secs.good]
    names = sorted(s.DETNAME.unique())
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(4, 3)
    dzb = np.linspace(0, 100, 11)
    cen = 0.5 * (dzb[1:] + dzb[:-1])
    zz = np.linspace(0, 100, 200)
    for i, name in enumerate(names[:6]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        g = s[s.DETNAME == name]
        p = pooled[pooled.group == name]
        ax.plot(g.dz_um, g.sigma_um, ".", color="0.6", ms=2)
        idx = np.digitize(g.dz_um, dzb) - 1
        med = [np.median(g.sigma_um[idx == j]) if (idx == j).sum() >= 3 else np.nan
               for j in range(10)]
        ax.plot(cen, med, "o", color="C3" if g.REB.iloc[0] == 0 else "C0", mec="k",
                ms=4)
        if not p.empty:
            p = p.iloc[0]
            ax.plot(zz, np.sqrt(np.clip(p.sig0sq + p.k2D * zz, 0, None)), "k-", lw=1)
            ax.set_title(f"{name}, {p.volts:.1f} V\nk = {p.k:.4f} +- {p.k_boot:.4f}, "
                         f"n_trk = {p.n_trk}", fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 12)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("dz (um)", fontsize=7)
        ax.set_ylabel("sigma (um)", fontsize=7)
    ax = fig.add_subplot(gs[2:, :])
    p = pooled[pooled.group.str.startswith("R21")].reset_index(drop=True)
    for reb, col in [(0, "C3"), (1, "C0")]:
        q = p[p.group.str[5] == str(reb)]
        ax.errorbar(q.index, q.k, q.k_boot, fmt="o", color=col, capsize=3,
                    label=f"REB{reb} ({q.volts.median():.1f} V), bootstrap over tracks")
        r = pooled[pooled.group == f"REB{reb}"]
        if not r.empty:
            ax.axhline(r.k.iloc[0], color=col, ls=":", lw=1)
            ax.axhline(r.k_th.iloc[0], color=col, ls="--", lw=1)
        q = q[np.isfinite(q.k_boot)]
        if len(q) > 1:
            w = 1 / q.k_boot ** 2
            mean = (w * q.k).sum() / w.sum()
            chi2 = (((q.k - mean) / q.k_boot) ** 2).sum()
            ax.text(0.02, 0.95 - 0.06 * reb, f"REB{reb}: chi2/dof = {chi2:.1f}/"
                    f"{len(q) - 1} for one common k", transform=ax.transAxes,
                    color=col, fontsize=8, va="top")
    ax.set_xticks(p.index)
    ax.set_xticklabels(p.group, fontsize=8)
    ax.set_ylabel("k = sigma_diff / sqrt(D dz) (dimensionless)")
    ax.set_title("k per sensor; dotted: per-REB pooled fit, dashed: sqrt(2kT/eV)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, None)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_k(pdf, tracks, pooled):
    ok = tracks[tracks.status == "ok"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
    ax = axes[0]
    vv = np.linspace(5, 60, 200)
    ax.plot(vv, theory_k(vv, 173.15), "k--", label="sqrt(2kT/eV), T = 173.15 K")
    jit = np.random.default_rng(1).normal(0, 0.4, len(ok))
    ax.plot(ok.BSSVBS + jit, ok.k, ".", color="0.6", label="single tracks")
    p = pooled[pooled.group.str.startswith("R21")]
    ax.errorbar(p.volts, p.k, p.k_err, fmt="o", color="C1", label="per sensor, pooled")
    p = pooled[pooled.group.str.startswith("REB")]
    ax.errorbar(p.volts, p.k, p.k_err, fmt="s", color="k", ms=7, label="per REB, pooled")
    ax.set_xlabel("back-bias voltage BSSVBS (V)")
    ax.set_ylabel("k = sigma_diff / sqrt(D dz) (dimensionless)")
    ax.set_ylim(0, max(0.15, np.nanpercentile(ok.k, 95) if len(ok) else 0.15))
    ax.legend(fontsize=8)
    ax.set_title("Diffusion coefficient against back bias")
    ax = axes[1]
    for reb, g in ok.groupby("REB"):
        ax.hist(np.sqrt(np.clip(g.sig0sq_um2, 0, None)), bins=np.linspace(0, 6, 31),
                histtype="step", lw=1.5, label=f"REB{reb}, n = {len(g)}")
    ax.axvline(SIGMA0_UM, color="k", ls="--", label="10 um / sqrt(12) = 2.89 um")
    ax.set_xlabel("per-track intercept sigma0 (um)")
    ax.set_ylabel("tracks per bin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_gallery(pdf, stamps, tracks, secs, per_page=6):
    order = tracks.sort_values("LENGTH", ascending=False)
    for start in range(0, len(order), per_page):
        chunk = order.iloc[start:start + per_page]
        fig, axes = plt.subplots(per_page, 2, figsize=(8.5, 11),
                                 gridspec_kw=dict(width_ratios=[1.3, 1]))
        for (ax_im, ax_w), (_, r) in zip(axes, chunk.iterrows()):
            img = stamps[(r.exposure, r.extname)]
            vmax = np.nanpercentile(img, 99.5)
            im = ax_im.imshow(img, origin="lower", cmap="magma", vmin=-20, vmax=vmax,
                              interpolation="nearest")
            fig.colorbar(im, ax=ax_im, fraction=0.04, label="electrons")
            ax_im.set_title(f"{r.exposure} {r.extname}  {r.BSSVBS:.1f} V  "
                            f"L = {r.LENGTH:.0f} px  [{r.status}]", fontsize=7)
            ax_im.tick_params(labelsize=6)
            g = secs[(secs.extname == r.extname) & (secs.exposure == r.exposure)]
            if not g.empty:
                x = g.dz_um if "dz_um" in g and g.dz_um.notna().all() else \
                    g.frac * THICKNESS_UM
                ax_w.errorbar(x[g.good], g.sigma_um[g.good], g.sigma_err_um[g.good],
                              fmt="o", ms=3)
                ax_w.plot(x[~g.good], g.sigma_um[~g.good], "x", color="0.5")
                if r.status in ("ok", "ambiguous"):
                    zz = np.linspace(0, 100, 100)
                    ax_w.plot(zz, np.sqrt(np.clip(r.sig0sq_um2 + r.k2D_um * zz, 0, None)),
                              "r-", lw=1)
                    ax_w.text(0.03, 0.85, f"k = {r.k:.3f}", transform=ax_w.transAxes,
                              fontsize=7)
            ax_w.set_ylim(0, 12)
            ax_w.set_xlim(0, 100)
            ax_w.tick_params(labelsize=6)
            ax_w.set_xlabel("dz (um)", fontsize=7)
            ax_w.set_ylabel("sigma (um)", fontsize=7)
        for ax_im, ax_w in axes[len(chunk):]:
            ax_im.axis("off")
            ax_w.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--find-run", nargs="+", required=True,
                   help="run directory names of find_cosmics.py in this study's output")
    p.add_argument("--label", default="diffusion")
    p.add_argument("--low-sigma", type=float, default=3.0)
    p.add_argument("--sec-len", type=float, default=3.0, help="pixels")
    p.add_argument("--max-sections", type=int, default=10)
    p.add_argument("--max-chi2", type=float, default=5.0, help="chi2/dof per section")
    p.add_argument("--max-halo", type=float, default=0.05,
                   help="fraction of track charge beyond 2.5 px from the line")
    p.add_argument("--min-length", type=float, default=15.0,
                   help="pixels; shorter tracks give too few sections")
    p.add_argument("--max-gallery", type=int, default=120)
    args = p.parse_args()

    from rubinwork.common import study_run
    from rubinwork.common.runs import data_root
    base = data_root() / "studies" / STUDY
    run = study_run(STUDY, args.label)

    tracks, secs, srcs, stamps = [], [], [], {}
    nvis = 0
    for fr in args.find_run:
        for f in sorted(glob.glob(str(base / fr / "visits" / "cosmics_*.fits"))):
            nvis += 1
            srcs.append(pd.read_parquet(f.replace("cosmics_", "tracks_")
                                        .replace(".fits", ".parquet")))
            with fits.open(f) as hl:
                args._exposure = hl[0].header["EXPOSURE"]
                for hdu in hl[1:]:
                    if hdu.header["LENGTH"] < args.min_length:
                        continue
                    t, s = analyze_track(hdu, args)
                    tracks.append(t)
                    secs.append(s)
                    stamps[(args._exposure, hdu.name)] = hdu.data.astype(float)
    tracks = pd.DataFrame(tracks)
    secs = pd.concat(secs, ignore_index=True)
    all_src = pd.concat(srcs, ignore_index=True)
    pooled = pooled_fits(tracks, secs)
    tracks.to_parquet(run.path / "tracks.parquet")
    secs.to_parquet(run.path / "sections.parquet")
    pooled.to_csv(run.path / "pooled.csv", index=False)
    print(tracks.status.value_counts().to_string())
    print(pooled.round(4).to_string())

    with PdfPages(run.path / "cosmic_diffusion.pdf") as pdf:
        page_summary(pdf, tracks, pooled, nvis, args.find_run)
        page_dedx(pdf, all_src)
        page_pooled(pdf, secs, pooled)
        page_per_ccd(pdf, secs, pooled)
        page_k(pdf, tracks, pooled)
        gal = tracks.sort_values("LENGTH", ascending=False).head(args.max_gallery)
        page_gallery(pdf, stamps, gal, secs)
    print(f"wrote {run.path / 'cosmic_diffusion.pdf'}")


if __name__ == "__main__":
    main()
