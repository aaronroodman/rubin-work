"""Shape-only fit of the M1 inner-edge roll-off (A, w) and M3 outer-edge turn-up (B, u).

Reads the ``model_profiles`` run (per-side model rim profiles on the grids) and the data
stamps of `giant_donuts`.

Per side and per 2 deg bin, the data radial profile is compared with each grid model in
the model's 50-pixel window around the nominal edge, less 6 pixels at each end. Each bin
has a free radial shift (the model moved by delta, on a 0.1 pixel grid over ±6 pixels,
refined by a parabola; the data stay at their own radii) and a free scale. The loss is Huber on residuals in units of the per-point
sigma, threshold 2 sigma (the scipy ``least_squares`` 'huber' loss with f_scale = 2); the
scale comes from iteratively reweighted least squares. Sigma per point is the photon plus
read/sky noise of the data mean, plus the model's ray-sampling noise. Bins whose nominal
model runs out of traced rays inside the fitted range (`trim_window`) are dropped.

Bin sets per side, from the model run's labels and per-edge spider cut:

- ``m1_inner``: inner edge set by M1, fitted with the M1 grid;
- ``inner_other``: inner edge set by another element (control, M1 grid);
- ``m3_outer``: outer edge set by M3, fitted with the M3 grid;
- ``baffle_outer``: outer edge set by M1Baffle1 (control, M3 grid).

The blur is free per side: for each grid point, each side takes the ``FWHM_GRID`` value
that minimizes its m1_inner plus m3_outer cost (joint over the two grids). Totals are
summed over sides; the best grid point is refined by a spline over (A, ln w) and
(B, ln u). Errors are a bootstrap over sides.

Writes ``costs/<key>.npz`` per side, then ``fit_rolloff.pdf`` and ``results.json``.

Usage, from the repo root::

    python aos/mirror_edge_rolloff/code/fit_rolloff.py --models 20261118_model_profiles \\
        --workers 60
    python aos/mirror_edge_rolloff/code/fit_rolloff.py --models 20261118_model_profiles \\
        --combine-only --out <run directory>
"""
import argparse
import json
import pathlib
import time

import numpy as np

from rubinwork.giant_donuts import edge_profiles as ep

STUDY = 'aos/mirror_edge_rolloff'
VARIANT = 'v402s_fitc'
SHIFTS_PIX = np.round(np.arange(-6.0, 6.0001, 0.1), 3)
HUBER_C = 2.0
IRLS_ITER = 4
MAX_NAN_FRAC = 0.2
PAD = 12  # window points dropped at each end: room for the ±6 pixel shifts
WINDOW_FIT_PIX = 19.0  # half-width of the fitted radial range, pixels
N_BOOT = 300
SETS = {'m1_inner': ('inner', 'M1', True), 'inner_other': ('inner', 'M1', False),
        'm3_outer': ('outer', 'M3', True), 'baffle_outer': ('outer', 'M1Baffle1', True)}
RIM_PEAK_PIX = 10.0
RIM_PLATEAU_PIX = (12.0, 25.0)


def _study_out():
    from rubinwork.common.utils import repo_root
    return pathlib.Path(repo_root()) / STUDY / 'output'


def bin_sets(m):
    """Bin masks per set, from a model npz."""
    out = {}
    for name, (edge, lab, match) in SETS.items():
        labels = m[f'label_{edge}']
        hit = (labels == lab) if match else ((labels != lab) & (labels != ''))
        out[name] = hit & ~m[f'cut_{edge}'] & np.isfinite(m[f'e_{"in" if edge == "inner" else "out"}'])
    return out


def _shift_index(nwin):
    """Model-window index and weight of each data point at each shift: (S, nwin - 2 PAD)."""
    u = PAD + np.arange(nwin - 2 * PAD)[None, :] - SHIFTS_PIX[:, None] / ep.DR_PIX
    i0 = np.clip(np.floor(u).astype(int), 0, nwin - 2)
    return i0, (u - i0).astype(np.float32)


def model_noise_frac(prof):
    """Ray-sampling noise of a model profile, as a fraction of its level (dimensionless).

    From the point-to-point scatter over the last 20 points (the plateau) of each bin.
    """
    p = prof[:, -20:]
    lvl = np.nanmedian(p, axis=1)
    d = np.diff(p, axis=1) / lvl[:, None] / np.sqrt(2)
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) if d.size else 0.0


def huber_costs(D, V, P, frac):
    """Huber cost, scale and shift per bin for one model.

    The model is shifted (linear interpolation of its 0.5 pixel samples) and compared
    with the data at the data's own radii, the window less `PAD` points at each end, so
    every shift uses the same points.

    Parameters
    ----------
    D, V : `numpy.ndarray`
        Data mean and variance, (B, nwin - 2 PAD) (electrons per pixel and its square).
    P : `numpy.ndarray`
        Model profile over the window, (B, nwin) (electrons per pixel).
    frac : `float`
        Model sampling noise, fraction of the model level (dimensionless).

    Returns
    -------
    cost, delta, scale : `numpy.ndarray`
        Per bin: the minimum Huber cost over shifts (dimensionless), its shift (pixels;
        positive means the data edge lies at larger radius than the model edge) and the
        scale (data over model, dimensionless).
    """
    i0, f = _shift_index(P.shape[1])
    M = P[:, i0] * (1 - f) + P[:, i0 + 1] * f  # (B, S, Nd)
    Q = (np.isfinite(D) & np.isfinite(V))[:, None, :] & np.isfinite(M)
    Dz = np.where(np.isfinite(D), D, 0.0)[:, None, :]
    Mm = np.where(Q, M, 0.0)
    Vt = np.where(Q, V[:, None, :], np.inf) + (frac * np.maximum(Mm, 0)) ** 2
    hw = np.ones(Mm.shape, np.float32)
    for _ in range(IRLS_ITER):
        w = Q * hw / Vt
        s = (w * Dz * Mm).sum(-1) / np.maximum((w * Mm * Mm).sum(-1), 1e-30)
        z = np.abs(Dz - s[..., None] * Mm) / np.sqrt(Vt)
        hw = np.where(z <= HUBER_C, 1.0, HUBER_C / np.maximum(z, 1e-30))
    rho = np.where(z <= HUBER_C, 0.5 * z * z, HUBER_C * z - 0.5 * HUBER_C ** 2)
    cost = (Q * rho).sum(-1)  # (B, S)
    j = np.argmin(cost, axis=1)
    jj = np.clip(j, 1, len(SHIFTS_PIX) - 2)
    b = np.arange(cost.shape[0])
    c0, c1, c2 = cost[b, jj - 1], cost[b, jj], cost[b, jj + 1]
    den = c0 - 2 * c1 + c2
    t = np.where(den > 0, 0.5 * (c0 - c2) / np.where(den > 0, den, 1), 0.0)
    t = np.clip(t, -1, 1)
    step = SHIFTS_PIX[1] - SHIFTS_PIX[0]
    delta = SHIFTS_PIX[jj] + t * step
    cmin = np.where(den > 0, c1 - 0.25 * (c0 - c2) * t, cost[b, j])
    return cmin.astype(np.float32), delta.astype(np.float32), s[b, j].astype(np.float32)


def trim_window(x, y, inward):
    """Blank a model window beyond where it runs out of traced rays.

    The dense trace keeps only `model_profiles.ZONES_M` stop radii, so on some edges the
    model falls off inside its window. Going from the edge into the flat top (``inward``
    is +1 for an inner edge, -1 for an outer one), points past the first that drops below
    0.6 of the 8-15 pixel median are set to NaN.
    """
    y = y.copy()
    for i in range(len(y)):
        d = inward * x[i]
        ref = np.nanmedian(np.where((d > 8) & (d < 15), y[i], np.nan))
        low = (d > 8) & (y[i] < 0.6 * ref)
        if low.any():
            y[i][d >= d[low].min()] = np.nan
    return y


def side_task(task):
    row, models_dir, out_dir = task
    from rubinwork.products import giant_donuts
    t0 = time.time()
    key = row['key']
    path = pathlib.Path(out_dir) / 'costs' / f'{key}.npz'
    if path.exists():
        return key, 'exists'
    m = dict(np.load(pathlib.Path(models_dir) / 'profiles' / f'{key}.npz'))
    img, msk, _ = giant_donuts.stamp(key, VARIANT)
    valid = ~msk.astype(bool)
    noise = float(row['noise_e'])
    prof, rad, *_rest, cnt = ep.profiles(img.astype(float), valid, float(row['cx_a']),
                                        float(row['cy_a']), float(row['r_out_pix']), noise)
    with np.errstate(divide='ignore', invalid='ignore'):
        var = (noise ** 2 + np.maximum(prof, 0)) / cnt
    var[cnt == 0] = np.nan
    nwin = m['prof_m1'].shape[-1]
    take = np.arange(nwin)
    rad_win = (np.arange(len(rad)) + 0.5) * ep.DR_PIX
    sets = bin_sets(m)
    res = dict(key=key, side=row['side'], fwhm_arcsec=m['fwhm_arcsec'])
    for edge, grid, start in (('inner', 'prof_m1', m['start_in']),
                              ('outer', 'prof_m3', m['start_out'])):
        use = np.zeros(ep.NBIN, bool)
        for name, (e, _, _) in SETS.items():
            if e == edge:
                use |= sets[name]
        bins = np.nonzero(use)[0]
        rm = rad_win[start[bins, None] + take[None, :]]
        di = start[bins, None] + PAD + take[None, :nwin - 2 * PAD]
        rows = np.arange(len(bins))[:, None]
        D, V = prof[bins][rows, di].astype(np.float32), var[bins][rows, di].astype(np.float32)
        P = m[grid][:, :, bins]  # (K, F, B, N)
        frac = model_noise_frac(P[0, list(m['fwhm_arcsec']).index(1.0)])
        ok = np.isfinite(D).mean(-1) >= 1 - MAX_NAN_FRAC
        # bins whose nominal model runs out of traced rays inside the fitted range
        xw = rm - (m['e_in'] if edge == 'inner' else m['e_out'])[bins, None]
        trimmed = trim_window(xw, P[0, list(m['fwhm_arcsec']).index(1.0)],
                              1 if edge == 'inner' else -1)
        lost = (np.isnan(trimmed) & np.isfinite(P[0, 0]) & (np.abs(xw) < WINDOW_FIT_PIX)).any(1)
        ok &= ~lost
        K, F = P.shape[:2]
        cost = np.full((K, F, len(bins)), np.nan, np.float32)
        delta, scale = cost.copy(), cost.copy()
        for k in range(K):
            for f in range(F):
                c, d, s = huber_costs(D[ok], V[ok], P[k, f, ok], frac)
                cost[k, f, ok], delta[k, f, ok], scale[k, f, ok] = c, d, s
        res.update({f'{edge}_bins': bins, f'{edge}_ok': ok, f'{edge}_cost': cost,
                    f'{edge}_delta': delta, f'{edge}_scale': scale,
                    f'{edge}_model_noise_frac': frac,
                    f'{edge}_data': D, f'{edge}_var': V,
                    f'{edge}_rad_data': rm[:, PAD:nwin - PAD].astype(np.float32),
                    f'{edge}_rad_model': rm.astype(np.float32)})
        for name, (e, _, _) in SETS.items():
            if e == edge:
                res[f'set_{name}'] = sets[name][bins]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **res)
    tmp.rename(path)
    print(json.dumps(dict(key=key, seconds=round(time.time() - t0),
                          n_inner=int(res['inner_ok'].sum()), n_outer=int(res['outer_ok'].sum()))),
          flush=True)
    return key, 'done'


def load_costs(out_dir, models_dir):
    """Per-side set costs: dict of arrays (sides, K, F) per set, plus side info."""
    files = sorted((pathlib.Path(out_dir) / 'costs').glob('*.npz'))
    g = np.load(next((pathlib.Path(models_dir) / 'profiles').glob('*.npz')))
    grids = {k: g[k] for k in ('A_nm', 'w_mm', 'B_nm', 'u_mm', 'fwhm_arcsec')}
    C = {name: [] for name in SETS}
    N = {name: [] for name in SETS}
    sides, keys = [], []
    for p in files:
        d = np.load(p, allow_pickle=False)
        keys.append(str(d['key']))
        sides.append(str(d['side']))
        for name, (edge, _, _) in SETS.items():
            sel = d[f'set_{name}'] & d[f'{edge}_ok']
            C[name].append(np.nansum(d[f'{edge}_cost'][:, :, sel], axis=-1))
            N[name].append(int(sel.sum()))
    return ({k: np.array(v) for k, v in C.items()}, {k: np.array(v) for k, v in N.items()},
            np.array(sides), np.array(keys), grids)


def best_blur(C):
    """Per side, per (k, j), the FWHM index minimizing m1_inner + m3_outer; (S, K, J)."""
    tot = C['m1_inner'][:, :, None, :] + C['m3_outer'][:, None, :, :]
    return np.argmin(tot, axis=-1), tot.min(axis=-1)


def refine(total, x, y, logy=True):
    """Spline-refined minimum of a cost over a grid (x, y): (x, y, cost)."""
    from scipy.interpolate import RectBivariateSpline
    if len(x) < 2 or len(y) < 2:  # test grids
        i, j = np.unravel_index(np.argmin(total), total.shape)
        return float(x[i]), float(y[j]), float(total[i, j])
    yy = np.log(y) if logy else y
    kx, ky = min(3, len(x) - 1), min(3, len(yy) - 1)
    sp = RectBivariateSpline(x, yy, total, kx=kx, ky=ky)
    fx = np.linspace(x.min(), x.max(), 301)
    fy = np.linspace(yy.min(), yy.max(), 301)
    z = sp(fx, fy)
    i, j = np.unravel_index(np.argmin(z), z.shape)
    return float(fx[i]), float(np.exp(fy[j]) if logy else fy[j]), float(z[i, j])


def fit_grid(C, idx, grids):
    """Best (A, w) and (B, u) for the sides in idx, joint blur per side."""
    sub = {k: v[idx] for k, v in C.items()}
    fb, tot = best_blur(sub)
    T = tot.sum(0)  # (K, J)
    A = grids['A_nm'].reshape(-1)
    nA, nw = len(np.unique(A)), len(np.unique(grids['w_mm']))
    nB, nu = len(np.unique(grids['B_nm'])), len(np.unique(grids['u_mm']))
    k, j = np.unravel_index(np.argmin(T), T.shape)
    m1 = T[:, j].reshape(nA, nw)
    m3 = T[k, :].reshape(nB, nu)
    a, w, _ = refine(m1, np.unique(grids['A_nm']), np.unique(grids['w_mm']))
    b, u, _ = refine(m3, np.unique(grids['B_nm']).astype(float), np.unique(grids['u_mm']).astype(float))
    return dict(A_nm=a, w_mm=w, B_nm=b, u_mm=u, k=int(k), j=int(j), m1_surface=m1, m3_surface=m3,
                blur_index=fb[:, k, j])


def control_curve(C, name, fb_index):
    """A set's cost summed over sides against its grid, each side at blur index fb_index."""
    s = np.arange(C[name].shape[0])
    return C[name][s[:, None], np.arange(C[name].shape[1])[None, :], fb_index[:, None]].sum(0)


def combine(out_dir, models_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    C, N, sides, keys, grids = load_costs(out_dir, models_dir)
    nA, nw = len(np.unique(grids['A_nm'])), len(np.unique(grids['w_mm']))
    nB, nu = len(np.unique(grids['B_nm'])), len(np.unique(grids['u_mm']))
    Au, wu = np.unique(grids['A_nm']), np.unique(grids['w_mm'])
    Bu, uu = np.unique(grids['B_nm']).astype(float), np.unique(grids['u_mm']).astype(float)
    groups = {'all': np.arange(len(keys)), 'extra': np.nonzero(sides == 'extra')[0],
              'intra': np.nonzero(sides == 'intra')[0]}
    groups = {g: idx for g, idx in groups.items() if len(idx)}
    fits = {g: fit_grid(C, idx, grids) for g, idx in groups.items()}
    rng = np.random.default_rng(1)
    boot = {g: [] for g in groups}
    for g, idx in groups.items():
        for _ in range(N_BOOT):
            r = fit_grid(C, rng.choice(idx, len(idx)), grids)
            boot[g].append([r['A_nm'], r['w_mm'], r['B_nm'], r['u_mm']])
        boot[g] = np.array(boot[g])
    names = ('A_nm', 'w_mm', 'B_nm', 'u_mm')
    results = {g: {n: dict(value=fits[g][n], boot_median=float(np.median(boot[g][:, i])),
                           boot_p16=float(np.percentile(boot[g][:, i], 16)),
                           boot_p84=float(np.percentile(boot[g][:, i], 84)))
                   for i, n in enumerate(names)} | {'n_sides': int(len(groups[g]))}
               for g in groups}
    best = fits['all']
    fb = best['blur_index']
    results['blur_fwhm_arcsec_median'] = float(np.median(grids['fwhm_arcsec'][fb]))
    results['n_bins'] = {k: int(v.sum()) for k, v in N.items()}
    pathlib.Path(out_dir, 'results.json').write_text(json.dumps(results, indent=1))

    with PdfPages(pathlib.Path(out_dir) / 'fit_rolloff.pdf') as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        lines = ['M1 inner roll-off dz = -A exp(-(r - 2.558 m)/w); M3 outer turn-up dz = +B exp(-(R3 - r)/u)',
                 'Shape-only fit: free shift and scale per 2 deg bin, Huber loss (2 sigma), blur per side.',
                 'Values: spline-refined grid minimum; [16, 84] percentiles of a bootstrap over sides.',
                 'Mirror-lab map (SUMMARY.md): A = 76.4 +- 8.9 nm, w = 27.1 +- 2.8 mm; M3 about +80 nm over 50 mm.', '']
        for g in groups:
            lines.append(f'{g} ({results[g]["n_sides"]} sides):')
            for n, unit in zip(names, ('nm', 'mm', 'nm', 'mm')):
                r = results[g][n]
                lines.append(f'   {n[0]} = {r["value"]:.1f} {unit}  [{r["boot_p16"]:.1f}, {r["boot_p84"]:.1f}]')
        lines += ['', f'bins used (sum over sides): ' + ', '.join(f'{k} {v}' for k, v in results['n_bins'].items()),
                  f'median per-side blur FWHM at the best fit: {results["blur_fwhm_arcsec_median"]:.2f} arcsec']
        fig.text(0.06, 0.95, '\n'.join(lines), va='top', family='monospace', fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)

        # cost surfaces
        fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
        for row, g in zip(axes, groups):
            for ax, surf, x, y, lab, bx, by in (
                    (row[0], fits[g]['m1_surface'], Au, wu, ('A (nm)', 'w (mm)'), 'A_nm', 'w_mm'),
                    (row[1], fits[g]['m3_surface'], Bu, uu, ('B (nm)', 'u (mm)'), 'B_nm', 'u_mm')):
                if min(surf.shape) < 2:
                    continue
                z = surf - surf.min()
                cs = ax.contourf(x, y, np.log10(z.T + 1), 20, cmap='viridis_r')
                fig.colorbar(cs, ax=ax, label='log10(cost - min + 1) (dimensionless)')
                ax.plot(fits[g][bx], fits[g][by], 'r+', ms=12)
                if bx == 'A_nm':
                    ax.plot(76.4, 27.1, 'wx', ms=10)
                ax.set_yscale('log')
                ax.set_xlabel(lab[0])
                ax.set_ylabel(lab[1])
                ax.set_title(f'{g}: {"M1 inner" if bx == "A_nm" else "M3 outer"} Huber cost', fontsize=9)
        fig.suptitle('Cost surfaces (+ best; x mirror-lab map)', fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # controls: cost vs grid with the joint best blur
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        for ax, (main, ctrl, gk, lab) in zip(axes, (('m1_inner', 'inner_other', 'w_mm', 'A (nm)'),
                                                     ('m3_outer', 'baffle_outer', 'u_mm', 'B (nm)'))):
            amp = grids['A_nm'] if gk == 'w_mm' else grids['B_nm']
            wid = grids[gk]
            wb = wid[best['k'] if gk == 'w_mm' else best['j']]
            for name, ls in ((main, '-'), (ctrl, '--')):
                cc = control_curve(C, name, fb)
                sel = wid == wb
                y = cc[sel] - cc[sel].min()
                ax.plot(amp[sel], y, ls, marker='o', label=f'{name} ({results["n_bins"][name]} bins), width {wb:g} mm')
            ax.set_xlabel(lab)
            ax.set_ylabel('Huber cost - min (dimensionless)')
            ax.legend(fontsize=8)
        fig.suptitle('Set costs against amplitude, at the best width and each side\'s best blur', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        stack_pages(pdf, out_dir, models_dir, keys, sides, best, fb, grids)
    return results


def _rim(r, p):
    pk = np.nanmax(np.where((r >= 0) & (r <= RIM_PEAK_PIX), p, np.nan))
    pl = np.nanmedian(np.where((r >= RIM_PLATEAU_PIX[0]) & (r <= RIM_PLATEAU_PIX[1]), p, np.nan))
    return pk / pl


def stack_pages(pdf, out_dir, models_dir, keys, sides, best, fb, grids):
    """Stacked data against model profiles, edge-aligned, per side and set."""
    import matplotlib.pyplot as plt
    k0 = int(np.nonzero((grids['A_nm'] == 0) & (grids['w_mm'] == grids['w_mm'][best['k']]))[0][0])
    j0 = int(np.nonzero((grids['B_nm'] == 0) & (grids['u_mm'] == grids['u_mm'][best['j']]))[0][0])
    acc = {}
    for s_i, key in enumerate(keys):
        d = np.load(pathlib.Path(out_dir) / 'costs' / f'{key}.npz')
        m = np.load(pathlib.Path(models_dir) / 'profiles' / f'{key}.npz')
        f = fb[s_i]
        for name, (edge, _, _) in SETS.items():
            sel = d[f'set_{name}'] & d[f'{edge}_ok']
            if not sel.any():
                continue
            kb, kn = (best['k'], k0) if edge == 'inner' else (best['j'], j0)
            grid = 'prof_m1' if edge == 'inner' else 'prof_m3'
            bins = d[f'{edge}_bins'][sel]
            e_nom = (m['e_in'] if edge == 'inner' else m['e_out'])[bins]
            sign = 1.0 if edge == 'inner' else -1.0  # bright side positive
            rm = d[f'{edge}_rad_model'][sel]
            rd = d[f'{edge}_rad_data'][sel]
            xm = sign * (rm - e_nom[:, None])
            for tag, kk in (('best', kb), ('nominal', kn)):
                P = m[grid][kk, f, bins]
                dl = d[f'{edge}_delta'][kk, f, sel]
                sc = d[f'{edge}_scale'][kk, f, sel]
                xd = sign * (rd - dl[:, None] - e_nom[:, None])
                lvl = np.nanmedian(np.where((xm >= RIM_PLATEAU_PIX[0]) & (xm <= RIM_PLATEAU_PIX[1]),
                                            P, np.nan), axis=1)
                a = acc.setdefault((name, sides[s_i], tag), dict(x_data=[], data=[], x_model=[],
                                                                 model=[]))
                a['x_data'].append(xd)
                a['data'].append(d[f'{edge}_data'][sel] / sc[:, None] / lvl[:, None])
                a['x_model'].append(xm)
                a['model'].append(P / lvl[:, None])
    grid_x = np.arange(-25, 25.01, 0.5)
    for name in SETS:
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        for ax, side in zip(axes, ('intra', 'extra')):
            n_bins = 0
            for tag, col in (('nominal', 'tab:blue'), ('best', 'tab:red')):
                a = acc.get((name, side, tag))
                if not a:
                    continue
                for what, ls in (('data', '.'), ('model', '-')):
                    x = np.concatenate(a[f'x_{what}'])
                    y = np.concatenate(a[what])
                    kb = np.digitize(x.ravel(), grid_x)
                    yy = y.ravel()
                    good = np.isfinite(yy)
                    med = np.array([np.median(yy[good & (kb == i)]) if (good & (kb == i)).any()
                                    else np.nan for i in range(1, len(grid_x))])
                    xc = 0.5 * (grid_x[1:] + grid_x[:-1])
                    if what == 'data':
                        ax.plot(xc, med, ls, color=col, ms=3,
                                label=f'data, aligned to {tag} (rim {_rim(xc, med):.2f})')
                    else:
                        ax.plot(xc, med, ls, color=col, lw=1,
                                label=f'model {tag} (rim {_rim(xc, med):.2f})')
                n_bins = sum(len(v) for v in a['x_data'])
            ax.set_title(f'{name}, {side}-focal ({n_bins} bins)', fontsize=9)
            ax.set_xlabel('distance from nominal edge, bright side positive (pixels)')
            ax.set_ylabel('profile / model plateau (dimensionless)')
            ax.legend(fontsize=7)
        fig.suptitle('Median edge-aligned profile; rim = peak within 10 px over the 12-25 px plateau '
                     '(dimensionless)', fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    import multiprocessing as mp
    import pandas as pd
    from rubinwork.common import study_run
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--models', required=True, help='model_profiles run directory name')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', default=None, help='output directory; a new study run by default')
    p.add_argument('--combine-only', action='store_true')
    p.add_argument('--limit', type=int, default=None, help='first N sides only')
    a = p.parse_args()
    models_dir = pathlib.Path(a.models)
    if not models_dir.is_absolute():
        models_dir = _study_out() / a.models
    out = pathlib.Path(a.out) if a.out else study_run(STUDY, 'fit_rolloff').path
    out.mkdir(parents=True, exist_ok=True)
    if not a.combine_only:
        S = pd.read_csv(models_dir / 'sides.csv')
        S = S[[(models_dir / 'profiles' / f'{k}.npz').exists() for k in S.key]]
        if a.limit:
            S = S.head(a.limit)
        tasks = [(r, str(models_dir), str(out)) for r in S.to_dict('records')]
        print(f'{len(tasks)} sides with model profiles', flush=True)
        if a.workers > 1:
            with mp.get_context('spawn').Pool(min(a.workers, len(tasks))) as pool:
                for _ in pool.imap_unordered(side_task, tasks):
                    pass
        else:
            for t in tasks:
                side_task(t)
    res = combine(out, models_dir)
    print(json.dumps({g: {k: round(v['value'], 2) for k, v in res[g].items() if isinstance(v, dict)}
                      for g in ('all', 'extra', 'intra') if g in res}), flush=True)
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
