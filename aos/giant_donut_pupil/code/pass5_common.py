"""Shared pieces of pupil pass 5: the side list, stamp loading, fit c re-rendering and the
model-guided spider tracer.

Sides: the 180 pass 4 band sides (run 20261021_pupil4) plus the 8 pass 3 sample sides not
in a band (run 20261020_pupil3), 188 in all, each once. ``in_sample`` marks the 49 pass 3
sample sides, ``in_bands`` the 180.

Guided tracer (`guided_shadows`). Pass 4 traced the shadows blind: the eight deepest dips
at the innermost radius (0.68 r_out) define the shadows, and later radii match to them
within 10 deg. Where 0.68 r_out cuts the vignetted inner edge, that first radius picks
dips in the dark part of the ring, and the real shadows then never match. Here:

1. the model shadows come from the fit c model with and without the vanes: the deficit
   ``1 - with / without`` (where the no-vane model is above 0.3 of its median) per narrow
   annulus, its eight peaks at 0.80 r_out, then walked inward and outward radius by
   radius within ``WALK_DEG``;
2. at each radius the model and data shadows are the dip of each image's flattened
   azimuthal profile (as `spider_trajectory`, bins whose running median is under 0.3 of
   the profile median dropped) within ``MODEL_WIN_DEG`` of the guide (model) and
   ``DATA_WIN_DEG`` of the model shadow (data), and at least ``MIN_DEPTH`` deep;
3. a shadow with fewer than ``MIN_POINTS`` radii, in the model or the data, is missing.

Both images are traced with the same estimator about the same centre, as pass 4.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (blitz path first)
import fit_donuts as fd  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import obscuration_v402g as og  # noqa: E402
import pupil_pass3 as p3  # noqa: E402
import spider_pass4 as sp4  # noqa: E402

from rubinwork.giant_donuts import spider_trajectory as stj  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
RUN_LABEL = 'pupil5'
RUN_DATE = '20261022'
PASS3_RUN = p3.STUDY_DIR / 'output' / '20261020_pupil3'
PASS4_RUN = p3.STUDY_DIR / 'output' / '20261021_pupil4'
RADII = stj.TRACE_RADII_NORM
GUIDE_RADIUS = 0.80
WALK_DEG = 3.0
MODEL_WIN_DEG = 2.0
DATA_WIN_DEG = 4.0
MIN_DEPTH = 0.1
MIN_POINTS = 4
DARK_FRAC = 0.3
N_SHADOW = stj.N_SPIDER_SHADOWS


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- sides

def sides():
    """One row per distinct side: metadata, fit c parameters and where its stamp is."""
    f4 = pd.read_parquet(PASS4_RUN / 'fits.parquet')
    c4 = f4[f4.fit == 'c'].copy()
    c4['source'] = 'pass4'
    c4['in_bands'] = True
    f3 = pd.read_parquet(PASS3_RUN / 'fits.parquet')
    c3 = f3[f3.fit == 'c'].copy()
    s3 = pd.read_parquet(PASS3_RUN / 'sample.parquet')
    c3 = c3.merge(s3[['sample_id', 'side', 'r_out_pix']].rename(columns={'r_out_pix': 'r_s3'}),
                  on=['sample_id', 'side'], how='left')
    have = set(zip(c4.pass3_id, c4.side))
    c3 = c3[[(a, b) not in have for a, b in zip(c3.sample_id, c3.side)]].copy()
    c3['pass3_id'] = c3.sample_id
    c3['band'] = ''
    c3['source'] = 'pass3'
    c3['in_bands'] = False
    keep = ['sample_id', 'pass3_id', 'band', 'side', 'visit', 'detector', 'sensor_class', 'mag',
            'field_r_deg', 'field_az_ocs_deg', 'field_x_ccs_deg', 'field_y_ccs_deg', 'rtp_deg',
            'r_out_pix', 'noise_e', 'flux_e', 'dx_pix', 'dy_pix', 'chi2_dof', 'fit_ok', 'error',
            'source', 'in_bands'] + [f'z{j}_ccs_um' for j in p3.FITS['c']]
    t = pd.concat([c4[keep], c3[keep]], ignore_index=True)
    t['error'] = t.error.fillna('')
    t['in_sample'] = t.pass3_id.fillna('') != ''
    t['pass3_id'] = t.pass3_id.fillna('')
    t['label'] = np.where(t.pass3_id != '', t.pass3_id, t.sample_id.str[:9])
    t['key'] = t.sample_id + ':' + t.side
    return t.reset_index(drop=True)


def load_side(row):
    """Stamp, mask, centre, noise, bin flags and the stored fit c model of one side."""
    run = PASS4_RUN if row.source == 'pass4' else PASS3_RUN
    d = np.load(run / 'images' / f'{row.sample_id}_{row.side}.npz')
    out = dict(img=d['img'].astype(float), masked=d['masked'].astype(bool), cx=float(d['cx']),
               cy=float(d['cy']), model_c=d['model_c'].astype(float))
    if 'spider' in d:
        out['flags'] = (d['spider'], d['dark'], d['m_in'], d['m_out'])
        out['noise'] = float(d['noise'])
        out['zk_ta'] = d['zk_ta']
    else:
        b = pd.read_parquet(PASS3_RUN / 'edges_bins.parquet')
        b = b[(b.sample_id == row.sample_id) & (b.side == row.side)].sort_values('bin')
        out['flags'] = tuple(b[c].to_numpy() for c in
                             ('spider', 'vignetted', 'masked_inner', 'masked_outer'))
        out['noise'] = float(row.noise_e)
        out['zk_ta'] = None
    return out


def edges(image, valid, cx, cy, row, noise, flags):
    return p3.edges_of(image, valid, cx, cy, row.r_out_pix, noise, flags)


# ---------------------------------------------------------------- fit c re-rendering

class Renderer:
    """Re-render fit c of one side at any spider angle, blur or with the vanes off."""

    def __init__(self, row, zk_ta=None):
        import yaml
        import danish
        offsets = fd.offsets_m(row.side, row.sensor_class)
        params = yaml.safe_load(og.mask_path(p3.MASK_RUN, offsets).read_text())
        r_outer, r_inner = ob._annulus(params)
        self.thx, self.thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
        if zk_ta is None:
            zk_ta, _ = p3.reference_zk(row.field_x_ccs_deg, row.field_y_ccs_deg, offsets, row.side,
                                       r_inner / r_outer)
        self.zk_ta = zk_ta
        self.rtp = float(row.rtp_deg)
        self.row = row
        self.fac = {True: p3._factory(params, self.rtp), False: p3._factory(params, self.rtp,
                                                                              spider=False)}
        self.danish = danish

    def model(self, npix, spider=True):
        return self.danish.SingleDonutModel(self.fac[spider], z_ref=self.zk_ta,
                                            z_terms=p3.FITS['c'], thx=self.thx, thy=self.thy,
                                            npix=npix, bkg_order=-1)

    def render(self, npix, delta_deg=0.0, fwhm=p3.FWHM_ARCSEC, spider=True, x=None):
        """Fit c image; `x` (flux, dx, dy in arcsec, deviations in m) overrides the table."""
        m = self.model(npix, spider)
        self.fac[True].spider_angle = self.rtp + delta_deg
        r = self.row
        if x is None:
            z = np.array([r[f'z{j}_ccs_um'] for j in p3.FITS['c']]) * 1e-6
            x = (r.flux_e, r.dx_pix * m.sky_scale, r.dy_pix * m.sky_scale, z)
        img = np.asarray(m.model(x[0], x[1], x[2], fwhm, x[3]), float)
        self.fac[True].spider_angle = self.rtp
        return img


# ---------------------------------------------------------------- guided tracer

def flat_profile(image, center, r_mid):
    """Azimuthal profile in a narrow annulus over a 10 deg running median; dark bins NaN."""
    ang, flux = stj._annulus_azimuthal(image, center, r_mid, stj.TRACE_ANNULUS_PIX,
                                       stj.TRACE_BIN_DEG)
    n = len(flux)
    half = int(round(10.0 / stj.TRACE_BIN_DEG))
    pad = np.concatenate([flux[-half:], flux, flux[:half]])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        base = np.nanmedian(win, axis=1)[:n]
        lvl = np.nanmedian(flux)
        flat = np.where(base > DARK_FRAC * lvl, flux / base, np.nan)
    return ang, flat


def dip_near(ang, flat, az0, half_deg, min_depth=MIN_DEPTH):
    """Deficit-weighted centroid (deg) of the deepest dip within `half_deg` of `az0`."""
    if not np.isfinite(az0):
        return np.nan, np.nan
    bin_deg = ang[1] - ang[0]
    n = len(ang)
    w = np.abs(wrap(ang - az0)) <= half_deg
    f = np.where(w & np.isfinite(flat), flat, np.inf)
    i = int(np.argmin(f))
    if not np.isfinite(f[i]) or 1 - f[i] < min_depth:
        return np.nan, np.nan
    wh = int(round(stj.DIP_WINDOW_DEG / bin_deg))
    idx = np.arange(i - wh, i + wh + 1) % n
    deficit = np.clip(1 - flat[idx], 0, None)
    deficit[~np.isfinite(deficit)] = 0
    off = np.arange(-wh, wh + 1) * bin_deg
    az = ang[i] + (deficit * off).sum() / deficit.sum() if deficit.sum() > 0 else ang[i]
    return float(az % 360), float(1 - f[i])


def guide_azimuths(with_v, without, center, r_edge):
    """(8, n_radii) shadow azimuths (deg, pixel frame) from the vane deficit of the model."""
    lit = without > DARK_FRAC * np.median(without[without > 0])
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(lit, with_v / without, np.nan)
    j0 = int(np.argmin(np.abs(np.array(RADII) - GUIDE_RADIUS)))
    ang, prof = stj._annulus_azimuthal(ratio, center, RADII[j0] * r_edge, stj.TRACE_ANNULUS_PIX,
                                       stj.TRACE_BIN_DEG)
    found, _ = stj._dip_azimuths(ang, prof)
    g = np.full((N_SHADOW, len(RADII)), np.nan)
    g[:len(found), j0] = found
    for step in (+1, -1):
        j = j0 + step
        while 0 <= j < len(RADII):
            ang, prof = stj._annulus_azimuthal(ratio, center, RADII[j] * r_edge,
                                               stj.TRACE_ANNULUS_PIX, stj.TRACE_BIN_DEG)
            for k in range(N_SHADOW):
                prev = g[k, j - step]
                if not np.isfinite(prev):
                    prev = g[k, j0]
                g[k, j] = dip_near(ang, prof, prev, WALK_DEG, min_depth=0.02)[0]
            j += step
    return g


def trace_near(image, center, r_edge, ref, half_deg):
    """Per shadow and radius, the dip of `image` within `half_deg` of `ref` (deg)."""
    az = np.full(ref.shape, np.nan)
    for j, rn in enumerate(RADII):
        if not np.isfinite(ref[:, j]).any():
            continue
        ang, flat = flat_profile(image, center, rn * r_edge)
        for k in range(ref.shape[0]):
            az[k, j] = dip_near(ang, flat, ref[k, j], half_deg)[0]
    return az


def centres_of(az, center, r_edge):
    rp = np.asarray(RADII) * r_edge
    return np.stack([center[0] + rp[None, :] * np.cos(np.deg2rad(az)),
                     center[1] + rp[None, :] * np.sin(np.deg2rad(az))], axis=-1)


def chords(az, image, center, r_edge, edges_fit, rtp_deg, profiles=True):
    """Shadow rows from traced azimuths: chord, r_half azimuth (OCS), depth and width."""
    cen = centres_of(az, center, r_edge)
    for k in range(len(cen)):
        if np.isfinite(cen[k]).all(axis=1).sum() < MIN_POINTS:
            cen[k] = np.nan
    tr = sp4._robust_trajectories(cen, center)
    az_bin = (np.arange(len(edges_fit[0])) + 0.5) * 360 / len(edges_fit[0])
    out = []
    for t, c in zip(tr, cen):
        row = dict(n_points=t['n_points'], rms_pix=t['rms_pix'], b_pix=t['impact_pix'],
                   phi0_ocs_deg=np.nan, az_half_ocs_deg=np.nan, r_half_pix=np.nan, depth=np.nan,
                   fwhm_pix=np.nan, profile=None, centres=c, traj=t)
        if np.isfinite(t['impact_pix']):
            k = int(np.argmin(np.abs(wrap(az_bin - t['azimuth_deg']))))
            r_half = 0.5 * (edges_fit[0][k] + edges_fit[1][k])
            az_h, d = sp4._az_at(t, center, r_half)
            row.update(phi0_ocs_deg=float((np.degrees(np.arctan2(d[1], d[0])) + rtp_deg) % 360),
                       az_half_ocs_deg=(az_h + rtp_deg) % 360 if np.isfinite(az_h) else np.nan,
                       r_half_pix=r_half)
            if profiles:
                off, flux, _, _ = stj.trajectory_profile(image, t, center, r_edge)
                depth, fwhm, _ = stj.vane_depth(off, flux)
                row.update(depth=depth, fwhm_pix=fwhm,
                           profile=(off.astype(np.float32), flux.astype(np.float32)))
        out.append(row)
    return out


def guided_shadows(data, model, without, center, r_edge, edges_fit, rtp_deg, profiles=True):
    """Model and data shadow rows (matched by index) and the missing reason per shadow."""
    guide = guide_azimuths(model, without, center, r_edge)
    az_m = trace_near(model, center, r_edge, guide, MODEL_WIN_DEG)
    az_d = trace_near(data, center, r_edge, az_m, DATA_WIN_DEG)
    sm = chords(az_m, model, center, r_edge, edges_fit, rtp_deg, profiles)
    sd = chords(az_d, data, center, r_edge, edges_fit, rtp_deg, profiles)
    reasons = []
    for k in range(N_SHADOW):
        n_g = int(np.isfinite(guide[k]).sum())
        if n_g < MIN_POINTS:
            reasons.append('model vane shadow not in the lit ring' if n_g == 0 else
                           f'model vane shadow lit at {n_g} of 7 radii')
        elif not np.isfinite(sm[k]['az_half_ocs_deg']):
            reasons.append('model dip under min depth')
        elif not np.isfinite(sd[k]['az_half_ocs_deg']):
            n_d = int(np.isfinite(az_d[k]).sum())
            reasons.append(f'data dip within {DATA_WIN_DEG:.0f} deg at {n_d} of 7 radii')
        elif sd[k]['rms_pix'] > sp4.MAX_RMS_PIX:
            reasons.append(f'data chord rms over {sp4.MAX_RMS_PIX} pixels')
        else:
            reasons.append('')
    return sd, sm, reasons, dict(guide=guide, az_m=az_m, az_d=az_d)


def m1_fit(y_deg, phi_deg):
    """Huber fit y = c + a cos(phi) + b sin(phi); returns c, amplitude, phase (deg)."""
    import statsmodels.api as sm
    p = np.deg2rad(np.asarray(phi_deg))
    y = np.asarray(y_deg)
    ok = np.isfinite(y) & np.isfinite(p)
    X = np.column_stack([np.ones(ok.sum()), np.cos(p[ok]), np.sin(p[ok])])
    r = sm.RLM(y[ok], X, M=sm.robust.norms.HuberT()).fit()
    c, a, b = r.params
    return dict(common=float(c), amp=float(np.hypot(a, b)),
                phase_deg=float(np.degrees(np.arctan2(b, a)) % 360), n=int(ok.sum()),
                resid_nmad=float(np.median(np.abs(r.resid)) * 1.4826))
