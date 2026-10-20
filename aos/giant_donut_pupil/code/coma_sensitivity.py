"""Inner-minus-outer edge-centre shift per µm of coma, Batoid v4.02 at the giant configuration (step 3a).

For each chosen donut side the side's Batoid configuration from
`batoid_images.py` (rtp, offsets, camera piston) is traced three times: as is,
with a constant +1 µm Noll Z7 phase screen at the stop and with +1 µm Z8. Each
is rendered with the step 2 kernel and its edges are measured with
`pupil_edges.edges` about one fixed centre. A circle (centre and radius) is
fitted to the inner and to the outer edge; the quantity is the inner centre
minus the outer centre, pixels on the stamp (CCS column, row).

The screen is a `batoid.OPDScreen` before M1 at the stop plane, annular Zernike
on ``R_outer = pupilSize / 2``, ``R_inner = 0.61 R_outer``, signed so that
`batoid.zernike` of the screened optic gains +1 µm of the Noll term; the run
records that check.

`invert` converts measured edge offsets (data minus Batoid, per bin) into Z7 and
Z8 with the sensitivity matrix.

Run, from ``code/``::

    python coma_sensitivity.py --pupil-run ../output/20261018_pupil \\
        --run ../output/20261019_pupil2
"""
import argparse
import json
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import batoid_images as bi  # noqa: E402  (puts the blitz checkout first on sys.path)
import fit_donuts as fd  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import pupil_edges as pe  # noqa: E402

SIDES = ('G0005_extra', 'G0005_intra', 'G0515_extra', 'G0515_intra', 'G0329_extra',
         'G0329_intra', 'G0618_extra', 'G0618_intra')
RAYS_PER_PIX = 30
COMA_UM = 1.0
EPS = 0.61

__all__ = ['with_screen', 'circle_centre', 'side_sensitivity', 'invert']


def with_screen(optic, noll, amp_m):
    """`optic` with a constant Zernike OPD screen before M1, `amp_m` of wavefront in Noll `noll`.

    A `batoid.OPDScreen` coefficient of +a gives -a in `batoid.zernike` of the
    traced optic, so the coefficient is negated.
    """
    import batoid
    import galsim
    r_out = optic.pupilSize / 2
    coef = np.zeros(noll + 1)
    coef[noll] = -amp_m
    screen = batoid.OPDScreen(name='ComaScreen', surface=batoid.Plane(),
                              screen=batoid.Zernike(coef, R_outer=r_out, R_inner=EPS * r_out),
                              coordSys=optic.stopSurface.coordSys)
    del galsim
    return optic.withInsertedOptic(before='M1', item=screen)


def circle_centre(r, az_deg):
    """Least-squares circle (x0, y0, radius) from polar edge points about the origin, pixels.

    Linear in the small-offset limit: ``r(phi) = R + x0 cos(phi) + y0 sin(phi)``.
    """
    ok = np.isfinite(r)
    phi = np.deg2rad(az_deg[ok])
    a = np.column_stack([np.ones(ok.sum()), np.cos(phi), np.sin(phi)])
    sol, *_ = np.linalg.lstsq(a, r[ok], rcond=None)
    return sol[1], sol[2], sol[0]


def _render(optic, thx, thy, spacing, rng, r_out_pix, npix, kernel, x0y0=None):
    xs, ys = [], []
    gx, gy = bi.pupil_grid(spacing, rng=rng)
    for i in range(0, len(gx), bi.CHUNK):
        keep, x, y = bi.transmitted(optic, gx[i:i + bi.CHUNK], gy[i:i + bi.CHUNK], thx, thy)
        xs.append(x[keep])
        ys.append(y[keep])
    x, y = np.concatenate(xs), np.concatenate(ys)
    if x0y0 is None:
        x0y0 = (x.mean(), y.mean())
    u = (x - x0y0[0]) / bi.PIXEL_M + npix // 2
    v = (y - x0y0[1]) / bi.PIXEL_M + npix // 2
    return bi.render(u, v, npix, kernel)[1], x0y0


def side_sensitivity(task):
    """Edge-centre shifts with no screen, +1 µm Z7 and +1 µm Z8 for one donut side."""
    import batoid
    label, row, b, pupil_run = task
    t0 = time.time()
    thx, thy = np.radians(row['field_x_ccs_deg']), np.radians(row['field_y_ccs_deg'])
    offsets = (row['det_offset_mm'] * 1e-3, row['cam_offset_mm'] * 1e-3, row['m2_offset_mm'] * 1e-3)
    leaf = tuple(b['clippers'].split(';'))
    base = bi.configure(ob.reduced_v402(keep=tuple(set(leaf) | set(bi.OPTICAL))), row['rtp_deg'],
                        offsets, b['piston_um'] * 1e-6)
    npix = int(2 * row['stamp_half_pix'] + 1)
    sky_scale = 3600 * np.rad2deg(bi.PIXEL_M / 10.312)
    kernel = bi.atm_kernel(npix, bi.FWHM_ARCSEC, sky_scale)
    spacing = 4.18 / (row['r_out_pix'] * np.sqrt(RAYS_PER_PIX)) / 1.05
    out = dict(side=label, field_r_deg=row['field_r_deg'], clean=bool(row['clean']))
    centres = {}
    x0y0 = None
    nominal = batoid.Optic.fromYaml('LSST_r.yaml')
    for tag, noll in (('none', None), ('z7', 7), ('z8', 8)):
        optic = base if noll is None else with_screen(base, noll, COMA_UM * 1e-6)
        rng = np.random.default_rng(zlib.crc32(label.encode()))
        img, x0y0 = _render(optic, thx, thy, spacing, rng, row['r_out_pix'], npix, kernel, x0y0)
        valid = np.ones(img.shape, bool)
        c = npix // 2
        r_in, r_ot, _, _, _ = pe.edges(img, valid, c, c, row['r_out_pix'])
        az = (np.arange(pe.NBIN) + 0.5) * 360 / pe.NBIN
        xi, yi, ri = circle_centre(r_in, az)
        xo, yo, ro = circle_centre(r_ot, az)
        centres[tag] = np.array([xi - xo, yi - yo, xi, yi, xo, yo])
        out[f'r_inner_{tag}_pix'] = ri
        out[f'r_outer_{tag}_pix'] = ro
        if noll is not None:
            # Sign check of the screen on the nominal design optic, on axis.
            dz = (batoid.zernike(with_screen(nominal, noll, COMA_UM * 1e-6), 0, 0, 622e-9, jmax=11,
                                 eps=EPS, nx=128)
                  - batoid.zernike(nominal, 0, 0, 622e-9, jmax=11, eps=EPS, nx=128)) * 0.622
            out[f'screen_check_{tag}_um'] = float(dz[noll])
    for tag in ('z7', 'z8'):
        d = centres[tag] - centres['none']
        for k, name in enumerate(('rel_dx', 'rel_dy', 'in_dx', 'in_dy', 'out_dx', 'out_dy')):
            out[f'{tag}_{name}_pix_per_um'] = float(d[k] / COMA_UM)
    out['seconds'] = time.time() - t0
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}),
          flush=True)
    return out


def invert(sens, d_inner_pix, d_outer_pix, az_deg):
    """Z7, Z8 (µm of wavefront) from per-bin edge offsets, data minus Batoid (pixels).

    The relative centre shift (inner minus outer) of the offsets is solved
    against the 2 x 2 sensitivity ``[[z7_rel_dx, z8_rel_dx], [z7_rel_dy, z8_rel_dy]]``.
    """
    xi, yi, _ = circle_centre(d_inner_pix, az_deg)
    xo, yo, _ = circle_centre(d_outer_pix, az_deg)
    m = np.array([[sens['z7_rel_dx_pix_per_um'], sens['z8_rel_dx_pix_per_um']],
                  [sens['z7_rel_dy_pix_per_um'], sens['z8_rel_dy_pix_per_um']]])
    z7, z8 = np.linalg.solve(m, [xi - xo, yi - yo])
    return float(z7), float(z8), float(xi - xo), float(yi - yo)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pupil-run', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--sides', default=','.join(SIDES))
    args = ap.parse_args()
    import multiprocessing as mp
    pupil_run = pathlib.Path(args.pupil_run).resolve()
    run = pathlib.Path(args.run).resolve()
    z4 = pd.read_parquet(pupil_run / 'z4_fits.parquet')
    z4 = z4[z4.pupil == 'v4.02'].set_index(z4[z4.pupil == 'v4.02'].cand_id + '_'
                                            + z4[z4.pupil == 'v4.02'].side)
    bat = pd.read_parquet(pupil_run / 'batoid_images.parquet')
    bat = bat.set_index(bat.cand_id + '_' + bat.side)
    fd.pupil('v4.02', str(ob.pathlib.Path(__file__).resolve().parents[1] / 'output'
                          / '20261018_obscuration_v402' / 'RubinObsc_v4.02_r_rtpp0_pp0d0.yaml'))
    tasks = [(s, z4.loc[s].to_dict(), bat.loc[s].to_dict(), str(pupil_run))
             for s in args.sides.split(',')]
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        rows = pool.map(side_sensitivity, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_parquet(run / 'coma_sensitivity.parquet')
    df.to_csv(run / 'coma_sensitivity.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 200)
    print(df[['side'] + [c for c in df if c.endswith('pix_per_um') and 'rel' in c]
             + [c for c in df if c.startswith('screen')]].round(3).to_string())


if __name__ == '__main__':
    main()
