"""Fine Batoid lit maps (5 mm) over the silhouette arc, stop plane x -3.45..-2.25 m, y -1.8..1.8 m."""
import base64, json, sys
from multiprocessing import Pool
import numpy as np, batoid
import build_data as bd

STEP = 0.005
XS = np.arange(-3.45, -2.25 + 1e-9, STEP)
YS = np.arange(-1.80, 1.80 + 1e-9, STEP)

def patch(args):
    cfg, th = args
    det = float(cfg.split('_')[0][3:]) * 1e-3
    cam = float(cfg.split('_')[1][3:]) * 1e-3
    m2 = float(cfg.split('_')[2][2:]) * 1e-3
    tel = batoid.Optic.fromYaml(str(bd.MODEL))
    tel = tel.withGloballyShiftedOptic('LSSTCamera', [0, 0, cam]).withGloballyShiftedOptic('M2', [0, 0, m2])
    for it in [i.name for i in tel.items if i.name.startswith('Spider')]:
        tel = tel.withGloballyShiftedOptic(it, [1e3, 0, 0])
    ref = batoid.RayVector.asGrid(optic=tel, wavelength=620e-9, theta_x=np.deg2rad(th), theta_y=0.0, nx=4, lx=8.8)
    z0, vx, vy, vz = ref.z[0], ref.vx[0], ref.vy[0], ref.vz[0]
    z_stop = tel.stopSurface.coordSys.origin[2]
    X, Y = np.meshgrid(XS, YS)
    t = (z_stop - z0) / vz
    x0 = X.ravel() - vx * t
    y0 = Y.ravel() - vy * t
    n = x0.size
    rays = batoid.RayVector(x0, y0, np.full(n, z0), np.full(n, vx), np.full(n, vy), np.full(n, vz),
                            t=np.zeros(n), wavelength=np.full(n, 620e-9))
    out = tel.trace(rays)
    lit = ~out.vignetted
    bits = np.packbits(lit.astype(np.uint8))
    return dict(cfg=cfg, theta=th, nx=len(XS), ny=len(YS), x0=float(XS[0]), y0=float(YS[0]), step=STEP,
                bits=base64.b64encode(bits.tobytes()).decode())

if __name__ == '__main__':
    D = json.load(open('viewer_data.json'))
    jobs = [(c, t) for c in bd.CONFIGS[:2] for t in bd.THETAS if t >= 1.2]
    with Pool(8) as p:
        D['patch'] = p.map(patch, jobs)
    json.dump(D, open('viewer_data.json', 'w'))
    print(len(D['patch']), 'patches;', len(json.dumps(D)), 'bytes total')
