"""Mask coefficients (v4.02g, v4.02s) and Batoid v4.02 lit maps in the stop plane, for the viewer."""
import base64, json, pathlib, sys
from multiprocessing import Pool
import numpy as np, yaml, batoid

OUT = pathlib.Path(sys.argv[1])
MASKS = pathlib.Path.home() / 'Astrophysics/Claude/rubin-work/aos/giant_donut_pupil/output'
MODEL = pathlib.Path.home() / 'Astrophysics/Claude/rubin-optical-model/LSST_r_v4.02.yaml'
CONFIGS = ['det+0.0_cam+4.0_m2+4.0', 'det+0.0_cam-4.0_m2-4.0', 'det+1.5_cam+4.0_m2+4.0',
           'det+1.5_cam-4.0_m2-4.0', 'det-1.5_cam+4.0_m2+4.0', 'det-1.5_cam-4.0_m2-4.0']
THETAS = [0.0, 1.0, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75]
NX = 360

def masks():
    out = {}
    for ver, run in (('v4.02g', '20261019_pupil2'), ('v4.02s', '20261023_pupil6')):
        for cfg in CONFIGS:
            d = yaml.safe_load(open(MASKS / run / 'masks' / f'RubinObsc_{ver}_r_{cfg}.yaml'))
            els = []
            for name, edges in d.items():
                if name == 'Spider_3D':
                    continue
                for edge, e in edges.items():
                    els.append(dict(name=name, edge=edge, clear=bool(e['clear']), tmin=float(e['thetaMin']),
                                    tmax=float(e['thetaMax']), c=[float(v) for v in e['center']],
                                    r=[float(v) for v in e['radius']]))
            out[f'{ver}|{cfg}'] = els
    return out

def trace(args):
    cfg, th = args
    det = float(cfg.split('_')[0][3:]) * 1e-3
    cam = float(cfg.split('_')[1][3:]) * 1e-3
    m2 = float(cfg.split('_')[2][2:]) * 1e-3
    tel = batoid.Optic.fromYaml(str(MODEL))
    tel = tel.withGloballyShiftedOptic('LSSTCamera', [0, 0, cam]).withGloballyShiftedOptic('M2', [0, 0, m2])
    if det:
        tel = tel.withLocallyShiftedOptic('Detector', [0, 0, det])
    for it in [i.name for i in tel.items if i.name.startswith('Spider')]:
        tel = tel.withGloballyShiftedOptic(it, [1e3, 0, 0])   # spider out of the beam
    rays = batoid.RayVector.asGrid(optic=tel, wavelength=620e-9, theta_x=np.deg2rad(th), theta_y=0.0,
                                   nx=NX, lx=8.8)
    z_stop = tel.stopSurface.coordSys.origin[2]
    t = (z_stop - rays.z) / rays.vz
    xs, ys = rays.x + rays.vx * t, rays.y + rays.vy * t        # stop-plane coordinates (m)
    out = tel.trace(rays.copy())
    lit = (~out.vignetted).reshape(NX, NX)
    x0, y0 = float(xs.reshape(NX, NX)[0, 0]), float(ys.reshape(NX, NX)[0, 0])
    dx = float(xs.reshape(NX, NX)[0, 1] - xs.reshape(NX, NX)[0, 0])
    dy = float(ys.reshape(NX, NX)[1, 0] - ys.reshape(NX, NX)[0, 0])
    bits = np.packbits(lit.ravel().astype(np.uint8))
    return dict(cfg=cfg, theta=th, nx=NX, x0=x0, y0=y0, dx=dx, dy=dy, z_stop=float(z_stop),
                bits=base64.b64encode(bits.tobytes()).decode())

if __name__ == '__main__':
    jobs = [(c, t) for c in CONFIGS[:2] for t in THETAS]
    with Pool(8) as p:
        maps = p.map(trace, jobs)
    OUT.write_text(json.dumps(dict(masks=masks(), batoid=maps, thetas=THETAS)))
    print('wrote', OUT, OUT.stat().st_size, 'bytes;', len(maps), 'Batoid maps; z_stop', maps[0]['z_stop'],
          'grid spacing', maps[0]['dx'], maps[0]['dy'])
