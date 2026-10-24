"""CCD-height sign check: the zkStart height term against batoid traced with the height map.
import numpy as np, batoid
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils import DefocalType
from batoid_rubin.builder import det_height_maps
from lsst.ts.intrinsic.wavefront.ccd_height import HEIGHT_TO_Z4_UM_PER_MM
import os
HMAP = os.path.expanduser('~/u/LSST/packages/batoid_rubin_data/ccd_height_map')
maps = det_height_maps(HMAP)
inst = Instrument(configFile='policy:instruments/LsstCam.yaml')
band = 'r'
tel = inst.getBatoidModel(band)
wl = inst.wavelength[band]
eps = tel.pupilObscuration
off = inst.defocalOffset
print(f'defocalOffset {off*1e3:.3f} mm, wavelength {wl*1e9:.0f} nm, eps {eps:.3f}')

def ta(optic, thx, thy):
    return batoid.zernikeTA(optic, thx, thy, wl, eps=eps, jmax=11, nrad=20, naz=120)*wl*1e6  # um

def land(optic, thx, thy):
    r = batoid.RayVector.fromStop(0, 0, optic, wavelength=wl, theta_x=thx, theta_y=thy)
    optic.trace(r, reverse=False)
    return float(r.x[0]), float(r.y[0])

# Corner rafts: SW0 extra (+1.5 mm), SW1 intra (-1.5 mm), per the ts_intrinsic_wavefront convention.
# Field points near each corner sensor (deg, batoid frame), refined so the chief ray lands on the map domain.
rows = []
for raft in ['R00', 'R04', 'R40', 'R44']:
    for side, sw, sign in [('extra', 'SW0', +1), ('intra', 'SW1', -1)]:
        m = maps[f'{raft}_{sw}']
        xs, ys = m.xs, m.ys
        # aim at the map centre: find field angle whose chief ray lands there (linear plate scale)
        xc, yc = 0.5*(xs[0]+xs[-1]), 0.5*(ys[0]+ys[-1])
        shifted = tel.withLocallyShiftedOptic('Detector', [0, 0, sign*off])
        th = np.array([xc, yc])/10.31
        for _ in range(8):
            p = np.array(land(shifted, *th))
            if np.hypot(*(p-[xc, yc])) < 1e-7: break
            J = np.empty((2, 2)); d = 1e-6
            for k in range(2):
                t2 = th.copy(); t2[k] += d; J[:, k] = (np.array(land(shifted, *t2))-p)/d
            th = th + np.linalg.solve(J, np.array([xc, yc])-p)
        thx, thy = th
        x, y = land(shifted, thx, thy)
        h_mm = float(m.sag(x, y))*1e3
        withh = tel.withPerturbedSurface('Detector', m).withLocallyShiftedOptic('Detector', [0, 0, sign*off])
        z0 = ta(shifted, thx, thy); zh = ta(withh, thx, thy)
        dz4_batoid = zh[4]-z0[4]
        dz4_tswep = HEIGHT_TO_Z4_UM_PER_MM*h_mm        # zkStart[Z4] += factor * h, both sides
        rows.append((raft, side, np.rad2deg(thx), np.rad2deg(thy), h_mm*1e3, z0[4], dz4_batoid*1e3, dz4_tswep*1e3))
print(f"{'raft':5s} {'side':6s} {'thx':>7s} {'thy':>7s} {'h':>8s} {'Z4_TA':>8s} {'dZ4_batoid':>11s} {'dZ4_tswep':>10s} {'ratio':>6s}")
print(f"{'':5s} {'':6s} {'[deg]':>7s} {'[deg]':>7s} {'[um]':>8s} {'[um]':>8s} {'[nm]':>11s} {'[nm]':>10s} {'':>6s}")
for r in rows:
    print(f'{r[0]:5s} {r[1]:6s} {r[2]:+7.3f} {r[3]:+7.3f} {r[4]:+8.2f} {r[5]:+8.3f} {r[6]:+11.2f} {r[7]:+10.2f} {r[6]/r[7] if r[7] else np.nan:6.3f}')
