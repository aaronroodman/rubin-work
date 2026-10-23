"""Geometry of Rubin_v1000_r and LSST_r_v4.02 for the model viewer: surfaces, obscuration outlines, rays."""
import json
import pathlib
import sys

import numpy as np

import batoid

HERE = pathlib.Path(__file__).resolve().parent
MODELS = {
    'v1000': HERE / 'Rubin_v1000_r.yaml',
    'v4.02': pathlib.Path.home() / 'Astrophysics/Claude/rubin-optical-model/LSST_r_v4.02.yaml',
}
MAX_VERTS = 120


def group_of(name):
    for prefix, g in (('CameraSilhouette', 'Camera silhouette (165 slices)'),
                      ('CameraBody', 'Camera body disk'),
                      ('CamAperture', 'In-camera openings'),
                      ('Spider', 'Spider vanes'),
                      ('TEACentralShade', 'Top End Assembly'), ('TopEnd', 'Top End Assembly'),
                      ('M2Cell', 'M2 cell'), ('M2BaffleCAD', 'M2 baffle'),
                      ('MidLevelBaffle', 'Mid-level baffle'), ('MidBaffleExt', 'Mid-level baffle'),
                      ('TMA', 'TMA structure baffle'), ('M1Baffle', 'M1 baffles')):
        if name.startswith(prefix):
            return g
    if name in ('M1', 'M2', 'M3'):
        return 'Mirrors'
    return 'Camera optics'


def walk(it, out):
    for c in getattr(it, 'items', []) or []:
        if hasattr(c, 'items'):
            walk(c, out)
        else:
            out.append(c)
    return out


def outlines(ob, n=72):
    """Local (x, y) loops of an obscuration's edges, and the radii of a clear annulus if it is one."""
    t = type(ob).__name__
    loops, clear = [], None
    ang = np.linspace(0, 2 * np.pi, n)
    if t == 'ObscNegation':
        sub, _ = outlines(ob.original, n)
        orig = ob.original
        if type(orig).__name__ == 'ObscAnnulus':
            clear = (orig.inner, orig.outer, orig.x, orig.y)
        elif type(orig).__name__ == 'ObscCircle':
            clear = (0.0, orig.radius, orig.x, orig.y)
        return sub, clear
    if t == 'ObscCircle':
        loops.append(np.c_[ob.x + ob.radius * np.cos(ang), ob.y + ob.radius * np.sin(ang)])
    elif t == 'ObscAnnulus':
        for r in (ob.inner, ob.outer):
            loops.append(np.c_[ob.x + r * np.cos(ang), ob.y + r * np.sin(ang)])
    elif t == 'ObscRectangle':
        w, h, th = ob.width / 2, ob.height / 2, getattr(ob, 'theta', 0.0)
        pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h], [-w, -h]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        loops.append(pts @ R.T + [ob.x, ob.y])
    elif t == 'ObscPolygon':
        xs, ys = np.asarray(ob.xs), np.asarray(ob.ys)
        k = max(1, int(np.ceil(len(xs) / MAX_VERTS)))
        xs, ys = xs[::k], ys[::k]
        loops.append(np.c_[np.r_[xs, xs[0]], np.r_[ys, ys[0]]])
    elif t == 'ObscUnion':
        for sub in ob.items:
            loops += outlines(sub, n)[0]
    else:
        print('unhandled obscuration', t, file=sys.stderr)
    return loops, clear


def to_global(item, xy, z_local=None):
    if z_local is None:
        z_local = item.surface.sag(xy[:, 0], xy[:, 1])
    loc = np.c_[xy, z_local]
    cs = item.coordSys
    return (cs.rot @ loc.T).T + cs.origin


def surface_mesh(item, r_in, r_out, cx, cy, nr=8, nphi=48):
    r = np.linspace(max(r_in, 1e-3), r_out, nr)
    p = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    R, P = np.meshgrid(r, p, indexing='ij')
    xy = np.c_[(cx + R * np.cos(P)).ravel(), (cy + R * np.sin(P)).ravel()]
    g = to_global(item, xy)
    tri = []
    for i in range(nr - 1):
        for j in range(nphi):
            a, b = i * nphi + j, i * nphi + (j + 1) % nphi
            c, d = (i + 1) * nphi + j, (i + 1) * nphi + (j + 1) % nphi
            tri += [(a, b, c), (b, d, c)]
    return dict(x=np.round(g[:, 0], 4).tolist(), y=np.round(g[:, 1], 4).tolist(), z=np.round(g[:, 2], 4).tolist(),
                i=[t[0] for t in tri], j=[t[1] for t in tri], k=[t[2] for t in tri])


def model_geometry(tag, path):
    optic = batoid.Optic.fromYaml(str(path))
    groups = {}
    meshes = []
    for item in walk(optic, []):
        g = group_of(item.name)
        ob = getattr(item, 'obscuration', None)
        if ob is None:
            continue
        loops, clear = outlines(ob)
        entry = groups.setdefault(g, dict(x=[], y=[], z=[], names=[]))
        for loop in loops:
            p = to_global(item, loop)
            entry['x'] += np.round(p[:, 0], 4).tolist() + [None]
            entry['y'] += np.round(p[:, 1], 4).tolist() + [None]
            entry['z'] += np.round(p[:, 2], 4).tolist() + [None]
        entry['names'].append(item.name)
        if g in ('Mirrors', 'Camera optics') and clear is not None:
            meshes.append(dict(name=item.name, group=g, **surface_mesh(item, *clear)))
    return dict(groups=groups, meshes=meshes)


def rays(path, field_deg):
    """Meridional fan of rays through the nominal model, with each ray's first blocking item."""
    optic = batoid.Optic.fromYaml(str(path))
    xs = np.unique(np.round(np.r_[np.linspace(-4.15, 4.15, 23), np.arange(-3.10, -2.54, 0.05)], 3))
    out = []
    names = [it.name for it in walk(optic, [])]
    for x in xs:
        rv = batoid.RayVector.asGrid(optic=optic, wavelength=620e-9, theta_x=np.deg2rad(field_deg), theta_y=0.0,
                                     nx=2, lx=0.1)
        # build one ray through stop-plane point (x, 0)
        z_stop = optic.stopSurface.coordSys.origin[2]
        t = (z_stop - rv.z[0]) / rv.vz[0]
        x0 = x - rv.vx[0] * t
        ray = batoid.RayVector(np.array([x0]), np.array([0.0 - rv.vy[0] * t]), np.array([rv.z[0]]),
                               np.array([rv.vx[0]]), np.array([rv.vy[0]]), np.array([rv.vz[0]]),
                               t=np.zeros(1), wavelength=np.array([620e-9]))
        tf = optic.traceFull(ray)
        pts, blocker = [[float(ray.x[0]), float(ray.y[0]), float(ray.z[0])]], None
        for n in names:
            if n not in tf:
                continue
            o = tf[n]['out']
            if blocker is None and o.vignetted[0]:
                blocker = n
            if n in ('M1', 'M2', 'M3') or n.endswith('_entrance') or n.endswith('_exit') or n == 'Detector':
                g = o.toCoordSys(batoid.CoordSys()) if hasattr(o, 'toCoordSys') else o
                pts.append([float(g.x[0]), float(g.y[0]), float(g.z[0])])
        out.append(dict(stop_x=float(x), blocker=blocker, pts=np.round(pts, 4).tolist()))
    return out


if __name__ == '__main__':
    data = {tag: model_geometry(tag, p) for tag, p in MODELS.items()}
    data['rays'] = {f'{tag}|{f}': rays(p, f) for tag, p in MODELS.items() for f in (0.0, 1.7)}
    data['silhouette_z'] = sorted(set(round(float(it.coordSys.origin[2]), 4) for it in walk(
        batoid.Optic.fromYaml(str(MODELS['v4.02'])), []) if it.name.startswith('CameraSilhouette')))
    out = HERE / 'model_data.json'
    out.write_text(json.dumps(data, separators=(',', ':')))
    for tag in MODELS:
        print(tag, {g: len(v['names']) for g, v in data[tag]['groups'].items()}, 'meshes', len(data[tag]['meshes']))
    print('rays blocked (v4.02, 1.7 deg):', [r['blocker'] for r in data['rays']['v4.02|1.7']])
    print('bytes', out.stat().st_size)
