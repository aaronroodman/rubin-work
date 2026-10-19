"""Tests for `obscuration_v402`.

1. `danish_keep` applies ``clear`` and ``thetaMin``/``thetaMax`` as danish does,
   on a synthetic one-circle file.
2. `_fit_circle_fixed` recovers a circle from a synthetic edge (the numpy-2 fix).
3. The CAD ``Spider_3D`` vanes cast the same stop-plane shadow as the batoid
   v4.02 spider planes, at 0 and 1.75 deg.
4. The vanes load into a danish factory and give a finite image.

Invocation::

    python test_obscuration_v402.py
    pytest test_obscuration_v402.py
"""
import numpy as np

import obscuration_v402 as ob


def test_danish_keep_circle():
    params = {'A': {'outer': dict(clear=True, thetaMin=0.0, thetaMax=1.0,
                                  center=[0.0], radius=[1.0])}}
    u = np.array([0.0, 0.5, 1.5])
    v = np.zeros(3)
    assert list(ob.danish_keep(params, u, v, 0.0, 0.0)) == [True, True, False]
    # Outside thetaMax the edge is inactive.
    assert ob.danish_keep(params, u, v, np.deg2rad(1.5), 0.0).all()
    params['A']['outer']['clear'] = False
    assert list(ob.danish_keep(params, u, v, 0.0, 0.0)) == [False, False, True]


def test_fit_circle_fixed():
    # An item edge of radius 1 m that maps one-to-one onto the pupil; the fitted
    # circle in the pupil is then that same circle.
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-1.2, 1.2, (2, 20000))
    y = np.sort(y)
    result = ob._fit_circle_fixed(x, y, x, y, 4.18, 0.5, 1.0)
    assert np.all(np.isfinite(result))
    assert abs(abs(result[1]) - 1.0) < 0.02


def test_spider_vanes_match_batoid():
    import yaml
    optic = ob.planes_optic(ob.SPIDER_PREFIXES)
    z_stop = float(yaml.safe_load(ob.V402_YAML.read_text())
                   ['opticalSystem']['stopSurface']['coordSys']['z'])
    params = {'Spider_3D': ob.cad_spider_vanes(z_stop)}
    for theta in (0.0, 1.75):
        u, v, kept = ob.batoid_keep(optic, (0.0, 0.0, 0.0), theta, nrad=80, naz=500)
        inside = (np.hypot(u, v) < 4.18) & (np.hypot(u, v) > 2.58)
        dk = ob.danish_keep(params, u, v, np.deg2rad(theta), 0.0)
        blocked = (~kept & inside).sum() / inside.sum()
        xor = (dk != kept)[inside].sum() / inside.sum()
        assert 0.01 < blocked < 0.05
        assert xor < 0.002


def test_vanes_in_danish_factory():
    import danish
    import yaml
    z_stop = float(yaml.safe_load(ob.V402_YAML.read_text())
                   ['opticalSystem']['stopSurface']['coordSys']['z'])
    params = {'Spider_3D': ob.cad_spider_vanes(z_stop)}
    factory = danish.DonutFactory(R_outer=4.18, R_inner=2.5498, mask_params=params,
                                  spider_angle=0.0, focal_length=10.31, pixel_scale=10e-6)
    img = factory.image(aberrations=np.array([0.0] * 4 + [3e-5]), thx=0.0, thy=0.0,
                        npix=181)
    assert np.isfinite(img).all() and img.sum() > 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
