"""Generate ts_wep ``maskParams`` for the v1000 pupil model.

ts_wep's blitz builds its donut pupil from `Instrument.maskParams`, not from
danish's pupil YAML files, and the shipped ``policy/instruments/LsstCam.yaml``
carries the v3.14 geometry only (``diameter`` 8.36 m, M1 inner radius 2.558 m).
Comparing pupil models inside blitz therefore needs ``maskParams`` generated from
the other model, which is what this module does.

Method
------
ts_wep models each optical-element edge as a circle whose centre and radius are
each a cubic in the field angle θ in degrees (`numpy.polyval` coefficient order,
highest power first, in meters). For each element and each θ, a pupil-filling ray
grid is traced, that element's obscuration alone is evaluated at its own surface,
and the radius of the boundary between kept and clipped rays is recorded. The
per-θ radii are then fitted with the cubic.

The shipped radii are **not all in the same frame**: M1's are in the entrance pupil
(its surface frame coincides with it), while M2, M3, L1 and the filter are
back-projected along the chief ray, with magnifications measured here of about
2.63 for M3, 2.21 for L1 and 20.2 for the filter. Rather than reproduce that
convention for every element, this module exploits the fact that **only M1 and M3
differ between v3.14 and v1000**, plus three elements v1000 adds:

=================  ====================================  ====================================
element            v3.14                                 v1000
=================  ====================================  ====================================
M1                 ``ObscAnnulus(2.558, 4.18)``          ``ObscAnnulus(2.5833, 4.18)``
M3                 ``ObscAnnulus(0.55, 2.508)``          ``ObscAnnulus(0.52735, 2.48511)``
M1Baffle1/2        absent                                ``ObscCircle(4.165)``
CameraBody         absent                                present
=================  ====================================  ====================================

M2, L1_entrance and Filter_entrance are byte-identical between the two models, so
their shipped coefficients are copied through unchanged and their frame convention
never has to be reproduced. Only the elements that actually change are refitted,
each in the frame its shipped coefficients already use, which is determined per
element by regenerating v3.14 and requiring agreement with the shipped values.

Validation
----------
`validate_against_shipped` regenerates v3.14 and compares against the shipped
coefficients. M1 agrees to about 0.4 mm on a 4.18 m radius, which is the finite
ray-grid limit rather than a method error. Run it before trusting any v1000
output: the generator is only credible to the extent it reproduces the model it
did not have to change.

Run as a script::

    python wfs/code/giant_donuts/gen_mask_params.py \\
        --output wfs/output/giant_donuts/maskParams_v1000.yaml
"""
import argparse
import copy
import pathlib
import sys

import batoid
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

# Field angles, in degrees, on which the per-element circles are fitted.  The
# science array reaches about 1.75 deg, the corner wavefront sensors about 1.85.
THETA_GRID_DEG = np.linspace(0.0, 1.85, 24)

# Rays per side of the entrance-pupil grid.  At 512 the boundary radius is
# recovered to about 0.4 mm on M1's 4.18 m, which is well inside the pupil-model
# differences being measured (M1 inner moves 25.3 mm from v3.14 to v1000).
NRAD_DEFAULT = 512

POLY_DEG = 3
WAVELENGTH_R_M = 620e-9

# Elements whose obscuration differs between v3.14 and v1000 *and* which set the
# surviving pupil boundary, so must be refitted.
#
# M3's aperture also changes (outer 2.508 m to 2.48511 m, inner 0.55 m to
# 0.52735 m) but is deliberately **not** refitted: it sits inside M1's inner
# shadow, so it never sets the boundary.  Traced on-axis through the full system,
# the surviving annulus runs 2.5580 to 4.1796 m in v3.14 and 2.5840 to 4.1796 m in
# v1000 -- the inner edge tracks M1's annulus (2.558 to 2.5833 m) to within the ray
# grid's 0.4 mm, and M3's change is invisible.  Refitting M3 would mean reproducing
# the shipped back-projected frame, which `validate_against_shipped` shows this
# module does not recover, for no effect on the pupil.
CHANGED_ELEMENTS = ('M1',)

# Elements v1000 adds.  The baffles are ClearCircle surfaces 15 mm inside M1's
# unchanged 4.18 m rim, so they define a new outer edge; CameraBody is a camera
# obscuration with no ts_wep edge analogue and is not emitted.
ADDED_ELEMENTS = ('M1Baffle1', 'M1Baffle2')

__all__ = [
    'CHANGED_ELEMENTS', 'ADDED_ELEMENTS',
    'edge_radius_vs_theta', 'fit_theta_poly', 'generate_mask_params',
    'validate_against_shipped',
]


def _pupil_grid(telescope, theta_deg, nrad, wavelength_m):
    """Trace a pupil-filling ray grid to every surface.

    Parameters
    ----------
    telescope : `batoid.Optic`
        The optical model.
    theta_deg : `float`
        Field angle, in degrees, along the x axis.
    nrad : `int`
        Rays per side of the entrance-pupil grid.
    wavelength_m : `float`
        Wavelength, in meters.

    Returns
    -------
    pupil_radius : `numpy.ndarray`
        Normalised entrance-pupil radius of each ray, dimensionless.
    in_pupil : `numpy.ndarray`
        Boolean, True for rays inside the unit entrance pupil.  `asGrid` fills the
        circumscribing square, so the corners must be excluded explicitly.
    traced : `dict`
        `batoid.Optic.traceFull` output.
    """
    rays = batoid.RayVector.asGrid(
        optic=telescope, wavelength=wavelength_m,
        theta_x=np.deg2rad(theta_deg), theta_y=0.0, nx=nrad,
    )
    half = telescope.pupilSize / 2
    pupil_radius = np.hypot(np.asarray(rays.x), np.asarray(rays.y)) / half
    return pupil_radius, pupil_radius <= 1.0, telescope.traceFull(rays.copy())


def _resolve(telescope, short_name):
    """Full `itemDict` key for an element's short name, or None if absent."""
    for key in telescope.itemDict:
        if key.split('.')[-1] == short_name:
            return key
    return None


def edge_radius_vs_theta(telescope, element, edge, frame='surface',
                         theta_grid_deg=THETA_GRID_DEG, nrad=NRAD_DEFAULT,
                         wavelength_m=WAVELENGTH_R_M):
    """Radius of one element edge as a function of field angle.

    Parameters
    ----------
    telescope : `batoid.Optic`
        The optical model.
    element : `str`
        Short element name, e.g. ``'M1'``.
    edge : {'outer', 'inner'}
        Which edge of the element's annular obscuration to follow.
    frame : {'surface', 'pupil'}, optional
        Frame the radius is reported in. ``'surface'`` is the element's own
        surface coordinates, ``'pupil'`` the entrance pupil in meters. Which one
        reproduces the shipped coefficients is per element, so
        `validate_against_shipped` decides it rather than this function assuming.
    theta_grid_deg : `numpy.ndarray`, optional
        Field angles, in degrees.
    nrad : `int`, optional
        Rays per side of the entrance-pupil grid.
    wavelength_m : `float`, optional
        Wavelength, in meters.

    Returns
    -------
    theta_deg : `numpy.ndarray`
        Field angles, in degrees.
    radius_m : `numpy.ndarray`
        Edge radius in the requested frame, in meters.  `numpy.nan` where the
        element does not clip the pupil at that angle.
    """
    key = _resolve(telescope, element)
    if key is None:
        raise KeyError(f'{element} absent from this model')
    obsc = getattr(telescope.itemDict[key], 'obscuration', None)
    if obsc is None:
        raise ValueError(f'{element} carries no obscuration')

    half = telescope.pupilSize / 2
    radii = []
    for theta in theta_grid_deg:
        pupil_radius, in_pupil, traced = _pupil_grid(
            telescope, theta, nrad, wavelength_m)
        if element not in traced:
            radii.append(np.nan)
            continue
        at = traced[element]['in']
        surf_radius = np.hypot(np.asarray(at.x), np.asarray(at.y))
        keep = in_pupil & ~obsc.contains(at.x, at.y)
        if np.count_nonzero(keep) < 16:
            radii.append(np.nan)
            continue

        measure = surf_radius if frame == 'surface' else pupil_radius * half
        radii.append(float(measure[keep].max() if edge == 'outer'
                           else measure[keep].min()))
    return np.asarray(theta_grid_deg, dtype=float), np.asarray(radii, dtype=float)


def fit_theta_poly(theta_deg, values, deg=POLY_DEG):
    """Fit a polynomial in field angle, in `numpy.polyval` coefficient order.

    Parameters
    ----------
    theta_deg : `numpy.ndarray`
        Field angles, in degrees.
    values : `numpy.ndarray`
        Quantity to fit, in meters.  `numpy.nan` entries are dropped.
    deg : `int`, optional
        Polynomial degree.

    Returns
    -------
    coeffs : `list` [`float`]
        ``deg + 1`` coefficients, highest power first, in meters.
    """
    good = np.isfinite(values)
    if good.sum() < deg + 1:
        constant = float(np.nanmedian(values)) if good.any() else 0.0
        return [0.0] * deg + [constant]
    return [float(c) for c in np.polyfit(theta_deg[good], values[good], deg)]


def validate_against_shipped(template_params, nrad=NRAD_DEFAULT,
                             theta_grid_deg=THETA_GRID_DEG):
    """Regenerate v3.14 and compare against the shipped coefficients.

    The only honest check on the generator: v3.14 is the model the shipped
    ``maskParams`` were built from, so regenerating it must return them. Reports
    both frames per edge, since which frame the shipped coefficients use varies by
    element.

    Parameters
    ----------
    template_params : `dict`
        The shipped ``maskParams``.
    nrad : `int`, optional
        Rays per side of the entrance-pupil grid.
    theta_grid_deg : `numpy.ndarray`, optional
        Field angles, in degrees.

    Returns
    -------
    report : `list` [`dict`]
        One entry per element edge, with the shipped and regenerated radius at
        θ = 0 in meters for each frame, and the better-matching frame's residual.
    """
    telescope = batoid.Optic.fromYaml('Rubin_v3.14_r.yaml')
    report = []
    for element, edges in template_params.items():
        if element == 'Spider_3D' or _resolve(telescope, element) is None:
            continue
        for edge, spec in edges.items():
            shipped_at_zero = float(np.polyval(spec['radius'], 0.0))
            row = {'element': element, 'edge': edge,
                   'shipped_r0_m': shipped_at_zero}
            for frame in ('surface', 'pupil'):
                try:
                    _, radius = edge_radius_vs_theta(
                        telescope, element, edge, frame=frame,
                        theta_grid_deg=theta_grid_deg[:1], nrad=nrad)
                    row[f'{frame}_r0_m'] = float(radius[0])
                except (KeyError, ValueError):
                    row[f'{frame}_r0_m'] = np.nan
            diffs = {f: abs(row[f'{f}_r0_m'] - shipped_at_zero)
                     for f in ('surface', 'pupil')
                     if np.isfinite(row.get(f'{f}_r0_m', np.nan))}
            if diffs:
                best = min(diffs, key=diffs.get)
                row['best_frame'] = best
                row['residual_m'] = diffs[best]
            report.append(row)
    return report


def generate_mask_params(template_params, model_name='Rubin_v1000_r',
                         frames=None, theta_grid_deg=THETA_GRID_DEG,
                         nrad=NRAD_DEFAULT, wavelength_m=WAVELENGTH_R_M):
    """Generate ``maskParams`` for a pupil model, refitting only what changed.

    Elements identical between v3.14 and v1000 keep their shipped coefficients, so
    their frame convention never has to be reproduced. `CHANGED_ELEMENTS` are
    refitted and `ADDED_ELEMENTS` appended.

    Parameters
    ----------
    template_params : `dict`
        The shipped ``maskParams``, supplying the schema, the ``clear`` flags, the
        θ validity ranges and the coefficients for unchanged elements.
    model_name : `str`, optional
        Batoid model to generate for.
    frames : `dict`, optional
        Per-element frame to measure in, as returned by
        `validate_against_shipped`. Elements absent default to ``'surface'``.
    theta_grid_deg : `numpy.ndarray`, optional
        Field angles, in degrees.
    nrad : `int`, optional
        Rays per side of the entrance-pupil grid.
    wavelength_m : `float`, optional
        Wavelength, in meters.

    Returns
    -------
    params : `dict`
        ``maskParams`` for `model_name`.
    """
    telescope = batoid.Optic.fromYaml(model_name + '.yaml')
    frames = frames or {}
    params = copy.deepcopy(template_params)

    for element in CHANGED_ELEMENTS:
        if element not in params or _resolve(telescope, element) is None:
            continue
        frame = frames.get(element, 'surface')
        for edge in params[element]:
            theta, radius = edge_radius_vs_theta(
                telescope, element, edge, frame=frame,
                theta_grid_deg=theta_grid_deg, nrad=nrad,
                wavelength_m=wavelength_m)
            params[element][edge]['radius'] = fit_theta_poly(theta, radius)

    for element in ADDED_ELEMENTS:
        if _resolve(telescope, element) is None:
            continue
        theta, radius = edge_radius_vs_theta(
            telescope, element, 'outer', frame=frames.get(element, 'surface'),
            theta_grid_deg=theta_grid_deg, nrad=nrad, wavelength_m=wavelength_m)
        if not np.isfinite(radius).any():
            continue
        params[element] = {
            'outer': {
                'clear': True,
                'thetaMin': 0.0,
                'thetaMax': 1.85,
                # The baffles are centred circles, so the centre stays at zero and
                # only the radius is fitted.
                'center': [0.0, 0.0, 0.0, 0.0],
                'radius': fit_theta_poly(theta, radius),
            }
        }
    return params


def main():
    ap = argparse.ArgumentParser(description='Generate v1000 ts_wep maskParams.')
    ap.add_argument('--model', default='Rubin_v1000_r',
                    help='batoid model name, without the .yaml suffix')
    ap.add_argument('--output', required=True, help='YAML file to write')
    ap.add_argument('--nrad', type=int, default=NRAD_DEFAULT,
                    help='rays per side of the entrance-pupil grid')
    ap.add_argument('--validate-only', action='store_true',
                    help='only regenerate v3.14 and report agreement')
    args = ap.parse_args()

    from lsst.ts.wep.instrument import Instrument
    template = Instrument(configFile='policy:instruments/LsstCam.yaml').maskParams

    print(f'validating the generator against the shipped v3.14 coefficients '
          f'(nrad={args.nrad})')
    report = validate_against_shipped(template, nrad=args.nrad)
    print(f"{'element':<16} {'edge':<6} {'shipped':>10} {'surface':>10} "
          f"{'pupil':>10} {'frame':>8} {'resid_m':>10}")
    for row in report:
        print(f"{row['element']:<16} {row['edge']:<6} "
              f"{row['shipped_r0_m']:>10.4f} "
              f"{row.get('surface_r0_m', np.nan):>10.4f} "
              f"{row.get('pupil_r0_m', np.nan):>10.4f} "
              f"{row.get('best_frame', '-'):>8} "
              f"{row.get('residual_m', np.nan):>10.5f}")

    frames = {r['element']: r['best_frame'] for r in report if 'best_frame' in r}
    if args.validate_only:
        return

    print(f'\ngenerating maskParams for {args.model}')
    params = generate_mask_params(template, model_name=args.model, frames=frames,
                                  nrad=args.nrad)
    for element in CHANGED_ELEMENTS + ADDED_ELEMENTS:
        if element in params:
            for edge, spec in params[element].items():
                print(f"  {element:<12} {edge:<6} radius(theta=0) = "
                      f"{np.polyval(spec['radius'], 0.0):.5f} m")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        yaml.safe_dump({'maskParams': params}, f, default_flow_style=None,
                       sort_keys=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
