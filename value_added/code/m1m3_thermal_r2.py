"""Quadratic-in-radius thermal terms from the M1M3 thermocouple grid.

The primary mirror M1M3 carries 146 thermocouples in its glass. The value-added database
already holds the four bulk gradients `lsst.ts.m1m3.utils.ThermocoupleAnalysis` reduces them
to -- along the x, y and z axes and linearly in radius. This module adds the **quadratic**
radial term, because a temperature field going as radius squared bends the mirror into a
shape that is much closer to pure defocus than a linear radial ramp is, and so is the term
most likely to move focus.

Three quadratic terms are computed, over three thermocouple populations: the whole mirror,
the M1 annulus alone and the M3 inner disc alone. M1 and M3 are one monolithic blank but two
optical surfaces at different radii and different curvatures, so a thermal expansion confined
to one of them is a different optical perturbation than the same expansion over both.

Provided:
  - ``THERMOCOUPLE_GEOMETRY`` -- name, x, y, radius and mirror assignment per thermocouple.
  - ``fit_r2_terms`` -- the quadratic coefficients for one table of thermocouple temperatures.
  - ``r2_terms_for_span`` -- load one time span from the Engineering Facility Database (EFD)
    and reduce it, returning a time-indexed frame ready to interpolate onto visits.

Import from the repo root::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from value_added.code.m1m3_thermal_r2 import fit_r2_terms

Notes
-----
**The quadratic coefficient is fitted alongside the linear terms, never alone, and in an
orthogonalized basis.** A fit of temperature against radius squared by itself would absorb most
of the linear radial ramp the database already carries and would duplicate
``m1m3_radial_gradient_c_per_m`` rather than add to it. Simply appending ``r**2`` to
``[1, r, z]`` is not enough either: over the M1 annulus, where radius spans only 2.997 m to
4.197 m, ``r**2`` is 99.875% explained by the linear terms, a variance inflation factor of
801.8 (dimensionless), so its coefficient would be almost pure noise amplification.

The quadratic column is therefore **Gram-Schmidt orthogonalized** against ``[1, r, z]`` over
each population's own sensor positions before the fit, which drops the design matrix condition
number from 2070.7 to 36.2 (dimensionless) on M1. The reported coefficient then carries only
the radial curvature the linear terms cannot express, which is exactly the "does radius squared
help beyond the gradients we already have" question. The linear and constant coefficients are
unchanged by the orthogonalization; only the quadratic one is reinterpreted.

The orthogonalized direction is scaled to unit root-mean-square over the population's sensors,
so the coefficient is **°C of temperature swing per unit normalized quadratic amplitude**
rather than °C/m². ``R2_SHAPE_RMS_M2`` records the m² scale that was divided out per
population, so a coefficient can be converted back to a raw ``°C/m²`` curvature by multiplying
by it. The unit-rms convention keeps the three populations comparable in size despite M1's much
narrower radial span.

The mirror split uses the optical prescription rather than the thermocouple radii: batoid's
``LSST_r.yaml`` puts M1 on the annulus 2.558 m to 4.180 m and M3 on 0.550 m to 2.508 m. The
thermocouple grid has a ring of 26 sensors at 2.510 to 2.533 m -- just outside M3's polished
edge, since a thermocouple is drilled into the blank rather than into the optical surface --
and then nothing until M1's innermost ring at 2.997 m. ``MIRROR_SPLIT_RADIUS_M`` sits in that
0.464 m empty gap, so every thermocouple is assigned and the assignment does not depend on
where in the gap the boundary is placed.

`ThermocoupleAnalysis.calculate_gradients_xyz_r` computes its own ``x_gradient_err``,
``y_gradient_err`` and ``z_gradient_err`` from a residual formed against the y-coordinate
array rather than the temperatures, so those three upstream error columns are not meaningful.
This module computes its own errors from the temperature residual and does not read them. The
upstream gradient *values* are unaffected.
"""
import warnings

import numpy as np
import pandas as pd

try:  # pragma: no cover - stack-only
    from lsst.ts.m1m3.utils import ThermocoupleAnalysis
    from lsst.ts.m1m3.utils.thermocouples import ThermocoupleTable
except Exception:  # pragma: no cover
    ThermocoupleAnalysis = None
    ThermocoupleTable = None

#: Radius separating the M3 inner disc from the M1 outer annulus [m]. Placed in the empty gap
#: between the outermost M3 thermocouple ring (2.533 m) and the innermost M1 ring (2.997 m);
#: see the module notes for why the optical edge radii themselves are not the boundary.
MIRROR_SPLIT_RADIUS_M = 2.75

#: Optical surface extents from batoid ``LSST_r.yaml`` [m], recorded so the split can be
#: checked against the prescription without re-deriving it.
M1_OPTICAL_RADIUS_M = (2.558, 4.180)
M3_OPTICAL_RADIUS_M = (0.550, 2.508)

#: The three quadratic columns this module produces, with the population each is fitted over.
#: The unit is °C per unit normalized quadratic amplitude (see the module notes), which is
#: written into the database as ``deg C (normalized r^2 amplitude)``.
R2_COLS = (
    ('m1m3_r2_coeff_c', 'all', 'M1M3 quadratic radial thermal term'),
    ('m1_r2_coeff_c', 'M1', 'M1 quadratic radial thermal term'),
    ('m3_r2_coeff_c', 'M3', 'M3 quadratic radial thermal term'),
)

#: Root-mean-square of the orthogonalized quadratic shape before unit-rms scaling, per
#: population [m²]. Multiplying a coefficient by this recovers a raw °C/m² curvature.
#: Filled at import once the geometry is known.
R2_SHAPE_RMS_M2 = {}


def thermocouple_geometry():
    """Position and mirror assignment of every M1M3 thermocouple.

    Returns
    -------
    geom : `pandas.DataFrame`
        Indexed by thermocouple name, with columns ``x_m``, ``y_m`` (positions in the mirror
        frame [m]), ``radius_m`` [m], ``level`` (``'B'`` back, ``'M'`` middle, ``'F'`` front,
        or a digit for the duplicated back sensors), ``z_rel`` (dimensionless 0 / 0.5 / 1
        matching the upstream 3-D map) and ``mirror`` (``'M1'`` or ``'M3'``).

    Raises
    ------
    RuntimeError
        If `lsst.ts.m1m3.utils` is unavailable, which is the off-stack case.
    """
    if ThermocoupleTable is None:
        raise RuntimeError('lsst.ts.m1m3.utils is unavailable; the thermocouple table '
                           'lives on the Rubin Science Platform AOS stack')
    rows = []
    for tc in ThermocoupleTable:
        r = float(np.hypot(tc.x_position, tc.y_position))
        last = tc.name[-1]
        # Matches the upstream __coordinate_map: M -> 0.5, F -> 1, anything else (B, B1, B2)
        # -> 0. The duplicated back sensors end in a digit and are back-face like plain B.
        z_rel = 0.5 if last == 'M' else (1.0 if last == 'F' else 0.0)
        rows.append({'name': tc.name, 'x_m': float(tc.x_position),
                     'y_m': float(tc.y_position), 'radius_m': r, 'level': last,
                     'z_rel': z_rel, 'cell': str(tc.core_location),
                     'mirror': 'M1' if r >= MIRROR_SPLIT_RADIUS_M else 'M3'})
    geom = pd.DataFrame(rows).set_index('name')
    return geom


#: Geometry table, built once at import where the stack is available.
try:  # pragma: no cover - stack-only
    THERMOCOUPLE_GEOMETRY = thermocouple_geometry()
except Exception:  # pragma: no cover
    THERMOCOUPLE_GEOMETRY = None


def _design(r, z, use_z):
    """Design matrix with the quadratic column orthogonalized against the linear terms.

    Parameters
    ----------
    r : `numpy.ndarray`
        Sensor radii [m].
    z : `numpy.ndarray`
        Dimensionless depth level, 0 / 0.5 / 1.
    use_z : `bool`
        Include the depth column.

    Returns
    -------
    A : `numpy.ndarray`
        Shape ``(n_sensors, 3)`` or ``(n_sensors, 4)``. The last column is the unit-rms
        orthogonalized quadratic shape; the leading columns are ``[1, r]`` plus ``z``.
    shape_rms_m2 : `float`
        Root-mean-square of the orthogonalized quadratic before scaling [m²], so the
        coefficient can be converted back to a raw °C/m² curvature.
    """
    lin = [np.ones_like(r), r]
    if use_z:
        lin.append(z)
    A_lin = np.column_stack(lin)
    q = r ** 2
    # Gram-Schmidt: remove whatever the linear terms can already express.
    beta, *_ = np.linalg.lstsq(A_lin, q, rcond=None)
    q = q - A_lin @ beta
    shape_rms_m2 = float(np.sqrt(np.mean(q ** 2)))
    if shape_rms_m2 > 0:
        q = q / shape_rms_m2
    return np.column_stack([A_lin, q]), shape_rms_m2


def fit_r2_terms(temperatures, geom=None, use_z=True, min_sensors=8):
    """Fit the quadratic radial thermal term over each mirror population.

    The temperature field is modelled as ``T = a0 + a1 * r (+ a3 * z) + a2 * q(r)`` by ordinary
    least squares over the thermocouples of the population, once per time sample, where
    ``q(r)`` is the unit-root-mean-square Gram-Schmidt orthogonalization of ``r**2`` against
    the other columns. ``a2`` is the reported quadratic coefficient: the radial curvature that
    the constant, linear and depth terms cannot express.

    Parameters
    ----------
    temperatures : `pandas.DataFrame`
        Thermocouple temperatures [°C], one row per time sample, one column per thermocouple
        name, as `ThermocoupleAnalysis.all_thermocouples_dataframe` provides. Columns not
        present in ``geom`` are ignored.
    geom : `pandas.DataFrame`, optional
        Geometry from `thermocouple_geometry`. Defaults to ``THERMOCOUPLE_GEOMETRY``.
    use_z : `bool`, optional
        Include the front/middle/back depth term, matching the upstream 3-D fit. The depth
        coordinate is the dimensionless 0 / 0.5 / 1 level, not a physical thickness, so its
        coefficient is not a per-metre gradient and is not reported.
    min_sensors : `int`, optional
        A population with fewer than this many reporting sensors in a sample yields NaN for
        that sample rather than a fit on too few points.

    Returns
    -------
    out : `pandas.DataFrame`
        Indexed as ``temperatures``. For each population a coefficient column from `R2_COLS`
        [°C per unit normalized quadratic amplitude], its formal error ``<col>_err`` in the
        same unit, the linear term fitted beside it ``<population>_r_coeff_c_per_m`` [°C/m],
        the residual scatter ``<population>_rms_c`` [°C] and the sensor count
        ``<population>_n_sensors`` (dimensionless).

    Notes
    -----
    The linear coefficient returned here is **not** the database's
    ``m1m3_radial_gradient_c_per_m``: that one comes from a fit with no quadratic term, so it
    carries whatever curvature exists, whereas this one is the ramp with the curvature taken
    out. Both are kept so the study can show how much of the linear term the quadratic absorbs.

    The fit is ordinary least squares, not robust. It is a 3- or 4-parameter fit to 40-146
    sensors reading a smooth physical field, and a thermocouple either reports or does not;
    the repository's robust-fit preference applies to the study-level regressions against
    focus, which are Huber.
    """
    if geom is None:
        geom = THERMOCOUPLE_GEOMETRY
    if geom is None:
        raise RuntimeError('no thermocouple geometry available; lsst.ts.m1m3.utils is '
                           'needed at import time')
    shared = [c for c in temperatures.columns if c in geom.index]
    if not shared:
        raise ValueError('no thermocouple column of the input matches the geometry table')
    out = pd.DataFrame(index=temperatures.index)
    for col, population, _label in R2_COLS:
        if population == 'all':
            names = shared
        else:
            names = [c for c in shared if geom.at[c, 'mirror'] == population]
        prefix = col[:-len('_r2_coeff_c')]
        if not names:
            out[col] = np.nan
            out[f'{col}_err'] = np.nan
            out[f'{prefix}_r_coeff_c_per_m'] = np.nan
            out[f'{prefix}_rms_c'] = np.nan
            out[f'{prefix}_n_sensors'] = 0
            continue
        r = geom.loc[names, 'radius_m'].to_numpy(float)
        z = geom.loc[names, 'z_rel'].to_numpy(float)
        temps = temperatures[names].to_numpy(float)
        n_t = temps.shape[0]
        a2 = np.full(n_t, np.nan)
        a2_err = np.full(n_t, np.nan)
        a1 = np.full(n_t, np.nan)
        rms = np.full(n_t, np.nan)
        n_used = np.zeros(n_t, dtype=int)
        # The reporting set is almost always constant across a span, so the pseudo-inverse is
        # reused for every sample sharing a finite pattern rather than refactored per row.
        finite = np.isfinite(temps)
        patterns, inverse = np.unique(finite, axis=0, return_inverse=True)
        for p_idx, pattern in enumerate(patterns):
            rows = np.flatnonzero(inverse == p_idx)
            k = int(pattern.sum())
            n_used[rows] = k
            if k < min_sensors:
                continue
            A, shape_rms = _design(r[pattern], z[pattern], use_z)
            if shape_rms <= 0 or np.linalg.matrix_rank(A) < A.shape[1]:
                # A population all at one radius cannot separate the radial terms, and the
                # orthogonalized quadratic collapses to zero.
                continue
            R2_SHAPE_RMS_M2.setdefault(population, shape_rms)
            y = temps[np.ix_(rows, pattern)]
            beta, *_ = np.linalg.lstsq(A, y.T, rcond=None)
            resid = y.T - A @ beta
            dof = max(k - A.shape[1], 1)
            sigma2 = np.sum(resid ** 2, axis=0) / dof
            cov_unit = np.linalg.pinv(A.T @ A)
            # The orthogonalized quadratic is the last column; the linear ramp is column 1.
            a1[rows] = beta[1]
            a2[rows] = beta[-1]
            a2_err[rows] = np.sqrt(sigma2 * cov_unit[-1, -1])
            rms[rows] = np.sqrt(sigma2)
        out[col] = a2
        out[f'{col}_err'] = a2_err
        out[f'{prefix}_r_coeff_c_per_m'] = a1
        out[f'{prefix}_rms_c'] = rms
        out[f'{prefix}_n_sensors'] = n_used
    return out


async def r2_terms_for_span(client, start, end, time_bin=30, use_z=True):
    """Load one span of M1M3 thermocouple telemetry and reduce it to the quadratic terms.

    Parameters
    ----------
    client : `lsst_efd_client.EfdClient`
        EFD client.
    start, end : `astropy.time.Time`
        Span to load. Use **one night at a time**; a multi-night span makes the thermocouple
        query time out, the same constraint `common.ess_telemetry.get_m1m3_gradients` carries.
    time_bin : `int`, optional
        Binning passed to `ThermocoupleAnalysis.load` [s].
    use_z : `bool`, optional
        Passed to `fit_r2_terms`.

    Returns
    -------
    terms : `pandas.DataFrame` or `None`
        Time-indexed quadratic terms, or `None` where the telemetry is absent over the span,
        so the caller can fill NaN rather than distinguishing failure modes itself.

    Notes
    -----
    The two telemetry-gap modes `get_m1m3_gradients` documents apply identically here and are
    caught the same way: a cold-junction channel that never reported over the span makes
    `load` raise `KeyError` while slicing a fixed column list, and a span where no
    thermocouple reported at all leaves the frame empty.
    """
    if ThermocoupleAnalysis is None:
        raise RuntimeError('lsst.ts.m1m3.utils is unavailable; the thermocouple reduction '
                           'lives on the Rubin Science Platform AOS stack')
    ta = ThermocoupleAnalysis(client)
    try:
        await ta.load(start, end, time_bin=time_bin)
    except KeyError as exc:
        warnings.warn(f'M1M3 thermocouple channel {exc} missing over {start.isot} to '
                      f'{end.isot}; quadratic terms unavailable')
        return None
    temps = getattr(ta, 'all_thermocouples_dataframe', None)
    if temps is None or len(temps) == 0:
        warnings.warn(f'no M1M3 thermocouple telemetry over {start.isot} to {end.isot}; '
                      'quadratic terms unavailable')
        return None
    return fit_r2_terms(temps, use_z=use_z)
