"""Standalone thermal-focus trim calculator: predict the focus degree-of-freedom Trim.

Goal: determine the degree-of-freedom (DOF) Trim values that set the focus v-mode (v1) from
thermal telemetry. Method: v1 is predicted from five thermal telemetry values with a Huber robust
linear fit, converted to microns of equivalent hexapod dz at 1110.1 um of hexapod dz per unit v1,
and back-projected into the four DOF that v-mode 1 contains.

**This file deliberately imports nothing but numpy.** It is meant to be copied to a summit
machine and run there, so every constant is inlined below.

Invocation, as a command::

    python trim_calculator.py --truss-temp-c 11.3 --z-gradient-c-per-m -0.0656 \
        --y-gradient-c-per-m -0.0196 --radial-gradient-c-per-m -0.0168 \
        --x-gradient-c-per-m 0.0017

    python trim_calculator.py --self-test

or as a function::

    from trim_calculator import predict_focus_error, predict_trim
    v1, v1_dz = predict_focus_error(11.3, -0.0656, -0.0196, -0.0168, 0.0017)
    out = predict_trim(11.3, -0.0656, -0.0196, -0.0168, 0.0017)

``truss_temp_c`` is the mean of the two TMA truss thermometers ``tma_truss_temp_pxpy`` and
``tma_truss_temp_mxmy``, interpolated within the night to the exposure midpoint; the four
gradients are bulk linear fits to the M1M3 thermocouple field. All five are derived quantities,
not raw telemetry channels.
"""
import argparse
import warnings

import numpy as np

# --------------------------------------------------------------------------------- constants

#: Response with every feature at zero [um of equivalent hexapod dz].
INTERCEPT_UM = -1392.01

#: Coefficient on the TMA truss temperature [um of equivalent hexapod dz per deg C].
TRUSS_UM_PER_C = +125.09

#: Coefficients on the four M1M3 bulk thermal gradients
#: [um of equivalent hexapod dz per (deg C per m)].
Z_GRADIENT_UM_PER_C_PER_M = -811.32
Y_GRADIENT_UM_PER_C_PER_M = -1254.02
RADIAL_GRADIENT_UM_PER_C_PER_M = -949.08
X_GRADIENT_UM_PER_C_PER_M = -3374.73

#: Mean of each feature over the fitted sample. Truss temperature in deg C, the four gradients
#: in deg C per m.
SAMPLE_FEATURE_MEANS = {'truss_temp_c': +11.27840,
                        'z_gradient_c_per_m': -0.06474,
                        'y_gradient_c_per_m': -0.01964,
                        'radial_gradient_c_per_m': -0.01671,
                        'x_gradient_c_per_m': +0.00169}

#: Full observed range of each feature over the fitted sample, as ``(low, high)``. Truss
#: temperature in deg C, the four gradients in deg C per m.
SAMPLE_FEATURE_RANGE = {'truss_temp_c': (+3.87652, +17.70162),
                        'z_gradient_c_per_m': (-0.76917, +0.68002),
                        'y_gradient_c_per_m': (-0.14575, +0.03324),
                        'radial_gradient_c_per_m': (-0.23442, +0.13742),
                        'x_gradient_c_per_m': (-0.01793, +0.04637)}

#: Residual scatter of the fit [um of equivalent hexapod dz, normalized median absolute
#: deviation], the uncertainty on a single prediction.
RESIDUAL_NMAD_UM = 58.2

#: Scatter of the open-loop focus over the same sample, before the correction
#: [um of equivalent hexapod dz].
UNCORRECTED_NMAD_UM = 336.8

#: v-mode-1 amplitude per um of total hexapod dz travel, split evenly between the camera and M2
#: hexapods [dimensionless per um]. The inverse is 1110.1 um of hexapod dz per unit v1.
V1_PER_UM_DZ = 9.00851e-04

#: Ratio of the two equivalent-dz conventions [dimensionless, camera-alone um over shared um].
#: A consumer that holds M2 still and moves only the camera hexapod multiplies the predicted dz
#: by this, 1121.8 um camera-alone against 1110.1 um shared per unit v1.
UM_CAMERA_ALONE_PER_UM_SHARED = 1121.8 / 1110.1

#: Equivalent hexapod dz per um of wavefront defocus [um of equivalent hexapod dz per um of
#: wavefront].
DZ_UM_PER_UM_WF = -63.9902

#: DOF content of one unit of v-mode-1 amplitude [um per unit v1], at the 50-DOF, 34-mode
#: projection. The ts_ofc DOF ordering is M2 hexapod first: dof0-4 are the M2 hexapod, dof5-9 the
#: camera hexapod, dof10-29 the M1M3 bending modes and dof30-49 the M2 bending modes. Every DOF
#: not listed carries below 2e-04 um per unit v1.
V1_DOF_UM_PER_UNIT = {'dof5': -645.657870,       # camera hexapod dz
                      'dof0': -463.898287,       # M2 hexapod dz
                      'dof12': +0.009390,        # M1M3 bending mode B3
                      'dof34': +0.007551}        # M2 bending mode B5

#: Name and reporting unit for each entry of `V1_DOF_UM_PER_UNIT`, in the order `predict_trim`
#: reports them.
V1_DOF_LABELS = (('dof5', 'camera hexapod dz', 'um'),
                 ('dof0', 'M2 hexapod dz', 'um'),
                 ('dof12', 'M1M3 bending mode B3', 'um'),
                 ('dof34', 'M2 bending mode B5', 'um'))


def predict_focus_error(truss_temp_c, z_gradient_c_per_m, y_gradient_c_per_m,
                        radial_gradient_c_per_m, x_gradient_c_per_m, warn_extrapolation=True):
    """Predict the focus v-mode from the five thermal telemetry values.

    Parameters
    ----------
    truss_temp_c : `float` or `array_like`
        TMA truss temperature, the mean of the two thermometers [deg C].
    z_gradient_c_per_m, y_gradient_c_per_m, radial_gradient_c_per_m, x_gradient_c_per_m : \
            `float` or `array_like`
        M1M3 bulk thermal gradients [deg C per m].
    warn_extrapolation : `bool`, optional
        Issue a `UserWarning` for any input outside `SAMPLE_FEATURE_RANGE`. The prediction is
        returned either way.

    Returns
    -------
    v1 : `float` or `numpy.ndarray`
        Predicted v-mode-1 amplitude [dimensionless].
    v1_dz : `float` or `numpy.ndarray`
        The same prediction as focus [um of equivalent hexapod dz, split evenly between the
        camera and M2 hexapods].

    Notes
    -----
    The uncertainty on one prediction is `RESIDUAL_NMAD_UM`, against an open-loop scatter of
    `UNCORRECTED_NMAD_UM`.
    """
    vals = {'truss_temp_c': np.asarray(truss_temp_c, float),
            'z_gradient_c_per_m': np.asarray(z_gradient_c_per_m, float),
            'y_gradient_c_per_m': np.asarray(y_gradient_c_per_m, float),
            'radial_gradient_c_per_m': np.asarray(radial_gradient_c_per_m, float),
            'x_gradient_c_per_m': np.asarray(x_gradient_c_per_m, float)}
    if warn_extrapolation:
        for name, v in vals.items():
            lo, hi = SAMPLE_FEATURE_RANGE[name]
            if np.any(v < lo) or np.any(v > hi):
                warnings.warn(f'{name} is outside the fitted range {lo:+.5f} to {hi:+.5f}; the '
                              f'prediction is an extrapolation', UserWarning, stacklevel=2)
    v1_dz = (INTERCEPT_UM
             + TRUSS_UM_PER_C * vals['truss_temp_c']
             + Z_GRADIENT_UM_PER_C_PER_M * vals['z_gradient_c_per_m']
             + Y_GRADIENT_UM_PER_C_PER_M * vals['y_gradient_c_per_m']
             + RADIAL_GRADIENT_UM_PER_C_PER_M * vals['radial_gradient_c_per_m']
             + X_GRADIENT_UM_PER_C_PER_M * vals['x_gradient_c_per_m'])
    v1 = v1_dz * V1_PER_UM_DZ
    if v1_dz.ndim == 0:
        return float(v1), float(v1_dz)
    return v1, v1_dz


def predict_trim(truss_temp_c, z_gradient_c_per_m, y_gradient_c_per_m,
                 radial_gradient_c_per_m, x_gradient_c_per_m, warn_extrapolation=True):
    """The four DOF Trim values that set the predicted focus v-mode.

    Parameters
    ----------
    truss_temp_c : `float` or `array_like`
        TMA truss temperature, the mean of the two thermometers [deg C].
    z_gradient_c_per_m, y_gradient_c_per_m, radial_gradient_c_per_m, x_gradient_c_per_m : \
            `float` or `array_like`
        M1M3 bulk thermal gradients [deg C per m].
    warn_extrapolation : `bool`, optional
        Passed to `predict_focus_error`.

    Returns
    -------
    out : `dict`
        ``v1`` [dimensionless v-mode-1 amplitude] and ``v1_dz`` [um of equivalent hexapod dz]
        from `predict_focus_error`, one entry per DOF keyed as in `V1_DOF_UM_PER_UNIT` —
        ``dof5`` and ``dof0`` the camera and M2 hexapod dz, ``dof12`` the M1M3 bending mode B3
        and ``dof34`` the M2 bending mode B5, all [um] — and ``uncertainty_um``, the scatter on
        ``v1_dz`` [um of equivalent hexapod dz].

    Notes
    -----
    Each DOF is ``V1_DOF_UM_PER_UNIT[dof] * v1``, with no sign flip: the commanded term enters
    the wavefront response positively, so the Trim amplitude that sets a predicted v1 is that v1
    itself. The hexapod split in the back-projection is uneven, 58.2% of the travel on the camera
    hexapod against 41.8% on M2, because that is the shape of the optical mode.

    The two bending-mode amplitudes stay below about 7 nm over the fitted sample, far under the
    scatter of the mirror figure Trim the observatory runs. They are reported for completeness.
    """
    v1, v1_dz = predict_focus_error(truss_temp_c, z_gradient_c_per_m, y_gradient_c_per_m,
                                    radial_gradient_c_per_m, x_gradient_c_per_m,
                                    warn_extrapolation=warn_extrapolation)
    v1_arr = np.asarray(v1, float)
    scalar = v1_arr.ndim == 0
    out = dict(v1=v1, v1_dz=v1_dz, uncertainty_um=RESIDUAL_NMAD_UM)
    for dof, unit_content in V1_DOF_UM_PER_UNIT.items():
        val = unit_content * v1_arr
        out[dof] = float(val) if scalar else val
    return out


#: Worked test cases, each ``(label, inputs, expected v1_dz in um of equivalent hexapod dz)``.
#: The expected values are the fitted equation evaluated independently, so they verify the
#: constants above have been transcribed correctly.
TEST_CASES = (
    ('the sample mean, all five features',
     dict(truss_temp_c=11.27840, z_gradient_c_per_m=-0.06474, y_gradient_c_per_m=-0.01964,
          radial_gradient_c_per_m=-0.01671, x_gradient_c_per_m=0.00169),
     +106.11),
    ('a cold night, gradients at zero',
     dict(truss_temp_c=5.0, z_gradient_c_per_m=0.0, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     -766.56),
    ('a warm night, gradients at zero',
     dict(truss_temp_c=17.0, z_gradient_c_per_m=0.0, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     +734.52),
    ('the truss at its mean, a strong z gradient only',
     dict(truss_temp_c=11.27840, z_gradient_c_per_m=-0.50, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     +424.47),
    ('the truss at its mean, a strong y gradient only',
     dict(truss_temp_c=11.27840, z_gradient_c_per_m=0.0, y_gradient_c_per_m=-0.10,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     +144.21),
    ('the truss at its mean, the x gradient at its upper edge',
     dict(truss_temp_c=11.27840, z_gradient_c_per_m=0.0, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.03),
     -82.43),
)


def self_test(tol_um=0.05, verbose=True):
    """Check the inlined constants and the DOF back-projection.

    Parameters
    ----------
    tol_um : `float`, optional
        Tolerance on each worked case [um of equivalent hexapod dz].
    verbose : `bool`, optional
        Print each case.

    Returns
    -------
    max_abs_diff : `float`
        Largest disagreement over `TEST_CASES` [um of equivalent hexapod dz].

    Raises
    ------
    AssertionError
        If any case disagrees by more than `tol_um`, or if the two hexapod dz entries of the DOF
        back-projection do not sum to the total travel `V1_PER_UM_DZ` inverts.
    """
    worst = 0.0
    for label, inputs, expected in TEST_CASES:
        _, got = predict_focus_error(**inputs, warn_extrapolation=False)
        diff = abs(got - expected)
        worst = max(worst, diff)
        if verbose:
            print(f'  {label:44s} expected {expected:+9.2f}  got {got:+9.2f}  '
                  f'difference {diff:.3f} um of equivalent hexapod dz')
    assert worst < tol_um, (f'a constant is mistyped: worst case differs by {worst:.3f} um of '
                           f'equivalent hexapod dz, over the tolerance {tol_um}')
    if verbose:
        print(f'  all {len(TEST_CASES)} cases agree to {worst:.3f} um of equivalent hexapod dz '
              f'(tolerance {tol_um})')

    # The two hexapod dz entries must sum to the total travel V1_PER_UM_DZ inverts, since that is
    # the convention the fitted coefficients are in.
    label, inputs, _ = TEST_CASES[0]
    out = predict_trim(**inputs, warn_extrapolation=False)
    total = out['dof5'] + out['dof0']
    expect_total = -out['v1_dz']
    rel = abs(total - expect_total) / max(abs(expect_total), 1e-12)
    if verbose:
        print(f'  DOF Trim on "{label}":')
        print(f'    v1 {out["v1"]:+.6f} (dimensionless), v1_dz {out["v1_dz"]:+.2f} um of '
              f'equivalent hexapod dz')
        for dof, name, unit in V1_DOF_LABELS:
            print(f'    {name:24s} {out[dof]:+12.6f} {unit}')
        print(f'    hexapod dz sum {total:+.4f} um against the total travel v1_dz implies '
              f'{expect_total:+.4f} um, relative disagreement {rel:.2e} (dimensionless)')
    assert rel < 1e-3, (f'the v-mode-1 DOF content is inconsistent with V1_PER_UM_DZ: the two '
                        f'hexapod dz sum to {total:+.4f} um against {expect_total:+.4f} um '
                        f'expected, a relative disagreement of {rel:.2e}')

    # Array input must give the same numbers as the scalar path, element by element.
    arr = predict_trim(np.array([11.27840, 5.0]), np.array([-0.06474, 0.0]),
                       np.array([-0.01964, 0.0]), np.array([-0.01671, 0.0]),
                       np.array([0.00169, 0.0]), warn_extrapolation=False)
    assert np.allclose(arr['v1_dz'], [TEST_CASES[0][2], TEST_CASES[1][2]], atol=tol_um), \
        'the array path disagrees with the scalar worked cases'
    if verbose:
        print(f'  the array path reproduces the scalar cases to {tol_um} um of equivalent '
              f'hexapod dz')
    return worst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--truss-temp-c', type=float,
                    help='TMA truss temperature, the mean of tma_truss_temp_pxpy and '
                         'tma_truss_temp_mxmy interpolated within the night [deg C]')
    ap.add_argument('--z-gradient-c-per-m', type=float, default=0.0,
                    help='M1M3 z thermal gradient [deg C per m]')
    ap.add_argument('--y-gradient-c-per-m', type=float, default=0.0,
                    help='M1M3 y thermal gradient [deg C per m]')
    ap.add_argument('--radial-gradient-c-per-m', type=float, default=0.0,
                    help='M1M3 radial thermal gradient [deg C per m]')
    ap.add_argument('--x-gradient-c-per-m', type=float, default=0.0,
                    help='M1M3 x thermal gradient [deg C per m]')
    ap.add_argument('--self-test', action='store_true',
                    help='check the inlined constants against the worked test cases and exit')
    args = ap.parse_args()

    if args.self_test:
        print('worked test cases [um of equivalent hexapod dz]')
        self_test()
        return
    if args.truss_temp_c is None:
        ap.error('--truss-temp-c is required; or pass --self-test')

    out = predict_trim(args.truss_temp_c, args.z_gradient_c_per_m, args.y_gradient_c_per_m,
                       args.radial_gradient_c_per_m, args.x_gradient_c_per_m)
    print(f'predicted v1     {out["v1"]:+12.6f} (dimensionless)')
    print(f'predicted v1_dz  {out["v1_dz"]:+12.1f} +/- {out["uncertainty_um"]:.1f} um of '
          f'equivalent hexapod dz')
    print('DOF Trim to apply')
    for dof, name, unit in V1_DOF_LABELS:
        print(f'  {dof:6s} {name:24s} {out[dof]:+12.6f} {unit}')


if __name__ == '__main__':
    main()
