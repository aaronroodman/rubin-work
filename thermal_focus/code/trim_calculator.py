"""Standalone thermal-focus trim calculator: predict the focus correction from temperatures.

Given the Telescope Mount Assembly (TMA) truss temperature and the four M1M3 bulk thermal
gradients, this returns the uniform-defocus error the Active Optics System (AOS) is expected to
have accumulated, and the hexapod trim adjustment that cancels it.

**This file deliberately imports nothing but numpy.** It is meant to be copied to a summit
machine and run there, so every constant is inlined below with its units and its provenance.
There is no database, no Engineering Facilities Database (EFD) client and no repository
dependency; the caller supplies the five numbers.

Invocation, as a command::

    python trim_calculator.py --truss-temp-c 11.3 --z-gradient-c-per-m -0.0656 \
        --y-gradient-c-per-m -0.0196 --radial-gradient-c-per-m -0.0168 \
        --x-gradient-c-per-m 0.0017

    python trim_calculator.py --self-test

or as a function::

    from trim_calculator import predict_focus_error_um, trim_adjustment
    dz = predict_focus_error_um(truss_temp_c=11.3, z_gradient_c_per_m=-0.0656,
                               y_gradient_c_per_m=-0.0196,
                               radial_gradient_c_per_m=-0.0168,
                               x_gradient_c_per_m=0.0017)

Where the inputs come from
--------------------------
``truss_temp_c`` is the **mean of two thermometers**, ``tma_truss_temp_pxpy`` and
``tma_truss_temp_mxmy``, from the TMA thermal telemetry, interpolated in time within the night
to the exposure midpoint. It is not a single stored channel — a consumer that reads one
thermometer and calls it the truss temperature will be offset from the value these coefficients
were fitted against.

The four gradients are bulk linear fits to the M1M3 thermocouple field in °C per m, along the
mirror x, y and z axes and radially. They are derived quantities too, not raw channels.

What the output means
---------------------
The returned focus error is in **µm of equivalent hexapod dz, split evenly between the camera
and M2 hexapods** — 1 µm means 0.5 µm on each. That convention is baked into the fitted
coefficients through the v-mode-1 projection, so a consumer that moves only the camera hexapod
must scale by the factor `UM_CAMERA_ALONE_PER_UM_SHARED` below.

Notes
-----
This is a prediction of the *uncorrected* focus error from temperatures alone. It does not read
the wavefront and does not know what the AOS has already commanded, so it is a feed-forward
term, not a replacement for the closed loop. Applied blind on top of an already-converged loop
it would double-count the correction.

The coefficients are fitted **between nights**, where the truss temperature moves by degrees.
Within a single observing block it moves by a median of 0.0658 °C, which is telemetry noise, and
feeding that noise through a coefficient of +124.49 µm of equivalent hexapod dz per °C produces
a prediction that swings almost as much as the drift it would be correcting — over 45 Full Array
Mode blocks the median within-block peak-to-peak is 34.9 µm of equivalent hexapod dz of real
drift against a 20.8 µm prediction swing, and subtracting the prediction makes the within-block
scatter *worse*, 46.9 µm, a ratio of 1.35 (dimensionless, corrected over uncorrected), improving
only 5 of the 45 blocks. So do not use this to chase focus within a block; it is a
night-to-night term.
"""
import argparse

import numpy as np

# --------------------------------------------------------------------------------- constants
#
# Fitted with a Huber robust linear model on 68,296 ordinary science visits over 149 nights,
# day_obs 20251103 to 20260713, holding whole nights out of the fit. The response is
#
#     (v1_trim - v1_measured) / V1_PER_UM_DZ
#
# where v1 is the amplitude of the first singular vector of the AOS sensitivity matrix, which is
# essentially uniform defocus. The hexapod look-up-table (LUT) term is excluded: it is a known
# commanded function of elevation, so including it would inject an elevation dependence that is
# not a focus error.

#: Response with every feature at zero [µm of equivalent hexapod dz]. This is a long
#: extrapolation from the sample, whose feature means are given in `SAMPLE_FEATURE_MEANS`, so it
#: is not a physically meaningful standalone offset — only the whole equation is.
INTERCEPT_UM = -1385.31

#: Coefficient on the TMA truss temperature [µm of equivalent hexapod dz per °C].
TRUSS_UM_PER_C = +124.49

#: Coefficients on the four M1M3 bulk thermal gradients
#: [µm of equivalent hexapod dz per (°C per m)].
Z_GRADIENT_UM_PER_C_PER_M = -805.99
Y_GRADIENT_UM_PER_C_PER_M = -1271.73
RADIAL_GRADIENT_UM_PER_C_PER_M = -946.57
X_GRADIENT_UM_PER_C_PER_M = -3414.66

#: Feature means over the fitted sample, for judging whether an input is an extrapolation.
#: Truss temperature in °C, the four gradients in °C per m.
SAMPLE_FEATURE_MEANS = {'truss_temp_c': 11.31893,
                        'z_gradient_c_per_m': -0.06559,
                        'y_gradient_c_per_m': -0.01959,
                        'radial_gradient_c_per_m': -0.01678,
                        'x_gradient_c_per_m': +0.00168}

#: Full observed range of each feature over the fitted sample, as ``(low, high)``. Outside this
#: the prediction is an extrapolation and `predict_focus_error_um` says so. These are the actual
#: minimum and maximum, not a percentile clip; note how narrow the x gradient is — it spans
#: 0.064 °C per m in total, so its large coefficient acts over a small lever arm.
SAMPLE_FEATURE_RANGE = {'truss_temp_c': (3.87652, 25.07418),
                        'z_gradient_c_per_m': (-0.76917, +0.68002),
                        'y_gradient_c_per_m': (-0.14575, +0.04549),
                        'radial_gradient_c_per_m': (-0.23442, +0.13742),
                        'x_gradient_c_per_m': (-0.01793, +0.04637)}

#: Night-grouped residual scatter of the fit [µm of equivalent hexapod dz, normalized median
#: absolute deviation]. The uncertainty on any single prediction is about this, against an
#: uncorrected scatter of `UNCORRECTED_NMAD_UM`.
RESIDUAL_NMAD_UM = 60.1

#: Scatter of the uncorrected focus error over the same sample [µm of equivalent hexapod dz].
UNCORRECTED_NMAD_UM = 337.0

#: v-mode-1 amplitude per µm of total hexapod dz travel [dimensionless per µm], at the
#: 10-degree-of-freedom, 1-mode projection, which is the one the online Optical Feedback Control
#: system uses. Derived from the AOS sensitivity matrix rather than assumed to carry over from
#: the projection the fit was done at: the full 50-degree-of-freedom, 34-mode projection gives
#: 9.00851e-04 and the 22/12 projection 9.00942e-04, so the three agree to 0.108%
#: (dimensionless, spread over the 50/34 value). That spread is far below the fit's own
#: uncertainty, so the choice of projection does not matter here.
V1_PER_UM_DZ = 9.018277e-04

#: Ratio of the two "equivalent dz" conventions [dimensionless, camera-alone µm over shared µm].
#: The fitted coefficients are in the shared convention, 0.5 µm on each hexapod. A consumer that
#: holds M2 still and moves only the camera hexapod must multiply the predicted dz by this. At
#: the 10/1 projection one unit of v-mode-1 amplitude is 1108.859 µm of shared travel against
#: 1120.559 µm of camera-alone travel, both derived from the sensitivity matrix. This is a
#: definition choice between two ways of expressing the same optical state, not an uncertainty.
UM_CAMERA_ALONE_PER_UM_SHARED = 1120.559 / 1108.859

#: Equivalent hexapod dz per µm of wavefront defocus [µm of equivalent hexapod dz per µm of
#: wavefront]. Negative because positive hexapod dz produces negative defocus. Provided so a
#: prediction can be quoted as a wavefront amplitude.
DZ_UM_PER_UM_WF = -63.9902


def predict_focus_error_um(truss_temp_c, z_gradient_c_per_m, y_gradient_c_per_m,
                           radial_gradient_c_per_m, x_gradient_c_per_m,
                           warn_extrapolation=True):
    """Predicted uniform-defocus error from the five thermal inputs.

    Parameters
    ----------
    truss_temp_c : `float` or `array_like`
        TMA truss temperature: the mean of ``tma_truss_temp_pxpy`` and ``tma_truss_temp_mxmy``,
        interpolated within the night [°C].
    z_gradient_c_per_m, y_gradient_c_per_m, radial_gradient_c_per_m, x_gradient_c_per_m : \
            `float` or `array_like`
        M1M3 bulk thermal gradients [°C per m].
    warn_extrapolation : `bool`, optional
        Print a warning for any input outside `SAMPLE_FEATURE_RANGE`.

    Returns
    -------
    dz_um : `float` or `numpy.ndarray`
        Predicted focus error [µm of equivalent hexapod dz, split evenly between the camera and
        M2 hexapods]. Positive means the telescope has drifted in the direction a positive
        hexapod dz would produce.

    Notes
    -----
    The uncertainty on one prediction is about `RESIDUAL_NMAD_UM`, from an uncorrected scatter of
    `UNCORRECTED_NMAD_UM` — a factor of 5.6 (dimensionless, uncorrected over residual scatter).
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
                print(f'WARNING: {name} outside the fitted range {lo} to {hi}; the prediction '
                      f'is an extrapolation')
    dz_um = (INTERCEPT_UM
             + TRUSS_UM_PER_C * vals['truss_temp_c']
             + Z_GRADIENT_UM_PER_C_PER_M * vals['z_gradient_c_per_m']
             + Y_GRADIENT_UM_PER_C_PER_M * vals['y_gradient_c_per_m']
             + RADIAL_GRADIENT_UM_PER_C_PER_M * vals['radial_gradient_c_per_m']
             + X_GRADIENT_UM_PER_C_PER_M * vals['x_gradient_c_per_m'])
    return float(dz_um) if dz_um.ndim == 0 else dz_um


def trim_adjustment(truss_temp_c, z_gradient_c_per_m, y_gradient_c_per_m,
                    radial_gradient_c_per_m, x_gradient_c_per_m, camera_alone=False,
                    warn_extrapolation=True):
    """The hexapod trim adjustment that cancels the predicted focus error.

    Parameters
    ----------
    truss_temp_c : `float` or `array_like`
        TMA truss temperature [°C].
    z_gradient_c_per_m, y_gradient_c_per_m, radial_gradient_c_per_m, x_gradient_c_per_m : \
            `float` or `array_like`
        M1M3 bulk thermal gradients [°C per m].
    camera_alone : `bool`, optional
        Return the motion for the camera hexapod holding M2 still, instead of the shared
        convention. Scales by `UM_CAMERA_ALONE_PER_UM_SHARED`.
    warn_extrapolation : `bool`, optional
        Passed to `predict_focus_error_um`.

    Returns
    -------
    out : `dict`
        ``focus_error_um`` (the prediction), ``trim_dz_um`` (the correction, its negative),
        ``camera_hexapod_dz_um`` and ``m2_hexapod_dz_um`` (how to split it) — all in µm — plus
        ``v1_amplitude`` (dimensionless v-mode-1 amplitude), ``wavefront_um``
        (µm of wavefront defocus) and ``uncertainty_um``.

    Notes
    -----
    The correction is the negative of the error, so applying ``trim_dz_um`` is what removes it.
    In the shared convention the total travel is split evenly, so each hexapod moves half.
    """
    err = predict_focus_error_um(truss_temp_c, z_gradient_c_per_m, y_gradient_c_per_m,
                                 radial_gradient_c_per_m, x_gradient_c_per_m,
                                 warn_extrapolation=warn_extrapolation)
    trim = -np.asarray(err, float)
    if camera_alone:
        cam = trim * UM_CAMERA_ALONE_PER_UM_SHARED
        m2 = np.zeros_like(cam)
    else:
        cam = trim / 2.0
        m2 = trim / 2.0
    scalar = np.ndim(trim) == 0
    out = dict(focus_error_um=err,
               trim_dz_um=float(trim) if scalar else trim,
               camera_hexapod_dz_um=float(cam) if scalar else cam,
               m2_hexapod_dz_um=float(m2) if scalar else m2,
               v1_amplitude=float(np.asarray(err) * V1_PER_UM_DZ) if scalar
               else np.asarray(err) * V1_PER_UM_DZ,
               wavefront_um=float(np.asarray(err) / DZ_UM_PER_UM_WF) if scalar
               else np.asarray(err) / DZ_UM_PER_UM_WF,
               uncertainty_um=RESIDUAL_NMAD_UM)
    return out


#: Worked test cases, each ``(label, inputs, expected focus error in µm of equivalent hexapod
#: dz)``. The expected values are the fitted equation evaluated by hand, so they verify the
#: constants above have been transcribed correctly. The analysis stage reproduces them from the
#: fitted pipeline itself.
TEST_CASES = (
    ('the sample mean, all five features',
     dict(truss_temp_c=11.31893, z_gradient_c_per_m=-0.06559, y_gradient_c_per_m=-0.01959,
          radial_gradient_c_per_m=-0.01678, x_gradient_c_per_m=0.00168),
     +111.71),
    ('a cold night, gradients at zero',
     dict(truss_temp_c=5.0, z_gradient_c_per_m=0.0, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     -762.86),
    ('a warm night, gradients at zero',
     dict(truss_temp_c=20.0, z_gradient_c_per_m=0.0, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     +1104.49),
    ('the truss at its mean, a strong z gradient only',
     dict(truss_temp_c=11.31893, z_gradient_c_per_m=-0.50, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     +426.77),
    ('the truss at its mean, a strong y gradient only',
     dict(truss_temp_c=11.31893, z_gradient_c_per_m=0.0, y_gradient_c_per_m=-0.10,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.0),
     +150.96),
    ('the truss at its mean, the x gradient at its upper edge',
     dict(truss_temp_c=11.31893, z_gradient_c_per_m=0.0, y_gradient_c_per_m=0.0,
          radial_gradient_c_per_m=0.0, x_gradient_c_per_m=0.03),
     -78.66),
)


def self_test(tol_um=0.05, verbose=True):
    """Check the inlined constants against the worked test cases.

    Parameters
    ----------
    tol_um : `float`, optional
        Tolerance [µm of equivalent hexapod dz].
    verbose : `bool`, optional
        Print each case.

    Returns
    -------
    max_abs_diff : `float`
        Largest disagreement over `TEST_CASES` [µm of equivalent hexapod dz].

    Raises
    ------
    AssertionError
        If any case disagrees by more than `tol_um`, which means a constant was mistyped.
    """
    worst = 0.0
    for label, inputs, expected in TEST_CASES:
        got = predict_focus_error_um(**inputs, warn_extrapolation=False)
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
    ap.add_argument('--camera-alone', action='store_true',
                    help='report the camera hexapod moving alone, M2 held still')
    ap.add_argument('--self-test', action='store_true',
                    help='check the inlined constants against the worked test cases and exit')
    args = ap.parse_args()

    if args.self_test:
        print('worked test cases [um of equivalent hexapod dz]')
        self_test()
        return
    if args.truss_temp_c is None:
        ap.error('--truss-temp-c is required; or pass --self-test')

    out = trim_adjustment(args.truss_temp_c, args.z_gradient_c_per_m, args.y_gradient_c_per_m,
                          args.radial_gradient_c_per_m, args.x_gradient_c_per_m,
                          camera_alone=args.camera_alone)
    conv = 'camera hexapod alone, M2 held still' if args.camera_alone else \
        'split evenly between the camera and M2 hexapods'
    print(f'predicted focus error   {out["focus_error_um"]:+9.1f} +/- {out["uncertainty_um"]:.1f}'
          f' um of equivalent hexapod dz')
    print(f'  as a v-mode-1 amplitude {out["v1_amplitude"]:+9.5f} (dimensionless)')
    print(f'  as wavefront defocus    {out["wavefront_um"]:+9.4f} um of wavefront')
    print(f'trim adjustment to apply {out["trim_dz_um"]:+9.1f} um of hexapod dz ({conv})')
    print(f'  camera hexapod dz      {out["camera_hexapod_dz_um"]:+9.1f} um')
    print(f'  M2 hexapod dz          {out["m2_hexapod_dz_um"]:+9.1f} um')


if __name__ == '__main__':
    main()
