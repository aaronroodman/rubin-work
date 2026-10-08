"""Standalone thermal-focus trim calculator: predict the focus degree-of-freedom Trim.

We predict the degree-of-freedom (DOF) Trim values that correct focus based on thermal telemetry.
The correction was derived from over 60k science visits, where the open-loop v-Mode 1 (v1) was
determined from the DoF Trim minus the measured value from CWFS wavefront. Then v1 was modeled via
a robust (Huber) linear fit to thermal telemetry values: mean TMA truss temperature, M1M3
z-gradient, r-gradient, x-gradient and y-gradient. For convenience we also convert v1 to microns of
equivalent hexapod focus (v1_dz) with conversion factor 1110.1 um of hexapod dz per unit v1.

Invocation, as a command::

    python trim_calculator.py --truss-temp-c 11.3 --z-gradient-c-per-m -0.0656 \
        --y-gradient-c-per-m -0.0196 --radial-gradient-c-per-m -0.0168 \
        --x-gradient-c-per-m 0.0017

or as a class::

    from trim_calculator import TrimCalculator
    calc = TrimCalculator()
    v1, v1_dz, dof_dict = calc.predict_trim(11.3, -0.0656, -0.0196, -0.0168, 0.0017)

``truss_temp_c`` is the mean of the two TMA truss thermometers ``tma_truss_temp_pxpy`` and
``tma_truss_temp_mxmy``, interpolated within the night to the exposure midpoint; the four
gradients are bulk linear fits to the M1M3 thermocouple field.

Every fitted coefficient lives in ``trim_coefficients.yaml`` next to this file, read by
`TrimCalculator` at construction, so copy both files together. Pass a path to use a refitted file.
The tests live in ``test_trim_calculator.py``.
"""
import argparse
import pathlib
import warnings

import numpy as np
import yaml

#: Default coefficient file, alongside this module.
COEFFICIENT_PATH = pathlib.Path(__file__).resolve().parent / 'trim_coefficients.yaml'


class TrimCalculator:
    """The fitted thermal-focus correction, read from a coefficient file.

    Parameters
    ----------
    path : `str` or `pathlib.Path`, optional
        Coefficient file to read. Defaults to `COEFFICIENT_PATH`.

    Attributes
    ----------
    coefficients : `dict`
        The coefficient file as parsed.
    intercept_um : `float`
        Response with every feature at zero [um of equivalent hexapod dz].
    truss_um_per_c : `float`
        Coefficient on the TMA truss temperature [um of equivalent hexapod dz per deg C].
    gradient_um_per_c_per_m : `dict`
        Coefficients on the four M1M3 bulk thermal gradients, keyed ``z``, ``y``, ``radial`` and
        ``x`` [um of equivalent hexapod dz per (deg C per m)].
    sample_feature_range : `dict`
        Full observed range of each feature over the fitted sample, as ``(low, high)``. Truss
        temperature in deg C, the four gradients in deg C per m.
    v1_per_um_dz : `float`
        v-mode-1 amplitude per um of total hexapod dz travel, split evenly between the camera and
        M2 hexapods [dimensionless per um]. The inverse is 1110.1 um of hexapod dz per unit v1.
    dz_um_per_um_wf : `float`
        Equivalent hexapod dz per um of wavefront defocus [um of equivalent hexapod dz per um of
        wavefront].
    v1_dof_um_per_unit : `dict`
        DOF content of one unit of v-mode-1 amplitude, keyed by DOF name [um per unit v1], at the
        50-DOF, 34-mode projection.
    v1_dof_labels : `tuple`
        ``(name, label, unit)`` for each entry of `v1_dof_um_per_unit`, in the order
        `predict_trim` reports them.
    """

    #: Feature names, in the order `predict_trim` takes them.
    FEATURES = ('truss_temp_c', 'z_gradient_c_per_m', 'y_gradient_c_per_m',
                'radial_gradient_c_per_m', 'x_gradient_c_per_m')

    def __init__(self, path=COEFFICIENT_PATH):
        with open(path) as handle:
            coefficients = yaml.safe_load(handle)

        self.path = pathlib.Path(path)
        self.coefficients = coefficients
        self.intercept_um = float(coefficients['intercept_um'])
        self.truss_um_per_c = float(coefficients['truss_um_per_c'])
        self.gradient_um_per_c_per_m = {axis: float(value) for axis, value
                                        in coefficients['gradient_um_per_c_per_m'].items()}
        self.sample_feature_range = {name: (float(lo), float(hi)) for name, (lo, hi)
                                     in coefficients['sample_feature_range'].items()}
        self.v1_per_um_dz = float(coefficients['v1_per_um_dz'])
        self.dz_um_per_um_wf = float(coefficients['dz_um_per_um_wf'])

        dof = coefficients['v1_dof']
        self.v1_dof_um_per_unit = {name: float(entry['um_per_unit'])
                                   for name, entry in dof.items()}
        self.v1_dof_labels = tuple((name, entry['label'], entry['unit'])
                                   for name, entry in dof.items())

    def __repr__(self):
        return f'{type(self).__name__}({str(self.path)!r})'

    def predict_trim(self, truss_temp_c, z_gradient_c_per_m, y_gradient_c_per_m,
                     radial_gradient_c_per_m, x_gradient_c_per_m, warn_extrapolation=True):
        """Predict the focus v-mode and the DOF Trim from the five thermal telemetry values.

        Parameters
        ----------
        truss_temp_c : `float` or `array_like`
            TMA truss temperature, the mean of the two thermometers [deg C].
        z_gradient_c_per_m, y_gradient_c_per_m, radial_gradient_c_per_m, x_gradient_c_per_m : \
                `float` or `array_like`
            M1M3 bulk thermal gradients [deg C per m].
        warn_extrapolation : `bool`, optional
            Issue a `UserWarning` for any input outside `sample_feature_range`. The prediction is
            returned either way.

        Returns
        -------
        v1 : `float` or `numpy.ndarray`
            Predicted v-mode-1 amplitude [dimensionless].
        v1_dz : `float` or `numpy.ndarray`
            The same prediction as focus [um of equivalent hexapod dz, split evenly between the
            camera and M2 hexapods].
        dof_dict : `dict`
            DOF Trim keyed by DOF name: ``M2_dz`` and ``Cam_dz`` the M2 and camera hexapod dz,
            ``B1_3`` the M1M3 bending mode B3 and ``B2_5`` the M2 bending mode B5, all [um].
        """
        vals = {'truss_temp_c': np.asarray(truss_temp_c, float),
                'z_gradient_c_per_m': np.asarray(z_gradient_c_per_m, float),
                'y_gradient_c_per_m': np.asarray(y_gradient_c_per_m, float),
                'radial_gradient_c_per_m': np.asarray(radial_gradient_c_per_m, float),
                'x_gradient_c_per_m': np.asarray(x_gradient_c_per_m, float)}
        if warn_extrapolation:
            for name, v in vals.items():
                lo, hi = self.sample_feature_range[name]
                if np.any(v < lo) or np.any(v > hi):
                    warnings.warn(f'{name} is outside the fitted range {lo:+.5f} to {hi:+.5f}; '
                                  f'the prediction is an extrapolation', UserWarning,
                                  stacklevel=2)
        grad = self.gradient_um_per_c_per_m
        v1_dz = (self.intercept_um
                 + self.truss_um_per_c * vals['truss_temp_c']
                 + grad['z'] * vals['z_gradient_c_per_m']
                 + grad['y'] * vals['y_gradient_c_per_m']
                 + grad['radial'] * vals['radial_gradient_c_per_m']
                 + grad['x'] * vals['x_gradient_c_per_m'])
        v1 = v1_dz * self.v1_per_um_dz

        scalar = v1_dz.ndim == 0
        dof_dict = {}
        for dof, unit_content in self.v1_dof_um_per_unit.items():
            val = unit_content * v1
            dof_dict[dof] = float(val) if scalar else val
        if scalar:
            return float(v1), float(v1_dz), dof_dict
        return v1, v1_dz, dof_dict


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--truss-temp-c', type=float, required=True,
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
    ap.add_argument('--coefficients', default=COEFFICIENT_PATH,
                    help=f'fitted coefficient file [default {COEFFICIENT_PATH.name} beside this '
                         f'module]')
    args = ap.parse_args()

    calc = TrimCalculator(args.coefficients)
    v1, v1_dz, dof_dict = calc.predict_trim(
        args.truss_temp_c, args.z_gradient_c_per_m, args.y_gradient_c_per_m,
        args.radial_gradient_c_per_m, args.x_gradient_c_per_m)

    print(f'predicted v1     {v1:+12.6f} (dimensionless)')
    print(f'predicted v1_dz  {v1_dz:+12.1f} um of equivalent hexapod dz')
    print('DOF Trim to apply')
    for dof, name, unit in calc.v1_dof_labels:
        print(f'  {dof:6s} {name:24s} {dof_dict[dof]:+12.6f} {unit}')


if __name__ == '__main__':
    main()
