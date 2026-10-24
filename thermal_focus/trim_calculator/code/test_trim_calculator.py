"""Tests for the standalone thermal-focus trim calculator.

Three checks, all reading their data from ``trim_test_cases.yaml``:

1. Each worked case reproduces its independently evaluated ``v1_dz``, which catches a mistyped
   coefficient in ``trim_coefficients.yaml``.
2. The two hexapod dz entries of the DOF back-projection sum to the total travel `TrimCalculator.v1_per_um_dz`
   inverts, which is the convention the fitted coefficients are in.
3. The array path agrees with the scalar path, element by element.

Invocation::

    python test_trim_calculator.py
    pytest test_trim_calculator.py
"""
import pathlib

import numpy as np
import yaml

from trim_calculator import TrimCalculator

#: Worked test cases, alongside this module.
TEST_CASE_PATH = pathlib.Path(__file__).resolve().parent / 'trim_test_cases.yaml'


def load_test_cases(path=TEST_CASE_PATH):
    """Read the worked test cases.

    Parameters
    ----------
    path : `str` or `pathlib.Path`, optional
        Test case file to read. Defaults to `TEST_CASE_PATH`.

    Returns
    -------
    tolerance_um : `float`
        Tolerance on each worked case [um of equivalent hexapod dz].
    cases : `tuple`
        Each entry ``(label, inputs, expected v1_dz in um of equivalent hexapod dz)``.
    """
    with open(path) as handle:
        data = yaml.safe_load(handle)
    cases = tuple((case['label'], dict(case['inputs']), float(case['expected_v1_dz_um']))
                  for case in data['test_cases'])
    return float(data['tolerance_um']), cases


def check_worked_cases(calc, cases, tol_um, verbose=True):
    """Check each worked case against its independently evaluated value.

    Parameters
    ----------
    calc : `trim_calculator.TrimCalculator`
        The calculator under test.
    cases : `tuple`
        As returned by `load_test_cases`.
    tol_um : `float`
        Tolerance on each case [um of equivalent hexapod dz].
    verbose : `bool`, optional
        Print each case.

    Returns
    -------
    max_abs_diff : `float`
        Largest disagreement over `cases` [um of equivalent hexapod dz].

    Raises
    ------
    AssertionError
        If any case disagrees by more than `tol_um`.
    """
    worst = 0.0
    for label, inputs, expected in cases:
        _, got, _ = calc.predict_trim(**inputs, warn_extrapolation=False)
        diff = abs(got - expected)
        worst = max(worst, diff)
        if verbose:
            print(f'  {label:44s} expected {expected:+9.2f}  got {got:+9.2f}  '
                  f'difference {diff:.3f} um of equivalent hexapod dz')
    assert worst < tol_um, (f'a coefficient is mistyped: worst case differs by {worst:.3f} um of '
                            f'equivalent hexapod dz, over the tolerance {tol_um}')
    if verbose:
        print(f'  all {len(cases)} cases agree to {worst:.3f} um of equivalent hexapod dz '
              f'(tolerance {tol_um})')
    return worst


def check_dof_split(calc, cases, verbose=True):
    """Check the DOF back-projection against the dz-equivalent convention.

    Parameters
    ----------
    calc : `trim_calculator.TrimCalculator`
        The calculator under test.
    cases : `tuple`
        As returned by `load_test_cases`. Only the first case is used.
    verbose : `bool`, optional
        Print the DOF Trim and the sum.

    Returns
    -------
    rel : `float`
        Relative disagreement between the two hexapod dz entries summed and the total travel
        [dimensionless, difference over total].

    Raises
    ------
    AssertionError
        If the two hexapod dz entries do not sum to the total travel `TrimCalculator.v1_per_um_dz` inverts.
    """
    label, inputs, _ = cases[0]
    v1, v1_dz, dof_dict = calc.predict_trim(**inputs, warn_extrapolation=False)
    total = dof_dict['M2_dz'] + dof_dict['Cam_dz']
    expect_total = -v1_dz
    rel = abs(total - expect_total) / max(abs(expect_total), 1e-12)
    if verbose:
        print(f'  DOF Trim on "{label}":')
        print(f'    v1 {v1:+.6f} (dimensionless), v1_dz {v1_dz:+.2f} um of '
              f'equivalent hexapod dz')
        for dof, name, unit in calc.v1_dof_labels:
            print(f'    {name:24s} {dof_dict[dof]:+12.6f} {unit}')
        print(f'    hexapod dz sum {total:+.4f} um against the total travel v1_dz implies '
              f'{expect_total:+.4f} um, relative disagreement {rel:.2e} (dimensionless)')
    assert rel < 1e-3, (f'the v-mode-1 DOF content is inconsistent with v1_per_um_dz: the two '
                        f'hexapod dz sum to {total:+.4f} um against {expect_total:+.4f} um '
                        f'expected, a relative disagreement of {rel:.2e}')
    return rel


def check_array_path(calc, cases, tol_um, verbose=True):
    """Check that array input gives the scalar numbers, element by element.

    Parameters
    ----------
    calc : `trim_calculator.TrimCalculator`
        The calculator under test.
    cases : `tuple`
        As returned by `load_test_cases`. The first two cases are stacked into arrays.
    tol_um : `float`
        Tolerance on each case [um of equivalent hexapod dz].
    verbose : `bool`, optional
        Print the outcome.

    Raises
    ------
    AssertionError
        If the array path disagrees with the scalar worked cases or the scalar DOF Trim.
    """
    names = ('truss_temp_c', 'z_gradient_c_per_m', 'y_gradient_c_per_m',
             'radial_gradient_c_per_m', 'x_gradient_c_per_m')
    pair = cases[:2]
    stacked = [np.array([inputs[name] for _, inputs, _ in pair], float) for name in names]
    _, arr_v1_dz, arr_dof = calc.predict_trim(*stacked, warn_extrapolation=False)
    assert np.allclose(arr_v1_dz, [expected for _, _, expected in pair], atol=tol_um), \
        'the array path disagrees with the scalar worked cases'
    _, _, dof_dict = calc.predict_trim(**pair[0][1], warn_extrapolation=False)
    assert np.allclose(arr_dof['M2_dz'][0], dof_dict['M2_dz']), \
        'the array DOF Trim disagrees with the scalar DOF Trim'
    if verbose:
        print(f'  the array path reproduces the scalar cases to {tol_um} um of equivalent '
              f'hexapod dz')


def self_test(calc=None, tol_um=None, verbose=True):
    """Run every check against a calculator's coefficients.

    Parameters
    ----------
    calc : `trim_calculator.TrimCalculator`, optional
        The calculator under test. Defaults to one built from the default coefficient file.
    tol_um : `float`, optional
        Tolerance on each worked case [um of equivalent hexapod dz]. Defaults to the
        ``tolerance_um`` in the test case file.
    verbose : `bool`, optional
        Print each case.

    Returns
    -------
    max_abs_diff : `float`
        Largest disagreement over the worked cases [um of equivalent hexapod dz].
    """
    if calc is None:
        calc = TrimCalculator()
    file_tol_um, cases = load_test_cases()
    if tol_um is None:
        tol_um = file_tol_um
    worst = check_worked_cases(calc, cases, tol_um, verbose=verbose)
    check_dof_split(calc, cases, verbose=verbose)
    check_array_path(calc, cases, tol_um, verbose=verbose)
    return worst


# ------------------------------------------------------------------------------- pytest entry points

def test_worked_cases():
    tol_um, cases = load_test_cases()
    check_worked_cases(TrimCalculator(), cases, tol_um, verbose=False)


def test_dof_split():
    _, cases = load_test_cases()
    check_dof_split(TrimCalculator(), cases, verbose=False)


def test_array_path():
    tol_um, cases = load_test_cases()
    check_array_path(TrimCalculator(), cases, tol_um, verbose=False)


if __name__ == '__main__':
    print('worked test cases [um of equivalent hexapod dz]')
    self_test()
