"""Wall-clock comparison of batoid ``fftPSF`` and the Debye-weighted Huygens sum.

Both methods are timed on the same entrance-pupil ray grid and the same output
lattice (the ``fftPSF`` lattice), for the Rubin r-band design on axis:

- in focus, 256^2 and 512^2 pupil grids, ``pad_factor`` 2;
- camera pistoned +1.5 mm (extra-focal donut), 2560^2 pupil grid,
  ``pad_factor`` 1, ``sphereRadius`` 20 m.

The Debye-Huygens time is split into the ray trace, the Debye weights, and the
two type-1 non-uniform FFTs (field and energy flux). Each step is the best of
three runs. The agreement of the two images, sum |difference| as a fraction of
total flux, is printed as a check.

Run with ``/opt/local/bin/python3 code/fresnel/time_fftpsf_vs_debye.py`` from
``optics/``. Prints only; writes no files.
"""

import time

import numpy as np

import fresnel_donut as fd     # imports finufft before batoid (OpenMP order)
import batoid

WAVELENGTH = 622e-9    # m
N_REPEAT = 3


def best_time(func):
    """Minimum wall-clock time (s) over `N_REPEAT` calls, and the last result."""
    times, out = [], None
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        out = func()
        times.append(time.perf_counter() - t0)
    return min(times), out


def compare(label, optic, nx, pad_factor, sphere_radius):
    """Time fftPSF and Debye-Huygens for one optic and print the results.

    Parameters
    ----------
    label : `str`
        Case name for the printout.
    optic : `batoid.Optic`
        Optic to image, on axis.
    nx : `int`
        Pupil ray-grid size.
    pad_factor : `int`
        ``fftPSF`` padding factor.
    sphere_radius : `float` or `None`
        ``fftPSF`` reference-sphere radius, m; None uses the optic's value.
    """
    print(f"\n== {label}: nx={nx}, pad_factor={pad_factor}, "
          f"sphereRadius={sphere_radius} m")
    fd.fftpsf_samples(optic, WAVELENGTH, 64, pad_factor=pad_factor,
                      sphere_radius=sphere_radius)    # warm-up
    t_wf, _ = best_time(lambda: batoid.analysis.wavefront(
        optic, 0.0, 0.0, WAVELENGTH, nx=nx, sphereRadius=sphere_radius))
    t_fft, (x, y, value, spacing) = best_time(lambda: fd.fftpsf_samples(
        optic, WAVELENGTH, nx, pad_factor=pad_factor,
        sphere_radius=sphere_radius))
    n = int(round(np.sqrt(len(x))))
    print(f"fftPSF total {t_fft:7.2f} s  (wavefront trace {t_wf:.2f} s; "
          f"lattice {n}x{n}, spacing {spacing*1e6:.3f} um)")

    t_trace, rays = best_time(lambda: fd.trace_pupil_grid(optic, WAVELENGTH,
                                                          nx))
    t_weights, (w, g) = best_time(lambda: fd.debye_weights(rays, nx))
    t_nufft, image = best_time(lambda: fd.huygens_on_lattice(
        rays, WAVELENGTH, x.reshape(n, n), y.reshape(n, n), weights=w,
        gamma=g))
    total = t_trace + t_weights + t_nufft
    print(f"Debye-Huygens total {total:7.2f} s  (trace {t_trace:.2f} s, "
          f"weights {t_weights:.2f} s, NUFFT x2 {t_nufft:.2f} s)")
    diff = np.abs(value.reshape(n, n) - image).sum()
    print(f"   sum|fftPSF - Debye-Huygens| = {diff:.2e} of total flux")


def main():
    telescope = batoid.Optic.fromYaml('LSST_r.yaml')
    for nx in (256, 512):
        compare('in focus', telescope, nx, 2, None)
    donut = fd.shift_camera(telescope, 1.5e-3)
    compare('+1.5 mm camera piston', donut, 2560, 1, 20.0)

    # The same donut on a pixel-commensurate 0.5 um grid, 3200^2 samples.
    rays = fd.trace_pupil_grid(donut, WAVELENGTH, 2560)
    w, g = fd.debye_weights(rays, 2560)
    center = fd.chief_ray_point(donut, WAVELENGTH)
    t, _ = best_time(lambda: fd.huygens_image(rays, WAVELENGTH, center, 0.5e-6,
                                              3200, weights=w, gamma=g))
    print(f"\nDebye-Huygens NUFFT only, 0.5 um x 3200^2 grid: {t:.2f} s")


if __name__ == '__main__':
    main()
