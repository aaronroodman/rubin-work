"""Diffraction models of defocused ("donut") star images from batoid ray traces.

Four image models are computed from the same batoid optic so that they can be
compared directly:

- geometric: random rays traced to the detector and histogrammed;
- ``batoid.analysis.fftPSF``: FFT of the exit-pupil wavefront (Fraunhofer with the
  exact traced defocus phase, uniform pupil amplitude, linearised pupil-to-k map);
- Huygens (Debye) sum ``U(x) = sum_j w_j exp(i[k_j.(x - r_j) + k0 t_j])`` over the
  traced rays, which is what ``batoid.analysis.huygensPSF`` evaluates one pixel at a
  time. Here it is evaluated with a type-1 non-uniform FFT (FINUFFT), which is exact
  to the requested tolerance and ~1e6 times faster (ratio of wall-clock times).
  ``w_j = 1`` reproduces batoid; ``w_j = sqrt(dOmega/d^2u)`` is the
  energy-conserving Debye-Wolf weight;
- semi-analytic references for an on-axis annular paraboloid (a perfect converging
  spherical wave): the non-paraxial scalar Debye integral, the paraxial
  Fresnel (Lommel) integral, and the analytic geometric irradiance.

Units: lengths in metres, wavelengths in metres, angles in radians, unless a
name says otherwise. Images are irradiance on the detector plane, normalised to
unit total flux (dimensionless fraction of total flux per sample or per pixel).
"""

import os
import time

# The PyPI wheels of batoid and finufft each vendor their own libomp on macOS;
# loading both aborts unless duplicate runtimes are allowed. The NUFFT results
# are checked point by point against batoid's own sumAmplitude in the notebooks.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from scipy.special import j0

# finufft must be imported (and its OpenMP runtime initialised) before
# batoid: in the other order, multi-threaded finufft calls segfault once
# batoid has run a threaded trace (macOS, PyPI wheels).
import finufft
import batoid


# ---------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------

def make_parabola(focal_length=10.31, diameter=8.36, obscuration=0.612,
                  defocus=0.0):
    """On-axis annular paraboloid mirror with a flat detector.

    On axis a paraboloid is stigmatic, so the reflected beam is a perfect
    spherical wave converging on the focus: the ideal test case for diffraction
    codes. The default focal ratio, diameter, and obscuration match Rubin
    (f/1.234, 8.36 m, 0.612).

    Parameters
    ----------
    focal_length : `float`
        Mirror focal length in m (vertex radius of curvature is twice this).
    diameter : `float`
        Outer clear-aperture diameter in m.
    obscuration : `float`
        Inner/outer clear-aperture diameter ratio (dimensionless).
    defocus : `float`
        Detector displacement from focus along the reflected beam, in m.
        Positive is beyond focus (extra-focal).

    Returns
    -------
    optic : `batoid.CompoundOptic`
    """
    radius_out = diameter / 2
    m1 = batoid.Mirror(
        batoid.Paraboloid(2 * focal_length), name='M1',
        obscuration=batoid.ObscNegation(
            batoid.ObscAnnulus(obscuration * radius_out, radius_out)))
    det = batoid.Detector(
        batoid.Plane(), name='Detector',
        coordSys=batoid.CoordSys(origin=[0, 0, focal_length + defocus]))
    return batoid.CompoundOptic(
        [m1, det], name='Parabola', backDist=15.0,
        stopSurface=batoid.Interface(batoid.Plane()),
        sphereRadius=focal_length + defocus, pupilSize=diameter,
        pupilObscuration=obscuration)


def shift_detector(optic, defocus, detector='LSST.LSSTCamera.Detector'):
    """Return `optic` with only the detector moved `defocus` m along its z axis.

    For the Rubin prescriptions in batoid the beam travels toward +z at the
    detector, so positive `defocus` is extra-focal and negative is intra-focal.
    This is the corner-wavefront-sensor geometry (sensor offset, lenses fixed),
    not the full-array-mode camera piston.
    """
    return optic.withGloballyShiftedOptic(detector, [0.0, 0.0, defocus])


def shift_camera(optic, defocus, camera='LSST.LSSTCamera'):
    """Return `optic` with the whole camera moved `defocus` m along global z.

    This is the full-array-mode (FAM) geometry: lenses, filter and detector
    move together (camera hexapod piston). For the Rubin prescriptions in
    batoid the beam travels toward +z at the camera, so positive `defocus` is
    extra-focal and negative is intra-focal.
    """
    return optic.withGloballyShiftedOptic(camera, [0.0, 0.0, defocus])


def dircos(theta_x, theta_y):
    """Direction cosines for field angle (rad), postel projection."""
    return batoid.utils.fieldToDirCos(theta_x, theta_y, projection='postel')


def chief_ray_point(optic, wavelength, theta_x=0.0, theta_y=0.0):
    """Chief-ray (stop-centre) intersection with the detector, local coords, m.

    The ray is inside the central obscuration and is flagged vignetted; its
    position is still the geometric chief-ray position.
    """
    ray = batoid.RayVector.fromStop(0.0, 0.0, optic=optic,
                                    wavelength=wavelength,
                                    dirCos=dircos(theta_x, theta_y))
    optic.trace(ray)
    return np.array([ray.x[0], ray.y[0]])


def shift_hexapods(optic, piston, camera='LSST.LSSTCamera', m2='LSST.M2'):
    """Return `optic` with the camera and M2 both moved `piston` m along global z.

    For the Rubin prescriptions in batoid, negative `piston` is intra-focal and
    positive is extra-focal; 4 mm on both hexapods gives about the defocus of an
    8 mm camera piston.
    """
    return optic.withGloballyShiftedOptic(camera, [0.0, 0.0, piston]) \
        .withGloballyShiftedOptic(m2, [0.0, 0.0, piston])


def turned_down_edge(depth, width, r_edge=2.558, half_size=4.25, spacing=5e-3):
    """Bicubic surface perturbation for a turned-down inner edge of M1.

    The perturbation is ``-depth * exp(-(r - r_edge) / width)`` for
    ``r >= r_edge`` and ``-depth`` inside the edge, in the surface's own z
    (negative is away from the incoming light). It steepens the concave surface
    near the edge, so reflected rays there are deflected toward the axis by up
    to ``2 * depth / width`` rad. Add it with
    ``optic.withPerturbedSurface('M1', turned_down_edge(...))``.

    Parameters
    ----------
    depth : `float`
        Surface depression at the edge, m.
    width : `float`
        Exponential roll-off scale outward from the edge, m.
    r_edge : `float`
        Edge radius, m (the M1 inner clear-aperture radius by default).
    half_size : `float`
        Half-width of the square interpolation grid, m.
    spacing : `float`
        Grid spacing, m; must be well below `width`.

    Returns
    -------
    surface : `batoid.Bicubic`
    """
    xs = np.arange(-half_size, half_size + spacing / 2, spacing)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    dz = -depth * np.exp(-np.clip(r - r_edge, 0.0, None) / width)
    return batoid.Bicubic(xs, xs, dz)


def accumulate_donut(optic, wavelength, n_rays, center, sample, npix, r_edges,
                     seed=1, batch=4_000_000):
    """On-axis geometric donut from rays drawn uniformly over the entrance pupil.

    The same `seed`, `n_rays` and `batch` give the same pupil samples, so two
    optics traced this way differ only through the optics.

    Parameters
    ----------
    center : `array_like`
        Image centre (x, y) on the detector, m.
    sample : `float`
        Image sample size, m.
    npix : `int`
        Image size, samples per side.
    r_edges : `numpy.ndarray`
        Edges of the radial annuli about `center`, m.

    Returns
    -------
    image : `numpy.ndarray`
        Ray counts per sample [y, x], normalised to unit sum.
    counts : `numpy.ndarray`
        Ray counts per annulus.
    radii : `numpy.ndarray`
        0.01 and 99.99 percentile ray radii of the first batch, m.
    """
    rng = np.random.default_rng(seed)
    half = optic.pupilSize / 2
    edges = (np.arange(npix + 1) - npix / 2) * sample
    image = np.zeros((npix, npix))
    counts = np.zeros(len(r_edges) - 1)
    radii = None
    remaining = n_rays
    while remaining > 0:
        n = min(batch, remaining)
        u = rng.uniform(-half, half, n)
        v = rng.uniform(-half, half, n)
        rv = batoid.RayVector.fromStop(u, v, optic=optic, wavelength=wavelength,
                                       dirCos=dircos(0.0, 0.0))
        optic.trace(rv)
        ok = ~rv.vignetted & ~rv.failed
        dx = rv.x[ok] - center[0]
        dy = rv.y[ok] - center[1]
        image += np.histogram2d(dy, dx, bins=[edges, edges])[0]
        r = np.hypot(dx, dy)
        counts += np.histogram(r, r_edges)[0]
        if radii is None:
            radii = np.percentile(r, [0.01, 99.99])
        remaining -= n
    return image / image.sum(), counts, radii


# ---------------------------------------------------------------------------
# Ray grids and Debye weights
# ---------------------------------------------------------------------------

def trace_pupil_grid(optic, wavelength, nx, theta_x=0.0, theta_y=0.0):
    """Trace an `nx` x `nx` square grid of rays filling the entrance pupil.

    Returns the traced `batoid.RayVector` in the detector's local frame. The
    ray order is the `nx` x `nx` grid flattened, which `debye_weights` relies on.
    """
    rays = batoid.RayVector.asGrid(optic=optic, wavelength=wavelength,
                                   dirCos=dircos(theta_x, theta_y), nx=nx)
    optic.trace(rays)
    return rays


def debye_weights(rays, nx):
    """Energy-conserving amplitude weights for a uniform entrance-pupil grid.

    In the Debye-Wolf plane-wave representation each ray carries amplitude
    ``a ~ sqrt(dP/dOmega)``, and the integral over directions discretises to
    ``a dOmega``. With uniform power per unit entrance-pupil area, the weight
    per ray is therefore ``sqrt(dOmega / d^2u)``, where ``dOmega = dalpha dbeta /
    gamma`` and (alpha, beta, gamma) are the ray direction cosines in the
    detector frame. The Jacobian is evaluated by finite differences on the
    pupil grid (vignetted rays are still traced by batoid, so the grid is
    complete).

    Returns
    -------
    weight : `numpy.ndarray`
        Relative amplitude weight per ray, flattened grid order, normalised to
        a mean of 1 over unvignetted rays (dimensionless).
    gamma : `numpy.ndarray`
        Direction cosine along the detector normal per ray (dimensionless).
    """
    v = np.stack([rays.vx, rays.vy, rays.vz])
    v = v / np.sqrt(np.sum(v**2, axis=0))
    alpha = v[0].reshape(nx, nx)
    beta = v[1].reshape(nx, nx)
    gamma = v[2]
    da_d0, da_d1 = np.gradient(alpha)
    db_d0, db_d1 = np.gradient(beta)
    jac = np.abs(da_d0 * db_d1 - da_d1 * db_d0).ravel()
    weight = np.sqrt(jac / np.abs(gamma))
    good = ~rays.vignetted & ~rays.failed & np.isfinite(weight)
    weight = weight / np.mean(weight[good])
    return weight, np.abs(gamma)


# ---------------------------------------------------------------------------
# Huygens / Debye sum via NUFFT
# ---------------------------------------------------------------------------

def _ray_coefficients(rays, wavelength, center, weights):
    """Complex coefficients and scaled k for the plane-wave sum on the detector."""
    good = ~rays.vignetted & ~rays.failed
    vx, vy, vz = rays.vx[good], rays.vy[good], rays.vz[good]
    vsq = vx**2 + vy**2 + vz**2        # |v| = 1/n
    k0 = 2 * np.pi / wavelength
    kx, ky, kz = k0 * vx / vsq, k0 * vy / vsq, k0 * vz / vsq
    t = rays.t[good]
    # Same phase convention as batoid RayVector.amplitude:
    #   exp(i[k.(r - r_j) - k0 (T - t_j)]),  r on the detector plane (z = 0).
    phase = (kx * (center[0] - rays.x[good]) + ky * (center[1] - rays.y[good])
             - kz * rays.z[good] + k0 * (t - np.median(t)))
    amp = np.ones_like(phase) if weights is None else weights[good]
    return kx, ky, amp * np.exp(1j * phase), good


def huygens_image(rays, wavelength, center, dx, npix, weights=None,
                  gamma=None, eps=1e-10):
    """Huygens/Debye irradiance on an `npix` x `npix` grid via a type-1 NUFFT.

    Evaluates ``U(x_m) = sum_j c_j exp(i k_j . x_m)`` with ``x_m = center +
    (m + 1/2) dx``, m = -npix/2 ... npix/2-1, which is a type-1 non-uniform FFT
    with non-uniform points ``k_j dx``. The half-sample offset puts the samples
    symmetrically about `center` and never on a pixel boundary that is a
    multiple of `dx` from it.

    Parameters
    ----------
    rays : `batoid.RayVector`
        Rays traced to the detector (local frame).
    wavelength : `float`
        Vacuum wavelength in m.
    center : `array_like`
        Grid centre (x, y) on the detector, m.
    dx : `float`
        Grid spacing on the detector, m. Must satisfy dx < lambda / (2 sin
        theta_max) so that ``|k_j dx| < pi``.
    npix : `int`
        Grid size (even).
    weights : `numpy.ndarray` or `None`
        Per-ray amplitude weights (flattened grid order). None gives unit
        weights, i.e. exactly what ``batoid.analysis.huygensPSF`` sums.
    gamma : `numpy.ndarray` or `None`
        Per-ray direction cosine along the detector normal. If given, the
        returned irradiance is the scalar energy flux through the detector
        plane, ``Re(U* dU/dz) / k0 = Re(U* V)`` with ``V = sum c_j gamma_j e^..``.
        If None, the returned image is ``|U|^2``.
    eps : `float`
        FINUFFT relative tolerance (dimensionless).

    Returns
    -------
    image : `numpy.ndarray`
        Irradiance [y, x], normalised to unit sum (fraction of flux per sample).
    xs, ys : `numpy.ndarray`
        Sample coordinates on the detector, m.
    """
    origin = np.asarray(center[:2], dtype=float) + 0.5 * dx
    kx, ky, c, good = _ray_coefficients(rays, wavelength, origin, weights)
    X, Y = kx * dx, ky * dx
    if np.max(np.abs(X)) >= np.pi or np.max(np.abs(Y)) >= np.pi:
        raise ValueError("dx too coarse: |k dx| >= pi")
    if gamma is None:
        U = finufft.nufft2d1(X, Y, c, (npix, npix), eps=eps, isign=1)
        image = np.abs(U.T)**2
    else:
        cc = np.stack([c, c * gamma[good]])
        U, V = finufft.nufft2d1(X, Y, cc, (npix, npix), eps=eps, isign=1)
        image = np.real(np.conj(U.T) * V.T)
    m = np.arange(npix) - npix // 2
    xs = origin[0] + m * dx
    ys = origin[1] + m * dx
    return image / image.sum(), xs, ys


def batoid_sum_amplitude(rays, points_xy, time_ref=None):
    """Batoid's own Huygens sum at a set of detector points (slow reference).

    Loops over points calling `batoid.RayVector.sumAmplitude`, exactly as
    ``batoid.analysis.huygensPSF`` does.

    Returns
    -------
    amplitude : `numpy.ndarray` of `complex`
    seconds_per_point : `float`
        Wall-clock time per evaluation point, s.
    """
    if time_ref is None:
        time_ref = np.median(rays.t[~rays.vignetted])
    out = np.empty(len(points_xy), dtype=complex)
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(points_xy):
        out[i] = rays.sumAmplitude(np.array([x, y, 0.0]), time_ref)
    return out, (time.perf_counter() - t0) / len(points_xy)


def nufft_amplitude_at(rays, wavelength, points_xy, weights=None,
                       time_ref=None, eps=1e-12):
    """Type-3 NUFFT of the same sum at arbitrary points, with batoid's phase.

    Used to verify `huygens_image` against `batoid_sum_amplitude` point by
    point (same time reference, so absolute phases are comparable).
    """
    good = ~rays.vignetted & ~rays.failed
    if time_ref is None:
        time_ref = np.median(rays.t[good])
    vx, vy, vz = rays.vx[good], rays.vy[good], rays.vz[good]
    vsq = vx**2 + vy**2 + vz**2
    k0 = 2 * np.pi / wavelength
    kx, ky, kz = k0 * vx / vsq, k0 * vy / vsq, k0 * vz / vsq
    phase = (-kx * rays.x[good] - ky * rays.y[good] - kz * rays.z[good]
             - k0 * (time_ref - rays.t[good]))
    amp = np.ones_like(phase) if weights is None else weights[good]
    c = amp * np.exp(1j * phase)
    pts = np.asarray(points_xy)
    return finufft.nufft2d3(kx, ky, c, pts[:, 0], pts[:, 1], eps=eps, isign=1)


# ---------------------------------------------------------------------------
# batoid fftPSF and geometric images, as samples
# ---------------------------------------------------------------------------

def fftpsf_samples(optic, wavelength, nx, pad_factor=1, theta_x=0.0,
                   theta_y=0.0, sphere_radius=None):
    """Run ``batoid.analysis.fftPSF`` and return its samples as coordinates.

    `sphere_radius` (m) is the reference-sphere radius passed to batoid; None
    uses the optic's ``sphereRadius`` (5 m for the Rubin yaml files).

    Returns
    -------
    x, y : `numpy.ndarray`
        Sample positions relative to the chief-ray point, m (flattened).
    value : `numpy.ndarray`
        Irradiance samples, normalised to unit sum (flattened).
    spacing : `float`
        Lattice spacing, m.
    """
    psf = batoid.analysis.fftPSF(optic, theta_x, theta_y, wavelength,
                                 nx=nx, pad_factor=pad_factor,
                                 sphereRadius=sphere_radius)
    # Same array/coordinate convention as batoid.analysis.huygensPSF.
    x = psf.coords[..., 0].T.ravel()
    y = psf.coords[..., 1].T.ravel()
    value = psf.array.ravel()
    spacing = np.sqrt(np.abs(np.linalg.det(psf.primitiveVectors)))
    return x, y, value / value.sum(), spacing


def huygens_on_lattice(rays, wavelength, x, y, weights=None, gamma=None,
                       eps=1e-10):
    """Huygens image on a square lattice given as sample coordinates.

    `x`, `y` are [y, x]-ordered coordinate arrays of a uniform square lattice
    (for example the `fftpsf_samples` output reshaped), possibly with negative
    spacing. Used to compare `fftPSF` with the Huygens sum sample by sample, so
    that neither is affected by re-binning.

    Returns
    -------
    image : `numpy.ndarray`
        Irradiance on the lattice [y, x], unit sum.
    """
    n = x.shape[1]
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    if not np.isclose(dx, dy, rtol=1e-6) or x.shape[0] != n:
        raise ValueError("lattice must be square with equal spacing")
    # huygens_image samples at center + (m + 1/2) dx, m = -n/2 ... n/2-1
    center = np.array([x[0, 0], y[0, 0]]) + (n // 2 - 0.5) * dx
    image, xs, ys = huygens_image(rays, wavelength, center, dx, n,
                                  weights=weights, gamma=gamma, eps=eps)
    if (np.max(np.abs(xs - x[0])) > 1e-3 * abs(dx)
            or np.max(np.abs(ys - y[:, 0])) > 1e-3 * abs(dx)):
        raise ValueError("lattice reconstruction failed")
    return image


def geometric_positions(optic, wavelength, nrays, theta_x=0.0, theta_y=0.0,
                        seed=0, batch=4_000_000):
    """Detector positions of rays drawn uniformly over the entrance pupil.

    Returns positions (m, detector local frame) of the unvignetted rays.
    """
    rng = np.random.default_rng(seed)
    half = optic.pupilSize / 2
    xs, ys = [], []
    remaining = nrays
    while remaining > 0:
        n = min(batch, remaining)
        u = rng.uniform(-half, half, n)
        v = rng.uniform(-half, half, n)
        rv = batoid.RayVector.fromStop(u, v, optic=optic,
                                       wavelength=wavelength,
                                       dirCos=dircos(theta_x, theta_y))
        optic.trace(rv)
        ok = ~rv.vignetted & ~rv.failed
        xs.append(rv.x[ok])
        ys.append(rv.y[ok])
        remaining -= n
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Binning and profiles
# ---------------------------------------------------------------------------

def bin_to_pixels(x, y, value, center, pixel, npix):
    """Sum samples into an `npix` x `npix` pixel image [y, x], unit total.

    Parameters
    ----------
    x, y : `numpy.ndarray`
        Sample positions, m.
    value : `numpy.ndarray` or `None`
        Sample weights (None for unit-weight rays).
    center : `array_like`
        Image centre, m.
    pixel : `float`
        Pixel size, m.
    """
    edges_x = center[0] + (np.arange(npix + 1) - npix / 2) * pixel
    edges_y = center[1] + (np.arange(npix + 1) - npix / 2) * pixel
    img, _, _ = np.histogram2d(y, x, bins=[edges_y, edges_x], weights=value)
    return img / img.sum()


def radial_profile(x, y, value, center, r_edges):
    """Azimuthally averaged surface brightness in annuli.

    Returns flux per unit area in each annulus, normalised so that
    ``sum(profile * annulus_area) = 1`` over the annuli (units 1/m^2).
    """
    r = np.hypot(x - center[0], y - center[1])
    flux, _ = np.histogram(r, bins=r_edges, weights=value)
    area = np.pi * np.diff(r_edges**2)
    return flux / area / flux.sum()


def pixel_radial_profile(image, pixel, center_index=None):
    """Azimuthal profile of a pixelised image: mean pixel value per annulus.

    Use this, not `radial_profile`, for gridded images. Dividing the flux in an
    annulus by its geometric area (as `radial_profile` does for scattered
    samples) imprints the ring-to-ring fluctuation in the number of pixel
    centres per annulus (~4% rms for annuli one pixel wide) as spurious ringing.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image [y, x] of odd size, centred on the central pixel unless
        `center_index` is given.
    pixel : `float`
        Pixel size, m; annuli are one pixel wide.

    Returns
    -------
    r : `numpy.ndarray`
        Annulus mid radii, m.
    profile : `numpy.ndarray`
        Mean pixel value / pixel area, for an image normalised to unit total
        flux (1/m^2).
    """
    n = image.shape[0]
    ci = n // 2 if center_index is None else center_index
    c = (np.arange(n) - ci) * pixel
    X, Y = np.meshgrid(c, c)
    r = np.hypot(X, Y).ravel()
    edges = np.arange(0, ci * pixel, pixel)
    cnt, _ = np.histogram(r, edges)
    tot, _ = np.histogram(r, edges, weights=(image / image.sum()).ravel())
    return 0.5 * (edges[1:] + edges[:-1]), tot / np.maximum(cnt, 1) / pixel**2


def normalise_profile(r, profile):
    """Normalise a sampled radial profile so that int profile 2 pi r dr = 1."""
    return profile / np.trapezoid(profile * 2 * np.pi * r, r)


# ---------------------------------------------------------------------------
# Semi-analytic references for the annular paraboloid
# ---------------------------------------------------------------------------

def _gauss_nodes(a, b, n_panels, n_gauss):
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    edges = np.linspace(a, b, n_panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * xg[None, :]).ravel()
    weights = (half[:, None] * wg[None, :]).ravel()
    return nodes, weights


def parabola_debye_profile(r, wavelength, focal_length, diameter, obscuration,
                           defocus, n_panels=1500, n_gauss=32, chunk=64):
    """Non-paraxial scalar Debye irradiance of the annular paraboloid.

    For a perfect spherical wave with uniform power per unit entrance-pupil
    area (pupil height h, ray angle theta with tan(theta/2) = h / 2f),

    ``U(r) = int w(h) exp(i k dz cos theta) J0(k r sin theta) h dh``
    ``V(r) = int w(h) cos theta exp(...) J0(...) h dh``
    ``E(r) = Re(U* V)``  (scalar energy flux through the detector plane),

    with ``w = sqrt(dOmega / d^2u) = cos^2(theta/2) / f``.

    Parameters
    ----------
    r : `numpy.ndarray`
        Radii on the detector, m.
    defocus : `float`
        Detector distance from focus, m.

    Returns
    -------
    E : `numpy.ndarray`
        Irradiance normalised to ``int E 2 pi r dr = 1`` over `r` (1/m^2).
    """
    k = 2 * np.pi / wavelength
    h, gw = _gauss_nodes(obscuration * diameter / 2, diameter / 2,
                         n_panels, n_gauss)
    theta = 2 * np.arctan(h / (2 * focal_length))
    base = np.cos(theta / 2)**2 * np.exp(1j * k * defocus * np.cos(theta)) \
        * h * gw
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    E = np.empty(len(r))
    for i in range(0, len(r), chunk):
        J = j0(k * np.outer(r[i:i + chunk], sin_t))
        U = J @ base
        V = J @ (base * cos_t)
        E[i:i + chunk] = np.real(np.conj(U) * V)
    return normalise_profile(r, E)


def parabola_paraxial_profile(r, wavelength, focal_length, diameter,
                              obscuration, defocus, n_panels=1500,
                              n_gauss=32, chunk=64):
    """Paraxial Fresnel (Lommel) irradiance of a uniform annular pupil.

    ``U(r) = int exp(-i k dz h^2 / 2f^2) J0(k r h / f) h dh``, ``E = |U|^2``.
    This is what every paraxial Fresnel or Fraunhofer-plus-defocus code
    computes for this geometry. Normalised as `parabola_debye_profile`.
    """
    k = 2 * np.pi / wavelength
    h, gw = _gauss_nodes(obscuration * diameter / 2, diameter / 2,
                         n_panels, n_gauss)
    base = np.exp(-1j * k * defocus * h**2 / (2 * focal_length**2)) * h * gw
    E = np.empty(len(r))
    for i in range(0, len(r), chunk):
        U = j0(k * np.outer(r[i:i + chunk], h / focal_length)) @ base
        E[i:i + chunk] = np.abs(U)**2
    return normalise_profile(r, E)


def parabola_geometric_profile(r, focal_length, diameter, obscuration,
                               defocus):
    """Analytic geometric irradiance of the defocused annular paraboloid.

    Rays at pupil height h cross the detector at ``rho = |dz| tan theta(h)``,
    so ``E(rho) = h dh / (rho drho)`` for uniform pupil illumination.
    Normalised as `parabola_debye_profile`.
    """
    h = np.linspace(obscuration * diameter / 2, diameter / 2, 20001)
    theta = 2 * np.arctan(h / (2 * focal_length))
    rho = np.abs(defocus) * np.tan(theta)
    E_h = h * np.gradient(h) / (rho * np.gradient(rho))
    E = np.interp(r, rho, E_h, left=0.0, right=0.0)
    return normalise_profile(r, E)


def parabola_donut_radii(focal_length, diameter, obscuration, defocus):
    """Geometric inner and outer donut radii, exact and paraxial, in m."""
    h = np.array([obscuration, 1.0]) * diameter / 2
    theta = 2 * np.arctan(h / (2 * focal_length))
    exact = np.abs(defocus) * np.tan(theta)
    paraxial = np.abs(defocus) * h / focal_length
    return exact, paraxial
