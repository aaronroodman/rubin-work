"""
Standalone JAX Optics+Atmosphere PSF moment model with autodiff.

    PSF(theta) = |FFT{ A(x) exp(i 2pi W(x)/lambda) }|^2   (x)   Atm(theta)

  * W = sum_j c_j Z_j(x)   -- Fraunhofer optical wavefront (Noll Zernikes).
  * Atm is applied as a differentiable MTF in the image-Fourier plane
    (Kolmogorov by default; von Karman is a drop-in), with 2 shape params.
  * Moments are computed under a FIXED elliptical-Gaussian weight (the HSM
    adaptive weight of the data star, supplied once), matching the estimator in
    moments_hsm.py / PIFF calculate_moments but held constant so the whole
    moment vector is a smooth, autodiff-able function of the model parameters.

Everything downstream of the parameter vector is jax; use jax.grad / jax.jacobian
on `moments(params)` for the fit.

Conventions match psf_model.py (Rubin: D=8.36 m, obsc=0.61, lambda=750 nm,
0.2 arcsec/pix) and galsim.OpticalPSF (circular Noll Zernikes, annular mask).
"""

import numpy as np
import galsim
import jax
import jax.numpy as jnp

import jax_hsm

jax.config.update("jax_enable_x64", True)

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0
RAD_PER_ARCSEC = 1.0 / ARCSEC_PER_RAD


class JaxOptAtmoPSF:
    def __init__(self, jmax=22, diam=8.36, obscuration=0.61, lam_nm=750.0,
                 pixel_scale=0.2, stamp=32, oversample=16, kernel='Kolmogorov',
                 L0=25.0, annular=False):
        """
        :param jmax:        max Noll index of the wavefront basis (Z1..Zjmax).
        :param stamp:       detector stamp size (pixels) at pixel_scale.
        :param oversample:  FFT oversampling B; detector pixel = B native pixels.
        :param annular:     use annular Noll Zernikes (R_inner=obscuration),
                            matching ts_wep / the MIW & CWFS convention; else
                            circular (galsim.OpticalPSF default).
        """
        self.jmax = jmax
        self.annular = annular
        self.diam = diam
        self.obscuration = obscuration
        self.lam_um = lam_nm * 1e-3
        self.pixel_scale = pixel_scale        # arcsec/detector pixel
        self.stamp = stamp
        self.B = oversample
        self.N = stamp * oversample           # native FFT grid
        self.kernel = kernel
        self.L0 = L0
        self.n_center_iter = 3                 # weighted-centroid iterations

        # native image angular pixel (rad); require pupil to fit: dtheta<=lam/D
        self.dtheta = (pixel_scale / oversample) * RAD_PER_ARCSEC  # rad
        # padded pupil extent so that dtheta = lam / (N*dx) = lam / D_pad
        D_pad = (self.lam_um * 1e-6) / self.dtheta
        assert D_pad >= diam, (f"oversample too small: D_pad={D_pad:.2f} < D={diam}"
                               f"; increase oversample")
        self.D_pad = D_pad
        dx = D_pad / self.N                    # pupil sample (m)

        # pupil-plane coordinate grid (m), centred
        ax = (np.arange(self.N) - self.N / 2) * dx
        xx, yy = np.meshgrid(ax, ax)
        r = np.hypot(xx, yy)
        aperture = ((r <= diam / 2) & (r >= obscuration * diam / 2)).astype(float)
        self.aperture = jnp.asarray(aperture)

        # circular Noll Zernike basis on the pupil (units: wavefront per unit
        # coefficient); matches galsim.OpticalPSF (R_inner=0, masked by aperture)
        u = xx / (diam / 2)
        v = yy / (diam / 2)
        R_inner = obscuration if annular else 0.0
        basis = galsim.zernike.zernikeBasis(jmax, u, v, R_outer=1.0,
                                            R_inner=R_inner)  # (jmax+1,N,N)
        self.Zbasis = jnp.asarray(basis) * self.aperture[None, :, :]

        # image-plane angular-frequency grid (cycles/rad) for the atmosphere MTF
        nu = np.fft.fftfreq(self.N, d=self.dtheta)   # cycles/rad
        nux, nuy = np.meshgrid(nu, nu)
        self.nux = jnp.asarray(nux)
        self.nuy = jnp.asarray(nuy)

        # detector-plane coordinate grid (arcsec), centred, for moments
        det = (np.arange(stamp) - stamp / 2 + 0.5) * pixel_scale
        dX, dY = np.meshgrid(det, det)
        self.detX = jnp.asarray(dX)
        self.detY = jnp.asarray(dY)

        if kernel == 'VonKarman':
            self._build_vonkarman_tables()

    def _build_vonkarman_tables(self):
        """Precompute (numpy, one-time) the galsim-calibrated VonKarman MTF.

        At fixed L0 the phase structure function scales exactly as r0^{-5/3},
        so D_ref(nu) = -2 ln[MTF(nu; r0_ref, L0)] (taken from galsim.VonKarman,
        exact) fully determines the kernel; r0 enters analytically and
        differentiably.  Also tabulate fwhm(r0) so the size parameter can be
        specified as an atmospheric FWHM.
        """
        self.r0_ref = 0.15
        nu_arcsec = np.linspace(0.0, 60.0, 4000)         # cycles/arcsec
        vk = galsim.VonKarman(lam=self.lam_um * 1e3, r0=self.r0_ref,
                              L0=self.L0, flux=1.0)
        mtf = np.array([vk.kValue(galsim.PositionD(2 * np.pi * f, 0.0)).real
                        for f in nu_arcsec])
        mtf = np.clip(mtf, 1e-12, 1.0)
        Dref = -2.0 * np.log(mtf)                          # >= 0, grows with nu
        self._vk_nu = jnp.asarray(nu_arcsec)
        self._vk_Dref = jnp.asarray(Dref)
        # fwhm(r0) table at this L0 for the size parametrisation
        r0s = np.linspace(0.05, 0.40, 60)
        fwhms = np.array([galsim.VonKarman(lam=self.lam_um * 1e3, r0=r,
                                           L0=self.L0).calculateFWHM()
                          for r in r0s])
        order = np.argsort(fwhms)
        self._vk_fwhm_tab = jnp.asarray(fwhms[order])
        self._vk_r0_tab = jnp.asarray(r0s[order])

    # ---------- optical PSF ----------
    def _optical_psf_native(self, zernikes):
        """|FFT(pupil field)|^2 on the native grid (fft-ordered, not shifted)."""
        # zernikes: length jmax+1 (index 0 unused), microns
        W = jnp.tensordot(zernikes, self.Zbasis, axes=(0, 0))   # microns
        phase = 2.0 * jnp.pi * W / self.lam_um
        field = self.aperture * jnp.exp(1j * phase)
        # ifft2 (rather than fft2) gives the image-plane parity that matches
        # galsim.OpticalPSF; |ifft2(f)|^2[k] ~ |fft2(f)|^2[-k].
        E = jnp.fft.ifft2(jnp.fft.ifftshift(field))
        psf = jnp.abs(E) ** 2
        return psf / jnp.sum(psf)

    # ---------- atmospheric MTF ----------
    def _atm_mtf(self, atm):
        """Atmospheric OTF on the (fft-ordered) frequency grid.

        atm = [fwhm_arcsec, g1, g2].
        """
        fwhm, g1, g2 = atm[0], atm[1], atm[2]
        # sheared frequency coordinate (ellipticity): apply (1+/-g) scaling
        nux = self.nux * (1.0 + g1) + self.nuy * g2
        nuy = self.nuy * (1.0 - g1) + self.nux * g2
        # floor nu^2 so the DC term (nu=0) has an exactly-zero shear gradient
        # (the fractional power of sqrt(nu^2) is otherwise singular there).
        nu2 = jnp.maximum(nux ** 2 + nuy ** 2, 1e-20)
        lam_m = self.lam_um * 1e-6
        if self.kernel == 'Kolmogorov':
            fwhm_rad = fwhm * RAD_PER_ARCSEC
            r0 = 0.9758634 * lam_m / fwhm_rad          # Kolmogorov FWHM relation
            H = jnp.exp(-3.44 * ((lam_m / r0) ** 2 * nu2) ** (5.0 / 6.0))
        elif self.kernel == 'VonKarman':
            # frequency in cycles/arcsec (galsim convention for the table)
            nu_arcsec = jnp.sqrt(nu2) / ARCSEC_PER_RAD
            Dref = jnp.interp(nu_arcsec, self._vk_nu, self._vk_Dref)
            r0 = jnp.interp(fwhm, self._vk_fwhm_tab, self._vk_r0_tab)
            D = Dref * (self.r0_ref / r0) ** (5.0 / 3.0)
            H = jnp.exp(-0.5 * D)
        else:
            raise NotImplementedError(self.kernel)
        return H

    # ---------- full PSF on detector ----------
    def psf(self, zernikes, atm):
        psf_native = self._optical_psf_native(zernikes)     # fft-ordered
        H = self._atm_mtf(atm)
        conv = jnp.fft.ifft2(jnp.fft.fft2(psf_native) * H).real
        conv = jnp.fft.fftshift(conv)                        # centre it
        conv = jnp.clip(conv, 0.0, None)
        # rebin native -> detector by summing B x B blocks
        B, S = self.B, self.stamp
        conv = conv.reshape(S, B, S, B).sum(axis=(1, 3))
        return conv / jnp.sum(conv)

    # ---------- adaptive (HSM-style) moments: matches galsim/PIFF ----------
    def moments_adaptive(self, zernikes, atm, n_iter=25):
        """Differentiable adaptive moments of the model PSF.

        Reproduces galsim FindAdaptiveMom + PIFF calculate_moments; the weight
        adapts to the PSF, so no external weight is needed and the result
        matches what the pipeline measures on real stars.
        """
        img = self.psf(zernikes, atm)
        mom, _ = jax_hsm.adaptive_moments(img, self.detX, self.detY,
                                          n_iter=n_iter)
        return mom

    def adaptive_weight(self, zernikes, atm, n_iter=25):
        """Return the converged (cu, cv, sigma, e1, e2) of the adaptive weight."""
        img = self.psf(zernikes, atm)
        cu, cv, Mxx, Mxy, Myy = jax_hsm.adaptive_weight(
            img, self.detX, self.detY, n_iter=n_iter)
        sigma = (Mxx * Myy - Mxy ** 2) ** 0.25
        T = Mxx + Myy
        return cu, cv, sigma, (Mxx - Myy) / T, 2 * Mxy / T

    # ---------- weighted moments under a fixed Gaussian weight ----------
    def moments(self, zernikes, atm, weight):
        """Weighted moments (arcsec^n) of the model PSF under a fixed weight.

        :param weight: dict/array [u0, v0, sigma, e1, e2] (arcsec) -- the HSM
                       adaptive weight of the data star.
        :returns: jnp array [e0,e1,e2, M21,M12,M30,M03, M22,M31,M13,M40,M04].
        """
        img = self.psf(zernikes, atm)
        u0, v0, sigma, we1, we2 = weight
        # elliptical Gaussian weight (fixed shape); inverse 2nd-moment matrix
        denom = jnp.sqrt(jnp.clip(1.0 - we1 ** 2 - we2 ** 2, 1e-8, None))
        Mxx = sigma ** 2 * (1 + we1) / denom
        Myy = sigma ** 2 * (1 - we1) / denom
        Mxy = sigma ** 2 * we2 / denom
        det = Mxx * Myy - Mxy ** 2
        iXX = Myy / det
        iYY = Mxx / det
        iXY = -Mxy / det

        # Recentre the weight on the image's weighted-mean centroid (fixed
        # shape), then take central moments about it.  This defines a single
        # self-consistent estimator applied identically to data and model
        # (see moments_fixedweight.measure_fixedweight_moments).  Unrolled ->
        # fully differentiable.
        cu, cv = u0, v0
        for _ in range(self.n_center_iter):
            u = self.detX - cu
            v = self.detY - cv
            W = jnp.exp(-0.5 * (iXX * u * u + 2 * iXY * u * v + iYY * v * v))
            WI = W * img
            s = jnp.sum(WI)
            cu = jnp.sum(WI * self.detX) / s
            cv = jnp.sum(WI * self.detY) / s

        u = self.detX - cu
        v = self.detY - cv
        W = jnp.exp(-0.5 * (iXX * u * u + 2 * iXY * u * v + iYY * v * v))
        WI = W * img
        norm = jnp.sum(WI)
        WI = WI / norm

        usq = u * u
        vsq = v * v
        uv = u * v
        rsq = usq + vsq
        usqmvsq = usq - vsq

        e0 = jnp.sum(WI * rsq)
        e1 = jnp.sum(WI * usqmvsq)
        e2 = 2.0 * jnp.sum(WI * uv)
        M21 = jnp.sum(WI * u * rsq)
        M12 = jnp.sum(WI * v * rsq)
        M30 = jnp.sum(WI * u * (usq - 3 * vsq))
        M03 = jnp.sum(WI * v * (3 * usq - vsq))
        M22 = jnp.sum(WI * rsq * rsq)
        M31 = jnp.sum(WI * rsq * usqmvsq)
        M13 = 2.0 * jnp.sum(WI * rsq * uv)
        M40 = jnp.sum(WI * (usq * usq - 6 * usq * vsq + vsq * vsq))
        M04 = 4.0 * jnp.sum(WI * usqmvsq * uv)
        return jnp.array([e0, e1, e2, M21, M12, M30, M03,
                          M22, M31, M13, M40, M04])


MOMENT_LABELS = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03',
                 'M22', 'M31', 'M13', 'M40', 'M04']
