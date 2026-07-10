"""
Measured Intrinsic Wavefront (MIW) from the official ip_isr ``intrinsicZernikes``
calibration in the Butler (RUN ON USDF -- needs Butler + ip_isr).

Reconstructs the total intrinsic wavefront Zernike vector in the OCS frame using
the calib object's OWN interpolators (scipy LinearNDInterpolator, NaN outside the
sample hull -- no re-derivation, no parquet copy):

  * OCS (telescope-fixed, shared across detectors): ``calib.interpolator_ocs``
  * CCS (camera-fixed, per detector; CCD focal-plane height folded into Z4):
    the star's OWN detector ``calib.interpolator``, evaluated strictly within
    that CCD's footprint.

Combined with the spin-phase convention (from ts_intrinsic_wavefront, s=+1,
CCS = R(-theta) OCS): for a spin-|m| doublet Z_cos + i Z_sin,

    total(theta) = O(field) + exp(i|m| s theta) * C(R_{-s theta} field)

and total = O + C for a scalar (m=0) mode.  OCS-frame output -- consistent with
the OCS-rotated PSF moments, the v-mode (DZ) deviation, and the CWFS zk_OCS.
"""

import numpy as np

# Noll -> (radial n, signed azimuthal m); from ts_intrinsic_wavefront.
NOLL_NM = {
    4: (2, 0), 5: (2, -2), 6: (2, 2), 7: (3, -1), 8: (3, 1),
    9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2), 13: (4, -2),
    14: (4, 4), 15: (4, -4), 16: (5, 1), 17: (5, -1), 18: (5, 3),
    19: (5, -3), 20: (5, 5), 21: (5, -5), 22: (6, 0), 23: (6, -2),
    24: (6, 2), 25: (6, -4), 26: (6, 4),
}


def _group(noll_list):
    """Group Noll indices into singlets (m=0) and cos/sin doublets (spin |m|)."""
    s = set(int(j) for j in noll_list)
    groups, used = [], set()
    for j in sorted(s):
        if j in used:
            continue
        n, m = NOLL_NM[j]
        if m == 0:
            groups.append(('single', 0, j, None))
            used.add(j)
            continue
        partner = next((k for k in s if k not in used and k != j
                        and NOLL_NM[k] == (n, -m)), None)
        if partner is None:
            groups.append(('single', abs(m), j, None))  # orphan -> treat scalar
            used.add(j)
            continue
        j_cos, j_sin = (j, partner) if m > 0 else (partner, j)
        groups.append(('pair', abs(m), j_cos, j_sin))
        used.update((j, partner))
    return groups


class MIWCalib:
    """MIW from the Butler ``intrinsicZernikes`` calib, queried per detector.

    :param collection: Butler CALIBRATION collection holding intrinsicZernikes.
    :param physical_filter: e.g. ``i_39`` (the calib is filter-replicated).
    :param repo/butler: Butler repo path, or an existing Butler instance.
    Detector calibs are loaded lazily (and cached) as they are first queried, so
    only the detectors actually used are fetched.
    """

    def __init__(self, collection, physical_filter='i_39', repo='/repo/main',
                 instrument='LSSTCam', rotation_sign=1, butler=None):
        from lsst.daf.butler import Butler
        self.b = butler if butler is not None else Butler(repo)
        self.collection = collection
        self.filter = physical_filter
        self.instrument = instrument
        self.rotation_sign = rotation_sign
        refs = list(self.b.registry.queryDatasets(
            'intrinsicZernikes', collections=collection,
            where=f"instrument='{instrument}' and physical_filter='{physical_filter}'",
            findFirst=True))
        self.dets = sorted({r.dataId['detector'] for r in refs})
        if not self.dets:
            raise RuntimeError(f'no intrinsicZernikes in {collection} '
                               f'for filter {physical_filter}')
        cal0 = self._load(self.dets[0])
        self.noll = [int(j) for j in np.asarray(cal0.noll_indices)]
        self.k = {j: i for i, j in enumerate(self.noll)}
        self._ocs = cal0.interpolator_ocs          # shared telescope-fixed field
        self.groups = _group(self.noll)
        self._cache = {int(self.dets[0]): cal0}

    def _load(self, det):
        return self.b.get('intrinsicZernikes', collections=self.collection,
                          instrument=self.instrument, detector=int(det),
                          physical_filter=self.filter)

    def _ccs_interp(self, det):
        c = self._cache.get(int(det))
        if c is None:
            c = self._load(det)
            self._cache[int(det)] = c
        return c.interpolator

    def zernikes(self, thx_deg, thy_deg, rotator_rad, jmax, detector):
        """Total intrinsic Zernike vector (microns, index 0 unused), OCS frame.

        :param detector: per-star detector id -- selects the CCS interpolator.
        :returns: ndarray (n_stars, jmax+1); NaN where OCS/CCS support is missing.
        """
        thx = np.atleast_1d(np.asarray(thx_deg, float))
        thy = np.atleast_1d(np.asarray(thy_deg, float))
        det = np.broadcast_to(np.atleast_1d(np.asarray(detector, int)), thx.shape)
        th = self.rotation_sign * np.broadcast_to(
            np.atleast_1d(np.asarray(rotator_rad, float)), thx.shape)
        c, s = np.cos(th), np.sin(th)
        cx = c * thx + s * thy          # CCS query point R_{-theta}(field)
        cy = -s * thx + c * thy

        # OCS: shared interpolator, returns (N, n_noll); NaN outside the hull
        voc = np.asarray(self._ocs(np.column_stack([thx, thy])), float)
        O = {j: voc[:, self.k[j]] for j in self.noll}
        # CCS: each star's own detector interpolator, strictly within that CCD
        C = {j: np.full(thx.size, np.nan) for j in self.noll}
        for d in np.unique(det):
            m = det == d
            vcc = np.asarray(self._ccs_interp(int(d))(
                np.column_stack([cx[m], cy[m]])), float)
            for j in self.noll:
                C[j][m] = vcc[:, self.k[j]]

        z = np.zeros((thx.size, jmax + 1))
        for kind, spin, ja, jb in self.groups:
            if ja > jmax and (jb is None or jb > jmax):
                continue
            if kind == 'single':
                if ja <= jmax:
                    z[:, ja] = O[ja] + C[ja]
            else:
                jc, js = ja, jb
                phase = np.exp(1j * spin * th)
                tot = (O[jc] + 1j * O[js]) + phase * (C[jc] + 1j * C[js])
                if jc <= jmax:
                    z[:, jc] = tot.real
                if js <= jmax:
                    z[:, js] = tot.imag
        return z
