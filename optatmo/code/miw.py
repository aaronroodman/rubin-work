"""
Measured Intrinsic Wavefront (MIW) loader: OCS + CCS -> Zernike vector.

Reads the ts_intrinsic_wavefront ``intrinsic_split_maps`` parquet (telescope-
fixed OCS and camera-fixed CCS Zernike field maps, microns, Noll Z4..Z26 on a
(thx_deg, thy_deg) grid) and reconstructs the total intrinsic wavefront at
arbitrary field positions and rotator angle.

Reconstruction convention (from ts_intrinsic_wavefront.intrinsic_split, s=+1,
CCS = R(-theta).OCS): for a spin-|m| Zernike doublet with complex field
Z_cos + i Z_sin,

    total(theta) = O(field)  +  exp(i|m| s theta) * C(R_{-s theta} field)

and for a scalar (m=0) mode, total = O(field) + C(R_{-s theta} field).
The CCS map is sampled at the field position rotated by -s*theta and the
complex coefficient picks up the spin phase.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import Delaunay


def _bary_interp(tri, values, qx, qy):
    """Barycentric-linear interp of `values` on triangulation `tri`; NaN outside."""
    q = np.column_stack([np.atleast_1d(qx), np.atleast_1d(qy)])
    out = np.full(len(q), np.nan)
    if tri is None:
        return out
    simp = tri.find_simplex(q)
    ok = simp >= 0
    if np.any(ok):
        T = tri.transform[simp[ok]]
        d = q[ok] - T[:, 2]
        bary = np.einsum('ijk,ik->ij', T[:, :2], d)
        w = np.column_stack([bary, 1 - bary.sum(1)])
        verts = tri.simplices[simp[ok]]
        out[ok] = np.einsum('ij,ij->i', w, values[verts])
    return out

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


class MIW:
    def __init__(self, parquet_path, rotation_sign=1):
        t = pq.read_table(parquet_path).to_pydict()
        self.thx = np.asarray(t['thx_deg'], float)
        self.thy = np.asarray(t['thy_deg'], float)
        self.pts = np.column_stack([self.thx, self.thy])
        self.tri = Delaunay(self.pts)
        self.rotation_sign = rotation_sign
        # available Noll indices (both OCS and CCS present)
        self.nolls = sorted(int(k[1:-4]) for k in t
                            if k.endswith('_OCS') and (k[:-4] + '_CCS') in t)
        self.ocs = {j: np.asarray(t[f'Z{j}_OCS'], float) for j in self.nolls}
        self.ccs = {j: np.asarray(t[f'Z{j}_CCS'], float) for j in self.nolls}
        self.groups = _group(self.nolls)

    def _interp(self, values, qx, qy):
        """Barycentric-linear interpolation of `values` at (qx,qy); NaN outside."""
        q = np.column_stack([np.atleast_1d(qx), np.atleast_1d(qy)])
        simp = self.tri.find_simplex(q)
        out = np.full(len(q), np.nan)
        ok = simp >= 0
        if np.any(ok):
            T = self.tri.transform[simp[ok]]
            d = q[ok] - T[:, 2]
            bary = np.einsum('ijk,ik->ij', T[:, :2], d)
            w = np.column_stack([bary, 1 - bary.sum(1)])
            verts = self.tri.simplices[simp[ok]]
            out[ok] = np.einsum('ij,ij->i', w, values[verts])
        return out

    def zernikes(self, thx_deg, thy_deg, rotator_rad, jmax):
        """Total intrinsic Zernike vector (microns, index 0 unused) per star.

        :returns: ndarray (n_stars, jmax+1).
        """
        thx = np.atleast_1d(np.asarray(thx_deg, float))
        thy = np.atleast_1d(np.asarray(thy_deg, float))
        th = self.rotation_sign * np.atleast_1d(np.asarray(rotator_rad, float))
        th = np.broadcast_to(th, thx.shape)
        c, s = np.cos(th), np.sin(th)
        # CCS sampled at R_{-theta} field position
        cx = c * thx + s * thy
        cy = -s * thx + c * thy

        z = np.zeros((thx.size, jmax + 1))
        for kind, spin, ja, jb in self.groups:
            if ja > jmax and (jb is None or jb > jmax):
                continue
            if kind == 'single':
                o = self._interp(self.ocs[ja], thx, thy)
                cc = self._interp(self.ccs[ja], cx, cy)
                if ja <= jmax:
                    z[:, ja] = o + cc
            else:
                jc, js = ja, jb
                Oc = self._interp(self.ocs[jc], thx, thy)
                Os = self._interp(self.ocs[js], thx, thy)
                Cc = self._interp(self.ccs[jc], cx, cy)
                Cs = self._interp(self.ccs[js], cx, cy)
                phase = np.exp(1j * spin * th)
                tot = (Oc + 1j * Os) + phase * (Cc + 1j * Cs)
                if jc <= jmax:
                    z[:, jc] = tot.real
                if js <= jmax:
                    z[:, js] = tot.imag
        return z


class MIWOfficial:
    """MIW from the official ip_isr ``intrinsicZernikes`` calib product.

    Reads the two parquets written by ``export_official_miw.py``:
      * ``intrinsic_official_ocs.parquet`` -- shared telescope-fixed OCS map
        (thx_deg, thy_deg, Z{j}_OCS), interpolated globally.
      * ``intrinsic_official_ccs.parquet`` -- per-detector camera-fixed CCS map
        (detector, thx_deg, thy_deg, Z{j}_CCS) with the per-CCD focal-plane
        height already folded into Z4 at every footprint point.  Interpolated
        **per detector** so the per-CCD height steps are preserved (a global
        interp would smear them out).

    Reconstruction is identical to `MIW` (OCS spin-phase convention, OCS-frame
    output): total(theta) = O(field) + exp(i|m| s theta) * C(R_{-s theta} field),
    with C taken from the star's own detector footprint.
    """

    def __init__(self, ocs_parquet, ccs_parquet, rotation_sign=1):
        ocs = pd.read_parquet(ocs_parquet)
        ccs = pd.read_parquet(ccs_parquet)
        self.rotation_sign = rotation_sign
        self.nolls = sorted(
            int(c[1:-4]) for c in ocs.columns
            if c.endswith('_OCS') and c[1:-4].isdigit()
            and f'Z{c[1:-4]}_CCS' in ccs.columns)
        # global OCS triangulation
        self.ocs_tri = Delaunay(np.column_stack(
            [ocs['thx_deg'].to_numpy(float), ocs['thy_deg'].to_numpy(float)]))
        self.ocs = {j: ocs[f'Z{j}_OCS'].to_numpy(float) for j in self.nolls}
        # per-detector CCS triangulations + values
        self.ccs_tri, self.ccs = {}, {}
        for det, g in ccs.groupby('detector'):
            det = int(det)
            pts = np.column_stack([g['thx_deg'].to_numpy(float),
                                   g['thy_deg'].to_numpy(float)])
            try:
                self.ccs_tri[det] = Delaunay(pts)
            except Exception:
                self.ccs_tri[det] = None       # degenerate (collinear) footprint
            self.ccs[det] = {j: g[f'Z{j}_CCS'].to_numpy(float)
                             for j in self.nolls}
        self.groups = _group(self.nolls)

    def zernikes(self, thx_deg, thy_deg, rotator_rad, jmax, detector):
        """Total intrinsic Zernike vector (microns, index 0 unused) per star.

        :param detector: per-star detector id (int), selects the CCS footprint.
        :returns: ndarray (n_stars, jmax+1); NaN where OCS/CCS support is missing.
        """
        thx = np.atleast_1d(np.asarray(thx_deg, float))
        thy = np.atleast_1d(np.asarray(thy_deg, float))
        det = np.broadcast_to(np.atleast_1d(np.asarray(detector, int)),
                              thx.shape)
        th = self.rotation_sign * np.atleast_1d(np.asarray(rotator_rad, float))
        th = np.broadcast_to(th, thx.shape)
        c, s = np.cos(th), np.sin(th)
        cx = c * thx + s * thy          # CCS query point R_{-theta}(field)
        cy = -s * thx + c * thy

        # OCS: one global interp per Noll
        O = {j: _bary_interp(self.ocs_tri, self.ocs[j], thx, thy)
             for j in self.nolls}
        # CCS: per detector interp per Noll
        C = {j: np.full(thx.size, np.nan) for j in self.nolls}
        for d in np.unique(det):
            m = det == d
            tri = self.ccs_tri.get(int(d))
            vals = self.ccs.get(int(d))
            if tri is None or vals is None:
                continue
            for j in self.nolls:
                C[j][m] = _bary_interp(tri, vals[j], cx[m], cy[m])

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
