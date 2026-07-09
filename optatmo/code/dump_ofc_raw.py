"""Dump the raw AOS OFC sensitivity matrix + normalization weights (one-time).

Run on the USDF in an env where lsst.ts.ofc + lsst.ts.intrinsic import
(the AOS/CWFS environment), from rubin-work/optatmo:

    python code/dump_ofc_raw.py

Writes optatmo/data/ofc_raw.npz (S_full, nw_full).  From this single dump the
laptop side (code/build_svd_local.py) reconstructs any (iZs, k_min, k_max,
n_keep, dof_idx) SVD in pure numpy — no further ts_ofc round-trips.
"""
import os
import numpy as np

import lsst.ts.intrinsic.wavefront.ofc_svd as osv
from lsst.ts.ofc import OFCData

S_full = np.asarray(OFCData('lsst').sensitivity_matrix)   # (n_focal, n_noll, 50)
nw_full = np.asarray(osv.load_normalization_weights())    # (50,) default (geom_mean)

out = os.path.join(os.path.dirname(__file__), '..', 'data', 'ofc_raw.npz')
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, S_full=S_full, nw_full=nw_full)
print(f'S_full {S_full.shape}, nw_full {nw_full.shape}')
print(f'wrote {os.path.abspath(out)}')
