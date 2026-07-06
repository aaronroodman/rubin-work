#!/usr/bin/env python3
"""plot_vmode_dof_matrix — DoF-vs-v-mode matrix (V) for an OFC SVD scheme.

Rebuilds the OFC sensitivity-matrix SVD (via ofc_svd.build_ofc_svd) and renders
the right-singular-vector block V[:, :n_keep]: the (dimensionless) DOF composition
of each retained v-mode, exactly as in smatrix_vmode_info §3.  Companion panel:
the singular-value spectrum with the truncation cut.  Defaults to the 22-DoF /
12-v-mode scheme used by wfs_dof_compare.

The V matrix is data-independent; only the pupil-Zernike set (iZs) enters, read
from the param_set's visits.parquet so it matches the wfs_dof_compare SVD.

Needs ts_ofc (build_ofc_svd) + TS_CONFIG_MTTCS_DIR; runs in the LSST stack env.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))
SCHEMES = {'22_12': (DOF22, 12), '50_34': (None, 34)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', default='fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x')
    ap.add_argument('--scheme', default='22_12', choices=list(SCHEMES))
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--annotate-min', type=float, default=0.10,
                    help='annotate cells with |V_im| >= this (0 = none)')
    args = ap.parse_args()
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd
    n_dof, n_keep = SCHEMES[args.scheme]

    base = Path(args.output_root) / args.param_set
    noll = [int(x) for x in np.asarray(
        pq.read_table(str(base / 'visits.parquet'), columns=['nollIndices']).to_pandas()['nollIndices'].iloc[0])]

    svd = build_ofc_svd(list(noll), k_min=1, k_max=6, n_keep=n_keep, n_dof=n_dof)
    V = svd.V[:, :svd.n_keep_eff]                 # (n_dof, n_keep) DOF composition
    labels = svd.dof_labels()[0]
    n_d = V.shape[0]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    out = base / f'vmode_dof_matrix_{args.scheme}.pdf'
    with PdfPages(str(out)) as pdf:
        fig = plt.figure(figsize=(12, max(6, 0.32 * n_d)), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
        ax0, ax1 = plt.subplot(gs[0]), plt.subplot(gs[1])

        im = ax0.imshow(V, cmap='seismic', vmin=-1, vmax=1, aspect='auto')
        ax0.set_xlabel('V-mode m'); ax0.set_ylabel('Normalized DOF')
        ax0.set_xticks(range(svd.n_keep_eff)); ax0.set_xticklabels([str(m + 1) for m in range(svd.n_keep_eff)])
        ax0.set_yticks(range(n_d)); ax0.set_yticklabels(labels, fontsize=7)
        ax0.set_title(f'V matrix — normalized (dimensionless) DOF coefficients\n'
                      f'({args.scheme.replace("_", " DoF / ")} v-modes)')
        fig.colorbar(im, ax=ax0, shrink=0.8)
        if args.annotate_min > 0:
            for i in range(n_d):
                for m in range(svd.n_keep_eff):
                    if abs(V[i, m]) >= args.annotate_min:
                        ax0.text(m, i, f'{V[i, m]:.2f}', ha='center', va='center', fontsize=5,
                                 color='k' if abs(V[i, m]) < 0.6 else 'w')

        ax1.semilogy(np.arange(1, len(svd.Sigma) + 1), svd.Sigma, 'o-', ms=4)
        ax1.axvline(svd.n_keep_eff + 0.5, color='green', alpha=0.6, label=f'truncation at {svd.n_keep_eff}')
        ax1.set_xlabel('V-mode m'); ax1.set_ylabel(r'$\sigma_m$')
        ax1.set_title('Singular values'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f'wrote {out}  (V {V.shape[0]}x{V.shape[1]}, {len(noll)} pupil Zernikes)')


if __name__ == '__main__':
    main()
