#!/usr/bin/env python3
"""plot_vmode_dof_matrix — OFC SVD mode diagnostics for a DOF/v-mode scheme.

Rebuilds the OFC sensitivity-matrix SVD (via ofc_svd.build_ofc_svd, which uses
ts_ofc for the sensitivity matrix and the ts_config_mttcs normalization yaml) and
renders a multi-page PDF:

  Page 1  V matrix (right singular vectors): the normalized (dimensionless) DOF
          composition of each retained v-mode, + the singular-value spectrum.
  Page 2  microns of Double-Zernike produced per unit v-mode (sigma_m * u_m =
          S v_m) -- rows are the (focal k, pupil Zj) DZ terms ordered k=1
          j=4..26, then k=2, ...; columns the v-modes.  This is the physical
          um-of-DZ each unit v-mode maps to (U_eff alone is only the unit shape).
  Page 3  the per-DOF OFC normalization weights that were applied, as a table
          (printed to stdout too; they span orders of magnitude).

Data-independent apart from the pupil-Zernike set (iZs), read from the param_set's
visits.parquet so it matches the wfs_dof_compare SVD.  Defaults to the 22-DoF /
12-v-mode scheme.  Needs ts_ofc (build_ofc_svd) + TS_CONFIG_MTTCS_DIR.

`--check` runs a regression test instead of plotting: it asserts build_ofc_svd
reproduces ts_ofc's StateEstimator.get_dofs_from_vmodes (DOF-per-v-mode = N.V) on
identical inputs, and exits 0=PASS / 1=FAIL.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))
SCHEMES = {'22_12': (DOF22, 12), '50_34': (None, 34)}


def run_ofc_check(instrument='lsst'):
    """Regression check: build_ofc_svd must reproduce ts_ofc's
    StateEstimator.get_dofs_from_vmodes on IDENTICAL inputs.

    ts_ofc and ofc_svd both do svd(sensitivity[:, dof_idx] @ diag(norm)); ts_ofc's
    get_dofs_from_vmodes(e_m) = N . V[:, m] = physical DOF per unit v-mode, which
    is exactly our normalization_weights * svd.V.  Building our SVD from the SAME
    full focal-k x pupil-zn matrix, DOF set, normalization yaml and truncation,
    the two must agree to numerical precision.  Returns True on PASS.
    """
    from lsst.ts.ofc import OFCData, StateEstimator
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd
    ofc = OFCData(instrument); se = StateEstimator(ofc)
    S = np.asarray(ofc.sensitivity_matrix)               # (n_k, n_zn, n_dof)
    n_k, n_zn, _ = S.shape
    dof_idx = [int(d) for d in ofc.dof_idx]
    norm_yaml = ofc.controller['normalization_weights_filename']
    n_keep = int(se.truncate_index) if se.truncate_index else se.Vh.shape[0]
    n_modes = se.Vh.shape[0]
    # ts_ofc DOF-per-v-mode: get_dofs_from_vmodes on each unit v-mode
    M_ofc = np.column_stack(
        [se.get_dofs_from_vmodes(np.eye(n_modes)[m]) for m in range(n_keep)])
    # our ofc_svd, same inputs (all focal-k, all pupil-zn, same DOF + norm)
    svd = build_ofc_svd(list(range(n_zn)), k_min=0, k_max=n_k - 1,
                        n_keep=n_keep, n_dof=dof_idx, norm_yaml_name=norm_yaml)
    M_ours = np.asarray(svd.normalization_weights)[:, None] * svd.V[:, :n_keep]
    norm_ok = np.allclose(np.asarray(ofc.normalization_weights)[dof_idx],
                          svd.normalization_weights)
    sig_ok = np.allclose(se.S[:n_keep], svd.Sigma[:n_keep], rtol=1e-6)
    for m in range(n_keep):                              # SVD sign is per-mode arbitrary
        if np.dot(M_ofc[:, m], M_ours[:, m]) < 0:
            M_ours[:, m] *= -1
    d = float(np.abs(M_ofc - M_ours).max())
    ok = bool(norm_ok and sig_ok and d < 1e-8)
    print(f'[check vs ts_ofc StateEstimator.get_dofs_from_vmodes]  '
          f'n_dof={len(dof_idx)} n_keep={n_keep} norm={norm_yaml}\n'
          f'  normalization arrays match: {norm_ok}\n'
          f'  singular values match:      {sig_ok} '
          f'(max |dS|={np.abs(se.S[:n_keep]-svd.Sigma[:n_keep]).max():.2e})\n'
          f'  DOF-per-v-mode max |ts_ofc - ours| = {d:.3e} '
          f'(scale ~{np.abs(M_ofc).max():.3g})\n'
          f'  -> {"PASS" if ok else "FAIL"}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', default='fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x')
    ap.add_argument('--scheme', default='22_12', choices=list(SCHEMES))
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--annotate-min', type=float, default=0.10,
                    help='annotate cells with |V_im| >= this (0 = none)')
    ap.add_argument('--check', action='store_true',
                    help='regression check only: assert build_ofc_svd reproduces '
                         'ts_ofc StateEstimator.get_dofs_from_vmodes, then exit '
                         '(0=PASS, 1=FAIL); makes no plot')
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if run_ofc_check() else 1)

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

        # ---- Page 2: MICRONS of Double-Zernike produced per unit v-mode ----
        # S v_m = sigma_m u_m, so (sigma_m * U_eff[:, m]) is the physical DZ
        # wavefront (um) a unit-amplitude v-mode m produces, broken down by DZ
        # term.  Rows are the (focal k, pupil Zj) terms in kj_grid order
        # (k outer, j inner: k=1 j=4..26, then k=2, ...).
        sig = np.asarray(svd.Sigma)[svd._keep()]        # kept singular values
        DZ = np.asarray(svd.U_eff) * sig[None, :]        # (n_kj, n_keep_eff), um
        kj = list(svd.kj_grid)
        n_kj = DZ.shape[0]
        karr = np.array([k for k, j in kj])
        fig = plt.figure(figsize=(max(8, 0.42 * svd.n_keep_eff + 3),
                                  max(8, 0.11 * n_kj + 2)), dpi=150)
        ax = fig.add_subplot(111)
        vmax = float(np.nanpercentile(np.abs(DZ), 99)) or 1.0
        im = ax.imshow(DZ, cmap='seismic', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xlabel('v-mode m')
        ax.set_ylabel('Double-Zernike term  (pupil Zj within each focal-k block)')
        ax.set_xticks(range(svd.n_keep_eff))
        ax.set_xticklabels([str(m + 1) for m in range(svd.n_keep_eff)], fontsize=7)
        ax.set_yticks(range(n_kj))
        ax.set_yticklabels([f'Z{j}' for k, j in kj], fontsize=4)
        for b in np.where(karr[1:] != karr[:-1])[0]:    # k-block separators
            ax.axhline(b + 0.5, color='k', lw=0.8)
        for k in dict.fromkeys(karr):                   # k=N labels per block
            ax.text(-0.085, float(np.where(karr == k)[0].mean()), f'k={k}',
                    transform=ax.get_yaxis_transform(), ha='right', va='center',
                    fontsize=9, fontweight='bold')
        ax.set_title(f'Double-Zernike produced per unit v-mode  '
                     f'(sigma_m * u_m, um of DZ)  '
                     f'({args.scheme.replace("_", " DoF / ")})')
        fig.colorbar(im, ax=ax, shrink=0.8, label='um of DZ per unit v-mode')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- Page 3: per-DOF OFC normalization weights (TABLE) ----
        # weights span orders of magnitude, so a table reads better than a bar.
        nw = np.asarray(svd.normalization_weights, float)
        print(f'\nOFC per-DOF normalization weights ({args.scheme}, '
              f'ts_config_mttcs range0.5_fwhm-0.15):')
        for lab, w in zip(labels, nw):
            print(f'  {lab:24s} {w:14.6g}')
        half = (len(nw) + 1) // 2
        fig, axes = plt.subplots(1, 2, figsize=(11, max(5, 0.22 * half + 1.2)),
                                 dpi=150)
        for ax, lo, hi in [(axes[0], 0, half), (axes[1], half, len(nw))]:
            ax.axis('off')
            rows = [[labels[i], f'{nw[i]:.6g}'] for i in range(lo, hi)]
            if rows:
                t = ax.table(cellText=rows, colLabels=['DOF', 'norm weight'],
                             loc='center', cellLoc='left')
                t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.25)
        fig.suptitle('OFC per-DOF normalization weights applied '
                     f'(range0.5_fwhm-0.15)  [{args.scheme}]', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)

    print(f'wrote {out}  (3 pages: V {V.shape[0]}x{V.shape[1]}, '
          f'DZ-per-v-mode {DZ.shape[0]}x{DZ.shape[1]} um, '
          f'{len(nw)} DOF norm-weight table; {len(noll)} pupil Zernikes)')


if __name__ == '__main__':
    main()
