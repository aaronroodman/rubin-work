#!/usr/bin/env python3
"""plot_vmode_dof_matrix — OFC SVD mode diagnostics for a DOF/v-mode scheme.

Rebuilds the Optical Feedback Control (OFC) sensitivity-matrix singular value
decomposition (via ``ofc_svd.build_ofc_svd``, which takes the sensitivity matrix from
ts_ofc and the normalization weights from the ts_config_mttcs yaml) and renders one
figure per page:

  1  **V matrix** — the normalized, dimensionless degree-of-freedom (DOF) composition
     of each v-mode, drawn with square cells. All v-modes are shown, with a line
     marking the ``n_keep`` truncation, so the discarded modes are visible.
  2  **Singular values** — the spectrum, with the truncation marked.
  3  **Double Zernike per unit v-mode** — µm of DZ that a unit-amplitude v-mode
     produces (``sigma_m * u_m = S v_m``), rows being the (focal k, pupil Zj) terms.
     Only the retained v-modes are shown.
  4  **Reachability and residual per DZ term** — the fraction of each elementary DZ
     term that the retained v-modes can produce, and the irreducible remainder. Folded
     in from the former ``jk_coverage_plots.ipynb``.
  5  **Normalization weights** — a table of the per-DOF weight ``w_i`` applied,
     decomposed into its range factor ``r_i`` (DOF-units of stroke) and FWHM factor
     ``f_i`` (arcsec of PSF width per DOF-unit), since ``w_i = r_i^0.5 * f_i^-0.5``.

Data-independent apart from the pupil-Zernike set, which defaults to the standard
Z4-Z26 (omitting Z20, Z21) and is identical in every param_set built to date.
``--param-set`` reads it from that param_set's ``visits.parquet`` instead and warns on
a difference. Output goes to ``output/smatrix_vmode/`` — outside any param_set, because
nothing here depends on FAM data.

Needs ts_ofc and $TS_CONFIG_MTTCS_DIR.

``--check`` runs a regression test instead of plotting: it asserts ``build_ofc_svd``
reproduces ts_ofc's ``StateEstimator.get_dofs_from_vmodes`` (DOF-per-v-mode = N.V) on
identical inputs, and exits 0=PASS / 1=FAIL.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # aos/code
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'smatrix' / 'code'))
from aos_state import DOF22  # noqa: E402  canonical 22-DOF index list
SCHEMES = {'22_12': (DOF22, 12), '50_34': (None, 34)}

# Pupil (annular) Zernike Noll indices carried by the FAM donut tables: Z4-Z26 with
# Z20 and Z21 omitted, 21 terms. Identical in every param_set built to date, so it is
# the default here; --param-set reads it from that param_set's visits.parquet instead.
ZK_NOLL_DEFAULT = tuple(z for z in range(4, 27) if z not in (20, 21))


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
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd, DEFAULT_NORM_YAML
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
    ap.add_argument('--param-set', default=None,
                    help='optional: read the pupil-Zernike set from this param_set\'s '
                         'visits.parquet instead of using ZK_NOLL_DEFAULT')
    ap.add_argument('--scheme', default='22_12', choices=list(SCHEMES))
    ap.add_argument('--instrument', default='lsst')
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

    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd, DEFAULT_NORM_YAML
    n_dof, n_keep = SCHEMES[args.scheme]

    # The v-mode/DOF matrix is a property of the OFC sensitivity matrix and the DOF
    # scheme alone -- no FAM data enters it. The pupil-Zernike set is the only
    # data-derived input, and it is the same in every param_set built to date, so it
    # defaults to the standard set and the output is not keyed by param_set.
    out_dir = Path(args.output_root) / 'smatrix_vmode'
    if args.param_set:
        vis = Path(args.output_root) / args.param_set / 'visits.parquet'
        noll = [int(x) for x in np.asarray(
            pq.read_table(str(vis), columns=['nollIndices']).to_pandas()['nollIndices'].iloc[0])]
        if noll != list(ZK_NOLL_DEFAULT):
            print(f'note: {args.param_set} pupil-Zernike set differs from the default')
    else:
        noll = list(ZK_NOLL_DEFAULT)

    # Build with ALL modes kept so the discarded ones can be shown, then mark the
    # truncation. n_keep_eff on the full SVD is the DOF count.
    svd = build_ofc_svd(list(noll), k_min=1, k_max=6, n_keep=n_keep, n_dof=n_dof)
    n_all = svd.V.shape[1]                        # every available v-mode
    n_kept = svd.n_keep_eff                       # retained by this scheme
    V_all = svd.V                                 # (n_dof, n_all) dimensionless
    labels = svd.dof_labels()[0]
    n_d = V_all.shape[0]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    scheme_txt = args.scheme.replace('_', ' DoF / ') + ' v-modes'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'vmode_dof_matrix_{args.scheme}.pdf'
    with PdfPages(str(out)) as pdf:

        # ---- Page 1: V matrix, square cells, all v-modes, truncation marked ----
        # 0.32 in per cell keeps the cells square at any (n_dof, n_all).
        cell = 0.32
        fig, ax = plt.subplots(figsize=(max(7, cell * n_all + 4.5),
                                       max(5, cell * n_d + 2.0)), dpi=150)
        im = ax.imshow(V_all, cmap='seismic', vmin=-1, vmax=1, aspect='equal')
        ax.set_xlabel('v-mode m')
        ax.set_ylabel('normalized DOF (dimensionless)')
        ax.set_xticks(range(n_all))
        ax.set_xticklabels([str(m + 1) for m in range(n_all)], fontsize=6)
        ax.set_yticks(range(n_d))
        ax.set_yticklabels(labels, fontsize=6)
        if n_kept < n_all:
            ax.axvline(n_kept - 0.5, color='lime', lw=2.0)
            ax.text(n_kept - 0.5, -0.9, f'  kept: {n_kept}  |  discarded: {n_all - n_kept}',
                    color='green', fontsize=8, va='bottom', ha='left')
        ax.set_title(f'V matrix — DOF composition of each v-mode ({scheme_txt})\n'
                     f'all {n_all} v-modes shown; green line marks the n_keep truncation')
        fig.colorbar(im, ax=ax, shrink=0.7, label='V element (dimensionless)')
        if args.annotate_min > 0:
            for i in range(n_d):
                for m in range(n_all):
                    if abs(V_all[i, m]) >= args.annotate_min:
                        ax.text(m, i, f'{V_all[i, m]:.2f}', ha='center', va='center',
                                fontsize=4, color='k' if abs(V_all[i, m]) < 0.6 else 'w')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- Page 2: singular values, on their own page ----
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        ax.semilogy(np.arange(1, len(svd.Sigma) + 1), svd.Sigma, 'o-', ms=5)
        ax.axvline(n_kept + 0.5, color='green', alpha=0.7,
                   label=f'truncation at n_keep = {n_kept}')
        ax.set_xlabel('v-mode m')
        ax.set_ylabel(r'singular value $\sigma_m$  (µm of DZ per unit v-mode)')
        ax.set_title(f'Singular-value spectrum ({scheme_txt})')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- Page 3: µm of DZ per unit v-mode (retained modes only) ----
        # S v_m = sigma_m u_m, so sigma_m * U_eff[:, m] is the physical DZ wavefront
        # in µm that a unit-amplitude v-mode m produces.
        sig = np.asarray(svd.Sigma)[svd._keep()]
        DZ = np.asarray(svd.U_eff) * sig[None, :]        # (n_kj, n_kept), µm
        kj = list(svd.kj_grid)
        n_kj = DZ.shape[0]
        karr = np.array([k for k, j in kj])
        fig, ax = plt.subplots(figsize=(max(8, 0.42 * n_kept + 4.5),
                                       max(8, 0.11 * n_kj + 2)), dpi=150)
        vmax = float(np.nanpercentile(np.abs(DZ), 99))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(DZ, cmap='seismic', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xlabel('v-mode m')
        ax.set_ylabel('Double-Zernike term  (pupil Zj within each focal-k block)')
        ax.set_xticks(range(n_kept))
        ax.set_xticklabels([str(m + 1) for m in range(n_kept)], fontsize=7)
        ax.set_yticks(range(n_kj))
        ax.set_yticklabels([f'Z{j}' for k, j in kj], fontsize=4)
        for b in np.where(karr[1:] != karr[:-1])[0]:
            ax.axhline(b + 0.5, color='k', lw=0.8)
        # k=N block labels, placed inside the axes so they are never clipped
        for k in dict.fromkeys(karr):
            ax.text(-0.4, float(np.where(karr == k)[0].mean()), f'k={k}',
                    ha='right', va='center', fontsize=9, fontweight='bold',
                    clip_on=False)
        ax.set_title(f'Double-Zernike produced per unit v-mode  '
                     rf'($\sigma_m u_m$, µm of DZ)  ({scheme_txt})')
        fig.colorbar(im, ax=ax, shrink=0.8, label='µm of DZ per unit v-mode')
        # extra left margin for the k= labels
        fig.subplots_adjust(left=0.16)
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 4: reachability / residual per DZ term ----
        # From the former jk_coverage_plots.ipynb: how much of each elementary DZ term
        # the retained v-modes can produce, and the irreducible remainder.
        U_eff = np.asarray(svd.U_eff)
        frac = (U_eff ** 2).sum(axis=1)              # reachable fraction per DZ term
        n_k = len(dict.fromkeys(karr)); n_j = n_kj // n_k
        frac_2d = frac.reshape(n_k, n_j)
        resid_2d = 1.0 - frac_2d
        jlabels = [f'Z{j}' for k, j in kj[:n_j]]
        klabels = [f'k={k}' for k in dict.fromkeys(karr)]
        fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * n_j + 3), 7.5), dpi=150)
        for ax, M, ttl, cm in (
                (axes[0], 100 * frac_2d, 'reachable', 'viridis'),
                (axes[1], 100 * resid_2d, 'residual after the v-mode fit', 'magma')):
            im = ax.imshow(M, cmap=cm, vmin=0, vmax=100, aspect='auto')
            ax.set_xticks(range(n_j)); ax.set_xticklabels(jlabels, fontsize=6, rotation=90)
            ax.set_yticks(range(n_k)); ax.set_yticklabels(klabels, fontsize=8)
            ax.set_title(f'Per-DZ-term {ttl}, n_keep = {n_kept}  [% of power]')
            fig.colorbar(im, ax=ax, shrink=0.9, label='% of power')
        fig.suptitle(f'Reachability of each Double-Zernike term ({scheme_txt})\n'
                     f'mean residual {100 * resid_2d.mean():.1f}% of power; '
                     f'{int((frac_2d >= 0.95).sum())} of {n_kj} terms >= 95% reachable',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)

        # ---- Page 5: normalization weights, decomposed into range and FWHM ----
        # w_i = r_i^0.5 * f_i^-0.5 (alpha=0.5, beta=-0.5 for range0.5_fwhm-0.15).
        # f_i is recomputed from the sensitivity matrix as the field-averaged
        # quadrature PSF width per DOF-unit (arcsec/DOF-unit); r_i then follows as
        # w_i^2 * f_i, in DOF-units of usable stroke.
        nw = np.asarray(svd.normalization_weights, float)
        f_i = r_i = None
        try:
            import normalization_weights as NW
            from lsst.ts.ofc import OFCData
            sens = np.asarray(OFCData(args.instrument).sensitivity_matrix)
            f_full = NW.compute_f_quadrature(sens, rings=5, spokes=6, znmin=4, znmax=22)
            f_i = f_full[list(svd.dof_idx)]
            r_i = nw ** 2 * f_i
        except Exception as e:                                  # noqa: BLE001
            print(f'note: could not decompose the weights ({type(e).__name__}: {e});'
                  ' showing the combined weight only')

        print(f'\nOFC per-DOF normalization weights ({args.scheme}, '
              f'{DEFAULT_NORM_YAML}):')
        hdr = f'  {"DOF":22s} {"w_i":>13s}'
        if f_i is not None:
            hdr += f' {"r_i [DOF-unit]":>16s} {"f_i [arcsec/DOF-unit]":>22s}'
        print(hdr)
        for i, (lab, w) in enumerate(zip(labels, nw)):
            line = f'  {lab:22s} {w:13.6g}'
            if f_i is not None:
                line += f' {r_i[i]:16.6g} {f_i[i]:22.6g}'
            print(line)

        ncol = 4 if f_i is not None else 2
        colw = 12 if f_i is not None else 9
        half = (len(nw) + 1) // 2
        fig, axes = plt.subplots(1, 2, figsize=(colw * 1.7, max(5, 0.22 * half + 1.4)),
                                 dpi=150)
        head = ['DOF', 'w_i'] + (['r_i\n[DOF-unit]', 'f_i\n[arcsec / DOF-unit]']
                                 if f_i is not None else [])
        for ax, lo, hi in [(axes[0], 0, half), (axes[1], half, len(nw))]:
            ax.axis('off')
            rows = []
            for i in range(lo, hi):
                row = [labels[i], f'{nw[i]:.5g}']
                if f_i is not None:
                    row += [f'{r_i[i]:.5g}', f'{f_i[i]:.4g}']
                rows.append(row)
            if rows:
                t = ax.table(cellText=rows, colLabels=head, loc='center', cellLoc='left')
                t.auto_set_font_size(False); t.set_fontsize(6); t.scale(1, 1.3)
        sub = (r'  $w_i = r_i^{0.5} f_i^{-0.5}$;  $r_i$ = usable stroke in DOF units,  '
               r'$f_i$ = field-averaged PSF width per DOF unit [arcsec]'
               if f_i is not None else '')
        fig.suptitle(f'OFC per-DOF normalization weights  [{args.scheme}]\n{sub}',
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)

    print(f'wrote {out}  (5 pages: V {n_d}x{n_all} with {n_kept} kept; '
          f'singular values; DZ-per-v-mode {n_kj}x{n_kept} µm; '
          f'reachability; {len(nw)} DOF normalization table)')


if __name__ == '__main__':
    main()
