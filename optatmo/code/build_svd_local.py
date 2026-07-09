"""Build AOS sensitivity-matrix SVDs locally from a one-time raw dump
(ofc_raw.npz = S_full + nw_full), replicating ofc_svd.build_ofc_svd exactly.

Lets us make any (iZs, k_min, k_max, n_keep, dof_idx) SVD without ts_ofc.
"""
import numpy as np

RAW = 'data/ofc_raw.npz'
# 22-DOF reduced set: 10 rigid + first 7 M1M3 + first 5 M2
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))


def build(iZs, k_min, k_max, n_keep, dof_idx=None, raw=RAW, out=None):
    d = np.load(raw)
    S_full, nw_full = d['S_full'], d['nw_full']
    iZs = np.asarray(iZs, int)
    n_k, n_j = k_max - k_min + 1, len(iZs)
    S_slab = S_full[k_min:k_max + 1, iZs, :]                 # (n_k, n_j, n_dof_full)
    n_dof_full = S_slab.shape[-1]
    dof_idx = list(range(n_dof_full)) if dof_idx is None else list(dof_idx)
    norm_sub = nw_full[dof_idx]
    S = S_slab.reshape(-1, n_dof_full)[:, dof_idx] @ np.diag(norm_sub)
    kj_grid = np.array([(k_min + ki, int(iZs[ji]))
                        for ki in range(n_k) for ji in range(n_j)])
    U, Sigma, Vh = np.linalg.svd(S, full_matrices=False)
    keep = list(range(min(n_keep, U.shape[1]))) if np.isscalar(n_keep) else list(n_keep)
    res = dict(U_eff=U[:, keep], kj_grid=kj_grid, Sigma=Sigma,
               dof_idx=np.asarray(dof_idx), normalization_weights=norm_sub,
               iZs=iZs, k_min=k_min, k_max=k_max)
    if out:
        np.savez(out, **res)
        print(f'wrote {out}: U_eff {res["U_eff"].shape}, Sigma[:14] {Sigma[:14].round(3)}')
    return res


def validate_against(npz_path, iZs, k_min, k_max, n_keep, dof_idx):
    """Check local build matches a reference npz (built on USDF)."""
    ref = np.load(npz_path)
    loc = build(iZs, k_min, k_max, n_keep, dof_idx)
    # SVD sign/degenerate-subspace ambiguity: compare |U^T U_ref| block and Sigma
    dsig = np.max(np.abs(loc['Sigma'][:len(ref['Sigma'])] - ref['Sigma']))
    # subspace agreement: singular values match + spans align
    print(f'Sigma max abs diff: {dsig:.2e}')
    print(f'U_eff shapes local {loc["U_eff"].shape} ref {ref["U_eff"].shape}')
    # column-space overlap (rotation-invariant): ||U_ref^T U_loc|| per singular direction
    M = ref['U_eff'].T @ loc['U_eff']
    print(f'diag |U_ref^T U_loc| (1=aligned): {np.abs(np.diag(M)).round(3)}')


if __name__ == '__main__':
    import os
    # 1) validate replication against the 22/12 k6 npz already built on USDF
    if os.path.exists('data/ofc_svd_22_12_k6.npz') and os.path.exists(RAW):
        print('=== validate local build vs USDF ofc_svd_22_12_k6.npz ===')
        validate_against('data/ofc_svd_22_12_k6.npz', range(4, 23), 1, 6, 12, DOF22)
    # 2) build 50/34 k6 (all 50 DOF, keep 34)
    if os.path.exists(RAW):
        print('\n=== build 50/34 k6 ===')
        build(range(4, 23), 1, 6, 34, dof_idx=None, out='data/ofc_svd_50_34_k6.npz')
    else:
        print(f'{RAW} not found — run the raw-dump snippet on USDF and scp it here.')
