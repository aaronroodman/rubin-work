"""
Fit the OptAtmo DZ + atmosphere model to real PSF-star moments and compare the
fitted Double-Zernike coefficients to the CWFS donut-wavefront DZ.

Pipeline per in-focus visit:
  1. load the extracted per-star moment catalog (CCS frame, arcsec^n);
  2. rotate moments + field positions CCS -> OCS by the rotator angle (frames);
  3. bin stars onto a focal-plane grid (median, robust error);
  4. run the moment fit (annular DZ + atmosphere shear, SVD degeneracy control);
  5. compare fitted z{k}f{f} to CWFS fits.parquet z1toz3_z{k}_c{f}.
"""

import numpy as np
import pandas as pd

from jax_optatmo import MOMENT_LABELS
import frames
import fit as fitmod

CWFS_FITS = ('../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/fits.parquet')


def load_and_prep(parquet, sign=1, rot_deg=None, clip_thr=5.0):
    """Load catalog, rotate CCS->OCS. Returns dict of arrays (OCS frame).

    rot_deg overrides the (possibly missing) per-row rotator angle.
    """
    df = pd.read_parquet(parquet)
    if rot_deg is not None:
        df = df.assign(rot_deg=rot_deg)
    mom = np.column_stack([df[k].to_numpy() for k in MOMENT_LABELS])

    # robust MAD-based outlier rejection on the fit moments (blends/artifacts
    # give wild 3rd-order values); keep stars within clip_thr robust-sigma.
    keep = np.ones(len(df), bool)
    for k in ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03']:
        v = mom[:, MOMENT_LABELS.index(k)]
        med = np.nanmedian(v)
        mad = 1.4826 * np.nanmedian(np.abs(v - med)) + 1e-30
        keep &= np.abs(v - med) < clip_thr * mad
    df = df[keep].reset_index(drop=True)
    mom = mom[keep]

    rot = np.deg2rad(df['rot_deg'].to_numpy())
    mom_ocs = np.array([frames.rotate_moments(mom[i], rot[i], sign)
                        for i in range(len(df))])
    thx_ocs, thy_ocs = frames.rotate_field(
        df['thx_ccs_deg'].to_numpy(), df['thy_ccs_deg'].to_numpy(), rot, sign)
    err = np.column_stack([df.get(k + '_err', pd.Series(np.full(len(df), np.nan)))
                           .to_numpy() for k in MOMENT_LABELS])
    print(f'  {parquet.split("/")[-1]}: kept {keep.sum()}/{len(keep)} stars '
          f'(clipped {(~keep).sum()} outliers)')
    return dict(thx=thx_ocs, thy=thy_ocs, rot=rot, mom=mom_ocs, err=err,
                detector=df['detector'].to_numpy(), x=df['x'].to_numpy(),
                y=df['y'].to_numpy())


def bin_grid(prep, cell_deg=0.10, min_n=3):
    """Median-bin stars on a *per-detector* sub-CCD grid; empirical errors.

    Bins are keyed by (detector, sub-CCD field cell) so a cell never straddles
    two CCDs -- this preserves the per-CCD focal-plane-height Z4 step (the focus
    diversity that breaks the Z4<->seeing degeneracy) and gives each cell a
    single, well-defined detector for the per-detector CCS intrinsic lookup.
    cell_deg ~ half a CCD (~0.2 deg) yields a few cells per CCD.
    """
    thx, thy, mom, det = prep['thx'], prep['thy'], prep['mom'], prep['detector']
    ix = np.floor(thx / cell_deg).astype(np.int64)
    iy = np.floor(thy / cell_deg).astype(np.int64)
    keys = det.astype(np.int64) * 1_000_000 + (ix + 500) + 1000 * (iy + 500)
    out_thx, out_thy, out_mom, out_err, out_rot, out_det = [], [], [], [], [], []
    for k in np.unique(keys):
        m = keys == k
        if m.sum() < min_n:
            continue
        out_thx.append(np.median(thx[m]))
        out_thy.append(np.median(thy[m]))
        med = np.median(mom[m], axis=0)
        out_mom.append(med)
        # robust error on the median: 1.253 * std / sqrt(n)
        out_err.append(1.253 * np.std(mom[m], axis=0) / np.sqrt(m.sum()))
        out_rot.append(np.median(prep['rot'][m]))
        out_det.append(int(det[m][0]))          # single detector by construction
    return dict(thx=np.array(out_thx), thy=np.array(out_thy),
                rot=np.array(out_rot), mom=np.array(out_mom),
                err=np.array(out_err), detector=np.array(out_det, int))


def to_catalog(binned):
    err = binned['err'].copy()
    # floor tiny errors to avoid over-weighting
    for j in range(err.shape[1]):
        col = err[:, j]
        good = np.isfinite(col) & (col > 0)
        if good.any():
            col[~good | (col < 0.1 * np.median(col[good]))] = np.median(col[good])
        err[:, j] = col
    return {'thx_deg': binned['thx'], 'thy_deg': binned['thy'],
            'rotator_rad': binned['rot'], 'moments': binned['mom'],
            'errors': err, 'detector': binned['detector']}


def cwfs_dz(day_obs, infocus_seq):
    """CWFS Double-Zernike coeffs for the FAM key (= in-focus seq - 1)."""
    f = pd.read_parquet(CWFS_FITS)
    row = f[(f.day_obs == day_obs) & (f.seq_num == infocus_seq - 1)]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    out = {}
    for k in list(range(4, 12)) + [14, 15]:
        for c in (1, 2, 3):
            col = f'z1toz3_z{k}_c{c}'
            if col in row:
                out[f'z{k}f{c}'] = float(row[col])
    return out


def fit_visit(cfg, parquet, day_obs, infocus_seq, sign=1, cell_deg=0.25,
              rot_deg=None):
    prep = load_and_prep(parquet, sign=sign, rot_deg=rot_deg)
    binned = bin_grid(prep, cell_deg=cell_deg)
    catalog = to_catalog(binned)
    print(f'{parquet}: {len(prep["thx"])} stars -> {len(binned["thx"])} grid cells')
    res, layout, fwd = fitmod.run_fit(cfg, catalog, miw=None, use_svd=True)
    fitted = {n: float(v) for n, v in zip(layout.names, res.x)}
    cw = cwfs_dz(day_obs, infocus_seq)
    print(f'{"coeff":8} {"PSF-fit":>10} {"CWFS":>10}')
    for n in layout.dz_names:
        cwv = cw.get(n, np.nan) if cw else np.nan
        print(f'{n:8} {fitted[n]:10.4f} {cwv:10.4f}')
    for a in ['fwhm', 'g1', 'g2']:
        key = 'atm_' + a
        if key in fitted:
            print(f'{key:8} {fitted[key]:10.4f}')
    return fitted, cw, res, layout
