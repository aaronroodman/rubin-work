#!/usr/bin/env python3
"""wfs_mimic — FAM-derived corner-WFS covariance of the measured-intrinsic deviation.

Mimics the four corner wavefront sensors using FAM donuts.  In each image, donuts
in four annular wedges at the WFS radius (centred ``delta + [0, 90, 180, 270]``
deg) give four pseudo-WFS Zernike vectors.  The per-donut **deviation**
(measured ``zk`` minus the OCS/CCS measured-intrinsic from the ``zk_intrinsic``
sidecar) is median-pooled per wedge, so each image yields a (4, nZk) deviation.
Covariance over images then gives the corner-to-corner structure a single WFS
estimate cannot show.

Outputs (per param_set / mi_name, under ``output/<ps>/<mi>/wfs_mimic/``):
    wfs_mimic_cov84.parquet   (4*nZk)x(4*nZk) cross-corner covariance + correlation
    wfs_mimic_cov21.parquet   nZk x nZk single-corner covariance (mean of the four
                              diagonal corner blocks) + correlation
    wfs_mimic_cov_bins.parquet  per rotator subset 84x84 (tidy diagnostic)
    wfs_mimic.pdf             covariance / correlation heatmaps (pooled + per subset)

The 84x84 is sampled over IMAGES (not donuts), so it needs >= 84 images to be
full rank; pooling all kept images is why the rotator-aware OCS/CCS intrinsic
matters (it lets every rotator angle contribute one consistent deviation).

Image (rotator-angle) selection: the analysis_config ``wfs_mimic.rotator_keep``
is a list of ``[lo, hi]`` rotator-angle ranges (deg) to KEEP -- it drops
out-of-family rotator points from the covariance, and is INDEPENDENT of the
OCS/CCS split's ``rotator_select`` (the maps may be built from one subset while
the covariance excludes a different angle).  ``null`` -> keep every image.

Reuses the rotator-aware intrinsic from the zk_intrinsic sidecar, so it tracks
the OCS/CCS MIW automatically.  numpy / pandas / pyarrow only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from astropy.table import Table

from lsst.ts.intrinsic.wavefront import mi_config as mc

DEFAULT = dict(wfs_inner_radius_deg=1.60, wfs_outer_radius_deg=1.725,
               wfs_azimuth_width_deg=7.5, delta_deg=0.0,
               min_donuts_per_wedge=3, rotator_keep=None)
CORNER_OFFSETS = [0.0, 90.0, 180.0, 270.0]
N_CORNERS = len(CORNER_OFFSETS)


def _in_ranges(angle, ranges):
    """True if scalar ``angle`` falls in any (lo, hi) range; ranges None -> all."""
    if ranges is None:
        return True
    if not np.isfinite(angle):
        return False
    return any(lo <= angle <= hi for lo, hi in ranges)


def _wedge_medians(dev, thx_deg, thy_deg, sec):
    """Median deviation in each of the four WFS wedges for one image.

    ``dev`` is (n_donut, nZk).  Returns (N_CORNERS, nZk) of medians (NaN-aware)
    or ``None`` if any wedge has fewer than ``min_donuts_per_wedge`` donuts."""
    r = np.hypot(thx_deg, thy_deg)
    az = np.degrees(np.arctan2(thy_deg, thx_deg)) % 360.0
    radial = (r >= sec['wfs_inner_radius_deg']) & (r <= sec['wfs_outer_radius_deg'])
    half = sec['wfs_azimuth_width_deg'] / 2.0
    out = np.full((N_CORNERS, dev.shape[1]), np.nan)
    for i, off in enumerate(CORNER_OFFSETS):
        c = (sec['delta_deg'] + off) % 360.0
        lo, hi = (c - half) % 360.0, (c + half) % 360.0
        az_in = (az >= lo) & (az <= hi) if lo < hi else (az >= lo) | (az <= hi)
        w = radial & az_in
        if int(w.sum()) < sec['min_donuts_per_wedge']:
            return None
        out[i] = np.nanmedian(dev[w], axis=0)
    return out


def _image_matrix(dev, thx_deg, thy_deg, day, seq, rot_lut, ranges, sec):
    """Stack per-image flattened (N_CORNERS*nZk) deviation vectors over the kept
    images.  Returns (n_img, N_CORNERS*nZk); only images with all four wedges
    populated and a rotator angle inside ``ranges`` are kept."""
    rows = []
    order = np.lexsort((seq, day))
    keys = list(zip(day[order], seq[order]))
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            idx = order[start:i]
            d, s = keys[start]
            start = i
            if not _in_ranges(rot_lut.get((int(d), int(s)), np.nan), ranges):
                continue
            wm = _wedge_medians(dev[idx], thx_deg[idx], thy_deg[idx], sec)
            if wm is None or not np.all(np.isfinite(wm)):
                continue
            rows.append(wm.ravel())              # corner-major: c0(all j), c1, ...
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 0))


def _cov_corr(M):
    """Covariance + correlation over the rows (images) of M; (None, None, n) if
    fewer than 2 samples."""
    if M.ndim != 2 or M.shape[0] < 2:
        return None, None, int(M.shape[0]) if M.ndim == 2 else 0
    cov = np.cov(M, rowvar=False)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.outer(sd, sd)
    return cov, corr, M.shape[0]


def _corner_block_mean(cov, nZk):
    """Mean of the N_CORNERS diagonal (nZk x nZk) blocks of an 84x84 covariance
    -> the single-corner covariance comparable to the donut-sampled 21x21."""
    return np.mean([cov[c * nZk:(c + 1) * nZk, c * nZk:(c + 1) * nZk]
                    for c in range(N_CORNERS)], axis=0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--config', default=None)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()
    coord = args.coord_sys

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    sec = {**DEFAULT, **mc.analysis_section(
        'wfs_mimic', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    ranges = ([(float(lo), float(hi)) for lo, hi in sec['rotator_keep']]
              if sec.get('rotator_keep') else None)
    base_ps = Path(args.output_root) / args.param_set
    base_mi = base_ps / args.mi_name
    out = base_mi / 'wfs_mimic'
    out.mkdir(parents=True, exist_ok=True)

    # ---- measured per-donut zk (file order) ----
    dd = pq.read_table(str(base_ps / 'donuts.parquet')).to_pandas()
    zk_meas = np.stack(dd[f'zk_{coord}'].values).astype(float)        # (N, nzk_m)
    day = np.asarray(dd['day_obs']).astype(int)
    seq = np.asarray(dd['seq_num']).astype(int)
    thx_deg = np.rad2deg(np.asarray(dd[f'thx_{coord}'], dtype=float))
    thy_deg = np.rad2deg(np.asarray(dd[f'thy_{coord}'], dtype=float))

    # ---- visits: rotator angle per image + measured Noll ordering ----
    vt = pq.read_table(str(base_ps / 'visits.parquet')).to_pandas()
    rot_lut = {(int(r.day_obs), int(r.seq_num)): float(r.rotator_angle)
               for r in vt.itertuples()}
    if 'nollIndices' in vt.columns:
        noll_m = [int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]
    else:
        noll_m = list(range(4, 4 + zk_meas.shape[1]))

    # ---- intrinsic sidecar (row-aligned to donuts.parquet) ----
    sc = pq.read_table(str(base_mi / 'zk_intrinsic.parquet'))
    md = sc.schema.metadata or {}
    noll_i = (np.frombuffer(md[b'nollIndices'], dtype=int).tolist()
              if b'nollIndices' in md else noll_m)
    scp = sc.to_pandas()
    zk_int = np.stack(scp['zk_intrinsic_MI'].values).astype(float)    # (N, nzk_i)
    if len(scp) != len(dd):
        sys.exit(f'ERROR: sidecar rows ({len(scp)}) != donuts rows ({len(dd)})')
    if not (np.array_equal(np.asarray(scp['day_obs']).astype(int), day)
            and np.array_equal(np.asarray(scp['seq_num']).astype(int), seq)):
        sys.exit('ERROR: sidecar is not row-aligned to donuts.parquet '
                 '(day_obs/seq_num mismatch) -- rerun the intrinsic_sidecar step.')

    # ---- common Noll ordering; per-donut deviation = measured - intrinsic ----
    noll = sorted(set(noll_m) & set(noll_i))
    im = [noll_m.index(j) for j in noll]
    ii = [noll_i.index(j) for j in noll]
    dev = zk_meas[:, im] - zk_int[:, ii]                              # (N, nZk)
    nZk = len(noll)
    print(f'[wfs_mimic] {args.param_set}/{args.mi_name}: {len(dd)} donuts, '
          f'{nZk} Zernikes {noll}')
    print(f'  rotator_keep = {ranges}   wedge r=[{sec["wfs_inner_radius_deg"]},'
          f'{sec["wfs_outer_radius_deg"]}]deg, +/-{sec["wfs_azimuth_width_deg"]/2}deg, '
          f'delta={sec["delta_deg"]}deg, min_donuts/wedge={sec["min_donuts_per_wedge"]}')

    # ---- corner centres (deg) for metadata ----
    r_mid = 0.5 * (sec['wfs_inner_radius_deg'] + sec['wfs_outer_radius_deg'])
    cz = [np.radians((sec['delta_deg'] + o) % 360.0) for o in CORNER_OFFSETS]
    corner_thx = [float(r_mid * np.cos(a)) for a in cz]
    corner_thy = [float(r_mid * np.sin(a)) for a in cz]
    labels84 = [f'Z{j}_c{c}' for c in range(N_CORNERS) for j in noll]
    labels21 = [f'Z{j}' for j in noll]

    # ---- pooled (all kept images) ----
    M = _image_matrix(dev, thx_deg, thy_deg, day, seq, rot_lut, ranges, sec)
    cov84, corr84, n_img = _cov_corr(M)
    if cov84 is None:
        sys.exit(f'ERROR: only {n_img} usable images -- cannot form a covariance.')
    rank = int(np.linalg.matrix_rank(cov84))
    cov21 = _corner_block_mean(cov84, nZk)
    sd21 = np.sqrt(np.clip(np.diag(cov21), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr21 = cov21 / np.outer(sd21, sd21)
    print(f'  pooled: {n_img} images -> 84x84 (rank {rank}/{cov84.shape[0]})'
          + ('  [rank-deficient: need >= 84 images]' if rank < cov84.shape[0] else ''))

    base_meta = dict(
        mi_name=args.mi_name, coord_sys=coord, n_images=int(n_img),
        rank=rank, quantity='deviation = measured - OCS/CCS measured-intrinsic',
        cov_units='um^2', noll=[int(j) for j in noll], n_corners=N_CORNERS,
        corner_offsets_deg=[float(o) for o in CORNER_OFFSETS],
        corner_thx_deg=corner_thx, corner_thy_deg=corner_thy,
        rotator_keep=[[float(lo), float(hi)] for lo, hi in ranges] if ranges else None,
        wfs_inner_radius_deg=float(sec['wfs_inner_radius_deg']),
        wfs_outer_radius_deg=float(sec['wfs_outer_radius_deg']),
        wfs_azimuth_width_deg=float(sec['wfs_azimuth_width_deg']),
        delta_deg=float(sec['delta_deg']),
        min_donuts_per_wedge=int(sec['min_donuts_per_wedge']))

    # ---- write the two colleague-facing matrices ----
    t84 = Table(dict(
        label=labels84,
        corner=[c for c in range(N_CORNERS) for _ in noll],
        j=[int(j) for _ in range(N_CORNERS) for j in noll],
        cov=[cov84[r].astype(np.float64) for r in range(cov84.shape[0])],
        corr=[corr84[r].astype(np.float64) for r in range(corr84.shape[0])]))
    t84.meta.update({**base_meta, 'labels': labels84})
    t84.write(str(out / 'wfs_mimic_cov84.parquet'), format='parquet', overwrite=True)

    t21 = Table(dict(
        label=labels21, j=[int(j) for j in noll],
        cov=[cov21[r].astype(np.float64) for r in range(nZk)],
        corr=[corr21[r].astype(np.float64) for r in range(nZk)]))
    t21.meta.update({**base_meta, 'labels': labels21,
                     'note': 'single-corner covariance = mean of the four diagonal '
                             'corner blocks of the 84x84'})
    t21.write(str(out / 'wfs_mimic_cov21.parquet'), format='parquet', overwrite=True)

    # ---- per rotator subset (diagnostic): pooled + each kept/config bin ----
    diag_bins = ranges if ranges is not None else mc.rotator_bins(cfg)
    subsets = [('pooled', ranges)] + [
        (f'rot_{lo:g}_{hi:g}', [(lo, hi)]) for lo, hi in diag_bins]
    rows = dict(subset=[], n_images=[], rank=[], label=[], cov=[], corr=[])
    for name, rng in subsets:
        Ms = _image_matrix(dev, thx_deg, thy_deg, day, seq, rot_lut, rng, sec)
        cv, cr, ni = _cov_corr(Ms)
        if cv is None:
            print(f'  subset {name}: {ni} images -- skipped (need >= 2)')
            continue
        rk = int(np.linalg.matrix_rank(cv))
        for r in range(cv.shape[0]):
            rows['subset'].append(name); rows['n_images'].append(int(ni))
            rows['rank'].append(rk); rows['label'].append(labels84[r])
            rows['cov'].append(cv[r].astype(np.float64))
            rows['corr'].append(cr[r].astype(np.float64))
        print(f'  subset {name}: {ni} images (rank {rk}/{cv.shape[0]})')
    tb = Table(rows)
    tb.meta.update({**base_meta, 'labels': labels84})
    tb.write(str(out / 'wfs_mimic_cov_bins.parquet'), format='parquet', overwrite=True)

    _heatmap_pdf(out / 'wfs_mimic.pdf', cov84, corr84, cov21, corr21,
                 labels84, labels21, base_meta)
    print(f'  wrote wfs_mimic_cov84.parquet, wfs_mimic_cov21.parquet, '
          f'wfs_mimic_cov_bins.parquet, wfs_mimic.pdf')


def _heatmap_pdf(path, cov84, corr84, cov21, corr21, labels84, labels21, meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def _panel(ax, A, title, labels, diverging, unit):
        finite = A[np.isfinite(A)]
        vmax = (float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0) or 1e-9
        vmin = -vmax if diverging else 0.0
        if diverging:
            im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(A, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        step = max(1, len(labels) // 28)
        ticks = range(0, len(labels), step)
        ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks))
        ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=5)
        ax.set_yticklabels([labels[i] for i in ticks], fontsize=5)
        plt.colorbar(im, ax=ax, shrink=0.8, label=unit)

    with PdfPages(str(path)) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 14), layout='constrained')
        _panel(axes[0, 0], cov84, '84x84 covariance', labels84, True, 'um^2')
        _panel(axes[0, 1], corr84, '84x84 correlation', labels84, True, 'r')
        _panel(axes[1, 0], cov21, '21x21 covariance (corner-mean)', labels21, True, 'um^2')
        _panel(axes[1, 1], corr21, '21x21 correlation (corner-mean)', labels21, True, 'r')
        fig.suptitle(f"WFS-mimic deviation covariance  ({meta['n_images']} images, "
                     f"rank {meta['rank']}/{cov84.shape[0]})\n"
                     f"rotator_keep={meta['rotator_keep']}", fontsize=12)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
