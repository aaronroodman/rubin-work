"""Convert the binned ISR caches from float32 ``.npy`` to Rice-compressed FITS, in place.

Each ``<visit>/<detector>.npy`` becomes ``<detector>.fits.fz`` (`isr_cache.save_binned`,
quantize level 16). The ``.npy`` is removed only after the FITS reads back with the same
shape and non-finite pixels, and with a residual (FITS − npy) whose maximum in each row is
below ``MAX_FRAC`` of that row's pixel noise (nMAD of neighbour differences / sqrt 2): Rice
quantizes row by row, so a row through a saturated star gets a coarser step. A file that fails
keeps its ``.npy`` and loses the FITS. Prints one line per visit and a total.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/compress_isr_cache.py --workers 32 \\
        20261018_isr_bin4 20261118_night2_20261107/isr_bin4
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import isr_cache  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / 'output'
MAX_FRAC = 0.125  # largest allowed |FITS - npy| over the row's pixel noise; quantization gives ~1/25


def convert(npy, dry_run=False):
    """Convert one file; returns (npy bytes, fits bytes, max |d| / noise, rms d / noise, ok)."""
    a = np.load(npy)
    name = npy.name[:-len('.npy')]
    fz = isr_cache.save_binned(npy.parent, name, a)
    b = _read(fz)
    fin = np.isfinite(a)
    dx = np.diff(a.astype(float), axis=1)
    noise = nmad(dx[np.isfinite(dx)]) / np.sqrt(2)
    ok = b.shape == a.shape and np.array_equal(fin, np.isfinite(b)) and noise > 0
    if ok:
        d = np.where(fin, b - a, 0.0)
        row_noise = np.array([nmad(r[np.isfinite(r)]) for r in dx]) / np.sqrt(2)
        good = row_noise > 0
        mx = float((np.abs(d).max(axis=1)[good] / row_noise[good]).max())
        rms = float(np.sqrt(np.mean(d[fin] ** 2)) / noise)
    else:
        mx = rms = np.inf
    ok = ok and mx < MAX_FRAC
    nb, fb = npy.stat().st_size, fz.stat().st_size
    if dry_run or not ok:
        fz.unlink()
    else:
        npy.unlink()
    return nb, fb, mx, rms, ok


def _read(fz):
    from astropy.io import fits
    with fits.open(fz) as h:
        return h['IMAGE'].data.astype(float)


def visit_task(task):
    d, dry_run = task
    rows = [convert(p, dry_run) for p in sorted(pathlib.Path(d).glob('*.npy'))]
    if not rows:
        return d, 0, 0, 0.0, 0.0, 0, 0
    nb, fb, mx, rms, ok = map(np.array, zip(*rows))
    return d, int(nb.sum()), int(fb.sum()), float(mx[ok].max(initial=0)), \
        float(np.median(rms[ok])) if ok.any() else np.nan, int(ok.sum()), int((~ok).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('caches', nargs='+', help='cache directories under output/, holding <visit>/ dirs')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--dry-run', action='store_true', help='convert and check, keep the npy, drop the FITS')
    a = p.parse_args()
    visits = sorted(str(v) for c in a.caches for v in (OUT / c).iterdir() if v.is_dir())
    tasks = [(v, a.dry_run) for v in visits]
    tot = np.zeros(4)
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context('spawn').Pool(min(a.workers, len(tasks)))
        it = pool.imap_unordered(visit_task, tasks)
    else:
        it = map(visit_task, tasks)
    for d, nb, fb, mx, rms, n_ok, n_bad in it:
        tot += (nb, fb, n_ok, n_bad)
        print(f'{pathlib.Path(d).relative_to(OUT)}: {n_ok} converted, {n_bad} failed; '
              f'{nb / 1e6:.0f} -> {fb / 1e6:.0f} MB; max |d| / noise {mx:.3f}, '
              f'median rms d / noise {rms:.4f} (dimensionless)', flush=True)
    print(f'total: {int(tot[2])} converted, {int(tot[3])} failed; {tot[0] / 1e9:.2f} -> '
          f'{tot[1] / 1e9:.2f} GB{" (dry run, nothing removed)" if a.dry_run else ""}', flush=True)


if __name__ == '__main__':
    main()
