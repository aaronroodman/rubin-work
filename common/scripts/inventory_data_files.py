"""Inventory data files under the output trees and the code that references them.

Finds every data file (parquet, DuckDB, npz, FITS, HDF5, CSV) under the given
roots, groups them by file name, and lists which topic/study code mentions that
name. A name referenced from two or more code units is a candidate product.

Stdlib only. Run from anywhere:

    python common/scripts/inventory_data_files.py
    python common/scripts/inventory_data_files.py --roots /some/dir --csv out.csv

Default roots are the repository's ``*/output`` directories (symlinks followed)
and ``/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work`` when it exists.
"""
import argparse
import collections
import csv
import os
import pathlib
import re
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_EXT = {'.parquet', '.duckdb', '.db', '.npz', '.npy', '.fits', '.h5', '.hdf5', '.csv', '.pkl'}
CODE_EXT = {'.py', '.ipynb', '.yaml', '.yml', '.sh'}
SDF_ROOT = pathlib.Path('/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work')


def code_unit(path):
    """``topic/study`` for a code file, ``topic`` for flat or top-level files."""
    parts = path.relative_to(REPO).parts
    if len(parts) > 3 and parts[1] in ('code', 'notebooks'):
        return f'{parts[0]}/{parts[2]}'
    return parts[0] if len(parts) > 1 else '(root)'


def generic_name(name):
    """Collapse dates, day_obs, seq_nums and chunk indices so per-night files group."""
    return re.sub(r'\d{3,}', 'N', name)


def scan_data(roots):
    groups = collections.defaultdict(lambda: {'n': 0, 'bytes': 0, 'dirs': set(), 'newest': 0.0})
    seen = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            real = os.path.realpath(dirpath)
            if real in seen:
                dirnames[:] = []
                continue
            seen.add(real)
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for fn in filenames:
                p = pathlib.Path(dirpath, fn)
                if p.suffix.lower() not in DATA_EXT:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                g = groups[generic_name(fn)]
                g['n'] += 1
                g['bytes'] += st.st_size
                g['newest'] = max(g['newest'], st.st_mtime)
                g['dirs'].add(re.sub(r'\d{3,}', 'N', str(pathlib.Path(dirpath).relative_to(root))))
                g.setdefault('example', fn)
    return groups


REF = re.compile(r'[\w{}\-.]*\.(?:parquet|duckdb|db|npz|npy|fits|h5|hdf5|csv|pkl)\b')


def index_code_references():
    """Map each data-file name mentioned in code (generic form) to the code units citing it."""
    index = collections.defaultdict(set)
    for p in REPO.rglob('*'):
        if p.suffix not in CODE_EXT or '.git' in p.parts or 'output' in p.parts:
            continue
        try:
            text = p.read_text(errors='ignore')
        except OSError:
            continue
        unit = code_unit(p)
        for tok in set(REF.findall(text)):
            index[generic_name(re.sub(r'\{[^}]*\}', '000', tok))].add(unit)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--roots', nargs='*', type=pathlib.Path)
    ap.add_argument('--csv', type=pathlib.Path, default=pathlib.Path('data_inventory.csv'))
    args = ap.parse_args()

    roots = args.roots or sorted(p for p in REPO.glob('*/output') if p.exists())
    if not args.roots and SDF_ROOT.exists():
        roots.append(SDF_ROOT)
    print('roots:', *roots, sep='\n  ')

    groups = scan_data(roots)
    index = index_code_references()
    rows = []
    for name, g in groups.items():
        units = index.get(name, set())
        rows.append({
            'file_name': name, 'n_files': g['n'], 'size_GB': round(g['bytes'] / 1e9, 3),
            'newest': time.strftime('%Y-%m-%d', time.localtime(g['newest'])),
            'n_units': len(units), 'units': ' '.join(sorted(units)),
            'dirs': ' | '.join(sorted(g['dirs'])[:4]) + (' | ...' if len(g['dirs']) > 4 else ''),
        })
    rows.sort(key=lambda r: (-r['n_units'], -r['size_GB']))

    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['file_name'])
        w.writeheader()
        w.writerows(rows)

    print(f'\n{"file name":40s} {"files":>6s} {"size (GB)":>9s} {"newest":>10s} {"units":>5s}  referenced by')
    for r in rows:
        if r['n_units'] >= 2:
            print(f'{r["file_name"][:40]:40s} {r["n_files"]:6d} {r["size_GB"]:9.3f} {r["newest"]:>10s} '
                  f'{r["n_units"]:5d}  {r["units"]}')
    n_shared = sum(r['n_units'] >= 2 for r in rows)
    total = sum(r['size_GB'] for r in rows)
    print(f'\n{len(rows)} distinct file names, {total:.1f} GB in total; '
          f'{n_shared} referenced by 2 or more code units (shown above).')
    print(f'Full table, including single-unit and unreferenced files: {args.csv.resolve()}')


if __name__ == '__main__':
    main()
