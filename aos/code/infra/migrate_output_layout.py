#!/usr/bin/env python3
"""One-shot migration of ``aos/output/`` to the repo output convention.

The old layout nested the two data axes and put the study innermost::

    output/<param_set>/<mi_name>/<study>/...

The convention in the root ``CLAUDE.md`` is study outermost, exactly one data
level, and the data axes joined into a single directory name::

    output/<study>/<param_set_dir>[_<mi_dir>]/...

``<param_set_dir>`` and ``<mi_dir>`` are the short ``dir_name`` values from
``param_sets.yaml`` and ``mi_config.yaml``; the long keys stay the identity that
``--param-set``, the ``value_added`` database rows and the frozen provenance
resolve against.  This script reads the same config the Snakefile reads, so the
mapping cannot drift from it.

Every move is an ``os.rename`` of a single file — no copy and no delete — so a
37 GB tree migrates in seconds provided source and destination share one
filesystem, which the script verifies via ``st_dev`` and refuses otherwise.
Emptied directories are left in place to be removed by hand.
``output/archive/`` is never touched; its layout is frozen provenance.

Usage (dry run is the default; it prints every move and the collision count)::

    python code/infra/migrate_output_layout.py
    python code/infra/migrate_output_layout.py --apply

Notes
-----
Two naming mismatches are fixed in passing: the on-disk ``closedloop/`` becomes
``closed_loop/`` to match ``code/closed_loop/``, and ``coadd_50_34`` /
``coadd_50_34_v2`` are coadd *variants* rather than studies, so they nest as
``coadd/<data>/{50_34,50_34_v2}/``.

The coadd's ``mi_name`` dependence is not recorded anywhere in the old tree —
neither the directory name nor the archive provenance notes state which MIW build
was used — so these move to the param_set-only data level
``coadd/<param_set_dir>/<variant>/`` rather than being labelled with a guessed
MIW tag.  A rerun writes the joined ``<P>_<M>`` name that the Snakefile declares.

``.ipynb_checkpoints/`` directories are skipped: they are gitignored editor
strays, not output, and are left in place with the emptied directories.
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

import yaml

from lsst.ts.intrinsic.wavefront import mi_config as mc

TOPIC = pathlib.Path(__file__).resolve().parents[2]          # aos/

# ---------------------------------------------------------------------------
# Study assignment for the directories that sat under a param_set or a MIW
# entry.  Key is the old directory name; value is the new study directory.
# A study whose name already matches needs no entry but is listed for clarity.
# ---------------------------------------------------------------------------
PS_STUDY = {                       # output/<ps>/<dir>/  ->  output/<study>/<P>/
    'chunks':             'fam_processing',      # stays a subdir: see ps_map()
    'fam_processing':     'fam_processing',
    'dzfit':              'dzfit',
    'correlations':       'correlations',
    'processing_compare': 'processing_compare',
}
MI_STUDY = {                       # output/<ps>/<mi>/<dir>/ -> output/<study>/<P>_<M>/
    'bounce':       'bounce',
    'correlations': 'correlations',
    'lut':          'lut',
    'psf':          'psf',
    'closedloop':   'closed_loop',               # match code/closed_loop/
    'wfs_mimic':    'wfs_mimic',
    'build':        'miw',                       # stays a subdir: see mi_map()
}
# Phase-1 tables and the loose per-param_set products, all phase-1.
PS_FILES = {
    'donuts.parquet':   'fam_processing',
    'visits.parquet':   'fam_processing',
    'fits.parquet':     'fam_processing',
    'visits_check.pdf': 'fam_processing',
}
# The MIW build's own products, loose in the old <ps>/<mi>/ directory.
MI_FILES_STUDY = 'miw'
# Files in <ps>/wfs/<variant>/ split by which study produced them: the ingest
# products (run_wfs_mktable) against the comparison (run_wfs_corner_compare).
WFS_INGEST_FILES = {'donuts.parquet', 'visits.parquet',
                    'wfs_mktable_validation.pdf',
                    'wfs_mktable_validation_bydonutave.pdf'}
# Loose files in <ps>/wfs/ belong to whichever study wrote them, not to the ingest.
WFS_LOOSE_STUDY = {'fam_wfs_triplet_compare.pdf': 'wfs_fam_compare'}
SKIP_DIRS = {'.ipynb_checkpoints'}          # gitignored editor strays, not output


def load_dir_names():
    """Return ``(ps_dir, mi_dir)`` mappings from the two config files.

    Returns
    -------
    ps_dir : `dict`
        param_set key -> short output directory name.
    mi_dir : `dict`
        ``(param_set key, mi_name key)`` -> short output directory name.
    """
    psets = yaml.safe_load((TOPIC / 'param_sets.yaml').read_text())
    psets = psets.get('param_sets', psets)
    ps_dir = {k: (v or {}).get('dir_name') or k for k, v in psets.items()}

    # mi_config entries are a per-param_set LIST keyed by `name`; go through the
    # package's own accessors so this cannot drift from what the Snakefile reads.
    doc = mc.load_doc(TOPIC / 'mi_config.yaml')
    mi_dir = {}
    for ps in ps_dir:
        for mi in mc.mi_names(ps, doc=doc):
            cfg = mc.load_mi_config(ps, mi, doc=doc)
            mi_dir[(ps, mi)] = mc.dir_name(cfg, mi)
    return ps_dir, mi_dir


def ps_map(rel, ps, P):
    """New path for a file that lived under ``output/<ps>/`` but not under a
    MIW entry.  ``rel`` is the path relative to that param_set directory.

    Returns `None` when the file has no mapping (caller reports it).
    """
    parts = rel.parts
    if len(parts) == 1:                                  # loose phase-1 product
        study = PS_FILES.get(parts[0])
        return None if study is None else f'{study}/{P}/{parts[0]}'
    head, rest = parts[0], '/'.join(parts[1:])
    if head == 'chunks':                                 # keep the chunks/ level
        return f'fam_processing/{P}/chunks/{rest}'
    if head == 'wfs':                                    # CWFS: variant nests
        if len(parts) < 3:                               # loose file in wfs/
            study = WFS_LOOSE_STUDY.get(parts[-1])
            return None if study is None else f'{study}/{P}/{rest}'
        variant, fname = parts[1], '/'.join(parts[2:])
        study = ('wfs_ingest' if parts[-1] in WFS_INGEST_FILES
                 else 'wfs_corner_compare')
        return f'{study}/{P}/{variant}/{fname}'
    if head.startswith('coadd_'):                        # variant, not a study
        return f'coadd/{P}/{head[len("coadd_"):]}/{rest}'
    study = PS_STUDY.get(head)
    return None if study is None else f'{study}/{P}/{rest}'


def mi_map(rel, D):
    """New path for a file under ``output/<ps>/<mi>/``.  ``rel`` is relative to
    that MIW directory and ``D`` is the joined ``<P>_<M>`` directory name."""
    parts = rel.parts
    if len(parts) == 1:                                  # the MIW build itself
        return f'{MI_FILES_STUDY}/{D}/{parts[0]}'
    head, rest = parts[0], '/'.join(parts[1:])
    if head == 'build':                                  # keep the build/ level
        return f'miw/{D}/build/{rest}'
    if head == 'wfs':                                    # CWFS: variant nests
        if len(parts) < 3:
            return f'wfs_dof_compare/{D}/{rest}'
        return f'wfs_dof_compare/{D}/{parts[1]}/' + '/'.join(parts[2:])
    if head == 'plots':                                  # the coadd-vs-MIW maps
        return f'coadd/{D}/{rest}'
    study = MI_STUDY.get(head)
    return None if study is None else f'{study}/{D}/{rest}'


def build_moves(out, ps_dir, mi_dir):
    """Walk the old tree and return ``(moves, unmapped)``.

    Parameters
    ----------
    out : `pathlib.Path`
        The ``aos/output`` directory.

    Returns
    -------
    moves : `list`
        ``(src, dst)`` `pathlib.Path` pairs, both absolute.
    unmapped : `list`
        Files under a param_set directory for which no rule matched.
    """
    moves, unmapped = [], []
    for ps, P in sorted(ps_dir.items()):
        root = out / ps
        if not root.is_dir():
            continue
        mi_names = {mi for (p, mi) in mi_dir if p == ps and (root / mi).is_dir()}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS]
            d = pathlib.Path(dirpath)
            rel = d.relative_to(root)
            mi = rel.parts[0] if rel.parts and rel.parts[0] in mi_names else None
            for fn in filenames:
                src = d / fn
                if mi is None:
                    new = ps_map(rel / fn, ps, P)
                else:
                    D = f'{P}_{mi_dir[(ps, mi)]}'
                    new = mi_map((rel / fn).relative_to(mi), D)
                if new is None:
                    unmapped.append(src)
                else:
                    moves.append((src, out / new))
    return moves, unmapped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='perform the renames (default: dry run, print only)')
    ap.add_argument('--output', default=None,
                    help='output directory to migrate (default: aos/output)')
    ap.add_argument('--quiet', action='store_true',
                    help='print only the summary, not every move')
    args = ap.parse_args()

    out = pathlib.Path(args.output) if args.output else TOPIC / 'output'
    if not out.is_dir():
        raise SystemExit(f'no such output directory: {out}')
    ps_dir, mi_dir = load_dir_names()
    print(f'[migrate_output_layout] {out}'
          f'{"  (DRY RUN — nothing is moved)" if not args.apply else ""}')
    for ps, P in sorted(ps_dir.items()):
        mark = 'present' if (out / ps).is_dir() else 'absent'
        print(f'  param_set {ps}  ->  {P}   ({mark} on disk)')

    moves, unmapped = build_moves(out, ps_dir, mi_dir)

    # ---- collisions: two sources landing on one destination, or a
    # destination that already exists outside the set being moved ----
    seen, collide = {}, []
    srcs = {s for s, _ in moves}
    for src, dst in moves:
        if dst in seen:
            collide.append((seen[dst], src, dst))
        seen[dst] = src
        if dst.exists() and dst not in srcs:
            collide.append((None, src, dst))

    if not args.quiet:
        for src, dst in moves:
            print(f'  {src.relative_to(out)}  ->  {dst.relative_to(out)}')
    if unmapped:
        print(f'\n  {len(unmapped)} file(s) with NO mapping (left in place):')
        for p in unmapped:
            print(f'    {p.relative_to(out)}')
    if collide:
        print(f'\n  {len(collide)} COLLISION(S):')
        for a, b, dst in collide:
            other = a.relative_to(out) if a else '(already on disk)'
            print(f'    {dst.relative_to(out)}  <-  {b.relative_to(out)}  '
                  f'and {other}')

    # Account for every file in the tree, so a mapping gap cannot hide.
    n_skip = sum(1 for ps in ps_dir if (out / ps).is_dir()
                 for p in (out / ps).rglob('*')
                 if p.is_file() and any(s in p.parts for s in SKIP_DIRS))
    n_bytes = sum(s.stat().st_size for s, _ in moves)
    print(f'\n  {len(moves)} file(s), {n_bytes / 2**30:.2f} GiB, '
          f'{len(unmapped)} unmapped, {n_skip} skipped '
          f'({"/".join(sorted(SKIP_DIRS))}), {len(collide)} collision(s)')

    if not args.apply:
        print('  dry run — re-run with --apply to perform the renames')
        return
    if collide:
        raise SystemExit('refusing to move anything while collisions remain')

    # os.rename cannot cross filesystems; a mismatch would mean a 37 GB copy.
    dev = out.stat().st_dev
    for src, _ in moves:
        if src.stat().st_dev != dev:
            raise SystemExit(f'{src} is on a different filesystem than {out} — '
                             f'refusing to rename across devices')

    n = 0
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        n += 1
    print(f'  moved {n} file(s).  The emptied directories are left in place; '
          f'remove them by hand once you have checked them.')


if __name__ == '__main__':
    main()
