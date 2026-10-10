"""Generate ``aos/param_sets.yaml`` from the two products' ``variants.yaml`` files.

The generated file is a merge of two sources of truth, each owning its own half:

``rubinwork/products/fam_tables/variants.yaml``
    the Full Array Mode (FAM) configuration -- collections, programs, date chunks.
``rubinwork/products/cwfs_tables/variants.yaml``
    the in-focus corner-wavefront-sensor (CWFS) collections, which were the
    ``wfs_collections`` maps here until phase 3a of
    ``notes/status/organization_plan.md``.

This script writes the legacy ``aos/param_sets.yaml`` view of both, keyed by the long
``param_set`` name. The CWFS half comes back through
`rubinwork.products.cwfs_tables.wfs_collections_for`, keyed by each CWFS variant's
``upstream_name``, so the generated file is unchanged by the move.

The legacy file cannot be retired, now or in phase 5: the external package
``lsst.ts.intrinsic.wavefront.intrinsics_lib.load_param_sets`` opens
``Path('param_sets.yaml')`` relative to the current working directory, and four scripts in
``aos/code/`` call it.

Run from anywhere::

    python -m rubinwork.products.fam_tables.gen_param_sets

``--check`` exits non-zero if the file on disk differs from what would be generated, for
use in a test or a pre-commit hook.
"""

import argparse
import pathlib
import sys

import yaml

from rubinwork.common.utils import repo_root
from .reader import variant_config, variants

# Fields copied straight through to the generated entry, in the order they are written.
# `dir_name` is derived from the variant name instead, and the product-only keys
# (`param_set`, `builder`, `chunks`, `coord_sys`, `registered`) are dropped: chunks live in
# aos/snake_config.yaml and coord_sys is read from there too.
#
# `wfs_collections` is NOT here: it is owned by the cwfs_tables product and injected at
# its historical position, between `fam_collections` and `collection_phrase`.
PASSTHROUGH = (
    "description",
    "butler_repo",
    "fam_programs",
    "fam_collections",
    "collection_phrase",
    "day_obs_min",
    "day_obs_max",
)

WFS_AFTER = "fam_collections"
"""The generated entry writes ``wfs_collections`` immediately after this field.

Its position has to be preserved, not merely its content: ``aos/param_sets.yaml`` is
compared byte for byte across the phase 3a move.
"""

HEADER = """\
# GENERATED FILE — DO NOT EDIT BY HAND.
#
# Source of truth: rubinwork/products/fam_tables/variants.yaml
# Regenerate with:
#
#     python -m rubinwork.products.fam_tables.gen_param_sets
#
# Every comment explaining these entries lives in variants.yaml. This file exists because
# the external package lsst.ts.intrinsic.wavefront.intrinsics_lib.load_param_sets() opens
# Path('param_sets.yaml') relative to the current working directory, and
# aos/code/cwfs/run_wfs_mktable.py, run_wfs_fam_compare.py, run_wfs_refit_ensemble.py and
# aos/code/fam_processing/check_chunk.py all call it. Editing this file instead of
# variants.yaml means the next regeneration silently discards the change.
#
# The key is the long param_set name, which stays the identity for everything that
# recorded it: the value-added DB rows, the frozen provenance files, and --param-set on
# the command line. `dir_name` is the short output-directory name.
"""


def _entry(variant):
    """The generated ``param_sets.yaml`` entry for one variant.

    Parameters
    ----------
    variant : `str`
        Variant name in ``variants.yaml``, e.g. ``"danish_1_2"``.

    Returns
    -------
    param_set : `str`
        The long key the entry is filed under.
    entry : `dict`
        The entry body, with ``dir_name`` first and the ``cwfs_tables``-owned
        ``wfs_collections`` map at its historical position.
    """
    from rubinwork.products import cwfs_tables

    cfg = variant_config(variant)
    wfs = cwfs_tables.wfs_collections_for(variant)
    entry = {"dir_name": variant}
    for key in PASSTHROUGH:
        if key in cfg:
            entry[key] = cfg[key]
        if key == WFS_AFTER and wfs:
            entry["wfs_collections"] = wfs
    return cfg["param_set"], entry


def render():
    """The full text of the generated ``param_sets.yaml``.

    Returns
    -------
    text : `str`
        Header comment followed by one block per variant, including the unregistered
        ones, in ``variants.yaml`` order.
    """
    parts = [HEADER]
    for variant in variants(registered_only=False):
        param_set, entry = _entry(variant)
        block = yaml.safe_dump({param_set: entry}, sort_keys=False,
                               default_flow_style=False, width=100)
        parts.append("\n" + block)
    return "".join(parts)


def default_path():
    """Path of the generated file, ``aos/param_sets.yaml`` under the repo root."""
    return pathlib.Path(repo_root()) / "aos" / "param_sets.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="output path (default: aos/param_sets.yaml under the repo root)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file on disk differs; write nothing")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out) if args.out else default_path()
    text = render()

    if args.check:
        current = out.read_text() if out.exists() else ""
        if current == text:
            print(f"{out} is up to date")
            return 0
        print(f"{out} is STALE — regenerate with "
              f"python -m rubinwork.products.fam_tables.gen_param_sets")
        return 1

    out.write_text(text)
    n = len(variants(registered_only=False))
    print(f"wrote {out} ({n} param_sets, dimensionless count)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
