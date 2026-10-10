"""Write the ``manifest.json`` that finishes a ``fam_tables`` build.

The last step of either build path -- the Snakefile's ``manifest`` rule and
``run_blitz_mktable`` -- so a build directory becomes a catalog entry only after
every table is on disk. Nothing here moves the variant's ``current`` symlink;
that is ``python -m rubinwork.products.catalog set-current``.

The manifest's ``config`` is the variant's whole ``variants.yaml`` entry, so a
composite variant such as ``danish_1_2`` records its **per-chunk** collections,
Butler repos and program filters rather than one collection field that would be
wrong for most of its visits.

Command line, used by the Snakefile rule::

    python -m rubinwork.products.fam_tables.write_manifest \\
        --variant danish_1_2 --build 20261010
"""

import argparse
import pathlib
import sys

from .. import manifest
from . import reader

__all__ = ["build_config", "write", "main"]


def build_config(variant, extra=None):
    """The manifest ``config`` for one variant: its expanded ``variants.yaml`` entry.

    Parameters
    ----------
    variant : `str`
        Variant name.
    extra : `dict`, optional
        Build-time settings to record alongside the variant entry -- the
        command-line flags of the run, which are not in ``variants.yaml``.

    Returns
    -------
    config : `dict`
        A copy of the variant entry, with ``variant`` and (when given) a
        ``build_options`` key holding `extra`.
    """
    config = dict(reader.variant_config(variant))
    config["variant"] = variant
    if extra:
        config["build_options"] = dict(extra)
    return config


def write(variant, build=None, build_dir=None, status="complete",
          build_options=None, inputs=None):
    """Write the manifest of one ``fam_tables`` build.

    Parameters
    ----------
    variant : `str`
        Variant name, or the long ``param_set`` key.
    build : `str`, optional
        Build name, normally the date (``"20261010"``). Taken from
        `build_dir`'s name when that is given instead.
    build_dir : `str` or `pathlib.Path`, optional
        The build directory. By default the catalog directory of
        ``variant``/``build``, which is where a normal build writes.
    status : `str`, optional
        Manifest ``status``, ``"complete"`` by default. Anything else keeps the
        build out of the catalog.
    build_options : `dict`, optional
        Build-time settings, recorded under ``config.build_options``.
    inputs : `dict`, optional
        Upstream builds read, as ``{product: "variant@build"}``.

    Returns
    -------
    manifest_path : `pathlib.Path`
        The manifest that was written.

    Notes
    -----
    Row counts come from the parquet files themselves
    (`rubinwork.products.manifest.file_stats`), not from the builder's own
    counters, so the manifest reports what is on disk.
    """
    variant = reader.variant_for_param_set(variant)
    if build_dir is None:
        if not build:
            raise ValueError("write() needs `build` or `build_dir`")
        build_dir = reader.build_dir(variant, build)
    build_dir = pathlib.Path(build_dir)
    build = build or build_dir.name
    return manifest.write(
        build_dir, product=reader.PRODUCT, variant=variant, build=build,
        config=build_config(variant, extra=build_options),
        inputs=inputs, status=status)


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.fam_tables.write_manifest",
        description="Write manifest.json for a fam_tables build, as its last step.")
    parser.add_argument("--variant", required=True,
                        help="Variant name, or the long param_set key")
    parser.add_argument("--build", default=None,
                        help="Build name (YYYYMMDD); taken from --build-dir if omitted")
    parser.add_argument("--build-dir", default=None,
                        help="Build directory (default: the catalog directory of "
                             "<variant>/<build>)")
    parser.add_argument("--status", default="complete",
                        help="Manifest status (default: complete)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Build-time setting to record under config.build_options; "
                             "repeatable")
    args = parser.parse_args(argv)

    options = {}
    for item in args.option:
        key, _, value = item.partition("=")
        options[key] = value
    path = write(args.variant, build=args.build, build_dir=args.build_dir,
                 status=args.status, build_options=options or None)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
