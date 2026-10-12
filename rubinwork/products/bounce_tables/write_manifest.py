"""Write the ``manifest.json`` that finishes a ``bounce_tables`` build.

The last step of a build, so a build directory becomes a catalog entry only after all
four tables are on disk. Nothing here moves the variant's ``current`` symlink; that is
``python -m rubinwork.products.catalog set-current``.

**One upstream**: the ``miw`` build whose Double-Zernike fit table was reduced. The
``fam_tables`` variant behind it goes under ``config``, not ``inputs`` -- a build reads
the MIW build's fit table, not the FAM tables -- so recording it as an input would claim
a dependency the build does not have. This is the opposite assignment from ``dof_lut``,
which reads the plain FAM tables and records ``fam_tables`` as its only input.

Command line, used by the Snakefile rule::

    python -m rubinwork.products.bounce_tables.write_manifest \\
        --variant d12-A_50_34_i_5rot_july --build 20261012 --miw-build 20261007
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
        Build-time settings to record alongside the variant entry -- the command-line
        flags of the run, which are not in ``variants.yaml``.

    Returns
    -------
    config : `dict`
        A copy of the variant entry, with ``variant``, the resolved ``fam_variant``, and
        the ``param_set`` and ``mi_name`` the configuration was looked up by -- all three
        derived through the MIW variant rather than stated in the entry, so the manifest
        records what was actually used. When given, a ``build_options`` key holds `extra`.
    """
    config = dict(reader.variant_config(variant))
    config["variant"] = variant
    config["fam_variant"] = reader.fam_variant(variant)
    param_set, mi_name = reader.mi_keys(variant)
    config["param_set"] = param_set
    config["mi_name"] = mi_name
    if extra:
        config["build_options"] = dict(extra)
    return config


def write(variant, build=None, build_dir=None, status="complete",
          build_options=None, miw_build=None, inputs=None):
    """Write the manifest of one ``bounce_tables`` build.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot_july"``.
    build : `str`, optional
        Build name, normally the date (``"20261012"``). Taken from `build_dir`'s name
        when that is given instead.
    build_dir : `str` or `pathlib.Path`, optional
        The build directory. By default the catalog directory of ``variant``/``build``,
        which is where a normal build writes.
    status : `str`, optional
        Manifest ``status``, ``"complete"`` by default. Anything else keeps the build out
        of the catalog.
    build_options : `dict`, optional
        Build-time settings, recorded under ``config.build_options``.
    miw_build : `str`, optional
        The ``miw`` build whose fit table was reduced. Recorded under ``inputs`` as
        ``{"miw": "<miw_variant>@<miw_build>"}``.
    inputs : `dict`, optional
        Further upstream builds read, as ``{product: "variant@build"}``. Merged over the
        entry `miw_build` produces.

    Returns
    -------
    manifest_path : `pathlib.Path`
        The manifest that was written.

    Raises
    ------
    `ValueError`
        If `build_dir` and `build` are both absent.

    Notes
    -----
    Row counts come from the parquet files themselves
    (`rubinwork.products.manifest.file_stats`), not from the builder's own counters, so
    the manifest reports what is on disk.
    """
    if build_dir is None:
        if not build:
            raise ValueError("write() needs `build` or `build_dir`")
        build_dir = reader.build_dir(variant, build)
    build_dir = pathlib.Path(build_dir)
    build = build or build_dir.name

    all_inputs = {}
    if miw_build:
        all_inputs["miw"] = f"{reader.miw_variant(variant)}@{miw_build}"
    if inputs:
        all_inputs.update(inputs)

    return manifest.write(
        build_dir, product=reader.PRODUCT, variant=variant, build=build,
        config=build_config(variant, extra=build_options),
        inputs=all_inputs or None, status=status)


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.bounce_tables.write_manifest",
        description="Write manifest.json for a bounce_tables build, as its last step.")
    parser.add_argument("--variant", required=True, help="Variant name")
    parser.add_argument("--build", default=None,
                        help="Build name (YYYYMMDD); taken from --build-dir if omitted")
    parser.add_argument("--build-dir", default=None,
                        help="Build directory (default: the catalog directory of "
                             "<variant>/<build>)")
    parser.add_argument("--miw-build", default=None,
                        help="The miw build whose fit table was reduced, recorded "
                             "under inputs")
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
                 status=args.status, build_options=options or None,
                 miw_build=args.miw_build)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
