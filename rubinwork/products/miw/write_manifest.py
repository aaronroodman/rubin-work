"""Write the ``manifest.json`` that finishes a ``miw`` build, or registers an external one.

The last step of a build, so a build directory becomes a catalog entry only after every
table is on disk. Nothing here moves the variant's ``current`` symlink; that is
``python -m rubinwork.products.catalog set-current``.

The manifest records the ``fam_tables`` build the MIW was built from under ``inputs``,
because that sets which visits entered the per-rotator-bin recovery, and -- for an entry
carrying ``build_from`` -- the ``miw`` build whose grids were reused, because such an entry
runs only the split and so inherits its parent's grids wholesale.

Two command lines. A build::

    python -m rubinwork.products.miw.write_manifest \\
        --variant d12-A_50_34_i_5rot --build 20261010 --fam-build 20260920

An external variant -- an official calibration collection, or the staged in-repo copy --
which records a ``location`` and no files::

    python -m rubinwork.products.miw.write_manifest --register-external official-20260528a
    python -m rubinwork.products.miw.write_manifest --register-external all
"""

import argparse
import pathlib
import sys

from .. import catalog, manifest
from . import reader

__all__ = ["build_config", "write", "register_external", "main"]

EXTERNAL_BUILD = "external"
"""Build name every external variant is registered under.

An external variant has one entry, not dated builds: the ``location`` is the identity, and
a new release is a new variant (``official-20260528a`` beside ``official-gmegias-v3``).
"""


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
        A copy of the variant entry, with ``variant`` and (when given) a
        ``build_options`` key holding `extra`. For a built variant the entry's literal
        ``text`` block is kept, so the manifest records the exact configuration YAML the
        runners were given.
    """
    config = dict(reader.variant_config(variant))
    config["variant"] = variant
    if extra:
        config["build_options"] = dict(extra)
    return config


def write(variant, build=None, build_dir=None, status="complete",
          build_options=None, fam_build=None, inputs=None):
    """Write the manifest of one ``miw`` build.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot"``.
    build : `str`, optional
        Build name, normally the date (``"20261010"``). Taken from `build_dir`'s name
        when that is given instead.
    build_dir : `str` or `pathlib.Path`, optional
        The build directory. By default the catalog directory of
        ``variant``/``build``, which is where a normal build writes.
    status : `str`, optional
        Manifest ``status``, ``"complete"`` by default. Anything else keeps the build
        out of the catalog.
    build_options : `dict`, optional
        Build-time settings, recorded under ``config.build_options``.
    fam_build : `str`, optional
        The ``fam_tables`` build this MIW was built from. Recorded under ``inputs`` as
        ``{"fam_tables": "<fam_variant>@<fam_build>"}``.
    inputs : `dict`, optional
        Upstream builds read, as ``{product: "variant@build"}``. Merged over the entries
        `fam_build` and ``build_from`` produce, so a caller can name the ``miw`` build
        whose grids were reused.

    Returns
    -------
    manifest_path : `pathlib.Path`
        The manifest that was written.

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
    if fam_build:
        all_inputs["fam_tables"] = f"{reader.fam_variant(variant)}@{fam_build}"
    source = reader.build_source(variant)
    if source != variant:
        # A build_from entry runs only the split, so its grids ARE the parent's: record
        # which parent, or the provenance of the maps is unrecoverable from the manifest.
        all_inputs["miw"] = source
    if inputs:
        all_inputs.update(inputs)

    return manifest.write(
        build_dir, product=reader.PRODUCT, variant=variant, build=build,
        config=build_config(variant, extra=build_options),
        inputs=all_inputs or None, status=status)


def register_external(variant):
    """Register one external variant: a manifest with a ``location`` and no files.

    Parameters
    ----------
    variant : `str`
        An `rubinwork.products.miw.external_variants` name, e.g.
        ``"official-20260528a"``.

    Returns
    -------
    build_dir : `pathlib.Path`
        The registered build directory, which holds only the manifest.

    Raises
    ------
    `KeyError`
        If `variant` is not an external variant.

    Notes
    -----
    The ``location`` is a Butler collection name for an ``official-*`` entry and a
    repository-relative path for ``staged-v1``. Neither is copied: the manifest points at
    data this repository does not own, which is the whole purpose of an external variant.
    Nothing the official MIW needs comes from here.
    """
    if variant not in reader.external_variants():
        raise KeyError(f"{variant!r} is not an external miw variant; defined: "
                       f"{reader.external_variants()}")
    config = reader.variant_config(variant)
    return catalog.register(reader.PRODUCT, variant, EXTERNAL_BUILD,
                            location=config["location"],
                            config=dict(config, variant=variant))


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.miw.write_manifest",
        description="Write manifest.json for a miw build, as its last step, or register "
                    "an external variant.")
    parser.add_argument("--variant", default=None, help="Variant name")
    parser.add_argument("--build", default=None,
                        help="Build name (YYYYMMDD); taken from --build-dir if omitted")
    parser.add_argument("--build-dir", default=None,
                        help="Build directory (default: the catalog directory of "
                             "<variant>/<build>)")
    parser.add_argument("--fam-build", default=None,
                        help="The fam_tables build this MIW was built from, recorded "
                             "under inputs")
    parser.add_argument("--status", default="complete",
                        help="Manifest status (default: complete)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Build-time setting to record under config.build_options; "
                             "repeatable")
    parser.add_argument("--register-external", default=None, metavar="VARIANT",
                        help="Register an external variant (a manifest with a location "
                             "and no files), or `all` for every one defined")
    args = parser.parse_args(argv)

    if args.register_external:
        names = (reader.external_variants() if args.register_external == "all"
                 else [args.register_external])
        for name in names:
            build_dir = register_external(name)
            print(f"registered {name} -> {build_dir / 'manifest.json'}")
        return 0

    if not args.variant:
        parser.error("--variant is required unless --register-external is given")
    options = {}
    for item in args.option:
        key, _, value = item.partition("=")
        options[key] = value
    path = write(args.variant, build=args.build, build_dir=args.build_dir,
                 status=args.status, build_options=options or None,
                 fam_build=args.fam_build)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
