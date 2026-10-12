"""Write the ``manifest.json`` that registers one night of ``guider_moments``.

The last step of a night's build, so a night becomes a catalog partition only after its
tables are on disk. Nothing here moves a ``current`` symlink: an incremental product has
no single current build, and a night is addressed by its ``day_obs``.

**A scratch build writes no manifest.** `is_catalog_build` decides by **path**, not by a
flag, so a reference build or an ad-hoc night written outside the catalog's own
``nights/`` directory cannot be registered even if the wrong argument is passed. That is
the `value_added` rule, applied per night here.

**No upstream build is recorded.** A night's inputs are Butler raw guider data and
``summit_utils``, both external to this repository, so they go under ``config`` -- the
Butler repository, the collections and the ``summit_utils`` commit -- rather than under
``inputs``, which names one build per product.

Command line::

    python -m rubinwork.products.guider_moments.write_manifest \\
        --night 20260706 --summit-utils-commit e57e0f4
"""

import argparse
import pathlib
import subprocess
import sys

from .. import manifest
from . import reader

__all__ = ["summit_utils_state", "is_catalog_build", "build_config", "write", "main"]


def summit_utils_state():
    """Commit and branch of the ``summit_utils`` clone a build used.

    Returns
    -------
    state : `dict`
        ``commit`` (short SHA), ``branch`` and ``version``, each ``'unknown'`` when it
        cannot be read. ``summit_utils`` is **external** -- a clone on branch
        ``u/aaronroodman/guider-fixes`` -- and is never modified by a build, so this is
        provenance only.
    """
    state = {"commit": "unknown", "branch": "unknown", "version": "unknown"}
    try:
        import lsst.summit.utils as summit_utils

        state["version"] = str(getattr(summit_utils, "__version__", "unknown"))
        repo = pathlib.Path(summit_utils.__file__).resolve().parents[4]
    except Exception:  # noqa: BLE001
        return state

    def _git(*args):
        try:
            out = subprocess.run(("git", "-C", str(repo)) + args, capture_output=True,
                                 text=True, timeout=10)
        except Exception:  # noqa: BLE001
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    state["commit"] = _git("rev-parse", "--short", "HEAD") or "unknown"
    state["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    return state


def is_catalog_build(night_root, variant=reader.PRODUCT_VARIANT):
    """Whether `night_root` is this product's catalog partition for a night.

    Parameters
    ----------
    night_root : `str` or `pathlib.Path`
        The directory a build is writing a night into.
    variant : `str`, optional
        Product variant, ``'v6'`` by default.

    Returns
    -------
    catalog_build : `bool`
        `True` when the path is the catalog's own
        ``<products>/guider_moments/<variant>/nights/<night>/`` directory, so the build
        should end by writing a manifest. `False` for a scratch directory -- a reference
        build, an ad-hoc night -- which must not be registered.

    Notes
    -----
    Decided by the path alone, not by a flag, so there is no way to register a scratch
    build by passing the wrong argument.
    """
    night_root = pathlib.Path(night_root).resolve()
    from .. import catalog

    try:
        nights_root = (catalog.products_root() / reader.PRODUCT / variant
                       / reader.NIGHTS_DIR).resolve()
    except Exception:  # noqa: BLE001
        return False
    return nights_root == night_root.parent


def build_config(night, variant=reader.PRODUCT_VARIANT, extra=None):
    """The manifest ``config``: the variant entry, the night, and the external provenance.

    Parameters
    ----------
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.
    variant : `str`, optional
        Product variant, ``'v6'`` by default.
    extra : `dict`, optional
        Build-time settings to record under ``build_options``, such as an exposure limit.

    Returns
    -------
    config : `dict`
        The expanded variant entry, the night, and the ``summit_utils`` commit, branch and
        version the build ran against.
    """
    config = dict(reader.variant_config(variant))
    config["day_obs"] = int(night)
    config["summit_utils"] = summit_utils_state()
    if extra:
        config["build_options"] = dict(extra)
    return config


def write(night, variant=reader.PRODUCT_VARIANT, night_root=None, status="complete",
          build_options=None):
    """Write ``manifest.json`` for one night.

    Parameters
    ----------
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.
    variant : `str`, optional
        Product variant, ``'v6'`` by default.
    night_root : `str` or `pathlib.Path`, optional
        The night's directory. The catalog's own partition for that night by default.
    status : `str`, optional
        ``"complete"`` by default. Any other value keeps the night out of the catalog.
    build_options : `dict`, optional
        Build-time settings recorded under ``config.build_options``.

    Returns
    -------
    manifest_path : `pathlib.Path`
        The manifest that was written.

    Raises
    ------
    `ValueError`
        If `night_root` is given but is not this product's catalog partition for `night`,
        which is how a scratch build is kept out of the catalog.

    Notes
    -----
    Row counts come from the parquet files themselves
    (`rubinwork.products.manifest.file_stats`), not from the builder's own counters, so
    the manifest reports what is on disk.
    """
    night = int(night)
    directory = (pathlib.Path(night_root) if night_root is not None
                 else reader.night_dir(night, variant))
    if not is_catalog_build(directory, variant):
        raise ValueError(
            f"{directory} is not the catalog partition of {reader.PRODUCT}/{variant} "
            f"night {night}; a scratch build writes no manifest")
    return manifest.write(directory, product=reader.PRODUCT, variant=variant,
                          build=f"{reader.NIGHTS_DIR}/{night}",
                          config=build_config(night, variant, extra=build_options),
                          status=status)


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.guider_moments.write_manifest",
        description="Write manifest.json for one night of guider_moments.")
    parser.add_argument("--night", required=True, help="day_obs as YYYYMMDD")
    parser.add_argument("--variant", default=reader.PRODUCT_VARIANT,
                        help=f"Variant name (default: {reader.PRODUCT_VARIANT})")
    parser.add_argument("--night-root", default=None,
                        help="The night's directory (default: the catalog partition)")
    parser.add_argument("--status", default="complete",
                        help="Manifest status (default: complete)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Build-time setting recorded under config.build_options; "
                             "repeatable")
    args = parser.parse_args(argv)

    options = {}
    for item in args.option:
        key, _, value = item.partition("=")
        options[key] = value
    path = write(args.night, variant=args.variant, night_root=args.night_root,
                 status=args.status, build_options=options or None)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
