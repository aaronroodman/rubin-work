"""The product catalog: find and read product builds through their manifests.

A product build lives at ``<data root>/products/<product>/<variant>/<build>/`` and
carries a ``manifest.json`` written by `rubinwork.products.manifest`. The default
build of a variant is the one ``current`` points at, a symlink the build writes.

This module is the only sanctioned way to reach product data. Building a path by
hand couples a reader to a layout that the reorganization is replacing; going
through `path` or `load` means a product can move, gain a variant or get a new
build without any reader changing.

External variants -- an official release, a Butler collection -- are registered as
a manifest with a ``location`` field and no files of their own.

Examples
--------
Read the current build of a variant::

    from rubinwork.products import catalog
    df = catalog.load("fam_tables", "danish_1_2")

Pin a build, for a result that must not move under you::

    build_dir = catalog.path("miw", "d12-A_50_34_i_5rot", build="20261007")
"""

import json
import os
import pathlib

__all__ = [
    "DATA_ROOT_DEFAULT", "data_root", "products_root", "studies_root",
    "path", "manifest", "load", "list_products", "list", "register",
    "ProductNotFound",
]

DATA_ROOT_DEFAULT = "/sdf/group/rubin/u/roodman/LSST/rubin-work"
"""Default data root on S3DF, overridden by ``$RUBINWORK_DATA``."""

CURRENT = "current"
"""Name of the symlink in a variant directory naming its default build."""

MANIFEST_NAME = "manifest.json"


class ProductNotFound(FileNotFoundError):
    """A requested product, variant or build is not in the catalog."""


def data_root():
    """Root of the data tree, from ``$RUBINWORK_DATA`` or the S3DF default.

    Returns
    -------
    root : `pathlib.Path`
        The data root. Not required to exist, so a caller can report a clear
        error rather than this function raising during import.
    """
    return pathlib.Path(os.environ.get("RUBINWORK_DATA", DATA_ROOT_DEFAULT))


def products_root():
    """Directory holding every product, ``<data root>/products``.

    Returns
    -------
    root : `pathlib.Path`
        The products directory.
    """
    return data_root() / "products"


def studies_root():
    """Directory holding per-study run output, ``<data root>/studies``.

    Returns
    -------
    root : `pathlib.Path`
        The studies directory.
    """
    return data_root() / "studies"


def _resolve_build(product, variant, build):
    """Return the build directory for one variant, resolving ``"current"``.

    Parameters
    ----------
    product : `str`
        Product name, the directory under ``products/``.
    variant : `str`
        Variant name, as defined in the product's ``variants.yaml``.
    build : `str`
        Build name (a date such as ``"20261007"``), or ``"current"`` for the
        build the ``current`` symlink points at.

    Returns
    -------
    build_dir : `pathlib.Path`
        The build directory.

    Raises
    ------
    `ProductNotFound`
        If the product, variant or build directory does not exist.
    """
    variant_dir = products_root() / product / variant
    if not variant_dir.is_dir():
        if not (products_root() / product).is_dir():
            known = _names(products_root())
            raise ProductNotFound(
                f"no product {product!r} under {products_root()}; "
                f"known products: {known or 'none'}")
        known = _names(products_root() / product)
        raise ProductNotFound(
            f"no variant {variant!r} of product {product!r}; "
            f"known variants: {known or 'none'}")
    build_dir = variant_dir / build
    if not build_dir.exists():
        known = _names(variant_dir)
        raise ProductNotFound(
            f"no build {build!r} of {product}/{variant}; "
            f"known builds: {known or 'none'}")
    return build_dir


def _names(directory):
    """Sorted names of the entries in `directory`, excluding work directories.

    Returns
    -------
    names : `list` [`str`]
        Entry names, or an empty list if `directory` does not exist.
    """
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir()
                  if not p.name.startswith((".", "_")))


def path(product, variant, build=CURRENT):
    """Directory of one product build.

    Parameters
    ----------
    product : `str`
        Product name.
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"`` for the variant's default build.

    Returns
    -------
    build_dir : `pathlib.Path`
        The build directory, with ``current`` resolved to the real build.

    Raises
    ------
    `ProductNotFound`
        If the product, variant or build does not exist.
    """
    return _resolve_build(product, variant, build).resolve()


def manifest(product, variant, build=CURRENT):
    """The manifest of one product build.

    Parameters
    ----------
    product : `str`
        Product name.
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    manifest : `dict`
        The parsed ``manifest.json``.

    Raises
    ------
    `ProductNotFound`
        If the build or its manifest does not exist.
    """
    build_dir = _resolve_build(product, variant, build)
    manifest_path = build_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ProductNotFound(
            f"{product}/{variant}/{build} has no {MANIFEST_NAME}; an incomplete "
            "or hand-made build directory is not in the catalog")
    with open(manifest_path) as f:
        return json.load(f)


def load(product, variant, build=CURRENT, file=None):
    """Read one file from a product build.

    Parameters
    ----------
    product : `str`
        Product name.
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.
    file : `str`, optional
        Name of the file to read, as it appears in the manifest's ``files``. May
        be omitted when the build has exactly one file.

    Returns
    -------
    data : `pandas.DataFrame` or `pathlib.Path`
        A parquet or csv file is read into a `~pandas.DataFrame`. Any other
        file, including a DuckDB database, is returned as its path, for the
        caller to open with the right reader.

    Raises
    ------
    `ProductNotFound`
        If the build does not exist, or `file` is absent from it.
    `ValueError`
        If `file` is omitted and the build does not hold exactly one file.

    Notes
    -----
    A product package may define its own ``load`` with product-specific
    behaviour (a DuckDB connection, several files joined); this generic version
    is what a product gets before it has one.
    """
    build_dir = _resolve_build(product, variant, build)
    files = sorted(manifest(product, variant, build).get("files", {}))
    if file is None:
        if len(files) != 1:
            raise ValueError(
                f"{product}/{variant}/{build} holds {len(files)} files "
                f"(count, dimensionless), so `file` is required; "
                f"files: {files or 'none'}")
        file = files[0]
    elif files and file not in files:
        raise ProductNotFound(
            f"{file!r} is not in {product}/{variant}/{build}; files: {files}")
    target = build_dir / file
    if not target.exists():
        raise ProductNotFound(
            f"{target} is named in the manifest but missing on disk")
    if target.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(target)
    if target.suffix == ".csv":
        import pandas as pd
        return pd.read_csv(target)
    return target.resolve()


def list_products(product=None):
    """Every registered build, read from the manifests.

    Parameters
    ----------
    product : `str`, optional
        Restrict the listing to one product. All products by default.

    Returns
    -------
    builds : `list` [`dict`]
        One entry per build, sorted by product, variant then build, each with
        keys ``product``, ``variant``, ``build``, ``current`` (`bool`, whether
        the variant's ``current`` symlink points here), ``status``, ``created``,
        ``git_commit`` and ``n_files`` (count, dimensionless). A build with an
        unreadable manifest gets ``status`` ``"no-manifest"``.
    """
    root = products_root()
    names = [product] if product else _names(root)
    builds = []
    for prod in names:
        for variant in _names(root / prod):
            variant_dir = root / prod / variant
            current = variant_dir / CURRENT
            current_target = current.resolve() if current.exists() else None
            for build in _names(variant_dir):
                build_dir = variant_dir / build
                if build == CURRENT or not build_dir.is_dir():
                    continue
                try:
                    man = manifest(prod, variant, build)
                except (ProductNotFound, json.JSONDecodeError):
                    man = {"status": "no-manifest"}
                builds.append({
                    "product": prod,
                    "variant": variant,
                    "build": build,
                    "current": build_dir.resolve() == current_target,
                    "status": man.get("status"),
                    "created": man.get("created"),
                    "git_commit": man.get("git_commit"),
                    "n_files": len(man.get("files", {})),
                })
    return builds


# `list` shadows the builtin inside this module, so it is defined last and the
# builtin is not used below. Section 5 of the plan names the function `list()`.
list = list_products  # noqa: A001


def register(product, variant, build, location=None, config=None, inputs=None,
             status="complete", set_current=True):
    """Register a build that exists outside the product tree, or on disk already.

    Used for an external variant -- an official release, a Butler collection -- and
    for a pre-reorganization build copied into the new tree. The build directory
    is created if needed and given a manifest; no data is copied or moved.

    Parameters
    ----------
    product : `str`
        Product name.
    variant : `str`
        Variant name.
    build : `str`
        Build name, usually the date the data was produced.
    location : `str` or `pathlib.Path`, optional
        Where the data actually lives, for a build with no files of its own
        (an external release path, a Butler collection name).
    config : `dict`, optional
        The configuration this build was produced with, as the manifest's
        ``config``.
    inputs : `dict`, optional
        Upstream builds, as ``{product: "variant@build"}``.
    status : `str`, optional
        Manifest ``status``, ``"complete"`` by default.
    set_current : `bool`, optional
        Whether to point the variant's ``current`` symlink at this build.

    Returns
    -------
    build_dir : `pathlib.Path`
        The registered build directory.
    """
    # Imported here, by full name: this module's own `manifest` function shadows
    # the submodule, and importing it at the top would be circular.
    import importlib
    manifest_module = importlib.import_module("rubinwork.products.manifest")

    build_dir = products_root() / product / variant / build
    build_dir.mkdir(parents=True, exist_ok=True)
    manifest_module.write(
        build_dir, product=product, variant=variant, build=build,
        config=config, inputs=inputs, location=location, status=status,
        set_current=set_current,
    )
    return build_dir
