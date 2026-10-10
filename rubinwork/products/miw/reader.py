"""Read ``miw`` builds, and the variant definitions they were built from.

The Measured Intrinsic Wavefront (MIW) is the intrinsic wavefront measured from Full
Array Mode (FAM) data, as opposed to the batoid ray-trace prediction. A build holds it in
three forms, each read by different work:

``intrinsic_split_maps.parquet``
    the sampled field grid: ``thx_deg``, ``thy_deg`` and ``Z<j>_{OCS,CCS}``, one row per
    grid cell. This is the form every analysis treating the MIW *as data* wants -- surface
    fits, back-projection, focal-plane maps. `load_maps` reads it.
``intrinsic_split_decomp.parquet``
    the rotator-angle decomposition: per Zernike term, a telescope-fixed component in the
    Observatory Coordinate System (OCS) and a camera-fixed component in the Camera
    Coordinate System (CCS), the latter rotating with the camera rotator.
    `lsst.ts.intrinsic.wavefront.intrinsic_split.reconstruct_at` combines them at a given
    rotator angle. `rubinwork.miw_corner` evaluates it at the corner sensors.
``zk_intrinsic.parquet``
    the per-donut sidecar, row-aligned to the FAM ``donuts.parquet``.

A build also holds ``fits.parquet``, the FAM Double Zernike (DZ) fit refitted with this
MIW subtracted, and ``build/rot_<lo>_<hi>/intrinsic_grid.parquet``, the per-rotator-bin
grids the split is computed from.

The MIW also exists in a Butler as an `lsst.ip.isr.IntrinsicZernikes` calibration
(dataset type ``intrinsicZernikes``), which is an interpolator queried at arbitrary field
positions. Those are registered here as **external** variants (``official-*``) carrying a
``location`` and no files, and are deliberately not wrapped: their only in-repo consumers
call ``getIntrinsicZernikes`` directly, which is what
``aos/notebooks/cwfs/aos_miw_cwfs_intrinsic_check.ipynb`` exists to verify.
"""

import functools
import pathlib

import numpy as np
import pandas as pd
import yaml

from .. import catalog

PRODUCT = "miw"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("intrinsic_split_maps", "intrinsic_split_decomp", "intrinsic_split_rms",
          "zk_intrinsic", "fits")
"""The tables a complete build holds, in the order `load` returns them."""

JS_DEFAULT = tuple(j for j in range(4, 27) if j not in (20, 21))
"""Pupil (annular) Zernike Noll indices carried by the FAM MIW maps.

Z4-Z26 omitting Z20 and Z21, 21 terms. The same set as `rubinwork.aos_state.ZK_NOLL`.
"""

MAP_KEY = ("thx_deg", "thy_deg")
"""Columns that make an ``intrinsic_split_maps`` row unique: one row per field grid cell."""

DECOMP_KEY = ("j", "part")
"""Columns that make an ``intrinsic_split_decomp`` row unique: one row per (Noll j, part)."""

RMS_KEY = ("j",)
"""Column that makes an ``intrinsic_split_rms`` row unique: one row per Noll index."""

FIT_KEY = ("day_obs", "seq_num")
"""Columns that make a ``fits.parquet`` row unique: one row per FAM visit."""

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")


@functools.lru_cache(maxsize=1)
def _variants_doc():
    """Parsed ``variants.yaml``, read once per process."""
    with open(VARIANTS_FILE) as f:
        return yaml.safe_load(f) or {}


def groups():
    """The whole ``variants.yaml`` document.

    `rubinwork.products.miw.gen_mi_config` needs the literal text blocks and the group
    structure, not just the per-variant index.

    Returns
    -------
    doc : `dict`
        ``defaults_text``, ``lead_text``, ``groups`` and ``external``.
    """
    return _variants_doc()


@functools.lru_cache(maxsize=1)
def _index():
    """``variant`` -> its entry merged with its group's fields, in file order."""
    out = {}
    for group in _variants_doc().get("groups") or []:
        for entry in group.get("entries") or []:
            out[entry["variant"]] = dict(
                entry,
                fam_variant=group["fam_variant"],
                param_set=group["param_set"],
                short_code=group["short_code"])
    return out


def variants(registered_only=True):
    """Names of every built variant.

    Parameters
    ----------
    registered_only : `bool`, optional
        Omit entries carrying ``registered: false`` (the default).

    Returns
    -------
    names : `list` [`str`]
        Variant names, in ``variants.yaml`` order, which is the order the generated
        ``aos/mi_config.yaml`` writes them. External variants are **not** included; use
        `external_variants`. A defined variant need not have a build on disk; use
        `rubinwork.products.catalog.list_products` for what is built.
    """
    return [v for v, cfg in _index().items()
            if cfg.get("registered", True) or not registered_only]


def external_variants():
    """Names of the external variants: official releases and the staged in-repo copy.

    Returns
    -------
    names : `list` [`str`]
        Variant names, sorted. Each is registered as a manifest with a ``location`` and
        no files of its own.
    """
    return sorted(_variants_doc().get("external") or {})


def variant_config(variant):
    """The full configuration of one variant, built or external.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot"`` or ``"official-20260528a"``.

    Returns
    -------
    config : `dict`
        For a built variant: its ``variants.yaml`` entry -- ``mi_name``, ``dir_name``, the
        literal ``text`` block -- plus its group's ``fam_variant``, ``param_set`` and
        ``short_code``. For an external one: its ``external`` entry.

    Raises
    ------
    `KeyError`
        If `variant` is defined in neither the ``groups`` nor the ``external`` block.
    """
    index = _index()
    if variant in index:
        return index[variant]
    external = _variants_doc().get("external") or {}
    if variant in external:
        return external[variant]
    raise KeyError(f"{variant!r} is not a miw variant; defined: "
                   f"{list(index) + external_variants()}")


def fam_variant(variant):
    """The ``fam_tables`` variant whose tables this MIW variant is built from.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    fam_variant : `str`
        A ``fam_tables`` variant name, e.g. ``"danish_1_2"``.

    Raises
    ------
    `KeyError`
        If `variant` is external, and so has no upstream FAM variant here.
    """
    cfg = variant_config(variant)
    if "fam_variant" not in cfg:
        raise KeyError(f"{variant!r} is an external miw variant and has no fam_variant")
    return cfg["fam_variant"]


def mi_keys(variant):
    """The ``(param_set, mi_name)`` pair the external package's runners take.

    The long ``param_set`` and the ``mi_name`` remain the identity for everything that
    recorded them, and are what ``--param-set`` and ``--mi-name`` expect.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    param_set : `str`
        The long ``aos/param_sets.yaml`` key of the upstream FAM variant.
    mi_name : `str`
        The entry's ``name`` in the generated ``aos/mi_config.yaml``.
    """
    cfg = variant_config(variant)
    return cfg["param_set"], cfg["mi_name"]


def dir_name(variant):
    """The entry's ``dir_name``: its output subdirectory in the old ``aos/`` tree.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    dir_name : `str`
        e.g. ``"A_50_34_i_5rot"``. The old tree's joined directory is
        ``aos/output/miw/<fam dir_name>_<this>/``.
    """
    return variant_config(variant)["dir_name"]


def _entry_config(variant):
    """One variant's ``mi_config`` entry, parsed from its literal text block.

    The block is the YAML the generated file carries, one list item under a param_set key,
    so it parses on its own once the two levels of indentation are removed.

    Returns
    -------
    entry : `dict`
        The entry mapping: ``name``, ``dir_name``, ``n_dof``, ``n_keep`` and whatever else
        it sets. NOT merged with ``defaults``; use
        `lsst.ts.intrinsic.wavefront.mi_config.load_mi_config` for the merged form, which
        is what the builders read.
    """
    text = variant_config(variant)["text"]
    dedented = "\n".join(line[4:] if line.startswith("    ") else line
                          for line in text.split("\n"))
    return (yaml.safe_load(dedented) or [{}])[0]


def build_source(variant):
    """The variant whose per-rotator-bin grids `variant` is split from.

    An entry carrying ``build_from`` reuses another variant's already-built grids and runs
    only the split and what follows, so it triggers no ``build_intrinsic``.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    source : `str`
        The product variant whose grids are read, or `variant` itself when it builds its
        own. ``build_from`` in the text block names an ``mi_name``, which is resolved back
        to a product variant of the same ``fam_variant``.
    """
    parent_mi = _entry_config(variant).get("build_from")
    if not parent_mi:
        return variant
    fam = fam_variant(variant)
    for name, cfg in _index().items():
        if cfg["fam_variant"] == fam and cfg["mi_name"] == parent_mi:
            return name
    raise KeyError(f"{variant!r} builds from mi_name {parent_mi!r}, which is not a "
                   f"miw variant of fam_variant {fam!r}")


def rotator_bins(variant):
    """The rotator-angle bins this variant's grids are built in.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    bins : `list` [`tuple` [`float`, `float`]]
        Camera rotator angle windows in degrees, from the entry's own ``rotator_bins`` or
        the ``defaults`` block.
    """
    own = _entry_config(variant).get("rotator_bins")
    if own is None:
        defaults = yaml.safe_load(_variants_doc()["defaults_text"]) or {}
        own = defaults["defaults"]["rotator_bins"]
    return [(float(lo), float(hi)) for lo, hi in own]


def split_rotator_bins(variant):
    """The rotator bins this variant's OCS/CCS split decomposes.

    An entry may set ``split.rotator_select`` to decompose only a subset -- dropping an
    out-of-family epoch that sits at distinct rotator angles -- while the full set of
    build grids still exists, so no rebuild is needed.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    bins : `list` [`tuple` [`float`, `float`]]
        Windows in degrees: ``split.rotator_select`` if set, else every bin of the build
        source.
    """
    sel = (_entry_config(variant).get("split") or {}).get("rotator_select")
    if sel is None:
        return rotator_bins(build_source(variant))
    return [(float(lo), float(hi)) for lo, hi in sel]


def build_dir(variant, build):
    """Where a build of `variant` is written, whether or not it exists yet.

    `rubinwork.products.catalog.path` resolves only builds already on disk; a builder
    needs the path to write into.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`
        Build name, usually the date the data is produced (``"20261010"``).

    Returns
    -------
    build_dir : `pathlib.Path`
        The build directory under the products root. Not created.
    """
    return catalog.products_root() / PRODUCT / variant / build


def load_table(table, variant, build=catalog.CURRENT):
    """Read one table of a build.

    Parameters
    ----------
    table : `str`
        One of `TABLES`.
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    data : `pandas.DataFrame`
        The table.

    Raises
    ------
    `ValueError`
        If `table` is not one of `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build or the table is missing.
    """
    if table not in TABLES:
        raise ValueError(f"{table!r} is not a miw table; expected one of {TABLES}")
    return catalog.load(PRODUCT, variant, build=build, file=f"{table}.parquet")


def load(variant, build=catalog.CURRENT, tables=TABLES):
    """Read a build's tables.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot"``.
    build : `str`, optional
        Build name, or ``"current"`` (the default) for the variant's default build.
    tables : `tuple` [`str`], optional
        Which tables to read, in the order they are returned. `TABLES` by default.

    Returns
    -------
    tables : `tuple` [`pandas.DataFrame`]
        One DataFrame per requested table.

    Raises
    ------
    `ValueError`
        If `tables` names something that is not in `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build does not exist, or a requested table is missing from it.

    Notes
    -----
    ``zk_intrinsic.parquet`` is row-aligned to the FAM ``donuts.parquet`` and so is
    millions of rows for a full build; name only the tables wanted rather than taking the
    default when that one is not needed.

    Examples
    --------
    >>> maps, = load("d12-A_50_34_i_5rot",
    ...              tables=("intrinsic_split_maps",))         # doctest: +SKIP
    """
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise ValueError(f"not miw tables: {bad}; expected from {TABLES}")
    return tuple(load_table(t, variant, build=build) for t in tables)


def load_maps(source, stride=1, coord="OCS", js=JS_DEFAULT, require=None):
    """Load an MIW field map as a grid of Zernike coefficients.

    Parameters
    ----------
    source : `str` or `pathlib.Path`
        Path to an ``intrinsic_split_maps`` parquet. Resolve one from the catalog with
        ``catalog.path("miw", variant) / "intrinsic_split_maps.parquet"`` rather than
        building a path.
    stride : `int`, optional
        Keep every `stride`-th surviving grid row. 1 (default) keeps all.
    coord : {'OCS', 'CCS'}, optional
        Which frame's columns to read. **OCS** (default) is the telescope-fixed
        component, and is what the static-optics and back-projection analyses want; CCS
        is the camera-fixed component, which rotates with the rotator.
    js : `iterable` [`int`], optional
        Pupil Zernike Noll indices to extract. `JS_DEFAULT` by default.
    require : `iterable` [`int`] or `None`, optional
        Which Zernikes must be finite for a row to be kept. `None` (default) requires
        all of `js`. Pass a subset to keep rows complete only in the terms an analysis
        actually uses -- e.g. ``require=(5, 6, 7, 8)`` for an astigmatism/coma study,
        which keeps 108 extra field points in the current maps where ``Z4_OCS``
        (defocus, CCD-height sensitive) is non-finite but Z5-Z8 are fine.

    Returns
    -------
    pts : `numpy.ndarray`, shape (N, 2)
        Field positions, in degrees, as (thx, thy).
    zk : `numpy.ndarray`, shape (N, max(js) + 1)
        Zernike coefficients in µm of wavefront, **Noll-indexed**: column `j` holds
        Z\\ :sub:`j`, so columns 0-3 (and 20, 21 for the default set) are zero.
    df : `pandas.DataFrame`
        The surviving rows, with `thx_deg`, `thy_deg` and the ``Z<j>_<coord>`` columns.

    Raises
    ------
    KeyError
        If the map lacks any requested ``Z<j>_<coord>`` column.

    Notes
    -----
    Rows with a non-finite value in any *required* Zernike are dropped, so every returned
    row is complete across `require`. Columns outside `require` may still hold NaN, so a
    caller that widens its Zernike use later must widen `require` too.

    The canonical MIW for downstream use is the ``_5rot`` variant's
    ``intrinsic_split_maps.parquet`` with its **OCS** columns; the frozen, version-tracked
    copy is the ``staged-v1`` external variant.
    """
    js = list(js)
    df = pd.read_parquet(source)
    cols = [f"Z{j}_{coord}" for j in js]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"MIW map {source} is missing {len(missing)} column(s), "
                       f"first few: {missing[:4]}")

    req_cols = [f"Z{j}_{coord}" for j in (js if require is None else list(require))]
    ok = np.all(np.isfinite(df[req_cols].to_numpy()), axis=1)
    df = df[ok].iloc[::stride].reset_index(drop=True)

    pts = np.column_stack([df["thx_deg"].to_numpy(), df["thy_deg"].to_numpy()])
    zk = np.zeros((len(df), max(js) + 1))
    for j in js:
        zk[:, j] = df[f"Z{j}_{coord}"].to_numpy()
    return pts, zk, df


load_miw = load_maps
"""Alias of `load_maps`, the name this reader had as ``aos/code/miw_io.load_miw``.

Kept so the four ``aos/code/static_optics/`` scripts and the four ``aos/code/miw/``
comparison scripts keep working through the shim at the old path. New code calls
`load_maps`, which says what it reads.
"""
