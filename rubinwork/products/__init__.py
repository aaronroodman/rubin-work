"""Data products and the catalog that reads them.

A product is code that writes data other work reads. Each product has named
variants, dated builds and a ``manifest.json`` per build. Code reaches product
data only through `rubinwork.products.catalog`, never by building a path.

The data root comes from the ``RUBINWORK_DATA`` environment variable, defaulting
to the S3DF path, so the same code runs on the laptop, the Rubin Science
Platform (RSP) and batch nodes.
"""

from .catalog import DATA_ROOT_DEFAULT, data_root, list_products, load, path, register

__all__ = [
    "DATA_ROOT_DEFAULT", "data_root", "list_products", "load", "path", "register",
]
