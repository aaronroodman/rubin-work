"""Tests for the ``bounce_aba`` product: variant definitions, reader and the A–B–A matrix."""

import numpy as np
import pandas as pd
import pytest

from rubinwork.bounce import aba
from rubinwork.products import bounce_aba
from rubinwork.products.bounce_aba import reader


def test_variants_resolve():
    for v in bounce_aba.variants():
        cfg = bounce_aba.variant_config(v)
        assert cfg['scheme'] in aba.SCHEMES
        assert cfg['intrinsic'] in ('batoid', 'miw')
        prod, _ = bounce_aba.fits_input(v)
        assert prod == {'batoid': 'fam_tables', 'miw': 'miw'}[cfg['intrinsic']]
        param_set, mi_name = bounce_aba.mi_keys(v)
        assert param_set
        assert (mi_name is None) == (cfg['intrinsic'] == 'batoid')


def test_defaults_under_entry():
    cfg = bounce_aba.variant_config('d13-batoid-22_12')
    assert cfg['n_mc'] == 1000 and cfg['bounce'] == 'T720_elevation'


def test_unknown_variant():
    with pytest.raises(KeyError):
        bounce_aba.variant_config('nope')


def test_aba_matrix_interpolates_a():
    v = pd.DataFrame({'setting': ['A', 'B', 'B', 'A', 'B'], 't_min': [0.0, 1.0, 3.0, 4.0, 5.0]})
    M = aba.aba_matrix(v)
    # the last B has no A after it, so only two rows
    assert M.shape == (2, 5)
    np.testing.assert_allclose(M[0], [-0.75, 1, 0, -0.25, 0])
    np.testing.assert_allclose(M[1], [-0.25, 0, 1, -0.75, 0])
    # a constant offset cancels
    np.testing.assert_allclose(M.sum(1), 0)


def test_load_reads_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(reader.catalog, 'path', lambda *a, **k: tmp_path)
    for t in bounce_aba.TABLES:
        pd.DataFrame({'x': [1]}).to_parquet(tmp_path / f'{t}.parquet')
    out = bounce_aba.load('d13-batoid-22_12')
    assert len(out) == len(bounce_aba.TABLES)
    with pytest.raises(KeyError):
        bounce_aba.load_table('nope', 'd13-batoid-22_12')
