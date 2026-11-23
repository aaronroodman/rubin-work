from rubinwork.products import sim_donut_visit_fits


def test_variants_resolve():
    for v in sim_donut_visit_fits.variants():
        cfg = sim_donut_visit_fits.variant_config(v)
        assert cfg["plane"] in ("observed", "expected")
        assert all(isinstance(x, float) for k, x in cfg["lstsq"].items() if k.endswith("tol"))
