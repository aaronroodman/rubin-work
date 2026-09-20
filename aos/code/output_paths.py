"""Resolve `output/` directories for a ``param_set`` and ``mi_name``.

Output is laid out study-outermost with exactly one data level, the two data axes
joined into a single directory name::

    output/<study>/<P>/           depends on the param_set only
    output/<study>/<P>_<M>/       depends on the param_set and the MIW build

``<P>`` and ``<M>`` are **short** directory names given by the ``dir_name`` key in
``param_sets.yaml`` and in the ``measured_intrinsics`` entries of ``mi_config.yaml``,
defaulting to the key when absent. The long keys stay the identity that ``--param-set``,
the value-added database rows and the frozen provenance resolve against; the short forms
appear only in paths.

The Snakefile owns the layout for every rule it runs, passing each script an explicit
``--out-dir``. This module is for the scripts run **by hand**, which have no Snakefile to
tell them where to write and would otherwise hardcode a path that goes stale.

Examples
--------
>>> study_dir('fam_processing', 'fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x')
PosixPath('output/fam_processing/danish_1_2')
>>> study_dir('miw', 'fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x', 'pathA_50_34_i_5rot')
PosixPath('output/miw/danish_1_2_A_50_34_i_5rot')
"""

from pathlib import Path

import yaml

from lsst.ts.intrinsic.wavefront import mi_config as mc

TOPIC = Path(__file__).resolve().parent.parent


def ps_dir(param_set, config_path=None):
    """Short output directory name for a ``param_set``.

    Parameters
    ----------
    param_set : `str`
        The long ``param_sets.yaml`` key.
    config_path : `pathlib.Path`, optional
        ``param_sets.yaml`` to read (default: the one beside this topic).

    Returns
    -------
    name : `str`
        The entry's ``dir_name``, or ``param_set`` itself when it has none.
    """
    path = Path(config_path) if config_path else TOPIC / 'param_sets.yaml'
    doc = yaml.safe_load(path.read_text())
    doc = doc.get('param_sets', doc)
    return (doc.get(param_set) or {}).get('dir_name') or param_set


def mi_dir(param_set, mi_name, config_path=None, doc=None):
    """Short output directory name for a ``mi_name`` under one ``param_set``.

    Parameters
    ----------
    param_set : `str`
        The long ``param_sets.yaml`` key.
    mi_name : `str`
        The ``measured_intrinsics`` entry name.
    config_path, doc
        Passed through to `lsst.ts.intrinsic.wavefront.mi_config.load_mi_config`.

    Returns
    -------
    name : `str`
        The entry's ``dir_name``, or ``mi_name`` itself when it has none.
    """
    cfg = mc.load_mi_config(param_set, mi_name, config_path=config_path, doc=doc)
    return mc.dir_name(cfg, mi_name)


def dtag(param_set, mi_name=None, ps_config=None, mi_config=None):
    """Joined data-axis directory name: ``<P>`` or ``<P>_<M>``.

    Parameters
    ----------
    param_set : `str`
        The long ``param_sets.yaml`` key.
    mi_name : `str`, optional
        The MIW build. Omit for a product that depends on the ``param_set`` alone.
    ps_config, mi_config : `pathlib.Path`, optional
        Override config paths, for testing.

    Returns
    -------
    name : `str`
        The single directory name holding both data axes.
    """
    P = ps_dir(param_set, config_path=ps_config)
    if mi_name is None:
        return P
    return f'{P}_{mi_dir(param_set, mi_name, config_path=mi_config)}'


def study_dir(study, param_set, mi_name=None, output_root='output',
              ps_config=None, mi_config=None):
    """Directory a study's products go in for one data set.

    Parameters
    ----------
    study : `str`
        Study directory name, e.g. ``'fam_processing'``, ``'miw'``, ``'coadd'``.
    param_set : `str`
        The long ``param_sets.yaml`` key.
    mi_name : `str`, optional
        The MIW build. Omit for a product that depends on the ``param_set`` alone.
    output_root : `str` or `pathlib.Path`, optional
        Root of the output tree (default ``'output'``, relative to ``aos/``).
    ps_config, mi_config : `pathlib.Path`, optional
        Override config paths, for testing.

    Returns
    -------
    path : `pathlib.Path`
        ``<output_root>/<study>/<P>[_<M>]``. Not created.
    """
    return (Path(output_root) / study
            / dtag(param_set, mi_name, ps_config=ps_config, mi_config=mi_config))
