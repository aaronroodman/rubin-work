"""Loader for mi_config.yaml (Phase-2 measured-intrinsic configs).

Merges the top-level ``defaults`` block into each per-param_set entry (deep
merge, so nested ``build`` / ``split`` dicts combine key-by-key), and resolves
the scalar-or-list ``n_dof`` / ``n_keep`` convention.
"""
import copy
from pathlib import Path

import yaml


def _deep_merge(base, over):
    out = copy.deepcopy(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config_path():
    return Path(__file__).resolve().parent.parent / 'mi_config.yaml'


def load_doc(config_path=None):
    return yaml.safe_load(open(config_path or default_config_path()))


def mi_names(param_set, config_path=None, doc=None):
    doc = doc or load_doc(config_path)
    return [e['name'] for e in doc.get('measured_intrinsics', {}).get(param_set, [])]


def load_mi_config(param_set, mi_name, config_path=None, doc=None):
    """Fully merged config dict for one (param_set, mi_name): defaults + entry."""
    doc = doc or load_doc(config_path)
    defaults = doc.get('defaults', {})
    for e in doc.get('measured_intrinsics', {}).get(param_set, []):
        if e.get('name') == mi_name:
            return _deep_merge(defaults, e)
    raise KeyError(f'MI config {mi_name!r} not found for param_set {param_set!r}')


def resolve_indices(spec):
    """Resolve the scalar-or-list convention to a sorted list of int indices.
    ``int n`` -> [0, 1, ..., n-1];  ``list`` -> exactly those indices."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return sorted(int(x) for x in spec)
    return list(range(int(spec)))


def rotator_bins(cfg):
    return [(float(lo), float(hi)) for lo, hi in cfg['rotator_bins']]


def as_band_list(filt):
    if filt is None:
        return None
    return [filt] if isinstance(filt, str) else list(filt)
