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


def analysis_config_path():
    return Path(__file__).resolve().parent.parent / 'analysis_config.yaml'


def load_analysis_doc(config_path=None):
    """Load analysis_config.yaml (LUT / aberration-pairs knobs).  Returns {} if
    the file is absent so callers can fall back to code-level defaults."""
    p = Path(config_path or analysis_config_path())
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def analysis_section(section, param_set=None, mi_name=None,
                     config_path=None, doc=None):
    """Merged analysis knobs for one ``section`` ('lut' | 'aberration_pairs').

    Resolution (deep-merge, later wins):
      defaults[section]
        <- overrides[param_set][section]
        <- overrides[param_set][mi_name][section]
    Kept separate from mi_config.yaml so editing these does not re-trigger the
    measured-intrinsic build (which lists mi_config.yaml as an input)."""
    doc = load_analysis_doc(config_path) if doc is None else doc
    out = (doc.get('defaults') or {}).get(section) or {}
    ov = ((doc.get('overrides') or {}).get(param_set) or {}) if param_set else {}
    out = _deep_merge(out, ov.get(section) or {})
    if mi_name and isinstance(ov.get(mi_name), dict):
        out = _deep_merge(out, ov[mi_name].get(section) or {})
    return out


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
