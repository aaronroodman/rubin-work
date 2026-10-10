"""Quick undefined-name scan for a Jupyter notebook.

Walks cells in order, accumulating module-level definitions, and
reports any Name(Load) reference whose identifier is never bound
anywhere up to and including the current cell (counting any name
introduced as an assignment target / def / class / param / import /
for-target / with-target / except-as inside or outside functions).

It's deliberately permissive (would-be false positives skipped):
* Builtins + common runtime names allowed.
* Names defined ANYWHERE in the cumulative set (incl. function locals)
  treated as bound; we don't try to model scope precisely.

Run:  python3 check_undefined.py path/to/notebook.ipynb
"""
import ast
import builtins
import json
import sys


COMMON_RUNTIME = {
    '__name__', '__doc__', '__file__', '__builtins__',
    'get_ipython', 'In', 'Out', 'exit', 'quit', 'display',
}


def _names_from_target(t):
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        out = set()
        for x in t.elts:
            out |= _names_from_target(x)
        return out
    if isinstance(t, ast.Starred):
        return _names_from_target(t.value)
    return set()


def collect_defined(tree):
    """Every name that gets bound anywhere in `tree` (incl. nested scopes)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                out |= _names_from_target(t)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            out |= _names_from_target(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
            # params
            args = node.args
            for a in (args.args + args.posonlyargs + args.kwonlyargs):
                out.add(a.arg)
            if args.vararg:
                out.add(args.vararg.arg)
            if args.kwarg:
                out.add(args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            args = node.args
            for a in (args.args + args.posonlyargs + args.kwonlyargs):
                out.add(a.arg)
            if args.vararg:
                out.add(args.vararg.arg)
            if args.kwarg:
                out.add(args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == '*':
                    # can't know — skip
                    continue
                out.add(alias.asname or alias.name)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            out |= _names_from_target(node.target)
        elif isinstance(node, ast.comprehension):
            out |= _names_from_target(node.target)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    out |= _names_from_target(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                out.add(node.name)
        elif isinstance(node, ast.Global):
            out.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            out.update(node.names)
        elif isinstance(node, ast.NamedExpr):
            out |= _names_from_target(node.target)
    return out


def collect_loads(tree):
    """Names used (Load context) anywhere in `tree`."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.add(node.id)
    return out


def main(path):
    nb = json.load(open(path))
    defined = set(dir(builtins)) | COMMON_RUNTIME
    any_issue = False
    for ci, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f'  cell {ci} ({c.get("id")}): SyntaxError {e}')
            continue
        used = collect_loads(tree)
        local = collect_defined(tree)
        undef = used - defined - local
        if undef:
            any_issue = True
            print(f'\ncell {ci} ({c.get("id")}): possibly undefined:')
            for n in sorted(undef):
                # locate first occurrence in source for context
                lineno = next((node.lineno for node in ast.walk(tree)
                               if isinstance(node, ast.Name)
                               and isinstance(node.ctx, ast.Load)
                               and node.id == n), None)
                line_text = src.splitlines()[lineno - 1].strip()[:90] if lineno else ''
                print(f'  {n!r:30s} (first at line {lineno}: {line_text})')
        defined |= local
    if not any_issue:
        print('  No undefined names detected.')


if __name__ == '__main__':
    main(sys.argv[1])
