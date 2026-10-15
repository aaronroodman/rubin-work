"""Split the compound statements flake8 reports as E702 (semicolon) and E701 (colon).

E702 `a(); b()` becomes two lines at the same indent.  E701 `if c: return x` becomes a
two-line if/body.  Both splits are done with the tokenizer, so a semicolon or colon inside
a string literal or a comment is never mistaken for a separator, and a line carrying a
trailing comment keeps it on the first statement.

Lines whose split is not purely mechanical (a one-line `def`/`class`, a match/case, an
annotated target, anything with more than one colon at depth 0) are left alone and
reported.
"""
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

PKG = Path('/sdf/home/r/roodman/u/LSST/packages/ts_intrinsic_wavefront')


def flagged(paths, code):
    out = subprocess.run(
        ['flake8'] + list(paths), cwd=PKG, capture_output=True, text=True).stdout
    hits = {}
    for line in out.splitlines():
        if f' {code} ' not in line:
            continue
        path, lineno = line.split(':')[0], int(line.split(':')[1])
        hits.setdefault(path, set()).add(lineno)
    return {p: sorted(v) for p, v in hits.items()}


def depth0_positions(text, want):
    """Column offsets of `want` tokens at bracket depth 0, outside strings/comments."""
    out, depth = [], 0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError):
        return []
    for tok in toks:
        if tok.type == tokenize.OP:
            if tok.string in '([{':
                depth += 1
            elif tok.string in ')]}':
                depth -= 1
            elif tok.string == want and depth == 0:
                out.append(tok.start[1])
    return out


def split_semicolons(line):
    """`a(); b()` -> ['a()', 'b()'] at the original indent, or None."""
    indent = re.match(r'^(\s*)', line).group(1)
    body = line[len(indent):]
    cols = depth0_positions(body, ';')
    if not cols:
        return None
    parts, prev = [], 0
    for c in cols:
        parts.append(body[prev:c])
        prev = c + 1
    parts.append(body[prev:])
    parts = [p.strip() for p in parts]
    # A trailing empty part is just `a();` -- drop it rather than emit a blank line.
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return [indent + p for p in parts]


def split_colon(line):
    """`if c: body` -> ['if c:', '    body'], or None when not a simple one-liner."""
    indent = re.match(r'^(\s*)', line).group(1)
    body = line[len(indent):]
    if re.match(r'^(def|class|async|match|case)\b', body):
        return None
    if not re.match(r'^(if|elif|else|for|while|try|except|finally|with)\b', body):
        return None
    cols = depth0_positions(body, ':')
    if len(cols) != 1:
        return None
    head, tail = body[:cols[0] + 1], body[cols[0] + 1:].strip()
    if not tail or tail.startswith('#'):
        return None
    return [indent + head, indent + '    ' + tail]


def main(paths):
    report = []
    for code, splitter in (('E702', split_semicolons), ('E701', split_colon)):
        for path, linenos in sorted(flagged(paths, code).items()):
            f = PKG / path
            lines = f.read_text().splitlines()
            for n in sorted(linenos, reverse=True):
                new = splitter(lines[n - 1])
                if new is None:
                    report.append(f'{path}:{n} SKIP {code} not a mechanical split')
                    continue
                lines[n - 1:n] = new
            f.write_text('\n'.join(lines) + '\n')
            print(f'{code} {path}: {len(linenos)} flagged')
    for r in report:
        print(r)


if __name__ == '__main__':
    main(sys.argv[1:])
