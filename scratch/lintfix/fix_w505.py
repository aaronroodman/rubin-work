"""Reflow the comment blocks and docstring paragraphs that flake8 reports as W505.

The unit of work is a PARAGRAPH, not a line.  Wrapping a single long line on its own
leaves an orphaned fragment on the next line ("(default)", "set,"), which reads far worse
than the warning it silences; so for each flagged line this collects the contiguous run of
sibling lines at the same indent that form one paragraph, joins them, and rewraps the
whole run to 79 columns.

Only pure comment blocks and docstring prose are touched.  A paragraph is left alone, and
reported, when it holds aligned structure (two or more runs of 2+ spaces inside the text,
a `--` arrow table, a numpydoc `name : type` header) or a token too long to wrap.
"""
import io
import re
import subprocess
import sys
import textwrap
import tokenize
from pathlib import Path

PKG = Path('/sdf/home/r/roodman/u/LSST/packages/ts_intrinsic_wavefront')
LIMIT = 79


def flagged(paths):
    """Map of path -> sorted line numbers flake8 reports as W505 (1-based)."""
    out = subprocess.run(
        ['flake8'] + list(paths), cwd=PKG, capture_output=True, text=True).stdout
    hits = {}
    for line in out.splitlines():
        if ' W505 ' not in line:
            continue
        path, lineno = line.split(':')[0], int(line.split(':')[1])
        hits.setdefault(path, set()).add(lineno)
    return {p: sorted(v) for p, v in hits.items()}


def string_spans(src):
    """(start, end) 1-based line ranges of every STRING token, for docstring detection."""
    spans = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING:
                spans.append((tok.start[0], tok.end[0]))
    except (tokenize.TokenError, IndentationError):
        pass
    return spans


def in_docstring(n, spans):
    return any(a <= n <= b for a, b in spans)


def classify(line):
    """('comment', indent, marker, body) / ('doc', indent, '', body) / (None, ...)."""
    m = re.match(r'^(\s*)(#+)(\s?)(.*)$', line)
    if m:
        return 'comment', m.group(1), m.group(2) + (m.group(3) or ' '), m.group(4)
    m = re.match(r'^(\s*)(.*)$', line)
    return 'doc', m.group(1), '', m.group(2)


def is_structure(body):
    """True when the text is an aligned table or a numpydoc header, not flowing prose."""
    if len(re.findall(r'\S {2,}\S', body)) >= 1 and re.search(r'\s{3,}', body):
        return True
    if re.match(r'^\s*[\w`*\[\]()., ]+ : ', body):          # numpydoc `name : type`
        return True
    if re.match(r'^\s*[-*+•]\s', body) or re.match(r'^\s*\d+[.)]\s', body):
        return True                                          # list item: own indent
    if '--' in body and re.search(r'\s{2,}', body):
        return True
    if re.match(r'^\s*(>>>|\$|python |snakemake )', body):   # a runnable example
        return True
    return False


def paragraph_bounds(lines, n, kind, indent, marker):
    """Half-open [lo, hi) run of lines forming one paragraph around index n (0-based)."""
    def same(i):
        if not 0 <= i < len(lines):
            return False
        k, ind, mk, body = classify(lines[i])
        if not body.strip():
            return False
        if k != kind or ind != indent:
            return False
        if kind == 'comment' and mk != marker:
            return False
        if is_structure(body):
            return False
        # A docstring opener/closer is a boundary: reflowing across it moves the quotes.
        if kind == 'doc' and ('"""' in lines[i] or "'''" in lines[i]):
            return False
        return True

    if not same(n):
        return None
    lo = hi = n
    while same(lo - 1):
        lo -= 1
    while same(hi + 1):
        hi += 1
    return lo, hi + 1


def main(paths):
    report = []
    for path, linenos in sorted(flagged(paths).items()):
        f = PKG / path
        lines = f.read_text().splitlines()
        spans = string_spans('\n'.join(lines))
        done = set()
        fixed = 0
        for n1 in sorted(linenos, reverse=True):
            n = n1 - 1
            if n in done:
                continue
            kind, indent, marker, body = classify(lines[n])
            if kind == 'doc' and not in_docstring(n1, spans):
                report.append(f'{path}:{n1} SKIP not comment or docstring')
                continue
            if is_structure(body) or ('"""' in lines[n] or "'''" in lines[n]):
                report.append(f'{path}:{n1} SKIP aligned structure or quote line')
                continue
            bounds = paragraph_bounds(lines, n, kind, indent, marker)
            if bounds is None:
                report.append(f'{path}:{n1} SKIP no paragraph')
                continue
            lo, hi = bounds
            prefix = indent + marker
            text = ' '.join(classify(lines[i])[3].strip() for i in range(lo, hi))
            wrapped = textwrap.wrap(
                text, width=LIMIT - len(prefix), break_long_words=False,
                break_on_hyphens=False)
            if not wrapped or max(len(prefix + w) for w in wrapped) > LIMIT:
                report.append(f'{path}:{n1} SKIP unsplittable token')
                continue
            lines[lo:hi] = [prefix + w for w in wrapped]
            done.update(range(lo, hi))
            fixed += 1
        f.write_text('\n'.join(lines) + '\n')
        print(f'{path}: {fixed} paragraph(s) reflowed of {len(linenos)} flagged')
    for r in report:
        print(r)


if __name__ == '__main__':
    main(sys.argv[1:])
