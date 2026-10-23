"""Index of every PDF page in the study's run directories: ``figure_index.csv``.

One row per page of every ``*.pdf`` under ``output/<run>/`` (not in ``images/`` or a
``*_isr_bin4/`` cache). Columns:

- ``id``: ``P<pass>-<file>-<page>`` (``P6-pupil_pass6-12``); pass from `PASS_OF_RUN`,
  ``S`` for the selection runs;
- ``pass``, ``run``, ``file``, ``page`` (1-based) and ``title``: the page's title, the
  topmost text line in the largest font that is not a bare number (matplotlib's
  suptitle, or a text page's heading);
- ``relevant``: ``yes``, ``superseded`` or ``context``, from ``findings.md``: a page is
  ``yes`` when a settled or open finding cites it, ``superseded`` when only superseded
  findings cite it, and ``context`` otherwise. A finding cites pages in its evidence
  column as ``<run>/<file>.pdf p<N>`` or ``p<N>-<M>``, or a whole PDF.

Rerunnable: a later pass reruns it after adding its run to `PASS_OF_RUN` and its rows to
``findings.md``. Writes ``figure_index.csv`` in the study directory.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/figure_index.py
"""
import pathlib
import re

import pandas as pd

STUDY_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = STUDY_DIR / 'output'
FINDINGS = STUDY_DIR / 'findings.md'
PASS_OF_RUN = {
    '20261018_candidates': 'S', '20261018_fits': '1', '20261018_pupil': '1',
    '20261018_spider_validation': '1', '20261019_pupil2': '2', '20261020_candidates_v2': 'S',
    '20261020_selection_diag': 'S', '20261020_test_v2': 'S', '20261020_pupil3': '3',
    '20261021_pupil4': '4', '20261022_pupil5': '5', '20261023_pupil6': '6',
    '20261023_night2_review': 'N2', '20261023_night2_20260127': 'N2',
    '20261023_night2_20250920': 'N2', '20261023_night2_20251027': 'N2',
}
_NUMBER = re.compile(r'^[\s\d.,+\-−eE%()]*$')


def page_title(page):
    spans = [(s['size'], s['bbox'][1], s['text'].strip())
             for b in page.get_text('dict')['blocks'] for ln in b.get('lines', [])
             for s in ln['spans'] if s['text'].strip() and not _NUMBER.match(s['text'])]
    if not spans:
        return ''
    top = max(s[0] for s in spans)
    line = sorted((s for s in spans if s[0] >= top - 0.1), key=lambda s: s[1])
    y0 = line[0][1]
    return ' '.join(s[2] for s in line if abs(s[1] - y0) < 2)[:200]


def citations():
    """{(run, file): {page or 0 (whole file): set of statuses}} from findings.md."""
    if not FINDINGS.exists():
        return {}
    cites = {}
    for ln in FINDINGS.read_text().splitlines():
        if not ln.startswith('|') or ln.startswith('|---'):
            continue
        cells = [c.strip() for c in ln.strip('|').split('|')]
        if len(cells) < 5:
            continue
        status = cells[2].split()[0].lower() if cells[2] else ''
        for m in re.finditer(r'(\d{8}_\w+)/([\w.\-]+\.pdf)(?:\s+p(\d+)(?:[-–](\d+))?)?', cells[3]):
            run, f, a, b = m.groups()
            pages = range(int(a), int(b or a) + 1) if a else [0]
            for p in pages:
                cites.setdefault((run, f), {}).setdefault(p, set()).add(status)
    return cites


def relevant(statuses):
    if not statuses:
        return 'context'
    if statuses & {'settled', 'open'}:
        return 'yes'
    return 'superseded' if 'superseded' in statuses else 'context'


def main():
    import fitz
    cites = citations()
    rows = []
    for pdf in sorted(OUTPUT.glob('*/*.pdf')):
        run = pdf.parent.name
        if run.endswith('_isr_bin4'):
            continue
        p = PASS_OF_RUN.get(run, '?')
        doc = fitz.open(pdf)
        c = cites.get((run, pdf.name), {})
        for i, page in enumerate(doc, start=1):
            st = c.get(i, set()) | c.get(0, set())
            rows.append(dict(id=f'P{p}-{pdf.stem}-{i}', **{'pass': p}, run=run, file=pdf.name, page=i,
                             title=page_title(page), relevant=relevant(st)))
    df = pd.DataFrame(rows)
    df.to_csv(STUDY_DIR / 'figure_index.csv', index=False)
    print(len(df), 'pages in', df.file.nunique(), 'PDFs;', df.relevant.value_counts().to_dict())


if __name__ == '__main__':
    main()
