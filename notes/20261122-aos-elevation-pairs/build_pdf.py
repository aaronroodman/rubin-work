"""Render note.md to note.pdf: pandoc to HTML, then PyMuPDF's HTML layout (Story).

No LaTeX or browser is needed, so it runs on S3DF. For the tech note use
``convert_to_latex.sh`` instead.

    python build_pdf.py
"""
import pathlib
import re
import subprocess

import fitz

HERE = pathlib.Path(__file__).resolve().parent
PAGE = fitz.paper_rect('letter')
MARGIN = 54  # pt, 0.75 in
CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.3; }
h1 { font-size: 17pt; margin-bottom: 4pt; }
h2 { font-size: 13pt; margin-top: 14pt; }
blockquote { color: #555; font-size: 9pt; margin-left: 0; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.5pt; width: 100%; }
th, td { border: 0.5pt solid #999; padding: 2pt 4pt; }
th { background-color: #eee; }
code { font-family: monospace; font-size: 8.5pt; }
figcaption, .caption { font-size: 8.5pt; color: #333; margin-bottom: 8pt; }
img { width: 100%; }
"""


def main():
    html = subprocess.run(['pandoc', 'note.md', '-f', 'markdown', '-t', 'html5'],
                          cwd=HERE, check=True, capture_output=True, text=True).stdout
    # Story collapses tables that carry pandoc's relative column widths.
    html = re.sub(r'<colgroup>.*?</colgroup>', '', html, flags=re.S)
    story = fitz.Story(html=html, user_css=CSS, archive=str(HERE))
    out = HERE / 'note.pdf'
    writer = fitz.DocumentWriter(str(out))
    where = PAGE + (MARGIN, MARGIN, -MARGIN, -MARGIN)
    more = True
    while more:
        dev = writer.begin_page(PAGE)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    print(f'wrote {out} ({fitz.open(out).page_count} pages)')


if __name__ == '__main__':
    main()
