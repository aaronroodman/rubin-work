"""Render a note's Markdown to PDF on S3DF, without LaTeX or a browser.

pandoc turns the Markdown into HTML; each TeX math span is drawn to a PNG with
matplotlib mathtext; PyMuPDF's HTML layout (``fitz.Story``) writes the PDF. Figures are
read relative to the note's directory. Mathtext covers most of what the notes use; an
expression it cannot draw is left as its TeX source.

    python md_to_pdf.py 20261122-aos-bounce-aba/note.md          # writes note.pdf beside it
    python md_to_pdf.py note.md -o out.pdf
"""
import argparse
import html as htmllib
import pathlib
import re
import subprocess
import tempfile

# matplotlib before fitz: fitz (user site) otherwise loads the system libstdc++, too old
# for the stack's matplotlib.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

PAGE = fitz.paper_rect('letter')
MARGIN = 54  # pt, 0.75 in
FONT_PT = 10
MATH_DPI = 300
CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.3; }
h1 { font-size: 17pt; margin-bottom: 4pt; }
h2 { font-size: 13pt; margin-top: 14pt; }
h3 { font-size: 11pt; margin-top: 10pt; }
blockquote { color: #555; font-size: 9pt; margin-left: 0; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.5pt; width: 100%; }
th, td { border: 0.5pt solid #999; padding: 2pt 4pt; }
th { background-color: #eee; }
code { font-family: monospace; font-size: 8.5pt; }
figcaption, .caption { font-size: 8.5pt; color: #333; margin-bottom: 8pt; }
.display { text-align: center; margin: 6pt 0; }
"""
# TeX that mathtext lacks, mapped to what it has.
TEX_FIXES = [(r'\Big', ''), (r'\big', ''), (r'\lvert', '|'), (r'\rvert', '|'),
             (r'\qquad', r'\quad\quad'), (r'\;', r'\ '), (r'\top', 'T'),
             (r'{\rm min}', r'\mathrm{min}'), (r'\ge ', r'\geq '), (r'\le ', r'\leq ')]


def render_math(tex, path, display):
    for a, b in TEX_FIXES:
        tex = tex.replace(a, b)
    size = FONT_PT * (1.15 if display else 1.0)
    fig = plt.figure()
    fig.text(0, 0, f'${tex}$', fontsize=size)
    try:
        fig.savefig(path, dpi=MATH_DPI, bbox_inches='tight', pad_inches=0.01,
                    transparent=True)
    except (ValueError, RuntimeError):
        return None
    finally:
        plt.close(fig)
    w, h = fitz.Pixmap(str(path)).width, fitz.Pixmap(str(path)).height
    return w * 72 / MATH_DPI, h * 72 / MATH_DPI


def math_to_images(html, tmp):
    n = [0]

    def repl(m):
        display = m.group(1) == 'display'
        tex = htmllib.unescape(m.group(2))
        name = f'math{n[0]:03d}.png'
        n[0] += 1
        size = render_math(tex, tmp / name, display)
        if size is None:
            return f'<code>{htmllib.escape(tex)}</code>'
        img = f'<img src="{name}" width="{size[0]:.1f}" height="{size[1]:.1f}"/>'
        return f'<div class="display">{img}</div>' if display else img

    pat = r'<span\s+class="math (inline|display)">\\[\(\[](.*?)\\[\)\]]</span>'
    return re.sub(pat, repl, html, flags=re.S), n[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('markdown', type=pathlib.Path)
    ap.add_argument('-o', '--output', type=pathlib.Path)
    args = ap.parse_args()
    md = args.markdown.resolve()
    out = (args.output or md.with_suffix('.pdf')).resolve()
    html = subprocess.run(['pandoc', md.name, '-f', 'markdown+lists_without_preceding_blankline', '-t', 'html5', '--mathjax',
                           '--wrap=none'],
                          cwd=md.parent, check=True, capture_output=True, text=True).stdout
    # Story collapses tables that carry pandoc's relative column widths.
    html = re.sub(r'<colgroup>.*?</colgroup>', '', html, flags=re.S)
    # Full-width figures; math images keep their own size.
    html = re.sub(r'<img src="(?!math\d)', '<img style="width: 100%" src="', html)
    with tempfile.TemporaryDirectory() as tmp:
        html, n_math = math_to_images(html, pathlib.Path(tmp))
        archive = fitz.Archive(str(md.parent))
        archive.add(tmp)
        story = fitz.Story(html=html, user_css=CSS, archive=archive)
        writer = fitz.DocumentWriter(str(out))
        where = PAGE + (MARGIN, MARGIN, -MARGIN, -MARGIN)
        more = True
        while more:
            dev = writer.begin_page(PAGE)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
    # Story stores images uncompressed: about 20 times larger than needed.
    doc = fitz.open(out)
    n_pages = doc.page_count
    packed = out.with_suffix('.packed.pdf')
    doc.save(packed, garbage=4, deflate=True, deflate_images=True)
    doc.close()
    packed.replace(out)
    print(f'wrote {out} ({n_pages} pages, {n_math} math spans, '
          f'{out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
