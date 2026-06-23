#!/usr/bin/env bash
# Build a browser-rendered note that matches the VS Code Markdown preview:
#   note.html        — self-contained (images + CSS embedded); open it and
#                      Cmd+P -> Save as PDF for a faithful render.
#   note_render.pdf  — auto-printed from note.html via headless Chrome.
# Uses --katex: KaTeX is the same engine VS Code's preview uses, so it renders
# the note's \bold and \cal math (MathJax/MathML do not define \bold).  Images
# are inline at full width (no LaTeX float-scaling), which is why this looks
# better than the pandoc->LaTeX PDF.  The Chrome virtual-time budget lets KaTeX
# finish typesetting before the PDF is captured.
set -euo pipefail
cd "$(dirname "$0")"
# (no --metadata title: the note's own H1 is the title; a synthetic one duplicates it)
pandoc note.md -s --katex --embed-resources --css style.css --metadata title=" " -o note.html
echo "wrote note.html"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --run-all-compositor-stages-before-draw --virtual-time-budget=25000 \
    --print-to-pdf="$PWD/note_render.pdf" "file://$PWD/note.html" >/dev/null 2>&1
  echo "wrote note_render.pdf"
else
  echo "(no Chrome found — open note.html and Cmd+P -> Save as PDF)"
fi
