#!/usr/bin/env bash
# Convert note.md -> two_epochs_body.tex : an lsstdoc-ready LaTeX fragment.
#   - drops the H1 (it becomes \title in two_epochs.tex)
#   - strips .png from \includegraphics so pdflatex picks the vector figures/*.pdf
#   - drops pandoc's caption-less-longtable "\def\LTcaptype{none}" (trips a counter)
#   - maps the Unicode glyphs used in the Markdown to LaTeX so it builds under pdflatex
# Re-run whenever note.md changes; then build two_epochs.tex with lsst-texmf.
set -euo pipefail
cd "$(dirname "$0")"
tail -n +2 note.md \
  | pandoc -f markdown -t latex --top-level-division=section \
  | sed 's#\.png}#}#g; s#\\def\\LTcaptype{none}##g' \
  | python3 -c 'import sys
repl = {"−":"-", "–":"--", "—":"---", "±":r"$\pm$", "°":r"$^\circ$", "μ":r"$\mu$",
        "×":r"$\times$", "≥":r"$\ge$", "≤":r"$\le$", "≈":r"$\approx$", "→":r"$\to$",
        "·":r"$\cdot$", "∇":r"$\nabla$", "Δ":r"$\Delta$", "σ":r"$\sigma$"}
t = sys.stdin.read()
for k, v in repl.items():
    t = t.replace(k, v)
sys.stdout.write(t)' \
  > two_epochs_body.tex
echo "wrote two_epochs_body.tex ($(wc -l < two_epochs_body.tex) lines)"
