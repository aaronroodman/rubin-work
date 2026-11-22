#!/usr/bin/env bash
# Convert note.md -> elevation_pairs_body.tex : an lsstdoc-ready LaTeX fragment.
#   - drops the H1 (it becomes \title in an lsstdoc wrapper)
#   - strips .png from \includegraphics so pdflatex picks the vector figures/*.pdf
#   - drops pandoc's caption-less-longtable "\def\LTcaptype{none}" (trips a counter)
#   - maps the Unicode glyphs used in the Markdown to LaTeX so it builds under pdflatex
# Re-run whenever note.md changes; then build an lsstdoc wrapper with lsst-texmf.
set -euo pipefail
cd "$(dirname "$0")"
tail -n +2 note.md \
  | pandoc -f markdown -t latex --top-level-division=section \
  | sed 's#\.png}#}#g; s#\\def\\LTcaptype{none}##g' \
  | python3 -c 'import sys
repl = {"−":"-", "–":"--", "—":"---", "±":r"$\pm$", "°":r"$^\circ$", "μ":r"$\mu$",
        "×":r"$\times$", "≥":r"$\ge$", "≤":r"$\le$", "≈":r"$\approx$", "→":r"$\to$",
        "·":r"$\cdot$", "∇":r"$\nabla$", "Δ":r"$\Delta$", "σ":r"$\sigma$",
        "µ":r"$\mu$", "ρ":r"$\rho$", "⁻⁵":r"$^{-5}$", "₁":r"$_1$", "₂":r"$_2$"}
t = sys.stdin.read()
for k, v in repl.items():
    t = t.replace(k, v)
sys.stdout.write(t)' \
  > elevation_pairs_body.tex
echo "wrote elevation_pairs_body.tex ($(wc -l < elevation_pairs_body.tex) lines)"
