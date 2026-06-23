#!/usr/bin/env bash
# Build note.pdf from note.md for posting (pandoc -> xelatex).
# Notes:
#   - xelatex handles the Unicode in note.md (−, ±, μ, °, Δ, …) natively.
#   - STIX Two Math is used so bold Greek exists (Latin Modern Math lacks a bold Σ).
#   - unicode-math already defines \bold, and its bold routes Greek to a glyph
#     Latin Modern lacks; we override \bold -> \symbf *after* packages load.
set -euo pipefail
cd "$(dirname "$0")"
hdr=$(mktemp /tmp/notehdr.XXXX.tex)
printf '%s\n' '\AtBeginDocument{\providecommand{\bold}{}\renewcommand{\bold}[1]{\symbf{#1}}}' > "$hdr"
pandoc note.md -o note.pdf --pdf-engine=xelatex -H "$hdr" \
    -V geometry:margin=1in -V mathfont="STIX Two Math"
rm -f "$hdr"
echo "wrote $(pwd)/note.pdf"
