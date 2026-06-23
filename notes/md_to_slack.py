#!/usr/bin/env python3
"""Convert standard (GitHub-flavored) Markdown to Slack mrkdwn.

    python md_to_slack.py note.md > note.slack.txt

Conversions:
  # Heading            -> *Heading*           (Slack has no heading syntax)
  **bold** / __bold__  -> *bold*
  *italic* / _italic_  -> _italic_
  ~~strike~~           -> ~strike~
  [text](url)          -> <url|text>
  - / * / + bullet     -> • bullet
  ![alt](src)          -> ":framed_picture: alt  (attach `src`)"

Passed through unchanged: fenced ``` code blocks and inline `code`.

Slack CANNOT render LaTeX math or tables. Math ($…$, $$…$$) is left raw — post
those as figures. Pipe-tables are wrapped in a ``` fence so columns stay
monospaced. Counts are reported on stderr.
"""
import re
import sys

C0, C1, C2 = '\x00', '\x01', '\x02'   # sentinels: inline-code, bold, inline-math


def convert(text):
    out, table = [], []
    in_fence = in_math = False
    warn = {'math': 0, 'table': 0, 'image': 0}

    def flush_table():
        if table:
            out.append('```'); out.extend(table); out.append('```')
            warn['table'] += 1; table.clear()

    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('```'):
            flush_table(); out.append(line); in_fence = not in_fence; continue
        if in_fence:
            out.append(line); continue
        if s == '$$':
            flush_table(); out.append(line)
            in_math = not in_math
            if in_math:
                warn['math'] += 1
            continue
        if in_math:
            out.append(line); continue
        # table row? (>=2 pipes, or a |---| separator)
        if line.count('|') >= 2 or re.match(r'^\s*\|?[\s:|-]+\|[\s:|-]*$', line):
            table.append(line); continue
        flush_table()
        out.append(_line(line, warn))
    flush_table()
    sys.stderr.write(
        f"[md_to_slack] {warn['math']} math block(s) left raw (Slack won't render "
        f"-> post as figures); {warn['table']} table(s) wrapped in code fences; "
        f"{warn['image']} image(s) -> attach to the Slack post manually.\n")
    return '\n'.join(out)


def _line(line, warn):
    codes, maths = [], []
    line = re.sub(r'`[^`]+`', lambda m: codes.append(m.group(0)) or f'{C0}{len(codes)-1}{C0}', line)
    line = re.sub(r'\$[^$]+\$', lambda m: maths.append(m.group(0)) or f'{C2}{len(maths)-1}{C2}', line)

    def img(m):
        warn['image'] += 1
        return f":framed_picture: {m.group(1) or 'figure'}  (attach `{m.group(2)}`)"
    line = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img, line)
    line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', line)

    h = re.match(r'^(#{1,6})\s+(.*)$', line)
    if h:
        line = f'*{h.group(2).strip()}*'
    else:
        line = re.sub(r'\*\*(.+?)\*\*', rf'{C1}\1{C1}', line)
        line = re.sub(r'__(.+?)__', rf'{C1}\1{C1}', line)
        line = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'_\1_', line)
        line = line.replace(C1, '*')
        line = re.sub(r'~~(.+?)~~', r'~\1~', line)
        line = re.sub(r'^(\s*)[-*+]\s+', r'\1• ', line)

    for i, mm in enumerate(maths):
        line = line.replace(f'{C2}{i}{C2}', mm)
    for i, c in enumerate(codes):
        line = line.replace(f'{C0}{i}{C0}', c)
    return line


if __name__ == '__main__':
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    sys.stdout.write(convert(src))
