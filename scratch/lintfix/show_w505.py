"""Print every W505 line flake8 reports, with its file and length, for hand rewrapping."""
import subprocess
import sys
from pathlib import Path

PKG = Path('/sdf/home/r/roodman/u/LSST/packages/ts_intrinsic_wavefront')

out = subprocess.run(
    ['flake8'] + sys.argv[1:], cwd=PKG, capture_output=True, text=True).stdout

for line in out.splitlines():
    if ' W505 ' not in line:
        continue
    path, lineno = line.split(':')[0], int(line.split(':')[1])
    text = (PKG / path).read_text().splitlines()[lineno - 1]
    print(f'{path}:{lineno}: len={len(text)}')
    print(f'    {text}')
