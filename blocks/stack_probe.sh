#!/usr/bin/env bash
echo "=== shell flags (has i=interactive, l=login?): [$-] ==="
echo "host=$(hostname)  user=$(whoami)"
echo "--- did startup files run? ---"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "EUPS_PATH=$EUPS_PATH"
echo "--- python resolution ---"
which python || echo ">> python NOT on PATH"
python --version 2>&1 || true
python -c 'import sys; print("sys.executable:", sys.executable)' 2>&1 || true
echo "--- key stack imports ---"
python -c 'import lsst.summit.utils, lsst_efd_client, lsst.ts.ofc; print("imports OK")' 2>&1 || true
echo "--- PATH ---"
echo "$PATH" | tr ':' '\n'
