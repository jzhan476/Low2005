#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-cache-low2005"
mkdir -p "$MPLCONFIGDIR"
export MATPLOTLIB_BACKEND="${MATPLOTLIB_BACKEND:-Agg}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/Code/Python/Low2005.py"

MODE="${1:-run}"

case "$MODE" in
  run)
    mkdir -p "$ROOT/Figures"
    cd "$ROOT/Code/Python"
    python Low2005.py
    ;;
  notebook)
    echo "Open or run: $ROOT/Code/Python/Low2005.ipynb"
    if command -v jupyter >/dev/null 2>&1; then
      exec jupyter notebook "$ROOT/Code/Python/Low2005.ipynb"
    else
      echo "Install Jupyter (optional): uv sync --group jupyter"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [run|notebook]"
    echo "  run      — python Code/Python/Low2005.py (default)"
    echo "  notebook — launch Jupyter on Low2005.ipynb"
    exit 1
    ;;
esac
