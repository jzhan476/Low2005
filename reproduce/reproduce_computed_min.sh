#!/bin/bash
# Minimal computational reproduction: Low (2005) HARK lifecycle replication.
# Writes PNG figures under Figures/ (paths in Code/Python/Low2005.py are relative to Code/Python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/reproduce_environment.sh"

export MATPLOTLIB_BACKEND=Agg
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-cache-low2005"
mkdir -p "$MPLCONFIGDIR"

LOW2005_PY="$PROJECT_ROOT/Code/Python/Low2005.py"
if [[ ! -f "$LOW2005_PY" ]]; then
    echo "❌ Missing $LOW2005_PY"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/Figures"

echo "========================================"
echo "Low (2005) computational replication"
echo "========================================"
echo ""
echo "Running: python Code/Python/Low2005.py"
echo ""

cd "$PROJECT_ROOT/Code/Python"
python Low2005.py

echo ""
echo "✅ Low (2005) replication script finished"
