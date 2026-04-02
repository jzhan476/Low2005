#!/bin/bash
# Legacy no-op: IMPC/LP pipeline removed; Low2005 figures come from Code/Python/Low2005.py.
# This Low2005 package builds paper figures via Code/Python/Low2005.py instead
# (run by ./reproduce.sh --comp min|full). Keep this as a no-op so --all and
# --data IMPC|LP|all do not fail or download obsolete artifacts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIGURE_TYPE="${1:-all}"

echo "========================================"
echo "Figures from results (Low2005)"
echo "========================================"
echo ""
echo "Paper figures for Low (2005) are produced by:"
echo "  Code/Python/Low2005.py"
echo "Run:"
echo "  ./reproduce.sh --comp min"
echo "  # or: ./reproduce_low2005.sh run"
echo ""
echo "Skipping legacy HA-Models figure pipeline (requested: ${FIGURE_TYPE})."
echo "✅ Done"
exit 0
