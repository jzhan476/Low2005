#!/bin/bash
# Full computational reproduction for this repo: same HARK replication as --comp min.
# (Legacy multi-day HA-Models pipeline removed; this runs Low2005 HARK replication.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/reproduce_environment.sh"

MIN_SCRIPT="$SCRIPT_DIR/reproduce_computed_min.sh"
if [[ -f "$MIN_SCRIPT" ]]; then
    bash "$MIN_SCRIPT"
else
    echo "❌ $MIN_SCRIPT not found"
    exit 1
fi

FLAG_FILE="$PROJECT_ROOT/reproduce/.results_pregenerated"
if [[ -f "$FLAG_FILE" ]]; then
    echo ""
    echo "========================================"
    echo "✅ Computation Complete"
    echo "========================================"
    echo ""
    echo "Removing PREGENERATED flag file..."
    rm -f "$FLAG_FILE"
    echo "✓ Flag removed - recompile Low2005.tex if captions referenced PREGENERATED"
    echo ""
else
    echo ""
    echo "ℹ️  No PREGENERATED flag file found (already removed or never created)"
    echo ""
fi
