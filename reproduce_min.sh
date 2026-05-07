#!/bin/bash
# =============================================================================
# Low2005 - Minimal reproduction entry point (REMARK convention)
# =============================================================================
# REMARK STANDARD.md allows a `reproduce_min.sh` at the repository root for
# projects whose full `reproduce.sh` takes ≥5 minutes. This script provides a
# fast validation path that runs ONLY the computational replication (the HARK
# lifecycle solve) and skips the LaTeX paper build.
#
# For the full reproduction (computation + figures + paper PDF), use:
#   ./reproduce.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec ./reproduce.sh --comp min "$@"
