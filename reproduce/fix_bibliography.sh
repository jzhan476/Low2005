#!/bin/bash
# Fix script for missing Low2005.bib bibliography file
# Downloads from GitHub raw URL (avoids git fetch which bloats .git/objects/)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Bibliography fix (Low2005.bib)"
echo "========================================"
echo ""

if [[ -f "Low2005.bib" ]] || [[ -f "HAFiscal.bib" ]]; then
    echo "✅ Low2005.bib or HAFiscal.bib already exists"
    exit 0
fi

echo "❌ Low2005.bib not found"
echo ""

# Upstream HAFiscal-QE branch hosts HAFiscal.bib; we save as Low2005.bib locally
GITHUB_REPO="${GITHUB_REPO:-llorracc/HAFiscal-QE}"
PRECOMPUTED_BRANCH="${PRECOMPUTED_BRANCH:-with-precomputed-artifacts}"
RAW_URL="https://raw.githubusercontent.com/${GITHUB_REPO}/${PRECOMPUTED_BRANCH}/HAFiscal.bib"

echo "Attempting to download from GitHub (saved as Low2005.bib)..."
echo "URL: $RAW_URL"
echo ""

if curl -L --fail --progress-bar -o Low2005.bib "$RAW_URL" 2>&1; then
    if [[ -f "Low2005.bib" && -s "Low2005.bib" ]]; then
        FILE_SIZE=$(du -h "Low2005.bib" 2>/dev/null | cut -f1)
        echo ""
        echo "✅ Successfully downloaded Low2005.bib ($FILE_SIZE)"
        exit 0
    fi
fi

rm -f Low2005.bib 2>/dev/null || true

echo ""
echo "⚠️  Could not download from GitHub"
echo ""
echo "Manual fixes:"
echo ""
echo "1. Check Figures/:"
if [[ -f "Figures/Low2005.bib" ]]; then
    cp Figures/Low2005.bib Low2005.bib
    echo "   ✅ Copied Figures/Low2005.bib"
    exit 0
elif [[ -f "Figures/HAFiscal.bib" ]]; then
    cp Figures/HAFiscal.bib Low2005.bib
    echo "   ✅ Copied Figures/HAFiscal.bib → Low2005.bib"
    exit 0
else
    echo "   ❌ No bibliography in Figures/"
fi
echo ""
echo "2. touch Low2005.bib  (empty; citations will be missing)"
echo ""
exit 1
