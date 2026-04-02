#!/bin/bash
# Check system dependencies for Low2005 reproduction
# System tools cannot be installed via uv/pip - they must be installed separately
#
# Usage: ./check-system-deps.sh [--install]
#   --install: Attempt to install missing dependencies (requires sudo)

set -e

INSTALL_MODE=false
if [[ "$1" == "--install" ]]; then
    INSTALL_MODE=true
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Low2005 System Dependencies Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Note: These are system-level tools (not Python packages)"
echo "      They cannot be installed via uv/pip"
echo ""

MISSING_DEPS=()
OPTIONAL_MISSING=()
ALL_OK=true

# ============================================================================
# Required: LaTeX (for PDF generation)
# ============================================================================
echo "📄 Checking LaTeX..."
if command -v pdflatex >/dev/null 2>&1; then
    VERSION=$(pdflatex --version | head -1)
    echo "  ✅ pdflatex: $VERSION"
    
    # Check if it's minimal or full
    if command -v kpsewhich >/dev/null 2>&1; then
        if kpsewhich mathkerncmssi8.tfm >/dev/null 2>&1; then
            echo "     (Full TeX Live - all fonts available)"
        else
            echo "     ⚠️  Minimal TeX Live - some fonts missing"
            echo "     Recommendation: Run ./setup-tex.sh --full"
        fi
    fi
else
    echo "  ❌ pdflatex NOT FOUND"
    MISSING_DEPS+=("texlive")
    ALL_OK=false
fi

if command -v latexmk >/dev/null 2>&1; then
    echo "  ✅ latexmk: $(latexmk -v | head -1)"
else
    echo "  ❌ latexmk NOT FOUND"
    MISSING_DEPS+=("latexmk")
    ALL_OK=false
fi

if command -v bibtex >/dev/null 2>&1; then
    echo "  ✅ bibtex: $(bibtex --version | head -1)"
else
    echo "  ❌ bibtex NOT FOUND"
    MISSING_DEPS+=("bibtex")
    ALL_OK=false
fi

# ============================================================================
# Optional: Docker (for DevContainer development)
# ============================================================================
echo ""
echo "🐳 Checking Docker (optional - for DevContainer)..."
if command -v docker >/dev/null 2>&1; then
    VERSION=$(docker --version)
    echo "  ✅ Docker installed: $VERSION"
    
    # Check if daemon is running
    if docker ps >/dev/null 2>&1; then
        echo "  ✅ Docker daemon running"
        echo "  ✅ User has Docker permissions"
    else
        # Check if it's a permission issue
        if docker ps 2>&1 | grep -q "permission denied"; then
            echo "  ⚠️  Docker daemon running, but permission denied"
            echo "     Fix: sudo usermod -aG docker \$USER"
            echo "     Then logout and login again"
        else
            echo "  ⚠️  Docker daemon not running"
            echo "     Fix: sudo systemctl start docker"
        fi
    fi
else
    echo "  ℹ️  Docker not installed (optional)"
    echo "     Only needed for DevContainer development"
    OPTIONAL_MISSING+=("docker")
fi

# ============================================================================
# Optional: Git (usually pre-installed)
# ============================================================================
echo ""
echo "📦 Checking Git..."
if command -v git >/dev/null 2>&1; then
    VERSION=$(git --version)
    echo "  ✅ $VERSION"
else
    echo "  ❌ git NOT FOUND"
    MISSING_DEPS+=("git")
    ALL_OK=false
fi

# ============================================================================
# Summary and Installation Instructions
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$ALL_OK" == true ]]; then
    echo "✅ All required dependencies installed!"
    
    if [[ ${#OPTIONAL_MISSING[@]} -gt 0 ]]; then
        echo ""
        echo "ℹ️  Optional dependencies missing:"
        for dep in "${OPTIONAL_MISSING[@]}"; do
            echo "   - $dep"
        done
        echo ""
        echo "These are optional - you can proceed without them."
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Install Python dependencies: uv sync --all-groups"
    echo "  2. Build PDFs: ./reproduce.sh --docs main"
    exit 0
else
    echo "❌ Missing required dependencies:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "   - $dep"
    done
    echo ""
    
    if [[ "$INSTALL_MODE" == true ]]; then
        echo "🔧 Attempting to install missing dependencies..."
        echo ""
        
        # Detect OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            if command -v apt-get >/dev/null 2>&1; then
                echo "Using apt-get (Debian/Ubuntu)..."
                
                if [[ " ${MISSING_DEPS[*]} " =~ " texlive " ]]; then
                    echo "Installing TeX Live..."
                    sudo apt-get update
                    sudo apt-get install -y texlive-latex-base texlive-latex-recommended latexmk
                fi
                
                if [[ " ${MISSING_DEPS[*]} " =~ " git " ]]; then
                    echo "Installing git..."
                    sudo apt-get install -y git
                fi
                
                if [[ " ${OPTIONAL_MISSING[*]} " =~ " docker " ]]; then
                    read -p "Install Docker? (y/N) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        echo "Installing Docker..."
                        sudo apt-get install -y docker.io
                        sudo systemctl start docker
                        sudo systemctl enable docker
                        sudo usermod -aG docker "$USER"
                        echo "⚠️  You need to logout and login for Docker permissions to take effect"
                    fi
                fi
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew >/dev/null 2>&1; then
                echo "Using Homebrew (macOS)..."
                
                if [[ " ${MISSING_DEPS[*]} " =~ " texlive " ]]; then
                    echo "Installing BasicTeX..."
                    brew install --cask basictex
                fi
                
                if [[ " ${MISSING_DEPS[*]} " =~ " git " ]]; then
                    echo "Installing git..."
                    brew install git
                fi
                
                if [[ " ${OPTIONAL_MISSING[*]} " =~ " docker " ]]; then
                    read -p "Install Docker Desktop? (y/N) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        echo "Installing Docker Desktop..."
                        brew install --cask docker
                        echo "⚠️  Start Docker Desktop from Applications folder"
                    fi
                fi
            fi
        fi
        
        echo ""
        echo "Re-run this script to verify installation:"
        echo "  ./check-system-deps.sh"
    else
        echo "📝 Installation instructions:"
        echo ""
        echo "Option 1: Run automated installer (requires sudo):"
        echo "  ./check-system-deps.sh --install"
        echo ""
        echo "Option 2: Manual installation:"
        echo ""
        
        if [[ " ${MISSING_DEPS[*]} " =~ " texlive " ]]; then
            echo "  LaTeX:"
            echo "    Linux:  ./setup-tex.sh --full"
            echo "    macOS:  brew install --cask mactex"
            echo ""
        fi
        
        if [[ " ${MISSING_DEPS[*]} " =~ " git " ]]; then
            echo "  Git:"
            echo "    Linux:  sudo apt-get install git"
            echo "    macOS:  brew install git"
            echo ""
        fi
        
        if [[ " ${OPTIONAL_MISSING[*]} " =~ " docker " ]]; then
            echo "  Docker (optional):"
            echo "    Linux:  sudo apt-get install docker.io"
            echo "    macOS:  brew install --cask docker"
            echo "    See: https://docs.docker.com/get-docker/"
            echo ""
        fi
        
        echo "Option 3: Use DevContainer (Docker-based, installs everything):"
        echo "  Open in VS Code → 'Reopen in Container'"
        echo ""
    fi
    
    exit 1
fi

