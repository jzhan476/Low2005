# Low2005 Reproduction Scripts

Scripts for building `Low2005.pdf` and running the HARK replication (`Code/Python/Low2005.py`) via `reproduce.sh`.

## 📋 Quick Reference

```bash
../reproduce.sh --docs main     # Build Low2005.tex → Low2005.pdf
../reproduce.sh --comp min      # Run Code/Python/Low2005.py
../reproduce.sh --comp full     # Same as min (legacy name)
../reproduce.sh --envt texlive  # Test LaTeX environment
../reproduce.sh --envt comp_uv  # Test Python/UV environment
```

The scripts in this directory are typically called by `reproduce.sh` or `reproduce.py`, but can also be used directly for specific tasks.

---

## 🔧 Environment Setup Scripts

### `reproduce_environment.sh`
**Purpose:** Main environment setup wrapper that delegates to UV-based setup  
**When to use:** Called automatically by `reproduce.sh`, but can be sourced directly  
**Details:** Attempts UV environment setup first (via `reproduce_environment_comp_uv.sh`), falls back to conda if UV unavailable

### `reproduce_environment_comp_uv.sh`
**Purpose:** Sets up Python computational environment using UV package manager  
**When to use:** First-time setup, or when recreating Python environment  
**What it does:**

- Installs UV if not present
- Creates `.venv` virtual environment
- Installs Python dependencies from `pyproject.toml`
- Creates verification marker on success

**Usage:**

```bash
./reproduce/reproduce_environment_comp_uv.sh
```

### `reproduce_environment_texlive.sh`
**Purpose:** Verifies TeX Live installation and required LaTeX packages  
**When to use:** To verify LaTeX environment is properly configured  
**What it does:**

- Checks for `pdflatex`, `bibtex`, `latexmk`
- Verifies required LaTeX packages from `required_latex_packages.txt`
- Creates verification marker on success

**Usage:**

```bash
./reproduce/reproduce_environment_texlive.sh
```

---

## 📄 Document Reproduction Scripts

> **Note:** For HTML generation (optional), see [`reproduce_html_README.md`](reproduce_html_README.md)

### `reproduce_documents.sh`
**Purpose:** Main LaTeX document compilation script  
**When to use:** Called by `reproduce.sh --docs [target]`  
**What it does:**

- Sets up proper `BIBINPUTS` and `BSTINPUTS` environment variables
- Compiles specified LaTeX documents using `latexmk`
- Handles both main document and subfiles
- Supports different document variants (main, appendices)

**Arguments:**

- `main` - Compile Low2005.tex (main document)
- `subfiles` - Compile individual subfiles
- `all` - Compile everything

**Usage:**

```bash
./reproduce/reproduce_documents.sh main
```

### `reproduce-standalone-files.sh`
**Purpose:** Compile standalone LaTeX files (figures, tables, subfiles)  
**When to use:** To compile individual components without full document build  
**Options:**

- `--figures` - Compile all .tex files in Figures/
- `--tables` - Compile all .tex files in Tables/
- `--subfiles` - Compile all .tex files in Subfiles/
- `--all` - Compile all standalone files
- `--clean-first` - Clean auxiliary files before compilation
- `--continue` - Continue even if some files fail

**Usage:**

```bash
./reproduce/reproduce-standalone-files.sh --figures
./reproduce/reproduce-standalone-files.sh --all --clean-first
```

---

## 🧮 Computational reproduction

### `reproduce_computed_min.sh`
Runs `Code/Python/Low2005.py` (from `Code/Python/`), creates `Figures/*.png`, and ensures `Figures/` exists.

### `reproduce_computed.sh`
Calls `reproduce_computed_min.sh`, then removes `reproduce/.results_pregenerated` if present.

### `reproduce_figures_from_results.sh`
Legacy no-op: Low2005 figures are produced by `Low2005.py`, not from pre-built HA-Model results.

---

## 📊 Benchmarking System

The `benchmarks/` subdirectory contains scripts for performance measurement and system information capture.

### `benchmarks/benchmark.sh`
**Purpose:** Wrapper script that times reproduction runs and captures system info  
**Usage:**

```bash
./reproduce/benchmarks/benchmark.sh ../reproduce.sh --docs main
```

### `benchmarks/capture_system_info.py`
**Purpose:** Capture detailed system information for benchmark reports  
**Output:** JSON files in `benchmarks/results/`

See [`benchmarks/README.md`](benchmarks/README.md) for detailed benchmarking documentation.

---

## 📁 Support Files

### `required_latex_packages.txt`
**Purpose:** List of required LaTeX packages for environment verification  
**Format:** One package name per line  
**Used by:** `reproduce_environment_texlive.sh`

---

## 🔍 Verification Markers

Successful environment verifications create marker files:

- `reproduce_environment_texlive_YYYYMMDD-HHMM.verified` - LaTeX environment verified
- `reproduce_environment_comp_uv_YYYYMMDD-HHMM.verified` - Python environment verified

These markers indicate when the environment was last successfully verified and are automatically ignored by git.

---

## 🏛️ Old Scripts

The `old/` subdirectory contains deprecated scripts kept for reference:

- Legacy testing scripts
- Private environment setup scripts
- Old cross-platform test implementations

These are not used in current workflows.

---

## 🚀 Typical Workflows

### First-Time Setup

```bash
./reproduce/reproduce_environment_comp_uv.sh
../reproduce.sh --envt texlive
../reproduce.sh --envt comp_uv
../reproduce.sh --comp min
../reproduce.sh --docs main
```

### Quick Document Build

```bash
# Just rebuild the LaTeX document (no computations)
../reproduce.sh --docs main
```

### Testing After Code Changes

```bash
../reproduce.sh --comp min
../reproduce.sh --docs main
```

### Full workflow

```bash
../reproduce.sh --comp min
../reproduce.sh --docs main
```

---

## 🔗 Related Documentation

- **Main README:** [`../README.md`](../README.md) - Project overview and quick start
- **REMARK:** [`../REMARK.md`](../REMARK.md) - Replication notes
- **Benchmarking:** [`benchmarks/README.md`](benchmarks/README.md) - Performance testing

---

## 💡 Tips

1. **Always use `reproduce.sh`** in the parent directory rather than calling these scripts directly, unless you know what you're doing
2. **Check verification markers** - If environment setup scripts have already created recent `.verified` files, you may not need to run them again
3. **Computational reproduction** - Requires `Code/HA-Models/` (restore from git if removed in a minimal checkout)
4. **Check benchmarks/** for performance data from previous runs
5. **Read script headers** - Each script has detailed comments explaining its purpose and usage

---

**Last Updated:** 2025-10-30  
**Maintainer:** See main repository README
