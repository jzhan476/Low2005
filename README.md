# Low2005 Replication

[![Reproducibility](https://github.com/jzhan476/Low2005/actions/workflows/reproduce.yml/badge.svg?branch=main)](https://github.com/jzhan476/Low2005/actions/workflows/reproduce.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python 3.9-3.12](https://img.shields.io/badge/Python-3.9--3.12-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![REMARK Tier 2](https://img.shields.io/badge/REMARK-Tier%202-brightgreen)](./REMARK.md)

Replication package for:

Hamish W. Low (2005), *Self-Insurance in a Life-Cycle Model of Labour Supply and Savings*,
*Review of Economic Dynamics*, 8(4), 945-975.  
DOI: <https://doi.org/10.1016/j.red.2005.03.002>

This repository reproduces the paper using **HARK** (`HARK.ConsumptionSaving.ConsLaborModel`
and `ConsIndShockModel`) and a LaTeX paper build rooted at `Low2005.tex`.

## What This Repository Produces

- a computational replication in `Code/Python/Low2005.py`
- a notebook version in `Code/Python/Low2005.ipynb`
- generated figures under `Figures/`
- a compiled paper `Low2005.pdf`

The main entrypoint for reproduction is `reproduce.sh`. Run it with `--all`
to execute the full pipeline (computational replication + paper), or with
`--help` to see all available subcommands. For a fast validation path, use
`./reproduce_min.sh`.

## REMARK-Oriented Quick Start

### Recommended setup: UV

Create the Python environment:

```bash
./reproduce/reproduce_environment_comp_uv.sh
```

Then test the computational environment:

```bash
./reproduce.sh --envt comp_uv
```

Run the computational replication:

```bash
./reproduce.sh --comp min
```

Build the paper:

```bash
./reproduce.sh --docs main
```

### Conda / Binder-compatible setup

For REMARK/Binder compatibility, the canonical conda environment file is
`binder/environment.yml`. It is intentionally minimal: it installs **Python 3.11**
(from conda-forge) plus `uv`, and the actual pinned dependencies are then materialized from
`uv.lock` (this pattern is endorsed by the REMARK STANDARD for projects whose
primary environment manager is `uv`, `poetry`, or similar). The root `pyproject.toml`
declares `requires-python = ">=3.9,<3.13"`, so local installs on 3.9–3.12 are also supported.

```bash
conda env create -f binder/environment.yml
conda activate low2005
uv sync --frozen          # materialize pinned deps from uv.lock
```

When launched on mybinder.org, the same materialization happens automatically
via `binder/postBuild`.

After activation you can run the standard reproduction commands:

```bash
./reproduce.sh --all          # full reproduction
./reproduce.sh --comp min     # computation only
./reproduce.sh --docs main    # paper only
```

## Expected Outputs

After `./reproduce.sh --comp min`, the main expected outputs are:

- PNG figures under `Figures/`

After `./reproduce.sh --docs main`, the main expected output is:

```bash
Low2005.pdf
```

## Repository Layout

| Path | Role |
|------|------|
| `README.md` | Top-level overview and reproduction instructions |
| `REMARK.md` | REMARK metadata and short replication summary |
| `CITATION.cff` | Citation metadata for the repository (software; v1.1.0) |
| `LICENSE` | Apache-2.0 license |
| `Dockerfile` | Container image (TeX Live + `uv` environment) |
| `binder/environment.yml` | REMARK/Binder-compatible conda environment file |
| `Code/Python/` | `Low2005.py` and `Low2005.ipynb` |
| `Figures/` | Generated computational output figures |
| `Low2005.tex` | Main LaTeX paper |
| `reproduce.sh` | Main orchestration script (use `--all` for full reproduction, `--help` for usage) |
| `reproduce_min.sh` | Fast validation: computational replication only (no LaTeX build) |
| `reproduce/` | Supporting environment, build, and utility scripts |
| `binder/postBuild` | Materializes the pinned Python environment from `uv.lock` |
| `@local/` | Local LaTeX configuration |
| `@resources/` | Bundled LaTeX and project resources |

## Main Reproduction Commands

Show help and available workflows:

```bash
./reproduce.sh --help
```

Run the computational replication:

```bash
./reproduce.sh --comp min
```

Compile the paper:

```bash
./reproduce.sh --docs main
```

Test environments:

```bash
./reproduce.sh --envt comp_uv
./reproduce.sh --envt texlive
```

Run the full workflow:

```bash
./reproduce.sh --all
```

Fast validation path (computation only, skips LaTeX build):

```bash
./reproduce_min.sh
```

## Expected Runtime

Approximate wall-clock times (order of magnitude; depends on CPU and cache state):

- `./reproduce.sh --docs main` — often about **8–15 seconds**
- `./reproduce.sh --comp min` (same as `./reproduce_min.sh`) — often about **5–15 seconds**
- `./reproduce.sh --comp full` — similar to `--comp min` in this package
- `./reproduce.sh --all` — sum of the two steps above (typically **under 30 seconds**)

On a recent Apple‑silicon laptop with a warm `uv` environment, comp + docs each often finish in roughly **5–10 seconds**.

## System Dependencies

The Python environment is managed with UV or conda, but document reproduction also requires
system tools such as:

- `latexmk`
- `pdflatex`
- `bibtex`

See `reproduce/README.md` for more detail on environment helpers and lower-level scripts.

## Docker

A root `Dockerfile` is included so the repository can be built in a containerized environment.
From the repository root:

```bash
docker build -t low2005 .
docker run --rm -it low2005
```

Inside the container, the project lives at `/workspace`.

## Notes

- The notebook and script implement the same HARK-based workflow in interactive and script forms.
- If you only want the main paper, use `./reproduce.sh --docs main`.
