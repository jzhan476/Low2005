# Low2005 Replication

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

The main entrypoint for reproduction is `reproduce.sh`.

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

For REMARK/Binder compatibility, the canonical conda environment file is:

```bash
binder/environment.yml
```

Create the environment with conda:

```bash
conda env create -f binder/environment.yml
conda activate low2005
```

You can then run the same reproduction commands:

```bash
./reproduce.sh --comp min
./reproduce.sh --docs main
```

## Expected Outputs

After `./reproduce.sh --comp min`, the main expected outputs are PNG figures under `Figures/`.

After `./reproduce.sh --docs main`, the main expected output is:

```bash
Low2005.pdf
```

## Repository Layout

| Path | Role |
|------|------|
| `README.md` | Top-level overview and reproduction instructions |
| `REMARK.md` | REMARK metadata and short replication summary |
| `CITATION.cff` | Citation metadata for the repository |
| `binder/environment.yml` | REMARK/Binder-compatible conda environment file |
| `Code/Python/` | `Low2005.py` and `Low2005.ipynb` |
| `Figures/` | Generated computational output figures |
| `Low2005.tex` | Main LaTeX paper |
| `reproduce.sh` | Main orchestration script |
| `reproduce/` | Supporting environment, build, and utility scripts |
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

- The repository includes a Zenodo DOI in `CITATION.cff`, but the baseline reproduction workflow
  is centered on `reproduce.sh`, the environment files, and the generated outputs.
- The notebook and script implement the same HARK-based workflow in interactive and script forms.
- If you only want the main paper, use `./reproduce.sh --docs main`.
