# Low2005 Replication

Replication package for:

Hamish W. Low (2005), *Self-Insurance in a Life-Cycle Model of Labour Supply and Savings*,
*Review of Economic Dynamics*, 8(4), 945-975.  
DOI: <https://doi.org/10.1016/j.red.2005.03.002>

Core computational engine: **HARK** (`HARK.ConsumptionSaving.ConsLaborModel`, `ConsIndShockModel`).

## Layout

| Path | Role |
|------|------|
| `REMARK.md` | REMARK metadata + replication checklist |
| `Code/Python/` | `Low2005.py` / `Low2005.ipynb` — HARK lifecycle replication, figures → `Figures/` |
| `@local/` | Project LaTeX config (`local.sty`, `metadata.ltx`, …) |
| `@resources/` | Bundled `econark` class (`texlive/texmf-local/`), `latexmk/`, `econ-ark/` badge assets |
| `Low2005.tex` | Single-file LaTeX paper |
| `reproduce.sh` | Env + document build (`./reproduce.sh --docs main`) and `--comp` for Python |

## Quick run (Python replication)

```bash
./reproduce.sh --comp min
```

Or from `Code/Python/`: `python Low2005.py`

## Build the PDF (LaTeX)

```bash
./reproduce.sh --docs main
```

Output: `Low2005.pdf`. For environment setup see `reproduce/README.md`.
