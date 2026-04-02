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
| `@resources/econ-ark/` | Econ-ARK badge (`PoweredByEconARK.pdf`, …) for the title page |
| `Low2005.tex` | Single-file LaTeX paper |
| `reproduce.sh` | Env + document build (`./reproduce.sh --docs main`) and `--comp` for Python |
| `reproduce_low2005.sh` | Shortcut: `run` or `notebook` for `Code/Python/Low2005.*` |

## Quick run (Python replication)

```bash
./reproduce_low2005.sh run
# or
./reproduce.sh --comp min
```

## Build the PDF (LaTeX)

```bash
./reproduce.sh --docs main
```

Output: `Low2005.pdf`. For environment setup see `reproduce/README.md`.
