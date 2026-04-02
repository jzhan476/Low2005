# Low2005 Code (HARK)

Numerical replication of Low (2005) using **HARK** (`LaborIntMargConsumerType`, `IndShockConsumerType`).

| File | Role |
|------|------|
| `Python/Low2005.py` | Script form; writes PNGs under `Figures/`. |
| `Python/Low2005.ipynb` | Jupyter version of the same workflow. |

## Requirements

Python environment with **HARK** (via `econ-ark` in `pyproject.toml`).

## Run

From repository root:

```bash
./reproduce_low2005.sh run
# or
./reproduce.sh --comp min
```

Notebook:

```bash
./reproduce_low2005.sh notebook
```
