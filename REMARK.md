---
remark-version: 1.0
tier: 2
github_repo_url: https://github.com/jzhan476/Low2005
remark-name: Low2005
notebooks:
  - Code/Python/Low2005.ipynb
tags:
  - REMARK
  - Reproduction
  - LifeCycle
  - LabourSupply
keywords:
  - Self-Insurance
  - Labour Supply
  - Savings
  - Incomplete Markets
  - Life-Cycle
---
# Low (2005): Self-Insurance in a Life-Cycle Model of Labour Supply and Savings

This repository replicates:

Low, Hamish W. (2005). *Self-Insurance in a Life-Cycle Model of Labour Supply and Savings*,
**Review of Economic Dynamics**, 8(4), 945-975.
DOI: <https://doi.org/10.1016/j.red.2005.03.002>

## Version and citation

- Current git release: **v1.1.0** (see [GitHub Releases](https://github.com/jzhan476/Low2005/releases)).
- To cite this replication **software**, use metadata in `CITATION.cff` (v1.1.0). The
  underlying article remains Low (2005), *Review of Economic Dynamics*.

## Replication status

1. HARK lifecycle replication: `Code/Python/Low2005.py` / `Low2005.ipynb` (`LaborIntMargConsumerType`, `IndShockConsumerType`).
2. Figures written to `Figures/`; build the PDF with `./reproduce.sh --docs main`.
3. Environments: `pyproject.toml` / `uv.lock` with `requires-python = ">=3.9,<3.13"`; Binder bootstraps **Python 3.11** via `binder/environment.yml` + `binder/postBuild` (`uv sync --frozen`).
