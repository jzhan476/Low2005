---
# ============================================================================
# Metadata for indexing this REMARK in the econ-ark project.
# Schema: https://github.com/econ-ark/REMARK (STANDARD.md, schema.json)
# ============================================================================

# REMARK identity
remark-version: "1.0"
remark-name: Low2005
tier: 2
github_repo_url: https://github.com/jzhan476/Low2005

# Authors of this replication software
authors:
  - family-names: Zhang
    given-names: Jiaxuan

# Original paper being replicated
references:
  - type: article
    title: Self-Insurance in a Life-Cycle Model of Labour Supply and Savings
    authors:
      - family-names: Low
        given-names: Hamish W.
    journal: Review of Economic Dynamics
    year: 2005
    volume: 8
    issue: 4
    pages: 945-975
    doi: 10.1016/j.red.2005.03.002

# Notebooks shipped in this repo (paths relative to repo root)
notebooks:
  - Code/Python/Low2005.ipynb

tags:
  - REMARK
  - Notebook
  - Reproduction
  - LifeCycle
  - LaborSupply
keywords:
  - Self-Insurance
  - Labor Supply
  - Savings
  - Incomplete Markets
  - Life-Cycle
  - Precautionary Saving
  - Heterogeneous Agents
---

# Low (2005): Self-Insurance in a Life-Cycle Model of Labour Supply and Savings

This is a Tier-2 REMARK reproducing the main quantitative comparisons of
Low, Hamish W. (2005), *Self-Insurance in a Life-Cycle Model of Labour Supply
and Savings*, **Review of Economic Dynamics**, 8(4), 945-975
([DOI: 10.1016/j.red.2005.03.002](https://doi.org/10.1016/j.red.2005.03.002)).
The replication is implemented in Python on top of [HARK](https://github.com/econ-ark/HARK).

## Authors

- **Jiaxuan Zhang** — Johns Hopkins University, Econ 606 (Spring 2026).
  GitHub: [@jzhan476](https://github.com/jzhan476).

## Abstract (Low 2005)

The paper studies how households self-insure over the life cycle when future
wages are uncertain and asset markets are incomplete. The central mechanism is
that *labor supply itself is an insurance margin*: with flexible hours,
households respond to wage risk by adjusting consumption, assets, **and** work
effort. Under uncertainty, households work more and consume less early in life;
flexible hours generate stronger precautionary asset accumulation in middle age
and life-cycle profiles for hours, consumption, and wealth that are closer to
the empirical patterns the paper aims to explain.

## What is reproduced

All quantitative results are reproduced from scratch in Python using
`HARK.ConsumptionSaving`:

| Paper object             | Implementation                                                      |
|--------------------------|---------------------------------------------------------------------|
| Flexible-hours economy   | `HARK.ConsumptionSaving.ConsLaborModel.LaborIntMargConsumerType`    |
| Fixed-hours economy      | `HARK.ConsumptionSaving.ConsIndShockModel.IndShockConsumerType`     |
| Calibration (Table 1)    | Encoded in `Code/Python/Low2005.py` (matches Low 2005, p. 956)      |
| Figures 3-7              | Five PNGs written to `Figures/` (see "Expected outputs" below)      |
| Working-life summary     | Mean hours and peak-asset age printed by `reproduce_min.sh`         |

Working life (ages 25-64) is solved with HARK's life-cycle solvers; retirement
(ages 65-84) is appended as a closed-form deterministic forward extension for
plotting only — `LaborIntMargConsumerType` requires strictly positive wages
each period, so retirement income is handled outside the HARK problem.

## Replication targets

The two quantitative targets that Low (2005) Table 1 calibrates the
flexible-hours uncertainty economy to match:

| Statistic (working life, baseline calibration) | Low (2005) | This replication | Residual (this − target) |
|------------------------------------------------|:----------:|:----------------:|:------------------------:|
| Mean hours, fraction of time worked            | 0.40       | 0.36             | −0.04 (−10.0%)           |
| Peak-assets age (full life cycle)              | ~60        | 56               | ≈ −4 yr (≈ −6.7%)        |

Both absolute residuals are small. The slight underprediction of mean hours
and the approximately four-year-earlier peak are driven by ingredients of
Low (2005) that are deliberately *not* added here: no mortality risk during
retirement, no bequest motive, no separate unemployment-style transitory
state.

### Qualitative comparisons (Low 2005, Figs. 3–7)

Each row reports the model output that the replication script prints; the
sign and order-of-magnitude pattern matches Low (2005).

| Mechanism (Low 2005 figure)                                   | Replication output (peak / early-life)        | Sign matches |
|---------------------------------------------------------------|-----------------------------------------------|:------------:|
| Uncertainty raises early-life hours (Fig. 3)                  | 0.437 (uncert) vs 0.405 (cert), age 25        | ✓            |
| Uncertainty raises peak asset accumulation (Fig. 5)           | 1.45 (uncert) vs 0.16 (cert) — ≈ 9× increase  | ✓            |
| Flexible hours reduce precautionary saving (Fig. 6)           | 1.45 (flex) vs 2.85 (fixed) — ≈ 2× ratio      | ✓            |
| Hump-shaped life-cycle consumption (Fig. 4)                   | Yes; peak around mid-50s                      | ✓            |
| Asset profile peaks before retirement (Fig. 7)                | Peak at age 56 (Low: ~60)                     | ✓            |

### Implied Frisch elasticity

The Cobb–Douglas-in-CRRA utility delivers a closed-form Frisch labor
supply elasticity at the calibration point of

$$\mathrm{Frisch} \;=\; \frac{1-h}{h}\cdot\frac{1+\eta(\gamma-1)}{\gamma}
\;\approx\; 1.0 \quad\text{at}\ h=0.40,\ \eta=0.4,\ \gamma=2.2.$$

This is above standard cross-sectional micro estimates (Chetty 2012
preferred Frisch ≈ 0.5) and broadly consistent with lifecycle-corrected
estimates (Domeij & Florén 2006; Imai & Keane 2004), as discussed in
Section 3 of `Low2005.pdf`. Holding the same targeted hours share
$h = 0.40$, the closed-form Frisch formula there implies Frisch ≈ **0.93**
at $\eta = 0.3$, **1.01** at $\eta = 0.4$ (baseline), and **1.09** at
$\eta = 0.5$ (PDF Table "Sensitivity … to $\eta$"; full counterfactuals
would re-calibrate hours).


## Out of scope

- $\delta$ is **not** re-calibrated separately for the flexible- and fixed-hours
  economies (Low's headline rate $\delta = 0.032$ is reused for both); the
  replication targets the *shape* of the asset and consumption profiles.
- Mortality risk in retirement, bequest motives, and unemployment shocks are
  not modeled.
- The CES-aggregator and additive-separability robustness exercises (Low 2005,
  Section on alternative preferences) are discussed in the paper but not
  re-run.

## How to reproduce

```bash
git clone https://github.com/jzhan476/Low2005.git
cd Low2005
git checkout v1.1.0
uv sync --frozen --no-dev      # or: conda env create -f binder/environment.yml
./reproduce.sh --comp min      # computational replication only (~5-15 s)
./reproduce.sh --docs main     # build Low2005.pdf only          (~8-15 s)
./reproduce.sh --all           # full pipeline                    (<30 s)
```

Binder: `binder/environment.yml` (Python 3.11) plus `binder/postBuild`
materializes pinned dependencies via `uv sync --frozen`. Local installs
support Python 3.9-3.12 (`requires-python = ">=3.9,<3.13"` in `pyproject.toml`).

## Expected outputs

After `./reproduce.sh --comp min`, five PNGs are written to `Figures/`:

| File                              | Mirrors Low (2005) figure                       |
|-----------------------------------|-------------------------------------------------|
| `lifecycle_profiles.png`          | Figure 7 — life-cycle hours/consumption/assets  |
| `hours_cert_vs_uncert.png`        | Figure 3 — hours under certainty vs. uncertainty|
| `consumption_cert_vs_uncert.png`  | Figure 4 — mean consumption                     |
| `assets_cert_vs_uncert.png`       | Figure 5 — mean asset holdings                  |
| `flexible_vs_fixed.png`           | Figure 6 — flex vs. fixed hours under risk      |

`./reproduce.sh --docs main` then produces `Low2005.pdf`.

## Computational requirements

- Python 3.9-3.12 (tested on 3.11).
- No GPU, no compiled extensions beyond what HARK and SciPy ship with.
- Wall-clock on a recent Apple-silicon laptop with a warm `uv` environment:
  comp ~5-10 s, docs ~5-10 s, all <30 s.
- LaTeX (`latexmk`, `pdflatex`, `bibtex`) is required only for the docs step.

## Tier-2 compliance checklist

- [x] Tagged release **v1.1.0** on `main` plus a GitHub Release
- [x] `reproduce.sh` (full) and `reproduce_min.sh` (fast)
- [x] Root `Dockerfile`
- [x] `LICENSE` (Apache-2.0)
- [x] `binder/environment.yml` + `binder/postBuild`, pinned via `uv.lock`
- [x] `README.md` >= 100 non-empty lines
- [x] `REMARK.md` with `tier: 2` frontmatter and notebook list (this file)
- [x] Valid `CITATION.cff` (CFF 1.2.0)

## How to cite

To cite this **replication software**, use the metadata in
[`CITATION.cff`](./CITATION.cff) (currently v1.1.0). The underlying article
remains:

> Low, Hamish W. (2005). Self-Insurance in a Life-Cycle Model of Labour Supply
> and Savings. *Review of Economic Dynamics*, 8(4), 945-975.
> [DOI: 10.1016/j.red.2005.03.002](https://doi.org/10.1016/j.red.2005.03.002).

## License & contact

Apache-2.0 (see [`LICENSE`](./LICENSE)). Issues and pull requests welcome at
[github.com/jzhan476/Low2005/issues](https://github.com/jzhan476/Low2005/issues).
