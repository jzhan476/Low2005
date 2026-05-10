# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Replication of Low (2005) using HARK
#
# **Paper:** Hamish W. Low, "Self-Insurance in a Life-Cycle Model of Labour
# Supply and Savings," *Review of Economic Dynamics*, 8(4), 945-975, 2005.
#
# This notebook replicates the key results: life-cycle profiles of consumption,
# labor supply, and assets with and without uncertainty, and with flexible vs
# fixed hours. These correspond to Figures 3-7 of the paper.
#
# **Model scope:** Working life (ages 25-64, 40 periods) followed by a
# deterministic retirement phase (ages 65-84, 20 periods) at a 0.55
# replacement rate. HARK's `LaborIntMargConsumerType` requires a positive
# wage every period, so retirement is computed as a closed-form
# certainty-equivalent extension of the working-life simulation: each
# agent enters retirement with their simulated terminal assets and a
# constant pension, and consumes optimally subject to the Euler equation
# with no bequests.

# %% Imports and calibration
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from HARK.ConsumptionSaving.ConsLaborModel import LaborIntMargConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

# Resolve the Figures directory robustly so the script works whether it is
# invoked from `Code/Python/` (the wrapper's CWD) or as a notebook from
# the repo root.  Falls back to ``cwd`` when ``__file__`` is undefined
# (e.g. inside Jupyter).
try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()
for _candidate in (_SCRIPT_DIR.parents[1] / "Figures",
                   _SCRIPT_DIR / "Figures",
                   Path.cwd() / "Figures"):
    if _candidate.parent.exists():
        FIG_DIR = _candidate
        break
else:
    FIG_DIR = _SCRIPT_DIR.parents[1] / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# === Low (2005) Published Calibration (Table 1, p.959) ===
gamma_low = 2.2          # CRRA over composite good
eta_low = 0.4            # consumption share in Cobb-Douglas utility
beta_low = 1.0 / 1.032   # discount factor (1 + delta = 1.032)
r_low = 0.016            # real interest rate
replacement_rate = 0.55  # social-security replacement rate (Low 2005, Table 1)

# Wage profile: log w(t) = alpha1*t + alpha2*t^2
alpha1_wage = 0.0561
alpha2_wage = -0.000599

# Shock variances (Low 2005, Table 1).  In the paper's notation,
# `\nu` is the transitory shock added to log w_t directly and
# `\varepsilon` is the innovation to the permanent component ln w^P_t,
# so sigma^2_eps is the *permanent* variance (0.031) and sigma^2_nu is
# the *transitory* variance (0.030).  Earlier versions of this script
# had these labels swapped; the assignments below now match the paper.
sigma_eps_sq = 0.031      # permanent shock variance (innovation to ln w^P_t)
sigma_nu_sq = 0.030       # transitory shock variance (added to ln w_t directly)
sigma_perm = np.sqrt(sigma_eps_sq)   # ~0.176
sigma_tran = np.sqrt(sigma_nu_sq)    # ~0.173

# Life-cycle parameters
T_work = 40              # working years (age 25-64)
T_ret = 20               # retirement years (age 65-84)
T_total = T_work + T_ret
start_age = 25
fixed_hours = 0.40       # 2080 hours / 5200 available hours in Low (2005)

# Numerical settings
simulation_seed = 2005
agent_count = 10000
shock_count = 7
asset_grid_count = 48
asset_grid_max = 50.0

# === Low (2005) Working-Life Replication Targets (Table 1, p.956) ===
# Single source of truth -- consumed by the printed summary, the JSON
# artifact, and the LaTeX tables that the paper inputs.
#
# Headline values (the uncertainty + flexible-hours scenario reported
# in Low (2005)'s abstract / discussion):
low_2005_targets = {
    "mean_hours_working": 0.40,
    "peak_assets_age": 60,
    "median_assets_to_income_working": 1.84,
    "aggregate_assets_to_income_working": 2.14,
}
#
# Per-scenario calibration targets transcribed verbatim from
# Low (2005), Table 1 ("Calibrated discount rates", p.956).  In the
# paper's notation 1 + delta is reported; we store the calibrated
# delta (so DiscFac = 1 / (1 + delta)).  These are the *paper-reported*
# numbers; this replication's calibrated counterparts and the
# resulting aggregate A/Y are computed below in
# ``calibrated_discfacs`` and ``calibrated_per_scenario`` and
# compared against these targets in the LaTeX summary table.
low_2005_per_scenario = {
    "uncert_flex":  {"delta": 0.032, "median_AY": 1.84, "aggregate_AY": 2.14,
                     "label": "Uncertainty, flexible hours"},
    "cert_flex":    {"delta": 0.009, "median_AY": 1.84, "aggregate_AY": 2.22,
                     "label": "Certainty, flexible hours"},
    "uncert_fixed": {"delta": 0.028, "median_AY": 1.84, "aggregate_AY": 2.05,
                     "label": "Uncertainty, fixed hours"},
}

# === HARK Parameter Mapping ===
# Low: u = (c^eta * z^(1-eta))^(1-gamma) / (1-gamma), z = leisure
# HARK: u = (c * z^alpha)^(1-rho) / (1-rho)
# => alpha = (1-eta)/eta, rho = 1 - eta*(1-gamma)
CRRA_hark = 1.0 - eta_low * (1.0 - gamma_low)
alpha_hark = (1.0 - eta_low) / eta_low
DiscFac = beta_low
Rfree = 1.0 + r_low

calibration_rows = [
    (r"Real interest rate $r$", f"{r_low:.3f}"),
    (r"Relative risk aversion $\gamma$", f"{gamma_low:.1f}"),
    (r"Consumption share $\eta$", f"{eta_low:.1f}"),
    (r"Wage-growth coefficient $\alpha_1$", f"{alpha1_wage:.4f}"),
    (r"Wage-growth coefficient $\alpha_2$", f"{alpha2_wage:.6f}"),
    (r"Permanent-shock variance $\sigma^2_{\varepsilon}$", f"{sigma_eps_sq:.3f}"),
    (r"Transitory-shock variance $\sigma^2_{\nu}$", f"{sigma_nu_sq:.3f}"),
    ("Social-security replacement rate", f"{replacement_rate:.2f}"),
]

print(f"Low (2005): gamma={gamma_low}, eta={eta_low}")
print(f"HARK:       CRRA={CRRA_hark:.4f}, LbrCost alpha={alpha_hark:.4f}")
print(f"Discount factor: {DiscFac:.6f}")
print(f"Gross interest rate: {Rfree:.4f}")
print(f"Permanent shock std: {sigma_perm:.4f}  (variance {sigma_eps_sq})")
print(f"Transitory shock std: {sigma_tran:.4f}  (variance {sigma_nu_sq})")
print(f"Retirement: {T_ret} periods at {replacement_rate:.2f} replacement rate")
print(f"Figures directory: {FIG_DIR}")

# %% Deterministic wage growth factors
PermGroFac_list = [
    float(np.exp(alpha1_wage + alpha2_wage * (2 * t + 1)))
    for t in range(T_work)
]

ages_work = np.arange(start_age, start_age + T_work)
ages = np.arange(start_age, start_age + T_total)  # full life cycle (25-84)
cum_log_wage = np.cumsum([0.0] + [np.log(g) for g in PermGroFac_list[:-1]])
wage_profile = np.exp(cum_log_wage)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ages_work, wage_profile, "b-", linewidth=2)
ax.set_xlabel("Age")
ax.set_ylabel("Relative Wage Level")
ax.set_title("Deterministic Wage Profile (Low 2005)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Retirement: backward-induction terminal continuation
#
# Without modeling retirement explicitly, the working-life agent has no
# reason to save for the future, and median A/Y comes in at roughly
# 40% of Low (2005)'s reported value.  To inject a retirement-saving
# motive into the backward induction, we solve a deterministic
# retirement subproblem (no shocks, constant pension at the calibrated
# 0.55 replacement rate, no bequest) using `IndShockConsumerType`, and
# then inject its start-of-retirement solution as the terminal
# solution for the working-life problem.  Working-life agents then
# carry assets into retirement deliberately, in line with Low (2005).
#
# To get the cross-period normalization right we set
# `PermGroFac[T_work-1] = replacement_rate` (deterministic income
# drop at the retirement transition) and zero the shock variances at
# the same period.  Working-life dynamics for periods 0..T_work-2 are
# unchanged.  We also keep `retirement_paths` as a closed-form
# deterministic forward extension for plotting the retirement portion
# of the lifecycle, since `LaborIntMargConsumerType` only simulates
# the working-life cycle natively.

# %%
def retirement_paths(history, *, CRRA, T_ret, replacement_rate,
                     Rfree, DiscFac):
    """Closed-form deterministic retirement extension with a_t >= 0.

    Retirement is solved *in isolation* from working life: each agent
    enters retirement with their simulated terminal assets and a
    constant pension equal to ``replacement_rate`` times their final
    permanent income, and chooses consumption to satisfy the Euler
    equation subject to no-borrowing (``a_t >= 0``) and no bequests
    (``a_{T_ret} = 0``).  Because the working-life HARK agent does not
    anticipate retirement, there is a visible discontinuity at age 65
    in the life-cycle figures -- this reflects the resolution of
    income uncertainty, not an error in the code.  A fully
    lifecycle-consistent solution would require a HARK agent that
    supports both a labour choice and a zero-wage retirement stage in
    the same backwards-induction problem, which is left as a future
    extension.

    Parameters
    ----------
    history : dict
        HARK simulation history with ``aNrm`` and ``pLvl`` arrays of
        shape ``(T_work, AgentCount)``.
    CRRA : float
        Coefficient of relative risk aversion used during retirement.
    T_ret : int
        Number of retirement periods.
    replacement_rate : float
        Pension as a fraction of last working-life permanent income.
    Rfree, DiscFac : float
        Gross interest rate and discount factor.

    Returns
    -------
    mean_c_ret, mean_a_ret : ndarray of shape ``(T_ret,)``
        Cross-sectional mean consumption (level) and mean assets
        entering each retirement period (i.e. ``a_t`` *before*
        consumption at period ``t``).
    """
    a_terminal = np.asarray(history["aNrm"][-1] * history["pLvl"][-1],
                            dtype=float)
    p_terminal = np.asarray(history["pLvl"][-1], dtype=float)
    pension = replacement_rate * p_terminal
    R = float(Rfree)

    # Euler-equation consumption growth factor.
    g_c = (DiscFac * R) ** (1.0 / CRRA)

    # Annuity factors: PV of $1 received in each retirement period.
    if abs(R - 1.0) < 1e-12:
        ann_y = float(T_ret)
    else:
        ann_y = (1.0 - R ** (-T_ret)) / (1.0 - 1.0 / R)
    if abs(g_c / R - 1.0) < 1e-12:
        ann_c = float(T_ret)
    else:
        ann_c = (1.0 - (g_c / R) ** T_ret) / (1.0 - g_c / R)

    c0 = (a_terminal + pension * ann_y) / ann_c

    AgentCount = a_terminal.shape[0]
    c_ret = np.empty((T_ret, AgentCount))
    a_ret = np.empty((T_ret, AgentCount))
    a_t = a_terminal.copy()
    hand_to_mouth = np.zeros(AgentCount, dtype=bool)
    for t in range(T_ret):
        c_unconstrained = c0 * (g_c ** t)
        cash = a_t + pension
        # Agents already liquidity-constrained eat pension each period.
        c_t = np.where(hand_to_mouth, pension, c_unconstrained)
        # If the unconstrained plan would push assets below zero this
        # period, consume all cash-on-hand and switch to hand-to-mouth.
        would_go_negative = (~hand_to_mouth) & (cash - c_t < 0.0)
        if np.any(would_go_negative):
            c_t = np.where(would_go_negative, cash, c_t)
            hand_to_mouth |= would_go_negative
        c_ret[t] = c_t
        a_ret[t] = a_t
        a_t = R * (cash - c_t)
        hand_to_mouth |= (a_t <= 1e-10)

    return c_ret.mean(axis=1), a_ret.mean(axis=1)

# %% Retirement-continuation terminal solution
from HARK.ConsumptionSaving.ConsLaborModel import ConsumerLaborSolution
from HARK.interpolation import ConstantFunction


def _solve_retirement_continuation(*, T_ret, CRRA, DiscFac, Rfree,
                                   asset_grid_max, asset_grid_count):
    """Solve a deterministic retirement subproblem.

    Models retirement as an `IndShockConsumerType` with no income
    shocks, no growth, constant pension normalized to 1 in
    retirement-pLvl units (so that real pension equals
    ``replacement_rate * pLvl_{T_work-1}`` once we set
    ``PermGroFac[T_work-1] = replacement_rate`` upstream), no
    unemployment, no bequest, and no borrowing.  Returns the IndShock
    solution at the start of the first retirement period, which we
    splice in as the terminal solution for the working-life problem.
    """
    ret_agent = IndShockConsumerType(
        cycles=1,
        T_cycle=T_ret,
        CRRA=CRRA,
        DiscFac=DiscFac,
        Rfree=[Rfree] * T_ret,
        LivPrb=[1.0] * T_ret,
        PermGroFac=[1.0] * T_ret,
        PermShkStd=[0.0] * T_ret,
        TranShkStd=[0.0] * T_ret,
        PermShkCount=1,
        TranShkCount=1,
        UnempPrb=0.0,
        UnempPrbRet=0.0,
        IncUnemp=0.0,
        IncUnempRet=0.0,
        T_retire=0,
        BoroCnstArt=0.0,
        aXtraMin=0.001,
        aXtraMax=asset_grid_max,
        aXtraCount=asset_grid_count,
        aXtraNestFac=3,
        AgentCount=1,
        T_age=T_ret + 1,
        T_sim=1,
        aNrmInitMean=0.0,
        aNrmInitStd=0.0,
        pLvlInitMean=0.0,
        pLvlInitStd=0.0,
        PermGroFacAgg=1.0,
    )
    ret_agent.solve()
    return ret_agent.solution[0]


class _LaborTerminalVPfunc:
    """`vPfunc(bNrm, TranShk)` wrapper for the retirement boundary.

    The retirement subproblem is deterministic: cash-on-hand at the
    start of retirement equals ``bNrm + 1`` (pension normalized to 1
    in retirement-pLvl units).  Any TranShk values that the
    LaborIntMarg solver passes in at the boundary have no economic
    role, so we ignore them and forward ``bNrm + 1`` to the
    retirement-stage marginal value function.
    """

    def __init__(self, vPfunc_ret):
        self.vPfunc_ret = vPfunc_ret

    def __call__(self, bNrm, TranShk):
        bNrm_arr = np.asarray(bNrm, dtype=float)
        return self.vPfunc_ret(bNrm_arr + 1.0)


class _LaborTerminalBNrmMin:
    """Minimum bNrm at the retirement boundary, constant across TranShk."""

    def __init__(self, value):
        self.value = float(value)

    def __call__(self, TranShk):
        return np.full_like(np.asarray(TranShk, dtype=float), self.value)


def _make_labor_terminal_from_retirement(ret_solution):
    """Wrap an IndShock retirement-start solution as a `ConsumerLaborSolution`."""
    return ConsumerLaborSolution(
        cFunc=ConstantFunction(0.0),
        LbrFunc=ConstantFunction(0.0),
        vPfunc=_LaborTerminalVPfunc(ret_solution.vPfunc),
        bNrmMin=_LaborTerminalBNrmMin(0.0),
    )


class LaborIntMargWithRetirement(LaborIntMargConsumerType):
    """`LaborIntMargConsumerType` with an injected retirement terminal.

    HARK's `IndShockConsumerType.pre_solve` (which the labor model
    inherits) calls `update_solution_terminal` and overwrites
    ``solution_terminal`` with a default eat-everything terminal each
    time `solve()` runs.  We override that hook so the retirement
    continuation persists across `solve()` invocations.
    """

    def __init__(self, *args, retirement_terminal, **kwargs):
        self._retirement_terminal = retirement_terminal
        super().__init__(*args, **kwargs)

    def update_solution_terminal(self):
        self.solution_terminal = self._retirement_terminal


class IndShockWithRetirement(IndShockConsumerType):
    """`IndShockConsumerType` with an injected retirement terminal."""

    def __init__(self, *args, retirement_terminal, **kwargs):
        self._retirement_terminal = retirement_terminal
        super().__init__(*args, **kwargs)

    def update_solution_terminal(self):
        self.solution_terminal = self._retirement_terminal


# Last-period growth factor becomes the deterministic income drop
# into retirement.  HARK uses ``PermGroFac[t]`` as the growth from
# period ``t`` to period ``t+1`` inside the solver, and never
# evaluates ``PermGroFac[T_cycle-1]`` during simulation when
# ``T_sim == T_cycle`` (see ``IndShockConsumerType.get_shocks`` time
# shift), so this swap is felt only by the backward induction.  The
# wage profile across ages 25..63 is therefore unchanged.
PermGroFac_with_retirement = list(PermGroFac_list[:-1]) + [replacement_rate]


# %% [markdown]
# ## Per-scenario calibration of the discount factor
#
# Low (2005), Table 1, calibrates the discount rate $\delta$ separately
# for each scenario so that the median wealth-to-income ratio over
# working life equals 1.84 (the PSID 1995 target).  The headline value
# $\delta = 0.032$ shown in the published Table 1 corresponds to the
# uncertainty + flexible-hours scenario; the certainty + flexible case
# uses $\delta = 0.009$ and the fixed-hours scenarios use values close
# to those.  To follow the paper's calibration *method* rather than just
# its reported number, we re-calibrate ``DiscFac`` separately for each
# of the three scenarios run by this script via bisection over the same
# target.  The replacement-rate, shock variances, wage profile, CRRA,
# and interest rate are all kept at the values reported in Low (2005).

# %% Per-scenario calibration helpers
def _build_flex_agent(*, DiscFac, perm_shk_std, tran_shk_std, base_dict):
    """Construct a flexible-hours agent with retirement continuation.

    Solves the deterministic retirement subproblem at the given
    ``DiscFac`` and splices its start-of-retirement marginal value
    function in as the working-life terminal solution.
    """
    ret_sol = _solve_retirement_continuation(
        T_ret=T_ret, CRRA=CRRA_hark, DiscFac=DiscFac, Rfree=Rfree,
        asset_grid_max=asset_grid_max, asset_grid_count=asset_grid_count,
    )
    labor_term = _make_labor_terminal_from_retirement(ret_sol)
    d = base_dict.copy()
    d["DiscFac"] = DiscFac
    d["PermShkStd"] = [perm_shk_std] * T_work
    d["TranShkStd"] = [tran_shk_std] * T_work
    return LaborIntMargWithRetirement(retirement_terminal=labor_term, **d)


def _build_fixed_agent(*, DiscFac, fixed_dict_template):
    """Construct a fixed-hours IndShock agent with retirement continuation."""
    ret_sol = _solve_retirement_continuation(
        T_ret=T_ret, CRRA=CRRA_hark, DiscFac=DiscFac, Rfree=Rfree,
        asset_grid_max=asset_grid_max, asset_grid_count=asset_grid_count,
    )
    d = fixed_dict_template.copy()
    d["DiscFac"] = DiscFac
    return IndShockWithRetirement(retirement_terminal=ret_sol, **d)


def _solve_and_sim_flex(DiscFac, perm_shk_std, tran_shk_std, base_dict):
    a = _build_flex_agent(
        DiscFac=DiscFac, perm_shk_std=perm_shk_std,
        tran_shk_std=tran_shk_std, base_dict=base_dict,
    )
    a.solve()
    a.track_vars = ["cNrm", "Lbr", "aNrm", "pLvl", "TranShk"]
    a.initialize_sim()
    a.simulate()
    return a


def _solve_and_sim_fixed(DiscFac, fixed_dict_template):
    a = _build_fixed_agent(DiscFac=DiscFac, fixed_dict_template=fixed_dict_template)
    a.solve()
    a.track_vars = ["cNrm", "aNrm", "pLvl", "TranShk"]
    a.initialize_sim()
    a.simulate()
    return a


def _flex_median_AY(agent):
    """Cross-sectional median A/Y for the flex-hours model.

    Uses the same income definition as the post-hoc summary
    (``Y = pLvl * Lbr * TranShk``), so that the median targeted during
    bisection matches the median reported in the final summary.
    """
    pLvl = np.asarray(agent.history["pLvl"], dtype=float)
    Lbr  = np.asarray(agent.history["Lbr"],  dtype=float)
    Tran = np.asarray(agent.history["TranShk"], dtype=float)
    aNrm = np.asarray(agent.history["aNrm"], dtype=float)
    A = aNrm * pLvl
    Y = pLvl * Lbr * Tran
    pos = Y > 1e-8
    return float(np.nanmedian(np.where(pos, A / np.where(pos, Y, 1.0), np.nan)))


def _fixed_median_AY(agent):
    """Cross-sectional median A/Y for the fixed-hours model."""
    pLvl = np.asarray(agent.history["pLvl"], dtype=float)
    Tran = np.asarray(agent.history["TranShk"], dtype=float)
    aNrm = np.asarray(agent.history["aNrm"], dtype=float)
    A = aNrm * pLvl
    Y = pLvl * Tran
    pos = Y > 1e-8
    return float(np.nanmedian(np.where(pos, A / np.where(pos, Y, 1.0), np.nan)))


def _calibrate_discfac(eval_func, *, target=1.84, lo=0.90, hi=1.05,
                       tol=0.01, max_iter=22, label=""):
    """Bisect over ``DiscFac`` until ``eval_func(DiscFac)`` is within
    ``tol`` of ``target``.

    ``eval_func(DiscFac)`` should return the median A/Y produced by
    solving and simulating one scenario.  ``A/Y`` is monotone increasing
    in ``DiscFac`` (more patient = save more), so a standard bisection
    converges in O(log(1/tol)) iterations -- at the default settings
    well within the iteration cap.

    The default bounds intentionally allow ``DiscFac`` slightly above
    one (negative implicit delta) because the certainty + flexible
    hours scenario needs an unusually patient agent to reach the PSID
    median A/Y target absent precautionary saving incentives.

    Returns
    -------
    (DiscFac, median_AY, converged) : tuple
        ``converged`` is ``True`` when the search returns within
        ``tol`` of ``target``; ``False`` indicates an upper- or
        lower-bound bind.
    """
    last_mid = lo
    last_m = float("nan")
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        last_mid = mid
        m = eval_func(mid)
        last_m = m
        print(f"    [{label}] iter {i + 1:2d}: DiscFac={mid:.4f} -> medA/Y={m:.3f}")
        if abs(m - target) < tol:
            return mid, m, True
        if hi - lo < 0.0005:
            return mid, m, False
        if m < target:
            lo = mid
        else:
            hi = mid
    return last_mid, last_m, False


# %% [markdown]
# ## Model 1-3 setup: shared agent parameter dictionaries
#
# These are *templates* -- the discount factor is left at the paper's
# headline value here and is overridden per-scenario after the
# calibration step below.

# %% Common agent parameters
base_dict = {
    "cycles": 1,
    "T_cycle": T_work,
    "CRRA": CRRA_hark,
    "DiscFac": DiscFac,
    "Rfree": [Rfree] * T_work,
    "LivPrb": [1.0] * T_work,
    "PermGroFac": PermGroFac_with_retirement,
    "WageRte": [1.0] * T_work,
    "PermShkCount": shock_count,
    "TranShkCount": shock_count,
    "UnempPrb": 0.0,
    "UnempPrbRet": 0.0,
    "T_retire": 0,
    "IncUnemp": 0.0,
    "IncUnempRet": 0.0,
    # Use the natural borrowing limit (most-negative `a` for which the
    # consumer can repay in the worst-case shock realization).  The same
    # choice is used for the fixed-hours benchmark below so that the
    # flexible-vs-fixed comparison isolates labour-supply flexibility
    # rather than mixing it with a tighter ad-hoc constraint.
    "BoroCnstArt": None,
    "LbrCostCoeffs": [float(np.log(alpha_hark))],
    "aXtraMin": 0.001,
    "aXtraMax": asset_grid_max,
    "aXtraCount": asset_grid_count,
    "aXtraNestFac": 3,
    "aXtraExtra": None,
    "vFuncBool": False,
    "CubicBool": False,
    "AgentCount": agent_count,
    "T_age": T_work + 1,
    "T_sim": T_work,
    "aNrmInitMean": -10.0,
    "aNrmInitStd": 0.0,
    "pLvlInitMean": 0.0,
    "pLvlInitStd": 0.0,
    "PermGroFacAgg": 1.0,
    "NewbornTransShk": False,
    "PerfMITShk": False,
    "neutral_measure": False,
    "seed": simulation_seed,
}

# Fixed-hours template.  In Low (2005) the fixed-hours counterfactual
# keeps the same Cobb-Douglas utility u(c, l) = (c^eta * l^(1-eta))^
# (1-gamma)/(1-gamma) and only fixes labour exogenously.  With l fixed,
# that utility collapses to A * c^(eta*(1-gamma))/(1-gamma) for a
# constant A, so the effective CRRA over consumption is
# 1 - eta*(1-gamma) = CRRA_hark = 1.48 -- the *same* curvature used for
# consumption in the flexible-hours model.  Earlier versions of this
# script used CRRA = gamma_low = 2.2 here, which made the fixed agent
# artificially more risk-averse than the flexible agent and reversed
# Low's qualitative finding that flexible hours raise middle-age asset
# accumulation.
fixed_dict_template = {
    "cycles": 1,
    "T_cycle": T_work,
    "CRRA": CRRA_hark,
    "DiscFac": DiscFac,
    "Rfree": [Rfree] * T_work,
    "LivPrb": [1.0] * T_work,
    "PermGroFac": PermGroFac_with_retirement,
    "PermShkStd": [sigma_perm] * T_work,
    "TranShkStd": [sigma_tran] * T_work,
    "PermShkCount": shock_count,
    "TranShkCount": shock_count,
    "UnempPrb": 0.0,
    "UnempPrbRet": 0.0,
    "T_retire": 0,
    "IncUnemp": 0.0,
    "IncUnempRet": 0.0,
    # Match the borrowing-constraint specification used by the flexible
    # model so that `flexible_vs_fixed.png` reflects only the
    # labour-supply margin.
    "BoroCnstArt": None,
    "aXtraMin": 0.001,
    "aXtraMax": asset_grid_max,
    "aXtraCount": asset_grid_count,
    "aXtraNestFac": 3,
    "AgentCount": agent_count,
    "T_age": T_work + 1,
    "T_sim": T_work,
    "aNrmInitMean": -10.0,
    "aNrmInitStd": 0.0,
    # Initial permanent income scaled by fixed hours so that the
    # fixed-hours benchmark and the flexible-hours model share the same
    # level units for consumption and assets.
    "pLvlInitMean": float(np.log(fixed_hours)),
    "pLvlInitStd": 0.0,
    "PermGroFacAgg": 1.0,
    "seed": simulation_seed,
}

# %% [markdown]
# ## Calibrate ``DiscFac`` per scenario (Low 2005, Table 1 method)
#
# Low (2005) calibrates the discount rate $\delta$ separately for each
# scenario so that the median wealth-to-income ratio in working life
# matches the PSID 1995 target of 1.84.  We follow the same procedure
# via bisection: each iteration solves and simulates the full
# working-life + retirement model and adjusts $\delta$ until median A/Y
# is within the tolerance of the target.

# %%
print("Calibrating DiscFac per scenario to median A/Y target = "
      f"{low_2005_targets['median_assets_to_income_working']:.2f}...")

_AY_target = low_2005_targets["median_assets_to_income_working"]


def _report(label, df, m, ok):
    flag = "" if ok else "  [bisection bound binds; reporting closest feasible]"
    print(f"  {label:<13}-> DiscFac={df:.4f} "
          f"(delta={1.0 / df - 1.0:+.4f}), "
          f"medA/Y={m:.3f}{flag}")


DiscFac_uncert_flex, _med_uncert_flex, _ok_uncert_flex = _calibrate_discfac(
    lambda df: _flex_median_AY(
        _solve_and_sim_flex(df, sigma_perm, sigma_tran, base_dict)
    ),
    target=_AY_target, label="flex+uncert",
)
_report("flex+uncert", DiscFac_uncert_flex, _med_uncert_flex, _ok_uncert_flex)

DiscFac_cert_flex, _med_cert_flex, _ok_cert_flex = _calibrate_discfac(
    lambda df: _flex_median_AY(
        _solve_and_sim_flex(df, 0.001, 0.001, base_dict)
    ),
    target=_AY_target, label="flex+cert",
)
_report("flex+cert", DiscFac_cert_flex, _med_cert_flex, _ok_cert_flex)

DiscFac_uncert_fixed, _med_uncert_fixed, _ok_uncert_fixed = _calibrate_discfac(
    lambda df: _fixed_median_AY(
        _solve_and_sim_fixed(df, fixed_dict_template)
    ),
    target=_AY_target, label="fixed+uncert",
)
_report("fixed+uncert", DiscFac_uncert_fixed, _med_uncert_fixed, _ok_uncert_fixed)

# Per-scenario calibrated values, exposed both to the rest of the
# script (for the actual solve+simulate pass below) and to the LaTeX
# tables and JSON artifact written at the end of the notebook.
calibrated_discfacs = {
    "uncert_flex":  DiscFac_uncert_flex,
    "cert_flex":    DiscFac_cert_flex,
    "uncert_fixed": DiscFac_uncert_fixed,
}
calibrated_medians = {
    "uncert_flex":  _med_uncert_flex,
    "cert_flex":    _med_cert_flex,
    "uncert_fixed": _med_uncert_fixed,
}
calibrated_converged = {
    "uncert_flex":  _ok_uncert_flex,
    "cert_flex":    _ok_cert_flex,
    "uncert_fixed": _ok_uncert_fixed,
}

# %% [markdown]
# ## Model 1: Flexible hours with uncertainty
#
# The main model from Low (2005). Agents choose consumption and labor
# supply each period, facing permanent and transitory wage shocks.

# %% Solve: flexible hours + uncertainty
agent = _solve_and_sim_flex(
    DiscFac_uncert_flex, sigma_perm, sigma_tran, base_dict,
)
print(f"Flexible-hours + uncertainty model solved ({T_work} periods).")

flex_u_c_work = np.mean(agent.history["cNrm"] * agent.history["pLvl"], axis=1)
flex_u_L_work = np.mean(agent.history["Lbr"], axis=1)
flex_u_a_work = np.mean(agent.history["aNrm"] * agent.history["pLvl"], axis=1)

flex_u_c_ret, flex_u_a_ret = retirement_paths(
    agent.history, CRRA=CRRA_hark, T_ret=T_ret,
    replacement_rate=replacement_rate, Rfree=Rfree,
    DiscFac=DiscFac_uncert_flex,
)
flex_u_c = np.concatenate([flex_u_c_work, flex_u_c_ret])
flex_u_a = np.concatenate([flex_u_a_work, flex_u_a_ret])
flex_u_L = np.concatenate([flex_u_L_work, np.zeros(T_ret)])

print(f"Mean labor (working, frac of time): {np.mean(flex_u_L_work):.4f}  (target: 0.40)")
print(f"Peak assets age (full life cycle): {ages[np.argmax(flex_u_a)]}")

# %% [markdown]
# ## Model 2: Flexible hours, no uncertainty (certainty case)
#
# Under certainty, agents have no precautionary motive. Comparing certainty
# vs uncertainty isolates the self-insurance effects (Figures 3-5).

# %% Solve: flexible hours + certainty
cert_agent = _solve_and_sim_flex(
    DiscFac_cert_flex, 0.001, 0.001, base_dict,
)

flex_c_c_work = np.mean(cert_agent.history["cNrm"] * cert_agent.history["pLvl"], axis=1)
flex_c_L_work = np.mean(cert_agent.history["Lbr"], axis=1)
flex_c_a_work = np.mean(cert_agent.history["aNrm"] * cert_agent.history["pLvl"], axis=1)

flex_c_c_ret, flex_c_a_ret = retirement_paths(
    cert_agent.history, CRRA=CRRA_hark, T_ret=T_ret,
    replacement_rate=replacement_rate, Rfree=Rfree,
    DiscFac=DiscFac_cert_flex,
)
flex_c_c = np.concatenate([flex_c_c_work, flex_c_c_ret])
flex_c_a = np.concatenate([flex_c_a_work, flex_c_a_ret])
flex_c_L = np.concatenate([flex_c_L_work, np.zeros(T_ret)])

print("Flexible-hours + certainty model solved.")

# %% [markdown]
# ## Model 3: Fixed hours + uncertainty
#
# Using HARK's `IndShockConsumerType` (exogenous labor). This comparison
# shows how labor flexibility affects precautionary saving (Figure 6).

# %% Solve: fixed hours + uncertainty (IndShockConsumerType)
fixed_agent = _solve_and_sim_fixed(DiscFac_uncert_fixed, fixed_dict_template)

fix_u_c_work = np.mean(fixed_agent.history["cNrm"] * fixed_agent.history["pLvl"], axis=1)
fix_u_a_work = np.mean(fixed_agent.history["aNrm"] * fixed_agent.history["pLvl"], axis=1)

fix_u_c_ret, fix_u_a_ret = retirement_paths(
    fixed_agent.history, CRRA=CRRA_hark, T_ret=T_ret,
    replacement_rate=replacement_rate, Rfree=Rfree,
    DiscFac=DiscFac_uncert_fixed,
)
fix_u_c = np.concatenate([fix_u_c_work, fix_u_c_ret])
fix_u_a = np.concatenate([fix_u_a_work, fix_u_a_ret])

print("Fixed-hours + uncertainty model solved.")

# %% [markdown]
# ---
# ## Figure 1: Life-Cycle Profiles under Uncertainty, Flexible Hours
# (cf. Low 2005, Figure 7)

# %%
retire_age = start_age + T_work  # age at which retirement begins (65)


def _mark_retirement(ax):
    ax.axvline(retire_age, color="grey", linestyle=":", linewidth=1)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(ages, flex_u_c, "b-", linewidth=2)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Mean Consumption")
axes[0].set_title("(a) Consumption")
axes[0].grid(True, alpha=0.3)
_mark_retirement(axes[0])

axes[1].plot(ages, flex_u_L, "r-", linewidth=2)
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Labor Supply (frac of time)")
axes[1].set_title("(b) Hours Worked")
axes[1].grid(True, alpha=0.3)
_mark_retirement(axes[1])

axes[2].plot(ages, flex_u_a, "g-", linewidth=2)
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Mean Assets")
axes[2].set_title("(c) Asset Accumulation")
axes[2].grid(True, alpha=0.3)
_mark_retirement(axes[2])

fig.suptitle(
    "Life-Cycle Profiles: Uncertainty with Flexible Hours\n"
    "(cf. Low 2005, Figure 7; vertical line marks retirement at age 65)",
    fontsize=13, y=1.02,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "lifecycle_profiles.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'lifecycle_profiles.png'}")

# %% [markdown]
# ## Figure 2: Hours Worked — Certainty vs Uncertainty
# (cf. Low 2005, Figure 3)
#
# Under uncertainty, agents front-load work effort as a form of
# precautionary behavior: they work more hours early in life.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ages_work, flex_u_L_work, "b-", linewidth=2, label="Uncertainty")
ax.plot(ages_work, flex_c_L_work, "r--", linewidth=2, label="Certainty")
ax.set_xlabel("Age")
ax.set_ylabel("Labor Supply (frac of time)")
ax.set_title("Hours Worked, Working Life: Certainty vs Uncertainty\n"
             "(cf. Low 2005, Figure 3)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "hours_cert_vs_uncert.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'hours_cert_vs_uncert.png'}")

# %% [markdown]
# ## Figure 3: Consumption — Certainty vs Uncertainty
# (cf. Low 2005, Figure 4)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ages, flex_u_c, "b-", linewidth=2, label="Uncertainty")
ax.plot(ages, flex_c_c, "r--", linewidth=2, label="Certainty")
ax.set_xlabel("Age")
ax.set_ylabel("Mean Consumption")
ax.set_title("Consumption: Certainty vs Uncertainty\n(cf. Low 2005, Figure 4)")
ax.legend()
ax.grid(True, alpha=0.3)
_mark_retirement(ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "consumption_cert_vs_uncert.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'consumption_cert_vs_uncert.png'}")

# %% [markdown]
# ## Figure 4: Assets — Certainty vs Uncertainty
# (cf. Low 2005, Figure 5)
#
# Precautionary saving under uncertainty leads to substantially higher
# asset accumulation, especially in middle age.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ages, flex_u_a, "b-", linewidth=2, label="Uncertainty")
ax.plot(ages, flex_c_a, "r--", linewidth=2, label="Certainty")
ax.set_xlabel("Age")
ax.set_ylabel("Mean Assets")
ax.set_title("Assets: Certainty vs Uncertainty\n(cf. Low 2005, Figure 5)")
ax.legend()
ax.grid(True, alpha=0.3)
_mark_retirement(ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "assets_cert_vs_uncert.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'assets_cert_vs_uncert.png'}")

# %% [markdown]
# ## Figure 5: Flexible vs Fixed Hours under Uncertainty
# (cf. Low 2005, Figure 6)
#
# Low (2005) finds that when labor supply is flexible, agents accumulate
# more assets in middle age than in the fixed-hours benchmark.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(ages, flex_u_c, "b-", linewidth=2, label="Flexible Hours")
axes[0].plot(ages, fix_u_c, "r--", linewidth=2, label="Fixed Hours")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Mean Consumption")
axes[0].set_title("Consumption: Flexible vs Fixed Hours")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
_mark_retirement(axes[0])

axes[1].plot(ages, flex_u_a, "b-", linewidth=2, label="Flexible Hours")
axes[1].plot(ages, fix_u_a, "r--", linewidth=2, label="Fixed Hours")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Mean Assets")
axes[1].set_title("Assets: Flexible vs Fixed Hours")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
_mark_retirement(axes[1])

fig.suptitle(
    "Flexible vs Fixed Hours under Uncertainty\n(cf. Low 2005, Figure 6)",
    fontsize=13, y=1.02,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "flexible_vs_fixed.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'flexible_vs_fixed.png'}")

# %% [markdown]
# ## Summary Statistics
#
# Comparison with Low (2005) Table 1 calibration targets:
# - Mean hours share (working age): ~0.40
# - Peak assets age (full life cycle): ~60
# - Median assets/income ratio (working age): 1.84
# - Aggregate assets/income ratio (working age): 2.14

# %%
# Cross-sectional working-life A/Y for each scenario.
#
# Income conventions (in level units, matching the figures):
#   flex:  income_{i,t} = pLvl_{i,t} * Lbr_{i,t} * TranShk_{i,t}
#   fixed: income_{i,t} = pLvl_{i,t} * TranShk_{i,t}     (pLvl already
#                                                         scaled by
#                                                         fixed_hours)
#   assets:                a_{i,t} = aNrm_{i,t} * pLvl_{i,t}


def _scenario_AY_flex(history):
    pLvl = np.asarray(history["pLvl"], dtype=float)
    Lbr  = np.asarray(history["Lbr"],  dtype=float)
    Tran = np.asarray(history["TranShk"], dtype=float)
    aNrm = np.asarray(history["aNrm"], dtype=float)
    income = pLvl * Lbr * Tran
    assets = aNrm * pLvl
    pos = income > 1e-8
    pointwise = np.where(pos, assets / np.where(pos, income, 1.0), np.nan)
    return (
        float(np.nanmedian(pointwise)),
        float(assets.sum() / income[pos].sum()),
    )


def _scenario_AY_fixed(history):
    pLvl = np.asarray(history["pLvl"], dtype=float)
    Tran = np.asarray(history["TranShk"], dtype=float)
    aNrm = np.asarray(history["aNrm"], dtype=float)
    income = pLvl * Tran
    assets = aNrm * pLvl
    pos = income > 1e-8
    pointwise = np.where(pos, assets / np.where(pos, income, 1.0), np.nan)
    return (
        float(np.nanmedian(pointwise)),
        float(assets.sum() / income[pos].sum()),
    )


median_AY_uncert_flex,  aggregate_AY_uncert_flex  = _scenario_AY_flex(agent.history)
median_AY_cert_flex,    aggregate_AY_cert_flex    = _scenario_AY_flex(cert_agent.history)
median_AY_uncert_fixed, aggregate_AY_uncert_fixed = _scenario_AY_fixed(fixed_agent.history)

# Per-scenario container -- consumed by both the JSON artifact and the
# LaTeX summary table.
calibrated_per_scenario = {
    "uncert_flex": {
        "DiscFac": DiscFac_uncert_flex,
        "delta": 1.0 / DiscFac_uncert_flex - 1.0,
        "median_AY": median_AY_uncert_flex,
        "aggregate_AY": aggregate_AY_uncert_flex,
        "converged": calibrated_converged["uncert_flex"],
    },
    "cert_flex": {
        "DiscFac": DiscFac_cert_flex,
        "delta": 1.0 / DiscFac_cert_flex - 1.0,
        "median_AY": median_AY_cert_flex,
        "aggregate_AY": aggregate_AY_cert_flex,
        "converged": calibrated_converged["cert_flex"],
    },
    "uncert_fixed": {
        "DiscFac": DiscFac_uncert_fixed,
        "delta": 1.0 / DiscFac_uncert_fixed - 1.0,
        "median_AY": median_AY_uncert_fixed,
        "aggregate_AY": aggregate_AY_uncert_fixed,
        "converged": calibrated_converged["uncert_fixed"],
    },
}

# Headline (uncertainty + flexible) statistics retained for the
# narrative summary text and qualitative comparisons below.
median_AY_working = median_AY_uncert_flex
aggregate_AY_working = aggregate_AY_uncert_flex

mean_hours = float(np.mean(flex_u_L_work))
peak_assets_age = int(ages[np.argmax(flex_u_a)])
mean_cons = float(np.mean(flex_u_c_work))

# Qualitative comparisons (uncertainty vs certainty; flexible vs fixed).
hours_age25_uncert = float(flex_u_L_work[0])
hours_age25_cert = float(flex_c_L_work[0])
peak_assets_uncert = float(np.max(flex_u_a))
peak_assets_cert = float(np.max(flex_c_a))
peak_assets_fixed = float(np.max(fix_u_a))

# Replication-target rows: (printed label, model value, target value,
# tex label, tex target string, format spec).  Driving the printed
# summary, the JSON artifact, and the LaTeX table from this single
# list keeps numbers from drifting between the script, the paper, and
# the JSON.
target_rows = [
    ("Mean hours (working, frac of time)", mean_hours,
     low_2005_targets["mean_hours_working"],
     "Mean hours, fraction of time worked", "0.40", "{:.3f}"),
    ("Peak assets age (full life cycle)", peak_assets_age,
     low_2005_targets["peak_assets_age"],
     "Peak-assets age (full life cycle)", "$\\sim$60", "{:d}"),
    ("Median A/Y, working life", median_AY_working,
     low_2005_targets["median_assets_to_income_working"],
     "Median $A/Y$", "1.84", "{:.2f}"),
    ("Aggregate A/Y, working life", aggregate_AY_working,
     low_2005_targets["aggregate_assets_to_income_working"],
     "Aggregate $A/Y$", "2.14", "{:.2f}"),
]

print("\n" + "=" * 78)
print("Summary: Replication vs Low (2005) Targets")
print("=" * 78)
print(f"{'Statistic':<44} {'Model':>9} {'Target':>9}")
print("-" * 78)
for label, model_value, target_value, _, tex_target, fmt in target_rows:
    model_str = fmt.format(model_value)
    target_str = "~60" if label.startswith("Peak assets age") else fmt.format(target_value)
    print(f"{label:<44} {model_str:>9} {target_str:>9}")
print(f"{'Mean consumption (working)':<44} "
      f"{mean_cons:>9.4f} {'--':>9}")
print("=" * 78)

print("\nPer-scenario calibration (Low 2005, Table 1, p.956):")
print(f"  {'Scenario':<28} {'paper delta':>12} {'this delta':>12} "
      f"{'paper A/Y':>10} {'this A/Y':>10}")
print("  " + "-" * 76)
for key, scen in low_2005_per_scenario.items():
    cal = calibrated_per_scenario[key]
    print(f"  {scen['label']:<28} {scen['delta']:>12.4f} {cal['delta']:>12.4f} "
          f"{scen['aggregate_AY']:>10.2f} {cal['aggregate_AY']:>10.2f}")
_uncert_raises_early_hours = hours_age25_uncert > hours_age25_cert
_uncert_raises_peak_assets = peak_assets_uncert > peak_assets_cert
_flex_raises_peak_assets = peak_assets_uncert > peak_assets_fixed


def _agree(flag, expected=True):
    """Tag for a qualitative comparison vs Low (2005)'s reported sign."""
    return "agrees with Low (2005)" if flag is expected else "DIFFERS from Low (2005)"


print("\nQualitative comparisons (Low 2005 sign in parentheses):")
print(f"  Hours @ age 25, uncert vs cert: "
      f"{hours_age25_uncert:.3f} vs {hours_age25_cert:.3f}  "
      f"[Low: uncert > cert -> {_agree(_uncert_raises_early_hours)}]")
print(f"  Peak assets, uncert vs cert:    "
      f"{peak_assets_uncert:.2f} vs {peak_assets_cert:.2f}  "
      f"[Low: uncert > cert -> {_agree(_uncert_raises_peak_assets)}]")
print(f"  Peak assets, flex vs fixed:     "
      f"{peak_assets_uncert:.2f} vs {peak_assets_fixed:.2f}  "
      f"[Low: flex > fixed -> {_agree(_flex_raises_peak_assets)}]")

# %% Persist comparison artifacts so the LaTeX paper can render the
# calibration and replication-targets tables from auto-generated
# content rather than stale hand-typed numbers.
import json

_calibration_tex_lines = [
    "% AUTO-GENERATED by Code/Python/Low2005.py - do not edit by hand.",
    r"\begin{tabular}{lc}",
    r"\toprule",
    r"\multicolumn{2}{l}{\textit{Common parameters (held fixed across scenarios)}} \\",
    r"\midrule",
    r"Parameter & Value \\",
    r"\midrule",
]
_calibration_tex_lines.extend(
    f"{parameter} & {value} \\\\" for parameter, value in calibration_rows
)
_calibration_tex_lines.extend([
    r"\midrule",
    r"\multicolumn{2}{l}{\textit{Calibrated discount rate $\delta$ "
    r"(Low 2005 Table 1 vs.~this replication)}} \\",
    r"\midrule",
    r"Scenario & $\delta$ (paper $\to$ this) \\",
    r"\midrule",
])
for key, scen in low_2005_per_scenario.items():
    cal = calibrated_per_scenario[key]
    flag = "" if cal["converged"] else r"$^\dagger$"
    _calibration_tex_lines.append(
        f"{scen['label']} & "
        f"${scen['delta']:.4f} \\to {cal['delta']:+.4f}${flag} \\\\"
    )
_any_unconverged = not all(c["converged"] for c in calibrated_per_scenario.values())
if _any_unconverged:
    _calibration_tex_lines.append(
        r"\multicolumn{2}{p{0.85\linewidth}}{\footnotesize "
        r"$^\dagger$ This replication's bisection over $\delta$ "
        r"could not match the PSID median $A/Y = 1.84$ even at the "
        r"upper bound of the search range, indicating an omitted "
        r"saving incentive (mortality, bequests, unemployment) "
        r"relative to Low (2005); we report the closest feasible "
        r"$\delta$.} \\"
    )
_calibration_tex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
])
(FIG_DIR / "calibration_table.tex").write_text(
    "\n".join(_calibration_tex_lines) + "\n"
)
print(f"Saved: {FIG_DIR / 'calibration_table.tex'}")

_summary = {
    "calibration_common": {
        "real_interest_rate": r_low,
        "relative_risk_aversion": gamma_low,
        "consumption_share": eta_low,
        "wage_alpha1": alpha1_wage,
        "wage_alpha2": alpha2_wage,
        "permanent_shock_variance": sigma_eps_sq,
        "transitory_shock_variance": sigma_nu_sq,
        "social_security_replacement_rate": replacement_rate,
        "fixed_hours_share": fixed_hours,
    },
    "numerical_settings": {
        "simulation_seed": simulation_seed,
        "agent_count": agent_count,
        "shock_count": shock_count,
        "asset_grid_count": asset_grid_count,
        "asset_grid_max": asset_grid_max,
    },
    "calibrated_discfac_per_scenario": calibrated_per_scenario,
    "low_2005_per_scenario": low_2005_per_scenario,
    "model_headline": {
        "scenario": "uncert_flex",
        "mean_hours_working": mean_hours,
        "peak_assets_age": peak_assets_age,
        "median_assets_to_income_working": median_AY_working,
        "aggregate_assets_to_income_working": aggregate_AY_working,
        "mean_consumption_working": mean_cons,
    },
    "low_2005_target_headline": low_2005_targets,
    "qualitative_signs_agree_with_low_2005": {
        "uncertainty_raises_early_hours": hours_age25_uncert > hours_age25_cert,
        "uncertainty_raises_peak_assets": peak_assets_uncert > peak_assets_cert,
        "flex_raises_peak_assets": peak_assets_uncert > peak_assets_fixed,
        "hours_age25_uncert": hours_age25_uncert,
        "hours_age25_cert": hours_age25_cert,
        "peak_assets_uncert_flex": peak_assets_uncert,
        "peak_assets_cert_flex": peak_assets_cert,
        "peak_assets_uncert_fixed": peak_assets_fixed,
    },
}
(FIG_DIR / "replication_summary.json").write_text(
    json.dumps(_summary, indent=2) + "\n"
)
print(f"Saved: {FIG_DIR / 'replication_summary.json'}")

_tex_lines = [
    "% AUTO-GENERATED by Code/Python/Low2005.py - do not edit by hand.",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"\multicolumn{3}{l}{\textit{Headline (uncertainty + flexible hours)}} \\",
    r"\midrule",
    r"Statistic (working-life cross-section) & "
    r"Low (2005) target & This replication \\",
    r"\midrule",
]
for _, model_value, target_value, tex_label, tex_target, fmt in target_rows:
    _tex_lines.append(
        f"{tex_label} & {tex_target} & {fmt.format(model_value)} \\\\"
    )
_tex_lines.extend([
    r"\midrule",
    r"\multicolumn{3}{l}{\textit{Per-scenario aggregate $A/Y$ after "
    r"calibrating $\delta$ to median $A/Y = 1.84$}} \\",
    r"\midrule",
    r"Scenario & Low (2005) Table 1 & This replication \\",
    r"\midrule",
])
for key, scen in low_2005_per_scenario.items():
    cal = calibrated_per_scenario[key]
    flag = "" if cal["converged"] else r"$^\dagger$"
    _tex_lines.append(
        f"{scen['label']} & {scen['aggregate_AY']:.2f} & "
        f"{cal['aggregate_AY']:.2f}{flag} \\\\"
    )
_any_unconverged_summary = not all(
    c["converged"] for c in calibrated_per_scenario.values()
)
if _any_unconverged_summary:
    _tex_lines.append(
        r"\multicolumn{3}{p{0.85\linewidth}}{\footnotesize "
        r"$^\dagger$ Median $A/Y$ could not reach the calibration "
        r"target of 1.84 even with the most patient agent in the "
        r"bisection range; the reported aggregate $A/Y$ is therefore "
        r"computed at the closest feasible $\delta$.} \\"
    )
_tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
(FIG_DIR / "replication_summary.tex").write_text(
    "\n".join(_tex_lines) + "\n"
)
print(f"Saved: {FIG_DIR / 'replication_summary.tex'}")
