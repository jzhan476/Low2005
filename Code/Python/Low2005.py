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

# === HARK Parameter Mapping ===
# Low: u = (c^eta * z^(1-eta))^(1-gamma) / (1-gamma), z = leisure
# HARK: u = (c * z^alpha)^(1-rho) / (1-rho)
# => alpha = (1-eta)/eta, rho = 1 - eta*(1-gamma)
CRRA_hark = 1.0 - eta_low * (1.0 - gamma_low)
alpha_hark = (1.0 - eta_low) / eta_low
DiscFac = beta_low
Rfree = 1.0 + r_low

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
# ## Deterministic retirement extension
#
# `LaborIntMargConsumerType` requires a strictly positive wage in every
# period, so retirement cannot be solved natively in the same agent.
# Instead we treat retirement as a deterministic, perfect-foresight
# continuation of the working-life simulation: each agent enters
# retirement holding their terminal assets and a constant pension equal
# to `replacement_rate` times their final permanent income, and consumes
# optimally subject to the Euler equation with no bequests.  This is the
# certainty-equivalent extension; agents face no further income risk
# during retirement.

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

# %% [markdown]
# ## Model 1: Flexible hours with uncertainty
#
# The main model from Low (2005). Agents choose consumption and labor
# supply each period, facing permanent and transitory wage shocks.

# %% Common agent parameters
base_dict = {
    "cycles": 1,
    "T_cycle": T_work,
    "CRRA": CRRA_hark,
    "DiscFac": DiscFac,
    "Rfree": [Rfree] * T_work,
    "LivPrb": [1.0] * T_work,
    "PermGroFac": PermGroFac_list,
    "WageRte": [1.0] * T_work,
    "PermShkCount": 7,
    "TranShkCount": 7,
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
    "aXtraMax": 50.0,
    "aXtraCount": 48,
    "aXtraNestFac": 3,
    "aXtraExtra": None,
    "vFuncBool": False,
    "CubicBool": False,
    "AgentCount": 10000,
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
}

# %% Solve: flexible hours + uncertainty
uncert_dict = base_dict.copy()
uncert_dict["PermShkStd"] = [sigma_perm] * T_work
uncert_dict["TranShkStd"] = [sigma_tran] * T_work

agent = LaborIntMargConsumerType(**uncert_dict)
agent.solve()
print(f"Flexible-hours + uncertainty model solved ({T_work} periods).")

agent.track_vars = ["cNrm", "Lbr", "aNrm", "pLvl"]
agent.initialize_sim()
agent.simulate()

flex_u_c_work = np.mean(agent.history["cNrm"] * agent.history["pLvl"], axis=1)
flex_u_L_work = np.mean(agent.history["Lbr"], axis=1)
flex_u_a_work = np.mean(agent.history["aNrm"] * agent.history["pLvl"], axis=1)

flex_u_c_ret, flex_u_a_ret = retirement_paths(
    agent.history, CRRA=CRRA_hark, T_ret=T_ret,
    replacement_rate=replacement_rate, Rfree=Rfree, DiscFac=DiscFac,
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
cert_dict = base_dict.copy()
cert_dict["PermShkStd"] = [0.001] * T_work
cert_dict["TranShkStd"] = [0.001] * T_work

cert_agent = LaborIntMargConsumerType(**cert_dict)
cert_agent.solve()

cert_agent.track_vars = ["cNrm", "Lbr", "aNrm", "pLvl"]
cert_agent.initialize_sim()
cert_agent.simulate()

flex_c_c_work = np.mean(cert_agent.history["cNrm"] * cert_agent.history["pLvl"], axis=1)
flex_c_L_work = np.mean(cert_agent.history["Lbr"], axis=1)
flex_c_a_work = np.mean(cert_agent.history["aNrm"] * cert_agent.history["pLvl"], axis=1)

flex_c_c_ret, flex_c_a_ret = retirement_paths(
    cert_agent.history, CRRA=CRRA_hark, T_ret=T_ret,
    replacement_rate=replacement_rate, Rfree=Rfree, DiscFac=DiscFac,
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
fixed_dict = {
    "cycles": 1,
    "T_cycle": T_work,
    "CRRA": gamma_low,
    "DiscFac": DiscFac,
    "Rfree": [Rfree] * T_work,
    "LivPrb": [1.0] * T_work,
    "PermGroFac": PermGroFac_list,
    "PermShkStd": [sigma_perm] * T_work,
    "TranShkStd": [sigma_tran] * T_work,
    "PermShkCount": 7,
    "TranShkCount": 7,
    "UnempPrb": 0.0,
    "UnempPrbRet": 0.0,
    "T_retire": 0,
    "IncUnemp": 0.0,
    "IncUnempRet": 0.0,
    # Match the borrowing-constraint specification used by the flexible
    # model above so that `flexible_vs_fixed.png` reflects only the
    # labour-supply margin.  Earlier versions used `BoroCnstArt=0.0`
    # here while the flexible model used `None`, which conflated
    # labour flexibility with a tighter ad-hoc no-borrowing constraint.
    "BoroCnstArt": None,
    "aXtraMin": 0.001,
    "aXtraMax": 50.0,
    "aXtraCount": 48,
    "aXtraNestFac": 3,
    "AgentCount": 10000,
    "T_age": T_work + 1,
    "T_sim": T_work,
    "aNrmInitMean": -10.0,
    "aNrmInitStd": 0.0,
    "pLvlInitMean": 0.0,
    "pLvlInitStd": 0.0,
    "PermGroFacAgg": 1.0,
}

fixed_agent = IndShockConsumerType(**fixed_dict)
fixed_agent.solve()

fixed_agent.track_vars = ["cNrm", "aNrm", "pLvl"]
fixed_agent.initialize_sim()
fixed_agent.simulate()

fix_u_c_work = np.mean(fixed_agent.history["cNrm"] * fixed_agent.history["pLvl"], axis=1)
fix_u_a_work = np.mean(fixed_agent.history["aNrm"] * fixed_agent.history["pLvl"], axis=1)

fix_u_c_ret, fix_u_a_ret = retirement_paths(
    fixed_agent.history, CRRA=gamma_low, T_ret=T_ret,
    replacement_rate=replacement_rate, Rfree=Rfree, DiscFac=DiscFac,
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
# When labor supply is flexible, agents use it as an additional
# self-insurance mechanism, reducing the need for precautionary saving.

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
# - Median assets/income ratio (working age): 1.84
# - Aggregate assets/income ratio (working age): 2.14

# %%
print("\n" + "=" * 60)
print("Summary: Replication vs Low (2005) Targets")
print("=" * 60)
print(f"{'Statistic':<40} {'Model':>10} {'Target':>10}")
print("-" * 60)
print(f"{'Mean hours (working, frac of time)':<40} "
      f"{np.mean(flex_u_L_work):.4f}     {'0.40':>5}")
print(f"{'Peak assets age (full life cycle)':<40} "
      f"{ages[np.argmax(flex_u_a)]:>5d}     {'~60':>5}")
print(f"{'Mean consumption (working)':<40} "
      f"{np.mean(flex_u_c_work):.4f}     {'--':>5}")
print("=" * 60)
print("\nKey qualitative results:")
print(f"  Uncertainty raises hours early in life: "
      f"{flex_u_L_work[0]:.3f} (uncert) vs {flex_c_L_work[0]:.3f} (cert)")
print(f"  Uncertainty raises asset accumulation: "
      f"{np.max(flex_u_a):.2f} (uncert) vs {np.max(flex_c_a):.2f} (cert)")
print(f"  Flexible hours reduce precautionary saving: "
      f"{np.max(flex_u_a):.2f} (flex) vs {np.max(fix_u_a):.2f} (fixed)")
