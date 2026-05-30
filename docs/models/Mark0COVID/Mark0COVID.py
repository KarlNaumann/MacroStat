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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mark-0 COVID Heterogeneous-Agent ABM

# %% [markdown]
# Mark-0 {cite:t}`BouchaudGualdiTarziaZamponi2018OptimalInflation` is a
# closed-economy ABM with $N$ firms, one representative household, one
# commercial bank, and one central bank. The COVID extension
# {cite:p}`SharmaBouchaudGualdiTarziaZamponi2021COVID` adds the
# parameter set used to study V-, U-, L-, and W-shaped recovery
# regimes. Each macro period executes an ordered 24-phase loop:
# price/wage adjustment, hiring/firing, production, household
# consumption, interest-rate setting, bankruptcy, and firm revival.
#
# Stochastic phases are reparameterised through three independent
# frozen $U(0, 1)$ noise buffers of shape ``(timesteps, N_firms)``
# pre-drawn in ``initialize()``. Boundary conditions on the two
# gradient-critical paths (``stay_alive`` indicator and the positive-$Y$
# guard) use ``Behavior.diffwhere``; all other branches use plain
# ``torch.where``, matching the abmstat reference pattern.

# %% [markdown]
# ## Module Contents
#
# Mark0COVID is divided into Variables, Parameters (fixed constants),
# Scenarios, and the Behavior (model initialization and steps). See the
# sibling pages linked from the model index for the per-component
# tables.

# %% [markdown]
# ## Implementation in MacroStat
#
# The forward pass is differentiable end-to-end. The 24-phase loop is
# decomposed into named methods on ``BehaviorMark0COVID`` so that each
# phase carries its own docstring and ``Equations`` section.
# ``ParametersMark0COVID.get_default_hyperparameters`` sets
# ``dtype = torch.float64`` to match the abmstat reference precision.

# %% [markdown]
# ## Model Dynamics

# %% [markdown]
# ### Preparatory Steps

# %%
# %load_ext autoreload
# %autoreload 2

import importlib
import logging
import sys

from matplotlib import pyplot as plt

from macrostat.models.Mark0COVID import Mark0COVID, ParametersMark0COVID

plt.style.use("../../macrostat.mplstyle")
importlib.reload(logging)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# %% [markdown]
# ### Four Gualdi regimes
#
# We run the model for the four named scenarios at ``T = 200``,
# ``N_firms = 500``. Each scenario varies the hiring/firing rate $R$
# and default threshold $\Theta$ through the additive-shock channel.

# %%
params = ParametersMark0COVID(
    hyperparameters={"timesteps": 200, "N_firms": 500, "seed": 0}
)
model = Mark0COVID(parameters=params)

regimes = {
    "Scenario 0: Baseline (R=2, Theta=2)": 0,
    "Scenario 1: Endogenous crises (R=0.5, Theta=3)": 1,
    "Scenario 2: Unstable (R=1.5, Theta=4)": 2,
    "Scenario 3: Full collapse (R=0.5, Theta=6)": 3,
}

paths = {}
for label, scenario_id in regimes.items():
    out = model.simulate(scenario=scenario_id)
    paths[label] = {
        "Unemployment": out["Unemployment"].detach().cpu().numpy().squeeze(),
        "Inflation": out["Inflation"].detach().cpu().numpy().squeeze(),
        "Production": out["TotalProduction"].detach().cpu().numpy().squeeze(),
        "BankruptcyRate": out["BankruptcyRate"].detach().cpu().numpy().squeeze(),
    }

# %% [markdown]
# ### Aggregate paths
#
# Unemployment, inflation, total production, and bankruptcy rate across
# the four regimes. The baseline scenario converges near full
# employment; the endogenous-crises regime shows large oscillations;
# the unstable regime drifts towards a high-unemployment fixed point;
# the full-collapse regime is absorbed near zero production.

# %% jupyter={"source_hidden": true}
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(11, 7))
colors = ["k", "tab:blue", "tab:orange", "tab:red"]

for color, (label, series) in zip(colors, paths.items()):
    axs[0, 0].plot(series["Unemployment"], color=color, linewidth=1.0, label=label)
    axs[0, 1].plot(series["Inflation"], color=color, linewidth=1.0, label=label)
    axs[1, 0].plot(series["Production"], color=color, linewidth=1.0, label=label)
    axs[1, 1].plot(series["BankruptcyRate"], color=color, linewidth=1.0, label=label)

axs[0, 0].set_title("Unemployment")
axs[0, 0].set_xlabel("period")
axs[0, 1].set_title("Inflation")
axs[0, 1].set_xlabel("period")
axs[1, 0].set_title("Total production")
axs[1, 0].set_xlabel("period")
axs[1, 1].set_title("Bankruptcy rate")
axs[1, 1].set_xlabel("period")

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
fig.suptitle("Figure Mark0COVID.1: Four Gualdi regimes")
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()

# %% [markdown]
# ### Notes
#
# - The shape of every het-agent state tensor in
#   ``model.variables.timeseries`` is ``(T + 1, N_firms)``; aggregates
#   are stored as ``(T + 1, 1)``.
# - The frozen-noise reparameterisation makes the forward pass
#   differentiable with respect to the price, wage, and revival
#   parameters. Reproducibility under fixed ``seed`` is verified in
#   :mod:`tests.models.mark0covid_test`.
# - The default ``dtype = torch.float64`` matches the abmstat reference.
#   Override via ``ParametersMark0COVID(hyperparameters={"dtype":
#   torch.float32})`` for faster GPU inference at the cost of larger
#   trajectory drift versus abmstat.
