# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Variables class for the Mark-0 COVID model.

Firm-level state lives on tensors of shape ``(N_firms,)`` and is resolved via
the het-agent ``sectors=["N_firms"]`` shape-axis convention introduced in
``MacroStat_HetAgentInfra``. Macro state (aggregate output, inflation,
interest rates, ...) lives on scalar tensors.

Mark-0 is non-SFC in the Godley-Lavoie sense: the household balance sheet is
implicit (savings :math:`S`) and the firm balance is the single :math:`A_i`
account. Variables therefore omit the ``sfc`` key and the SFC machinery skips
them silently (per the het-agent migration notes).

Frozen-noise buffers (price / wage / revival) are pre-drawn in
``Behavior.initialize`` and stored as private model state on the Behavior
instance, not as recorded Variables. The buffers are not part of the user-
visible timeseries; set ``hyper["record_noise"]=True`` to materialise them
out-of-band on the behavior instance (mirrors KirmansAnts ``_micro_trajectory``).
"""

import logging

from macrostat.core.variables import Variables
from macrostat.models.Mark0COVID.parameters import ParametersMark0COVID

logger = logging.getLogger(__name__)


class VariablesMark0COVID(Variables):
    r"""Variables for the Mark-0 COVID model.

    Firm-level state (price, wage, production, demand, assets, alive mask)
    carries ``sectors=["N_firms"]`` and resolves to shape ``(N_firms,)``
    tensors via the shape-axis machinery in ``Variables.new_state``.

    Macro state (output, inflation, interest rates, etc.) carries the empty
    ``sectors=[]`` and resolves to shape ``(1,)``.

    Parameters
    ----------
    parameters: ParametersMark0COVID | None
        Model parameters; defaults if ``None``.
    """

    version = "Mark0COVID"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersMark0COVID | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersMark0COVID()
        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        firm = ["N_firms"]
        scalar: list = []
        return {
            "FirmPrice": {
                "notation": r"P_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": firm,
            },
            "FirmWage": {
                "notation": r"W_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": firm,
            },
            "FirmProduction": {
                "notation": r"Y_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": firm,
            },
            "FirmDemand": {
                "notation": r"D_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": firm,
            },
            "FirmAssets": {
                "notation": r"A_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": firm,
            },
            "FirmProfits": {
                "notation": r"\Pi_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": firm,
            },
            "FirmAlive": {
                "notation": r"\alpha_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": firm,
            },
            "AveragePrice": {
                "notation": r"\bar P_t",
                "unit": ".",
                "history": 1,
                "sectors": scalar,
            },
            "AverageWage": {
                "notation": r"\bar W_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "MaxWage": {
                "notation": r"W^{\max}_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "TotalProduction": {
                "notation": r"Y_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "TotalPayroll": {
                "notation": r"\sum_i W_{i,t} Y_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "TotalDemand": {
                "notation": r"D_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "Inflation": {
                "notation": r"\pi_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "ExpectedInflationEWMA": {
                "notation": r"\pi^{ema}_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "ExpectedInflationUsed": {
                "notation": r"\hat\pi_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "DepositRateEWMA": {
                "notation": r"\bar\rho^d_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "LoanRateEWMA": {
                "notation": r"\bar\rho^l_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "Unemployment": {
                "notation": r"u_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "UnemploymentEWMA": {
                "notation": r"\bar u_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "HouseholdSavings": {
                "notation": r"S_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "FirmSavingsTotal": {
                "notation": r"\mathcal{E}^+_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "FirmDebtTotal": {
                "notation": r"\mathcal{E}^-_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "FirmAssetsTotal": {
                "notation": r"\sum_i A_{i,t}",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "DefaultedTotal": {
                "notation": r"\text{def}_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "BankruptcyRate": {
                "notation": r"b_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "CBRate": {
                "notation": r"\rho^0_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "LoanRate": {
                "notation": r"\rho^l_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "DepositRate": {
                "notation": r"\rho^d_t",
                "unit": "1 / period",
                "history": 0,
                "sectors": scalar,
            },
            "ConsumptionPropensity": {
                "notation": r"c_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "ConsumptionBudget": {
                "notation": r"B_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "FirmGamma": {
                "notation": r"\Gamma_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "Employment": {
                "notation": r"\varepsilon_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
            "M0Stock": {
                "notation": r"M^0_t",
                "unit": ".",
                "history": 0,
                "sectors": scalar,
            },
        }
