# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Parameters class for the Mark-0 COVID heterogeneous-agent ABM.

Reference: Bouchaud, J.-P., Gualdi, S., Tarzia, M., Zamponi, F. (2018).
"Optimal inflation target: insights from an agent-based model".
Economics 12 (15).

COVID-extension: Sharma, Bouchaud, Gualdi, Tarzia, Zamponi (2021).
"V–, U–, L– or W–shaped economic recovery after COVID-19: Insights from
an Agent Based Model."

The model is a closed-economy ABM with :math:`N` firms, one representative
household, one bank, one central bank. Each macro period :math:`t` runs a
24-phase loop covering price/wage adjustment, bankruptcy, household
consumption, interest-rate setting, and firm revival.
"""

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersMark0COVID(Parameters):
    r"""Parameters for the Mark-0 COVID model.

    Economic parameters (twenty-three) cover firm price/wage adjustment,
    central-bank reaction function, household consumption, bank interest-rate
    formation, and revival dynamics. Hyperparameters control population size
    ``N_firms``, simulation horizon ``timesteps``, smoothing constants for
    the differentiable masks, and the per-instance RNG seed.

    The frozen-noise reparameterisation pre-draws three independent
    ``(timesteps, N_firms)`` ``U(0,1)`` buffers in ``initialize()`` (price,
    wage, revival). Gradient flow through these stochastic phases requires
    the smooth ``tanhmask`` boundary at sites where a parameter controls a
    threshold — see ``behavior.py`` for the explicit sites.
    """

    version = "Mark0COVID"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parameters=parameters,
            hyperparameters=hyperparameters,
            *args,
            **kwargs,
        )

    def get_default_parameters(self):
        return {
            "HiringFiringRate": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"R",
                "unit": ".",
                "value": 2.0,
            },
            "LoanRateGammaSensitivity": {
                "lower bound": 0.0,
                "upper bound": 1000.0,
                "notation": r"\alpha_\Gamma",
                "unit": ".",
                "value": 50.0,
            },
            "GammaBaseline": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\Gamma_0",
                "unit": ".",
                "value": 0.0,
            },
            "WagePriceAdjustmentRatio": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\gamma_w / \gamma_p",
                "unit": ".",
                "value": 1.0,
            },
            "PriceAdjustmentSize": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\gamma_p",
                "unit": ".",
                "value": 0.1,
            },
            "FiringPropensity": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\eta_0",
                "unit": ".",
                "value": 0.1,
            },
            "WageInflationFactor": {
                "lower bound": 0.0,
                "upper bound": 2.0,
                "notation": r"f_w",
                "unit": ".",
                "value": 1.0,
            },
            "InterestRateBaseline": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\rho^\star",
                "unit": "1 / period",
                "value": 0.005,
            },
            "CBInflationReaction": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\phi_\pi",
                "unit": ".",
                "value": 0.0,
            },
            "CBEmploymentReaction": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\phi_\varepsilon",
                "unit": ".",
                "value": 0.0,
            },
            "CBInflationTarget": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\pi^\star",
                "unit": "1 / period",
                "value": 0.002,
            },
            "CBUnemploymentTarget": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\hat\varepsilon^\star",
                "unit": ".",
                "value": 0.05,
            },
            "DefaultThreshold": {
                "lower bound": 0.0,
                "upper bound": 100.0,
                "notation": r"\Theta",
                "unit": ".",
                "value": 2.0,
            },
            "BankruptcyInterestEffect": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"f",
                "unit": ".",
                "value": 0.5,
            },
            "ConsumptionRealRateSensitivity": {
                "lower bound": 0.0,
                "upper bound": 100.0,
                "notation": r"\alpha_c",
                "unit": ".",
                "value": 4.0,
            },
            "ConsumptionPropensityBaseline": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"c_0",
                "unit": ".",
                "value": 0.5,
            },
            "DividendShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\delta",
                "unit": ".",
                "value": 0.02,
            },
            "HouseholdIntensityOfChoice": {
                "lower bound": 0.0,
                "upper bound": 100.0,
                "notation": r"\beta",
                "unit": ".",
                "value": 2.0,
            },
            "ExpectedInflationEWMAWeight": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\tau^R",
                "unit": ".",
                "value": 0.5,
            },
            "ExpectedInflationTargetWeight": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\tau^T",
                "unit": ".",
                "value": 0.5,
            },
            "FirmRevivalProbability": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\phi",
                "unit": "1 / period",
                "value": 0.1,
            },
            "EWMAMemory": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\omega",
                "unit": ".",
                "value": 0.2,
            },
            "InitialProductionScale": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"y_0",
                "unit": ".",
                "value": 0.5,
            },
        }

    def get_default_hyperparameters(self):
        import torch

        hyper = super().get_default_hyperparameters()
        hyper["timesteps"] = 100
        hyper["timesteps_initialization"] = 0
        hyper["N_firms"] = 1000
        hyper["seed"] = 0
        hyper["seed_init"] = 0
        hyper["dtype"] = torch.float64
        hyper["tanh_constant"] = 1.0e6
        hyper["sigmoid_constant"] = 1.0e4
        hyper["min_constant"] = 100.0
        hyper["max_constant"] = 100.0
        hyper["epsilon"] = 1.0e-12
        hyper["sectors"] = ["Firms", "Household", "Bank", "CentralBank"]
        return hyper

    def verify_parameters(self):
        super().verify_parameters()

        if self.hyper["N_firms"] < 1:
            raise ValueError(
                f"N_firms must be a positive integer; got {self.hyper['N_firms']!r}."
            )

        phi = self["FirmRevivalProbability"]
        if hasattr(phi, "item"):
            phi = phi.item()
        if phi > 0.0 and self.hyper["tanh_constant"] < 50.0 / phi:
            logger.warning(
                "tanh_constant=%g is below the recommended 50/phi=%g; revival "
                "rate will bias high through the tanhmask reparameterisation.",
                self.hyper["tanh_constant"],
                50.0 / phi,
            )
