"""
Behavior classes for the New Keynesian 3-Equation (NK3E) model.
Reference equations:
 y_t = A - a1 * r_{t-1}
 pi_t = pi_{t-1} + a2 * (y_t - y_e)
 r_s = (A - y_e) / a1
 r_t = r_s + a3 * (pi_t - pi_T)

Where a3 = 1 / [a1 * (1/(a2*b) + a2)].

Reference: Carlin & Soskice (2014); implementation aligned with
Source: A New Keynesian 3-Equation Model — https://macrosimulation.org/a_new_keynesian_3_equation_model
"""

__author__ = ["Mitja Devetak"]
__credits__ = ["Mitja Devetak"]
__license__ = "MIT"
__maintainer__ = ["Mitja Devetak"]

import logging

import torch
from tqdm import tqdm

from macrostat.core.behavior import Behavior
from macrostat.models.NK3E.parameters import ParametersNK3E
from macrostat.models.NK3E.scenarios import ScenariosNK3E
from macrostat.models.NK3E.variables import VariablesNK3E

logger = logging.getLogger(__name__)


class BehaviorNK3E(Behavior):
    """Simulation logic for the NK3E model.

    This class advances the model one period at a time using the three core
    equations:
    - IS (goods demand): y_t = A - a1 * r_{t-1}
    - Phillips (inflation): pi_t = pi_{t-1} + a2 * (y_t - y_e)
    - Monetary policy: r_t = r_s + a3 * (pi_t - pi_T), with r_s = (A - y_e)/a1

    The central bank response slope a3 is computed from structural parameters
    each step: a3 = 1 / [a1 * (1/(a2*b) + a2)], so you only specify a1, a2, b.

    Design notes:
    - We treat parameters as potentially time-varying via the scenarios system.
      Any parameter shocks are applied upstream in ``apply_parameter_shocks``,
      so the step reads already-shocked values from ``params``.
    - We keep a minimal state: output (y), inflation (pi), real rate (r) and
      the stabilizing real rate (r_s). Both pi and r use one-period lags, so
      they are configured with history=1 in the variables.
    """

    version = "NK3E"

    def __init__(
        self,
        parameters: ParametersNK3E | None = None,
        scenarios: ScenariosNK3E | None = None,
        variables: VariablesNK3E | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        if parameters is None:
            parameters = ParametersNK3E()
        if scenarios is None:
            scenarios = ScenariosNK3E(parameters=parameters)
        if variables is None:
            variables = VariablesNK3E(parameters=parameters)

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            debug=debug,
        )

    def initialize(self):
        """Set the model at its steady state before shocks start.

        At steady state, by definition y = y_e, r = r_s and pi = pi_T. We use
        the current (pre-shock) parameter values to compute r_s and then set
        all state variables accordingly. The base class will record this initial
        state for the required number of initialization timesteps.
        """
        a1 = self.params["a1"]
        A = self.params["A"]
        y_e = self.params["y_e"]
        pi_T = self.params["pi_T"]

        r_s = (A - y_e) / a1
        y_ss = y_e
        pi_ss = pi_T
        r_ss = r_s

        self.state["y"] = torch.tensor([y_ss])
        self.state["pi"] = torch.tensor([pi_ss])
        self.state["r"] = torch.tensor([r_ss])
        self.state["r_s"] = torch.tensor([r_s])

    def step(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        """Advance the model by one period using the 3-equation system.

        Parameters
        ----------
        t : int
            Current period (for bookkeeping only; equations are time-homogeneous).
        scenario : dict
            Vectorized scenario values at time t (not directly used here since
            parameter shocks are already reflected in ``params``).
        params : dict
            Parameter values for time t with scenario shocks already applied.

        Notes
        -----
        - We re-compute a3 every period from (a1, a2, b) in case those are
          shocked over time.
        - IS uses the lagged real rate from ``self.prior['r']`` to produce y_t.
        - The Phillips curve uses the output gap to update inflation.
        - The policy rule sets the real rate relative to the stabilizing rate.
        """
        a1 = params["a1"]
        a2 = params["a2"]
        b = params["b"]
        # Parameter shocks already applied in params via apply_parameter_shocks
        A = params["A"]
        pi_T = params["pi_T"]
        y_e = params["y_e"]

        a3 = 1.0 / (a1 * (1.0 / (a2 * b) + a2))

        # r_s depends on current A and y_e
        r_s = (A - y_e) / a1

        # IS: y_t = A - a1 * r_{t-1}
        y_t = A - a1 * self.prior["r"]

        # PC: pi_t = pi_{t-1} + a2 * (y_t - y_e)
        pi_t = self.prior["pi"] + a2 * (y_t - y_e)

        # MP: r_t = r_s + a3 * (pi_t - pi_T)
        r_t = r_s + a3 * (pi_t - pi_T)

        self.state["y"] = y_t
        self.state["pi"] = pi_t
        self.state["r"] = r_t
        self.state["r_s"] = r_s

    def forward(self):
        """Run the full simulation, optionally with a tqdm progress bar.

        This mirrors the base class implementation but adds a progress bar when
        ``parameters.hyper['use_tqdm']`` is True. At each step we:
        1) build the scenario slice for time t,
        2) apply parameter shocks (so ``params`` reflects current-time values),
        3) call :meth:`step` to update the state,
        4) record the new state into the timeseries and history buffers.
        """
        torch.manual_seed(self.hyper["seed"])
        self.state, self.history = self.variables.initialize_tensors(
            t=self.hyper["timesteps"],
            dtype=torch.float32,
            requires_grad=self.hyper["requires_grad"],
            device=self.hyper["device"],
        )

        # initialize
        self.initialize()
        for t in range(self.hyper["timesteps_initialization"]):
            self.variables.record_state(t, self.state)
        for t in range(self.hyper["timesteps_initialization"]):
            self.history = self.variables.update_history(self.state)
        self.prior = self.state

        iterator = range(
            self.hyper["timesteps_initialization"] + 1, self.hyper["timesteps"]
        )
        if self.hyper.get("use_tqdm", False):
            iterator = tqdm(iterator, desc="NK3E Simulation", leave=False)

        for t in iterator:
            self.state = self.variables.new_state()
            idx = torch.where(
                torch.arange(self.hyper["timesteps"]) == t,
                torch.ones(1),
                torch.zeros(1),
            )
            scenario = {k: idx @ v for k, v in self.scenarios.items()}
            params = self.apply_parameter_shocks(t, scenario)
            self.step(t=t, scenario=scenario, params=params)
            self.variables.record_state(t, self.state)
            self.history = self.variables.update_history(self.state)
            self.prior = self.state

        return None
