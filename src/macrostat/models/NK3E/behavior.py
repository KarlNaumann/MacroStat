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

        self.state["y"] = y_ss.reshape(1)
        self.state["pi"] = pi_ss.reshape(1)
        self.state["r"] = r_ss.reshape(1)
        self.state["r_s"] = r_s.reshape(1)
        # a3 baseline for recording/graphing (depends on a1, a2, b)
        a2 = self.params["a2"]
        b = self.params["b"]
        a3 = 1.0 / (a1 * (1.0 / (a2 * b) + a2))
        self.state["a3"] = a3.reshape(1)

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
        # Compute per-period components via subfunctions with explicit dependencies
        self.central_bank_slope(t=t, scenario=scenario, params=params)
        self.stabilizing_real_rate(t=t, scenario=scenario, params=params)
        self.is_curve_output(t=t, scenario=scenario, params=params)
        # Additive state-variable shocks (see ScenariosNK3E.get_default_scenario_values).
        # Apply each shock immediately after its equation so the perturbation
        # propagates within the same period through the downstream equations:
        #   OutputShock  -> y_t is seen by Phillips (pi_t) and, via r_t, next IS
        #   InflationShock -> pi_t is seen by the policy rule (r_t)
        #   RateShock    -> r_t is seen by next period's IS curve
        # Use .get() so user-built scenarios that bypass the default values
        # still work, and fall back to a same-shape zero tensor on the state.
        self.state["y"] = self.state["y"] + scenario.get(
            "OutputShock", torch.zeros_like(self.state["y"])
        )
        self.phillips_curve_inflation(t=t, scenario=scenario, params=params)
        self.state["pi"] = self.state["pi"] + scenario.get(
            "InflationShock", torch.zeros_like(self.state["pi"])
        )
        self.monetary_policy_rate(t=t, scenario=scenario, params=params)
        self.state["r"] = self.state["r"] + scenario.get(
            "RateShock", torch.zeros_like(self.state["r"])
        )

    def central_bank_slope(self, t: int, scenario: dict, params: dict | None = None):
        r"""Compute the monetary policy reaction slope a3 from structural parameters.

        Parameters
        ----------
        t : int
            Current period (for bookkeeping only).
        scenario : dict
            Scenario dictionary (not used).
        params : dict | None
            Parameter values for time t with scenario shocks already applied.

        Equations
        ---------
        .. math::
            a_3 = \frac{1}{a_1\left(\frac{1}{a_2 b} + a_2\right)}

        Dependency
        ----------
        - parameters: a1
        - parameters: a2
        - parameters: b

        Sets
        -----
        - a3
        """
        a1 = params["a1"]
        a2 = params["a2"]
        b = params["b"]
        value = 1.0 / (a1 * (1.0 / (a2 * b) + a2))
        # preserve dtype/device/shape without requiring a Python scalar
        self.state["a3"] = torch.ones_like(self.state["a3"]) * value

    def stabilizing_real_rate(self, t: int, scenario: dict, params: dict | None = None):
        r"""Compute the stabilizing real rate r_s consistent with output at potential.

        Parameters
        ----------
        t : int
            Current period (for bookkeeping only).
        scenario : dict
            Scenario dictionary (not used).
        params : dict | None
            Parameter values for time t with scenario shocks already applied.

        Equations
        ---------
        .. math::
            r_s = \frac{A - y_e}{a_1}

        Dependency
        ----------
        - parameters: A
        - parameters: y_e
        - parameters: a1

        Sets
        -----
        - r_s
        """
        a1 = params["a1"]
        A = params["A"]
        y_e = params["y_e"]
        value = (A - y_e) / a1
        self.state["r_s"] = torch.ones_like(self.state["r_s"]) * value

    def is_curve_output(self, t: int, scenario: dict, params: dict | None = None):
        r"""IS curve: output as a function of demand shifter and lagged real rate.

        Parameters
        ----------
        t : int
            Current period (for bookkeeping only).
        scenario : dict
            Scenario dictionary (not used).
        params : dict | None
            Parameter values for time t with scenario shocks already applied.

        Equations
        ---------
        .. math::
            y_t = A - a_1 r_{t-1}

        Dependency
        ----------
        - parameters: A
        - parameters: a1
        - prior: r

        Sets
        -----
        - y
        """
        A = params["A"]
        a1 = params["a1"]
        self.state["y"] = A - a1 * self.prior["r"]

    def phillips_curve_inflation(
        self, t: int, scenario: dict, params: dict | None = None
    ):
        r"""Phillips curve: inflation responds to the output gap.

        Parameters
        ----------
        t : int
            Current period (for bookkeeping only).
        scenario : dict
            Scenario dictionary (not used).
        params : dict | None
            Parameter values for time t with scenario shocks already applied.

        Equations
        ---------
        .. math::
            \pi_t = \pi_{t-1} + a_2 (y_t - y_e)

        Dependency
        ----------
        - prior: pi
        - state: y
        - parameters: a2
        - parameters: y_e

        Sets
        -----
        - pi
        """
        a2 = params["a2"]
        y_e = params["y_e"]
        self.state["pi"] = self.prior["pi"] + a2 * (self.state["y"] - y_e)

    def monetary_policy_rate(self, t: int, scenario: dict, params: dict | None = None):
        r"""Monetary policy rule: real rate reacts to inflation deviations.

        Parameters
        ----------
        t : int
            Current period (for bookkeeping only).
        scenario : dict
            Scenario dictionary (not used).
        params : dict | None
            Parameter values for time t with scenario shocks already applied.

        Equations
        ---------
        .. math::
            r_t = r_s + a_3 (\pi_t - \pi^T)

        Dependency
        ----------
        - state: r_s
        - state: a3
        - state: pi
        - parameters: pi_T

        Sets
        -----
        - r
        """
        pi_T = params["pi_T"]
        self.state["r"] = self.state["r_s"] + self.state["a3"] * (
            self.state["pi"] - pi_T
        )
