"""
Behavior classes for the MacroStat model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import torch

from macrostat.core.parameters import Parameters
from macrostat.core.scenarios import Scenarios
from macrostat.core.variables import Variables

logger = logging.getLogger(__name__)


class Behavior(torch.nn.Module):
    """Base class for the behavior of the MacroStat model."""

    def __init__(
        self,
        parameters: Parameters,
        scenarios: Scenarios,
        variables: Variables,
        scenario: int = 0,
        differentiable: bool = False,
        debug: bool = False,
    ):
        """Initialize the behavior of the MacroStat model.

        Parameters
        ----------
        parameters: macrostat.core.parameters.Parameters
            The parameters of the model.
        scenarios: macrostat.core.scenarios.Scenarios
            The scenarios of the model.
        variables: macrostat.core.variables.Variables
            The variables of the model.
        scenario: int
            The scenario to use for the model run.
        debug: bool
            Whether to print debug information.
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the parameters
        self.params = parameters.to_nn_parameters()
        self.hyper = parameters.hyper

        # Initialize the scenarios
        self.scenarios = scenarios.to_nn_parameters(scenario=scenario)
        self.scenarioID = scenario

        # Initialize the variables
        self.variables = variables

        # Settings
        self.differentiable = differentiable
        self.debug = debug

    ############################################################################
    # Simulation of the model
    ############################################################################

    def forward(self):
        """Forward pass of the behavior.

        This should include the model's main loop, and is implemented as a placeholder.
        The idea is for users to implement an initialize() and step() function,
        which will be called by the forward() function.

        If there are additional steps necessary, users may wish to overwrite this function.
        """
        # Set the seed
        torch.manual_seed(self.hyper["seed"])

        # Initialize the output tensors
        self.state, self.history = self.variables.initialize_tensors(
            t=self.hyper["timesteps"],
            dtype=torch.float32,
            requires_grad=self.hyper["requires_grad"],
            device=self.hyper["device"],
        )

        # Initialize the model
        logger.debug(
            f"Initializing model (t=0...{self.hyper['timesteps_initialization']})"
        )
        self.initialize()

        for t in range(self.hyper["timesteps_initialization"]):
            self.variables.record_state(t, self.state)

        for t in range(self.hyper["timesteps_initialization"]):
            self.history = self.variables.update_history(self.state)

        # Initialize the prior and state
        self.prior = self.state

        # Run the model for the remaining timesteps
        logger.debug(
            f"Simulating model (t={self.hyper['timesteps_initialization'] + 1}...{self.hyper['timesteps']})"
        )

        for t in range(
            self.hyper["timesteps_initialization"] + 1, self.hyper["timesteps"]
        ):
            self.state = self.variables.new_state()
            # Get scenario series for this point in time
            idx = torch.where(
                torch.arange(self.hyper["timesteps"]) == t,
                torch.ones(1),
                torch.zeros(1),
            )
            scenario = {k: idx @ v for k, v in self.scenarios.items()}

            # Apply parameter shocks
            params = self.apply_parameter_shocks(t, scenario)

            # Step the model
            self.step(t=t, scenario=scenario, params=params)

            # Store the outputs
            self.variables.record_state(t, self.state)
            self.history = self.variables.update_history(self.state)
            self.prior = self.state

        return None

    def initialize(self):
        """Initialize the behavior.

        This should include the model's initialization steps, and set all of the
        necessary state variables. They only need to be set for one period, and
        will then be copied to the history and prior to be used in the step function.
        """
        raise NotImplementedError("Behavior.initialize() to be implemented by model")

    def step(self, t: int, scenario: dict, params: dict | None = None):
        """Step function of the behavior.

        This should include the model's main loop.

        Parameters
        ----------
        t: int
            The current timestep.
        scenario: dict
            The scenario information for the current timestep.
        """
        raise NotImplementedError("Behavior.step() to be implemented by model")

    def apply_parameter_shocks(self, t: int, scenario: dict):
        """Apply parameter shocks to the model.

        Any parameter in the model can be shocked/changed during the simulation
        using the scenario information. Specifically, for a parameter alpha, the
        user can pass two types of potential shocks:
        1. An multiplicative shock, generically named alpha_multiply
        2. An additive shock, generically named alpha_add

        This function will apply the shocks to the parameters, and return a
        dictionary with the updated parameters. The application of the shocks is
        independent, that is, the multiplicative shock does not affect the additive
        shock and vice versa. This is done by first applying the multiplicative
        shock, and then the additive shock.

        Parameters
        ----------
        t: int
            The current timestep.
        scenario: dict
            The scenario information for the current timestep.

        Returns
        -------
        dict
            A dictionary with the updated parameters.
        """
        params = {}

        for key, value in self.params.items():
            mul, add = torch.tensor(1.0), torch.tensor(0.0)
            if f"{key}_multiply" in scenario:
                mul = scenario[f"{key}_multiply"]
            if f"{key}_add" in scenario:
                add = scenario[f"{key}_add"]
            params[key] = value * mul + add

        return params

    ############################################################################
    # Steady State
    ############################################################################

    def compute_theoretical_steady_state(self, **kwargs):
        """Compute the theoretical steady state of the model.

        This process generally follows the structure of the forward() function,
        but instead of simulating the model, the steady state is computed at
        each timestep. Therefore, (1) the model is initialized, and (2) for
        each timestep the parameter and scenario information is passed to the
        compute_theoretical_steady_state_per_step() function that computes the
        steady state at that timestep.

        Parameters
        ----------
        **kwargs: dict
            Additional keyword arguments.
        """
        # Set the seed
        torch.manual_seed(self.hyper["seed"])

        # Initialize the output tensors
        self.state, _ = self.variables.initialize_tensors(
            t=self.hyper["timesteps"],
            dtype=torch.float32,
            requires_grad=self.hyper["requires_grad"],
            device=self.hyper["device"],
        )

        # Initialize the model
        info = f"(t=0...{self.hyper['timesteps_initialization']})"
        logger.debug(f"Initializing model {info}")
        self.initialize()

        for t in range(self.hyper["timesteps_initialization"]):
            self.variables.record_state(t, self.state)

        # Compute the steady state
        info = f"(t={self.hyper['timesteps_initialization'] + 1}...{self.hyper['timesteps']})"
        logger.debug(f"Computing theoretical steady state {info}")

        for t in range(
            self.hyper["timesteps_initialization"] + 1, self.hyper["timesteps"]
        ):
            self.state = self.variables.new_state()

            # Get scenario series for this point in time
            idx = torch.where(
                torch.arange(self.hyper["timesteps"]) == t,
                torch.ones(1),
                torch.zeros(1),
            )
            scenario = {k: idx @ v for k, v in self.scenarios.items()}

            # Apply parameter shocks
            params = self.apply_parameter_shocks(t, scenario)

            # Compute the steady state
            self.compute_theoretical_steady_state_per_step(
                t=t, params=params, scenario=scenario
            )

            # Store the outputs
            self.variables.record_state(t, self.state)

        return None

    def compute_theoretical_steady_state_per_step(self, **kwargs):
        """Compute the theoretical steady state of the model per step."""
        raise NotImplementedError(
            "Behavior.compute_theoretical_steady_state_per_step() to be implemented by model"
        )

    ############################################################################
    # Some Differentiable PyTorch Alternatives
    ############################################################################

    def diffwhere(self, condition, x1, x2):
        """Where condition that is differentiable with respect to the condition.

        Requires:
            self.hyper['diffwhere'] = True
            self.hyper['sigmoid_constant'] as a large number

        Note: For non-NaN/inf, where(x > eps, z, y) is (x - eps > 0) * (z - y) + y,
        so we can use the sigmoid function to approximate the where function.

        Parameters
        ----------
        condition : torch.Tensor
            Condition to be evaluated expressed as x - eps
        x1 : torch.Tensor
            Value to be returned if condition is True
        x2 : torch.Tensor
            Value to be returned if condition is False
        """
        sig = torch.sigmoid(torch.mul(condition, self.hyper["sigmoid_constant"]))
        return torch.add(torch.mul(sig, torch.sub(x1, x2)), x2)

    def tanhmask(self, x):
        """Convert a variable into 0 (x<0) and 1 (x>0)

        Requires:
            self.hyper['tanh_constant'] as a large number

        Parameters
        ----------
        x: torch.Tensor
            The variable to be converted.

        """
        kwg = {"dtype": torch.float64, "requires_grad": True}
        return torch.div(
            torch.add(
                torch.ones(x.size(), **kwg),
                torch.tanh(torch.mul(x, self.hyper["tanh_constant"])),
            ),
            torch.tensor(2.0, **kwg),
        )

    def diffmin(self, x1, x2):
        """Smooth approximation to the minimum
        B: https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function

        Requires:
            self.hyper['min_constant'] as a large number

        Parameters
        ----------
        x1: torch.Tensor
            The first variable to be compared.
        x2: torch.Tensor
            The second variable to be compared.
        """
        r = self.hyper["min_constant"]
        pt1 = torch.exp(torch.mul(x1, -1 * r))
        pt2 = torch.exp(torch.mul(x2, -1 * r))
        return torch.mul(-1 / r, torch.log(torch.add(pt1, pt2)))

    def diffmax(self, x1, x2):
        """Smooth approximation to the minimum
        B: https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function

        Requires:
            self.hyper['max_constant'] as a large number

        Parameters
        ----------
        x1: torch.Tensor
            The first variable to be compared.
        x2: torch.Tensor
            The second variable to be compared.
        """
        r = self.hyper["max_constant"]
        pt1 = torch.exp(torch.mul(x1, r))
        pt2 = torch.exp(torch.mul(x2, r))
        return torch.mul(1 / r, torch.log(torch.add(pt1, pt2)))

    def diffmin_v(self, x):
        """Smooth approximation to the minimum. See diffmin

        Parameters
        ----------
        x: torch.Tensor
            The variable to be converted.

        Requires:
            self.hyper['min_constant'] as a large number
        """
        r = self.hyper["min_constant"]
        temp = torch.exp(torch.mul(x, -1 * r))
        return torch.mul(-1 / r, torch.log(torch.sum(temp)))

    def diffmax_v(self, x):
        """Smooth approximation to the maximum for a tensor. See diffmax

        Requires:
            self.hyper['max_constant'] as a large number

        Parameters
        ----------
        x: torch.Tensor
            The variable to be converted.
        """
        r = self.hyper["max_constant"]
        temp = torch.exp(torch.mul(x, r))
        return torch.mul(1 / r, torch.log(torch.sum(temp)))


if __name__ == "__main__":
    pass
