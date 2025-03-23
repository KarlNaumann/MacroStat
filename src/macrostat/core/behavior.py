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
        record: bool = False,
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
        record: bool
            Whether to record the model output as a whole timeseries, or just
            the state variables (less memory-intensive).
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
        self.record = record
        self.debug = debug

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
        kwargs = {
            "dtype": torch.float32,
            "requires_grad": self.hyper["requires_grad"],
            "device": self.hyper["device"],
        }

        self.state, self.history = self.variables.initialize_tensors(
            t=self.hyper["timesteps"], **kwargs
        )

        # Initialize the model
        logger.debug(
            f"Initializing model (t=0...{self.hyper['timesteps_initialization']})"
        )
        self.initialize()
        if self.record:
            for t in range(self.hyper["timesteps_initialization"]):
                self.variables.record_state(t, self.state)

        for t in range(self.hyper["timesteps_initialization"]):
            self.history = self.variables.update_history(self.state)

        self.prior = self.state
        self.state = self.variables.new_state()

        # Run the model for the remaining timesteps
        logger.debug(
            f"Simulating model (t={self.hyper['timesteps_initialization'] + 1}...{self.hyper['timesteps']})"
        )

        for t in range(
            self.hyper["timesteps_initialization"] + 1, self.hyper["timesteps"]
        ):
            # Get scenario series for this point in time
            idx = torch.where(
                torch.arange(self.hyper["timesteps"]) == t,
                torch.ones(1),
                torch.zeros(1),
            )
            scenario = {k: idx @ v for k, v in self.scenarios.items()}

            self.step(t, scenario)

            self.variables.record_state(t, self.state)

            self.history = self.variables.update_history(self.state)
            self.prior = self.state
            self.state = self.variables.new_state()

        return None

    def initialize(self):
        """Initialize the behavior.

        This should include the model's initialization steps, and set all of the
        necessary state variables. They only need to be set for one period, and
        will then be copied to the history and prior to be used in the step function.
        """
        raise NotImplementedError("Behavior.initialize() to be implemented by model")

    def step(self, t: int, scenario: dict):
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

    # Some Differentiable PyTorch Alternatives

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
        if self.hyper["diffwhere"]:
            sig = torch.sigmoid(torch.mul(condition, self.hyper["sigmoid_constant"]))
            out = torch.add(torch.mul(sig, torch.sub(x1, x2)), x2)
        else:
            out = torch.where(condition > 0, x1, x2)
        return out

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
