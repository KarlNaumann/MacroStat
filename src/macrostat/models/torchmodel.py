# -*- coding: utf-8 -*-
"""
Generic model class for models implemented using PyTorch
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torchfunc

import macrostat.models.model as model


class TorchModel(model.Model, torch.nn.Module):
    """Generic model class for models implemented using PyTorch

    This class provides a wrapper for users to write their underlying model
    behavior while maintaining a uniformly accessible interface. Specifically,
    the user is expected to adapt the model.forward() function to their needs.
    The use of PyTorch allows for automatic differentiation which has computational
    advantages compared to finite differences.

    Attributes
    ----------
    name: str
        Name of the model, such as "model". Used for file and database names
    parameters : dict
        Dictionary of all parameters
    hyper_parameters : dict
        Dictionary of all hyperparameters
    output : pd.DataFrame
        None, or the latest simulation run for given parameters

    Example
    -------
    A general workflow for a model might look like

    >>> model = Model(parameters, hyper_parameters)
    >>> output = model.simulate()
    >>> model.save()

    """

    def __init__(
        self,
        parameters: dict = None,
        hyper_parameters: dict = None,
        name: str = "torchmodel",
    ):
        """Initialisation of the TorchModel class

        Parameters
        ----------
        parameters : dict
            dictionary of the named parameters of the model
        hyper_parameters : dict
            dictionary of hyper-parameters related to the model
        name : str (default 'model')
            name of the model (for use in filenaming)
        """
        # Check hyperparameters, adding defaults if necessary
        if hyper_parameters is None:
            hyper_parameters = self._default_hyper_parameters()
        else:
            for k, v in self._default_hyper_parameters().items():
                if k not in hyper_parameters:
                    hyper_parameters[k] = v

        # Initialize the parent classes
        model.Model.__init__(
            self, parameters=parameters, hyper_parameters=hyper_parameters, name=name
        )
        torch.nn.Module.__init__(self)

        # For PyTorch we define the parameter order
        self.parameter_order = tuple(self.parameters.keys())

        # Generate pytorch parameters
        self.tparam = {}
        self._update_tparam()

    def simulate(self) -> pd.DataFrame:
        """Simulate a model run using the stored parameters

        This function is designed to be overwritten by the user's
        specific implementation of their model. Note that it is
        expected for the user to set the ''self.output'' attribute
        to the output generated.

        The function will run ''self.initialize_simulation'' to set up the
        model and then run the forward pass of the model. Furthermore, in
        the pure simulation case, we omit the gradient calculation.

        Returns
        -------
        output : pd.DataFrame
            Output of the model. Generically it should have a "time"-like
            index and variables across the columns
        """
        self._update_tparam()
        self.initialize_simulation()

        with torch.no_grad():
            self.forward(**self.tparam)

        self.outputs = pd.DataFrame(
            {k: v.cpu().clone().detach().numpy() for k, v in self.outputs.items()}
        )

        return self.outputs

    def initialize_simulation(self):
        """Initialize the model's state variables before running the simulation
        For instance, if one wants to load in a model state to start from.
        """
        pass

    def forward(self, *args, **kwargs):
        """Run the model forward through time, e.g. the loop over timesteps goes here.

        This method is called by the simulation method and by the pytorch
        autograd system. Generally, it shouldn't be called directly by the user.
        """
        return NotImplementedError

    def _update_tparam(self):
        """Update the pytorch parameters from the parameters dictionary. Ensuring
        that the parameters are on the correct device and require a gradient
        """
        for k, v in self.parameters.items():
            self.tparam[k] = torch.tensor(
                v,
                dtype=torch.float64,
                requires_grad=True,
                device=self.hyper_parameters["device"],
            )

    def _default_hyper_parameters(self) -> dict:
        """Return the default hyper parameters. This is primarily a dictionary
        of the PyTorch specific hyperparameters such as constants and the device.

        Returns
        -------
        parameters : dict
            dictionary of the parameters
        """
        return {
            "device": "cpu",
            "requires_grad": True,
            "diffwhere": True,
            "sigmoid_constant": 100,
            "tanh_constant": 100,
            "min_constant": 10,
            "max_constant": 10,
        }

    ### Some Differentiable PyTorch Alternatives

    def diffwhere(self, condition, x1, x2):
        """Where condition that is differentiable with respect to the condition.

        Requires:
            self.hyper_parameters['diffwhere'] = True
            self.hyper_parameters['sigmoid_constant'] as a large number

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
        if self.hyper_parameters["diffwhere"]:
            sig = torch.sigmoid(
                torch.mul(condition, self.hyper_parameters["sigmoid_constant"])
            )
            out = torch.add(torch.mul(sig, torch.sub(x1, x2)), x2)
        else:
            out = torch.where(condition > 0, x1, x2)
        return out

    def tanhmask(self, x):
        """Convert a variable into 0 (x<0) and 1 (x>0)"""
        kwg = {"dtype": torch.float64, "requires_grad": True}
        return torch.div(
            torch.add(
                torch.ones(x.size(), **kwg),
                torch.tanh(torch.mul(x, self.hyper_parameters["tanh_constant"])),
            ),
            torch.tensor(2.0, **kwg),
        )

    def diffmin(self, x1, x2):
        """Smooth approximation to the minimum
        B: https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function

        Requires:
            self.hyper_parameters['min_constant'] as a large number
        """
        r = self.hyper_parameters["min_constant"]
        pt1 = torch.exp(torch.mul(x1, -1 * r))
        pt2 = torch.exp(torch.mul(x2, -1 * r))
        return torch.mul(-1 / r, torch.log(torch.add(pt1, pt2)))

    def diffmax(self, x1, x2):
        """Smooth approximation to the minimum
        B: https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function

        Requires:
            self.hyper_parameters['max_constant'] as a large number
        """
        r = self.hyper_parameters["max_constant"]
        pt1 = torch.exp(torch.mul(x1, r))
        pt2 = torch.exp(torch.mul(x2, r))
        return torch.mul(1 / r, torch.log(torch.add(pt1, pt2)))

    def diffmin_v(self, x):
        """Smooth approximation to the minimum. See diffmin

        Requires:
            self.hyper_parameters['min_constant'] as a large number
        """
        r = self.hyper_parameters["min_constant"]
        temp = torch.exp(torch.mul(x, -1 * r))
        return torch.mul(-1 / r, torch.log(torch.sum(temp)))

    def diffmax_v(self, x):
        """Smooth approximation to the maximum for a tensor. See diffmax

        Requires:
            self.hyper_parameters['max_constant'] as a large number
        """
        r = self.hyper_parameters["max_constant"]
        temp = torch.exp(torch.mul(x, r))
        return torch.mul(1 / r, torch.log(torch.sum(temp)))


if __name__ == "__main__":

    torch_model = TorchModel({})
    x = torch.tensor([1.0, 2.0, 3.0])
    torch_model.hyper_parameters["min_constant"] = 10.0
    result = torch_model.diffmax_v(x)
    print(result)
    pass
