"""
pytest code for the TorchModel class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import copy
import os

import numpy as np
import pandas as pd
import pytest
import torch

from macrostat.models.torchmodel import TorchModel


def allclose_dict_comparison(dict1: dict, dict2: dict):
    """Comparison of two dictionaries using the numpy allclose function

    Parameters
    ----------
    dict1 : dict
    dict2 : dict
    """
    assert dict1.keys() == dict2.keys()

    for k in dict1:
        if np.issubdtype(np.array(dict1[k]).dtype, np.number):
            assert np.allclose(dict1[k], dict2[k])
        else:
            assert dict1[k] == dict2[k]


@pytest.fixture
def default_params():
    return {"param1": 1.0, "param2": 2.0}


@pytest.fixture
def default_hyper_params():
    return {
        "device": "cpu",
        "requires_grad": True,
        "diffwhere": True,
        "sigmoid_constant": 100,
        "tanh_constant": 100,
        "min_constant": 10,
        "max_constant": 10,
    }


@pytest.fixture
def torch_model(default_params, default_hyper_params):
    return TorchModel(parameters=default_params, hyper_parameters=default_hyper_params)


def test_model_initialization(torch_model, default_params, default_hyper_params):
    assert torch_model.parameters == default_params
    assert torch_model.hyper_parameters == default_hyper_params
    assert torch_model.name == "torchmodel"
    assert torch_model.tparam is not None


def test_update_tparam(torch_model):
    torch_model._update_tparam()
    for param_name, param_value in torch_model.parameters.items():
        assert torch_model.tparam[param_name].item() == param_value
        assert isinstance(torch_model.tparam[param_name], torch.Tensor)
        assert torch_model.tparam[param_name].requires_grad is True
        assert (
            torch_model.tparam[param_name].device.type
            == torch_model.hyper_parameters["device"]
        )


def test_simulation_output(torch_model):
    # Mocking the forward method and initialize_simulation method
    torch_model.forward = lambda *args, **kwargs: None

    # Simulating outputs
    torch_model.outputs = {"param1": torch.tensor([1.0]), "param2": torch.tensor([2.0])}

    output_df = torch_model.simulate()

    assert isinstance(output_df, pd.DataFrame)
    assert "param1" in output_df.columns
    assert "param2" in output_df.columns
    assert np.isclose(output_df["param1"].values[0], 1.0)
    assert np.isclose(output_df["param2"].values[0], 2.0)


def test_diffwhere(torch_model):
    condition = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)
    x1 = torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([30.0, 40.0], dtype=torch.float64, requires_grad=True)

    # When diffwhere is False
    torch_model.hyper_parameters["diffwhere"] = False
    output = torch_model.diffwhere(condition, x1, x2)
    expected = torch.where(condition > 0, x1, x2)
    assert torch.equal(output, expected)

    # Expected approximation to where using sigmoid
    sigmoid_approx = torch.sigmoid(condition * 10000)
    expected = sigmoid_approx * (x1 - x2) + x2

    condition = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)
    x1 = torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([30.0, 40.0], dtype=torch.float64, requires_grad=True)
    # When diffwhere is True
    torch_model.hyper_parameters["diffwhere"] = True
    torch_model.hyper_parameters["sigmoid_constant"] = torch.tensor([10000])
    output = torch_model.diffwhere(condition, x1, x2)
    assert torch.allclose(output, expected, atol=1e-6)


def test_diffwhere_differentiability(torch_model):

    # Tensors with requires_grad=True to track gradients
    condition = torch.tensor([0.5, -0.5, 0.0], requires_grad=True)
    x1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    # Call the diffwhere function
    output = torch_model.diffwhere(condition, x1, x2)

    # Perform a reduction to enable backpropagation (e.g., sum all elements)
    output.sum().backward()

    # Check if gradients have been computed
    assert condition.grad is not None, "Gradient for condition is not computed"
    assert x1.grad is not None, "Gradient for x1 is not computed"
    assert x2.grad is not None, "Gradient for x2 is not computed"

    # Optional: Check if the gradients are non-zero
    assert torch.any(condition.grad != 0), "Gradient for condition is zero"
    assert torch.any(x1.grad != 0), "Gradient for x1 is zero"
    assert torch.any(x2.grad != 0), "Gradient for x2 is zero"


def test_diffmin(torch_model):
    x1 = torch.tensor([5.0], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    torch_model.hyper_parameters["min_constant"] = 10.0
    result = torch_model.diffmin(x1, x2)
    expected = torch.min(x1, x2)
    assert torch.allclose(result, expected, atol=1e-4)


def test_diffmax(torch_model):
    x1 = torch.tensor([5.0], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    torch_model.hyper_parameters["max_constant"] = 10
    result = torch_model.diffmax(x1, x2)
    expected = torch.max(x1, x2)
    assert torch.allclose(result, expected, atol=1e-4)


def test_tanhmask(torch_model):
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    torch_model.hyper_parameters["tanh_constant"] = 10.0
    result = torch_model.tanhmask(x)
    expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-4)


def test_tanhmask_differentiability(torch_model):
    # Input tensor with requires_grad=True to track gradients
    x = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float64, requires_grad=True)
    # Call the tanhmask function
    output = torch_model.tanhmask(x)
    # Perform a reduction to enable backpropagation (e.g., sum all elements)
    output.sum().backward()
    # Check if gradients have been computed
    assert x.grad is not None, "Gradient for x is not computed"
    # Optional: Check if the gradients are non-zero
    assert torch.any(x.grad != 0), "Gradient for x is zero"


def test_diffmin_v(torch_model):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    torch_model.hyper_parameters["min_constant"] = 10.0
    result = torch_model.diffmin_v(x)
    expected = torch.min(x)
    assert torch.allclose(result, expected, atol=1e-4)


def test_diffmin_differentiability(torch_model):
    # Input tensors with requires_grad=True to track gradients
    x1 = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float64, requires_grad=True)

    # Call the diffmin function
    output = torch_model.diffmin(x1, x2)

    # Perform a reduction to enable backpropagation (e.g., sum all elements)
    output.sum().backward()

    # Check if gradients have been computed
    assert x1.grad is not None, "Gradient for x1 is not computed"
    assert x2.grad is not None, "Gradient for x2 is not computed"

    # Optional: Check if the gradients are non-zero
    assert torch.any(x1.grad != 0), "Gradient for x1 is zero"
    assert torch.any(x2.grad != 0), "Gradient for x2 is zero"


def test_diffmax_v(torch_model):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    torch_model.hyper_parameters["max_constant"] = 10.0
    result = torch_model.diffmax_v(x)
    expected = torch.max(x)
    assert torch.allclose(result, expected, atol=1e-4)


def test_diffmax_differentiability(torch_model):
    # Input tensors with requires_grad=True to track gradients
    x1 = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float64, requires_grad=True)

    # Call the diffmax function
    output = torch_model.diffmax(x1, x2)

    # Perform a reduction to enable backpropagation (e.g., sum all elements)
    output.sum().backward()

    # Check if gradients have been computed
    assert x1.grad is not None, "Gradient for x1 is not computed"
    assert x2.grad is not None, "Gradient for x2 is not computed"

    # Optional: Check if the gradients are non-zero
    assert torch.any(x1.grad != 0), "Gradient for x1 is zero"
    assert torch.any(x2.grad != 0), "Gradient for x2 is zero"


def test_default_hyper_parameters(torch_model):
    # Ensure default hyperparameters are set correctly if not provided
    default_hyper_params = torch_model._default_hyper_parameters()
    for key, value in default_hyper_params.items():
        assert key in torch_model.hyper_parameters
        assert torch_model.hyper_parameters[key] == value

    torch_model = TorchModel({})
    default_hyper_params = torch_model._default_hyper_parameters()
    for key, value in default_hyper_params.items():
        assert key in torch_model.hyper_parameters
        assert torch_model.hyper_parameters[key] == value
