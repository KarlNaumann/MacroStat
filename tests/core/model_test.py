import os
import tempfile

import pytest
import torch

from macrostat.core import Behavior, Model, Parameters, Scenarios, Variables


@pytest.fixture
def test_model():
    params = Parameters({"test_param": 1.0}, hyperparameters={"T": 10})
    return Model(parameters=params)


def test_model_initialization():
    """Test basic model initialization"""
    params = Parameters({"test_param": 1.0}, hyperparameters={"T": 10})
    model = Model(parameters=params)

    assert isinstance(model.parameters, Parameters)
    assert isinstance(model.scenarios, Scenarios)
    assert isinstance(model.variables, Variables)
    assert model.name == "model"


def test_model_initialization_with_dict():
    """Test model initialization with dictionary parameters"""
    params = {"test_param": 1.0}
    hyperparams = {"T": 10}
    model = Model(parameters=params, hyperparameters=hyperparams)

    assert isinstance(model.parameters, Parameters)
    assert model.parameters["test_param"] == 1.0


def test_forward_pass(test_model):
    """Test model forward pass"""
    with pytest.raises(NotImplementedError):
        test_model.forward()


def test_simulate(test_model):
    """Test model simulation"""
    with pytest.raises(NotImplementedError):
        test_model.simulate()


def test_save_load():
    """Test model save and load functionality"""
    params = Parameters({"test_param": 1.0}, hyperparameters={"T": 10})
    model = Model(parameters=params)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model.save(tmp.name)
        loaded_model = Model.load(tmp.name)

    assert isinstance(loaded_model, Model)
    assert loaded_model.parameters["test_param"] == model.parameters["test_param"]

    os.unlink(tmp.name)

    # Test without passing a


def test_from_json(tmp_path):
    """Test model initialization from JSON files"""
    model = Model(parameters={"test_param": 1.0}, hyperparameters={"T": 10})
    print(model.parameters.hyper)
    model.to_json(tmp_path)

    model2 = Model.from_json(f"{tmp_path}_params.json", f"{tmp_path}_scenarios.json")
    assert isinstance(model2, Model)
    assert model2.parameters["test_param"] == 1.0


def test_custom_behavior():
    """Test model with custom behavior class"""

    class CustomBehavior(Behavior):
        def forward(self):
            return torch.tensor([1.0])

    params = Parameters({"test_param": 1.0}, hyperparameters={"T": 10})
    model = Model(parameters=params, behavior=CustomBehavior)

    output = model.forward()
    assert torch.equal(output, torch.tensor([1.0]))
