"""
pytest code for the Model class
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

from macrostat.models.model import Model


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


class TestModel:
    """Tests for the class Model found in models/model.py"""

    # General kwargs for potential testing
    kwargs = dict(
        parameters=dict(p1=1, p2=np.ones((2,)), p3=np.arange(4).reshape((2, 2))),
        hyper_parameters=dict(h1=10, h2=np.ones((2,)), h3="test"),
        name="test_model",
    )

    def test_init(self):
        """Check that the attributes are set correctly"""
        kwargs = copy.deepcopy(self.kwargs)
        model = Model(**kwargs)
        allclose_dict_comparison(model.parameters, kwargs["parameters"])
        allclose_dict_comparison(model.hyper_parameters, kwargs["hyper_parameters"])
        assert model.name == kwargs["name"]

    def test_parameter_validation(self):
        """Check that parameter validation accepts only numbers"""
        kwargs = copy.deepcopy(self.kwargs)
        kwargs["parameters"]["badType"] = "badType"
        with pytest.raises(ValueError):
            model = Model(**kwargs)

    def test_simulate_notimplemented(self):
        """Test raising of notimplementederror for base class"""
        print(self.kwargs)
        model = Model(**self.kwargs)
        with pytest.raises(NotImplementedError):
            model.simulate()

    def test_equivalence(self):
        """Test the model equivalence implementation"""
        model = Model(**self.kwargs)
        # No difference
        model2 = Model(**self.kwargs)
        assert model == model2

    def test_equivalence_parameters(self):
        """Test the model equivalence implementation"""
        model = Model(**self.kwargs)
        # Parameter difference: Addition of a parameter
        alt = copy.deepcopy(self.kwargs)
        alt["parameters"]["newparam"] = 1
        model2 = Model(**alt)
        assert model != model2
        # Parameter difference: Change of value
        alt = copy.deepcopy(self.kwargs)
        alt["parameters"]["p1"] += 1
        model2 = Model(**alt)
        assert model != model2

    def test_equivalence_hyper_parameters(self):
        """Test the model equivalence implementation"""
        model = Model(**self.kwargs)
        # Hyper Parameter difference: Addition of a parameter
        alt = copy.deepcopy(self.kwargs)
        alt["hyper_parameters"]["newparam"] = None
        model2 = Model(**alt)
        assert model != model2
        # Hyper Parameter difference: Change of value
        alt = copy.deepcopy(self.kwargs)
        alt["hyper_parameters"]["h1"] = None
        model2 = Model(**alt)
        assert model != model2

    def test_equivalence_name(self):
        """Test the model equivalence implementation"""
        model = Model(**self.kwargs)
        # Name difference: Change of value
        alt = copy.deepcopy(self.kwargs)
        alt["name"] = None
        model2 = Model(**alt)
        assert model != model2

    def test_equivalence_output(self):
        """Test the model equivalence implementation"""
        model = Model(**self.kwargs)
        model.output = pd.DataFrame(np.arange(9))
        # Output difference: None vs. Int
        model2 = Model(**self.kwargs)
        assert model != model2
        # Output difference: Change of value
        model2.output = model.output + 1
        assert model != model2

    def test_save_load(self, tmpdir):
        """Test whether the saved and loaded models are equivalent"""
        model = Model(**self.kwargs)
        # Check with a given path
        path = tmpdir.join("model.pkl")
        model.save(path=path)
        model2 = Model.load(path=path)
        assert model == model2

        # Check with the default path
        model.save()
        model2 = Model.load(path=f"{model.name}.pkl")
        assert model == model2
        os.remove(f"{model.name}.pkl")
