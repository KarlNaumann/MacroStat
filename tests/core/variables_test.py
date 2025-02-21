"""
pytest code for the Variables class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import os

import pytest
import torch

from macrostat.core import Parameters, Variables


class TestVariables:
    """Tests for the Variables class found in models/variables.py"""

    # Test data
    variable_info = {
        "var1": {"sectors": [0], "history": 0},
        "var2": {"sectors": [0, 1], "history": 2},
        "var3": {"sectors": [0], "history": 1},
    }

    params = Parameters(hyper_parameters={"T": 100})

    def test_init(self):
        """Test initialization of Variables class"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        assert v.info == self.variable_info
        assert v.parameters == self.params

    def test_initialize_tensors(self):
        """Test tensor initialization"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state, history = v.initialize_tensors(100)

        # Check state variables
        assert set(state.keys()) == {"var1", "var2", "var3"}
        assert state["var1"].shape == (1,)
        assert state["var2"].shape == (2,)
        assert state["var3"].shape == (1,)

        # Check history variables
        assert set(history.keys()) == {"var2", "var3"}
        assert len(history["var2"]) == 0
        assert len(history["var3"]) == 0

        # Check timeseries initialization
        assert set(v.timeseries.keys()) == {"var1", "var2", "var3"}
        assert v.timeseries["var1"].shape == (100, 1)
        assert v.timeseries["var2"].shape == (100, 2)
        assert v.timeseries["var3"].shape == (100, 1)

    def test_new_state(self):
        """Test new state initialization"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state = v.new_state()
        assert set(state.keys()) == {"var1", "var2", "var3"}
        assert state["var1"].shape == (1,)
        assert state["var2"].shape == (2,)
        assert state["var3"].shape == (1,)

    def test_update_history(self):
        """Test history update mechanism"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state, _ = v.initialize_tensors(100)

        # Update history multiple times
        for i in range(3):
            state["var1"] = torch.ones(1) * i
            state["var2"] = torch.ones(2) * i
            state["var3"] = torch.ones(1) * i
            history = v.update_history(state)

            if i < 2:
                assert len(v.history["var2"]) == i + 1
            else:
                assert len(v.history["var2"]) == 2
                assert len(v.history["var3"]) == 1

            if i > 0:
                assert torch.allclose(history["var2"][0], torch.ones(2) * i)
                assert torch.allclose(history["var3"][0], torch.ones(1) * i)

    def test_record_state(self, caplog):
        """Test recording of state variables"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state, _ = v.initialize_tensors(100)

        # Record state at different timesteps
        for t in range(3):
            state = {
                "var1": torch.ones(1) * t,
                "var2": torch.ones(2) * t,
                "var3": torch.ones(1) * t,
            }
            v.record_state(t, state)

            assert torch.allclose(v.timeseries["var1"][t], torch.ones(1) * t)
            assert torch.allclose(v.timeseries["var2"][t], torch.ones(2) * t)
            assert torch.allclose(v.timeseries["var3"][t], torch.ones(1) * t)

        # Test warning for keys in state but not timeseries
        state["extra_var"] = torch.ones(1)
        v.record_state(3, state)
        assert (
            "keys in state variables but not timeseries: {'extra_var'}" in caplog.text
        )

        # Test error handling for mismatched shapes
        state = {
            "var1": torch.ones(2),  # Wrong shape, should be (1,)
            "var2": torch.ones(2),
            "var3": torch.ones(1),
        }
        with pytest.raises(Exception):
            v.record_state(4, state)

    def test_json_io(self, tmpdir):
        """Test JSON file I/O"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state, _ = v.initialize_tensors(100)

        # Create some test data
        for t in range(3):
            state = {k: torch.ones_like(v) * t for k, v in state.items()}
            v.record_state(t, state)

        # Save to JSON
        json_path = os.path.join(tmpdir, "test_variables.json")
        v.to_json(json_path)

        # Load from JSON
        v2 = Variables.from_json(json_path)

        # Compare timeseries
        for var in v.timeseries:
            assert torch.allclose(v.timeseries[var], v2.timeseries[var])
