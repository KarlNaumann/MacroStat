"""
pytest code for the Variables class
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

from macrostat.core import Parameters, Variables


class TestVariables:
    """Tests for the Variables class found in models/variables.py"""

    # Test data
    variable_info = {
        # Wealth (to test that a stock is not included in the transaction matrix)
        "householdWealth": {
            "sectors": ["Household"],
            "history": 0,
            "unit": "currency",
            "notation": "W_h",
            "sfc": [("asset", "Household")],
        },
        "householdMoneyStock": {
            "sectors": ["Household"],
            "history": 0,
            "unit": "currency",
            "notation": "M_h",
            "sfc": [("asset", ["Household", "Current"])],
        },
        # A second stock in the same balance sheet section
        "householdBillStock": {
            "sectors": ["Household"],
            "history": 0,
            "unit": "currency",
            "notation": "B_h",
            "sfc": [("asset", ["Household", "Current"])],
        },
        "firmBillStock": {
            "sectors": ["Firm"],
            "history": 0,
            "unit": "currency",
            "notation": "B_f",
            "sfc": [("asset", ["Firm", "Current"])],
        },
        "householdConsumption": {
            "sectors": ["Household"],
            "history": 2,
            "unit": "currency/period",
            "notation": "C",
            "sfc": [("outflow", "Household")],
        },
        "foreignConsumption": {
            "sectors": ["Foreign"],
            "history": 2,
            "unit": "currency/period",
            "notation": "C_f",
            "sfc": [("outflow", "Foreign")],
        },
        "governmentConsumption": {
            "sectors": ["Government"],
            "history": 2,
            "unit": "currency/period",
            "notation": "C_g",
            "sfc": [("outflow", "Government")],
        },
        "firmProfit": {
            "sectors": ["Firm"],
            "history": 1,
            "unit": "currency/period",
            "notation": "\\Pi",
            "sfc": [("inflow", ["Firm", "Current"])],
        },
        "firmCapitalStock": {
            "sectors": ["Firm"],
            "history": 0,
            "unit": "currency",
            "notation": "K_f",
            "sfc": [("asset", ("Firm", "Capital"))],
        },
        "firmLoans": {
            "sectors": ["Firm"],
            "history": 0,
            "unit": "currency",
            "notation": "L_f",
            "sfc": [("liability", "Firm")],
        },
    }

    params = Parameters(hyperparameters={"sectors": ["Household", "Firm", "Other"]})

    def test_init(self):
        """Test initialization of Variables class"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        assert v.info == self.variable_info
        assert v.parameters == self.params

    ############################################################################
    # SFC Functions
    ############################################################################

    def test_get_stock_variables(self):
        """Test getting stock variables"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        stocks = v.get_stock_variables()
        true_stocks = {
            k
            for k, v in self.variable_info.items()
            if v["sfc"][0][0].lower() in ["asset", "liability"]
        }

        assert set(stocks.keys()) == true_stocks

    def test_get_flow_variables(self):
        """Test getting flow variables"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        flows = v.get_flow_variables()
        true_flows = {
            k
            for k, v in self.variable_info.items()
            if v["sfc"][0][0].lower() in ["inflow", "outflow"]
        }

        assert set(flows.keys()) == true_flows

    def test_get_index_variables(self):
        """Test getting index variables"""
        # Add an index variable to test
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        v.info["priceIndex"] = {
            "sectors": [],
            "history": 0,
            "unit": "index",
            "notation": "P",
            "sfc": [("index", "General")],
        }

        indices = v.get_index_variables()

        # Check that index variable is included
        assert set(indices.keys()) == {"priceIndex"}

    def test_balance_sheet_theoretical_incomplete_info(self):
        """Test theoretical balance sheet generation with incomplete info"""
        v = Variables(
            variable_info=copy.deepcopy(self.variable_info), parameters=self.params
        )
        v.info["householdMoneyStock"].pop("sfc")
        with pytest.raises(ValueError):
            v.balance_sheet_theoretical()

    def test_balance_sheet_theoretical_content(self):
        """Test theoretical balance sheet generation with content math format"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        bs = v.balance_sheet_theoretical()
        assert isinstance(bs, pd.DataFrame)
        assert "Household" in bs.columns
        assert "Firm" in bs.columns
        assert "Total" in bs.columns
        assert "Wealth" in bs.index

    def test_balance_sheet_theoretical_math_format(self):
        """Test theoretical balance sheet generation with math format"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        bs = v.balance_sheet_theoretical(mathfmt="sphinx")
        assert ":math:`+W_h`" in bs.values
        bs = v.balance_sheet_theoretical(mathfmt="latex")
        assert "$+W_h$" in bs.values

    def test_balance_sheet_theoretical_non_camel(self):
        """Test theoretical balance sheet generation with non-camel case"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        bs = v.balance_sheet_theoretical(non_camel_case=True)
        assert "Wealth" in bs.index

    def test_balance_sheet_theoretical_without_items(self):
        """Test theoretical balance sheet generation without items"""
        info = copy.deepcopy(self.variable_info)
        drop_items = []
        for k, v in info.items():
            if v["sfc"][0][0] in ["asset", "liability"]:
                drop_items.append(k)

        for k in drop_items:
            info.pop(k)

        v = Variables(variable_info=info, parameters=self.params)
        bs = v.balance_sheet_theoretical()

        # Check that the balance sheet has all sectors and the total column
        tgtcols = set(self.params.hyper["sectors"]) | {"Total"}
        assert set(bs.columns.get_level_values(0)) == tgtcols
        # Check that the total column sums to zero
        assert np.allclose(bs["Total"].sum(), 0)

    def test_transaction_matrix_theoretical_incomplete_info(self):
        """Test theoretical transaction matrix generation with incomplete info"""
        v = Variables(
            variable_info=copy.deepcopy(self.variable_info), parameters=self.params
        )
        v.info["householdMoneyStock"].pop("sfc")
        with pytest.raises(ValueError):
            v.transaction_matrix_theoretical()

    def test_transaction_matrix_theoretical_content(self):
        """Test theoretical transaction matrix generation with content"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        tm = v.transaction_matrix_theoretical()
        assert isinstance(tm, pd.DataFrame)
        assert "Household" in tm.columns
        assert "Firm" in tm.columns
        assert "Total" in tm.columns
        assert "householdConsumption" in tm.index
        assert "firmProfit" in tm.index
        assert "Change in MoneyStock" in tm.index

    def test_transaction_matrix_theoretical_math_format(self):
        """Test theoretical transaction matrix generation with math format"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        tm = v.transaction_matrix_theoretical(mathfmt="sphinx")
        assert ":math:`-C`" in tm.values
        assert ":math:`+\\Pi`" in tm.values
        tm = v.transaction_matrix_theoretical(mathfmt="latex")
        assert "$-C$" in tm.values
        assert "$+\\Pi$" in tm.values

    def test_transaction_matrix_theoretical_non_camel(self):
        """Test theoretical transaction matrix generation with non-camel case"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        tm = v.transaction_matrix_theoretical(non_camel_case=True)
        assert "household Consumption" in tm.index
        assert "firm Profit" in tm.index

    ############################################################################
    # Comparison Functions
    ############################################################################

    def test_compare_to_variables(self):
        """Test the compare function"""
        # Create two Variables objects with different values
        v1 = Variables(variable_info=self.variable_info, parameters=self.params)
        v2 = Variables(variable_info=self.variable_info, parameters=self.params)

        # Initialize tensors with different values
        v1.initialize_tensors(100)
        v2.initialize_tensors(100)

        # Set different values
        v1.timeseries["householdWealth"][:] = 100
        v2.timeseries["householdWealth"][:] = 200

        # Compare the variables
        diff, rel_diff = v1.compare(v2)

        # Check that the differences are calculated correctly
        assert isinstance(diff, pd.DataFrame)
        assert isinstance(rel_diff, pd.DataFrame)

        assert (diff["householdWealth"] == -100).all().all()  # 100 - 200
        assert (rel_diff["householdWealth"] == -50).all().all()  # (100-200)/200 * 100

        # Test comparing with a DataFrame directly
        df = v2.to_pandas()
        diff2, rel_diff2 = v1.compare(df)
        pd.testing.assert_frame_equal(diff, diff2)
        pd.testing.assert_frame_equal(rel_diff, rel_diff2)

    def test_compare_to_pandas(self):
        """Test the compare function"""
        # Create two Variables objects with different values
        v1 = Variables(variable_info=self.variable_info, parameters=self.params)
        v2 = Variables(variable_info=self.variable_info, parameters=self.params)

        # Initialize tensors with different values
        v1.initialize_tensors(100)
        v2.initialize_tensors(100)

        # Set different values
        v1.timeseries["householdWealth"][:] = 100
        v2.timeseries["householdWealth"][:] = 200

        # Compare the variables
        diff, rel_diff = v1.compare(v2.to_pandas())

        # Check that the differences are calculated correctly
        assert isinstance(diff, pd.DataFrame)
        assert isinstance(rel_diff, pd.DataFrame)
        assert (diff["householdWealth"][0] == -100).all().all()  # 100 - 200
        assert (
            (rel_diff["householdWealth"][0] == -50).all().all()
        )  # (100-200)/200 * 100

        # Test comparing with a DataFrame directly
        df = v2.to_pandas()
        diff2, rel_diff2 = v1.compare(df)
        pd.testing.assert_frame_equal(diff, diff2)
        pd.testing.assert_frame_equal(rel_diff, rel_diff2)

    ############################################################################
    # IO Functions
    ############################################################################

    def test_json_io(self, tmp_path):
        """Test JSON I/O operations"""
        # Create Variables object with some data
        v1 = Variables(variable_info=self.variable_info, parameters=self.params)
        state, _ = v1.initialize_tensors(100)
        state["householdWealth"][:] = 100
        state["householdConsumption"][:] = 50
        state["firmProfit"][:] = 25

        # Save to JSON
        json_path = tmp_path / "variables.json"
        v1.to_json(json_path)

        # Load from JSON
        v2 = Variables.from_json(json_path)

        # Compare the loaded data with original
        pd.testing.assert_frame_equal(v1.to_pandas(), v2.to_pandas())

        # Check specific values
        assert torch.allclose(
            v1.timeseries["householdWealth"], v2.timeseries["householdWealth"]
        )
        assert torch.allclose(
            v1.timeseries["householdConsumption"], v2.timeseries["householdConsumption"]
        )
        assert torch.allclose(v1.timeseries["firmProfit"], v2.timeseries["firmProfit"])

    def test_to_pandas(self):
        """Test conversion to pandas DataFrame"""
        # Create Variables object with some data
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        v.initialize_tensors(100)
        v.timeseries["householdWealth"][:] = 100
        v.timeseries["householdConsumption"][:] = 50
        v.timeseries["firmProfit"][:] = 25

        # Convert to pandas
        df = v.to_pandas()

        # Check that it's a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Check the shape
        assert df.shape == (self.params.hyper["timesteps"], len(self.variable_info))

        # Check the columns
        assert set(df.columns.get_level_values(0)) == set(self.variable_info.keys())

        # Check the values
        assert (df["householdWealth"] == 100).all().all()
        assert (df["householdConsumption"] == 50).all().all()
        assert (df["firmProfit"] == 25).all().all()

    def test_info_to_csv(self, tmp_path):
        """Test info to CSV"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        csv_path = tmp_path / "variables.csv"
        v.info_to_csv(csv_path, sphinx_math=False)
        assert os.path.exists(csv_path)

        # Check the content
        df = pd.read_csv(csv_path, index_col=0)

        # Check if all info is in columns
        info_set = set(
            [i.title() for _, v in self.variable_info.items() for i in v.keys()]
        )
        assert df.shape == (len(self.variable_info), len(info_set))
        assert set(df.columns.tolist()) == info_set

    def test_info_to_csv_sphinx_math(self, tmp_path):
        """Test info to CSV with Sphinx math"""
        # Save and load the CSV
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        csv_path = tmp_path / "variables.csv"
        v.info_to_csv(csv_path, sphinx_math=True)
        df = pd.read_csv(csv_path)

        # Check if all notation is in math format
        assert all(
            [
                i.startswith(":math:`") and i.endswith("`")
                for i in df["Notation"].tolist()
            ]
        )

    ############################################################################
    # General Functions
    ############################################################################

    def test_check_health_default_log_output(self, caplog):
        """Test check health default log output"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        assert v.check_health()
        assert "Check health not implemented for this model" in caplog.text

    def test_get_default_variables(self):
        """Test get default variables: there should be nothing in the core class"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        default_vars = v.get_default_variables()
        assert isinstance(default_vars, dict)
        assert len(default_vars) == 0

    def test_initialize_tensors(self):
        """Test tensor initialization"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state, history = v.initialize_tensors(100)

        # Check state variables
        assert set(state.keys()) == set(self.variable_info.keys())
        for k in state.keys():
            assert state[k].shape == (1,)

        # Check history variables
        true_history = {k for k, v in self.variable_info.items() if v["history"] > 0}
        assert set(history.keys()) == true_history
        # They are initialized as empty lists so no need to check the length

        # Check timeseries initialization
        assert set(v.timeseries.keys()) == set(self.variable_info.keys())
        for k in v.timeseries.keys():
            assert v.timeseries[k].shape == (100, 1)

    def test_new_state(self):
        """Test new state initialization"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state = v.new_state()
        assert set(state.keys()) == set(self.variable_info.keys())
        for k in state.keys():
            assert state[k].shape == (1,)

    def test_update_history(self):
        """Test history update mechanism"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state, _ = v.initialize_tensors(100)

        # Update history multiple times
        for i in range(3):
            state["householdWealth"] = torch.ones(1) * i
            state["householdConsumption"] = torch.ones(1) * i
            state["firmProfit"] = torch.ones(1) * i
            history = v.update_history(state)

            if i < 2:
                assert len(v.history["householdConsumption"]) == i + 1
            else:
                assert len(v.history["householdConsumption"]) == 2
                assert len(v.history["firmProfit"]) == 1

            if i > 0:
                assert torch.allclose(
                    history["householdConsumption"][0], torch.ones(1) * i
                )
                assert torch.allclose(history["firmProfit"][0], torch.ones(1) * i)

    def test_record_state(self, caplog):
        """Test recording of state variables"""
        v = Variables(variable_info=self.variable_info, parameters=self.params)
        state, _ = v.initialize_tensors(100)

        # Record state at different timesteps
        for t in range(3):
            state = {
                "householdWealth": torch.ones(1) * t,
                "householdConsumption": torch.ones(1) * t,
                "firmProfit": torch.ones(1) * t,
            }
            v.record_state(t, state)

            assert torch.allclose(v.timeseries["householdWealth"][t], torch.ones(1) * t)
            assert torch.allclose(
                v.timeseries["householdConsumption"][t], torch.ones(1) * t
            )
            assert torch.allclose(v.timeseries["firmProfit"][t], torch.ones(1) * t)

        # Test warning for keys in state but not timeseries
        state["extra_var"] = torch.ones(1)
        v.record_state(3, state)
        assert (
            "keys in state variables but not timeseries: {'extra_var'}" in caplog.text
        )

        # Test error handling for mismatched shapes
        state = {
            "householdWealth": torch.ones(2),  # Wrong shape, should be (1,)
            "householdConsumption": torch.ones(1),
            "firmProfit": torch.ones(1),
        }
        with pytest.raises(Exception):
            v.record_state(4, state)

    def test_verify_sfc_item_missing_sfc_incorrect_type(self, caplog):
        """Test verification of SFC item: missing SFC"""
        info = copy.deepcopy(self.variable_info)
        info["householdMoneyStock"]["sfc"] = "incorrect"
        v = Variables(variable_info=info, parameters=self.params)

        assert not v._verify_sfc_item(
            v.info["householdMoneyStock"]["sfc"], "householdMoneyStock"
        )
        assert (
            "sfc information for householdMoneyStock is not a valid item" in caplog.text
        )

    def test_verify_sfc_item_missing_sfc_incorrect_length(self, caplog):
        """Test verification of SFC item: missing SFC"""
        info = copy.deepcopy(self.variable_info)
        info["householdMoneyStock"]["sfc"] = ("asset", "Household", "Current")
        v = Variables(variable_info=info, parameters=self.params)

        assert not v._verify_sfc_item(
            v.info["householdMoneyStock"]["sfc"], "householdMoneyStock"
        )
        assert (
            "sfc information for householdMoneyStock is not a valid tuple of length 2: ('asset', 'Household', 'Current')"
            in caplog.text
        )

    def test_verify_sfc_info_correct(self):
        """Test verification of SFC info"""
        info = copy.deepcopy(self.variable_info)
        v = Variables(variable_info=info, parameters=self.params)

        assert v.verify_sfc_info()

    def test_verify_sfc_info_correct_list(self):
        """Test verification of SFC info: list of tuples"""
        info = copy.deepcopy(self.variable_info)
        info["householdMoneyStock"]["sfc"] = [
            ("asset", "Household"),
            ("liability", "Firm"),
        ]
        v = Variables(variable_info=info, parameters=self.params)

        assert v.verify_sfc_info()

    def test_verify_sfc_info_correct_tuple(self):
        """Test verification of SFC info: list of tuples"""
        info = copy.deepcopy(self.variable_info)
        info["householdMoneyStock"]["sfc"] = ("asset", "Household")
        v = Variables(variable_info=info, parameters=self.params)

        assert v.verify_sfc_info()

    def test_verify_sfc_info_incorrect_list(self):
        """Test verification of SFC info: list of tuples"""
        info = copy.deepcopy(self.variable_info)
        info["householdMoneyStock"]["sfc"] = np.array(
            [
                ("asset", "Household"),
                ("liability", "Firm"),
            ]
        )
        v = Variables(variable_info=info, parameters=self.params)

        assert not v.verify_sfc_info()

    def test_verify_sfc_info_missing_sfc(self, caplog):
        """Test verification of SFC info: missing SFC info"""
        info = copy.deepcopy(self.variable_info)
        info["householdMoneyStock"].pop("sfc")
        v = Variables(variable_info=info, parameters=self.params)

        assert not v.verify_sfc_info()
        assert "No SFC information for householdMoneyStock" in caplog.text

    def test_verify_sfc_info_missing_sfc_item(self, caplog):
        """Test verification of SFC info: missing SFC item"""
        info = copy.deepcopy(self.variable_info)
        info["householdMoneyStock"]["sfc"] = (("asset", "Household"), "Current")
        v = Variables(variable_info=info, parameters=self.params)

        assert not v.verify_sfc_info()
        assert (
            "sfc information for householdMoneyStock is not a valid item" in caplog.text
        )

    def test_apply_math_format_incorrect_format(self):
        """Test application of math format: incorrect format"""
        with pytest.raises(ValueError):
            Variables()._apply_math_format(pd.DataFrame(), "incorrect")

    def test_convert_sector_to_tuples_incorrect_type(self):
        """Test conversion of sector to tuples: not a tuple"""
        with pytest.raises(ValueError):
            Variables()._convert_sector_to_tuples(10)

    def test_convert_sector_to_tuples_sector_too_long(self):
        """Test conversion of sector to tuples: sector too long"""
        with pytest.raises(ValueError):
            Variables()._convert_sector_to_tuples(("Household", "Current", "Other"))
