"""
Unit tests for the sampler module
"""

import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from macrostat.sample.sampler import BaseSampler
from tests.sample.conftest import (
    MockModel,  # Use absolute import now that we have __init__.py files
)


@pytest.fixture
def sampler(mock_model, tmp_path):
    return BaseSampler(model=mock_model, output_folder=str(tmp_path), cpu_count=1)


class TestBaseSampler:
    def test_initialization(self, mock_model, mock_parameters, tmp_path):
        """Test initialization of BaseSampler"""
        sampler = BaseSampler(
            model=mock_model,
            output_folder=str(tmp_path),
            cpu_count=2,
            batchsize=10,
            output_filetype="csv",
            output_compression="gzip",
        )

        assert sampler.model == mock_model
        assert sampler.modelclass == MockModel
        assert sampler.base_parameters.is_equal(mock_parameters)
        assert sampler.cpu_count == 2
        assert sampler.batchsize == 10
        assert sampler.output_folder == tmp_path
        assert sampler.output_filetype == "csv"
        assert sampler.output_compression == "gzip"
        assert os.path.exists(tmp_path)

    def test_initialization_default_bounds(self, mock_model, tmp_path):
        """Test initialization with default bounds from model"""
        sampler = BaseSampler(
            model=mock_model, output_folder=str(tmp_path), bounds=None
        )

        assert sampler.bounds == mock_model.parameters.get_bounds()

    def testverify_bounds_valid(self, sampler, valid_bounds):
        """Test bounds verification with valid bounds"""
        # Should not raise any exception
        sampler.verify_bounds(valid_bounds)

    def testverify_bounds_invalid_parameter(self, sampler):
        """Test bounds verification with invalid parameter"""
        invalid_bounds = {"invalid_param": (0.1, 1.0), "param2": (1.0, 10.0)}

        with pytest.raises(
            ValueError, match="Parameter invalid_param not in the model's parameters"
        ):
            sampler.verify_bounds(invalid_bounds)

    def testverify_bounds_invalid_length(self, sampler):
        """Test bounds verification with invalid bound length"""
        invalid_bounds = {
            "param1": (0.1, 1.0, 2.0),  # Three values instead of two
            "param2": (1.0, 10.0),
        }

        with pytest.raises(
            ValueError, match="Bounds should be a list-like of length 2"
        ):
            sampler.verify_bounds(invalid_bounds)

    def testverify_bounds_invalid_order(self, sampler):
        """Test bounds verification with invalid bound order"""
        invalid_bounds = {
            "param1": (1.0, 0.1),  # Upper bound smaller than lower bound
            "param2": (1.0, 10.0),
        }

        with pytest.raises(
            ValueError, match="Lower bound should be smaller than the upper bound"
        ):
            sampler.verify_bounds(invalid_bounds)

    def testverify_bounds_logspace_invalid_signs(self, sampler):
        """Test bounds verification with invalid signs for logspace"""
        sampler.logspace = True
        invalid_bounds = {
            "param1": (-1.0, 1.0),  # Different signs
            "param2": (1.0, 10.0),
        }

        with pytest.raises(
            ValueError, match="Bounds should be either both positive or both negative"
        ):
            sampler.verify_bounds(invalid_bounds)

    def testverify_bounds_logspace_zero(self, sampler):
        """Test bounds verification with zero bounds in logspace"""
        sampler.logspace = True
        invalid_bounds = {"param1": (0.0, 1.0), "param2": (1.0, 10.0)}  # Zero bound

        with pytest.raises(
            ValueError, match="Bounds cannot be zero when using logspace"
        ):
            sampler.verify_bounds(invalid_bounds)

    def test_generate_tasks_not_implemented(self, sampler):
        """Test that generate_tasks raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            sampler.generate_tasks()

    @patch("macrostat.sample.sampler.msbatchprocessing.parallel_processor")
    def test_sample_method_batchsize_none(
        self, mock_parallel_processor, mock_model, tmp_path
    ):
        """Test the sample method with batchsize=None (all tasks in one batch)"""
        # Create sampler with batchsize=None
        sampler = BaseSampler(
            model=mock_model, output_folder=str(tmp_path), cpu_count=1, batchsize=None
        )

        # Create a larger set of tasks
        tasks = [(i, MockModel({"param1": float(i)})) for i in range(5)]
        sampler.generate_tasks = Mock(return_value=tasks)

        # Mock the parallel processor output
        mock_parallel_processor.return_value = [
            (
                i,
                MockModel({"param1": float(i)}),
                pd.DataFrame({"time": [1, 2], "value": [i, i + 1]}).set_index("time"),
            )
            for i in range(5)
        ]

        # Run the sample method
        sampler.sample()

        # Verify that parallel_processor was called exactly once
        assert mock_parallel_processor.call_count == 1

        # Check that parameters were saved
        assert os.path.exists(tmp_path / "parameters.csv")

        # Check that outputs were saved in a single file
        assert os.path.exists(tmp_path / "outputs_0.csv")
        assert not os.path.exists(tmp_path / "outputs_1.csv")  # No second batch file

        # Verify the content of the output file
        df = pd.read_csv(tmp_path / "outputs_0.csv")
        assert len(df) == 10  # 5 models * 2 timepoints

    @patch("macrostat.sample.sampler.msbatchprocessing.parallel_processor")
    def test_sample_method_batchsize_three(
        self, mock_parallel_processor, mock_model, tmp_path
    ):
        """Test the sample method with batchsize=3 (two batches)"""
        # Create sampler with batchsize=3
        sampler = BaseSampler(
            model=mock_model, output_folder=str(tmp_path), cpu_count=1, batchsize=3
        )

        # Create a larger set of tasks
        tasks = [(i, MockModel({"param1": float(i)})) for i in range(5)]
        sampler.generate_tasks = Mock(return_value=tasks)

        # Mock the parallel processor output for each batch
        def mock_processor_side_effect(tasks, *args, **kwargs):
            # Return results based on the task IDs in the batch
            return [
                (
                    i,
                    MockModel({"param1": float(i)}),
                    pd.DataFrame({"time": [1, 2], "value": [i, i + 1]}).set_index(
                        "time"
                    ),
                )
                for i in range(len(tasks))
            ]

        mock_parallel_processor.side_effect = mock_processor_side_effect

        # Run the sample method
        sampler.sample()

        # Verify that parallel_processor was called exactly twice
        assert mock_parallel_processor.call_count == 2

        # Check that parameters were saved
        assert os.path.exists(tmp_path / "parameters.csv")

        # Check that outputs were saved in two files
        assert os.path.exists(tmp_path / "outputs_0.csv")
        assert os.path.exists(tmp_path / "outputs_1.csv")

        # Verify the content of the first batch file (3 tasks)
        df_batch0 = pd.read_csv(tmp_path / "outputs_0.csv")
        assert len(df_batch0) == 6  # 3 models * 2 timepoints

        # Verify the content of the second batch file (2 tasks)
        df_batch1 = pd.read_csv(tmp_path / "outputs_1.csv")
        assert len(df_batch1) == 4  # 2 models * 2 timepoints

        # Verify the total number of rows across all files
        total_rows = len(df_batch0) + len(df_batch1)
        assert total_rows == 10  # 5 models * 2 timepoints

    def test_save_outputs_csv(self, sampler, tmp_path):
        """Test saving outputs in CSV format"""
        # Create sample data
        raw_outputs = [
            (
                0,
                MockModel(),
                pd.DataFrame({"time": [1, 2], "value": [1, 2]}).set_index("time"),
            ),
            (
                1,
                MockModel(),
                pd.DataFrame({"time": [1, 2], "value": [2, 3]}).set_index("time"),
            ),
        ]

        # Save outputs
        sampler.save_outputs(raw_outputs, batch=0)

        # Check file exists
        output_file = tmp_path / "outputs_0.csv"
        assert os.path.exists(output_file)

        # Check content
        df = pd.read_csv(output_file)
        assert len(df) == 4  # 2 models * 2 timepoints

    def test_save_outputs_parquet(self, tmp_path):
        """Test saving outputs in Parquet format"""
        sampler = BaseSampler(
            model=MockModel(), output_folder=str(tmp_path), output_filetype="parquet"
        )

        # Create sample data
        raw_outputs = [
            (
                0,
                MockModel(),
                pd.DataFrame({"time": [1, 2], "value": [1, 2]}).set_index("time"),
            ),
            (
                1,
                MockModel(),
                pd.DataFrame({"time": [1, 2], "value": [2, 3]}).set_index("time"),
            ),
        ]

        # Save outputs
        sampler.save_outputs(raw_outputs, batch=0)

        # Check file exists
        output_file = tmp_path / "outputs_0.parquet"
        assert os.path.exists(output_file)

        # Check content
        df = pd.read_parquet(output_file)
        assert len(df) == 4  # 2 models * 2 timepoints

    def test_invalid_output_filetype(self, tmp_path):
        """Test that invalid output filetype raises ValueError"""
        sampler = BaseSampler(
            model=MockModel(), output_folder=str(tmp_path), output_filetype="invalid"
        )

        raw_outputs = [
            (
                0,
                MockModel(),
                pd.DataFrame({"time": [1, 2], "value": [1, 2]}).set_index("time"),
            )
        ]

        with pytest.raises(ValueError, match="Invalid output filetype"):
            sampler.save_outputs(raw_outputs, batch=0)

    def test_generate_tasks_base_implementation(self, sampler):
        """Test the base implementation of generate_tasks"""
        # Mock the generate_parameters method to return a known set of points
        mock_points = pd.DataFrame({"param1": [0.5, 0.7], "param2": [5.0, 7.0]})
        sampler.generate_parameters = Mock(return_value=mock_points)

        # Generate tasks
        tasks = sampler.generate_tasks()

        # Check basic structure
        assert isinstance(tasks, list)
        assert len(tasks) == 2  # Two points in our mock data

        # Check first task
        task_id, model, *args = tasks[0]
        assert task_id == 0  # First index
        assert isinstance(model, MockModel)
        assert args == list(sampler.simulation_args)

        # Check model parameters
        assert model.parameters["param1"] == 0.5
        assert model.parameters["param2"] == 5.0

        # Check second task
        task_id, model, *args = tasks[1]
        assert task_id == 1  # Second index
        assert isinstance(model, MockModel)
        assert args == list(sampler.simulation_args)

        # Check model parameters
        assert model.parameters["param1"] == 0.7
        assert model.parameters["param2"] == 7.0

    def test_generate_tasks_preserves_base_parameters(self, sampler):
        """Test that generate_tasks preserves parameters not in the points"""
        # Mock the generate_parameters method to return a known set of points
        mock_points = pd.DataFrame({"param1": [0.5, 0.9]})
        sampler.generate_parameters = Mock(return_value=mock_points)

        # Generate tasks
        tasks = sampler.generate_tasks()

        # Check that param1 varies and param2 is fixed
        assert tasks[0][1].parameters["param1"] == 0.5
        assert tasks[0][1].parameters["param2"] == 2.0
        assert tasks[1][1].parameters["param1"] == 0.9
        assert tasks[1][1].parameters["param2"] == 2.0

    def test_generate_tasks_handles_empty_points(self, sampler):
        """Test that generate_tasks handles empty points DataFrame"""
        # Mock the generate_parameters method to return empty DataFrame
        mock_points = pd.DataFrame(columns=["param1", "param2"])
        sampler.generate_parameters = Mock(return_value=mock_points)

        # Generate tasks
        tasks = sampler.generate_tasks()

        # Check that we get an empty list
        assert isinstance(tasks, list)
        assert len(tasks) == 0

    def test_generate_tasks_handles_simulation_args(self, sampler):
        """Test that generate_tasks correctly includes simulation arguments"""
        # Set simulation arguments
        sampler.simulation_args = (1, 2, 3)

        # Mock the generate_parameters method
        mock_points = pd.DataFrame({"param1": [0.5], "param2": [5.0]})
        sampler.generate_parameters = Mock(return_value=mock_points)

        # Generate tasks
        tasks = sampler.generate_tasks()

        # Check that simulation args are included
        task_id, model, *args = tasks[0]
        assert args == [1, 2, 3]
