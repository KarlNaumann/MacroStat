"""
Unit tests for the sobol sampler module
"""

import os

import pandas as pd
import pytest

from macrostat.sample.sobol import SobolSampler


@pytest.fixture
def sampler(mock_model, valid_bounds, tmp_output_dir):
    return SobolSampler(
        model=mock_model,
        bounds=valid_bounds,
        output_folder=str(tmp_output_dir),
        cpu_count=1,
    )


class TestSobolSampler:
    def test_initialization(self, mock_model, valid_bounds, tmp_output_dir):
        """Test initialization of SobolSampler"""
        sampler = SobolSampler(
            model=mock_model,
            bounds=valid_bounds,
            sample_power=8,
            logspace=True,
            sobol_seed=42,
            simulation_args=(1, 2),
            output_folder=str(tmp_output_dir),
            cpu_count=2,
            batchsize=10,
        )

        assert sampler.model == mock_model
        assert sampler.bounds == valid_bounds
        assert sampler.sample_power == 8
        assert sampler.logspace is True
        assert sampler.sobol_seed == 42
        assert sampler.simulation_args == (1, 2)
        assert sampler.output_folder == tmp_output_dir
        assert sampler.cpu_count == 2
        assert sampler.batchsize == 10
        assert os.path.exists(tmp_output_dir)

    def test_generate_sobol_points(self, sampler):
        """Test generation of Sobol points"""
        points = sampler.generate_parameters()

        # Check DataFrame structure
        assert isinstance(points, pd.DataFrame)
        assert set(points.columns) == set(sampler.bounds.keys())
        assert len(points) == 2**sampler.sample_power

        # Check bounds
        for param, (lower, upper) in sampler.bounds.items():
            assert points[param].min() >= lower
            assert points[param].max() <= upper

    def test_generate_sobol_points_logspace(
        self, mock_model, valid_bounds, tmp_output_dir
    ):
        """Test generation of Sobol points in logspace"""
        sampler = SobolSampler(
            model=mock_model,
            bounds=valid_bounds,
            logspace=True,
            output_folder=str(tmp_output_dir),
        )

        points = sampler.generate_parameters()

        # Check DataFrame structure
        assert isinstance(points, pd.DataFrame)
        assert set(points.columns) == set(sampler.bounds.keys())
        assert len(points) == 2**sampler.sample_power

        # Check bounds
        for param, (lower, upper) in sampler.bounds.items():
            assert points[param].min() >= lower
            assert points[param].max() <= upper
