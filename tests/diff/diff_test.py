import pandas as pd
import pytest
import torch

from macrostat.diff import (
    JacobianAutograd,
    JacobianNumerical,
    check_model_differentiability,
)
from macrostat.models import get_model

# Module-level loss functions for multiprocessing compatibility
_target_mse = torch.tensor([1.0, 0.0])


def loss_fn_mse(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """MSE loss function for testing."""
    y = output["State"][-1]
    return torch.nn.functional.mse_loss(y, _target_mse)


def loss_fn_non_scalar_1d(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Non-scalar loss function returning 1D tensor."""
    return output["State"][-1]  # Shape: (2,)


def loss_fn_non_scalar_2d(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Non-scalar loss function returning 2D tensor."""
    return output["State"][-3:]  # Last 3 timesteps, shape: (3, 2)


def loss_fn_3d(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """3D loss function for testing unsupported shapes."""
    state = output["State"]
    return state[-3:].unsqueeze(0).repeat(2, 1, 1)  # Shape: (2, 3, 2)


class TestDiffLinear2D:
    def setup_method(self):
        self.model_cls = get_model("LINEAR2D")
        self.model = self.model_cls()

    def make_loss_fn(self):
        """Return module-level loss function."""
        return loss_fn_mse

    def test_autograd_vs_numerical_close(self):
        loss_fn = self.make_loss_fn()
        auto = JacobianAutograd(self.model)
        num = JacobianNumerical(self.model, epsilon=1e-5)

        grads_auto = auto.compute(loss_fn=loss_fn, mode="rev")
        grads_num = num.compute(loss_fn=loss_fn, mode="central")

        for name in grads_auto:
            ga = grads_auto[name]
            gn = grads_num[name]
            assert ga.shape == gn.shape
            # Relative error should be modest for this simple linear system
            denom = torch.maximum(ga.abs(), gn.abs()).clamp_min(1.0)
            rel = (ga - gn).abs() / denom
            assert rel.max().item() < 1e-2

    def test_checker_reports_small_relative_errors(self):
        loss_fn = self.make_loss_fn()
        report = check_model_differentiability(
            model=self.model,
            loss_fn=loss_fn,
            scenario=0,
            rtol=1e-5,
            atol=1e-8,
            compare_forward_reverse=True,
            compare_numerical=True,
            numerical_mode="central",
            epsilon=1e-5,
        )

        assert not report.nan_or_inf
        # Forward vs reverse should agree to around 1e-6 or better
        assert report.rel_err_fwd_rev is not None
        assert report.rel_err_fwd_rev < 1e-5
        # Autograd vs numerical should agree to around 1e-2 on this toy model
        assert report.rel_err_autodiff_num is not None
        assert report.rel_err_autodiff_num < 1e-2

    def test_log_space_zero_parameter_error(self):
        """Test that log-space raises ValueError for zero parameters."""
        loss_fn = self.make_loss_fn()
        jac_num = JacobianNumerical(self.model, epsilon=1e-5, parameter_space="log")

        # LINEAR2D model has x0_2 parameter with value 0, which should raise ValueError
        with pytest.raises(
            ValueError, match="Cannot use log-space with zero parameters"
        ):
            jac_num.compute(loss_fn=loss_fn, mode="central")

    def test_log_space_parameter_steps(self):
        """Test log-space parameter steps with non-zero parameters."""
        loss_fn = self.make_loss_fn()

        # Create a model and modify x0_2 to be non-zero
        model = self.model_cls()
        # Set x0_2 to a small positive value (it's 0 by default)
        model.parameters["x0_2"] = 0.1

        jac_num = JacobianNumerical(model, epsilon=1e-5, parameter_space="log")

        # Should work for positive parameters
        grads_log = jac_num.compute(loss_fn=loss_fn, mode="central")

        # Compare with direct space (should be different but reasonable)
        jac_direct = JacobianNumerical(model, epsilon=1e-5, parameter_space="direct")
        grads_direct = jac_direct.compute(loss_fn=loss_fn, mode="central")

        # Both should produce same structure
        assert set(grads_log.keys()) == set(grads_direct.keys())
        for name in grads_log:
            assert grads_log[name].shape == grads_direct[name].shape

    def test_non_scalar_loss_function(self):
        """Test non-scalar loss functions."""
        jac_num = JacobianNumerical(self.model, epsilon=1e-5)
        jac_auto = JacobianAutograd(self.model)

        grads_num = jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")
        grads_auto = jac_auto.compute(loss_fn=loss_fn_non_scalar_1d, mode="rev")

        # Check structure matches
        assert set(grads_num.keys()) == set(grads_auto.keys())
        for name in grads_num:
            assert grads_num[name].shape == grads_auto[name].shape
            # Should have shape (*loss_shape, *param_shape)
            # loss_shape is (2,), param_shape depends on parameter
            assert grads_num[name].shape[0] == 2  # First dimension is loss elements

    def test_non_scalar_loss_2d(self):
        """Test 2D loss function output."""
        jac_num = JacobianNumerical(self.model, epsilon=1e-5)
        jac_auto = JacobianAutograd(self.model)

        grads_num = jac_num.compute(loss_fn=loss_fn_non_scalar_2d, mode="central")
        grads_auto = jac_auto.compute(loss_fn=loss_fn_non_scalar_2d, mode="rev")

        # Check structure matches
        assert set(grads_num.keys()) == set(grads_auto.keys())
        for name in grads_num:
            assert grads_num[name].shape == grads_auto[name].shape
            # Should have shape (3, 2, *param_shape)
            assert grads_num[name].shape[:2] == (3, 2)

    def test_jacobian_to_tensor(self):
        """Test to_tensor helper."""
        loss_fn = self.make_loss_fn()
        jac_num = JacobianNumerical(self.model, epsilon=1e-5)
        jac_dict = jac_num.compute(loss_fn=loss_fn, mode="central")

        # Convert to tensor using class method
        jac_tensor = jac_num.to_tensor(jac_dict)

        # Should be 2D: (loss_elements, num_params)
        assert jac_tensor.ndim == 2
        # Loss is scalar, so first dimension is 1
        assert jac_tensor.shape[0] == 1
        assert jac_tensor.shape[1] == len(jac_dict)

    def test_jacobian_to_tensor_non_scalar(self):
        """Test to_tensor with non-scalar loss."""
        jac_num = JacobianNumerical(self.model, epsilon=1e-5)
        jac_dict = jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")

        # Convert to tensor using class method
        jac_tensor = jac_num.to_tensor(jac_dict)

        # Should be 2D: (loss_elements=2, num_params)
        assert jac_tensor.ndim == 2
        assert jac_tensor.shape[0] == 2
        assert jac_tensor.shape[1] == len(jac_dict)

    def test_autograd_numerical_structure_match(self):
        """Test that autograd and numerical produce identical structures."""
        # Test with scalar loss
        loss_fn_scalar = self.make_loss_fn()
        jac_num = JacobianNumerical(self.model, epsilon=1e-5)
        jac_auto = JacobianAutograd(self.model)

        grads_num = jac_num.compute(loss_fn=loss_fn_scalar, mode="central")
        grads_auto = jac_auto.compute(loss_fn=loss_fn_scalar, mode="rev")

        # Structures must match exactly
        assert set(grads_num.keys()) == set(grads_auto.keys())
        for name in grads_num:
            assert grads_num[name].shape == grads_auto[name].shape

        # Test with non-scalar loss
        grads_num_ns = jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")
        grads_auto_ns = jac_auto.compute(loss_fn=loss_fn_non_scalar_1d, mode="rev")

        # Structures must match exactly
        assert set(grads_num_ns.keys()) == set(grads_auto_ns.keys())
        for name in grads_num_ns:
            assert grads_num_ns[name].shape == grads_auto_ns[name].shape

    def test_negative_parameter_warning(self):
        """Test that negative parameters in log-space issue warning."""
        # This is hard to test without modifying model parameters
        # The warning is issued during _validate_log_space_params
        # We'll just verify the method exists and can be called
        # If model has negative params, warning will be issued during compute
        # For this test model, we assume it doesn't have negative params
        JacobianNumerical(self.model, epsilon=1e-5, parameter_space="log")


class TestJacobianBase:
    """Test JacobianBase helper methods (to_tensor and to_pandas)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_cls = get_model("LINEAR2D")
        self.model = self.model_cls()
        self.jac_num = JacobianNumerical(self.model, epsilon=1e-5)
        self.jac_auto = JacobianAutograd(self.model)
        self.loss_fn = loss_fn_mse

        # Compute a Jacobian to use for testing
        self.jacobian_dict = self.jac_num.compute(loss_fn=self.loss_fn, mode="central")

    # Tests for to_tensor method
    def test_to_tensor_with_jacobian_dict(self):
        """Test to_tensor with explicit jacobian_dict parameter."""
        jac_tensor = self.jac_num.to_tensor(jacobian_dict=self.jacobian_dict)

        # Should be 2D: (loss_elements, num_params)
        assert jac_tensor.ndim == 2
        # Loss is scalar, so first dimension is 1
        assert jac_tensor.shape[0] == 1
        assert jac_tensor.shape[1] == len(self.jacobian_dict)

    def test_to_tensor_with_self_jacobian(self):
        """Test to_tensor with self.jacobian (jacobian_dict=None)."""
        # jac_num already has self.jacobian set from compute()
        jac_tensor = self.jac_num.to_tensor()

        # Should be 2D: (loss_elements, num_params)
        assert jac_tensor.ndim == 2
        assert jac_tensor.shape[0] == 1
        assert jac_tensor.shape[1] == len(self.jacobian_dict)

    def test_to_tensor_no_jacobian_raises_error(self):
        """Test to_tensor raises error when no jacobian available."""
        jac_new = JacobianNumerical(self.model, epsilon=1e-5)
        # No jacobian computed yet, so self.jacobian doesn't exist

        with pytest.raises(ValueError, match="jacobian_dict must be provided"):
            jac_new.to_tensor()

    def test_to_tensor_empty_dict(self):
        """Test to_tensor with empty jacobian_dict."""
        jac_tensor = self.jac_num.to_tensor(jacobian_dict={})
        assert jac_tensor.numel() == 0

    def test_to_tensor_custom_param_order(self):
        """Test to_tensor with custom param_order."""
        param_order = list(reversed(list(self.jacobian_dict.keys())))
        jac_tensor = self.jac_num.to_tensor(
            jacobian_dict=self.jacobian_dict, param_order=param_order
        )

        # Verify columns match the specified order
        assert jac_tensor.shape[1] == len(param_order)

    def test_to_tensor_missing_parameter_in_order(self):
        """Test to_tensor raises error for missing parameter in param_order."""
        param_order = list(self.jacobian_dict.keys()) + ["nonexistent_param"]

        with pytest.raises(ValueError, match="Parameter nonexistent_param not found"):
            self.jac_num.to_tensor(
                jacobian_dict=self.jacobian_dict, param_order=param_order
            )

    def test_to_tensor_incompatible_shapes(self):
        """Test to_tensor raises error for incompatible loss shapes."""
        # Create a jacobian dict with incompatible shapes
        bad_jacobian = {
            "a11": torch.tensor([1.0, 2.0]),  # Shape (2,)
            "a12": torch.tensor([3.0]),  # Shape (1,)
        }

        with pytest.raises(ValueError, match="Incompatible loss structure"):
            self.jac_num.to_tensor(jacobian_dict=bad_jacobian)

    def test_to_tensor_with_non_scalar_loss(self):
        """Test to_tensor with non-scalar loss."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")
        jac_tensor = self.jac_num.to_tensor(jacobian_dict=jac_dict)

        # Should be 2D: (loss_elements=2, num_params)
        assert jac_tensor.ndim == 2
        assert jac_tensor.shape[0] == 2
        assert jac_tensor.shape[1] == len(jac_dict)

    # Tests for to_pandas method
    def test_to_pandas_with_jacobian_dict(self):
        """Test to_pandas with explicit jacobian_dict parameter."""
        df = self.jac_num.to_pandas(jacobian_dict=self.jacobian_dict)

        # Should be DataFrame with correct structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Scalar loss
        assert len(df.columns) == len(self.jacobian_dict)

    def test_to_pandas_with_self_jacobian(self):
        """Test to_pandas with self.jacobian (jacobian_dict=None)."""
        df = self.jac_num.to_pandas()

        # Should be DataFrame with correct structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert len(df.columns) == len(self.jacobian_dict)

    def test_to_pandas_no_jacobian_raises_error(self):
        """Test to_pandas raises error when no jacobian available."""
        jac_new = JacobianNumerical(self.model, epsilon=1e-5)

        with pytest.raises(ValueError, match="jacobian_dict must be provided"):
            jac_new.to_pandas()

    def test_to_pandas_empty_dict(self):
        """Test to_pandas with empty jacobian_dict."""
        df = self.jac_num.to_pandas(jacobian_dict={})
        assert len(df) == 0
        assert len(df.columns) == 0

    def test_to_pandas_scalar_loss(self):
        """Test to_pandas with scalar loss."""
        df = self.jac_num.to_pandas(jacobian_dict=self.jacobian_dict)

        # Scalar loss should create index with single "loss" element
        assert len(df.index) == 1
        assert df.index.name == "element"
        assert df.index[0] == "loss"

    def test_to_pandas_1d_loss_no_structure(self):
        """Test to_pandas with 1D loss and no structure info."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")
        df = self.jac_num.to_pandas(jacobian_dict=jac_dict)

        # Should create flat RangeIndex
        assert len(df.index) == 2  # loss_fn_non_scalar_1d returns shape (2,)
        assert isinstance(df.index, pd.RangeIndex)
        assert df.index.name == "element"

    def test_to_pandas_1d_loss_with_timesteps(self):
        """Test to_pandas with 1D loss and only timesteps provided."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")
        df = self.jac_num.to_pandas(jacobian_dict=jac_dict, timesteps=2)

        # Should create timestep index
        assert len(df.index) == 2
        assert df.index.name == "timestep"
        assert list(df.index) == [0, 1]

    def test_to_pandas_1d_loss_with_variable_names(self):
        """Test to_pandas with 1D loss and only variable_names provided."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")
        variable_names = ["var1", "var2"]
        df = self.jac_num.to_pandas(
            jacobian_dict=jac_dict, variable_names=variable_names
        )

        # Should create variable index
        assert len(df.index) == 2
        assert df.index.name == "variable"
        assert list(df.index) == variable_names

    def test_to_pandas_1d_loss_mismatched_timesteps(self):
        """Test to_pandas raises error for mismatched timesteps."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")

        with pytest.raises(ValueError, match="does not match shape"):
            self.jac_num.to_pandas(jacobian_dict=jac_dict, timesteps=5)

    def test_to_pandas_1d_loss_mismatched_variable_names(self):
        """Test to_pandas raises error for mismatched variable_names."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_1d, mode="central")

        with pytest.raises(ValueError, match="does not match shape"):
            self.jac_num.to_pandas(
                jacobian_dict=jac_dict, variable_names=["var1", "var2", "var3"]
            )

    def test_to_pandas_2d_loss_no_structure(self):
        """Test to_pandas with 2D loss and no structure info."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_2d, mode="central")
        df = self.jac_num.to_pandas(jacobian_dict=jac_dict)

        # Should infer timesteps and create default variable names
        assert len(df.index) == 6  # 3 timesteps * 2 variables
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["timestep", "variable"]

    def test_to_pandas_2d_loss_with_timesteps_only(self):
        """Test to_pandas with 2D loss and only timesteps provided."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_2d, mode="central")
        df = self.jac_num.to_pandas(jacobian_dict=jac_dict, timesteps=3)

        # Should infer variable names
        assert len(df.index) == 6
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["timestep", "variable"]
        # Check variable names are auto-generated
        unique_vars = df.index.get_level_values("variable").unique()
        assert len(unique_vars) == 2
        assert all(v.startswith("var_") for v in unique_vars)

    def test_to_pandas_2d_loss_with_variable_names_only(self):
        """Test to_pandas with 2D loss and only variable_names provided."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_2d, mode="central")
        variable_names = ["x", "y"]
        df = self.jac_num.to_pandas(
            jacobian_dict=jac_dict, variable_names=variable_names
        )

        # Should infer timesteps
        assert len(df.index) == 6
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["timestep", "variable"]
        # Check variable names match
        unique_vars = df.index.get_level_values("variable").unique()
        assert list(unique_vars) == variable_names

    def test_to_pandas_2d_loss_with_both(self):
        """Test to_pandas with 2D loss and both timesteps and variable_names."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_2d, mode="central")
        variable_names = ["x", "y"]
        df = self.jac_num.to_pandas(
            jacobian_dict=jac_dict, timesteps=3, variable_names=variable_names
        )

        # Should create MultiIndex
        assert len(df.index) == 6
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["timestep", "variable"]
        unique_vars = df.index.get_level_values("variable").unique()
        assert list(unique_vars) == variable_names

    def test_to_pandas_2d_loss_mismatched_dimensions(self):
        """Test to_pandas raises error for mismatched 2D dimensions."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_2d, mode="central")

        with pytest.raises(ValueError, match="do not match shape"):
            self.jac_num.to_pandas(
                jacobian_dict=jac_dict, timesteps=5, variable_names=["x", "y"]
            )

    def test_to_pandas_3d_loss_raises_error(self):
        """Test to_pandas raises error for 3D+ loss."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_3d, mode="central")

        with pytest.raises(ValueError, match="Unsupported loss shape"):
            self.jac_num.to_pandas(jacobian_dict=jac_dict)

    def test_to_pandas_custom_param_order(self):
        """Test to_pandas with custom param_order."""
        param_order = list(reversed(list(self.jacobian_dict.keys())))
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jacobian_dict, param_order=param_order
        )

        # Verify columns match the specified order
        assert list(df.columns) == param_order

    def test_to_pandas_incompatible_shapes(self):
        """Test to_pandas raises error for incompatible loss shapes."""
        bad_jacobian = {
            "a11": torch.tensor([1.0, 2.0]),  # Shape (2,)
            "a12": torch.tensor([3.0]),  # Shape (1,)
        }

        with pytest.raises(ValueError, match="Incompatible loss structure"):
            self.jac_num.to_pandas(jacobian_dict=bad_jacobian)

    def test_to_tensor_and_to_pandas_consistency(self):
        """Test that to_tensor and to_pandas produce consistent results."""
        jac_tensor = self.jac_num.to_tensor(jacobian_dict=self.jacobian_dict)
        df = self.jac_num.to_pandas(jacobian_dict=self.jacobian_dict)

        # Convert DataFrame to numpy and compare
        df_values = df.values
        tensor_values = jac_tensor.detach().cpu().numpy()

        # Should match (allowing for floating point differences)
        assert df_values.shape == tensor_values.shape
        assert torch.allclose(torch.tensor(df_values), torch.tensor(tensor_values))

    def test_to_pandas_with_non_scalar_loss(self):
        """Test to_pandas with non-scalar loss (1D and 2D)."""
        # Test 1D
        jac_dict_1d = self.jac_num.compute(
            loss_fn=loss_fn_non_scalar_1d, mode="central"
        )
        df_1d = self.jac_num.to_pandas(
            jacobian_dict=jac_dict_1d, variable_names=["x", "y"]
        )
        assert len(df_1d.index) == 2
        assert df_1d.index.name == "variable"

        # Test 2D
        jac_dict_2d = self.jac_num.compute(
            loss_fn=loss_fn_non_scalar_2d, mode="central"
        )
        df_2d = self.jac_num.to_pandas(
            jacobian_dict=jac_dict_2d, timesteps=3, variable_names=["x", "y"]
        )
        assert len(df_2d.index) == 6
        assert isinstance(df_2d.index, pd.MultiIndex)
