import numpy as np
import pandas as pd
import pytest
import torch

from macrostat.diff import (
    JacobianAutograd,
    JacobianComparisonReport,
    JacobianNumerical,
    check_model_differentiability,
    compare_jacobian_dicts,
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


def loss_fn_full_state(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Full State trajectory (T, 2) for analytic log-Jacobian cross-checks.

    Module-level so JacobianNumerical can pickle it for its worker pool.
    """
    return output["State"]


# LINEAR2D default free-parameter values, pinned in setup_method so the tests
# are independent of the model's shared-default mutation across instances.
_LINEAR2D_DEFAULTS = {
    "S1.S1.a": 0.9,
    "S1.S2.a": 0.1,
    "S2.S1.a": -0.2,
    "S2.S2.a": 0.8,
    "S1.x0": 1.0,
    "S2.x0": 0.1,  # non-zero so the default "relative" scheme is valid
}


def _linear2d_analytic_log_jacobian(model, timesteps: int) -> dict[str, np.ndarray]:
    """Closed-form d(A^t x0)/dlog|theta| for LINEAR2D, keyed by full param names.

    Row ``t`` matches recorded State row ``t = A^t x0`` (row 0 = x0). Columns:

    - matrix entry ``a_{rc}``:
      ``a_{rc} * (sum_{k=0}^{t-1} A^k E_{rc} A^{t-1-k}) x0`` (empty sum = 0 at t=0)
    - initial state ``x0_i``: ``x0_i * A^t e_i``

    Ground-truth reference validated against finite differences to ~1.9e-5.
    """
    p = model.parameters
    A = np.array(
        [
            [float(p["S1.S1.a"]), float(p["S1.S2.a"])],
            [float(p["S2.S1.a"]), float(p["S2.S2.a"])],
        ]
    )
    x0 = np.array([float(p["S1.x0"]), float(p["S2.x0"])])
    Apow = [np.linalg.matrix_power(A, t) for t in range(timesteps)]

    out: dict[str, np.ndarray] = {}
    for name, (r, c) in {
        "S1.S1.a": (0, 0),
        "S1.S2.a": (0, 1),
        "S2.S1.a": (1, 0),
        "S2.S2.a": (1, 1),
    }.items():
        E = np.zeros((2, 2))
        E[r, c] = 1.0
        col = np.zeros((timesteps, 2))
        for t in range(1, timesteps):
            s = sum(Apow[k] @ E @ Apow[t - 1 - k] for k in range(t))
            col[t] = float(p[name]) * (s @ x0)
        out[name] = col
    for name, i in {"S1.x0": 0, "S2.x0": 1}.items():
        col = np.zeros((timesteps, 2))
        for t in range(timesteps):
            col[t] = float(p[name]) * Apow[t][:, i]
        out[name] = col
    return out


class TestDiffLinear2D:
    def setup_method(self):
        self.model_cls = get_model("LINEAR2D")
        self.model = self.model_cls()
        # LINEAR2D shares its default parameter dict across instances, so a
        # prior test that mutated a parameter can leak into this fresh model.
        # Pin the known defaults so every test starts from a clean, order-
        # independent state (and the mutation also cleans the shared default).
        for name, value in _LINEAR2D_DEFAULTS.items():
            self.model.parameters[name] = value

    def make_loss_fn(self):
        """Return module-level loss function."""
        return loss_fn_mse

    def test_autograd_vs_numerical_close(self):
        loss_fn = self.make_loss_fn()
        grads_auto = JacobianAutograd(self.model).compute(loss_fn=loss_fn, mode="rev")
        grads_num = JacobianNumerical(
            self.model, epsilon=1e-5, parameter_space="direct"
        ).compute(loss_fn=loss_fn, mode="central")

        assert set(grads_auto.keys()) == set(grads_num.keys())
        for name in grads_auto:
            ga = grads_auto[name]
            gn = grads_num[name]
            assert ga.shape == gn.shape
            # Relative error should be modest for this simple linear system
            denom = torch.maximum(ga.abs(), gn.abs()).clamp_min(1.0)
            rel = (ga - gn).abs() / denom
            assert rel.max().item() < 1e-2

    def test_vectorized_params_expose_scalar_jacobian_keys(self):
        """Assembled matrix/vector leaves decompose to per-scalar Jacobian keys.

        LINEAR2D assembles its parameters into a 2x2 ``a`` matrix and a
        length-2 ``x0`` vector via ``vector_sectors``. The autograd backend
        must still report one gradient per scalar parameter, keyed by the
        free-parameter names, and matching the numerical backend element for
        element (which perturbs those scalars directly).
        """
        loss_fn = self.make_loss_fn()
        free_names = set(self.model.parameters.get_free_param_names())

        grads_auto = JacobianAutograd(self.model).compute(loss_fn=loss_fn, mode="rev")
        assert set(grads_auto.keys()) == free_names
        assert "S1.S2.a" in grads_auto  # off-diagonal entry differentiated on its own

        grads_num = JacobianNumerical(
            self.model, epsilon=1e-5, parameter_space="direct"
        ).compute(loss_fn=loss_fn, mode="central")
        # A transposed decomposition would read leaf[j, i] instead of leaf[i, j]
        # and so swap the two off-diagonal entries. They are distinct, so each
        # autograd entry must sit closer to its own numerical counterpart than
        # to the other one.
        for name, swapped in (("S1.S2.a", "S2.S1.a"), ("S2.S1.a", "S1.S2.a")):
            dist_self = (grads_auto[name] - grads_num[name]).abs().max()
            dist_swapped = (grads_auto[name] - grads_num[swapped]).abs().max()
            assert dist_self < dist_swapped

    def test_checker_reports_small_relative_errors(self):
        report = check_model_differentiability(
            model=self.model,
            loss_fn=self.make_loss_fn(),
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

    @pytest.mark.parametrize("space", ["relative", "log"])
    def test_relative_log_zero_parameter_error(self, space):
        """ "relative" and "log" both reject a zero-valued parameter."""
        # Force a parameter to zero so the guard is exercised for both schemes.
        self.model.parameters[self.model.parameters.get_free_param_names()[0]] = 0.0

        with pytest.raises(ValueError, match="zero parameters"):
            JacobianNumerical(self.model, epsilon=1e-5, parameter_space=space).compute(
                loss_fn=self.make_loss_fn(), mode="central"
            )

    def test_relative_space_parameter_steps(self):
        """ "relative" (renamed old "log") matches "direct" structure."""
        loss_fn = self.make_loss_fn()

        jac_num = JacobianNumerical(
            self.model, epsilon=1e-5, parameter_space="relative"
        )
        grads_rel = jac_num.compute(loss_fn=loss_fn, mode="central")

        # Compare with direct space (should be different but reasonable)
        jac_direct = JacobianNumerical(
            self.model, epsilon=1e-5, parameter_space="direct"
        )
        grads_direct = jac_direct.compute(loss_fn=loss_fn, mode="central")

        # Both should produce same structure
        assert set(grads_rel.keys()) == set(grads_direct.keys())
        for name in grads_rel:
            assert grads_rel[name].shape == grads_direct[name].shape

    def test_non_scalar_loss_function(self):
        """Test non-scalar loss functions."""
        grads_num = JacobianNumerical(self.model, epsilon=1e-5).compute(
            loss_fn=loss_fn_non_scalar_1d, mode="central"
        )
        grads_auto = JacobianAutograd(self.model).compute(
            loss_fn=loss_fn_non_scalar_1d, mode="rev"
        )

        # Check structure matches
        assert set(grads_num.keys()) == set(grads_auto.keys())
        for name in grads_num:
            assert grads_num[name].shape == grads_auto[name].shape
            # Should have shape (*loss_shape, *param_shape)
            # loss_shape is (2,), param_shape depends on parameter
            assert grads_num[name].shape[0] == 2  # First dimension is loss elements

    def test_non_scalar_loss_2d(self):
        """Test 2D loss function output."""
        grads_num = JacobianNumerical(self.model, epsilon=1e-5).compute(
            loss_fn=loss_fn_non_scalar_2d, mode="central"
        )
        grads_auto = JacobianAutograd(self.model).compute(
            loss_fn=loss_fn_non_scalar_2d, mode="rev"
        )

        # Check structure matches
        assert set(grads_num.keys()) == set(grads_auto.keys())
        for name in grads_num:
            assert grads_num[name].shape == grads_auto[name].shape
            # Should have shape (3, 2, *param_shape)
            assert grads_num[name].shape[:2] == (3, 2)

    def test_jacobian_to_tensor(self):
        """Test to_tensor helper."""
        self.model.parameters["S2.x0"] = 0.1  # non-zero so default log-space is valid
        jac_num = JacobianNumerical(self.model, epsilon=1e-5)
        jac_dict = jac_num.compute(loss_fn=self.make_loss_fn(), mode="central")

        # Convert to tensor using class method
        jac_tensor = jac_num.to_tensor(jac_dict)

        # Should be 2D: (loss_elements, num_params)
        assert jac_tensor.ndim == 2
        # Loss is scalar, so first dimension is 1
        assert jac_tensor.shape[0] == 1
        assert jac_tensor.shape[1] == len(jac_dict)

    def test_jacobian_to_tensor_non_scalar(self):
        """Test to_tensor with non-scalar loss."""
        self.model.parameters["S2.x0"] = 0.1  # non-zero so default log-space is valid
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

    def test_log_derivative_matches_analytic(self):
        """Numerical-central and autograd "log" match the closed-form log-Jacobian.

        Both backends are pinned to an independently-derived ground truth
        (not merely to each other), so a wrong theta factor or a row-offset
        error is caught.
        """
        # epsilon=1e-3 is the float32-recommended step; smaller steps suffer
        # cancellation in the finite-difference numerator.
        grads_num = JacobianNumerical(
            self.model, epsilon=1e-3, parameter_space="log"
        ).compute(loss_fn=loss_fn_full_state, mode="central")
        grads_auto = JacobianAutograd(self.model, parameter_space="log").compute(
            loss_fn=loss_fn_full_state, mode="rev"
        )

        timesteps = next(iter(grads_num.values())).shape[0]
        analytic = _linear2d_analytic_log_jacobian(self.model, timesteps)

        for name, ref in analytic.items():
            ref_t = torch.tensor(ref, dtype=grads_num[name].dtype)
            # autograd is near machine-exact for this linear system
            torch.testing.assert_close(grads_auto[name], ref_t, atol=1e-4, rtol=1e-3)
            # numerical central is float32 finite-difference accurate
            torch.testing.assert_close(grads_num[name], ref_t, atol=2e-3, rtol=1e-2)

    def test_log_equals_relative_times_theta(self):
        """ "log" output equals "relative" output scaled by the signed parameter."""
        grads_rel = JacobianNumerical(
            self.model, epsilon=1e-3, parameter_space="relative"
        ).compute(loss_fn=loss_fn_full_state, mode="central")
        grads_log = JacobianNumerical(
            self.model, epsilon=1e-3, parameter_space="log"
        ).compute(loss_fn=loss_fn_full_state, mode="central")

        # The identity log == relative*theta is exact in real arithmetic; in
        # float32 the "relative" denominator theta*(e^eps - e^-eps) carries an
        # exp-cancellation error that "log"'s 2*eps avoids, so allow ~1e-2.
        for name in grads_rel:
            theta = float(self.model.parameters[name])
            torch.testing.assert_close(
                grads_log[name], grads_rel[name] * theta, atol=1e-3, rtol=1e-2
            )

    def test_log_negative_theta_sign_and_warning(self):
        """For theta < 0 the log-derivative carries theta's sign; warning fires."""
        with pytest.warns(UserWarning, match="negative parameters"):
            grads_rel = JacobianNumerical(
                self.model, epsilon=1e-3, parameter_space="relative"
            ).compute(loss_fn=loss_fn_full_state, mode="central")
        with pytest.warns(UserWarning, match="negative parameters"):
            grads_log = JacobianNumerical(
                self.model, epsilon=1e-3, parameter_space="log"
            ).compute(loss_fn=loss_fn_full_state, mode="central")

        theta = float(self.model.parameters["S2.S1.a"])  # a21 = -0.2
        assert theta < 0
        rel = grads_rel["S2.S1.a"]
        log = grads_log["S2.S1.a"]
        torch.testing.assert_close(log, rel * theta, atol=1e-3, rtol=1e-2)
        nz = rel.abs() > 1e-6
        assert torch.all(torch.sign(log[nz]) == -torch.sign(rel[nz]))

    def test_log_forward_backward_smoke(self):
        """Forward/backward "log" pin the epsilon denominator against analytic."""
        timesteps = None
        for mode in ("forward", "backward"):
            grads = JacobianNumerical(
                self.model, epsilon=1e-3, parameter_space="log"
            ).compute(loss_fn=loss_fn_full_state, mode=mode)
            if timesteps is None:
                timesteps = next(iter(grads.values())).shape[0]
                analytic = _linear2d_analytic_log_jacobian(self.model, timesteps)
            for name, ref in analytic.items():
                ref_t = torch.tensor(ref, dtype=grads[name].dtype)
                # first-order (O(eps)) scheme: a wrong (theta-scaled) denominator
                # would be off by ~theta, far outside this band.
                torch.testing.assert_close(grads[name], ref_t, atol=5e-3, rtol=5e-2)

    def test_checker_forwards_log_space_to_both_backends(self):
        """check_model_differentiability in "log" compares both backends in-space."""
        report = check_model_differentiability(
            model=self.model,
            loss_fn=loss_fn_full_state,
            parameter_space="log",
            epsilon=1e-3,
            rtol=5e-2,
            compare_forward_reverse=True,
            compare_numerical=True,
        )
        assert not report.nan_or_inf
        assert report.autodiff_vs_numerical_ok
        assert report.rel_err_autodiff_num is not None
        assert report.rel_err_autodiff_num < 5e-2

    def test_future_warning_on_explicit_log(self):
        """Passing "log" explicitly warns; the default ("relative") does not."""
        with pytest.warns(FutureWarning, match="log"):
            JacobianNumerical(self.model, epsilon=1e-5, parameter_space="log")

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            JacobianNumerical(self.model, epsilon=1e-5)  # default -> relative
            JacobianNumerical(self.model, epsilon=1e-5, parameter_space="relative")


class TestJacobianBase:
    """Test JacobianBase helper methods (to_tensor and to_pandas)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_cls = get_model("LINEAR2D")
        self.model = self.model_cls()
        # S2.x0 is 0 by default; set it non-zero so the default log-space
        # perturbation is valid for these helper tests.
        self.model.parameters["S2.x0"] = 0.1
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
        # No jacobian computed yet, so self.jacobian doesn't exist
        with pytest.raises(ValueError, match="jacobian_dict must be provided"):
            JacobianNumerical(self.model, epsilon=1e-5).to_tensor()

    def test_to_tensor_empty_dict(self):
        """Test to_tensor with empty jacobian_dict."""
        assert self.jac_num.to_tensor(jacobian_dict={}).numel() == 0

    def test_to_tensor_custom_param_order(self):
        """Test to_tensor with custom param_order."""
        param_order = list(reversed(list(self.jacobian_dict.keys())))

        # Verify columns match the specified order
        assert self.jac_num.to_tensor(
            jacobian_dict=self.jacobian_dict, param_order=param_order
        ).shape[1] == len(param_order)

    def test_to_tensor_missing_parameter_in_order(self):
        """Test to_tensor raises error for missing parameter in param_order."""
        with pytest.raises(ValueError, match="Parameter nonexistent_param not found"):
            self.jac_num.to_tensor(
                jacobian_dict=self.jacobian_dict,
                param_order=list(self.jacobian_dict.keys()) + ["nonexistent_param"],
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

    def test_to_tensor_no_flatten_preserves_loss_shape(self):
        """Test to_tensor(flatten=False) keeps the native (T, N) loss shape."""
        jac_dict = self.jac_num.compute(loss_fn=loss_fn_non_scalar_2d, mode="central")
        flat = self.jac_num.to_tensor(jacobian_dict=jac_dict, flatten=True)
        unflat = self.jac_num.to_tensor(jacobian_dict=jac_dict, flatten=False)

        # Loss is (3, 2); flat is (6, P), unflat is (3, 2, P).
        n_params = len(jac_dict)
        assert flat.shape == (6, n_params)
        assert unflat.shape == (3, 2, n_params)
        # The two are related by a reshape (C-order): byte-equal.
        assert torch.equal(unflat, flat.reshape(3, 2, n_params))

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
        with pytest.raises(ValueError, match="jacobian_dict must be provided"):
            JacobianNumerical(self.model, epsilon=1e-5).to_pandas()

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
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_1d, mode="central"
            )
        )

        # Should create flat RangeIndex
        assert len(df.index) == 2  # loss_fn_non_scalar_1d returns shape (2,)
        assert isinstance(df.index, pd.RangeIndex)
        assert df.index.name == "element"

    def test_to_pandas_1d_loss_with_timesteps(self):
        """Test to_pandas with 1D loss and only timesteps provided."""
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_1d, mode="central"
            ),
            timesteps=2,
        )

        # Should create timestep index
        assert len(df.index) == 2
        assert df.index.name == "timestep"
        assert list(df.index) == [0, 1]

    def test_to_pandas_1d_loss_with_variable_names(self):
        """Test to_pandas with 1D loss and only variable_names provided."""
        variable_names = ["var1", "var2"]
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_1d, mode="central"
            ),
            variable_names=variable_names,
        )

        # Should create variable index
        assert len(df.index) == 2
        assert df.index.name == "variable"
        assert list(df.index) == variable_names

    def test_to_pandas_1d_loss_mismatched_timesteps(self):
        """Test to_pandas raises error for mismatched timesteps."""
        with pytest.raises(ValueError, match="does not match shape"):
            self.jac_num.to_pandas(
                jacobian_dict=self.jac_num.compute(
                    loss_fn=loss_fn_non_scalar_1d, mode="central"
                ),
                timesteps=5,
            )

    def test_to_pandas_1d_loss_mismatched_variable_names(self):
        """Test to_pandas raises error for mismatched variable_names."""
        with pytest.raises(ValueError, match="does not match shape"):
            self.jac_num.to_pandas(
                jacobian_dict=self.jac_num.compute(
                    loss_fn=loss_fn_non_scalar_1d, mode="central"
                ),
                variable_names=["var1", "var2", "var3"],
            )

    def test_to_pandas_2d_loss_no_structure(self):
        """Test to_pandas with 2D loss and no structure info."""
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_2d, mode="central"
            )
        )

        # Should infer timesteps and create default variable names
        assert len(df.index) == 6  # 3 timesteps * 2 variables
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["timestep", "variable"]

    def test_to_pandas_2d_loss_with_timesteps_only(self):
        """Test to_pandas with 2D loss and only timesteps provided."""
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_2d, mode="central"
            ),
            timesteps=3,
        )

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
        variable_names = ["x", "y"]
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_2d, mode="central"
            ),
            variable_names=variable_names,
        )

        # Should infer timesteps
        assert len(df.index) == 6
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["timestep", "variable"]
        # Check variable names match
        assert list(df.index.get_level_values("variable").unique()) == variable_names

    def test_to_pandas_2d_loss_with_both(self):
        """Test to_pandas with 2D loss and both timesteps and variable_names."""
        variable_names = ["x", "y"]
        df = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_2d, mode="central"
            ),
            timesteps=3,
            variable_names=variable_names,
        )

        # Should create MultiIndex
        assert len(df.index) == 6
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["timestep", "variable"]
        assert list(df.index.get_level_values("variable").unique()) == variable_names

    def test_to_pandas_2d_loss_mismatched_dimensions(self):
        """Test to_pandas raises error for mismatched 2D dimensions."""
        with pytest.raises(ValueError, match="do not match shape"):
            self.jac_num.to_pandas(
                jacobian_dict=self.jac_num.compute(
                    loss_fn=loss_fn_non_scalar_2d, mode="central"
                ),
                timesteps=5,
                variable_names=["x", "y"],
            )

    def test_to_pandas_3d_loss_raises_error(self):
        """Test to_pandas raises error for 3D+ loss."""
        with pytest.raises(ValueError, match="Unsupported loss shape"):
            self.jac_num.to_pandas(
                jacobian_dict=self.jac_num.compute(loss_fn=loss_fn_3d, mode="central")
            )

    def test_to_pandas_custom_param_order(self):
        """Test to_pandas with custom param_order."""
        param_order = list(reversed(list(self.jacobian_dict.keys())))

        # Verify columns match the specified order
        assert (
            list(
                self.jac_num.to_pandas(
                    jacobian_dict=self.jacobian_dict, param_order=param_order
                ).columns
            )
            == param_order
        )

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
        # Convert DataFrame to numpy and compare
        df_values = self.jac_num.to_pandas(jacobian_dict=self.jacobian_dict).values
        tensor_values = (
            self.jac_num.to_tensor(jacobian_dict=self.jacobian_dict)
            .detach()
            .cpu()
            .numpy()
        )

        # Should match (allowing for floating point differences)
        assert df_values.shape == tensor_values.shape
        assert torch.allclose(torch.tensor(df_values), torch.tensor(tensor_values))

    def test_to_pandas_with_non_scalar_loss(self):
        """Test to_pandas with non-scalar loss (1D and 2D)."""
        # Test 1D
        df_1d = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_1d, mode="central"
            ),
            variable_names=["x", "y"],
        )
        assert len(df_1d.index) == 2
        assert df_1d.index.name == "variable"

        # Test 2D
        df_2d = self.jac_num.to_pandas(
            jacobian_dict=self.jac_num.compute(
                loss_fn=loss_fn_non_scalar_2d, mode="central"
            ),
            timesteps=3,
            variable_names=["x", "y"],
        )
        assert len(df_2d.index) == 6
        assert isinstance(df_2d.index, pd.MultiIndex)


class TestCompareJacobianDicts:
    """Unit tests for compare_jacobian_dicts using synthetic data."""

    def test_identical_dicts(self):
        """Identical Jacobians produce zero diffs and all elements close."""
        jac = {"a": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([4.0, 5.0])}
        report = compare_jacobian_dicts(jac, jac, "X", "Y")

        assert isinstance(report, JacobianComparisonReport)
        assert report.method_a == "X"
        assert report.method_b == "Y"
        assert report.overall_max_abs_diff == 0.0
        assert report.overall_max_rel_diff == 0.0
        for pc in report.per_parameter.values():
            assert pc.max_abs_diff == 0.0
            assert pc.num_close == pc.num_elements
            assert not pc.has_nan_inf

    def test_known_difference(self):
        """Injected perturbation produces expected statistics."""
        jac_a = {"p1": torch.tensor([10.0, 20.0])}
        jac_b = {"p1": torch.tensor([10.1, 20.0])}

        pc = compare_jacobian_dicts(jac_a, jac_b).per_parameter["p1"]

        assert pc.max_abs_diff == pytest.approx(0.1, abs=1e-6)
        assert pc.mean_abs_diff == pytest.approx(0.05, abs=1e-6)
        # rel_diff = 0.1 / max(10.0, 10.1) = 0.1/10.1 ≈ 0.0099
        assert pc.max_rel_diff == pytest.approx(0.1 / 10.1, abs=1e-4)
        assert pc.num_elements == 2

    def test_nan_detection(self):
        """NaN in one dict is flagged in has_nan_inf."""
        jac_a = {"p1": torch.tensor([1.0, float("nan")])}
        jac_b = {"p1": torch.tensor([1.0, 2.0])}

        assert (
            compare_jacobian_dicts(jac_a, jac_b).per_parameter["p1"].has_nan_inf is True
        )

    def test_inf_detection(self):
        """Inf in one dict is flagged in has_nan_inf."""
        jac_a = {"p1": torch.tensor([1.0, float("inf")])}
        jac_b = {"p1": torch.tensor([1.0, 2.0])}

        assert (
            compare_jacobian_dicts(jac_a, jac_b).per_parameter["p1"].has_nan_inf is True
        )

    def test_missing_keys_only_shared(self):
        """Only shared keys are compared; disjoint keys are ignored."""
        jac_a = {"shared": torch.tensor([1.0]), "only_a": torch.tensor([2.0])}
        jac_b = {"shared": torch.tensor([1.0]), "only_b": torch.tensor([3.0])}

        assert set(compare_jacobian_dicts(jac_a, jac_b).per_parameter.keys()) == {
            "shared"
        }

    def test_scalar_jacobian(self):
        """Scalar (0-dim) tensors are handled."""
        jac_a = {"p": torch.tensor(5.0)}
        jac_b = {"p": torch.tensor(5.5)}

        pc = compare_jacobian_dicts(jac_a, jac_b).per_parameter["p"]
        assert pc.num_elements == 1
        assert pc.max_abs_diff == pytest.approx(0.5, abs=1e-6)

    def test_multidim_jacobian(self):
        """2D tensors (T x V) are compared element-wise."""
        jac_a = {"p": torch.ones(10, 5)}
        jac_b = {"p": torch.ones(10, 5) + 0.01}

        pc = compare_jacobian_dicts(jac_a, jac_b).per_parameter["p"]
        assert pc.num_elements == 50
        assert pc.max_abs_diff == pytest.approx(0.01, abs=1e-6)

    def test_worst_parameters_sorting(self):
        """worst_parameters returns parameters sorted by max_rel_diff desc."""
        jac_a = {
            "good": torch.tensor([100.0]),
            "bad": torch.tensor([100.0]),
            "ugly": torch.tensor([100.0]),
        }
        jac_b = {
            "good": torch.tensor([100.001]),  # tiny rel diff
            "bad": torch.tensor([110.0]),  # 10% rel diff
            "ugly": torch.tensor([150.0]),  # 50% rel diff
        }

        worst = compare_jacobian_dicts(jac_a, jac_b).worst_parameters(n=3)
        assert worst[0].name == "ugly"
        assert worst[1].name == "bad"
        assert worst[2].name == "good"

    def test_empty_dicts(self):
        """Empty dicts produce an empty report."""
        report = compare_jacobian_dicts({}, {})
        assert len(report.per_parameter) == 0
        assert report.overall_max_abs_diff == 0.0
        assert report.overall_max_rel_diff == 0.0

    def test_summary_runs(self):
        """summary() returns a non-empty string without errors."""
        jac = {"a": torch.tensor([1.0, 2.0])}
        text = compare_jacobian_dicts(jac, jac, "fwd", "rev").summary()
        assert isinstance(text, str)
        assert "fwd" in text
        assert "rev" in text
        assert "a" in text
