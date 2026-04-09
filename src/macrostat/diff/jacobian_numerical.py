"""
Numerical Jacobian computation using finite differences.
"""

import copy
import logging
import warnings
from typing import Callable, Dict, Literal

import torch
import torch.multiprocessing as mp

from macrostat.diff.jacobian_base import JacobianBase
from macrostat.util.batchprocessing import parallel_processor

logger = logging.getLogger(__name__)

LossFn = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


def jacobian_worker(task):
    """Worker function for parallel Jacobian computation.

    Parameters
    ----------
    task : tuple
        Task tuple: (model, loss_fn, task_id, scenario)
        - model: MacroStat Model instance with perturbed parameters
        - loss_fn: Loss function
        - task_id: Tuple (param_name, direction)
        - scenario: Scenario index or name to use

    Returns
    -------
    tuple
        (*task_id, loss) where loss is the output from loss_fn
    """
    model, loss_fn, task_id, scenario = task
    try:
        output = model.simulate(scenario=scenario)
        loss = loss_fn(output)
        if isinstance(loss, torch.Tensor):
            loss = loss.detach()

        return (*task_id, loss)
    except Exception as e:
        logger.error(f"Worker failed for task {task_id}: {str(e)}")
        raise


class JacobianNumerical(JacobianBase):
    """
    Compute Jacobians using numerical finite differences.

    This class supports:
    - Direct and log-space parameter perturbations
    - Multiprocessing for large-scale models
    - Non-scalar loss functions (e.g., time series outputs)
    """

    def __init__(
        self,
        model,
        scenario: int | str = 0,
        epsilon: float = 1e-3,
        parameter_space: Literal["direct", "log"] = "log",
    ):
        """
        Initialize numerical Jacobian computation.

        Parameters
        ----------
        model : Model
            MacroStat model instance
        scenario : int | str, optional
            Scenario to use for computation, by default 0
        epsilon : float, optional
            Perturbation size for finite differences, by default 1e-3.
            In log-space, this gives a relative perturbation of ~0.1%,
            which balances truncation error against float32 noise for
            typical SFC models running 50-200 timesteps.

            Epsilon guidance (log-space, float32):
            - 1e-3: Best general-purpose choice. Verified accurate for
              parameters spanning 4 orders of magnitude (2e-4 to 1.0).
            - 1e-4: Better for smooth, weakly-nonlinear parameters but
              noisier for small parameters (< 1e-3).
            - 1e-2: More robust to noise but higher truncation error
              for strongly nonlinear parameters.

            For float64 computation, 1e-5 to 1e-7 are viable.
        parameter_space : {"direct", "log"}, optional
            Space in which to apply perturbations, by default "log".
            Log-space (p -> p*exp(±eps)) gives scale-invariant relative
            perturbations, avoiding the problem where a fixed eps is too
            large for small parameters and too small for large ones.
            Use "direct" only for parameters that are exactly zero or
            when you need additive perturbations for a specific reason.
        """
        super().__init__(model, scenario)
        self.epsilon = epsilon
        self.parameter_space = parameter_space

    def _validate_log_space_params(self, param_names: list[str]):
        """Validate parameters for log-space computation.

        Parameters
        ----------
        param_names : list[str]
            List of parameter names to validate.

        Raises
        ------
        ValueError
            If any zero parameters found.

        Warns
        -----
        UserWarning
            If any negative parameters found.
        """
        zero_params = []
        negative_params = []

        for name in param_names:
            value = self.model.parameters[name]

            if value == 0:
                zero_params.append(name)
            if value < 0:
                negative_params.append(name)

        if zero_params:
            raise ValueError(
                f"Cannot use log-space with zero parameters: {zero_params}"
            )

        if negative_params:
            warnings.warn(
                f"Found negative parameters in log-space mode: {negative_params}. "
                "Using sign-preserving transformation: sign(p) * exp(log(abs(p)) + epsilon)"
            )

    def _apply_perturbation(
        self,
        param_value: float,
        direction: Literal["pos", "neg"],
    ) -> float:
        """Apply perturbation to a scalar parameter value.

        Parameters
        ----------
        param_value : float
            Original parameter value (scalar)
        direction : {"pos", "neg"}
            Direction of perturbation

        Returns
        -------
        float
            Perturbed parameter value
        """
        # Convert to tensor for computation
        value_tensor = torch.tensor(param_value)

        # Apply perturbation
        if self.parameter_space == "log":
            # Log-space perturbation
            sign = torch.sign(value_tensor)
            abs_value = torch.abs(value_tensor)
            log_value = torch.log(abs_value)

            if direction == "pos":
                new_value = sign * torch.exp(log_value + self.epsilon)
            else:  # neg
                new_value = sign * torch.exp(log_value - self.epsilon)
        else:
            # Direct-space perturbation
            if direction == "pos":
                new_value = value_tensor + self.epsilon
            else:  # neg
                new_value = value_tensor - self.epsilon

        return new_value.item()

    def _generate_tasks(
        self,
        param_names: list[str],
        loss_fn: LossFn,
        mode: Literal["central", "forward", "backward"],
    ) -> list:
        """Generate tasks for parallel processing.

        Parameters
        ----------
        param_names : list[str]
            List of parameter names to differentiate
        loss_fn : callable
            Loss function
        mode : {"central", "forward", "backward"}
            Finite difference mode

        Returns
        -------
        list
            List of task tuples: (model, loss_fn, task_id, scenario)
        """
        tasks = []

        if self.parameter_space == "log":
            self._validate_log_space_params(param_names)

        for param_name in param_names:
            base_value = self.model.parameters[param_name]

            if mode in {"central", "forward"}:
                # Positive perturbation
                par_pos = copy.deepcopy(self.model.parameters)
                perturbed_value = self._apply_perturbation(base_value, "pos")
                par_pos[param_name] = perturbed_value
                # Create new Scenarios with modified parameters and copy custom scenarios
                sce_pos = self.model.scenarios.__class__(parameters=par_pos)
                # Copy all custom scenarios (non-default) from original model
                for sc_id, sc_name in self.model.scenarios.info.items():
                    if sc_id != 0:  # Skip default scenario
                        ts_copy = copy.deepcopy(self.model.scenarios.timeseries[sc_id])
                        sce_pos.add_scenario(timeseries=ts_copy, name=sc_name["Name"])
                model_pos = self.model.__class__(parameters=par_pos, scenarios=sce_pos)
                task_id_pos = (param_name, "pos")
                tasks.append((model_pos, loss_fn, task_id_pos, self.scenario))

            if mode in {"central", "backward"}:
                # Negative perturbation
                par_neg = copy.deepcopy(self.model.parameters)
                perturbed_value = self._apply_perturbation(base_value, "neg")
                par_neg[param_name] = perturbed_value
                # Create new Scenarios with modified parameters and copy custom scenarios
                sce_neg = self.model.scenarios.__class__(parameters=par_neg)
                # Copy all custom scenarios (non-default) from original model
                for sc_id, sc_name in self.model.scenarios.info.items():
                    if sc_id != 0:  # Skip default scenario
                        ts_copy = copy.deepcopy(self.model.scenarios.timeseries[sc_id])
                        sce_neg.add_scenario(timeseries=ts_copy, name=sc_name["Name"])
                model_neg = self.model.__class__(parameters=par_neg, scenarios=sce_neg)
                task_id_neg = (param_name, "neg")
                tasks.append((model_neg, loss_fn, task_id_neg, self.scenario))

        return tasks

    def compute(
        self,
        loss_fn: LossFn,
        mode: Literal["central", "forward", "backward"] = "central",
        num_workers: int | None = None,
        progress_bar: bool = False,
        param_names: list[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the numerical Jacobian of the loss w.r.t. model parameters.

        Parameters
        ----------
        loss_fn : callable
            Loss function that takes model output dict and returns a tensor
        mode : {"central", "forward", "backward"}, optional
            Finite difference mode, by default "central"
        num_workers : int, optional
            Number of parallel workers, by default uses all CPUs
        progress_bar : bool, optional
            Whether to show progress bar, by default False
        param_names : list[str], optional
            List of parameter names to differentiate. If None, differentiates
            all parameters in model.parameters.values, by default None

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping parameter names to their Jacobian tensors.
            Each tensor has shape (*loss_shape, *param_shape).
        """
        if mode not in {"central", "forward", "backward"}:
            raise ValueError(
                f"Unsupported mode '{mode}'. Use 'central', 'forward', or 'backward'."
            )

        if param_names is None:
            param_names = self.model.parameters.get_free_param_names()

        output_base = self.model.simulate(scenario=self.scenario)
        loss_base = loss_fn(output_base)

        tasks = self._generate_tasks(param_names, loss_fn, mode)

        if not tasks:
            jacobian = {}
            return jacobian

        if num_workers is None:
            num_workers = mp.cpu_count()

        results = parallel_processor(
            tasks=tasks,
            worker=jacobian_worker,
            cpu_count=num_workers,
            sharing_strategy=None,  # No sharing - atomic tensors
            progress_bar=progress_bar,
        )

        results_by_param: Dict[str, Dict[str, torch.Tensor]] = {}
        for result in results:
            param_name, direction, loss_value = result
            if param_name not in results_by_param:
                results_by_param[param_name] = {}
            if not isinstance(loss_value, torch.Tensor):
                loss_value = torch.tensor(loss_value)
            results_by_param[param_name][direction] = loss_value

        jacobian: Dict[str, torch.Tensor] = {}
        for param_name in param_names:
            losses = results_by_param.get(param_name, {})
            base_value = self.model.parameters[param_name]

            # Compute the actual perturbation in parameter space.
            # In direct space: delta = epsilon.
            # In log space: p+ = p*exp(eps), p- = p*exp(-eps),
            #   so delta_central = p*(exp(eps) - exp(-eps)),
            #      delta_fwd    = p*(exp(eps) - 1),
            #      delta_bwd    = p*(1 - exp(-eps)).
            if self.parameter_space == "log" and base_value != 0:
                # p+ = p*exp(eps), p- = p*exp(-eps), so:
                # p+ - p- = p*(exp(eps) - exp(-eps))  [signed]
                exp_pos = torch.exp(torch.tensor(self.epsilon)).item()
                exp_neg = torch.exp(torch.tensor(-self.epsilon)).item()
                delta_central = base_value * (exp_pos - exp_neg)
                delta_fwd = base_value * (exp_pos - 1.0)
                delta_bwd = base_value * (1.0 - exp_neg)
            else:
                delta_central = 2.0 * self.epsilon
                delta_fwd = self.epsilon
                delta_bwd = self.epsilon

            if all([mode == "central", "pos" in losses, "neg" in losses]):
                grad = (losses["pos"] - losses["neg"]) / delta_central
            elif mode == "forward" and "pos" in losses:
                grad = (losses["pos"] - loss_base) / delta_fwd
            elif mode == "backward" and "neg" in losses:
                grad = (loss_base - losses["neg"]) / delta_bwd
            else:
                grad = torch.zeros_like(loss_base)

            jacobian[param_name] = grad

        self.jacobian = jacobian
        return jacobian
