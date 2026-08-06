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
    - Three perturbation schemes: "direct", "relative", and "log"
    - Multiprocessing for large-scale models
    - Non-scalar loss functions (e.g., time series outputs)
    """

    def __init__(
        self,
        model,
        scenario: int | str = 0,
        epsilon: float = 1e-3,
        parameter_space: Literal["direct", "relative", "log"] | None = None,
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
            In the "relative" and "log" schemes this gives a fractional
            perturbation of ~0.1%, which balances truncation error against
            float32 noise for typical SFC models running 50-200 timesteps.

            Epsilon guidance (relative/log scheme, float32):
            - 1e-3: Best general-purpose choice. Verified accurate for
              parameters spanning 4 orders of magnitude (2e-4 to 1.0).
            - 1e-4: Better for smooth, weakly-nonlinear parameters but
              noisier for small parameters (< 1e-3).
            - 1e-2: More robust to noise but higher truncation error
              for strongly nonlinear parameters.

            For float64 computation, 1e-5 to 1e-7 are viable.
        parameter_space : {"direct", "relative", "log"}, optional
            Scheme for perturbing parameters, by default "relative". The
            choice sets both the step size and the returned derivative:

            - "direct": additive step p -> p +/- eps; returns df/dp.
            - "relative": fractional step p -> p*exp(+/-eps); returns df/dp.
              Scale-invariant, so one eps suits parameters of very different
              magnitude. Recommended default.
            - "log": same fractional step as "relative", but returns the
              logarithmic derivative df/dlog|p| = p * df/dp. Differentiates
              w.r.t. log|p| (not log p, which is complex for p < 0); the sign
              of p is held fixed, so the result carries the sign of p and,
              for p < 0, is opposite in sign to df/d|p|.

            "relative" and "log" are undefined for p = 0 and raise; use
            "direct" in that case.

        Notes
        -----
        Passing ``parameter_space="log"`` explicitly emits a ``FutureWarning``:
        before v0.7.0 the "log" scheme returned the direct derivative df/dp
        (that behaviour is now named "relative"). Pass "relative" to keep the
        old numbers.
        """
        super().__init__(model, scenario)
        self.epsilon = epsilon

        # ``None`` is the silent default (-> "relative"); only an explicit
        # "log" string trips the deprecation warning about the semantic change.
        if parameter_space == "log":
            warnings.warn(
                "parameter_space='log' now returns the logarithmic derivative "
                "df/dlog|p| = p * df/dp. Before v0.7.0 it returned the direct "
                "derivative df/dp; pass parameter_space='relative' for that "
                "behaviour.",
                FutureWarning,
                stacklevel=2,
            )
        if parameter_space is None:
            parameter_space = "relative"
        if parameter_space not in {"direct", "relative", "log"}:
            raise ValueError(
                f"Unsupported parameter_space '{parameter_space}'. "
                "Use 'direct', 'relative', or 'log'."
            )

        self.parameter_space = parameter_space
        # Two independent axes resolved once, so the shared "2*eps" denominator
        # of "direct" and "log" can never be edited into each other's branch:
        #   _relative_step -> which perturbation _apply_perturbation applies
        #   _log_output    -> which derivative the denominator returns
        self._relative_step = parameter_space in {"relative", "log"}
        self._log_output = parameter_space == "log"

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
        if self._relative_step:
            # Fractional (relative) step, shared by "relative" and "log".
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

        # "relative" and "log" both step through log|p|, so a zero parameter
        # has no valid perturbation: numerical raises (it cannot subset the
        # column out the way autograd zero-fills it).
        if self._relative_step:
            self._validate_relative_space_params(param_names, raise_on_zero=True)

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

        self.output_base = self.model.simulate(scenario=self.scenario)
        self.loss_base = loss_fn(self.output_base)

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

            # Denominator = the step in the variable we differentiate against.
            if self._log_output:
                # Differentiate w.r.t. log|p|. The relative step lands at
                # log|p| +/- eps exactly, so the denominator is eps-based and
                # independent of p (candidate-a log finite difference). This
                # matches the "direct" denominator numerically but is reached
                # for a different reason and applied to a relative-step
                # numerator, giving df/dlog|p| = p * df/dp.
                delta_central = 2.0 * self.epsilon
                delta_fwd = self.epsilon
                delta_bwd = self.epsilon
            elif self._relative_step and base_value != 0:
                # Fractional step p -> p*exp(+/-eps); return df/dp, so divide
                # by the actual parameter-space perturbation p+ - p-.
                exp_pos = torch.exp(torch.tensor(self.epsilon)).item()
                exp_neg = torch.exp(torch.tensor(-self.epsilon)).item()
                delta_central = base_value * (exp_pos - exp_neg)
                delta_fwd = base_value * (exp_pos - 1.0)
                delta_bwd = base_value * (1.0 - exp_neg)
            else:
                # Additive step p -> p +/- eps.
                delta_central = 2.0 * self.epsilon
                delta_fwd = self.epsilon
                delta_bwd = self.epsilon

            if all([mode == "central", "pos" in losses, "neg" in losses]):
                grad = (losses["pos"] - losses["neg"]) / delta_central
            elif mode == "forward" and "pos" in losses:
                grad = (losses["pos"] - self.loss_base) / delta_fwd
            elif mode == "backward" and "neg" in losses:
                grad = (self.loss_base - losses["neg"]) / delta_bwd
            else:
                grad = torch.zeros_like(self.loss_base)

            jacobian[param_name] = grad

        self.jacobian = jacobian
        return jacobian
