"""
A class for handling parameters for a MacroStat model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

# Default libraries
import json
import logging
import os
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
import torch

from macrostat.core.constraints import (
    ConstraintError,
    ConstraintResolver,
    LinearConstraint,
    ParameterLocation,
)

logger = logging.getLogger(__name__)


class BoundaryError(Exception):
    """Exception raised for invalid bounds."""

    def __init__(self, message: str):
        self.message = message + " Please check the Excel, JSON or default bounds."


class Parameters:
    """A class for handling parameters for the MacroStat model."""

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        data_parameters: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters for the model. If no parameters are provided,
        the default parameters will be used, and if only some parameters are
        provided, the missing parameters will be set to their default values.

        Parameters
        ----------
        parameters: dict | None
            The parameters to initialize the model with. If None, the default
            parameters will be used.
        hyperparameters: dict | None
            The hyperparameters to initialize the model with. If None, the
            default hyperparameters will be used.
        data_parameters: dict | None
            Tensor-valued parameters to initialize the model with. Each entry
            should be a dict with keys "value" (tensor or nested list),
            "lower_bound", "upper_bound", "notation", "unit". If None, the
            defaults from get_default_data_parameters() are used.
        """

        self.values = self.get_default_parameters()
        if parameters is not None:
            new = {k: v for k, v in parameters.items() if k in self.values}
            self.values.update(new)

        self.hyper = self.get_default_hyperparameters()
        if hyperparameters is not None:
            new = {k: v for k, v in hyperparameters.items() if k in self.hyper}
            self.hyper.update(new)

        self.data = self.get_default_data_parameters()
        if data_parameters is not None:
            for k, v in data_parameters.items():
                if k in self.data:
                    if isinstance(v.get("value"), (list, np.ndarray)):
                        v = {**v, "value": torch.tensor(v["value"], dtype=torch.float)}
                    self.data[k].update(v)

        # Lazily built on first call to get_constraint_resolver; any future
        # mutation API that changes self.values must call _invalidate_resolver.
        self._constraint_resolver: ConstraintResolver | None = None

        self.verify_bounds()
        self.verify_constraints()
        self.enforce_constraints()
        self.verify_parameters()

    def __contains__(self, key: str):
        """Check if a key is in the parameters, data parameters, or hyperparameters.

        Parameters
        ----------
        key: str
            The name of the parameter or hyperparameter to check for.
        """
        return key in self.values or key in self.data or key in self.hyper

    def __getitem__(self, key: str):
        """Get an item from the parameters, data parameters, or hyperparameters.

        Parameters
        ----------
        key: str
            The name of the parameter or hyperparameter to get the item for.
        """
        if key in self.values:
            return self.values[key]["value"]
        if key in self.data:
            return self.data[key]["value"]
        return self.hyper[key]

    def __setitem__(self, key: str, value):
        """Set an item in the parameters, data parameters, or hyperparameters.

        Parameters
        ----------
        key: str
            The name of the parameter or hyperparameter to set.
        value: float | torch.Tensor
            The value to set for the item.
        """
        if key in self.values:
            self.values[key]["value"] = value
        elif key in self.data:
            if isinstance(value, (list, np.ndarray)):
                value = torch.tensor(value, dtype=torch.float)
            self.data[key]["value"] = value
        elif key in self.hyper:
            self.hyper[key] = value
        else:
            logger.warning(f"Key {key} not found in parameters or hyperparameters.")

    def __str__(self):  # pragma: no cover
        """Return a string representation of the parameters.

        This function returns a string representation of the parameters,
        with the hyperparameters and parameters aligned.
        """
        # Find the longest key for alignment
        all_keys = (
            list(self.hyper.keys()) + list(self.values.keys()) + list(self.data.keys())
        )
        max_key_length = max([len(key) for key in all_keys] + [10])

        # Create the output string, hyperparameters first
        output = "Hyperparameters:\n"
        for key, value in self.hyper.items():
            output += f"  {key:.<{max_key_length}} {value}\n"

        # Add the parameters
        output += "\nParameters:\n"
        for key, info in self.values.items():
            output += (
                f"  {key:.<{max_key_length}} {info['value']:.5g} ({info['unit']})\n"
            )

        # Add the data parameters
        if self.data:
            output += "\nData Parameters:\n"
            for key, info in self.data.items():
                shape = tuple(info["value"].shape)
                output += f"  {key:.<{max_key_length}} {shape} ({info['unit']})\n"

        return output

    @classmethod
    def from_json(cls, file_path: os.PathLike, *args, **kwargs):
        """Initialize the parameters from a JSON file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the JSON file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)

        data_params = None
        if "DataParameters" in data:
            data_params = {}
            for k, v in data["DataParameters"].items():
                data_params[k] = {
                    **v,
                    "value": torch.tensor(v["value"], dtype=torch.float),
                }

        return cls(
            parameters=data["Parameters"],
            hyperparameters=data["HyperParameters"],
            data_parameters=data_params,
        )

    @classmethod
    def from_excel(cls, file_path: os.PathLike, *args, **kwargs):
        """Initialize the parameters from an Excel file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the Excel file to load the parameters from.
        """
        raise NotImplementedError("Not implemented")

    @classmethod
    def from_csv(cls, file_path: os.PathLike, *args, **kwargs):
        """Initialize the parameters from a CSV file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the CSV file to load the parameters from.
        """
        df = pd.read_csv(file_path, index_col=0)

        # Parameters
        par = df[df["ParameterType"] == "Parameter"].drop(columns=["ParameterType"])
        par.columns = [i.lower() for i in par.columns]
        par["value"] = par["value"].astype(float)
        par = par.to_dict(orient="index")

        # Hyperparameters
        hyper = df[df["ParameterType"] == "HyperParameter"]["Value"]
        hyper.index = hyper.index.str.lower()
        hyper = hyper.to_dict()
        # Try to convert to int, else bool, else keep as string
        for key, value in hyper.items():
            if value.isdigit():
                hyper[key] = int(value)
            elif value.lower() == "true":
                hyper[key] = True
            elif value.lower() == "false":
                hyper[key] = False
            else:
                hyper[key] = value

        # Convert the dataframe to a dictionary
        return cls(
            parameters=par,
            hyperparameters=hyper,
        )

    def get_default_hyperparameters(self):
        """Return the default hyperparameters.

        The hyperparameters are the parameters that are not directly used in
        the model, but rather for the simulation and calibration. They must
        include:
        1. The number of timesteps to simulate
        2. The scenario trigger, i.e. the timestep at which the scenario starts
        3. The seed for the random number generator
        4. The device to use for the simulation
        5. May include other parameters, such as flags for the model

        """

        return {
            "timesteps": 100,
            "timesteps_initialization": 10,
            "scenario_trigger": 0,
            "seed": 42,
            "device": "cpu",
            "requires_grad": False,
        }

    def get_default_parameters(self):
        """Return the default parameters.

        Should return a dictionary with the keys being the parameter names,
        and a subdictionary with the keys including:
        - "value": The value of the parameter.
        - "lower bound": The lower bound of the parameter.
        - "upper bound": The upper bound of the parameter.
        - "unit": The unit of the parameter.
        - "notation": The notation of the parameter.
        """
        return {}

    def get_default_data_parameters(self):
        """Return the default data parameters.

        Data parameters are tensor-valued parameters (vectors, matrices) that
        are typically loaded from external data files (e.g. IO tables). They
        participate in the nn.ParameterDict pipeline and are fully
        differentiable.

        Should return a dictionary with the keys being the parameter names,
        and a subdictionary with the keys:
        - "value": torch.Tensor with the default value (e.g. zeros of the
          expected shape).
        - "lower_bound": float, element-wise lower bound.
        - "upper_bound": float, element-wise upper bound.
        - "notation": str, LaTeX notation.
        - "unit": str, unit description.
        """
        return {}

    def get_constraints(self) -> tuple[LinearConstraint, ...]:
        """Return parameter constraints for this model.

        Override in subclasses to declare adding-up constraints.
        Default: no constraints.

        Constraints are applied in declaration order. Chaining
        (a derived parameter in one constraint appearing as a free
        parameter in another) is explicitly rejected by
        :meth:`verify_constraints`.

        Returns
        -------
        tuple[LinearConstraint, ...]
            Constraints to enforce on this parameter set.
        """
        return ()

    def get_constraint_resolver(self) -> ConstraintResolver:
        """Return a :class:`ConstraintResolver` for the current parameter set.

        Built lazily on first call and cached on the instance. The
        resolver maps canonical (dotted) parameter names to their
        :class:`ParameterLocation` in the step-time tensor dict that
        :meth:`vectorize_parameters` produces. This lets the same
        :class:`LinearConstraint` apply to scalar, 1-D sector-indexed,
        and 2-D sector-by-sector parameter layouts uniformly.

        The build logic mirrors :meth:`vectorize_parameters`:

        * ``len(parts) == 2`` and first part in ``vector_sectors`` →
          1-D slot at ``tensor_key=parts[1]`` indexed by the sector
          position.
        * ``len(parts) == 3`` and first two parts in ``vector_sectors`` →
          2-D slot at ``tensor_key=parts[2]`` indexed by the row and
          column sector positions.
        * Otherwise → scalar slot at ``tensor_key=name.replace(".", "_")``
          with ``index=None``.

        Returns
        -------
        ConstraintResolver
            Resolver over the current parameter set.
        """
        if self._constraint_resolver is not None:
            return self._constraint_resolver

        if "vector_sectors" in self.hyper:
            vsecs = sorted(self.hyper["vector_sectors"])
        else:
            vsecs = []

        mapping: dict[str, ParameterLocation] = {}
        for name in self.values:
            parts = name.split(".")

            if len(parts) == 2 and parts[0] in vsecs:
                sec, par = parts
                mapping[name] = ParameterLocation(
                    tensor_key=par,
                    index=(vsecs.index(sec),),
                )
            elif len(parts) == 3 and parts[0] in vsecs and parts[1] in vsecs:
                rowsec, colsec, par = parts
                mapping[name] = ParameterLocation(
                    tensor_key=par,
                    index=(vsecs.index(rowsec), vsecs.index(colsec)),
                )
            else:
                mapping[name] = ParameterLocation(
                    tensor_key=name.replace(".", "_"),
                    index=None,
                )

        self._constraint_resolver = ConstraintResolver(mapping)
        return self._constraint_resolver

    def _invalidate_resolver(self) -> None:
        """Drop the cached :class:`ConstraintResolver`.

        Call from any future mutation API that changes the key set of
        ``self.values`` or the ``vector_sectors`` hyperparameter.
        Currently unused because parameters are effectively frozen
        after ``__init__``.
        """
        self._constraint_resolver = None

    def verify_constraints(self):
        """Verify that declared constraints are well-formed.

        Runs seven checks against the declared constraint set:

        1. Every referenced parameter exists in ``self.values``.
        2. No parameter is derived in more than one constraint.
        3. No derived parameter appears as a free parameter in another
           constraint (chaining is not supported).
        4. Every referenced parameter is locatable by the resolver
           returned from :meth:`get_constraint_resolver`. This is a
           defensive check against divergence between the parameter
           dictionary and the resolver build rules.
        5. Within each constraint, all parameter locations share a
           compatible shape class: all scalar, or all 1-D slots of the
           same ``tensor_key``, or all 2-D slots of the same
           ``tensor_key``. Mixed-shape constraints are rejected.
        6. The prospective post-enforcement value of the derived
           parameter (``target - sum(free)``) lies within its declared
           bounds. Catches misdeclared constraints at init time rather
           than after enforcement.
        7. If any constraint is violated by the provided defaults, a
           warning is logged and the derived parameter will be
           adjusted by :meth:`enforce_constraints`.

        Raises
        ------
        ConstraintError
            If any of checks 1-6 fail. :class:`ConstraintError`
            subclasses :class:`ValueError`, so callers that catch
            ``ValueError`` continue to work.
        """
        constraints = self.get_constraints()
        if not constraints:
            return

        # Check 1: all referenced parameter names exist.
        derived_params = []
        for c in constraints:
            for name in c.param_names:
                if name not in self.values:
                    raise ConstraintError(
                        f"Constraint references unknown parameter '{name}'. "
                        f"Available: {sorted(self.values.keys())}"
                    )
            derived_params.append(c.derived_param)

        # Check 2: no duplicate derived params.
        if len(derived_params) != len(set(derived_params)):
            seen = set()
            duplicates = []
            for p in derived_params:
                if p in seen:
                    duplicates.append(p)
                seen.add(p)
            raise ConstraintError(
                f"Parameter(s) {duplicates} appear as derived in multiple "
                f"constraints. Each parameter can be derived in at most one."
            )

        # Check 3: no derived param is free in another constraint.
        derived_set = set(derived_params)
        for c in constraints:
            overlap = derived_set & set(c.free_params)
            if overlap:
                raise ConstraintError(
                    f"Parameter(s) {sorted(overlap)} are derived in one "
                    f"constraint but appear as free in another. Chained "
                    f"constraints are not supported."
                )

        # Check 4: every name is resolvable. Uses a fresh resolver so
        # any divergence between the cache and the current parameter
        # dictionary surfaces here.
        resolver = self.get_constraint_resolver()
        for c in constraints:
            for name in c.param_names:
                if name not in resolver:
                    raise ConstraintError(
                        f"Constraint parameter '{name}' is not resolvable "
                        f"to a tensor slot. This usually indicates a "
                        f"mismatch between the parameter name and the "
                        f"vector_sectors hyperparameter."
                    )

        # Check 5: shape homogeneity within each constraint.
        for c in constraints:
            locs = [resolver.locate(name) for name in c.param_names]
            scalar_locs = [loc for loc in locs if loc.index is None]
            indexed_locs = [loc for loc in locs if loc.index is not None]
            if scalar_locs and indexed_locs:
                raise ConstraintError(
                    f"Constraint {c.param_names!r} mixes scalar and "
                    f"indexed parameter locations. All parameters in a "
                    f"single constraint must share the same shape class."
                )
            if indexed_locs:
                tensor_keys = {loc.tensor_key for loc in indexed_locs}
                if len(tensor_keys) > 1:
                    raise ConstraintError(
                        f"Constraint {c.param_names!r} spans multiple "
                        f"tensor keys {sorted(tensor_keys)!r}. All "
                        f"indexed parameters in a single constraint "
                        f"must live in the same tensor."
                    )
                index_dims = {len(loc.index) for loc in indexed_locs}
                if len(index_dims) > 1:
                    raise ConstraintError(
                        f"Constraint {c.param_names!r} mixes "
                        f"{sorted(index_dims)!r}-dimensional index "
                        f"shapes. All parameters must share the same "
                        f"index rank."
                    )

        # Check 6: prospective derived value lies within bounds.
        for c in constraints:
            free_sum = sum(self.values[name]["value"] for name in c.free_params)
            prospective = c.target - free_sum
            info = self.values[c.derived_param]
            lo = info.get("lower bound")
            hi = info.get("upper bound")
            if lo is not None and prospective < lo:
                raise ConstraintError(
                    f"Constraint would drive derived parameter "
                    f"'{c.derived_param}' to {prospective:.8g}, below "
                    f"its lower bound {lo:.8g}. Check the free "
                    f"parameter defaults or the constraint target."
                )
            if hi is not None and prospective > hi:
                raise ConstraintError(
                    f"Constraint would drive derived parameter "
                    f"'{c.derived_param}' to {prospective:.8g}, above "
                    f"its upper bound {hi:.8g}. Check the free "
                    f"parameter defaults or the constraint target."
                )

        # Check 7: warn if defaults violate constraints.
        for c in constraints:
            actual_sum = sum(self.values[name]["value"] for name in c.param_names)
            if abs(actual_sum - c.target) > 1e-6:
                logger.warning(
                    "Constraint violation in defaults: sum(%s) = %.8f, "
                    "target = %.8f. Derived parameter '%s' will be adjusted.",
                    c.param_names,
                    actual_sum,
                    c.target,
                    c.derived_param,
                )

    def enforce_constraints(self):
        """Enforce all constraints by adjusting derived parameter values.

        Modifies ``self.values[derived_param]["value"]`` in place so that
        the adding-up constraint is satisfied. Called during ``__init__``
        and before any export (``to_nn_parameters``, ``to_json``, etc.).
        """
        for c in self.get_constraints():
            free_sum = sum(self.values[name]["value"] for name in c.free_params)
            new_value = c.target - free_sum
            old_value = self.values[c.derived_param]["value"]
            if abs(old_value - new_value) > 1e-12:
                logger.debug(
                    "Constraint: %s adjusted from %.8g to %.8g",
                    c.derived_param,
                    old_value,
                    new_value,
                )
            self.values[c.derived_param]["value"] = new_value

    def get_free_param_names(self) -> list[str]:
        """Return parameter names excluding derived (constrained) parameters.

        Returns
        -------
        list[str]
            All parameter names that are free (not derived by any constraint).
        """
        derived = {c.derived_param for c in self.get_constraints()}
        return [name for name in self.values if name not in derived]

    def get_bounds(self):
        """Return the bounds for the parameters (scalar and data)."""
        bounds = {
            key: (info["lower bound"], info["upper bound"])
            for key, info in self.values.items()
        }
        bounds.update(
            {
                key: (info["lower_bound"], info["upper_bound"])
                for key, info in self.data.items()
            }
        )
        return bounds

    def get_values(self):
        """Return the values for the parameters (scalar and data)."""
        self.enforce_constraints()
        values = {key: info["value"] for key, info in self.values.items()}
        values.update({key: info["value"] for key, info in self.data.items()})
        return values

    def is_equal(self, other: "Parameters"):
        """Compare the parameters to another Parameters object."""
        if self.values != other.values or self.hyper != other.hyper:
            return False
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        for k in self.data:
            if not torch.equal(self.data[k]["value"], other.data[k]["value"]):
                return False
        return True

    def set_bound(self, key: str, value: tuple):
        """Set the bounds for a single parameter

        Parameters
        ----------
        key: str
            The key of the parameter to set the bounds for.
        value: tuple
            The bounds to set for the parameter.
        """
        self.values[key]["Lower Bound"] = value[0]
        self.values[key]["Upper Bound"] = value[1]
        self.verify_bounds()

    def set_notation(self, key: str, value: str):
        """Set the notation for a single parameter."""
        self.values[key]["notation"] = value

    def set_unit(self, key: str, value: str):
        """Set the unit for a single parameter."""
        self.values[key]["unit"] = value

    def load_data_parameter_from_file(
        self,
        name: str,
        file_path: os.PathLike,
        index_col: int | None = 0,
        header: int | None = 0,
    ):
        """Load a single data parameter from a CSV file.

        The CSV is read into a pandas DataFrame and converted to a tensor.
        Shape validation is strict: the loaded shape must match the shape of
        the default data parameter.

        Parameters
        ----------
        name: str
            The name of the data parameter to load into.
        file_path: os.PathLike
            Path to the CSV file.
        index_col: int | None
            Column to use as row labels (passed to pd.read_csv). Default 0
            skips the first column (assumes row labels).
        header: int | None
            Row to use as column labels (passed to pd.read_csv). Default 0
            skips the first row (assumes column headers).
        """
        if name not in self.data:
            raise KeyError(
                f"Data parameter '{name}' not found. "
                f"Available: {list(self.data.keys())}"
            )

        df = pd.read_csv(file_path, index_col=index_col, header=header)
        tensor = torch.tensor(df.values, dtype=torch.float).squeeze()

        expected_shape = self.data[name]["value"].shape
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for data parameter '{name}': "
                f"loaded {tuple(tensor.shape)}, expected {tuple(expected_shape)}"
            )

        self.data[name]["value"] = tensor

    def load_data_parameters_from_directory(
        self,
        data_dir: os.PathLike,
        mapping: dict[str, str] | None = None,
        **csv_kwargs,
    ):
        """Batch-load data parameters from a directory of CSV files.

        Parameters
        ----------
        data_dir: os.PathLike
            Path to the directory containing CSV files.
        mapping: dict[str, str] | None
            A mapping of ``{parameter_name: filename}``. If None, looks for
            ``{ParameterName}.csv`` for each defined data parameter.
        **csv_kwargs
            Additional keyword arguments passed to
            ``load_data_parameter_from_file`` (e.g. ``index_col``, ``header``).
        """
        data_dir = Path(data_dir)

        if mapping is None:
            mapping = {name: f"{name}.csv" for name in self.data}

        for param_name, filename in mapping.items():
            file_path = data_dir / filename
            if not file_path.exists():
                logger.warning(
                    f"File not found for data parameter '{param_name}': {file_path}"
                )
                continue
            self.load_data_parameter_from_file(param_name, file_path, **csv_kwargs)

    def to_csv(self, file_path: os.PathLike, sphinx_math: bool = False):
        """Convert the parameters to a CSV file.

        Scalar parameters and hyperparameters are written to the main CSV file.
        Data parameters are written as individual CSVs in a sibling directory
        named ``{stem}_data/``.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the CSV file to save the parameters to.
        sphinx_math: bool
            Whether to use Sphinx math notation in the CSV file.
        """
        self.enforce_constraints()
        par = pd.DataFrame.from_dict(self.values, orient="index").sort_index()
        par = par[["notation", "unit", "value", "lower bound", "upper bound"]]
        par.columns = ["Notation", "Unit", "Value", "Lower Bound", "Upper Bound"]
        if sphinx_math:  # pragma: no cover
            par["Notation"] = par["Notation"].apply(lambda x: r":math:`" + x + r"`")
        par["ParameterType"] = "Parameter"

        hyper = pd.DataFrame.from_dict(self.hyper, orient="index").sort_index()
        hyper.columns = ["Value"]
        hyper["ParameterType"] = "HyperParameter"

        df = pd.concat([par, hyper], axis=0)
        df.to_csv(file_path)

        if self.data:
            file_path = Path(file_path)
            data_dir = file_path.parent / f"{file_path.stem}_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            for name, info in self.data.items():
                tensor = info["value"]
                if tensor.dim() <= 1:
                    pd.Series(tensor.detach().numpy()).to_csv(data_dir / f"{name}.csv")
                else:
                    pd.DataFrame(tensor.detach().numpy()).to_csv(
                        data_dir / f"{name}.csv"
                    )

        return df

    def to_excel(self, file_path: os.PathLike, *args, **kwargs):
        """Convert the parameters to an Excel file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the Excel file to save the parameters to.
        """
        raise NotImplementedError("Not implemented")

    def to_json(self, file_path: os.PathLike, *args, **kwargs):
        """Convert the parameters to a JSON file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the JSON file to save the parameters to.
        """
        self.enforce_constraints()
        payload = {
            "Parameters": self.values,
            "HyperParameters": self.hyper,
        }

        if self.data:
            payload["DataParameters"] = {
                k: {
                    **{dk: dv for dk, dv in v.items() if dk != "value"},
                    "value": v["value"].tolist(),
                }
                for k, v in self.data.items()
            }

        with open(file_path, "w") as file:
            json.dump(payload, file)

    def to_nn_parameters(self):
        """Convert the parameters to a nn.ParameterDict."""
        self.enforce_constraints()
        vectorized = self.vectorize_parameters()
        return torch.nn.ParameterDict(vectorized)

    def vectorize_parameters(self):
        """Vectorize the parameters.

        This function generates vectors out of the list of individual parameters.
        It does so by respecting the "vector_sectors" hyperparameter. If this
        hyperparameter exists, then:

        1. for each parameter of the form "sector.name" a vector of zeros of
        size (len(vector_sectors),1) will be generated and populated with
        the values of the sector. The sectors index is computed as the index
        of the sector name in the sorted vector_sectors list.
        2. for each parameter of the form "rowsec.colsec.name" a matrix of zeros
        of size (len(vector_sectors),len(vector_sectors)) will be generated and
        populated with the values of the sector. The sectors index is computed
        as the index of the sector name in the sorted vector_sectors list. The
        first sector is the row, the second the column of the matrix to be
        populated

        Data parameters (tensors) are included directly by name.
        """
        # Allow users to specify that only a subset of sectors fit the vector/matrix scheme
        if "vector_sectors" in self.hyper:
            vsecs = sorted(self.hyper["vector_sectors"])
        else:
            vsecs = []

        # Parse info dict to generate: individual, vector, and matrix setups
        kwargs = dict(device=self.hyper["device"], dtype=torch.float)
        pvectors = {}
        for key, info in self.values.items():
            parts = key.split(".")

            # Case 1: parameter vectors (if desired)
            if len(parts) == 2 and parts[0] in vsecs:
                sec, par = parts

                if par not in pvectors:
                    pvectors[par] = torch.zeros(len(vsecs), **kwargs)

                pvectors[par][vsecs.index(sec)] = info["value"]

            # Case 2: parameter matrices, e.g. input-output coefficients
            # We assume that the order is row-sector.col-sector.name
            elif len(parts) == 3 and parts[0] in vsecs and parts[1] in vsecs:
                rowsec, colsec, par = parts

                if par not in pvectors:
                    pvectors[par] = torch.zeros(len(vsecs), len(vsecs), **kwargs)

                row = (vsecs.index(rowsec),)
                col = vsecs.index(colsec)
                pvectors[par][row, col] = info["value"]

            # Case 3: everything else (incl. if more than 2 periods in the name)
            else:
                v = torch.tensor(info["value"], **kwargs)
                pvectors[key.replace(".", "_")] = v

        # Include data parameters directly (already tensors)
        for key, info in self.data.items():
            tensor = info["value"]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, **kwargs)
            pvectors[key] = tensor.to(device=kwargs["device"], dtype=kwargs["dtype"])

        return pvectors

    def verify_bounds(self):
        """Verify that the bounds are valid. By testing first that all
        parameters have bounds, and then that the bounds are valid.
        """
        # Check scalar parameters
        needed_bounds = set(self.get_default_parameters().keys())
        found_bounds = {}
        for key, info in self.values.items():
            conditions = [
                "lower bound" in info and info["lower bound"] is not None,
                "upper bound" in info and info["upper bound"] is not None,
            ]
            if all(conditions):
                found_bounds[key] = (info["lower bound"], info["upper bound"])

        found_bound_params = set(found_bounds.keys())
        if needed_bounds.difference(found_bound_params):
            raise BoundaryError(
                f"Missing bounds for parameters: {needed_bounds - found_bound_params}"
            )

        for param, bounds in found_bounds.items():
            if bounds[0] > bounds[1]:
                raise BoundaryError(f"Parameter {param} has invalid bounds: {bounds}")

        # Check data parameters
        for key, info in self.data.items():
            lb, ub = info.get("lower_bound"), info.get("upper_bound")
            if lb is not None and ub is not None and lb > ub:
                raise BoundaryError(
                    f"Data parameter {key} has invalid bounds: ({lb}, {ub})"
                )

    def verify_parameters(self):
        """Verify that the parameters are within the bounds."""
        # Scalar parameters
        for param, info in self.values.items():
            if (
                info["value"] < info["lower bound"]
                or info["value"] > info["upper bound"]
            ):
                msg = f"Parameter {param} has invalid value: {info['value']}"
                raise BoundaryError(
                    f"{msg} (bounds: {info['lower bound']}, {info['upper bound']})"
                )

        # Data parameters
        for param, info in self.data.items():
            tensor = info["value"]
            lb, ub = info.get("lower_bound"), info.get("upper_bound")
            if lb is not None and tensor.min().item() < lb:
                raise BoundaryError(
                    f"Data parameter {param} has values below lower bound {lb} "
                    f"(min: {tensor.min().item():.6g})"
                )
            if ub is not None and tensor.max().item() > ub:
                raise BoundaryError(
                    f"Data parameter {param} has values above upper bound {ub} "
                    f"(max: {tensor.max().item():.6g})"
                )


if __name__ == "__main__":
    pass
