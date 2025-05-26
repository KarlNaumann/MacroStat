import ast
import inspect
import logging
from typing import Dict, Type

import pandas as pd

from macrostat.causality import CausalityAnalyzer
from macrostat.core import Model

logger = logging.getLogger(__name__)


class DocstringCausalityAnalyzer(CausalityAnalyzer):
    def __init__(self, model_class: Type[Model]):
        super().__init__(model_class=model_class)

    def analyze(self):
        """Analyze a model class and return dependency dictionary"""

        # Gather the docstrings
        self._parse_behavior_docstrings()

        # Parse the docstrings of each method called by step()
        self._relations = {
            k: self._parse_docstring(v) for k, v in self._docstrings.items()
        }

        # Build the adjacency matrix
        self.build_adjacency_matrix()

        return self.adjacency_matrix

    def build_adjacency_matrix(self):
        """Build adjacency matrix from dependencies

        The adjacency matrix maps scenarios, state variables, and parameters to each other.
        The rows represent the prior, scenario, parameters and state variables, i.e. the
        dependency section of the docstring. The columns represent the state variables, i.e.
        the sets section of the docstring.
        """

        # Gather all of the (type, name) pairs in the dependencies and sets
        dependency_rows, set_columns = set(), set()
        for components in self._relations.values():
            # Handle dependencies
            for type_name, names in components["Dependency"].items():
                for name in names:
                    dependency_rows.add((type_name, name))

            # Handle sets
            for name in components["Sets"]["state"]:
                set_columns.add(("state", name))

        all_entities = set(dependency_rows) | set(set_columns)

        # Generate the empty adjacency matrix
        # Build the adjacency matrix
        self.adjacency_matrix = pd.DataFrame(
            0.0,
            index=pd.MultiIndex.from_tuples(
                all_entities, names=["source_type", "source_name"]
            ),
            columns=pd.MultiIndex.from_tuples(
                all_entities, names=["target_type", "target_name"]
            ),
            dtype=float,
        )
        self.adjacency_matrix.sort_index(axis=0, inplace=True)
        self.adjacency_matrix.sort_index(axis=1, inplace=True)

        # Fill adjacency matrix with weight 1 for all dependencies
        for components in self._relations.values():
            for target in components["Sets"]["state"]:
                for type_name, names in components["Dependency"].items():
                    for name in names:
                        self.adjacency_matrix.loc[
                            (type_name, name), ("state", target)
                        ] = 1

        return self.adjacency_matrix

    ###########################################################################
    # Docstring parsing methods
    ###########################################################################

    def _get_methods_called_by_step(self):
        """Extract docstrings from methods called by step() that have
        a Dependency and/or Sets section.

        This function is used to extract the docstrings of the methods called by the
        step() methods of a Behavior class. It handles inheritance by checking both
        the current class and its parent classes for method implementations.

        Sets
        -------
        docstrings : Dict[str, str]
            Dictionary mapping method names to their docstrings.
        order : Tuple[str, ...]
            Tuple with the order of the methods.
        """
        behavior = self.model_class().behavior

        # Get the correct step method
        step_method = getattr(behavior, "step")
        if inspect.isfunction(step_method) or inspect.ismethod(step_method):
            step_method = step_method

        # Get the source code for the step method
        source = inspect.getsource(step_method)
        # Remove any common leading whitespace from every line
        lines = source.splitlines()
        if not lines:
            logger.warning(f"Empty source code for step method in {behavior.__name__}")
            return

        # Find the minimum indentation
        min_indent = float("inf")
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        # Remove the common indentation
        if min_indent < float("inf"):
            source = "\n".join(line[min_indent:] for line in lines)

        # Parse the step method definition
        step_node = ast.parse(source)

        # Visit all nodes in the AST to find method calls
        self._called_methods = []
        for node in ast.walk(step_node):
            # Look for method calls (self.method_name())
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr not in self._called_methods
            ):
                self._called_methods.append(node.func.attr)

        # For each called method, find its actual implementation
        resolved_methods = []
        for method_name in self._called_methods:
            # Find the first class in MRO that implements this method
            for cls in behavior.__mro__:
                if hasattr(cls, method_name):
                    resolved_methods.append(method_name)

        # Convert to tuple to avoid permutation issues
        self._called_methods = tuple(resolved_methods)

    def _parse_behavior_docstrings(self):
        """Extract docstrings from methods called by step() that have
        a Dependency and/or Sets section.

        This function is used to extract the docstrings of the methods called by the
        step() methods of a Behavior class. It then returns a dictionary mapping method
        names to their docstrings, and a tuple with the order of the methods.

        Sets
        -------
        docstrings : Dict[str, str]
            Dictionary mapping method names to their docstrings.
        order : Tuple[str, ...]
            Tuple with the order of the methods.
        """
        behavior = self.model_class().behavior
        self._docstrings = {}
        self._get_methods_called_by_step()

        # Get all methods from the class
        for name, method in inspect.getmembers(behavior, predicate=inspect.isfunction):
            # Skip private methods and methods not called by step()
            if name.startswith("_") or name not in self._called_methods:
                continue

            # Extract docstring
            doc = method.__doc__
            if doc and ("Dependency" in doc or "Sets" in doc):
                self._docstrings[name] = doc
            else:
                logger.warning(f"No Dependency or Sets section in {name}")

    def _parse_docstring(self, docstring: str) -> Dict[str, Dict[str, str]]:
        """Parse a docstring and return a dictionary of dependencies and sets

        Docstring titles are "underlined" with a variable number of "-" characters.
        We extract the Dependency and Sets sections. Then for each line in that section,
        we extract the item type (pre-colon) and the item name (post-colon).

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary mapping item type to a dictionary mapping item name to the item value.
        """
        result = {"Dependency": {}, "Sets": {"state": []}}

        # Split docstring into lines and remove empty lines
        lines = [line.strip() for line in docstring.split("\n") if line.strip()]

        current_section = None

        for i, line in enumerate(lines):
            if line.replace("-", "").strip() == "":
                # Check section header (based on all dash underlines)
                current_section = lines[i - 1].strip()
                continue
            elif lines[min(i + 1, len(lines) - 1)].replace("-", "").strip() == "":
                # If the next line is also an underline, skip it
                continue
            elif current_section == "Dependency" and ":" in line:
                # For Dependency section, parse type:value pairs
                type_name, value = line.split(":", 1)
                type_name = type_name.replace("-", "").strip()
                value = value.strip()
                # Initialize list if this is the first value for this type
                if type_name not in result[current_section]:
                    result[current_section][type_name] = []
                # Append the value to the list
                result[current_section][type_name].append(value)
            elif current_section == "Sets":
                # For Sets section, just add the state variable name
                result[current_section]["state"].append(line.replace("-", "").strip())

        return result
