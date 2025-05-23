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

        # Parse the docstrings
        self._relations = {
            k: self._parse_docstring(v) for k, v in self._docstrings.items()
        }

        # Build the adjacency matrix
        self._build_adjacency_matrix()

        return self.adjacency_matrix

    def build_adjacency_matrix(self) -> pd.DataFrame:
        """Build adjacency matrix from dependencies"""
        raise NotImplementedError("Subclasses must implement this method")

    ###########################################################################
    # Docstring parsing methods
    ###########################################################################

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

        # Get the source code for the entire class
        source = inspect.getsource(behavior)

        # Parse the class definition
        class_node = ast.parse(source)

        # Find the class definition node
        class_def = None
        for node in ast.walk(class_node):
            if isinstance(node, ast.ClassDef):
                class_def = node
                break

        if class_def is None:
            logger.warning(f"No class definition found in {behavior.__name__}")
            return

        # Find step method
        step_node = None
        for node in class_def.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == "step":
                    step_node = node

        # Visit all nodes in the AST
        self._called_methods = []
        for node in ast.walk(step_node):
            # Look for method calls (self.method_name)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "self" and node.attr not in self._called_methods:
                    self._called_methods.append(node.attr)

        # Convert to tuple to avoid permutation issues
        self._called_methods = tuple(self._called_methods)

        # Get all methods from the class
        for name, method in inspect.getmembers(behavior, predicate=inspect.isfunction):
            # Skip private methods and methods not called by step()
            if name.startswith("_") or name not in self._called_methods:
                continue

            # Extract docstring
            doc = method.__doc__
            if doc and ("Dependency" in doc or "Sets" in doc):
                self._docstrings[name] = doc

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

        if not docstring:
            return result

        # Split docstring into lines and remove empty lines
        lines = [line.strip() for line in docstring.split("\n") if line.strip()]

        current_section = None

        for line in lines:
            # Check if this is a section header
            if line in ["Dependency", "Sets"]:
                current_section = line
                continue

            # Skip lines that are just dashes (section underlines)
            if line.replace("-", "").strip() == "":
                continue

            # Handle different sections differently
            if current_section == "Dependency" and ":" in line:
                # For Dependency section, parse type:value pairs
                type_name, value = line.split(":", 1)
                type_name = type_name.replace("-", "").strip()
                value = value.strip()

                if type_name and value:
                    # Initialize list if this is the first value for this type
                    if type_name not in result[current_section]:
                        result[current_section][type_name] = []
                    # Append the value to the list
                    result[current_section][type_name].append(value)

            elif current_section == "Sets":
                # For Sets section, just add the state variable name
                result[current_section]["state"].append(line.replace("-", "").strip())

        return result

    def _build_adjacency_matrix(self):
        """Build adjacency matrix from dependencies

        The adjacency matrix maps scenarios, state variables, and parameters to each other.
        The rows represent the prior, scenario, parameters and state variables, i.e. the
        dependency section of the docstring. The columns represent the state variables, i.e.
        the sets section of the docstring.
        """

        dependency_rows, set_columns = set(), set()

        for components in self._relations.values():
            # Handle dependencies
            for type_name, names in components["Dependency"].items():
                for name in names:
                    dependency_rows.add((type_name, name))

            # Handle sets
            for name in components["Sets"]["state"]:
                set_columns.add(("state", name))

        # Build the adjacency matrix
        self.adjacency_matrix = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                dependency_rows, names=["source_type", "source_name"]
            ),
            columns=pd.MultiIndex.from_tuples(
                set_columns, names=["target_type", "target_name"]
            ),
        ).fillna(0)
        self.adjacency_matrix.sort_index(axis=0, inplace=True)
        self.adjacency_matrix.sort_index(axis=1, inplace=True)

        for components in self._relations.values():
            for target in components["Sets"]["state"]:
                for type_name, names in components["Dependency"].items():
                    for name in names:
                        self.adjacency_matrix.loc[
                            (type_name, name), ("state", target)
                        ] = 1

        return self.adjacency_matrix
