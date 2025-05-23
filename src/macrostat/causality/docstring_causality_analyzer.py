import ast
import inspect
import logging
from typing import Type

import pandas as pd

from macrostat.causality import CausalityAnalyzer
from macrostat.core import Model

logger = logging.getLogger(__name__)


class DocstringCausalityAnalyzer(CausalityAnalyzer):
    def __init__(self, model_class: Type[Model]):
        super().__init__(model_class=model_class)

    def analyze(self):
        """Analyze a model class and return dependency dictionary"""
        self._parse_behavior_docstrings()

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
