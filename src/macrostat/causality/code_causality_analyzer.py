from typing import Type

import pandas as pd

from macrostat.causality import CausalityAnalyzer
from macrostat.core import Model


class CodeCausalityAnalyzer(CausalityAnalyzer):
    def __init__(self, model_class: Type[Model]):
        super().__init__(model_class=model_class)

    def analyze(self):
        """Analyze a model class and return dependency dictionary"""
        raise NotImplementedError("Subclasses must implement this method")

    def build_adjacency_matrix(self) -> pd.DataFrame:
        """Build adjacency matrix from dependencies"""
        raise NotImplementedError("Subclasses must implement this method")
