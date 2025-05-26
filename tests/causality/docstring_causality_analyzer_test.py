import unittest

import pandas as pd

from macrostat.causality.docstring_causality_analyzer import DocstringCausalityAnalyzer
from macrostat.core import Model


class TestBehavior:
    def step(self):
        self.method1()
        self.method2()
        self.method3()

    def method1(self):
        """Update state variables based on dependencies

        Dependency
        -------
        scenario: scenario1
        parameter: parameter1
        parameter: parameter2

        Sets
        -------
        state1
        """
        pass

    def method2(self):
        """Calculate parameters based on state

        Dependency
        -------
        state: state1
        parameter: parameter3

        Sets
        -------
        state2
        """
        pass

    def method3(self):
        pass


class TestModel(Model):
    def __init__(self):
        self.behavior = TestBehavior


class TestDocstringCausalityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DocstringCausalityAnalyzer(TestModel)

    def test_initialization(self):
        """Test that the analyzer initializes correctly with a model class"""
        self.assertIsInstance(self.analyzer, DocstringCausalityAnalyzer)
        self.assertEqual(self.analyzer.model_class, TestModel)

    def test_parse_docstring(self):
        """Test docstring parsing functionality

        The docstrings are parsed from the behavior methods and stored in
        self._docstrings. We check whether the docstring is parsed correctly,
        splitting its components into dependency and sets.
        """

        docstring = """
        This is a test docstring.

        Prior Heading
        -------------
        This is a test prior heading.

        Dependency
        -------
        scenario: scenario1
        parameter: parameter1
        state: state1

        Sets
        -------
        state2

        Posterior Heading
        -----------------
        This is a test posterior heading.
        """
        result = self.analyzer._parse_docstring(docstring)

        expected = {
            "Dependency": {
                "scenario": ["scenario1"],
                "parameter": ["parameter1"],
                "state": ["state1"],
            },
            "Sets": {"state": ["state2"]},
        }
        self.assertEqual(result, expected)

    def test_get_methods_called_by_step(self):
        """Test extraction of methods called by step()

        The methods are extracted from the behavior methods and stored in
        self._called_methods. We check if all methods are present.
        """
        self.analyzer._get_methods_called_by_step()
        expected_methods = ("method1", "method2", "method3")
        self.assertEqual(self.analyzer._called_methods, expected_methods)

    def test_parse_behavior_docstrings(self):
        """Test extraction of docstrings from behavior methods

        The docstrings are parsed from the behavior methods and stored in
        self._docstrings. We check if all methods are present. The content
        of the docstrings is checked in the test_parse_docstring method.
        """
        self.analyzer._parse_behavior_docstrings()
        self.assertIn("method1", self.analyzer._docstrings)
        self.assertIn("method2", self.analyzer._docstrings)
        # We do not parse method docstrings that do not have a Dependency or Sets section
        self.assertNotIn("method3", self.analyzer._docstrings)

    def test_build_adjacency_matrix(self):
        """Test building of adjacency matrix from dependencies"""
        # First parse the docstrings
        self.analyzer.analyze()

        # Check that the matrix is a DataFrame
        self.assertIsInstance(self.analyzer.adjacency_matrix, pd.DataFrame)

        # Check that it has the correct structure
        self.assertTrue(
            all(
                col in self.analyzer.adjacency_matrix.columns.names
                for col in ["target_type", "target_name"]
            )
        )
        self.assertTrue(
            all(
                idx in self.analyzer.adjacency_matrix.index.names
                for idx in ["source_type", "source_name"]
            )
        )

    def test_analyze(self):
        """Test the complete analysis workflow"""
        result = self.analyzer.analyze()

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that it has the correct structure
        self.assertTrue(
            all(col in result.columns.names for col in ["target_type", "target_name"])
        )
        self.assertTrue(
            all(idx in result.index.names for idx in ["source_type", "source_name"])
        )


if __name__ == "__main__":
    unittest.main()
