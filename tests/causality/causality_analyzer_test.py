import pandas as pd

from macrostat.causality.causality_analyzer import CausalityAnalyzer
from macrostat.core import Model


class TestModel(Model):
    """A simple test model for testing CausalityAnalyzer"""

    def __init__(self):
        super().__init__()
        self.add_state("x", 1.0)
        self.add_state("y", 2.0)
        self.add_parameter("a", 3.0)
        self.add_parameter("b", 4.0)


class TestCausalityAnalyzer(CausalityAnalyzer):
    """A concrete implementation of CausalityAnalyzer for testing"""

    _relations = {
        "function1": {
            "Dependency": {"state": ["s1", "s2"], "parameter": ["p1", "p2"]},
            "Sets": {"state": ["s3"]},
        },
        "function2": {
            "Dependency": {"state": ["s3"], "scenario": ["sc1"]},
            "Sets": {"state": ["s5"]},
        },
    }

    def analyze(self):
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

    def build_adjacency_matrix(self):
        return self.adjacency_matrix


def test_check_for_cycles_no_cycles():
    """Test check_for_cycles when there are no cycles in the graph"""
    analyzer = TestCausalityAnalyzer(TestModel)
    analyzer.analyze()
    cycles = analyzer.check_for_cycles()
    assert len(cycles) == 0


def test_check_for_cycles_with_cycles():
    """Test check_for_cycles when there are cycles in the graph"""
    analyzer = TestCausalityAnalyzer(TestModel)
    analyzer._relations["cycle"] = {
        "Dependency": {"state": ["s1", "s2"]},
        "Sets": {"state": ["s1"]},
    }
    analyzer.analyze()
    cycles = analyzer.check_for_cycles()
    assert len(cycles) > 0
    assert any("state:s1" in cycle for cycle in cycles)


def test_create_edgelist():
    """Test _create_edgelist method"""
    analyzer = TestCausalityAnalyzer(TestModel)
    analyzer.analyze()
    edgelist = analyzer._create_edgelist()

    # Check that edgelist is a DataFrame
    assert isinstance(edgelist, pd.DataFrame)

    # Check required columns
    required_columns = [
        "source_type",
        "source_name",
        "target_type",
        "target_name",
        "weight",
        "source",
        "target",
    ]
    assert all(col in edgelist.columns for col in required_columns)

    # Check that weights are non-zero
    assert all(edgelist["weight"] != 0)

    # Check source and target format
    assert all(":" in source for source in edgelist["source"])
    assert all(":" in target for target in edgelist["target"])

    # Check that source and target are properly formatted
    for _, row in edgelist.iterrows():
        assert row["source"] == f"{row['source_type']}:{row['source_name']}"
        assert row["target"] == f"{row['target_type']}:{row['target_name']}"
