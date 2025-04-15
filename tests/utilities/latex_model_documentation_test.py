import ast
from unittest import TestCase
from unittest.mock import MagicMock, patch

from macrostat.core.behavior import Behavior
from macrostat.util.latex_model_documentation import (
    convert_docstring_to_latex,
    create_latex_content,
    extract_equations_from_docstring,
    find_called_methods,
    generate_latex_documentation,
    parse_behavior_docstrings,
)


class MockBehavior(Behavior):
    def __init__(self):
        super().__init__()

    def initialize(self):
        """Initialize the model.

        This is a test initialization method.

        Equations
        ---------
        x = y + z
        """
        self.helper_method()

    def step(self):
        """Step the model forward.

        This is a test step method.

        Equations
        ---------
        a = b + c
        """
        self.another_helper()

    def helper_method(self):
        """Helper method for initialization.

        Equations
        ---------
        y = 2 * z
        """
        pass

    def another_helper(self):
        """Another helper method for step.

        Equations
        ---------
        b = 3 * c
        """
        pass


class TestLatexModelDocumentation(TestCase):
    def setUp(self):
        # Create a mock Behavior class for testing

        self.mock_behavior = MockBehavior

    def test_find_called_methods(self):
        # Create a simple AST node for testing
        source = """def test_method(self):
    self.helper1()
    x = self.helper2()
    self.helper3()"""
        node = ast.parse(source).body[0]

        result = find_called_methods(node)
        expected = {"helper1", "helper2", "helper3"}
        self.assertEqual(result, expected)

    def test_extract_equations_from_docstring(self):
        docstring = """Test method.

Equations
---------
x = y + z
a = b + c"""
        result = extract_equations_from_docstring(docstring)
        expected = "x = y + z\na = b + c"
        self.assertEqual(result, expected)

    def test_extract_equations_from_docstring_with_align(self):
        docstring = """Test method.

Equations
---------
.. math::
    :nowrap:

    \\begin{align}
    x &= y + z \\\\
    a &= b + c
    \\end{align}"""
        result = extract_equations_from_docstring(docstring)
        expected = "x &= y + z \\\\\na &= b + c"
        self.assertEqual(result, expected)

    def test_convert_docstring_to_latex(self):
        docstring = """Test method description.

Equations
---------
x = y + z"""
        result = convert_docstring_to_latex(docstring, "test")
        expected = """Test method description.

Equations
---------
x = y + z

\\begin{align}\\label{eq:test}
x = y + z
\\end{align}"""
        self.assertEqual(result, expected)

    def test_parse_behavior_docstrings(self):
        result = parse_behavior_docstrings(self.mock_behavior)

        # Check structure
        self.assertIn("initialize", result)
        self.assertIn("step", result)

        # Check content
        self.assertIn("helper_method", result["initialize"])
        self.assertIn("another_helper", result["step"])

    def test_create_latex_content(self):
        result = create_latex_content(
            self.mock_behavior,
            title="Test Model",
            subsec=True,
            preamble="\\usepackage{test}",
        )

        # Check basic structure
        self.assertIn(r"\title{Test Model}", result)
        self.assertIn(r"\usepackage{test}", result)

        # Check content sections
        self.assertIn(r"\section{Initialization Equations}", result)
        self.assertIn(r"\section{Step Equations}", result)

        # Check equations
        self.assertIn("x = y + z", result)
        self.assertIn("a = b + c", result)

    def test_generate_latex_documentation(self):
        with patch("builtins.open", MagicMock()) as mock_open:
            result = generate_latex_documentation(
                self.mock_behavior, output_file="test.tex", title="Test Model"
            )

            # Check that file was written
            mock_open.assert_called_once_with("test.tex", "w")

            # Check content
            self.assertIn(r"\documentclass{article}", result)
            self.assertIn(r"\title{Test Model}", result)
