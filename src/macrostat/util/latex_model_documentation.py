"""
Utility functions for extracting and formatting docstring information from Behavior classes.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import ast
import inspect
import re
from typing import Dict, Set, Type

from macrostat.core.behavior import Behavior


def generate_latex_documentation(
    behavior_class: Type[Behavior],
    output_file: str = None,
    title: str = None,
    subsec: bool = True,
    preamble: str = None,
) -> str:
    r"""Make a LaTeX model description from a Behavior class.

    This function takes a Behavior class and returns a LaTeX model description by parsing
    the docstrings of the initialize() and the step() methods. It then copies the docstrings
    of those methods and all of the methods that they call. From each, it extracts the description
    and the equations section, and then formats them for LaTeX. It then saves the LaTeX code to a
    file if an output file is provided.

    Parameters
    ----------
    behavior_class : Type[Behavior]
        The Behavior class to make a LaTeX model description from.
    output_file : str, optional
        The file to save the LaTeX model description to.
    title : str, optional
        The title of the LaTeX model description.
    subsec : bool, optional
        If True, add a subsection for each method. If False, just append the description and equations.
    preamble : str, optional
        A string of LaTeX code to add to the preamble of the document, i.e. before the \begin{document} command.

    Returns
    -------
    str
        The LaTeX model description.

    Examples
    --------
    >>> from macrostat.models import get_model
    >>> GL06SIM = get_model("GL06SIM")
    >>> tex = generate_documentation(GL06SIM().behavior)
    >>> print(tex)
    """
    tex = create_latex_content(behavior_class, title, subsec, preamble)
    if output_file:
        with open(output_file, "w") as f:
            f.write(tex)
    return tex


def create_latex_content(
    behavior_class: Type[Behavior],
    title: str = None,
    subsec: bool = True,
    preamble: str = None,
) -> str:
    """Generate LaTeX content for a model's documentation.

    This function creates a complete LaTeX document structure for documenting a model's
    behavior class. It starts with a preamble and title, then adds a section for the initialization
    of the model, including any methods called by initialize(). It then adds a section for the
    step() method, including any methods called by step().

    Parameters
    ----------
    behavior_class : Type[Behavior]
        The Behavior class to document
    title : str, optional
        Optional title for the document. If None, uses the class name
    subsec : bool, optional
        If True, creates a subsection instead of a section
    preamble : str, optional
        Optional LaTeX preamble to include before the document content

    Returns
    -------
    str
        Complete LaTeX document as a string
    """
    title = title if title is not None else behavior_class.__name__

    # Extract the docstrings from the initialize() and step() methods
    docstrings = parse_behavior_docstrings(behavior_class)

    # Start LaTeX document with the preamble if provided, otherwise use the default preamble
    if preamble:
        latex = [preamble]
    else:
        latex = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
        ]

    # Start the document
    latex.append(r"\begin{document}")
    latex.append(f"\\title{{{title}}}")
    latex.append(r"\maketitle")

    # Add initialization equations
    if docstrings["initialize"]:
        latex.append(r"\section{Initialization Equations}")
        if "initialize" in docstrings["initialize"]:
            latex.append(
                convert_docstring_to_latex(
                    docstrings["initialize"]["initialize"], "initialize"
                )
            )

        # Go through any methods that have been called
        for method_name, docstring in docstrings["initialize"].items():
            if subsec:
                latex.append(f"\\subsection{{{method_name.replace('_', ' ').title()}}}")
            latex.append(convert_docstring_to_latex(docstring, method_name))

    # Add step equations
    if docstrings["step"]:
        latex.append(r"\section{Step Equations}")
        if "step" in docstrings["step"]:
            latex.append(convert_docstring_to_latex(docstrings["step"]["step"], "step"))

        for method_name, docstring in docstrings["step"].items():
            latex.append(f"\\subsection{{{method_name.replace('_', ' ').title()}}}")
            latex.append(convert_docstring_to_latex(docstring, method_name))
            latex.append("\n")

    # End LaTeX document
    latex.append(r"\end{document}")

    return "\n".join(latex)


def parse_behavior_docstrings(
    behavior_class: Type[Behavior],
) -> Dict[str, Dict[str, str]]:
    """Extract docstrings from methods called by initialize() or step() that have
    an Equations section.

    This function is used to extract the docstrings of the methods called by the
    initialize() and step() methods of a Behavior class. It then returns a dictionary
    with two keys: 'initialize' and 'step', each containing a dictionary mapping method
    names to their docstrings.

    Parameters
    ----------
    behavior_class : Type[Behavior]
        The Behavior class to extract docstrings from.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary with two keys: 'initialize' and 'step', each containing a dictionary
        mapping method names to their docstrings.
    """
    docstrings = {"initialize": {}, "step": {}}

    # Get the source code for the entire class
    source = inspect.getsource(behavior_class)

    # Parse the class definition
    class_node = ast.parse(source)

    # Find the class definition node
    class_def = None
    for node in ast.walk(class_node):
        if isinstance(node, ast.ClassDef):
            class_def = node
            break

    if class_def is None:
        return docstrings

    # Find initialize and step methods
    initialize_node = None
    step_node = None
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "initialize":
                initialize_node = node
            elif node.name == "step":
                step_node = node

    # Get methods called by initialize and step
    initialize_methods = (
        find_called_methods(initialize_node) if initialize_node else set()
    )
    step_methods = find_called_methods(step_node) if step_node else set()

    # Get all methods from the class
    methods = inspect.getmembers(behavior_class, predicate=inspect.isfunction)

    for name, method in methods:
        # Skip private methods and the step/initialize methods themselves
        if name.startswith("_"):  # or name in ["step", "initialize"]:
            continue

        doc = method.__doc__
        if doc and "Equations" in doc:
            # Check if method is called by initialize or step
            if name in initialize_methods or name == "initialize":
                docstrings["initialize"][name] = doc
            if name in step_methods or name == "step":
                docstrings["step"][name] = doc

    return docstrings


def find_called_methods(method_node: ast.FunctionDef) -> Set[str]:
    """Extract the names of methods called within a method's AST node.

    This function is used to extract the names of the methods called within a method's
    AST node. It is used to determine which methods are called by the initialize() and
    step() methods of a Behavior class.

    Parameters
    ----------
    method_node : ast.FunctionDef
        The AST node of the method.

    Returns
    -------
    Set[str]
        Set of method names called within the method.
    """
    called_methods = set()

    # Visit all nodes in the AST
    for node in ast.walk(method_node):
        # Look for method calls (self.method_name)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "self":
                called_methods.add(node.attr)

    return called_methods


def convert_docstring_to_latex(docstring: str, label: str = None) -> str:
    """Convert a docstring to LaTeX text and align equations.

    This function is used to convert a docstring to LaTeX text and add the equations
    to the docstring in an align environment.

    Parameters
    ----------
    docstring : str
        The docstring to convert to LaTeX.
    label : str, optional
        The label of the equation. If None, no label is added.

    Returns
    -------
    str
        The LaTeX text and align equations.
    """
    tex = []

    # Description of the method
    description = docstring.split("Parameters")[0].strip()
    if description:
        tex.append(description + "\n")

    # Equations of the method
    equations = extract_equations_from_docstring(docstring)
    if equations:
        tex.append(r"\begin{align}\label{eq:" + label + r"}")
        tex.append(equations)
        tex.append(r"\end{align}")

    return "\n".join(tex)


def extract_equations_from_docstring(docstring: str) -> str:
    """Extract the Equations section from a docstring and format it for LaTeX.

    This function is used to extract the Equations section from a docstring and format
    it for LaTeX. It eliminates any sphinx directives and other text that is not part
    of the equations, and, if it finds an align environment, it removes the outermost
    align environment.

    Parameters
    ----------
    docstring : str
        The docstring containing an Equations section.

    Returns
    -------
    str
        The formatted LaTeX equations.
    """
    # Split the docstring into sections
    sections = re.split(r"\n\s*([A-Za-z]+)\s*\n\s*-+\n", docstring)

    # Find the Equations section
    for i, section in enumerate(sections):
        if section.strip() == "Equations":
            equations_text = sections[i + 1].strip()
            break
    else:
        return ""

    # Format the equations for LaTeX
    equations = []
    outer_align = 0  # Track the outermost align environment

    for line in equations_text.split("\n"):
        line = line.strip()
        # Skip lines that are not equations (e.g. sphinx math directive and its options) or empty lines
        if line.startswith(".. math::") or line.startswith(":") or not line:
            continue
        # If the sphinx uses :nowrap: and there is an align environment, skip the outermost align environment
        if line.startswith("\\begin{align}"):
            outer_align += 1
            if outer_align == 1:  # Skip only the outermost align
                continue
        elif line.startswith("\\end{align}"):
            outer_align -= 1
            if outer_align == 0:  # Skip only the outermost align
                continue
        # Remove indentation
        if line.startswith("   "):
            line = line[3:]
        # Skip empty lines
        if line:
            equations.append(line)

    return "\n".join(equations)
