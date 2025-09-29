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
from typing import Dict, Set, Type

from macrostat.core.behavior import Behavior


def generate_docs(
    behavior_class: Type[Behavior],
    output_file: str = None,
    docstyle: str = "latex",
    title: str = None,
    subsec: bool = True,
    preamble: str = None,
) -> str:
    r"""Make a model description from a Behavior class.

    This function takes a Behavior class and returns a model description by parsing
    the docstrings of the initialize() and the step() methods. It then copies the docstrings
    of those methods and all of the methods that they call. From each, it extracts the description
    and the equations section, and then formats them. It then saves the docs to a
    file if an output file is provided.

    Parameters
    ----------
    behavior_class : Type[Behavior]
        The Behavior class to make a model description from.
    output_file : str, optional
        The file to save the model description to.
    docstyle : str, default "latex"
        The type of documentation to make
    title : str, optional
        The title of the model description.
    subsec : bool, optional
        If True, add a subsection for each method. If False, just append the description and equations.
    preamble : str, optional
        A string of LaTeX code to add to the preamble of the document, i.e. before the \begin{document} command.
        Only for LaTeX

    Returns
    -------
    str
        The model description.

    Examples
    --------
    >>> from macrostat.models import get_model
    >>> GL06SIM = get_model("GL06SIM")
    >>> tex = generate_docs(GL06SIM().behavior, dostyle="latex")
    >>> print(tex)
    """
    match docstyle.lower():
        case "latex":
            content = create_latex_content(behavior_class, title, subsec, preamble)
        case "rst":
            content = create_rst_content(behavior_class, title, subsec)
        case _:
            raise ValueError("Incorrect docstyle supplied. Accepted: [latex, rst]")

    if output_file:
        with open(output_file, "w") as f:
            f.write(content)
    return content


def create_rst_content(
    behavior_class: Type[Behavior],
    title: str = None,
    subsec: bool = False,
) -> str:  # pragma: no cover
    """Generate rst content for a model's documentation. Primarily for docs.

    This function creates a complete rst document structure for documenting a model's
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
        Complete rst document as a string
    """
    title = title if title is not None else behavior_class.__name__

    # Extract the docstrings from the initialize() and step() methods
    docstrings = parse_behavior_docstrings(behavior_class)

    rst = []

    rst.append(len(title) * "=")
    rst.append(title)
    rst.append(len(title) * "=")

    # Add initialization equations
    if docstrings["initialize"]:
        txt = "Initialization Equations"
        rst.append(len(txt) * "-")
        rst.append(txt)
        rst.append(len(txt) * "-")

        if "initialize" in docstrings["initialize"]:
            rst.append(
                convert_docstring_to_rst(
                    docstrings["initialize"]["initialize"], "initialize"
                )
            )

        # Go through any methods that have been called
        for method_name, docstring in docstrings["initialize"].items():
            if method_name == "initialize":
                continue
            rst.append(convert_docstring_to_rst(docstring, method_name))

    # Add step equations
    if docstrings["step"]:
        txt = "Step Equations"
        rst.append(len(txt) * "-")
        rst.append(txt)
        rst.append(len(txt) * "-")

        if "step" in docstrings["step"]:
            rst.append(convert_docstring_to_rst(docstrings["step"]["step"], "step"))

        for count, method_name in enumerate(docstrings["step"]):
            docstring = docstrings["step"][method_name]
            rst.append(f"{count+1}. {method_name.replace('_', ' ').title()}\n")
            rst.append(convert_docstring_to_rst(docstring, method_name))
            rst.append("\n")

    return "\n".join(rst)


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


def gather_docstring_sections(docstring: str) -> dict:
    """Convert a docstring into a dict of sections and their content. The first
    section is called Description

    Parameters
    ----------
    docstring : str
        The docstring to convert to LaTeX.

    Returns
    -------
    docparts : dict[str,str]
        section name: text separation of the docstring
    """

    section = "Description"
    docparts = {section: []}

    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        if len(line) == 0:
            continue
        elif line == len(line) * "-":
            docparts[section].pop(-1)
            section = lines[i - 1].strip()
            docparts[section] = []
        else:
            docparts[section].append(line)

    return {k: "\n".join(v) for k, v in docparts.items()}


def convert_docstring_to_rst(
    docstring: str, label: str = None
) -> str:  # pragma: no cover
    """Convert a docstring to the rst markdown syntax.

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
    rst = []
    docparts = gather_docstring_sections(docstring)

    rst.append(docparts["Description"])
    equations = extract_equations_from_docstring(docstring)
    if equations:
        rst.append("\n.. math::")
        rst.append("\t" + f":label: {label}")
        rst.append("\t:nowrap:\n")
        rst.append("\t" + r"\begin{align}")
        rst.append("\t" + equations.replace("\n", "\n\t"))
        rst.append("\t" + r"\end{align}")
        rst.append("\n")

    return "\n".join(rst)


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
    equations_text = gather_docstring_sections(docstring)["Equations"]

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
