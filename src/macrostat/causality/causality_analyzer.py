from typing import Type

import dash
import dash_cytoscape as cyto
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dash import html

from macrostat.core import Model

# Register the extra layouts
cyto.load_extra_layouts()


class CausalityAnalyzer:
    def __init__(self, model_class: Type[Model]):
        self.model_class = model_class
        self.adjacency_matrix = None
        self._dependency = {}

    def analyze(self):
        """Analyze a model class and return dependency dictionary"""
        raise NotImplementedError("Subclasses must implement this method")

    def build_adjacency_matrix(self) -> pd.DataFrame:
        """Build adjacency matrix from dependencies"""
        raise NotImplementedError("Subclasses must implement this method")

    def check_for_cycles(self):
        """Check for cycles in the model's dependency graph.

        While cycles are not necessarily a problem, cycling logic in a given
        step may indicate indeterminism. This method returns a list of cycles
        found in the created adjacency matrix.

        Returns
        -------
        list
            List of cycles found in the graph. Each cycle is a list of nodes.
            Returns empty list if no cycles are found.
        """
        if self.adjacency_matrix is None:
            self.adjacency_matrix = self.analyze()

        # Create edgelist from adjacency matrix
        edgelist = self._create_edgelist()

        # Create directed graph
        G = nx.DiGraph()
        G.add_edges_from(
            [(row["source"], row["target"]) for _, row in edgelist.iterrows()]
        )

        # Find cycles using simple_cycles
        cycles = list(nx.recursive_simple_cycles(G))

        return cycles

    def plot_heatmap(self):  # pragma: no cover
        """Visualize the adjacency matrix organized by trophic levels.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size in inches (width, height), by default (12, 10)
        cmap : str, optional
            Colormap to use, by default 'viridis'

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plot
        """
        if self.adjacency_matrix is None:
            self.analyze()

        # Get the matrix data
        matrix = self.adjacency_matrix.values

        # Calculate trophic levels using the adjacency matrix
        # A variable's trophic level is 1 + max(trophic level of its dependencies)
        n = len(matrix)
        trophic_levels = np.zeros(n)
        max_iter = n  # Maximum number of iterations to prevent infinite loops

        for _ in range(max_iter):
            new_levels = np.zeros(n)
            for i in range(n):
                # Find all variables that this variable depends on
                dependencies = np.where(matrix[i, :] > 0)[0]
                if len(dependencies) > 0:
                    new_levels[i] = 1 + np.max(trophic_levels[dependencies])
                else:
                    new_levels[i] = 1  # Base level for variables with no dependencies

            if np.allclose(new_levels, trophic_levels):
                break
            trophic_levels = new_levels

        # Sort indices by trophic level
        sorted_indices = np.argsort(trophic_levels)[::-1]

        # Create a new DataFrame with reordered indices
        reordered_matrix = self.adjacency_matrix.iloc[
            sorted_indices, sorted_indices
        ].copy()

        # Replace zeros with NaN for better visualization
        reordered_matrix = reordered_matrix.replace(0, np.nan)

        # Remove rows and columns that are all NaN
        reordered_matrix = reordered_matrix.dropna(axis=1, how="all")
        reordered_matrix = reordered_matrix.dropna(axis=0, how="all")

        # Create the figure
        # Calculate figure size based on number of variables
        n_rows, n_cols = reordered_matrix.shape
        figsize = (max(8, n_cols * 0.5), max(6, n_rows * 0.5))
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(reordered_matrix.values)
        ax.grid(which="major", color="lightgray", linewidth=0.5)

        # Set the labels
        ax.set_xticks(range(len(reordered_matrix.columns)))
        ax.set_yticks(range(len(reordered_matrix.index)))

        # Format the labels to show type and name
        x_labels = [f"{label[0]}\n{label[1]}" for label in reordered_matrix.columns]
        y_labels = [f"{label[0]}\n{label[1]}" for label in reordered_matrix.index]

        ax.set_xticklabels(x_labels, rotation=90, ha="center", va="top")
        ax.set_yticklabels(y_labels)

        # Add source/target labels
        ax.set_xlabel("Target Variables", labelpad=10)
        ax.set_ylabel("Source Variables", labelpad=10)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        return fig, ax

    def plot_with_cytoscape(
        self,
        port: int = 8050,
        node_styles: dict[str, dict[str, str]] = {
            "state": {"background-color": "lightblue", "border-color": "blue"},
            "parameters": {"background-color": "lightgreen", "border-color": "green"},
            "hyper": {"background-color": "lavender", "border-color": "purple"},
            "scenario": {"background-color": "lightyellow", "border-color": "yellow"},
            "prior": {"background-color": "lightpink", "border-color": "red"},
            "history": {"background-color": "lightgray", "border-color": "black"},
        },
    ):  # pragma: no cover
        """Create an interactive flowchart visualization using Dash and Cytoscape.

        This method creates a web-based interactive visualization of the model's structure
        using Dash and Cytoscape. The visualization allows for:
        - Zooming and panning
        - Node selection and highlighting
        - Edge highlighting
        - Node dragging and repositioning
        - Export to various formats

        Parameters
        ----------
        port : int, optional
            The port number to run the Dash server on, by default 8050

        Returns
        -------
        None
            Opens a web browser with the interactive visualization
        """
        if self.adjacency_matrix is None:
            self.adjacency_matrix = self.analyze()

        # Create nodes and edges for Cytoscape
        nodes = []
        edges = []

        # Create edgelist using the same method as NetworkX version
        edgelist = self._create_edgelist()
        edgelist = edgelist[
            edgelist["source_type"].isin(node_styles.keys())
            & edgelist["target_type"].isin(node_styles.keys())
        ]

        # Get unique nodes from both source and target
        all_nodes = pd.concat([edgelist["source"], edgelist["target"]]).unique()

        # Add nodes
        for node in all_nodes:
            node_type = node.split(":")[0]
            node_name = node.split(":")[1]

            node_style = node_styles.get(node_type, {})
            nodes.append(
                {
                    "data": {
                        "id": str(node),  # Ensure ID is string
                        "label": node_name,
                        "type": node_type,
                    },
                    "style": node_style,
                }
            )

        # Add edges from edgelist
        for _, row in edgelist.iterrows():
            edges.append(
                {
                    "data": {
                        "source": str(row["source"]),
                        "target": str(row["target"]),
                        "weight": float(row["weight"]),
                    }
                }
            )

        # Create Dash app
        app = dash.Dash(__name__)

        # Define the layout
        app.layout = html.Div(
            [
                html.H1("Model Structure", style={"textAlign": "center"}),
                cyto.Cytoscape(
                    id="model-graph",
                    layout={
                        "name": "dagre",
                        "rankDir": "LR",
                        "nodeSep": 50,
                        "rankSep": 200,
                        "spacingFactor": 2.0,
                    },
                    style={"width": "100%", "height": "1200px"},
                    elements=nodes + edges,
                    stylesheet=[
                        # Node styles
                        {
                            "selector": "node",
                            "style": {
                                "content": "data(label)",
                                "text-valign": "center",
                                "text-halign": "center",
                                "text-wrap": "wrap",
                                "text-max-width": "160px",
                                "font-size": "28px",
                                "font-weight": "normal",
                                "background-color": "data(background-color)",
                                "border-color": "data(border-color)",
                                "border-width": 3,
                                "width": "160px",
                                "height": "80px",
                                "padding": "25px",
                            },
                        },
                        # Edge styles
                        {
                            "selector": "edge",
                            "style": {
                                "width": 4,
                                "line-color": "#666",
                                "target-arrow-color": "#666",
                                "target-arrow-shape": "triangle",
                                "target-arrow-scale": 3,
                                "curve-style": "bezier",
                            },
                        },
                        # Hover effects
                        {
                            "selector": "node:hover",
                            "style": {
                                "background-color": "#BEE",
                                "line-color": "#000",
                                "target-arrow-color": "#000",
                                "source-arrow-color": "#000",
                                "text-outline-color": "#000",
                                "text-outline-width": 2,
                            },
                        },
                        {
                            "selector": "edge:hover",
                            "style": {
                                "width": 3,
                                "line-color": "#000",
                                "target-arrow-color": "#000",
                            },
                        },
                    ],
                ),
                html.Div(
                    # Create a legend entry for each node typeå
                    [
                        html.Div(
                            [
                                html.Div(
                                    style={
                                        "width": "20px",
                                        "height": "20px",
                                        "backgroundColor": style.get(
                                            "background-color", "#FFF"
                                        ),
                                        "border": f"3px solid {style.get('border-color', '#000')}",
                                        "marginRight": "10px",
                                        "display": "inline-block",
                                    }
                                ),
                                html.Span(
                                    node_type.title(),
                                    style={"fontSize": "16px"},
                                ),
                            ],
                            style={
                                "marginRight": (
                                    "20px" if i < len(node_styles) - 1 else ""
                                ),
                                "display": "inline-block",
                            },
                        )
                        for i, (node_type, style) in enumerate(node_styles.items())
                    ],
                    style={
                        "padding": "10px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "5px",
                        "marginBottom": "20px",
                        "display": "flex",
                        "justifyContent": "center",
                    },
                ),
            ],
        )

        # Run the app
        app.run(debug=True, port=port)

    def _create_edgelist(self):
        """Create edgelist from adjacency matrix"""
        edgelist = self.adjacency_matrix.copy(deep=True)
        edgelist = edgelist.stack([0, 1], future_stack=True)
        edgelist.name = "weight"
        edgelist = edgelist[edgelist != 0]
        edgelist = edgelist.reset_index()
        edgelist["source"] = edgelist["source_type"] + ":" + edgelist["source_name"]
        edgelist["target"] = edgelist["target_type"] + ":" + edgelist["target_name"]
        return edgelist
