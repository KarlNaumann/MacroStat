from typing import Type

import matplotlib.patches as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from macrostat.core import Model


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

    def plot_flowchart(self):
        """Plot a flowchart of the model based on the adjacency matrix.

        This method creates a directed graph visualization of the model's structure
        using NetworkX and Matplotlib. The graph shows the relationships between
        variables as directed edges, with nodes representing variables.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the flowchart visualization.
        """

        # Add column
        edgelist = self.adjacency_matrix.copy(deep=True)
        edgelist = edgelist.stack([0, 1], future_stack=True)
        edgelist.name = "weight"
        edgelist = edgelist[edgelist != 0]
        # Add the node names
        edgelist = edgelist.reset_index()
        edgelist["source"] = edgelist["source_type"] + ":" + edgelist["source_name"]
        edgelist["target"] = edgelist["target_type"] + ":" + edgelist["target_name"]
        # Reduce

        G = nx.from_pandas_edgelist(
            edgelist, source="source", target="target", create_using=nx.DiGraph
        )

        # Color nodes by type
        node_colormap = {
            "state": "blue",
            "parameters": "green",
            "scenario": "yellow",
            "prior": "red",
        }
        node_colors = nx.get_node_attributes(G, "color")
        if not node_colors:  # If attributes don't exist yet
            nx.set_node_attributes(
                G,
                {node: node_colormap[node.split(":")[0]] for node in G.nodes()},
                "color",
            )
            node_colors = nx.get_node_attributes(G, "color")
        node_colors = list(node_colors.values())

        # Node labels are just the item after the colon
        nx.set_node_attributes(
            G,
            {
                node: "".join(
                    c if i == 0 else ("\n" + c if c.isupper() else c)
                    for i, c in enumerate(node.split(":")[1])
                )
                for node in G.nodes()
            },
            "label",
        )
        nx.set_node_attributes(
            G, {node: node.split(":")[0] for node in G.nodes()}, "subset"
        )

        # Create figure
        plt.figure(figsize=(12, 12))

        pos = nx.spectral_layout(G)

        # Draw the graph
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=nx.get_node_attributes(G, "label"),
            node_color=nx.get_node_attributes(G, "color").values(),
            node_size=2000,
            font_size=10,
            arrows=True,
            arrowsize=20,
            edge_color="gray",
            width=1.5,
            node_shape="s",  # 's' specifies square/rectangular nodes
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )

        # Add legend
        legend_elements = [
            mpl.Patch(facecolor=v, label=k) for k, v in node_colormap.items()
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        plt.title("Model Structure Flowchart", pad=20)

        return plt.gcf()
