"""
Temporal Network Analysis: Network Properties Module

This module provides functions for computing structural properties of networks,
including metrics like density, diameter, clustering coefficients, and more.
Supports temporal analysis by accepting lists of networks with associated labels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional


def format_large_numbers(x, pos):
    """Format large numbers with appropriate units (k, M, B)."""
    if x >= 1_000_000_000:
        return f'{x/1_000_000_000:.1f}B'
    elif x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.1f}k'
    else:
        return f'{x:.0f}'


def network_properties(graphs: List, 
                      graph_labels: Optional[List[str]] = None,
                      filename: Optional[str] = None, 
                      save_path: str = "plots/", 
                      visualisation: bool = True) -> pd.DataFrame:
    """
    Compute comprehensive network properties for a collection of graphs.
    
    This function systematically analyzes a collection of networks, extracting 
    multiple properties including node/edge counts, density, connectivity measures,
    and clustering coefficients. Supports temporal analysis with proper handling of
    missing or sparse time periods.
    
    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If provided, these are used in plots to show temporal gaps correctly.
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        If provided, saves numerical results to CSV with this filename
    save_path : str, optional
        Directory path for saving visualizations (default: "plots/")
    visualisation : bool, optional
        If True (default), generates plots for each numerical property
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing network properties for each graph,
        with one row per graph and columns for each metric
        
    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import network_properties
    >>> G1 = ig.Graph.Barabasi(n=100, m=2)
    >>> G2 = ig.Graph.Barabasi(n=100, m=2)
    >>> graphs = [G1, G2]
    >>> labels = ["2019-01", "2019-02"]
    >>> props = network_properties(graphs, graph_labels=labels)
    
    Notes
    -----
    Properties computed include:
    - Basic metrics: node count, edge count, density
    - Connectivity: diameter, average path length, strongly connected components
    - Local structure: clustering coefficient, reciprocity
    - Graph type: directed, weighted, bipartite, etc.
    """
    
    # Validate inputs
    if not graphs:
        raise ValueError("graphs list cannot be empty")
    
    # Set up graph labels
    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(f"graph_labels length ({len(graph_labels)}) must match graphs length ({len(graphs)})")
    
    # Create output directory
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Initialize lists to store properties
    num_vertices = []
    num_edges = []
    density = []
    strongly_connected_components = []
    girth = []
    diameter = []
    avg_path_length = []
    mean_degree = []
    reciprocity = []
    transitivity = []
    
    # Additional graph type properties
    is_bipartite = []
    is_connected = []
    is_dag = []
    is_directed = []
    is_named = []
    is_simple = []
    is_weighted = []
    has_multiple = []

    # Loop over each graph and compute properties
    for graph in graphs:
        try:
            num_vertices.append(graph.vcount())
            num_edges.append(graph.ecount())
            density.append(graph.density())
            
            # Connectivity properties
            strongly_connected_components.append(len(graph.components(mode="STRONG")))
            
            # Only compute diameter/girth if graph is sufficiently connected
            try:
                diameter.append(graph.diameter())
            except:
                diameter.append(np.nan)
            
            try:
                girth.append(graph.girth())
            except:
                girth.append(np.nan)
            
            # Path length (handle potential errors for disconnected graphs)
            try:
                avg_path_length.append(np.mean(graph.shortest_paths()))
            except:
                avg_path_length.append(np.nan)
            
            mean_degree.append(np.mean(graph.degree()))
            reciprocity.append(graph.reciprocity())
            
            # Transitivity (clustering coefficient)
            try:
                transitivity.append(graph.transitivity_undirected())
            except:
                transitivity.append(np.nan)
            
            # Graph type properties
            is_bipartite.append(graph.is_bipartite())
            is_connected.append(graph.is_connected())
            is_dag.append(graph.is_dag())
            is_directed.append(graph.is_directed())
            is_named.append(graph.is_named())
            is_simple.append(graph.is_simple())
            is_weighted.append(graph.is_weighted())
            has_multiple.append(graph.has_multiple())
            
        except Exception as e:
            print(f"Warning: Error processing graph {graph_labels[len(num_vertices)]}: {e}")
            continue

    # Create DataFrame with network properties
    network_data = pd.DataFrame({
        "Graph": graph_labels[:len(num_vertices)],
        "Number of Nodes": num_vertices,
        "Number of Edges": num_edges,
        "Density": density,
        "Strongly Connected Components": strongly_connected_components,
        "Girth": girth,
        "Diameter": diameter,
        "Average Path Length": avg_path_length,
        "Mean Degree": mean_degree,
        "Reciprocity": reciprocity,
        "Transitivity": transitivity,
        "Is Bipartite": is_bipartite,
        "Is Connected": is_connected,
        "Is DAG": is_dag,
        "Is Directed": is_directed,
        "Is Named": is_named,
        "Is Simple": is_simple,
        "Is Weighted": is_weighted,
        "Has Multiple": has_multiple,
    })

    # Save to CSV if requested
    if filename:
        save_path_file = os.path.join(save_path, filename) if save_path else filename
        try:
            network_data.to_csv(save_path_file, index=False)
            print(f"✓ Network properties saved to {save_path_file}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    # Generate visualizations
    if visualisation:
        properties_to_plot = [
            "Number of Nodes",
            "Number of Edges",
            "Density",
            "Diameter",
            "Average Path Length",
            "Mean Degree",
            "Reciprocity",
            "Transitivity",
        ]

        for prop in properties_to_plot:
            if prop not in network_data.columns:
                continue
            
            try:
                fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
                
                # Use graph labels on x-axis (preserves gaps)
                x_range = range(len(network_data))
                y_values = network_data[prop]
                
                ax.plot(x_range, y_values, marker='o', linestyle='-', 
                       markersize=10, linewidth=2, color='#1f77b4')
                
                ax.set_xlabel("Year - Month", fontsize=14, fontweight='bold')
                ax.set_ylabel(prop, fontsize=14, fontweight='bold')
                
                # Set x-ticks to show labels at regular intervals
                tick_positions = range(0, len(network_data), max(1, len(network_data)//10))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([network_data["Graph"].iloc[i] for i in tick_positions], 
                                  rotation=45, ha='right', fontsize=12, fontweight='bold')
                
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
                plt.yticks(fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                plot_filename = os.path.join(save_path, f"{prop}.pdf")
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Plot saved: {plot_filename}")
                
            except Exception as e:
                print(f"Warning: Could not plot {prop}: {e}")

    return network_data
