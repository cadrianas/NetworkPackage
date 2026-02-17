"""
Temporal Network Analysis: Vertex Properties Module

This module provides functions for analyzing properties of individual nodes
across temporal networks, tracking how node importance and characteristics
change over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Optional


def vertex_properties(graphs: List,
                     node_name: str,
                     graph_labels: Optional[List[str]] = None,
                     filename: Optional[str] = None,
                     save_path: str = "plots/",
                     visualisation: bool = True) -> pd.DataFrame:
    """
    Compute and visualize properties of a specific node across multiple graphs.
    
    Tracks how various metrics for a single node evolve over time across a
    temporal network. Computes centrality measures, structural properties,
    and constraint metrics for the specified node.
    
    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects to analyze
    node_name : str
        Name/label of the node of interest (must match node attribute in graphs)
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        CSV filename for saving results. If None, results are not saved.
    save_path : str, optional
        Directory for saving visualizations (default: "plots/")
    visualisation : bool, optional
        If True (default), generates plots for each property over time
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per graph, showing how the node's properties
        evolve across time. Includes columns for each computed metric.
        
    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import vertex_properties
    >>> G1 = ig.Graph.Barabasi(n=50, m=2)
    >>> G2 = ig.Graph.Barabasi(n=50, m=2)
    >>> G1.vs["name"] = [f"Node_{i}" for i in range(50)]
    >>> G2.vs["name"] = [f"Node_{i}" for i in range(50)]
    >>> graphs = [G1, G2]
    >>> labels = ["2019-01", "2019-02"]
    >>> props = vertex_properties(graphs, node_name="Node_5", graph_labels=labels)
    
    Notes
    -----
    Properties computed for the node:
    - Centrality: degree, closeness, betweenness, eigenvector, PageRank,
      harmonic, eccentricity
    - Local structure: clustering coefficient, constraint
    - Importance: authority score, hub score, coreness
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
    if save_path and visualisation:
        os.makedirs(save_path, exist_ok=True)

    # Initialize dictionary to store properties
    properties = {
        "Graph": [],
        "Degree_Centrality": [],
        "Closeness_Centrality": [],
        "Betweenness_Centrality": [],
        "Eigenvector_Centrality": [],
        "PageRank": [],
        "Harmonic_Centrality": [],
        "Eccentricity": [],
        "Clustering_Coefficient": [],
        "Constraint": [],
        "Coreness": [],
        "Authority_Score": [],
        "Hub_Score": [],
    }

    # Loop over each graph and extract node properties
    for graph_idx, graph in enumerate(graphs):
        graph_name = graph_labels[graph_idx]
        
        # Find node index by name/label
        node_index = None
        if "name" in graph.vs.attributes() and node_name in graph.vs["name"]:
            node_index = graph.vs.find(name=node_name).index
        elif "label" in graph.vs.attributes() and node_name in graph.vs["label"]:
            node_index = graph.vs.find(label=node_name).index
        else:
            print(f"Warning: Node '{node_name}' not found in graph {graph_name}. Skipping.")
            continue
        
        # Record graph name
        properties["Graph"].append(graph_name)
        
        # Compute and store centrality measures
        try:
            properties["Degree_Centrality"].append(graph.degree()[node_index])
        except:
            properties["Degree_Centrality"].append(None)
        
        try:
            properties["Closeness_Centrality"].append(graph.closeness()[node_index])
        except:
            properties["Closeness_Centrality"].append(None)
        
        try:
            properties["Betweenness_Centrality"].append(graph.betweenness()[node_index])
        except:
            properties["Betweenness_Centrality"].append(None)
        
        try:
            properties["Eigenvector_Centrality"].append(graph.eigenvector_centrality()[node_index])
        except:
            properties["Eigenvector_Centrality"].append(None)
        
        try:
            properties["PageRank"].append(graph.pagerank()[node_index])
        except:
            properties["PageRank"].append(None)
        
        try:
            properties["Harmonic_Centrality"].append(graph.harmonic_centrality()[node_index])
        except:
            properties["Harmonic_Centrality"].append(None)
        
        try:
            properties["Eccentricity"].append(graph.eccentricity()[node_index])
        except:
            properties["Eccentricity"].append(None)
        
        try:
            properties["Clustering_Coefficient"].append(graph.transitivity_local_undirected()[node_index])
        except:
            properties["Clustering_Coefficient"].append(None)
        
        try:
            properties["Constraint"].append(graph.constraint()[node_index])
        except:
            properties["Constraint"].append(None)
        
        try:
            properties["Coreness"].append(graph.coreness()[node_index])
        except:
            properties["Coreness"].append(None)
        
        try:
            properties["Authority_Score"].append(graph.authority_score()[node_index])
        except:
            properties["Authority_Score"].append(None)
        
        try:
            properties["Hub_Score"].append(graph.hub_score()[node_index])
        except:
            properties["Hub_Score"].append(None)

    # Convert to DataFrame
    properties_df = pd.DataFrame(properties)

    # Save to CSV if requested
    if filename:
        save_path_file = os.path.join(save_path, filename) if save_path else filename
        try:
            properties_df.to_csv(save_path_file, index=False)
            print(f"✓ Vertex properties saved to {save_path_file}")
        except Exception as e:
            print(f"Error saving vertex properties: {e}")

    # Generate visualizations
    if visualisation:
        properties_to_plot = [
            "Degree_Centrality",
            "Closeness_Centrality",
            "Betweenness_Centrality",
            "Eigenvector_Centrality",
            "PageRank",
            "Harmonic_Centrality",
            "Eccentricity",
            "Clustering_Coefficient",
            "Constraint",
            "Coreness",
            "Authority_Score",
            "Hub_Score",
        ]
        
        for prop in properties_to_plot:
            try:
                # Check if property has valid data
                if prop not in properties_df.columns or properties_df[prop].isna().all():
                    continue
                
                fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
                
                x_range = range(len(properties_df))
                y_values = properties_df[prop]
                
                ax.plot(x_range, y_values, marker='o', linestyle='-',
                       markersize=10, linewidth=2, color='#1f77b4')
                
                ax.set_xlabel("Year - Month", fontsize=12, fontweight='bold')
                ax.set_ylabel(prop.replace('_', ' '), fontsize=12, fontweight='bold')
                ax.set_title(f"{prop.replace('_', ' ')} for {node_name}", fontsize=14, fontweight='bold')
                
                # Set x-ticks to show labels at regular intervals
                tick_positions = range(0, len(properties_df), max(1, len(properties_df)//10))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([properties_df["Graph"].iloc[i] for i in tick_positions],
                                  rotation=45, ha='right', fontsize=10, fontweight='bold')
                
                plt.yticks(fontsize=10, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                plot_filename = os.path.join(save_path, f"{node_name}_{prop}.pdf")
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Plot saved: {plot_filename}")
                
            except Exception as e:
                print(f"Warning: Could not plot {prop}: {e}")
        
        # Create combined plot with all properties
        try:
            fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
            
            x_range = range(len(properties_df))
            
            # Plot all numeric properties
            for prop in properties_to_plot:
                if prop in properties_df.columns and not properties_df[prop].isna().all():
                    # Normalize values for visibility (optional, comment out to skip)
                    y_values = properties_df[prop]
                    if y_values.max() > 0:
                        y_values = y_values / y_values.max()
                    ax.plot(x_range, y_values, marker='o', label=prop.replace('_', ' '), linewidth=2)
            
            ax.set_xlabel("Year - Month", fontsize=12, fontweight='bold')
            ax.set_ylabel("Normalized Value", fontsize=12, fontweight='bold')
            ax.set_title(f"All Properties for {node_name}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            
            # Set x-ticks
            tick_positions = range(0, len(properties_df), max(1, len(properties_df)//10))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([properties_df["Graph"].iloc[i] for i in tick_positions],
                              rotation=45, ha='right', fontsize=10, fontweight='bold')
            
            plt.yticks(fontsize=10, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save combined plot
            combined_filename = os.path.join(save_path, f"{node_name}_all_properties.pdf")
            fig.savefig(combined_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Combined plot saved: {combined_filename}")
            
        except Exception as e:
            print(f"Warning: Could not create combined plot: {e}")

    return properties_df
