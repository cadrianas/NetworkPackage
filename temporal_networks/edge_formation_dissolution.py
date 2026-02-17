"""
Temporal Network Analysis: Edge Dynamics Module

This module provides functions for computing and visualizing edge formation
and dissolution patterns across temporal networks. Properly handles temporal
gaps in the data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
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


def compute_edge_dynamics(graphs: List,
                         graph_labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute edge formation and dissolution between consecutive graphs.
    
    Analyzes how the edge structure changes from one temporal snapshot to
    the next, computing the number of edges that appear (formed) and disappear
    (dissolved) at each time step.
    
    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects representing consecutive time points
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - Graph: Label for the current time point
        - Edges_Formed: Number of new edges created
        - Edges_Dissolved: Number of edges removed
        - Edges_Formed_Percent: Percentage change relative to previous graph
        - Edges_Dissolved_Percent: Percentage change relative to previous graph
        
    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import compute_edge_dynamics
    >>> graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
    >>> labels = ["2019-01", "2019-02", ..., "2019-12"]
    >>> dynamics = compute_edge_dynamics(graphs, graph_labels=labels)
    
    Notes
    -----
    Comparison starts from the second graph. The first row corresponds to
    differences between Graph 0 and Graph 1.
    """
    
    # Validate inputs
    if len(graphs) < 2:
        raise ValueError("At least 2 graphs required to compute edge dynamics")
    
    # Set up graph labels
    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(f"graph_labels length ({len(graph_labels)}) must match graphs length ({len(graphs)})")
    
    dynamics_data = []
    
    # Compare consecutive graphs
    for i in range(1, len(graphs)):
        g_prev = graphs[i-1]
        g_curr = graphs[i]
        
        try:
            # Get edge sets (as tuples for set operations)
            prev_edges = set(g_prev.get_edgelist())
            curr_edges = set(g_curr.get_edgelist())
            
            # Compute edges formed and dissolved
            edges_formed = curr_edges - prev_edges
            edges_dissolved = prev_edges - curr_edges
            
            num_formed = len(edges_formed)
            num_dissolved = len(edges_dissolved)
            
            # Compute percentages (relative to previous graph edge count)
            prev_edge_count = len(prev_edges)
            formed_percent = (num_formed / prev_edge_count * 100) if prev_edge_count > 0 else 0
            dissolved_percent = (num_dissolved / prev_edge_count * 100) if prev_edge_count > 0 else 0
            
            dynamics_data.append({
                "Graph": graph_labels[i],
                "Edges_Formed": num_formed,
                "Edges_Dissolved": num_dissolved,
                "Edges_Formed_Percent": formed_percent,
                "Edges_Dissolved_Percent": dissolved_percent,
            })
            
        except Exception as e:
            print(f"Warning: Error comparing graphs {i-1} and {i}: {e}")
            continue
    
    dynamics_df = pd.DataFrame(dynamics_data)
    return dynamics_df


def plot_edge_dynamics(dynamics_df: pd.DataFrame,
                      metric: str = "Edges_Formed",
                      output_file: Optional[str] = None,
                      save_path: str = "plots/") -> None:
    """
    Plot edge formation/dissolution dynamics over time.
    
    Creates a line plot showing how edges are formed or dissolved at each
    time step. Properly shows gaps in temporal data (e.g., missing months).
    
    Parameters
    ----------
    dynamics_df : pd.DataFrame
        DataFrame from compute_edge_dynamics()
    metric : str, optional
        Metric to plot: "Edges_Formed", "Edges_Dissolved", etc.
        (default: "Edges_Formed")
    output_file : str, optional
        Filename for saving the plot. If None, uses metric name.
        (default: None → "{metric}.pdf")
    save_path : str, optional
        Directory for saving plots (default: "plots/")
        
    Returns
    -------
    None
        Saves plot to file
        
    Notes
    -----
    The plot uses x-axis positions that preserve temporal gaps. If you have
    data for Jan, Mar (Feb missing), the plot will show that gap visually.
    """
    
    if metric not in dynamics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe. "
                        f"Available: {list(dynamics_df.columns)}")
    
    os.makedirs(save_path, exist_ok=True)
    
    try:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
        
        # Use index positions (0, 1, 2, ...) for x-axis
        # This preserves gaps because we're only plotting rows that exist
        x_range = range(len(dynamics_df))
        y_values = dynamics_df[metric]
        
        # Plot line with markers
        ax.plot(x_range, y_values, marker='o', linestyle='-',
               markersize=12, linewidth=3, color='#1f77b4')
        
        ax.set_xlabel("Year - Month", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Edges", fontsize=14, fontweight='bold')
        ax.set_title(f"{metric.replace('_', ' ')} Over Time", fontsize=16, fontweight='bold')
        
        # Set x-ticks to show labels at regular intervals
        # This ensures we see dates, not just indices
        tick_positions = range(0, len(dynamics_df), max(1, len(dynamics_df)//12))
        tick_labels = [dynamics_df["Graph"].iloc[i] if i < len(dynamics_df) 
                      else "" for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', 
                          fontsize=12, fontweight='bold')
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
        plt.yticks(fontsize=12, fontweight='bold')
        
        # Grid and layout
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        if output_file is None:
            output_file = f"{metric}.pdf"
        
        plot_path = os.path.join(save_path, output_file)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Plot saved: {plot_path}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")


def edge_formation(graphs: List,
                  graph_labels: Optional[List[str]] = None,
                  save_path: str = "plots/") -> pd.DataFrame:
    """
    Analyze and plot edge formation over time.
    
    Convenience function combining edge dynamics computation and visualization
    for edge formation specifically.
    
    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects
    graph_labels : list of str, optional
        Labels for each graph
    save_path : str, optional
        Directory for saving plots
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with edge dynamics
    """
    
    print("Computing edge formation...")
    dynamics_df = compute_edge_dynamics(graphs, graph_labels=graph_labels)
    
    print("Plotting edge formation...")
    plot_edge_dynamics(dynamics_df, metric="Edges_Formed", 
                      output_file="edges_formed.pdf", save_path=save_path)
    
    return dynamics_df


def edge_dissolution(graphs: List,
                    graph_labels: Optional[List[str]] = None,
                    save_path: str = "plots/") -> pd.DataFrame:
    """
    Analyze and plot edge dissolution over time.
    
    Convenience function combining edge dynamics computation and visualization
    for edge dissolution specifically.
    
    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects
    graph_labels : list of str, optional
        Labels for each graph
    save_path : str, optional
        Directory for saving plots
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with edge dynamics
    """
    
    print("Computing edge dissolution...")
    dynamics_df = compute_edge_dynamics(graphs, graph_labels=graph_labels)
    
    print("Plotting edge dissolution...")
    plot_edge_dynamics(dynamics_df, metric="Edges_Dissolved",
                      output_file="edges_dissolved.pdf", save_path=save_path)
    
    return dynamics_df
