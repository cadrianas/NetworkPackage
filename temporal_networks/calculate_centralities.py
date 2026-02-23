"""
Temporal Network Analysis: Centrality Measures Module (WITH GAP REPORTING)

This module provides functions for computing various centrality measures that
quantify the importance and influence of individual nodes within networks.
Supports computation across temporal networks with proper handling of node labels
and temporal gaps.

KEY FEATURES:
- Computes multiple centrality measures (degree, betweenness, PageRank, etc.)
- Automatically detects and reports temporal gaps in your data
- Optionally visualizes how node centralities change over time
- Handles gapped data correctly in temporal plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Dict, Tuple
from datetime import datetime


# ============================================================================
# GAP DETECTION UTILITIES (same as network_properties)
# ============================================================================

def parse_flexible_datetime(label: str) -> Optional[datetime]:
    """Parse datetime label in multiple formats."""
    try:
        return datetime.strptime(label.strip(), "%Y-%m")
    except ValueError:
        pass

    try:
        return datetime.strptime(label.strip(), "%Y-%m-%d")
    except ValueError:
        pass

    try:
        year, week = label.strip().split('-W')
        return datetime.strptime(f"{year}-W{int(week)}-1", "%Y-W%W-%w")
    except (ValueError, AttributeError):
        pass

    try:
        year, quarter = label.strip().split('-Q')
        month = (int(quarter) - 1) * 3 + 1
        return datetime(int(year), month, 1)
    except (ValueError, IndexError):
        pass

    try:
        return datetime.strptime(label.strip(), "%Y")
    except ValueError:
        pass

    return None


def calculate_time_difference(date1: datetime, date2: datetime, unit: str = "months") -> float:
    """Calculate time difference between two dates."""
    if date1 > date2:
        date1, date2 = date2, date1

    if unit == "days":
        return (date2 - date1).days
    elif unit == "weeks":
        return (date2 - date1).days / 7
    elif unit == "months":
        return (date2.year - date1.year) * 12 + (date2.month - date1.month)
    elif unit == "years":
        return (date2.year - date1.year) + (date2.month - date1.month) / 12
    else:
        raise ValueError(f"Unknown unit: {unit}")


def detect_temporal_gaps(graph_labels: List[str],
                        gap_threshold: int = 1,
                        unit: str = "months") -> Dict:
    """Detect temporal gaps and return information."""

    if len(graph_labels) < 2:
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
        }

    parsed_dates = [parse_flexible_datetime(label) for label in graph_labels]

    if any(d is None for d in parsed_dates):
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
        }

    gaps = []
    segments = []
    segment_start = 0

    for i in range(1, len(parsed_dates)):
        date_prev = parsed_dates[i - 1]
        date_curr = parsed_dates[i]

        time_diff = calculate_time_difference(date_prev, date_curr, unit=unit)

        if time_diff > gap_threshold:
            gaps.append({
                "start_idx": i - 1,
                "end_idx": i,
                "start_label": graph_labels[i - 1],
                "end_label": graph_labels[i],
                "gap_size": time_diff,
            })

            segments.append((segment_start, i))
            segment_start = i

    segments.append((segment_start, len(graph_labels)))

    return {
        "has_gaps": len(gaps) > 0,
        "num_gaps": len(gaps),
        "gaps": gaps,
        "segments": segments,
    }


def print_gap_report(graph_labels: List[str], gap_info: Dict, unit: str = "months") -> None:
    """Print human-readable gap report."""

    print("\n" + "=" * 80)
    print("TEMPORAL DATA STRUCTURE ANALYSIS")
    print("=" * 80)

    print(f"\nDataset Overview:")
    print(f"  Number of observations: {len(graph_labels)}")
    print(f"  Time unit: {unit}")
    print(f"  Date range: {graph_labels[0]} to {graph_labels[-1]}")

    if not gap_info["has_gaps"]:
        print(f"\n✓ Data is CONTINUOUS (no gaps detected)")
    else:
        print(f"\n⚠ Data has GAPS: {gap_info['num_gaps']} gap(s) detected\n")

        for i, gap in enumerate(gap_info["gaps"], 1):
            print(f"  Gap #{i}:")
            print(f"    From: {gap['start_label']} (index {gap['start_idx']})")
            print(f"    To:   {gap['end_label']} (index {gap['end_idx']})")
            print(f"    Size: {gap['gap_size']:.1f} {unit}")
            print()

    print("Impact on Temporal Visualization:")
    if gap_info["has_gaps"]:
        print("  ✓ Centrality plots show SEPARATE LINE SEGMENTS for each continuous period")
        print("  ✓ No lines are drawn across gaps")
        print("  ✓ Visual breaks indicate where data is missing")
    else:
        print("  ✓ Centrality plots show CONTINUOUS LINES connecting all points")

    print("\n" + "=" * 80 + "\n")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def plot_with_gap_handling(ax, graph_labels: List[str], y_values, gap_segments: List[Tuple],
                          marker='o', linestyle='-', markersize=8,
                          linewidth=2, color='#1f77b4'):
    """Plot data with gap handling."""

    for segment_start, segment_end in gap_segments:
        x_indices = np.arange(segment_start, segment_end)
        y_segment = [y_values[i] for i in x_indices]

        ax.plot(x_indices, y_segment, marker=marker, linestyle=linestyle,
               markersize=markersize, linewidth=linewidth, color=color)

    ax.set_xticks(range(len(graph_labels)))
    ax.set_xticklabels(graph_labels, rotation=45, ha='right', fontsize=10)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def calculate_centralities(graphs: List,
                          graph_labels: Optional[List[str]] = None,
                          filename: Optional[str] = "centralities_results.csv",
                          report_gaps: bool = True,
                          visualize_evolution: bool = False,
                          save_path: str = "plots/") -> pd.DataFrame:
    """
    Compute comprehensive centrality measures for nodes across multiple graphs.

    Calculates various centrality metrics for each node in a collection of networks,
    including degree, closeness, betweenness, eigenvector, PageRank, harmonic,
    eccentricity, clustering coefficient, and HITS scores.

    **KEY FEATURE:** Automatically detects and reports temporal gaps in your data.
    Optionally visualizes how node centralities evolve over time with proper gap handling.

    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        Supports multiple formats: YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        CSV filename for saving results (default: "centralities_results.csv")
        If None, results are not saved to file
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console
    visualize_evolution : bool, optional
        If True, creates plots showing how centrality measures evolve over time
        (default: False). This is useful for tracking important nodes.
    save_path : str, optional
        Directory for saving visualizations (default: "plots/")

    Returns
    -------
    pandas.DataFrame
        DataFrame with centrality measures, one row per node per graph.
        Columns include: Graph, Node, and various centrality measures

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import calculate_centralities
    >>>
    >>> # Continuous data (no gaps)
    >>> graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
    >>> labels = ["2024-01", "2024-02", ..., "2024-12"]
    >>> centralities = calculate_centralities(graphs, graph_labels=labels)
    >>>
    >>> # Gapped data (seasonal operation)
    >>> labels_seasonal = ["2024-03", "2024-04", "2024-05", "2024-06",
    ...                    "2024-07", "2024-08", "2024-11", "2024-12"]
    >>> centralities = calculate_centralities(graphs[:8], graph_labels=labels_seasonal)
    >>> # Will report: Gap between 2024-08 and 2024-11

    Notes
    -----
    Centrality measures computed:
    - Degree: Number of direct connections
    - Closeness: Average distance to all other nodes
    - Betweenness: Number of shortest paths node lies on
    - Eigenvector: Importance based on connections to important nodes
    - PageRank: Probabilistic measure of node importance
    - Harmonic: Closeness variant less sensitive to disconnected components
    - Eccentricity: Greatest distance to any other node
    - Clustering Coefficient: Tendency of neighbors to be connected
    - HITS Authority/Hub: Measures from hyperlink-induced topic search

    Temporal Gaps:
    - Automatically analyzes temporal labels for gaps
    - Reports if data has missing periods
    - Temporal plots preserve gaps as visual breaks
    """

    # Validate inputs
    if not graphs:
        raise ValueError("graphs list cannot be empty")

    # Set up graph labels
    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(f"graph_labels length ({len(graph_labels)}) must match graphs length ({len(graphs)})")

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    # Create output directory if needed
    if save_path and visualize_evolution:
        os.makedirs(save_path, exist_ok=True)

    # Initialize list to store centrality measures for all nodes across all graphs
    centralities_list = []

    # Iterate over each graph
    for graph_idx, graph in enumerate(graphs):
        graph_name = graph_labels[graph_idx]

        # Get node labels from graph attributes
        if "name" in graph.vs.attributes():
            node_labels = graph.vs["name"]
        elif "label" in graph.vs.attributes():
            node_labels = graph.vs["label"]
        else:
            node_labels = [f"Node_{i}" for i in range(graph.vcount())]
            print(f"Warning: Graph {graph_name} has no 'name' or 'label' attribute. Using node indices.")

        # Compute centrality measures
        try:
            degree_centrality = graph.degree()
        except:
            degree_centrality = [None] * graph.vcount()

        try:
            closeness_centrality = graph.closeness()
        except:
            closeness_centrality = [None] * graph.vcount()

        try:
            betweenness_centrality = graph.betweenness(directed=graph.is_directed())
        except:
            betweenness_centrality = [None] * graph.vcount()

        try:
            eigenvector_centrality = graph.eigenvector_centrality()
        except:
            eigenvector_centrality = [None] * graph.vcount()

        try:
            pagerank = graph.pagerank()
        except:
            pagerank = [None] * graph.vcount()

        try:
            harmonic_centrality = graph.harmonic_centrality()
        except:
            harmonic_centrality = [None] * graph.vcount()

        try:
            eccentricity = graph.eccentricity()
        except:
            eccentricity = [None] * graph.vcount()

        try:
            clustering_coefficient = graph.transitivity_local_undirected()
        except:
            clustering_coefficient = [None] * graph.vcount()

        try:
            authority_score = graph.authority_score()
        except:
            authority_score = [None] * graph.vcount()

        try:
            hub_score = graph.hub_score()
        except:
            hub_score = [None] * graph.vcount()

        # For each node, store all centrality measures
        for node_idx, node_label in enumerate(node_labels):
            centrality_dict = {
                "Graph": graph_name,
                "Node": node_label,
                "Node_Index": node_idx,
                "Degree_Centrality": degree_centrality[node_idx],
                "Closeness_Centrality": closeness_centrality[node_idx],
                "Betweenness_Centrality": betweenness_centrality[node_idx],
                "Eigenvector_Centrality": eigenvector_centrality[node_idx],
                "PageRank": pagerank[node_idx],
                "Harmonic_Centrality": harmonic_centrality[node_idx],
                "Eccentricity": eccentricity[node_idx],
                "Clustering_Coefficient": clustering_coefficient[node_idx],
                "HITS_Authority": authority_score[node_idx],
                "HITS_Hub": hub_score[node_idx],
            }
            centralities_list.append(centrality_dict)

    # Convert to DataFrame
    centralities_df = pd.DataFrame(centralities_list)

    # Save to CSV if requested
    if filename:
        try:
            centralities_df.to_csv(filename, index=False)
            print(f"✓ Centralities results saved to {filename}")
        except Exception as e:
            print(f"Error saving centralities to CSV: {e}")

    # Optional: Visualize centrality evolution over time
    if visualize_evolution:
        _visualize_centrality_evolution(centralities_df, graph_labels, gap_info, save_path)

    return centralities_df


def _visualize_centrality_evolution(centralities_df: pd.DataFrame,
                                   graph_labels: List[str],
                                   gap_info: Dict,
                                   save_path: str) -> None:
    """
    Visualize how centrality measures evolve over time.

    Creates plots showing temporal evolution of average centrality measures
    across all nodes, with proper gap handling.
    """

    # Calculate average centrality measures per time step
    centrality_measures = [
        "Degree_Centrality",
        "Betweenness_Centrality",
        "PageRank",
        "Closeness_Centrality",
    ]

    for measure in centrality_measures:
        if measure not in centralities_df.columns:
            continue

        try:
            # Get average centrality for each graph
            avg_by_graph = centralities_df.groupby("Graph")[measure].mean().reset_index()
            avg_values = avg_by_graph[measure].values

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

            # Plot with gap handling
            plot_with_gap_handling(ax, graph_labels, avg_values,
                                  gap_info["segments"],
                                  marker='o', linestyle='-', markersize=8,
                                  linewidth=2, color='#1f77b4')

            ax.set_xlabel("Year - Month", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"Average {measure.replace('_', ' ')}", fontsize=12, fontweight='bold')
            ax.set_title(f"Temporal Evolution: {measure.replace('_', ' ')}", fontsize=14, fontweight='bold')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_filename = os.path.join(save_path, f"centrality_evolution_{measure}.pdf")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Centrality evolution plot saved: {plot_filename}")

        except Exception as e:
            print(f"Warning: Could not plot {measure}: {e}")