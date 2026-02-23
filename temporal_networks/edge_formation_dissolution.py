"""
Temporal Network Analysis: Edge Dynamics Module (WITH GAP REPORTING)

This module provides functions for computing and visualizing edge formation
and dissolution patterns across temporal networks. Properly handles temporal
gaps in the data.

KEY FEATURES:
- Automatically detects and reports temporal gaps
- Plots correctly show gaps as visual breaks
- Analyzes edge dynamics (routes that form and dissolve)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from typing import List, Optional, Dict, Tuple


# ============================================================================
# GAP DETECTION UTILITIES
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
        print("  ✓ Edge dynamics plots show SEPARATE LINE SEGMENTS for each continuous period")
        print("  ✓ No lines are drawn across gaps")
        print("  ✓ Visual breaks indicate where data is missing")
    else:
        print("  ✓ Edge dynamics plots show CONTINUOUS LINES connecting all points")

    print("\n" + "=" * 80 + "\n")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

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


def plot_with_gap_handling(ax, graph_labels: List[str], y_values, gap_segments: List[Tuple],
                          marker='o', linestyle='-', markersize=10,
                          linewidth=3, color='#1f77b4'):
    """Plot data with gap handling."""

    for segment_start, segment_end in gap_segments:
        x_indices = np.arange(segment_start, segment_end)
        y_segment = [y_values[i] for i in x_indices]

        ax.plot(x_indices, y_segment, marker=marker, linestyle=linestyle,
               markersize=markersize, linewidth=linewidth, color=color)

    ax.set_xticks(range(len(graph_labels)))
    ax.set_xticklabels(graph_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

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
                      graph_labels: List[str],
                      gap_info: Dict,
                      metric: str = "Edges_Formed",
                      output_file: Optional[str] = None,
                      save_path: str = "plots/") -> None:
    """
    Plot edge formation/dissolution dynamics over time with gap handling.

    Creates a line plot showing how edges are formed or dissolved at each
    time step. Properly shows gaps in temporal data as visual breaks.

    Parameters
    ----------
    dynamics_df : pd.DataFrame
        DataFrame from compute_edge_dynamics()
    graph_labels : list of str
        Temporal labels
    gap_info : dict
        Gap detection information
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
    The plot uses gap-aware plotting to preserve temporal gaps.
    If you have data for Jan, Mar (Feb missing), the plot will show that
    gap visually instead of drawing a false line.
    """

    if metric not in dynamics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe. "
                        f"Available: {list(dynamics_df.columns)}")

    os.makedirs(save_path, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

        y_values = dynamics_df[metric].values

        # FIXED: Use gap-aware plotting
        plot_with_gap_handling(ax, graph_labels, y_values,
                              gap_info["segments"],
                              marker='o', linestyle='-', markersize=12,
                              linewidth=3, color='#1f77b4')

        ax.set_xlabel("Year - Month", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Edges", fontsize=14, fontweight='bold')
        ax.set_title(f"{metric.replace('_', ' ')} Over Time", fontsize=16, fontweight='bold')

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
                  save_path: str = "plots/",
                  report_gaps: bool = True) -> pd.DataFrame:
    """
    Analyze and plot edge formation over time.

    Convenience function combining edge dynamics computation and visualization
    for edge formation specifically. Automatically detects and reports temporal
    gaps.

    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    save_path : str, optional
        Directory for saving plots (default: "plots/")
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    pandas.DataFrame
        DataFrame with edge dynamics
    """

    # Set up graph labels
    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(f"graph_labels length ({len(graph_labels)}) must match graphs length ({len(graphs)})")

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    print("Computing edge formation...")
    dynamics_df = compute_edge_dynamics(graphs, graph_labels=graph_labels)

    print("Plotting edge formation...")
    plot_edge_dynamics(dynamics_df, graph_labels, gap_info,
                      metric="Edges_Formed",
                      output_file="edges_formed.pdf", save_path=save_path)

    return dynamics_df


def edge_dissolution(graphs: List,
                    graph_labels: Optional[List[str]] = None,
                    save_path: str = "plots/",
                    report_gaps: bool = True) -> pd.DataFrame:
    """
    Analyze and plot edge dissolution over time.

    Convenience function combining edge dynamics computation and visualization
    for edge dissolution specifically. Automatically detects and reports temporal
    gaps.

    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    save_path : str, optional
        Directory for saving plots (default: "plots/")
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    pandas.DataFrame
        DataFrame with edge dynamics
    """

    # Set up graph labels
    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(f"graph_labels length ({len(graph_labels)}) must match graphs length ({len(graphs)})")

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    print("Computing edge dissolution...")
    dynamics_df = compute_edge_dynamics(graphs, graph_labels=graph_labels)

    print("Plotting edge dissolution...")
    plot_edge_dynamics(dynamics_df, graph_labels, gap_info,
                      metric="Edges_Dissolved",
                      output_file="edges_dissolved.pdf", save_path=save_path)

    return dynamics_df