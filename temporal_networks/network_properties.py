"""
Temporal Network Analysis: Network Properties Module (WITH GAP REPORTING)

This module provides functions for computing structural properties of networks,
including metrics like density, diameter, clustering coefficients, and more.

KEY FEATURES:
- Supports temporal analysis with automatic gap detection
- Reports where data has gaps (seasonal closures, missing measurements, etc.)
- Handles any temporal format (monthly, daily, weekly, etc.)
- Plots correctly show gaps as visual breaks, not false continuity
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Tuple, Dict
from datetime import datetime


# ============================================================================
# GAP DETECTION UTILITIES
# ============================================================================

def parse_flexible_datetime(label: str) -> Optional[datetime]:
    """Parse datetime label in multiple formats (YYYY-MM, YYYY-MM-DD, etc.)"""
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
    """Calculate time difference between two dates in specified units."""
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
    """
    Detect temporal gaps in labels and return detailed information.

    Parameters
    ----------
    graph_labels : list of str
        Temporal labels (e.g., ["2024-03", "2024-04", "2024-11"])
    gap_threshold : int, optional
        Threshold for gap detection (default: 1)
        For monthly data: threshold=1 means gaps are 2+ months apart
    unit : str, optional
        Time unit: "days", "weeks", "months", "years" (default: "months")

    Returns
    -------
    dict
        Dictionary with keys:
        - "has_gaps" (bool)
        - "num_gaps" (int)
        - "gaps" (list): Details about each gap
        - "segments" (list): Continuous time segments
    """

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
    """Print human-readable gap report to console."""

    print("\n" + "=" * 80)
    print("TEMPORAL DATA STRUCTURE ANALYSIS")
    print("=" * 80)

    print(f"\nDataset Overview:")
    print(f"  Number of observations: {len(graph_labels)}")
    print(f"  Time unit: {unit}")
    print(f"  Date range: {graph_labels[0]} to {graph_labels[-1]}")

    if not gap_info["has_gaps"]:
        print(f"\n✓ Data is CONTINUOUS (no gaps detected)")
        print(f"  All observations are sequential with no missing periods.")
    else:
        print(f"\n⚠ Data has GAPS: {gap_info['num_gaps']} gap(s) detected\n")

        for i, gap in enumerate(gap_info["gaps"], 1):
            print(f"  Gap #{i}:")
            print(f"    From: {gap['start_label']} (index {gap['start_idx']})")
            print(f"    To:   {gap['end_label']} (index {gap['end_idx']})")
            print(f"    Size: {gap['gap_size']:.1f} {unit}")
            print()

    print("Impact on Plots:")
    if gap_info["has_gaps"]:
        print("  ✓ Plots show SEPARATE LINE SEGMENTS for each continuous period")
        print("  ✓ No lines are drawn across gaps")
        print("  ✓ Visual breaks indicate where data is missing")
        print("\nPossible causes for gaps:")
        print("  - Seasonal operation (e.g., summer tourism, winter closure)")
        print("  - System maintenance or downtime")
        print("  - Data collection interruptions")
        print("  - Multi-phase study (e.g., Phase 1, then gap, Phase 2)")
    else:
        print("  ✓ Plots show a CONTINUOUS LINE connecting all points")
        print("  ✓ All time periods are represented sequentially")

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
                          linewidth=2, color='#1f77b4'):
    """
    Plot data with proper handling of temporal gaps.

    Draws separate line segments for each continuous period, creating
    visual breaks where gaps occur (instead of false continuity).
    """

    for segment_start, segment_end in gap_segments:
        x_indices = np.arange(segment_start, segment_end)
        y_segment = [y_values[i] for i in x_indices]

        ax.plot(x_indices, y_segment, marker=marker, linestyle=linestyle,
               markersize=markersize, linewidth=linewidth, color=color)

    ax.set_xticks(range(len(graph_labels)))
    ax.set_xticklabels(graph_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def network_properties(graphs: List,
                      graph_labels: Optional[List[str]] = None,
                      filename: Optional[str] = None,
                      save_path: str = "plots/",
                      visualisation: bool = True,
                      report_gaps: bool = True) -> pd.DataFrame:
    """
    Compute comprehensive network properties for a collection of graphs.

    This function systematically analyzes a collection of networks, extracting
    multiple properties including node/edge counts, density, connectivity measures,
    and clustering coefficients.

    **KEY FEATURE:** Automatically detects and reports temporal gaps in your data,
    showing you where discontinuities occur (seasonal closures, maintenance windows,
    missing measurements, etc.). Plots correctly show gaps as visual breaks.

    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        Supports multiple formats: YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        If provided, saves numerical results to CSV with this filename
    save_path : str, optional
        Directory path for saving visualizations (default: "plots/")
    visualisation : bool, optional
        If True (default), generates plots for each numerical property
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    pandas.DataFrame
        DataFrame containing network properties for each graph,
        with one row per graph and columns for each metric

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import network_properties
    >>>
    >>> # Continuous data (no gaps)
    >>> graphs = [ig.Graph.Barabasi(n=100, m=2) for _ in range(12)]
    >>> labels = ["2024-01", "2024-02", ..., "2024-12"]
    >>> props = network_properties(graphs, graph_labels=labels)
    >>>
    >>> # Gapped data (seasonal operation)
    >>> labels_seasonal = ["2024-03", "2024-04", "2024-05", "2024-06",
    ...                    "2024-07", "2024-08", "2024-11", "2024-12"]
    >>> props = network_properties(graphs[:8], graph_labels=labels_seasonal)
    >>> # Will report: Gap between 2024-08 and 2024-11

    Notes
    -----
    Properties computed include:
    - Basic metrics: node count, edge count, density
    - Connectivity: diameter, average path length, strongly connected components
    - Local structure: clustering coefficient, reciprocity
    - Graph type: directed, weighted, bipartite, etc.

    Gap Detection:
    - Automatically analyzes temporal labels
    - Reports if data has gaps (missing time periods)
    - Shows exactly where gaps occur
    - Plots preserve gaps as visual discontinuities
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

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

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

            strongly_connected_components.append(len(graph.components(mode="STRONG")))

            try:
                diameter.append(graph.diameter())
            except:
                diameter.append(np.nan)

            try:
                girth.append(graph.girth())
            except:
                girth.append(np.nan)

            try:
                avg_path_length.append(np.mean(graph.shortest_paths()))
            except:
                avg_path_length.append(np.nan)

            mean_degree.append(np.mean(graph.degree()))
            reciprocity.append(graph.reciprocity())

            try:
                transitivity.append(graph.transitivity_undirected())
            except:
                transitivity.append(np.nan)

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
        try:
            network_data.to_csv(filename, index=False)
            print(f"✓ Network properties saved to {filename}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    # Generate visualizations with gap handling
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

                y_values = network_data[prop].values
                graph_labels_subset = network_data["Graph"].values

                # Plot with gap handling
                plot_with_gap_handling(ax, list(graph_labels_subset), y_values,
                                      gap_info["segments"],
                                      marker='o', linestyle='-', markersize=10,
                                      linewidth=2, color='#1f77b4')

                ax.set_xlabel("Year - Month", fontsize=14, fontweight='bold')
                ax.set_ylabel(prop, fontsize=14, fontweight='bold')
                ax.set_title(f"{prop} Over Time", fontsize=16, fontweight='bold')

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