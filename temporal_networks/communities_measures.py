"""
Temporal Network Analysis: Community Detection Module (WITH GAP REPORTING)

This module provides functions for detecting and analyzing community structures
in networks using multiple algorithms. Supports temporal analysis by tracking
how communities evolve over time.

KEY FEATURES:
- Applies 7 different community detection algorithms
- Automatically detects and reports temporal gaps
- Plots correctly show gaps as visual breaks
- Tracks community structure evolution
"""

import pandas as pd
from igraph import Graph
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Dict, Tuple
from datetime import datetime


# ============================================================================
# GAP DETECTION UTILITIES (same as other functions)
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
        print("  ✓ Community plots show SEPARATE LINE SEGMENTS for each continuous period")
        print("  ✓ No lines are drawn across gaps")
        print("  ✓ Visual breaks indicate where data is missing")
    else:
        print("  ✓ Community plots show CONTINUOUS LINES connecting all points")

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
                          linewidth=2, color='#2ca02c'):
    """Plot data with gap handling."""

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

def communities_measures(graphs: List,
                        graph_labels: Optional[List[str]] = None,
                        save_path: str = "plots/",
                        visualisation: bool = True,
                        report_gaps: bool = True) -> dict:
    """
    Detect and analyze community structures using multiple algorithms.

    Applies various community detection algorithms to each graph in the temporal
    sequence and tracks how community structure evolves over time. Provides both
    detailed community membership and summary statistics.

    **KEY FEATURE:** Automatically detects and reports temporal gaps in your data.
    Plots correctly show gaps as visual breaks.

    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        Supports multiple formats: YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    save_path : str, optional
        Directory path for saving results and visualizations (default: "plots/")
    visualisation : bool, optional
        If True (default), generates plots for community evolution
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    dict
        Dictionary with results for each algorithm, mapping algorithm names to
        DataFrames containing community assignments

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import communities_measures
    >>> G1 = ig.Graph.Barabasi(n=100, m=2)
    >>> G2 = ig.Graph.Barabasi(n=100, m=2)
    >>> graphs = [G1, G2]
    >>> labels = ["2019-01", "2019-02"]
    >>> communities = communities_measures(graphs, graph_labels=labels)

    Notes
    -----
    Community detection algorithms used:
    - Leiden: Optimizes modularity with refined granularity
    - Louvain: Fast modularity optimization
    - Walktrap: Uses random walks to find communities
    - Fast Greedy: Greedy optimization of modularity
    - Label Propagation: Fast method based on label propagation
    - Spinglass: Based on statistical mechanics
    - Infomap: Information-theoretic approach

    Temporal Gaps:
    - Automatically analyzes temporal labels for gaps
    - Reports if data has missing periods
    - Community plots preserve gaps as visual breaks
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

    # Create output directory
    os.makedirs(save_path, exist_ok=True)

    # Define community detection algorithms
    # Note: Some require undirected graphs, so we convert as needed
    community_algorithms = [
        ("leiden", "community_leiden"),
        ("louvain", "community_multilevel"),
        ("walktrap", "community_walktrap"),
        ("fast_greedy", "community_fastgreedy"),
        ("label_prop", "community_label_propagation"),
        ("spinglass", "community_spinglass"),
        ("infomap", "community_infomap")
    ]

    results = {}

    # Apply each algorithm
    for algo_name, algo_func in community_algorithms:
        print(f"\nProcessing algorithm: {algo_name}")
        all_communities = []
        num_communities_list = []
        max_community_size_list = []

        # Apply algorithm to each graph
        for graph_idx, graph in enumerate(graphs):
            graph_label = graph_labels[graph_idx]

            try:
                # Convert to undirected for algorithms that require it
                g = graph.copy()
                if g.is_directed() and algo_func in ["community_walktrap",
                                                      "community_fastgreedy",
                                                      "community_label_propagation",
                                                      "community_spinglass"]:
                    g = g.as_undirected()

                # Simplify graph to remove multi-edges and loops
                g.simplify(multiple=True, loops=True, combine_edges=dict(weight="sum"))

                # Get weights if available
                weights = g.es["weight"] if "weight" in g.es.attributes() else None

                # Get node labels
                if "name" in g.vs.attributes():
                    node_labels = g.vs["name"]
                elif "label" in g.vs.attributes():
                    node_labels = g.vs["label"]
                else:
                    node_labels = [f"Node_{i}" for i in range(g.vcount())]

                # Detect communities using the specified algorithm
                try:
                    if algo_func in ["community_walktrap", "community_fastgreedy"]:
                        partition = getattr(g, algo_func)(weights=weights).as_clustering()
                    elif algo_func == "community_infomap":
                        partition = getattr(g, algo_func)(edge_weights=weights)
                    else:
                        partition = getattr(g, algo_func)(weights=weights)
                except Exception as e:
                    print(f"  Warning: Algorithm {algo_name} failed on graph {graph_label}: {e}")
                    continue

                # Store community assignments
                for comm_idx, comm in enumerate(partition):
                    for node_idx in comm:
                        all_communities.append({
                            "Graph": graph_label,
                            "Node": node_labels[node_idx],
                            "Community": comm_idx
                        })

                # Calculate statistics for this graph
                num_communities_list.append({
                    "Graph": graph_label,
                    "Number_of_Communities": len(partition),
                    "Max_Community_Size": max(len(c) for c in partition) if partition else 0,
                    "Min_Community_Size": min(len(c) for c in partition) if partition else 0,
                    "Mean_Community_Size": sum(len(c) for c in partition) / len(partition) if partition else 0,
                })

            except Exception as e:
                print(f"  Error processing graph {graph_label} with algorithm {algo_name}: {e}")
                continue

        # Convert to DataFrame and save
        if all_communities:
            communities_df = pd.DataFrame(all_communities)
            csv_filename = os.path.join(save_path, f"communities_{algo_name}_assignments.csv")
            communities_df.to_csv(csv_filename, index=False)
            print(f"  ✓ Community assignments saved: {csv_filename}")
            results[algo_name] = communities_df
        else:
            print(f"  Warning: No communities detected for {algo_name}")
            continue

        # Save statistics
        if num_communities_list:
            stats_df = pd.DataFrame(num_communities_list)
            stats_csv_filename = os.path.join(save_path, f"communities_{algo_name}_stats.csv")
            stats_df.to_csv(stats_csv_filename, index=False)
            print(f"  ✓ Community statistics saved: {stats_csv_filename}")

            # Generate visualizations with gap handling
            if visualisation:
                _plot_community_stats(stats_df, algo_name, graph_labels, gap_info, save_path)

    return results


def _plot_community_stats(stats_df: pd.DataFrame, algo_name: str,
                         graph_labels: List[str], gap_info: Dict,
                         save_path: str) -> None:
    """
    Helper function to plot community statistics with gap handling.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with community statistics
    algo_name : str
        Name of the algorithm
    graph_labels : list of str
        Temporal labels
    gap_info : dict
        Gap detection information
    save_path : str
        Path for saving plots
    """

    properties_to_plot = [
        ("Number_of_Communities", "Number of Communities"),
        ("Max_Community_Size", "Maximum Community Size"),
        ("Mean_Community_Size", "Mean Community Size"),
    ]

    for prop_col, prop_label in properties_to_plot:
        if prop_col not in stats_df.columns:
            continue

        try:
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

            y_values = stats_df[prop_col].values

            # FIXED: Use gap-aware plotting
            plot_with_gap_handling(ax, graph_labels, y_values,
                                  gap_info["segments"],
                                  marker='o', linestyle='-', markersize=10,
                                  linewidth=2, color='#2ca02c')

            ax.set_xlabel("Year - Month", fontsize=14, fontweight='bold')
            ax.set_ylabel(prop_label, fontsize=14, fontweight='bold')
            ax.set_title(f"{prop_label} ({algo_name})", fontsize=16, fontweight='bold')

            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
            plt.yticks(fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_filename = os.path.join(save_path, f"communities_{algo_name}_{prop_col}.pdf")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Plot saved: {plot_filename}")

        except Exception as e:
            print(f"  Warning: Could not plot {prop_label}: {e}")