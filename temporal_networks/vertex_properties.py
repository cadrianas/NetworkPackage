"""
Temporal Network Analysis: Vertex Properties Module (WITH GAP REPORTING)

This module provides functions for analyzing properties of individual nodes
across temporal networks, tracking how node importance and characteristics
change over time.

KEY FEATURES:
- Automatically detects and reports temporal gaps
- Plots correctly show gaps as visual breaks
- Tracks individual node evolution
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
        print("  ✓ Vertex property plots show SEPARATE LINE SEGMENTS for each continuous period")
        print("  ✓ No lines are drawn across gaps")
        print("  ✓ Visual breaks indicate where data is missing")
    else:
        print("  ✓ Vertex property plots show CONTINUOUS LINES connecting all points")

    print("\n" + "=" * 80 + "\n")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def plot_with_gap_handling(ax, graph_labels: List[str], y_values, gap_segments: List[Tuple],
                          marker='o', linestyle='-', markersize=10,
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

def vertex_properties(graphs: List,
                     node_name: str,
                     graph_labels: Optional[List[str]] = None,
                     filename: Optional[str] = None,
                     save_path: str = "plots/",
                     visualisation: bool = True,
                     report_gaps: bool = True) -> pd.DataFrame:
    """
    Compute and visualize properties of a specific node across multiple graphs.

    Tracks how various metrics for a single node evolve over time across a
    temporal network. Computes centrality measures, structural properties,
    and constraint metrics for the specified node.

    **KEY FEATURE:** Automatically detects and reports temporal gaps in your data.
    Plots correctly show gaps as visual breaks.

    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects to analyze
    node_name : str
        Name/label of the node of interest (must match node attribute in graphs)
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        Supports multiple formats: YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        CSV filename for saving results. If None, results are not saved.
    save_path : str, optional
        Directory for saving visualizations (default: "plots/")
    visualisation : bool, optional
        If True (default), generates plots for each property over time
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

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

    Temporal Gaps:
    - Automatically analyzes temporal labels for gaps
    - Reports if data has missing periods
    - Vertex property plots preserve gaps as visual breaks
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

    # Generate visualizations with gap handling
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

                y_values = properties_df[prop].values

                # FIXED: Use gap-aware plotting
                plot_with_gap_handling(ax, graph_labels, y_values,
                                      gap_info["segments"],
                                      marker='o', linestyle='-', markersize=10,
                                      linewidth=2, color='#1f77b4')

                ax.set_xlabel("Year - Month", fontsize=12, fontweight='bold')
                ax.set_ylabel(prop.replace('_', ' '), fontsize=12, fontweight='bold')
                ax.set_title(f"{prop.replace('_', ' ')} for {node_name}", fontsize=14, fontweight='bold')

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

            # Plot all numeric properties
            for prop in properties_to_plot:
                if prop in properties_df.columns and not properties_df[prop].isna().all():
                    y_values = properties_df[prop].values
                    if y_values.max() > 0:
                        y_values = y_values / y_values.max()

                    # FIXED: Use gap-aware plotting for combined plot
                    for segment_start, segment_end in gap_info["segments"]:
                        x_indices = np.arange(segment_start, segment_end)
                        y_segment = [y_values[i] for i in x_indices]
                        ax.plot(x_indices, y_segment, marker='o', linestyle='-',
                               label=prop.replace('_', ' ') if segment_start == gap_info["segments"][0][0] else "",
                               linewidth=2)

            ax.set_xlabel("Year - Month", fontsize=12, fontweight='bold')
            ax.set_ylabel("Normalized Value", fontsize=12, fontweight='bold')
            ax.set_title(f"All Properties for {node_name}", fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(graph_labels)))
            ax.set_xticklabels(graph_labels, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=9, loc='best')

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