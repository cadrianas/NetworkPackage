"""
Temporal Gap Utilities Module

This is a shared utility module for gap detection and plotting across the
temporal_networks package. Can be copied directly into the package as:
  temporal_networks/_gap_utilities.py

All plotting functions should import from this module for consistent gap handling.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Optional, Dict


# ============================================================================
# DATETIME PARSING
# ============================================================================

def parse_flexible_datetime(label: str) -> Optional[datetime]:
    """
    Parse a datetime label in multiple formats.

    Supports:
    - "YYYY-MM" (e.g., "2024-03") - RECOMMENDED for temporal networks
    - "YYYY-MM-DD" (e.g., "2024-03-15")
    - "YYYY-W##" (e.g., "2024-W12" for week 12)
    - "YYYY-Q#" (e.g., "2024-Q2" for quarter 2)
    - "YYYY" (e.g., "2024" for year only)

    Parameters
    ----------
    label : str
        Datetime label in any supported format

    Returns
    -------
    datetime or None
        Parsed datetime object, or None if parsing fails
    """
    # Try YYYY-MM first (most common)
    try:
        return datetime.strptime(label.strip(), "%Y-%m")
    except ValueError:
        pass

    # Try YYYY-MM-DD
    try:
        return datetime.strptime(label.strip(), "%Y-%m-%d")
    except ValueError:
        pass

    # Try YYYY-W## (ISO week format)
    try:
        year, week = label.strip().split('-W')
        return datetime.strptime(f"{year}-W{int(week)}-1", "%Y-W%W-%w")
    except (ValueError, AttributeError):
        pass

    # Try YYYY-Q# (quarter format)
    try:
        year, quarter = label.strip().split('-Q')
        month = (int(quarter) - 1) * 3 + 1
        return datetime(int(year), month, 1)
    except (ValueError, IndexError):
        pass

    # Try YYYY only
    try:
        return datetime.strptime(label.strip(), "%Y")
    except ValueError:
        pass

    return None


def calculate_time_difference(date1: datetime, date2: datetime,
                              unit: str = "months") -> float:
    """
    Calculate time difference between two dates in specified units.

    Parameters
    ----------
    date1 : datetime
        First date
    date2 : datetime
        Second date
    unit : str, optional
        Time unit: "days", "weeks", "months", "years" (default: "months")

    Returns
    -------
    float
        Time difference in specified units
    """
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
        raise ValueError(f"Unknown unit: {unit}. Use 'days', 'weeks', 'months', or 'years'")


# ============================================================================
# GAP DETECTION
# ============================================================================

def detect_temporal_gaps(graph_labels: List[str],
                         gap_threshold: int = 1,
                         unit: str = "months") -> Dict:
    """
    Detect temporal gaps in a list of labels.

    A gap occurs when consecutive labels are more than `gap_threshold` apart.
    For example, with monthly data and threshold=1, a gap of 2+ months is detected.

    Parameters
    ----------
    graph_labels : list of str
        Temporal labels (e.g., ["2024-03", "2024-04", "2024-11"])
    gap_threshold : int, optional
        Threshold for gap detection (default: 1)
    unit : str, optional
        Time unit: "days", "weeks", "months", "years" (default: "months")

    Returns
    -------
    dict
        Dictionary with:
        - "has_gaps" (bool): Whether gaps were detected
        - "num_gaps" (int): Number of gaps
        - "gaps" (list): Details about each gap
        - "segments" (list): Continuous time segments as (start, end) tuples

    Examples
    --------
    >>> labels = ["2024-03", "2024-04", "2024-05", "2024-11", "2024-12"]
    >>> result = detect_temporal_gaps(labels)
    >>> result["has_gaps"]
    True
    >>> result["num_gaps"]
    1
    """

    if len(graph_labels) < 2:
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
        }

    # Parse all labels
    parsed_dates = [parse_flexible_datetime(label) for label in graph_labels]

    # Check if all parsing was successful
    if any(d is None for d in parsed_dates):
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
        }

    # Find gaps
    gaps = []
    segments = []
    segment_start = 0

    for i in range(1, len(parsed_dates)):
        date_prev = parsed_dates[i - 1]
        date_curr = parsed_dates[i]

        time_diff = calculate_time_difference(date_prev, date_curr, unit=unit)

        # Gap detected if difference exceeds threshold
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

    # Add final segment
    segments.append((segment_start, len(graph_labels)))

    return {
        "has_gaps": len(gaps) > 0,
        "num_gaps": len(gaps),
        "gaps": gaps,
        "segments": segments,
    }


# ============================================================================
# REPORTING
# ============================================================================

def print_gap_report(graph_labels: List[str], gap_info: Dict,
                     unit: str = "months") -> None:
    """
    Print human-readable gap analysis report to console.

    Parameters
    ----------
    graph_labels : list of str
        Original temporal labels
    gap_info : dict
        Output from detect_temporal_gaps()
    unit : str, optional
        Time unit for reporting (default: "months")
    """

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

    print("Impact on Temporal Visualization:")
    if gap_info["has_gaps"]:
        print("  ✓ Plots show SEPARATE LINE SEGMENTS for each continuous period")
        print("  ✓ No lines are drawn across gaps")
        print("  ✓ Visual breaks indicate where data is missing")
        print("\nPossible causes for gaps:")
        print("  - Seasonal operation (system open only part of year)")
        print("  - System maintenance or downtime")
        print("  - Data collection interruptions")
        print("  - Multi-phase study (Phase 1, break, Phase 2)")
    else:
        print("  ✓ Plots show CONTINUOUS LINES connecting all points")
        print("  ✓ All time periods are represented sequentially")

    print("\n" + "=" * 80 + "\n")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def format_large_numbers(x, pos):
    """
    Format large numbers with appropriate units (k, M, B).

    Use with matplotlib's FuncFormatter:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
    """
    if x >= 1_000_000_000:
        return f'{x / 1_000_000_000:.1f}B'
    elif x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.1f}k'
    else:
        return f'{x:.0f}'


def plot_with_gap_handling(ax, graph_labels: List[str], y_values, gap_segments: List[Tuple],
                           marker='o', linestyle='-', markersize=10,
                           linewidth=2, color='#1f77b4', label: Optional[str] = None):
    """
    Plot data with proper handling of temporal gaps.

    Instead of drawing a continuous line across gaps, this function detects gaps
    and draws separate line segments for each continuous temporal period. This
    creates visual breaks where gaps occur, accurately representing the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    graph_labels : list of str
        Temporal labels (e.g., ["2024-03", "2024-04", "2024-11"])
    y_values : array-like
        Y-values to plot (one per label)
    gap_segments : list of tuple
        Continuous segments as (start_idx, end_idx) tuples
        (from detect_temporal_gaps()["segments"])
    marker : str, optional
        Marker style (default: 'o')
    linestyle : str, optional
        Line style (default: '-')
    markersize : int, optional
        Marker size (default: 10)
    linewidth : float, optional
        Line width (default: 2)
    color : str, optional
        Color (default: '#1f77b4')
    label : str, optional
        Label for legend (default: None)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from _gap_utilities import plot_with_gap_handling, detect_temporal_gaps
    >>>
    >>> labels = ["2024-03", "2024-04", "2024-05", "2024-11", "2024-12"]
    >>> y_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    >>> gap_info = detect_temporal_gaps(labels)
    >>>
    >>> fig, ax = plt.subplots()
    >>> plot_with_gap_handling(ax, labels, y_values, gap_info["segments"])
    >>> ax.set_title("Example with Gap")
    >>> plt.show()
    """

    for i, (segment_start, segment_end) in enumerate(gap_segments):
        x_indices = np.arange(segment_start, segment_end)
        y_segment = [y_values[idx] for idx in x_indices]

        # Only add label for first segment (for legend)
        segment_label = label if i == 0 else None

        ax.plot(x_indices, y_segment, marker=marker, linestyle=linestyle,
                markersize=markersize, linewidth=linewidth, color=color,
                label=segment_label)

    # Set x-ticks and labels to show all data points
    ax.set_xticks(range(len(graph_labels)))
    ax.set_xticklabels(graph_labels, rotation=45, ha='right', fontsize=10)


# ============================================================================
# USAGE SUMMARY
# ============================================================================

"""
How to use these utilities in your plotting functions:

1. At the top of your module:

   from ._gap_utilities import (
       detect_temporal_gaps,
       print_gap_report,
       plot_with_gap_handling,
       format_large_numbers,
   )

2. In your function (early):

   def my_plotting_function(graphs, graph_labels, ..., report_gaps=True):
       # ... validation ...

       # Detect gaps
       gap_info = detect_temporal_gaps(graph_labels)

       # Report to user
       if report_gaps:
           print_gap_report(graph_labels, gap_info)

       # ... compute metrics ...

       # Create plot
       fig, ax = plt.subplots(figsize=(14, 7))

       # Use gap-aware plotting instead of ax.plot()
       plot_with_gap_handling(ax, graph_labels, y_values,
                             gap_info["segments"],
                             marker='o', linestyle='-', color='#1f77b4')

       ax.set_ylabel("Your Metric")
       ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
       # ... rest of formatting ...

That's it! Gaps will be handled automatically.
"""