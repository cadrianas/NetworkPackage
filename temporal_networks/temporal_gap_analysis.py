"""
Temporal Gap Analysis and Reporting

This module provides comprehensive gap detection and reporting for temporal networks.
Users can identify where data has discontinuities (seasonal closures, maintenance windows,
missing measurements, etc.) to ensure accurate temporal analysis.

Supports flexible datetime formats and provides clear reporting of gap locations.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import pandas as pd


def parse_flexible_datetime(label: str) -> Optional[datetime]:
    """
    Parse a datetime label in multiple formats.

    Supports:
    - "YYYY-MM" (e.g., "2024-03")
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

    Examples
    --------
    >>> parse_flexible_datetime("2024-03")
    datetime.datetime(2024, 3, 1, 0, 0)

    >>> parse_flexible_datetime("2024-03-15")
    datetime.datetime(2024, 3, 15, 0, 0)
    """
    # Try YYYY-MM first (most common for temporal networks)
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
        # Convert week number to date (using first day of week)
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


def calculate_time_difference(date1: datetime, date2: datetime, unit: str = "months") -> float:
    """
    Calculate time difference between two dates in specified units.

    Parameters
    ----------
    date1 : datetime
        First date
    date2 : datetime
        Second date (should be after date1)
    unit : str, optional
        Unit for difference: "days", "weeks", "months", "years" (default: "months")

    Returns
    -------
    float
        Time difference in specified units

    Examples
    --------
    >>> d1 = datetime(2024, 3, 1)
    >>> d2 = datetime(2024, 5, 1)
    >>> calculate_time_difference(d1, d2, unit="months")
    2.0
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


def detect_temporal_gaps(graph_labels: List[str],
                         gap_threshold: int = 1,
                         unit: str = "months",
                         verbose: bool = True) -> Dict:
    """
    Detect temporal gaps in a list of labels with detailed reporting.

    A gap occurs when consecutive labels are more than `gap_threshold` apart
    (in specified units). This is useful for identifying seasonal closures,
    maintenance windows, data collection gaps, etc.

    Parameters
    ----------
    graph_labels : list of str
        Labels in datetime format (e.g., ["2024-03", "2024-04", "2024-11"])
    gap_threshold : int, optional
        Threshold for detecting gaps (default: 1)
        - For monthly data: threshold=1 means 2+ months apart is a gap
        - For daily data: threshold=7 means 8+ days apart is a gap
    unit : str, optional
        Time unit for calculation: "days", "weeks", "months", "years"
        (default: "months")
    verbose : bool, optional
        If True, print detailed gap report (default: True)

    Returns
    -------
    dict
        Dictionary with keys:
        - "has_gaps" (bool): Whether gaps were detected
        - "num_gaps" (int): Number of gaps found
        - "gaps" (list): List of gap information dicts with:
          - "start_idx" (int): Index of last point before gap
          - "end_idx" (int): Index of first point after gap
          - "start_label" (str): Label before gap
          - "end_label" (str): Label after gap
          - "gap_size" (float): Size of gap in specified units
        - "segments" (list): List of (start_idx, end_idx) tuples for continuous segments
        - "report" (str): Human-readable gap report

    Examples
    --------
    >>> labels = ["2024-03", "2024-04", "2024-05", "2024-11", "2024-12"]
    >>> result = detect_temporal_gaps(labels, verbose=False)
    >>> result["has_gaps"]
    True
    >>> result["num_gaps"]
    1
    >>> result["gaps"][0]["gap_size"]
    6.0
    """

    if len(graph_labels) < 2:
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
            "report": "No gaps: Data is continuous or contains fewer than 2 points."
        }

    # Parse all labels
    parsed_dates = [parse_flexible_datetime(label) for label in graph_labels]

    # Check if parsing was successful
    if any(d is None for d in parsed_dates):
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
            "report": "Warning: Could not parse all labels. Gap detection disabled."
        }

    # Detect gaps
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

            # End current segment and start new one
            segments.append((segment_start, i))
            segment_start = i

    # Add final segment
    segments.append((segment_start, len(graph_labels)))

    # Generate report
    report = _generate_gap_report(graph_labels, gaps, has_gaps=len(gaps) > 0, unit=unit)

    result = {
        "has_gaps": len(gaps) > 0,
        "num_gaps": len(gaps),
        "gaps": gaps,
        "segments": segments,
        "report": report
    }

    if verbose:
        print(report)

    return result


def _generate_gap_report(graph_labels: List[str], gaps: List[Dict],
                         has_gaps: bool, unit: str = "months") -> str:
    """Generate human-readable gap report."""

    lines = [
        "=" * 80,
        "TEMPORAL GAP ANALYSIS",
        "=" * 80,
    ]

    lines.append(f"\nDataset: {len(graph_labels)} temporal observations")
    lines.append(f"Time unit: {unit}")
    lines.append(f"Date range: {graph_labels[0]} to {graph_labels[-1]}")

    if not has_gaps:
        lines.append("\n✓ No gaps detected: Data is continuous")
    else:
        lines.append(f"\n⚠ Gaps detected: {len(gaps)} gap(s) found\n")

        for i, gap in enumerate(gaps, 1):
            lines.append(f"Gap #{i}:")
            lines.append(f"  Between: {gap['start_label']} (index {gap['start_idx']})")
            lines.append(f"       and: {gap['end_label']} (index {gap['end_idx']})")
            lines.append(f"  Size: {gap['gap_size']:.1f} {unit}")
            lines.append("")

    lines.append("\nImpact on plotting:")
    if has_gaps:
        lines.append("  - Plots will show SEPARATE LINE SEGMENTS for each continuous period")
        lines.append("  - No lines will be drawn across the gaps")
        lines.append("  - Visual breaks indicate where data is missing")
    else:
        lines.append("  - Plots will show a CONTINUOUS LINE connecting all points")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def print_gap_summary(gap_info: Dict) -> None:
    """
    Print a concise summary of detected gaps.

    Parameters
    ----------
    gap_info : dict
        Output from detect_temporal_gaps()
    """
    print(gap_info["report"])


def create_gap_dataframe(graph_labels: List[str], gap_info: Dict) -> pd.DataFrame:
    """
    Create a DataFrame summarizing detected gaps.

    Useful for including in analysis reports or exporting.

    Parameters
    ----------
    graph_labels : list of str
        Original temporal labels
    gap_info : dict
        Output from detect_temporal_gaps()

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per gap, including:
        - start_label, end_label, gap_size

    Examples
    --------
    >>> labels = ["2024-03", "2024-04", "2024-05", "2024-11", "2024-12"]
    >>> gaps = detect_temporal_gaps(labels, verbose=False)
    >>> df = create_gap_dataframe(labels, gaps)
    >>> df.to_csv("gaps.csv")
    """

    if not gap_info["gaps"]:
        return pd.DataFrame(columns=["gap_number", "start_label", "end_label", "gap_size"])

    gap_rows = []
    for i, gap in enumerate(gap_info["gaps"], 1):
        gap_rows.append({
            "gap_number": i,
            "start_label": gap["start_label"],
            "end_label": gap["end_label"],
            "gap_size": gap["gap_size"],
            "start_idx": gap["start_idx"],
            "end_idx": gap["end_idx"],
        })

    return pd.DataFrame(gap_rows)


# Example usage:
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Example 1: Continuous Data (No Gaps)")
    print("=" * 80 + "\n")

    labels_continuous = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    result1 = detect_temporal_gaps(labels_continuous)

    print("\n" + "=" * 80)
    print("Example 2: Gapped Data (Seasonal Closure)")
    print("=" * 80 + "\n")

    labels_gapped = ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07",
                     "2024-08", "2024-11", "2024-12", "2025-01", "2025-02"]
    result2 = detect_temporal_gaps(labels_gapped, verbose=True)

    # Create gap summary dataframe
    gap_df = create_gap_dataframe(labels_gapped, result2)
    print("\nGap Summary as DataFrame:")
    print(gap_df)

    print("\n" + "=" * 80)
    print("Example 3: Daily Data with Gap")
    print("=" * 80 + "\n")

    labels_daily = ["2024-03-01", "2024-03-02", "2024-03-03",
                    "2024-03-10", "2024-03-11"]  # 7-day gap
    result3 = detect_temporal_gaps(labels_daily, gap_threshold=3, unit="days")