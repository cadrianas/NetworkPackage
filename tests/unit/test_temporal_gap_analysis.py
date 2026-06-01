import unittest

import temporal_networks.temporal_gap_analysis as temporal_gap_analysis
import temporal_networks._gap_utilities as _gap_utilities


class TestTemporalGapAnalysis(unittest.TestCase):
    def test_all_exports(self):
        """Test that __all__ contains exactly the expected functions."""
        expected_exports = [
            "parse_flexible_datetime",
            "calculate_time_difference",
            "detect_temporal_gaps",
            "print_gap_report",
            "create_gap_dataframe",
        ]
        self.assertCountEqual(temporal_gap_analysis.__all__, expected_exports)
        self.assertEqual(temporal_gap_analysis.__all__, expected_exports)

    def test_exported_functions(self):
        """Test that the functions exported match the original implementations in _gap_utilities."""
        self.assertIs(
            temporal_gap_analysis.parse_flexible_datetime,
            _gap_utilities.parse_flexible_datetime
        )
        self.assertIs(
            temporal_gap_analysis.calculate_time_difference,
            _gap_utilities.calculate_time_difference
        )
        self.assertIs(
            temporal_gap_analysis.detect_temporal_gaps,
            _gap_utilities.detect_temporal_gaps
        )
        self.assertIs(
            temporal_gap_analysis.print_gap_report,
            _gap_utilities.print_gap_report
        )
        self.assertIs(
            temporal_gap_analysis.create_gap_dataframe,
            _gap_utilities.create_gap_dataframe
        )


if __name__ == '__main__':
    unittest.main()
