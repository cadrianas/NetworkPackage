import unittest
from datetime import datetime

from temporal_networks._gap_utilities import parse_flexible_datetime, format_large_numbers

class TestFormatLargeNumbers(unittest.TestCase):
    def test_format_small_numbers(self):
        """Test formatting of numbers less than 1,000."""
        self.assertEqual(format_large_numbers(0, 0), "0")
        self.assertEqual(format_large_numbers(500, 0), "500")
        self.assertEqual(format_large_numbers(999, 0), "999")

    def test_format_thousands(self):
        """Test formatting of numbers in the thousands (k)."""
        self.assertEqual(format_large_numbers(1_000, 0), "1.0k")
        self.assertEqual(format_large_numbers(1_500, 0), "1.5k")
        self.assertEqual(format_large_numbers(999_999, 0), "1000.0k") # Based on logic, < 1_000_000 is k

    def test_format_millions(self):
        """Test formatting of numbers in the millions (M)."""
        self.assertEqual(format_large_numbers(1_000_000, 0), "1.0M")
        self.assertEqual(format_large_numbers(2_500_000, 0), "2.5M")
        self.assertEqual(format_large_numbers(999_999_999, 0), "1000.0M") # Based on logic

    def test_format_billions(self):
        """Test formatting of numbers in the billions (B)."""
        self.assertEqual(format_large_numbers(1_000_000_000, 0), "1.0B")
        self.assertEqual(format_large_numbers(3_500_000_000, 0), "3.5B")
        self.assertEqual(format_large_numbers(10_000_000_000, 0), "10.0B")

    def test_format_negative_numbers(self):
        """Test formatting of negative numbers (current logic treats as < 1000)."""
        self.assertEqual(format_large_numbers(-500, 0), "-500")
        self.assertEqual(format_large_numbers(-1_000_000, 0), "-1000000")

    def test_format_floats(self):
        """Test formatting of floating point inputs."""
        self.assertEqual(format_large_numbers(1500.5, 0), "1.5k")
        self.assertEqual(format_large_numbers(999.9, 0), "1000") # 999.9 formatted with .0f is 1000


class TestParseFlexibleDatetime(unittest.TestCase):
    def test_parse_flexible_datetime_valid_formats(self):
        # YYYY-MM
        self.assertEqual(parse_flexible_datetime("2024-03"), datetime(2024, 3, 1))

        # YYYY-MM-DD
        self.assertEqual(parse_flexible_datetime("2024-03-15"), datetime(2024, 3, 15))

        # YYYY-W## (First day of ISO week 12 in 2024 is Monday, March 18)
        self.assertEqual(parse_flexible_datetime("2024-W12"), datetime(2024, 3, 18))

        # YYYY-Q# (First day of quarter 2 is April 1)
        self.assertEqual(parse_flexible_datetime("2024-Q2"), datetime(2024, 4, 1))

        # YYYY
        self.assertEqual(parse_flexible_datetime("2024"), datetime(2024, 1, 1))

        # Leading and trailing spaces
        self.assertEqual(parse_flexible_datetime("  2024-03  "), datetime(2024, 3, 1))
        self.assertEqual(parse_flexible_datetime("2024-03-15 "), datetime(2024, 3, 15))
        self.assertEqual(parse_flexible_datetime(" 2024-Q2"), datetime(2024, 4, 1))
        self.assertEqual(parse_flexible_datetime("2024 "), datetime(2024, 1, 1))

    def test_parse_flexible_datetime_invalid_fallback(self):
        # Invalid format
        self.assertIsNone(parse_flexible_datetime("invalid"))

        # Valid format but invalid date values
        self.assertIsNone(parse_flexible_datetime("2024-13-45"))

        # Another invalid string
        self.assertIsNone(parse_flexible_datetime("not-a-date"))

        # Empty string
        self.assertIsNone(parse_flexible_datetime(""))

if __name__ == '__main__':
    unittest.main()
