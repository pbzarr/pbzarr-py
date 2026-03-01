"""Tests for pbzarr.region — region parsing."""

import pytest

from pbzarr import InvalidRegionError, Region, parse_region


# ---------------------------------------------------------------------------
# String form: whole contig
# ---------------------------------------------------------------------------


class TestStringWholeContig:
    def test_simple(self):
        r = parse_region("chr1")
        assert r == Region(contig="chr1", start=None, end=None)

    def test_with_whitespace(self):
        r = parse_region("  chr1  ")
        assert r == Region(contig="chr1", start=None, end=None)

    def test_contig_with_underscore(self):
        r = parse_region("chr1_random")
        assert r == Region(contig="chr1_random")

    def test_contig_with_period(self):
        r = parse_region("scaffold.1")
        assert r == Region(contig="scaffold.1")

    def test_contig_with_hyphen(self):
        r = parse_region("NC-000001")
        assert r == Region(contig="NC-000001")


# ---------------------------------------------------------------------------
# String form: range
# ---------------------------------------------------------------------------


class TestStringRange:
    def test_basic(self):
        r = parse_region("chr1:1000-2000")
        assert r == Region(contig="chr1", start=1000, end=2000)

    def test_with_commas(self):
        r = parse_region("chr1:1,000,000-2,000,000")
        assert r == Region(contig="chr1", start=1_000_000, end=2_000_000)

    def test_zero_start(self):
        r = parse_region("chr1:0-100")
        assert r == Region(contig="chr1", start=0, end=100)


# ---------------------------------------------------------------------------
# String form: single position
# ---------------------------------------------------------------------------


class TestStringSinglePosition:
    def test_basic(self):
        r = parse_region("chr1:5000")
        assert r == Region(contig="chr1", start=5000, end=5001)

    def test_zero(self):
        r = parse_region("chr1:0")
        assert r == Region(contig="chr1", start=0, end=1)


# ---------------------------------------------------------------------------
# Tuple form
# ---------------------------------------------------------------------------


class TestTupleForm:
    def test_whole_contig(self):
        r = parse_region(("chr1",))
        assert r == Region(contig="chr1")

    def test_range(self):
        r = parse_region(("chr1", 1000, 2000))
        assert r == Region(contig="chr1", start=1000, end=2000)

    def test_single_position(self):
        r = parse_region(("chr1", 5000))
        assert r == Region(contig="chr1", start=5000, end=5001)


# ---------------------------------------------------------------------------
# One-based conversion
# ---------------------------------------------------------------------------


class TestOneBased:
    def test_string_range(self):
        # 1-based inclusive chr1:1000-2000 → 0-based half-open [999, 2000)
        r = parse_region("chr1:1000-2000", one_based=True)
        assert r == Region(contig="chr1", start=999, end=2000)

    def test_string_single_position(self):
        # 1-based position 5000 → 0-based [4999, 5000)
        r = parse_region("chr1:5000", one_based=True)
        assert r == Region(contig="chr1", start=4999, end=5000)

    def test_string_whole_contig(self):
        # Whole contig is unaffected by one_based
        r = parse_region("chr1", one_based=True)
        assert r == Region(contig="chr1")

    def test_tuple_range(self):
        r = parse_region(("chr1", 1000, 2000), one_based=True)
        assert r == Region(contig="chr1", start=999, end=2000)

    def test_tuple_single_position(self):
        r = parse_region(("chr1", 5000), one_based=True)
        assert r == Region(contig="chr1", start=4999, end=5000)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_empty_string(self):
        with pytest.raises(InvalidRegionError, match="empty"):
            parse_region("")

    def test_malformed_string(self):
        with pytest.raises(InvalidRegionError, match="Malformed"):
            parse_region("chr1:abc-def")

    def test_inverted_range(self):
        with pytest.raises(InvalidRegionError, match="less than"):
            parse_region("chr1:2000-1000")

    def test_equal_start_end(self):
        with pytest.raises(InvalidRegionError, match="less than"):
            parse_region(("chr1", 1000, 1000))

    def test_negative_start_tuple(self):
        with pytest.raises(InvalidRegionError, match="non-negative"):
            parse_region(("chr1", -1, 100))

    def test_negative_end_tuple(self):
        with pytest.raises(InvalidRegionError, match="non-negative"):
            parse_region(("chr1", 0, -1))

    def test_invalid_contig_in_tuple(self):
        with pytest.raises(InvalidRegionError, match="Invalid contig"):
            parse_region(("chr1:bad",))

    def test_non_string_contig_tuple(self):
        with pytest.raises(InvalidRegionError, match="string"):
            parse_region((123,))

    def test_wrong_tuple_length(self):
        with pytest.raises(InvalidRegionError, match="1, 2, or 3"):
            parse_region(("chr1", 1, 2, 3))

    def test_non_int_position_tuple(self):
        with pytest.raises(InvalidRegionError, match="integer"):
            parse_region(("chr1", "abc"))

    def test_wrong_type(self):
        with pytest.raises(InvalidRegionError, match="string or tuple"):
            parse_region(42)  # type: ignore[arg-type]

    def test_one_based_position_zero_is_invalid(self):
        # 1-based position 0 doesn't exist — would become -1 in 0-based
        with pytest.raises(InvalidRegionError, match="non-negative"):
            parse_region("chr1:0", one_based=True)
