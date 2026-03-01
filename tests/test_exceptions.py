"""Tests for pbzarr.exceptions."""

import pytest

from pbzarr import (
    ColumnNotFoundError,
    ContigNotFoundError,
    InvalidRegionError,
    PbzError,
    TrackNotFoundError,
)


class TestExceptionHierarchy:
    """All PBZ exceptions inherit from PbzError."""

    def test_contig_not_found_is_pbz_error(self):
        assert issubclass(ContigNotFoundError, PbzError)

    def test_track_not_found_is_pbz_error(self):
        assert issubclass(TrackNotFoundError, PbzError)

    def test_invalid_region_is_pbz_error(self):
        assert issubclass(InvalidRegionError, PbzError)

    def test_column_not_found_is_pbz_error(self):
        assert issubclass(ColumnNotFoundError, PbzError)


class TestContigNotFoundError:
    def test_message(self):
        e = ContigNotFoundError("chrX")
        assert "chrX" in str(e)

    def test_message_with_available(self):
        e = ContigNotFoundError("chrX", available=["chr1", "chr2"])
        assert "chrX" in str(e)
        assert "chr1" in str(e)

    def test_attrs(self):
        e = ContigNotFoundError("chrX", available=["chr1"])
        assert e.contig == "chrX"
        assert e.available == ["chr1"]


class TestTrackNotFoundError:
    def test_message(self):
        e = TrackNotFoundError("depths")
        assert "depths" in str(e)

    def test_message_with_available(self):
        e = TrackNotFoundError("depths", available=["masks"])
        assert "depths" in str(e)
        assert "masks" in str(e)


class TestColumnNotFoundError:
    def test_message(self):
        e = ColumnNotFoundError("sample_X")
        assert "sample_X" in str(e)

    def test_message_with_available(self):
        e = ColumnNotFoundError("sample_X", available=["sample_A"])
        assert "sample_X" in str(e)
        assert "sample_A" in str(e)


class TestCatchAllPbzError:
    """Users can catch PbzError to handle any PBZ-specific error."""

    def test_catch_contig(self):
        with pytest.raises(PbzError):
            raise ContigNotFoundError("chrX")

    def test_catch_track(self):
        with pytest.raises(PbzError):
            raise TrackNotFoundError("depths")

    def test_catch_region(self):
        with pytest.raises(PbzError):
            raise InvalidRegionError("bad region")

    def test_catch_column(self):
        with pytest.raises(PbzError):
            raise ColumnNotFoundError("sample_X")
