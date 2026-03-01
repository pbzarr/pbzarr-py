"""Tests for Track.query() and Track.__getitem__() — Phase 4."""

import numpy as np
import pytest

from pbzarr.exceptions import (
    ColumnNotFoundError,
    ContigNotFoundError,
    InvalidRegionError,
)
from pbzarr.store import create_store
from pbzarr.track import Track

CONTIGS = ["chr1", "chr2"]
CONTIG_LENGTHS = [1_000, 500]
COLUMNS = ["sample_A", "sample_B", "sample_C"]


@pytest.fixture()
def store():
    return create_store({}, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)


@pytest.fixture()
def col_track(store):
    """Columnar track with known data written to chr1."""
    track = store.create_track("depths", dtype="uint32", columns=COLUMNS)
    arr = track.zarr_array("chr1")
    data = np.arange(1_000 * 3, dtype="uint32").reshape(1_000, 3)
    arr[:] = data
    return track


@pytest.fixture()
def scalar_track(store):
    """Scalar (no columns) track with known data."""
    track = store.create_track("mean_depth", dtype="float32")
    arr = track.zarr_array("chr1")
    data = np.arange(1_000, dtype="float32")
    arr[:] = data
    return track


# ── query() with string regions ──────────────────────────────────


class TestQueryString:
    def test_whole_contig_columnar(self, col_track):
        result = col_track.query("chr1")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1_000, 3)

    def test_whole_contig_scalar(self, scalar_track):
        result = scalar_track.query("chr1")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1_000,)

    def test_range_columnar(self, col_track):
        result = col_track.query("chr1:100-200")
        assert result.shape == (100, 3)

    def test_range_scalar(self, scalar_track):
        result = scalar_track.query("chr1:100-200")
        assert result.shape == (100,)
        np.testing.assert_array_equal(result, np.arange(100, 200, dtype="float32"))

    def test_single_base(self, col_track):
        result = col_track.query("chr1:500")
        assert result.shape == (1, 3)

    def test_single_base_scalar(self, scalar_track):
        result = scalar_track.query("chr1:500")
        assert result.shape == (1,)
        assert result[0] == 500.0

    def test_data_values_columnar(self, col_track):
        result = col_track.query("chr1:0-5")
        expected = np.arange(5 * 3, dtype="uint32").reshape(5, 3)
        np.testing.assert_array_equal(result, expected)

    def test_end_of_contig(self, col_track):
        result = col_track.query("chr1:990-1000")
        assert result.shape == (10, 3)


# ── query() with tuple regions ───────────────────────────────────


class TestQueryTuple:
    def test_tuple_contig_only(self, col_track):
        result = col_track.query(("chr1",))
        assert result.shape == (1_000, 3)

    def test_tuple_range(self, col_track):
        result = col_track.query(("chr1", 100, 200))
        assert result.shape == (100, 3)

    def test_tuple_single_base(self, scalar_track):
        result = scalar_track.query(("chr1", 42))
        assert result.shape == (1,)
        assert result[0] == 42.0


# ── query() with one_based ───────────────────────────────────────


class TestQueryOneBased:
    def test_one_based_range(self, scalar_track):
        result = scalar_track.query("chr1:1-10", one_based=True)
        assert result.shape == (10,)
        np.testing.assert_array_equal(
            result, np.arange(0, 10, dtype="float32")
        )

    def test_one_based_single_base(self, scalar_track):
        result = scalar_track.query("chr1:1", one_based=True)
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_one_based_tuple(self, scalar_track):
        result = scalar_track.query(("chr1", 1, 10), one_based=True)
        assert result.shape == (10,)
        np.testing.assert_array_equal(
            result, np.arange(0, 10, dtype="float32")
        )


# ── query() with column filtering ───────────────────────────────


class TestQueryColumns:
    def test_single_column_string_squeeze(self, col_track):
        """Single string → 1D array."""
        result = col_track.query("chr1:0-10", columns="sample_A")
        assert result.shape == (10,)

    def test_single_column_list_no_squeeze(self, col_track):
        """List with one element → 2D array."""
        result = col_track.query("chr1:0-10", columns=["sample_A"])
        assert result.shape == (10, 1)

    def test_two_columns(self, col_track):
        result = col_track.query("chr1:0-10", columns=["sample_A", "sample_C"])
        assert result.shape == (10, 2)

    def test_all_columns_explicit(self, col_track):
        result = col_track.query("chr1:0-10", columns=["sample_A", "sample_B", "sample_C"])
        assert result.shape == (10, 3)

    def test_column_values_correct(self, col_track):
        """Verify that column resolution picks the right data."""
        result = col_track.query("chr1:0-5", columns="sample_C")
        expected_full = np.arange(5 * 3, dtype="uint32").reshape(5, 3)
        np.testing.assert_array_equal(result, expected_full[:, 2])

    def test_column_list_values_correct(self, col_track):
        result = col_track.query("chr1:0-5", columns=["sample_C", "sample_A"])
        expected_full = np.arange(5 * 3, dtype="uint32").reshape(5, 3)
        np.testing.assert_array_equal(result[:, 0], expected_full[:, 2])
        np.testing.assert_array_equal(result[:, 1], expected_full[:, 0])

    def test_columns_on_scalar_track_raises(self, scalar_track):
        with pytest.raises(ColumnNotFoundError):
            scalar_track.query("chr1:0-10", columns="sample_A")

    def test_columns_on_scalar_track_list_raises(self, scalar_track):
        with pytest.raises(ColumnNotFoundError):
            scalar_track.query("chr1:0-10", columns=["sample_A"])


# ── query() validation errors ────────────────────────────────────


class TestQueryValidation:
    def test_contig_not_found(self, col_track):
        with pytest.raises(ContigNotFoundError):
            col_track.query("chrX:0-100")

    def test_end_exceeds_length(self, col_track):
        with pytest.raises(InvalidRegionError, match="exceeds contig length"):
            col_track.query("chr1:0-9999")

    def test_invalid_region_string(self, col_track):
        with pytest.raises(InvalidRegionError):
            col_track.query("!!!bad!!!")

    def test_start_ge_end(self, col_track):
        with pytest.raises(InvalidRegionError):
            col_track.query(("chr1", 200, 100))

    def test_column_not_found(self, col_track):
        with pytest.raises(ColumnNotFoundError):
            col_track.query("chr1:0-10", columns="nonexistent")

    def test_column_list_not_found(self, col_track):
        with pytest.raises(ColumnNotFoundError):
            col_track.query("chr1:0-10", columns=["sample_A", "nonexistent"])


# ── __getitem__ basic access ─────────────────────────────────────


class TestGetitemBasic:
    def test_string_whole_contig(self, col_track):
        result = col_track["chr1"]
        assert result.shape == (1_000, 3)

    def test_string_whole_contig_scalar(self, scalar_track):
        result = scalar_track["chr1"]
        assert result.shape == (1_000,)

    def test_slice_range(self, col_track):
        result = col_track["chr1", 100:200]
        assert result.shape == (100, 3)

    def test_slice_range_scalar(self, scalar_track):
        result = scalar_track["chr1", 100:200]
        assert result.shape == (100,)

    def test_slice_all_columns(self, col_track):
        result = col_track["chr1", 100:200, :]
        assert result.shape == (100, 3)

    def test_int_position(self, col_track):
        """Integer position → single row, all columns."""
        result = col_track["chr1", 500]
        assert result.shape == (1, 3)

    def test_int_position_scalar(self, scalar_track):
        result = scalar_track["chr1", 500]
        assert result.shape == (1,)
        assert result[0] == 500.0

    def test_open_ended_slice(self, col_track):
        """slice(None, 10) → 0:10."""
        result = col_track["chr1", :10]
        assert result.shape == (10, 3)

    def test_open_start_slice(self, scalar_track):
        """slice(990, None) → 990:contig_length."""
        result = scalar_track["chr1", 990:]
        assert result.shape == (10,)

    def test_data_values(self, scalar_track):
        result = scalar_track["chr1", 0:5]
        np.testing.assert_array_equal(
            result, np.arange(5, dtype="float32")
        )


# ── __getitem__ column access ────────────────────────────────────


class TestGetitemColumns:
    def test_column_by_name(self, col_track):
        """String column → squeeze to 1D."""
        result = col_track["chr1", 0:10, "sample_B"]
        assert result.shape == (10,)

    def test_column_by_index(self, col_track):
        """Integer column → squeeze to 1D."""
        result = col_track["chr1", 0:10, 1]
        assert result.shape == (10,)

    def test_column_name_and_index_match(self, col_track):
        by_name = col_track["chr1", 0:10, "sample_B"]
        by_index = col_track["chr1", 0:10, 1]
        np.testing.assert_array_equal(by_name, by_index)

    def test_column_int_slice(self, col_track):
        result = col_track["chr1", 0:10, 0:2]
        assert result.shape == (10, 2)

    def test_column_list_of_names(self, col_track):
        result = col_track["chr1", 0:10, ["sample_A", "sample_C"]]
        assert result.shape == (10, 2)

    def test_column_list_values(self, col_track):
        result = col_track["chr1", 0:5, ["sample_C", "sample_A"]]
        full = np.arange(5 * 3, dtype="uint32").reshape(5, 3)
        np.testing.assert_array_equal(result[:, 0], full[:, 2])
        np.testing.assert_array_equal(result[:, 1], full[:, 0])


# ── __getitem__ validation ───────────────────────────────────────


class TestGetitemValidation:
    def test_contig_not_found(self, col_track):
        with pytest.raises(ContigNotFoundError):
            col_track["chrX"]

    def test_end_exceeds_length(self, col_track):
        with pytest.raises(InvalidRegionError, match="exceeds contig length"):
            col_track["chr1", 0:9999]

    def test_position_exceeds_length(self, col_track):
        with pytest.raises(InvalidRegionError, match="exceeds contig length"):
            col_track["chr1", 1000]

    def test_negative_position(self, col_track):
        with pytest.raises(InvalidRegionError, match="non-negative"):
            col_track["chr1", -1]

    def test_invalid_key_type(self, col_track):
        with pytest.raises(InvalidRegionError):
            col_track[42]

    def test_too_many_elements(self, col_track):
        with pytest.raises(InvalidRegionError):
            col_track["chr1", 0:10, 0, "extra"]

    def test_column_name_not_found(self, col_track):
        with pytest.raises(ColumnNotFoundError):
            col_track["chr1", 0:10, "nonexistent"]

    def test_column_index_out_of_range(self, col_track):
        with pytest.raises(ColumnNotFoundError):
            col_track["chr1", 0:10, 99]

    def test_column_negative_index(self, col_track):
        with pytest.raises(ColumnNotFoundError):
            col_track["chr1", 0:10, -1]

    def test_invalid_position_type(self, col_track):
        with pytest.raises(InvalidRegionError):
            col_track["chr1", "not_a_slice"]

    def test_invalid_column_type(self, col_track):
        with pytest.raises(InvalidRegionError):
            col_track["chr1", 0:10, 3.14]


# ── query() and __getitem__ equivalence ──────────────────────────


class TestQueryGetitemEquivalence:
    """query() and __getitem__ should return identical data for equivalent args."""

    def test_whole_contig(self, col_track):
        q = col_track.query("chr1")
        g = col_track["chr1"]
        np.testing.assert_array_equal(q, g)

    def test_range(self, col_track):
        q = col_track.query("chr1:100-200")
        g = col_track["chr1", 100:200]
        np.testing.assert_array_equal(q, g)

    def test_range_scalar(self, scalar_track):
        q = scalar_track.query("chr1:100-200")
        g = scalar_track["chr1", 100:200]
        np.testing.assert_array_equal(q, g)

    def test_single_column_string(self, col_track):
        q = col_track.query("chr1:0-50", columns="sample_B")
        g = col_track["chr1", 0:50, "sample_B"]
        np.testing.assert_array_equal(q, g)


# ── second contig ────────────────────────────────────────────────


class TestSecondContig:
    """Ensure queries work on chr2 (different length)."""

    def test_query_chr2(self, store):
        track = store.create_track("cov", dtype="uint16", columns=["s1"])
        arr = track.zarr_array("chr2")
        data = np.ones((500, 1), dtype="uint16") * 42
        arr[:] = data
        result = track.query("chr2:0-100", columns="s1")
        assert result.shape == (100,)
        assert np.all(result == 42)

    def test_getitem_chr2(self, store):
        track = store.create_track("cov2", dtype="uint16")
        arr = track.zarr_array("chr2")
        data = np.full(500, 7, dtype="uint16")
        arr[:] = data
        result = track["chr2", 0:100]
        assert result.shape == (100,)
        assert np.all(result == 7)

    def test_chr2_end_boundary(self, store):
        track = store.create_track("cov3", dtype="uint16")
        with pytest.raises(InvalidRegionError, match="exceeds contig length"):
            track.query("chr2:0-501")


# ── fill value (unwritten data) ──────────────────────────────────


class TestFillValue:
    """Querying unwritten regions returns the fill value."""

    def test_uint32_fill_value(self, col_track):
        result = col_track.query("chr2:0-10")
        assert result.shape == (10, 3)
        np.testing.assert_array_equal(result, 0)

    def test_float32_fill_value(self, scalar_track):
        result = scalar_track.query("chr2:0-10")
        assert result.shape == (10,)
        assert np.all(np.isnan(result))
