"""Tests for Track.__setitem__() — Phase 5 (write path)."""

import numpy as np
import pytest

from pbzarr.exceptions import (
    ColumnNotFoundError,
    ContigNotFoundError,
    InvalidRegionError,
)
from pbzarr.store import create_store

CONTIGS = ["chr1", "chr2"]
CONTIG_LENGTHS = [1_000, 500]
COLUMNS = ["sample_A", "sample_B", "sample_C"]


@pytest.fixture()
def store():
    return create_store({}, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)


@pytest.fixture()
def col_track(store):
    return store.create_track("depths", dtype="uint32", columns=COLUMNS)


@pytest.fixture()
def scalar_track(store):
    return store.create_track("mean_depth", dtype="float32")


# ── whole-contig writes ──────────────────────────────────────────


class TestWriteWholeContig:
    def test_columnar(self, col_track):
        data = np.ones((1_000, 3), dtype="uint32") * 42
        col_track["chr1"] = data
        result = col_track["chr1"]
        np.testing.assert_array_equal(result, data)

    def test_scalar(self, scalar_track):
        data = np.arange(1_000, dtype="float32")
        scalar_track["chr1"] = data
        result = scalar_track["chr1"]
        np.testing.assert_array_equal(result, data)

    def test_chr2(self, scalar_track):
        data = np.full(500, 7.0, dtype="float32")
        scalar_track["chr2"] = data
        result = scalar_track["chr2"]
        np.testing.assert_array_equal(result, data)


# ── range writes ─────────────────────────────────────────────────


class TestWriteRange:
    def test_columnar_range(self, col_track):
        data = np.ones((100, 3), dtype="uint32") * 99
        col_track["chr1", 100:200] = data
        result = col_track["chr1", 100:200]
        np.testing.assert_array_equal(result, data)

    def test_scalar_range(self, scalar_track):
        data = np.full(50, 3.14, dtype="float32")
        scalar_track["chr1", 200:250] = data
        result = scalar_track["chr1", 200:250]
        np.testing.assert_array_equal(result, data)

    def test_partial_write_doesnt_clobber(self, scalar_track):
        """Writing to a range shouldn't affect other positions."""
        full = np.arange(1_000, dtype="float32")
        scalar_track["chr1"] = full
        scalar_track["chr1", 500:600] = np.zeros(100, dtype="float32")
        before = scalar_track["chr1", 0:500]
        np.testing.assert_array_equal(before, np.arange(500, dtype="float32"))
        after = scalar_track["chr1", 600:1000]
        np.testing.assert_array_equal(
            after, np.arange(600, 1000, dtype="float32")
        )


# ── column-specific writes ───────────────────────────────────────


class TestWriteColumns:
    def test_write_single_column_by_name(self, col_track):
        col_track["chr1", 0:10, "sample_B"] = np.full(10, 77, dtype="uint32")
        result = col_track["chr1", 0:10, "sample_B"]
        np.testing.assert_array_equal(result, 77)

    def test_write_single_column_by_index(self, col_track):
        col_track["chr1", 0:10, 2] = np.full(10, 55, dtype="uint32")
        result = col_track["chr1", 0:10, 2]
        np.testing.assert_array_equal(result, 55)

    def test_write_column_slice(self, col_track):
        data = np.ones((10, 2), dtype="uint32") * 33
        col_track["chr1", 0:10, 0:2] = data
        result = col_track["chr1", 0:10, 0:2]
        np.testing.assert_array_equal(result, data)

    def test_write_column_doesnt_clobber_others(self, col_track):
        full = np.arange(1_000 * 3, dtype="uint32").reshape(1_000, 3)
        col_track["chr1"] = full
        col_track["chr1", 0:10, "sample_A"] = np.zeros(10, dtype="uint32")
        # sample_B and sample_C should be untouched
        result_b = col_track["chr1", 0:10, "sample_B"]
        np.testing.assert_array_equal(result_b, full[0:10, 1])
        result_c = col_track["chr1", 0:10, "sample_C"]
        np.testing.assert_array_equal(result_c, full[0:10, 2])

    def test_write_all_columns_explicit(self, col_track):
        data = np.ones((20, 3), dtype="uint32") * 11
        col_track["chr1", 0:20, :] = data
        result = col_track["chr1", 0:20]
        np.testing.assert_array_equal(result, data)


# ── write then read roundtrip ────────────────────────────────────


class TestWriteReadRoundtrip:
    def test_write_read_matches(self, col_track):
        data = np.random.default_rng(42).integers(0, 1000, size=(200, 3)).astype("uint32")
        col_track["chr1", 300:500] = data
        result = col_track.query("chr1:300-500")
        np.testing.assert_array_equal(result, data)

    def test_write_read_scalar_matches(self, scalar_track):
        data = np.random.default_rng(7).random(100).astype("float32")
        scalar_track["chr1", 0:100] = data
        result = scalar_track.query("chr1:0-100")
        np.testing.assert_array_equal(result, data)

    def test_overwrite(self, scalar_track):
        """Writing twice to the same region uses the second write."""
        scalar_track["chr1", 0:10] = np.ones(10, dtype="float32")
        scalar_track["chr1", 0:10] = np.full(10, 2.0, dtype="float32")
        result = scalar_track["chr1", 0:10]
        np.testing.assert_array_equal(result, 2.0)


# ── validation errors on write ───────────────────────────────────


class TestWriteValidation:
    def test_contig_not_found(self, col_track):
        with pytest.raises(ContigNotFoundError):
            col_track["chrX"] = np.zeros((1_000, 3), dtype="uint32")

    def test_end_exceeds_length(self, col_track):
        with pytest.raises(InvalidRegionError, match="exceeds contig length"):
            col_track["chr1", 0:9999] = np.zeros((9999, 3), dtype="uint32")

    def test_column_not_found(self, col_track):
        with pytest.raises(ColumnNotFoundError):
            col_track["chr1", 0:10, "nonexistent"] = np.zeros(10, dtype="uint32")

    def test_invalid_key_type(self, col_track):
        with pytest.raises(InvalidRegionError):
            col_track[42] = np.zeros(10, dtype="uint32")


# ── dtypes ───────────────────────────────────────────────────────


class TestWriteDtypes:
    def test_bool_track(self, store):
        track = store.create_track("mask", dtype="bool")
        data = np.array([True, False, True, True, False], dtype="bool")
        track["chr1", 0:5] = data
        result = track["chr1", 0:5]
        np.testing.assert_array_equal(result, data)

    def test_int8_track(self, store):
        track = store.create_track("scores", dtype="int8")
        data = np.array([-1, 0, 1, 127, -128], dtype="int8")
        track["chr1", 0:5] = data
        result = track["chr1", 0:5]
        np.testing.assert_array_equal(result, data)

    def test_float64_track(self, store):
        track = store.create_track("methyl", dtype="float64")
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float64")
        track["chr1", 0:5] = data
        result = track["chr1", 0:5]
        np.testing.assert_array_equal(result, data)
