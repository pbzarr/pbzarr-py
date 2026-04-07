"""Tests for pbzarr.track — Track class and Track.create."""

import numpy as np
import pytest
import zarr

import pbzarr
from pbzarr.exceptions import (
    ColumnNotFoundError,
    ContigNotFoundError,
    PbzError,
    TrackNotFoundError,
)
from pbzarr.store import create_store
from pbzarr.track import Track

# Use small contig lengths for fast tests.
CONTIGS = ["chr1", "chr2"]
CONTIG_LENGTHS = [10_000, 8_000]
COLUMNS = ["sample_A", "sample_B", "sample_C"]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def store():
    """In-memory PBZ store."""
    return create_store({}, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)


@pytest.fixture()
def columnar_track(store):
    """Track with columns (the common case)."""
    return store.create_track("depths", dtype="uint32", columns=COLUMNS)


@pytest.fixture()
def scalar_track(store):
    """Track without columns."""
    return store.create_track("mean_depth", dtype="float32")


# ------------------------------------------------------------------
# create_track — basic creation
# ------------------------------------------------------------------


class TestCreateTrack:
    """Tests for create_track() and store.create_track()."""

    def test_returns_track(self, columnar_track):
        assert isinstance(columnar_track, Track)

    def test_group_is_zarr_group(self, columnar_track):
        assert isinstance(columnar_track.group, zarr.Group)

    def test_store_reference(self, columnar_track, store):
        assert columnar_track.store is store

    def test_backend_inherited(self, columnar_track):
        assert columnar_track.backend == pbzarr.Backend.NUMPY


class TestTrackMetadata:
    """Tests for perbase_zarr_track metadata attributes."""

    def test_dtype(self, columnar_track):
        assert columnar_track.dtype == "uint32"

    def test_has_columns_true(self, columnar_track):
        assert columnar_track.has_columns is True

    def test_has_columns_false(self, scalar_track):
        assert scalar_track.has_columns is False

    def test_chunk_size_default(self, columnar_track):
        assert columnar_track.chunk_size == 1_000_000

    def test_column_chunk_size_default(self, columnar_track):
        assert columnar_track.column_chunk_size == 16

    def test_column_chunk_size_none_for_scalar(self, scalar_track):
        assert scalar_track.column_chunk_size is None

    def test_description_none_by_default(self, columnar_track):
        assert columnar_track.description is None

    def test_source_none_by_default(self, columnar_track):
        assert columnar_track.source is None

    def test_description_set(self, store):
        track = store.create_track(
            "depths2",
            dtype="uint32",
            columns=COLUMNS,
            description="Read depth from BAM files",
        )
        assert track.description == "Read depth from BAM files"

    def test_source_set(self, store):
        track = store.create_track(
            "depths2", dtype="uint32", columns=COLUMNS, source="perbase v0.1.0"
        )
        assert track.source == "perbase v0.1.0"

    def test_full_metadata_dict(self, columnar_track):
        meta = columnar_track.metadata
        assert meta["dtype"] == "uint32"
        assert meta["has_columns"] is True
        assert meta["chunk_size"] == 1_000_000
        assert meta["column_chunk_size"] == 16

    def test_extra_metadata(self, store):
        track = store.create_track(
            "depths2", dtype="uint32", columns=COLUMNS, extra_metadata={"min_depth": 10}
        )
        assert track.metadata["min_depth"] == 10


class TestColumnsArray:
    """Tests for the columns array."""

    def test_columns_list(self, columnar_track):
        assert columnar_track.columns == COLUMNS

    def test_num_columns(self, columnar_track):
        assert columnar_track.num_columns == 3

    def test_columns_none_for_scalar(self, scalar_track):
        assert scalar_track.columns is None

    def test_num_columns_none_for_scalar(self, scalar_track):
        assert scalar_track.num_columns is None


# ------------------------------------------------------------------
# Per-contig data arrays
# ------------------------------------------------------------------


class TestDataArrays:
    """Tests for per-contig data arrays created by create_track."""

    def test_array_exists_per_contig(self, columnar_track):
        for contig in CONTIGS:
            arr = columnar_track.zarr_array(contig)
            assert isinstance(arr, zarr.Array)

    def test_columnar_shape(self, columnar_track):
        arr = columnar_track.zarr_array("chr1")
        assert arr.shape == (10_000, 3)

    def test_columnar_shape_chr2(self, columnar_track):
        arr = columnar_track.zarr_array("chr2")
        assert arr.shape == (8_000, 3)

    def test_scalar_shape(self, scalar_track):
        arr = scalar_track.zarr_array("chr1")
        assert arr.shape == (10_000,)

    def test_dtype_matches(self, columnar_track):
        arr = columnar_track.zarr_array("chr1")
        assert arr.dtype == np.dtype("uint32")

    def test_scalar_dtype(self, scalar_track):
        arr = scalar_track.zarr_array("chr1")
        assert arr.dtype == np.dtype("float32")

    def test_array_dimensions_columnar(self, columnar_track):
        arr = columnar_track.zarr_array("chr1")
        assert arr.attrs["_ARRAY_DIMENSIONS"] == ["position", "column"]

    def test_array_dimensions_scalar(self, scalar_track):
        arr = scalar_track.zarr_array("chr1")
        assert arr.attrs["_ARRAY_DIMENSIONS"] == ["position"]


class TestFillValues:
    """Tests for default fill values (SPEC §7)."""

    def test_uint32_fill_zero(self, columnar_track):
        arr = columnar_track.zarr_array("chr1")
        assert arr.fill_value == 0

    def test_float32_fill_nan(self, scalar_track):
        arr = scalar_track.zarr_array("chr1")
        assert np.isnan(arr.fill_value)

    def test_bool_fill_false(self, store):
        track = store.create_track("mask", dtype="bool")
        arr = track.zarr_array("chr1")
        assert arr.fill_value is np.False_

    def test_custom_fill_value(self, store):
        track = store.create_track("depths2", dtype="uint32", fill_value=255)
        arr = track.zarr_array("chr1")
        assert arr.fill_value == 255


class TestChunking:
    """Tests for chunk sizing."""

    def test_chunks_capped_to_contig_length(self, columnar_track):
        # chr1 is 10_000 bp, default chunk_size is 1_000_000 → chunk = 10_000
        arr = columnar_track.zarr_array("chr1")
        assert arr.chunks[0] == 10_000

    def test_column_chunks_capped_to_num_columns(self, columnar_track):
        # 3 columns, default column_chunk_size is 16 → chunk = 3
        arr = columnar_track.zarr_array("chr1")
        assert arr.chunks[1] == 3

    def test_custom_chunk_size(self, store):
        track = store.create_track(
            "depths2", dtype="uint32", columns=COLUMNS, chunk_size=5_000
        )
        arr = track.zarr_array("chr1")
        assert arr.chunks[0] == 5_000

    def test_scalar_chunks(self, scalar_track):
        arr = scalar_track.zarr_array("chr1")
        # Only one dimension — capped to contig length
        assert arr.chunks == (10_000,)


# ------------------------------------------------------------------
# All supported dtypes
# ------------------------------------------------------------------


class TestAllDtypes:
    """Test that all SPEC §3.1 dtypes can be created."""

    @pytest.mark.parametrize(
        "dtype_str",
        [
            "uint8",
            "uint16",
            "uint32",
            "int8",
            "int16",
            "int32",
            "float32",
            "float64",
            "bool",
        ],
    )
    def test_create_dtype(self, store, dtype_str):
        track = store.create_track(f"track_{dtype_str}", dtype=dtype_str)
        assert track.dtype == dtype_str
        arr = track.zarr_array("chr1")
        assert arr.dtype == np.dtype(dtype_str)


# ------------------------------------------------------------------
# create_track validation
# ------------------------------------------------------------------


class TestCreateTrackValidation:
    """Validation tests for create_track."""

    def test_invalid_dtype_raises(self, store):
        with pytest.raises(PbzError, match="Invalid dtype"):
            store.create_track("bad", dtype="int64")

    def test_invalid_dtype_lists_valid(self, store):
        with pytest.raises(PbzError, match="uint32"):
            store.create_track("bad", dtype="complex128")


# ------------------------------------------------------------------
# Escape hatches (SPEC §11.1)
# ------------------------------------------------------------------


class TestEscapeHatches:
    """Tests for zarr_array() escape hatch."""

    def test_zarr_array_returns_zarr_array(self, columnar_track):
        arr = columnar_track.zarr_array("chr1")
        assert isinstance(arr, zarr.Array)

    def test_invalid_contig_raises(self, columnar_track):
        with pytest.raises(ContigNotFoundError, match="chrX"):
            columnar_track.zarr_array("chrX")


# ------------------------------------------------------------------
# Column name → index resolution
# ------------------------------------------------------------------


class TestColumnResolution:
    """Tests for column name → index mapping."""

    def test_resolve_existing(self, columnar_track):
        assert columnar_track._column_name_to_idx("sample_A") == 0
        assert columnar_track._column_name_to_idx("sample_B") == 1
        assert columnar_track._column_name_to_idx("sample_C") == 2

    def test_resolve_missing_raises(self, columnar_track):
        with pytest.raises(ColumnNotFoundError, match="sample_X"):
            columnar_track._column_name_to_idx("sample_X")

    def test_error_lists_available(self, columnar_track):
        with pytest.raises(ColumnNotFoundError) as exc_info:
            columnar_track._column_name_to_idx("sample_X")
        assert exc_info.value.available == COLUMNS

    def test_index_cached(self, columnar_track):
        """Column index map is built once and reused."""
        columnar_track._column_name_to_idx("sample_A")
        first = columnar_track._column_index
        columnar_track._column_name_to_idx("sample_B")
        second = columnar_track._column_index
        assert first is second


# ------------------------------------------------------------------
# Store-level track access (PbzStore.__getitem__, __contains__, tracks)
# ------------------------------------------------------------------


class TestStoreTrackAccess:
    """Tests for accessing tracks through PbzStore."""

    def test_tracks_lists_track(self, store, columnar_track):
        assert "depths" in store.tracks()

    def test_getitem_returns_track(self, store, columnar_track):
        track = store["depths"]
        assert isinstance(track, Track)
        assert track.dtype == "uint32"

    def test_contains_true(self, store, columnar_track):
        assert "depths" in store

    def test_contains_false(self, store):
        assert "nonexistent" not in store

    def test_getitem_missing_raises(self, store):
        with pytest.raises(TrackNotFoundError, match="nonexistent"):
            store["nonexistent"]

    def test_multiple_tracks(self, store):
        store.create_track("depths", dtype="uint32", columns=COLUMNS)
        store.create_track("mean_depth", dtype="float32")
        names = store.tracks()
        assert sorted(names) == ["depths", "mean_depth"]

    def test_nested_track(self, store):
        store.create_track("masks/callable", dtype="bool")
        assert "masks/callable" in store.tracks()
        track = store["masks/callable"]
        assert track.dtype == "bool"

    def test_repr(self, columnar_track):
        r = repr(columnar_track)
        assert "Track" in r
        assert "uint32" in r
        assert "columns=3" in r

    def test_repr_scalar(self, scalar_track):
        r = repr(scalar_track)
        assert "Track" in r
        assert "float32" in r
        assert "columns=" not in r


# ------------------------------------------------------------------
# Roundtrip: create → open → access
# ------------------------------------------------------------------


class TestRoundtrip:
    """Test create store + track, close, reopen, and verify."""

    def test_roundtrip_on_disk(self, tmp_path):
        path = tmp_path / "roundtrip.pbz.zarr"
        s = create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s.create_track("depths", dtype="uint32", columns=COLUMNS)

        # Re-open
        s2 = pbzarr.open(path, mode="r")
        assert "depths" in s2.tracks()
        track = s2["depths"]
        assert track.dtype == "uint32"
        assert track.columns == COLUMNS
        assert track.zarr_array("chr1").shape == (10_000, 3)

    def test_roundtrip_store_repr_with_tracks(self, tmp_path):
        path = tmp_path / "repr.pbz.zarr"
        s = create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s.create_track("depths", dtype="uint32", columns=COLUMNS)
        s.create_track("mask", dtype="bool")

        s2 = pbzarr.open(path)
        r = repr(s2)
        assert "tracks=2" in r
