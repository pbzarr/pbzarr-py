"""Tests for the dask backend — Phase 6."""

import numpy as np
import pytest

dask = pytest.importorskip("dask")
import dask.array as da  # noqa: E402

from pbzarr._backends import Backend  # noqa: E402
from pbzarr.store import create_store, open_store  # noqa: E402

CONTIGS = ["chr1", "chr2"]
CONTIG_LENGTHS = [1_000, 500]
COLUMNS = ["sample_A", "sample_B", "sample_C"]


@pytest.fixture()
def store(tmp_path):
    """On-disk store opened with dask backend."""
    path = tmp_path / "test.pbz.zarr"
    s = create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
    track = s.create_track("depths", dtype="uint32", columns=COLUMNS)
    arr = track.zarr_array("chr1")
    arr[:] = np.arange(1_000 * 3, dtype="uint32").reshape(1_000, 3)
    scalar = s.create_track("mean_depth", dtype="float32")
    arr_s = scalar.zarr_array("chr1")
    arr_s[:] = np.arange(1_000, dtype="float32")
    return open_store(path, backend="dask")


@pytest.fixture()
def col_track(store):
    return store["depths"]


@pytest.fixture()
def scalar_track(store):
    return store["mean_depth"]


class TestDaskBackendStore:
    def test_backend_is_dask(self, store):
        assert store.backend == Backend.DASK

    def test_track_inherits_backend(self, col_track):
        assert col_track.backend == Backend.DASK


class TestDaskQuery:
    def test_returns_dask_array(self, col_track):
        result = col_track.query("chr1:0-100")
        assert isinstance(result, da.Array)

    def test_shape(self, col_track):
        result = col_track.query("chr1:100-200")
        assert result.shape == (100, 3)

    def test_compute_values(self, col_track):
        result = col_track.query("chr1:0-5").compute()
        expected = np.arange(5 * 3, dtype="uint32").reshape(5, 3)
        np.testing.assert_array_equal(result, expected)

    def test_scalar_track(self, scalar_track):
        result = scalar_track.query("chr1:0-10")
        assert isinstance(result, da.Array)
        assert result.shape == (10,)
        np.testing.assert_array_equal(result.compute(), np.arange(10, dtype="float32"))

    def test_whole_contig(self, col_track):
        result = col_track.query("chr1")
        assert result.shape == (1_000, 3)

    def test_single_column_squeeze(self, col_track):
        result = col_track.query("chr1:0-10", columns="sample_A")
        assert isinstance(result, da.Array)
        assert result.shape == (10,)

    def test_column_list(self, col_track):
        result = col_track.query("chr1:0-10", columns=["sample_A", "sample_C"])
        assert result.shape == (10, 2)

    def test_one_based(self, scalar_track):
        result = scalar_track.query("chr1:1-10", one_based=True).compute()
        np.testing.assert_array_equal(result, np.arange(0, 10, dtype="float32"))

    def test_tuple_region(self, col_track):
        result = col_track.query(("chr1", 50, 100))
        assert isinstance(result, da.Array)
        assert result.shape == (50, 3)


class TestDaskGetitem:
    def test_returns_dask_array(self, col_track):
        result = col_track["chr1", 0:100]
        assert isinstance(result, da.Array)

    def test_string_key(self, col_track):
        result = col_track["chr1"]
        assert result.shape == (1_000, 3)

    def test_slice_range(self, col_track):
        result = col_track["chr1", 100:200]
        assert result.shape == (100, 3)

    def test_column_by_name(self, col_track):
        result = col_track["chr1", 0:10, "sample_B"]
        assert isinstance(result, da.Array)
        assert result.shape == (10,)

    def test_column_int_slice(self, col_track):
        result = col_track["chr1", 0:10, 0:2]
        assert result.shape == (10, 2)

    def test_compute_matches_numpy(self, col_track, tmp_path):
        """Dask .compute() should match numpy backend results."""
        path = tmp_path / "test.pbz.zarr"
        np_store = open_store(path, backend="numpy")
        np_track = np_store["depths"]
        dask_result = col_track.query("chr1:200-300").compute()
        numpy_result = np_track.query("chr1:200-300")
        np.testing.assert_array_equal(dask_result, numpy_result)


class TestDaskLazy:
    """Verify dask arrays are truly lazy."""

    def test_no_compute_on_query(self, col_track):
        result = col_track.query("chr1:0-100")
        assert isinstance(result, da.Array)
        assert not isinstance(result, np.ndarray)

    def test_chained_operations_stay_lazy(self, col_track):
        result = col_track.query("chr1:0-100")
        mean = result.mean(axis=0)
        assert isinstance(mean, da.Array)
