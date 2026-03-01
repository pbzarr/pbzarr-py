"""Tests for pbzarr.store — create_store, open_store, PbzStore."""

import numpy as np
import pytest
import zarr
from zarr.core.dtype.npy.string import VariableLengthUTF8

import pbzarr
from pbzarr.exceptions import ContigNotFoundError, PbzError
from pbzarr.store import PbzStore, _PBZ_VERSION, create_store, open_store


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

CONTIGS = ["chr1", "chr2", "chr3"]
CONTIG_LENGTHS = [248_956_422, 242_193_529, 198_295_559]


@pytest.fixture()
def store(tmp_path):
    """Create a fresh PBZ store on disk."""
    path = tmp_path / "test.pbz.zarr"
    return create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)


@pytest.fixture()
def mem_store():
    """Create a fresh in-memory PBZ store."""
    return create_store({}, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)


# ------------------------------------------------------------------
# create_store
# ------------------------------------------------------------------


class TestCreateStore:
    """Tests for create_store()."""

    def test_returns_pbzstore(self, store):
        assert isinstance(store, PbzStore)

    def test_root_is_zarr_group(self, store):
        assert isinstance(store.root, zarr.Group)

    def test_root_pbz_metadata(self, store):
        meta = store.root.attrs["pbz"]
        assert meta["version"] == _PBZ_VERSION

    def test_contigs_array(self, store):
        arr = store.root["contigs"][:]
        assert list(arr) == CONTIGS

    def test_contig_lengths_array(self, store):
        arr = store.root["contig_lengths"][:]
        np.testing.assert_array_equal(arr, CONTIG_LENGTHS)

    def test_tracks_group_exists(self, store):
        tracks = store.root["tracks"]
        assert isinstance(tracks, zarr.Group)

    def test_contigs_property(self, store):
        assert store.contigs == CONTIGS

    def test_contig_lengths_property(self, store):
        assert store.contig_lengths == dict(zip(CONTIGS, CONTIG_LENGTHS))

    def test_backend_default_numpy(self, store):
        assert store.backend == pbzarr.Backend.NUMPY

    def test_tracks_empty(self, store):
        assert store.tracks() == []

    def test_in_memory(self, mem_store):
        assert isinstance(mem_store, PbzStore)
        assert mem_store.contigs == CONTIGS

    def test_repr(self, store):
        r = repr(store)
        assert "PbzStore" in r
        assert "contigs=3" in r
        assert "tracks=0" in r
        assert "numpy" in r


class TestCreateStoreValidation:
    """Validation tests for create_store()."""

    def test_length_mismatch_raises(self, tmp_path):
        with pytest.raises(PbzError, match="same length"):
            create_store(
                tmp_path / "bad.pbz.zarr",
                contigs=["chr1"],
                contig_lengths=[100, 200],
            )

    def test_empty_contigs_raises(self, tmp_path):
        with pytest.raises(PbzError, match="At least one contig"):
            create_store(
                tmp_path / "bad.pbz.zarr",
                contigs=[],
                contig_lengths=[],
            )


# ------------------------------------------------------------------
# open_store
# ------------------------------------------------------------------


class TestOpenStore:
    """Tests for open_store()."""

    def test_open_existing(self, tmp_path):
        path = tmp_path / "test.pbz.zarr"
        create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s = open_store(path)
        assert isinstance(s, PbzStore)
        assert s.contigs == CONTIGS

    def test_open_readonly(self, tmp_path):
        path = tmp_path / "test.pbz.zarr"
        create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s = open_store(path, mode="r")
        assert isinstance(s, PbzStore)

    def test_open_readwrite(self, tmp_path):
        path = tmp_path / "test.pbz.zarr"
        create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s = open_store(path, mode="r+")
        assert isinstance(s, PbzStore)

    def test_open_with_backend_string(self, tmp_path):
        path = tmp_path / "test.pbz.zarr"
        create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s = open_store(path, backend="numpy")
        assert s.backend == pbzarr.Backend.NUMPY

    def test_open_with_backend_enum(self, tmp_path):
        path = tmp_path / "test.pbz.zarr"
        create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s = open_store(path, backend=pbzarr.Backend.NUMPY)
        assert s.backend == pbzarr.Backend.NUMPY

    def test_open_contig_lengths(self, tmp_path):
        path = tmp_path / "test.pbz.zarr"
        create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        s = open_store(path)
        assert s.contig_lengths == dict(zip(CONTIGS, CONTIG_LENGTHS))


class TestOpenStoreValidation:
    """Validation tests for open_store()."""

    def test_invalid_mode_raises(self, tmp_path):
        path = tmp_path / "test.pbz.zarr"
        create_store(path, contigs=CONTIGS, contig_lengths=CONTIG_LENGTHS)
        with pytest.raises(PbzError, match="Invalid mode"):
            open_store(path, mode="w")

    def test_missing_pbz_attr_raises(self, tmp_path):
        """Store without 'pbz' root attribute is rejected."""
        path = tmp_path / "not_pbz.zarr"
        root = zarr.open_group(str(path), mode="w")
        contigs_arr = root.create_array(
            "contigs", shape=(1,), dtype=VariableLengthUTF8()
        )
        contigs_arr[:] = np.array(["chr1"], dtype=object)
        root.create_array("contig_lengths", data=np.array([100], dtype="int64"))
        with pytest.raises(PbzError, match="missing 'pbz' attribute"):
            open_store(path)

    def test_missing_version_raises(self, tmp_path):
        """Store with 'pbz' attr but no 'version' key is rejected."""
        path = tmp_path / "no_ver.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs["pbz"] = {}
        contigs_arr = root.create_array(
            "contigs", shape=(1,), dtype=VariableLengthUTF8()
        )
        contigs_arr[:] = np.array(["chr1"], dtype=object)
        root.create_array("contig_lengths", data=np.array([100], dtype="int64"))
        with pytest.raises(PbzError, match="missing 'version'"):
            open_store(path)

    def test_missing_contigs_array_raises(self, tmp_path):
        """Store without 'contigs' array is rejected."""
        path = tmp_path / "no_contigs.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs["pbz"] = {"version": "0.1"}
        root.create_array("contig_lengths", data=np.array([100], dtype="int64"))
        with pytest.raises(PbzError, match="missing 'contigs' array"):
            open_store(path)

    def test_missing_contig_lengths_raises(self, tmp_path):
        """Store without 'contig_lengths' array is rejected."""
        path = tmp_path / "no_lengths.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs["pbz"] = {"version": "0.1"}
        contigs_arr = root.create_array(
            "contigs", shape=(1,), dtype=VariableLengthUTF8()
        )
        contigs_arr[:] = np.array(["chr1"], dtype=object)
        with pytest.raises(PbzError, match="missing 'contig_lengths' array"):
            open_store(path)


# ------------------------------------------------------------------
# PbzStore.validate_contig
# ------------------------------------------------------------------


class TestValidateContig:
    """Tests for PbzStore.validate_contig()."""

    def test_valid_contig(self, store):
        store.validate_contig("chr1")  # should not raise

    def test_invalid_contig_raises(self, store):
        with pytest.raises(ContigNotFoundError, match="chrX"):
            store.validate_contig("chrX")

    def test_error_lists_available(self, store):
        with pytest.raises(ContigNotFoundError) as exc_info:
            store.validate_contig("chrX")
        assert exc_info.value.available == CONTIGS


# ------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------


class TestCaching:
    """Verify contigs/contig_lengths are cached."""

    def test_contigs_cached(self, store):
        first = store.contigs
        second = store.contigs
        assert first is second

    def test_contig_lengths_cached(self, store):
        first = store.contig_lengths
        second = store.contig_lengths
        assert first is second


# ------------------------------------------------------------------
# Public API access via pbzarr namespace
# ------------------------------------------------------------------


class TestPublicAPI:
    """Verify create/open are accessible from the top-level pbzarr namespace."""

    def test_create_alias(self):
        assert pbzarr.create is create_store

    def test_open_alias(self):
        assert pbzarr.open is open_store

    def test_pbzstore_exported(self):
        assert pbzarr.PbzStore is PbzStore

    def test_create_via_namespace(self, tmp_path):
        path = tmp_path / "ns.pbz.zarr"
        s = pbzarr.create(path, contigs=["chr1"], contig_lengths=[1000])
        assert isinstance(s, PbzStore)
        assert s.contigs == ["chr1"]
