"""Track — a named group of per-base data arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zarr
from zarr.core.dtype.npy.string import VariableLengthUTF8

from pbzarr._backends import Backend
from pbzarr.exceptions import (
    ColumnNotFoundError,
    ContigNotFoundError,
)

if TYPE_CHECKING:
    from pbzarr.store import PbzStore

_DTYPE_MAP: dict[str, np.dtype] = {
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "bool": np.dtype("bool"),
}

_DEFAULT_FILL_VALUES: dict[str, object] = {
    "uint8": 0,
    "uint16": 0,
    "uint32": 0,
    "int8": 0,
    "int16": 0,
    "int32": 0,
    "float32": float("nan"),
    "float64": float("nan"),
    "bool": False,
}

_DEFAULT_CHUNK_SIZE = 1_000_000
_DEFAULT_COLUMN_CHUNK_SIZE = 16


class Track:
    """A PBZ track wrapping a zarr group under ``/tracks/{name}/``.

    Do not instantiate directly — use :meth:`PbzStore.create_track` or
    :meth:`PbzStore.__getitem__` instead.
    """

    def __init__(self, group: zarr.Group, store: PbzStore) -> None:
        self._group = group
        self._store = store
        self._column_index: dict[str, int] | None = None

    @property
    def group(self) -> zarr.Group:
        """Escape hatch: raw zarr group for this track."""
        return self._group

    @property
    def store(self) -> PbzStore:
        """Parent PbzStore."""
        return self._store

    @property
    def backend(self) -> Backend:
        """Active backend (inherited from store)."""
        return self._store.backend

    @property
    def metadata(self) -> dict:
        """Full ``pbz_track`` metadata dict."""
        meta = self._group.attrs["pbz_track"]
        assert isinstance(meta, dict)
        return dict(meta)

    @property
    def dtype(self) -> str:
        """Track dtype as a string (e.g. ``'uint32'``)."""
        result: str = self.metadata["dtype"]
        return result

    @property
    def has_columns(self) -> bool:
        """Whether this track has a column dimension."""
        result: bool = self.metadata["has_columns"]
        return result

    @property
    def chunk_size(self) -> int:
        """Chunk size along the position axis (bp)."""
        result: int = self.metadata["chunk_size"]
        return result

    @property
    def column_chunk_size(self) -> int | None:
        """Chunk size along the column axis, or None if no columns."""
        return self.metadata.get("column_chunk_size")

    @property
    def description(self) -> str | None:
        """Human-readable description, or None."""
        return self.metadata.get("description")

    @property
    def source(self) -> str | None:
        """Tool/version that created this track, or None."""
        return self.metadata.get("source")

    @property
    def columns(self) -> list[str] | None:
        """Column names, or None if the track has no column dimension."""
        if not self.has_columns:
            return None
        arr = self._group["columns"]
        assert isinstance(arr, zarr.Array)
        return list(arr[:])  # type: ignore[arg-type]

    @property
    def num_columns(self) -> int | None:
        """Number of columns, or None if the track has no column dimension."""
        cols = self.columns
        return len(cols) if cols is not None else None

    def zarr_array(self, contig: str) -> zarr.Array:
        """Get the raw zarr.Array for a specific contig.

        Parameters
        ----------
        contig
            Contig name (must exist in the store).

        Returns
        -------
        zarr.Array

        Raises
        ------
        ContigNotFoundError
            If the contig doesn't exist in this track.
        """
        self._store.validate_contig(contig)
        try:
            arr = self._group[contig]
        except KeyError:
            raise ContigNotFoundError(
                contig, available=self._store.contigs
            ) from None
        if not isinstance(arr, zarr.Array):
            raise ContigNotFoundError(
                contig, available=self._store.contigs
            )
        return arr

    def _resolve_column_index(self) -> dict[str, int]:
        """Build and cache column name -> index mapping."""
        if self._column_index is None:
            cols = self.columns
            if cols is None:
                self._column_index = {}
            else:
                self._column_index = {name: i for i, name in enumerate(cols)}
        return self._column_index

    def _column_name_to_idx(self, name: str) -> int:
        """Resolve a single column name to its integer index."""
        idx_map = self._resolve_column_index()
        try:
            return idx_map[name]
        except KeyError:
            raise ColumnNotFoundError(
                name, available=self.columns
            ) from None

    def __repr__(self) -> str:
        parts = [f"dtype={self.dtype!r}"]
        if self.has_columns:
            parts.append(f"columns={self.num_columns}")
        parts.append(f"backend={self.backend.value!r}")
        return f"Track({', '.join(parts)})"

    @classmethod
    def create(
        cls,
        store: PbzStore,
        name: str,
        *,
        dtype: str,
        columns: list[str] | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        column_chunk_size: int = _DEFAULT_COLUMN_CHUNK_SIZE,
        fill_value: object | None = None,
        description: str | None = None,
        source: str | None = None,
        compressors: object | None = None,
        shards: tuple[int, ...] | None = None,
        extra_metadata: dict | None = None,
    ) -> Track:
        """Create a new track in a PBZ store.

        Parameters
        ----------
        store
            The parent PbzStore (must be writable).
        name
            Track name/path (e.g. ``"depths"`` or ``"masks/callable"``).
        dtype
            Data type string — one of: ``uint8``, ``uint16``, ``uint32``,
            ``int8``, ``int16``, ``int32``, ``float32``, ``float64``, ``bool``.
        columns
            Column names. If ``None``, the track has no column dimension.
        chunk_size
            Chunk size along the position axis (default 1,000,000).
        column_chunk_size
            Chunk size along the column axis (default 16). Ignored if no columns.
        fill_value
            Zarr fill value for unwritten chunks. If ``None``, uses SPEC defaults.
        description
            Human-readable description.
        source
            Tool/version that created this track.
        compressors
            Zarr codec(s) for compression. Passed through to zarr.
        shards
            Shard shape tuple. Passed through to zarr. ``None`` means no sharding.
        extra_metadata
            Additional key-value pairs to include in the ``pbz_track`` metadata.

        Returns
        -------
        Track
        """
        from pbzarr.exceptions import PbzError

        if dtype not in _DTYPE_MAP:
            valid = list(_DTYPE_MAP.keys())
            raise PbzError(f"Invalid dtype: {dtype!r}. Valid dtypes: {valid}")

        has_columns = columns is not None

        if fill_value is None:
            fill_value = _DEFAULT_FILL_VALUES[dtype]

        np_dtype = _DTYPE_MAP[dtype]

        track_meta: dict = {
            "dtype": dtype,
            "chunk_size": chunk_size,
            "has_columns": has_columns,
        }
        if has_columns:
            track_meta["column_chunk_size"] = column_chunk_size
        if description is not None:
            track_meta["description"] = description
        if source is not None:
            track_meta["source"] = source
        if extra_metadata:
            track_meta.update(extra_metadata)

        tracks_group = store.root["tracks"]
        assert isinstance(tracks_group, zarr.Group)
        track_group = tracks_group.create_group(name)
        track_group.attrs["pbz_track"] = track_meta

        if has_columns:
            assert columns is not None  # for type checker
            cols_arr = track_group.create_array(
                "columns", shape=(len(columns),), dtype=VariableLengthUTF8()  # type: ignore[arg-type]
            )
            cols_arr[:] = np.array(columns, dtype=object)

        contigs = store.contigs
        contig_lengths = store.contig_lengths
        for contig in contigs:
            length = contig_lengths[contig]
            if has_columns:
                assert columns is not None
                shape: tuple[int, ...] = (length, len(columns))
                chunks: tuple[int, ...] = (
                    min(chunk_size, length),
                    min(column_chunk_size, len(columns)),
                )
            else:
                shape = (length,)
                chunks = (min(chunk_size, length),)

            dims = ["position", "column"] if has_columns else ["position"]

            create_kwargs: dict = {
                "shape": shape,
                "chunks": chunks,
                "dtype": np_dtype,
                "fill_value": fill_value,
            }
            if compressors is not None:
                create_kwargs["compressors"] = compressors
            if shards is not None:
                create_kwargs["shards"] = shards

            arr = track_group.create_array(contig, **create_kwargs)
            arr.attrs["_ARRAY_DIMENSIONS"] = dims

        return cls(group=track_group, store=store)
