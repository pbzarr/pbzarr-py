"""Track — a named group of per-base data arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zarr
from zarr.core.dtype.npy.string import VariableLengthUTF8

from pbzarr._backends import Backend, get_data
from pbzarr.exceptions import (
    ColumnNotFoundError,
    ContigNotFoundError,
    InvalidRegionError,
)
from pbzarr.region import Region, parse_region

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

    Do not instantiate directly — use [`PbzStore.create_track`][pbzarr.PbzStore.create_track]
    or `store[name]` indexing instead.
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
        meta = self._group.attrs["perbase_zarr_track"]
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

    def _resolve_position_slice(
        self, region: Region, contig_length: int
    ) -> slice:
        """Turn a Region into a validated position slice."""
        start = region.start if region.start is not None else 0
        end = region.end if region.end is not None else contig_length

        if end > contig_length:
            raise InvalidRegionError(
                f"End position ({end}) exceeds contig length ({contig_length}) "
                f"for contig {region.contig!r}."
            )
        return slice(start, end)

    def _resolve_column_slice(
        self, columns: str | list[str] | None
    ) -> int | list[int] | slice:
        """Resolve column argument to a zarr-compatible index.

        Returns an int for a single string (squeeze), a list[int] for a
        list of names (no squeeze), or slice(None) for None (all columns).
        """
        if columns is None:
            if self.has_columns:
                return slice(None)
            return slice(None)

        if not self.has_columns:
            raise ColumnNotFoundError(
                columns if isinstance(columns, str) else columns[0],
                available=None,
            )

        if isinstance(columns, str):
            return self._column_name_to_idx(columns)

        indices: list[int] = []
        for name in columns:
            indices.append(self._column_name_to_idx(name))
        return indices

    def query(
        self,
        region: str | tuple,
        *,
        columns: str | list[str] | None = None,
        one_based: bool = False,
    ) -> object:
        """Query data for a genomic region.

        Parameters
        ----------
        region
            Region string (``"chr1:1000-2000"``) or tuple
            (``("chr1", 1000, 2000)``).
        columns
            Column filter. A single string returns a 1D array (squeeze).
            A list of strings returns a 2D array. ``None`` returns all
            columns.
        one_based
            If ``True``, interpret coordinates as 1-based inclusive.

        Returns
        -------
        object
            numpy.ndarray or dask.array.Array depending on backend.
        """
        parsed = parse_region(region, one_based=one_based)
        self._store.validate_contig(parsed.contig)
        contig_length = self._store.contig_lengths[parsed.contig]
        pos_slice = self._resolve_position_slice(parsed, contig_length)
        col_index = self._resolve_column_slice(columns)

        arr = self.zarr_array(parsed.contig)

        slices: tuple = (
            (pos_slice, col_index) if self.has_columns else (pos_slice,)
        )
        return get_data(arr, slices, self.backend)

    def _parse_getitem_key(
        self, key: object
    ) -> tuple[str, slice, int | list[int] | slice]:
        """Parse ``__getitem__`` key into (contig, pos_slice, col_index).

        Supports:
          track["chr1"]                     -> whole contig
          track["chr1", 100:200]            -> position range, all columns
          track["chr1", 100:200, :]         -> same
          track["chr1", 100:200, "colA"]    -> single column (squeeze)
          track["chr1", 100:200, 0:5]       -> column integer slice
          track["chr1", 100:200, 3]         -> single column by index (squeeze)
        """
        if isinstance(key, str):
            contig = key
            pos_slice = slice(None)
            col_index: int | list[int] | slice = slice(None)
            self._store.validate_contig(contig)
            contig_length = self._store.contig_lengths[contig]
            pos_slice = slice(0, contig_length)
            return contig, pos_slice, col_index

        if not isinstance(key, tuple):
            raise InvalidRegionError(
                f"Track key must be a string or tuple, got {type(key).__name__}"
            )

        if len(key) < 1 or len(key) > 3:
            raise InvalidRegionError(
                f"Track key must have 1-3 elements, got {len(key)}"
            )

        contig = key[0]
        if not isinstance(contig, str):
            raise InvalidRegionError(
                f"First element (contig) must be a string, got {type(contig).__name__}"
            )
        self._store.validate_contig(contig)
        contig_length = self._store.contig_lengths[contig]

        if len(key) == 1:
            return contig, slice(0, contig_length), slice(None)

        pos_key = key[1]
        if isinstance(pos_key, slice):
            start = pos_key.start if pos_key.start is not None else 0
            stop = pos_key.stop if pos_key.stop is not None else contig_length
            if stop > contig_length:
                raise InvalidRegionError(
                    f"End position ({stop}) exceeds contig length "
                    f"({contig_length}) for contig {contig!r}."
                )
            pos_slice = slice(start, stop)
        elif isinstance(pos_key, int):
            if pos_key < 0:
                raise InvalidRegionError(
                    f"Position must be non-negative, got {pos_key}"
                )
            if pos_key >= contig_length:
                raise InvalidRegionError(
                    f"Position ({pos_key}) exceeds contig length "
                    f"({contig_length}) for contig {contig!r}."
                )
            pos_slice = slice(pos_key, pos_key + 1)
        else:
            raise InvalidRegionError(
                f"Position must be a slice or int, got {type(pos_key).__name__}"
            )

        if len(key) == 2:
            return contig, pos_slice, slice(None)

        col_key = key[2]
        col_index = self._resolve_getitem_col(col_key)
        return contig, pos_slice, col_index

    def _resolve_getitem_col(
        self, col_key: object
    ) -> int | list[int] | slice:
        """Resolve the column component of a __getitem__ key."""
        if isinstance(col_key, slice):
            return col_key
        if isinstance(col_key, int):
            if self.has_columns:
                num = self.num_columns
                assert num is not None
                if col_key < 0 or col_key >= num:
                    raise ColumnNotFoundError(
                        str(col_key),
                        available=self.columns,
                    )
            return col_key
        if isinstance(col_key, str):
            return self._column_name_to_idx(col_key)
        if isinstance(col_key, list):
            indices: list[int] = []
            for item in col_key:
                if isinstance(item, str):
                    indices.append(self._column_name_to_idx(item))
                elif isinstance(item, int):
                    indices.append(item)
                else:
                    raise InvalidRegionError(
                        f"Column list items must be str or int, "
                        f"got {type(item).__name__}"
                    )
            return indices
        raise InvalidRegionError(
            f"Column key must be a slice, int, str, or list, "
            f"got {type(col_key).__name__}"
        )

    def __getitem__(self, key: object) -> object:
        """Slice-based data access.

        Examples
        --------
        >>> track["chr1"]
        >>> track["chr1", 100:200]
        >>> track["chr1", 100:200, :]
        >>> track["chr1", 100:200, "sample_A"]
        >>> track["chr1", 100:200, 0:5]
        """
        contig, pos_slice, col_index = self._parse_getitem_key(key)
        arr = self.zarr_array(contig)

        slices: tuple = (
            (pos_slice, col_index) if self.has_columns else (pos_slice,)
        )
        return get_data(arr, slices, self.backend)

    def __setitem__(self, key: object, value: object) -> None:
        """Slice-based data write.

        Examples
        --------
        >>> track["chr1"] = data
        >>> track["chr1", 100:200] = data
        >>> track["chr1", 100:200, :] = data
        >>> track["chr1", 100:200, "sample_A"] = data
        >>> track["chr1", 100:200, 0:5] = data
        """
        contig, pos_slice, col_index = self._parse_getitem_key(key)
        arr = self.zarr_array(contig)

        if self.has_columns:
            arr[pos_slice, col_index] = value  # type: ignore[index,assignment]
        else:
            arr[pos_slice] = value  # type: ignore[assignment]

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
        track_group.attrs["perbase_zarr_track"] = track_meta

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
