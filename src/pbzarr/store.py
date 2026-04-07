"""PbzStore — PBZ store wrapping a zarr.Group root."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
import zarr.storage
from zarr.core.dtype.npy.string import VariableLengthUTF8

from pbzarr._backends import Backend
from pbzarr.exceptions import ContigNotFoundError, PbzError, TrackNotFoundError

if TYPE_CHECKING:
    from pbzarr.track import Track

_PBZ_VERSION = "0.1"


class PbzStore:
    """A PBZ store wrapping a zarr v3 root group.

    Do not instantiate directly — use [`pbzarr.create_store`][] or
    [`pbzarr.open_store`][] instead.
    """

    def __init__(self, root: zarr.Group, backend: Backend = Backend.NUMPY) -> None:
        self._root = root
        self._backend = backend
        self._contigs_list: list[str] | None = None
        self._contig_lengths_map: dict[str, int] | None = None

    @property
    def root(self) -> zarr.Group:
        """Escape hatch: raw zarr root group."""
        return self._root

    @property
    def backend(self) -> Backend:
        """Active backend."""
        return self._backend

    @property
    def contigs(self) -> list[str]:
        """Contig names in order."""
        if self._contigs_list is None:
            arr = self._root["contigs"]
            assert isinstance(arr, zarr.Array)
            self._contigs_list = list(arr[:])  # type: ignore[arg-type]
        return self._contigs_list

    @property
    def contig_lengths(self) -> dict[str, int]:
        """Mapping of contig name -> length in bp."""
        if self._contig_lengths_map is None:
            names = self.contigs
            arr = self._root["contig_lengths"]
            assert isinstance(arr, zarr.Array)
            lengths = arr[:]
            self._contig_lengths_map = dict(
                zip(names, (int(v) for v in lengths))  # type: ignore[union-attr]
            )
        return self._contig_lengths_map

    def tracks(self) -> list[str]:
        """List all track paths (e.g. ``['depths', 'masks/callable']``)."""
        tracks_group = self._root.get("tracks")
        if tracks_group is None or not isinstance(tracks_group, zarr.Group):
            return []
        return _find_tracks(tracks_group)

    def __getitem__(self, name: str) -> Track:
        """Get a Track by name. Raises TrackNotFoundError."""
        from pbzarr.track import Track

        tracks_group = self._root.get("tracks")
        if tracks_group is None:
            raise TrackNotFoundError(name, available=[])

        try:
            group = tracks_group[name]  # type: ignore[index]
        except KeyError:
            raise TrackNotFoundError(name, available=self.tracks()) from None

        if not isinstance(group, zarr.Group) or "perbase_zarr_track" not in group.attrs:
            raise TrackNotFoundError(name, available=self.tracks())

        return Track(group=group, store=self)

    def __contains__(self, name: str) -> bool:
        """Check if a track exists."""
        try:
            self[name]
            return True
        except TrackNotFoundError:
            return False

    def create_track(self, name: str, **kwargs: Any) -> Track:
        """Create a new track. See [`Track.create`][pbzarr.Track.create]."""
        from pbzarr.track import Track

        return Track.create(self, name, **kwargs)

    def validate_contig(self, contig: str) -> None:
        """Raise ContigNotFoundError if contig is not in this store."""
        if contig not in self.contig_lengths:
            raise ContigNotFoundError(contig, available=self.contigs)

    def __repr__(self) -> str:
        n_contigs = len(self.contigs)
        n_tracks = len(self.tracks())
        return (
            f"PbzStore(contigs={n_contigs}, tracks={n_tracks}, "
            f"backend={self._backend.value!r})"
        )


def create_store(
    path: str | Path | dict,
    *,
    contigs: list[str],
    contig_lengths: list[int],
    storage_options: dict | None = None,
) -> PbzStore:
    """Create a new PBZ store.

    Parameters
    ----------
    path
        Filesystem path or ``dict``/``MemoryStore`` for in-memory.
    contigs
        Contig/chromosome names.
    contig_lengths
        Length of each contig in base pairs (same order as *contigs*).
    storage_options
        Passed through to zarr for remote stores (e.g. S3).

    Returns
    -------
    PbzStore
    """
    if len(contigs) != len(contig_lengths):
        raise PbzError(
            f"contigs ({len(contigs)}) and contig_lengths ({len(contig_lengths)}) "
            "must have the same length."
        )
    if len(contigs) == 0:
        raise PbzError("At least one contig is required.")

    store = _resolve_store(path, storage_options=storage_options)
    root = zarr.open_group(store, mode="w")

    root.attrs["perbase_zarr"] = {"version": _PBZ_VERSION}

    contigs_arr = root.create_array(
        "contigs",
        shape=(len(contigs),),
        dtype=VariableLengthUTF8(),  # type: ignore[arg-type]
    )
    contigs_arr[:] = np.array(contigs, dtype=object)

    root.create_array(
        "contig_lengths",
        data=np.array(contig_lengths, dtype="int64"),
    )

    root.create_group("tracks")

    return PbzStore(root, backend=Backend.NUMPY)


def open_store(
    path: str | Path,
    *,
    mode: str = "r",
    backend: Backend | str = "numpy",
    storage_options: dict | None = None,
) -> PbzStore:
    """Open an existing PBZ store.

    Parameters
    ----------
    path
        Filesystem path or URL to the store.
    mode
        ``"r"`` (read-only) or ``"r+"`` (read-write).
    backend
        Array backend: ``"numpy"`` (default) or ``"dask"``.
    storage_options
        Passed through to zarr for remote stores.

    Returns
    -------
    PbzStore

    Raises
    ------
    PbzError
        If the store does not contain valid PBZ metadata.
    """
    backend_enum = Backend.from_value(backend)

    if mode not in ("r", "r+"):
        raise PbzError(f"Invalid mode: {mode!r}. Use 'r' or 'r+'.")

    store = _resolve_store(path, storage_options=storage_options)
    root = zarr.open_group(store, mode=mode)  # type: ignore[arg-type]

    pbz_meta = root.attrs.get("perbase_zarr")
    if pbz_meta is None:
        raise PbzError(
            "Not a PBZ store: missing 'perbase_zarr' attribute in root metadata."
        )
    if not isinstance(pbz_meta, dict):
        raise PbzError("Not a PBZ store: 'perbase_zarr' attribute must be a mapping.")
    version = pbz_meta.get("version")
    if version is None:
        raise PbzError(
            "Not a PBZ store: missing 'version' in root 'perbase_zarr' attribute."
        )

    if "contigs" not in root:
        raise PbzError("Not a PBZ store: missing 'contigs' array.")
    if "contig_lengths" not in root:
        raise PbzError("Not a PBZ store: missing 'contig_lengths' array.")

    return PbzStore(root, backend=backend_enum)


def _resolve_store(
    path: str | Path | dict,
    *,
    storage_options: dict | None = None,
) -> zarr.storage.MemoryStore | str:
    """Resolve a path/dict to a zarr store."""
    if isinstance(path, dict):
        return zarr.storage.MemoryStore()
    return str(path)


def _find_tracks(group: zarr.Group, prefix: str = "") -> list[str]:
    """Recursively find all track groups under a zarr group."""
    result: list[str] = []
    for name, obj in group.members():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(obj, zarr.Group) and "perbase_zarr_track" in obj.attrs:
            result.append(path)
        elif isinstance(obj, zarr.Group):
            result.extend(_find_tracks(obj, path))
    return result
