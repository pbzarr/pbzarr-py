"""Backend enum and dispatch registry for PBZ data retrieval."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zarr


class Backend(Enum):
    """Array backend for data retrieval."""

    NUMPY = "numpy"
    DASK = "dask"

    @classmethod
    def from_value(cls, value: Backend | str) -> Backend:
        """Convert a string or Backend to a Backend enum member.

        Raises ValueError if the string is not a valid backend.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            valid = [b.value for b in cls]
            raise ValueError(
                f"Unknown backend: {value!r}. Valid backends: {valid}"
            ) from None


def _import_dask():  # type: ignore[no-untyped-def]
    """Lazy import dask.array with a clear error message."""
    try:
        import dask.array as da  # type: ignore[no-untyped-def]

        return da
    except ImportError:
        raise ImportError(
            "dask is required for the 'dask' backend. "
            "Install with: pip install pbzarr[dask]"
        ) from None


def _get_numpy(zarr_array: zarr.Array, slices: tuple) -> object:
    """Retrieve data using the numpy (eager) backend."""
    return zarr_array[slices]


def _get_dask(zarr_array: zarr.Array, slices: tuple) -> object:
    """Retrieve data using the dask (lazy) backend."""
    da = _import_dask()
    return da.from_zarr(zarr_array)[slices]


_DISPATCH: dict = {
    Backend.NUMPY: _get_numpy,
    Backend.DASK: _get_dask,
}


def get_data(
    zarr_array: zarr.Array,
    slices: tuple,
    backend: Backend,
) -> object:
    """Dispatch data retrieval to the appropriate backend.

    Parameters
    ----------
    zarr_array
        The raw zarr.Array to read from.
    slices
        Tuple of slices/indices to apply.
    backend
        Which backend to use for retrieval.

    Returns
    -------
    object
        An array-like object (numpy.ndarray, dask.array.Array, etc.).
    """
    fn = _DISPATCH.get(backend)
    if fn is None:
        raise ValueError(
            f"No dispatch function registered for backend: {backend!r}"
        )
    return fn(zarr_array, slices)
