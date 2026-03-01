"""PBZ — Per-Base Zarr.

A convention and domain layer on top of Zarr v3 for per-base genomic data.
"""

from pbzarr._backends import Backend
from pbzarr.exceptions import (
    ColumnNotFoundError,
    ContigNotFoundError,
    InvalidRegionError,
    PbzError,
    TrackNotFoundError,
)
from pbzarr.region import Region, parse_region
from pbzarr.store import PbzStore, create_store, open_store
from pbzarr.track import Track

create = create_store
open = open_store

__all__ = [
    "Backend",
    "ColumnNotFoundError",
    "ContigNotFoundError",
    "InvalidRegionError",
    "PbzError",
    "PbzStore",
    "Region",
    "Track",
    "TrackNotFoundError",
    "create",
    "create_store",
    "open",
    "open_store",
    "parse_region",
]
