"""Region parsing for genomic queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pbzarr.exceptions import InvalidRegionError

_CONTIG_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_REGION_RE = re.compile(
    r"^(?P<contig>[A-Za-z0-9._-]+)"
    r"(?::(?P<start>[0-9,]+)(?:-(?P<end>[0-9,]+))?)?$"
)


@dataclass(frozen=True, slots=True)
class Region:
    """A parsed genomic region in 0-based, half-open coordinates.

    Attributes
    ----------
    contig
        Contig/chromosome name.
    start
        Start position (inclusive). None means start of contig (0).
    end
        End position (exclusive). None means end of contig (contig length).
    """

    contig: str
    start: int | None = None
    end: int | None = None


def _strip_commas(s: str) -> str:
    """Strip commas from numeric strings (e.g., '1,000,000' -> '1000000')."""
    return s.replace(",", "")


def _parse_int(s: str, label: str) -> int:
    """Parse a string as a non-negative integer, raising InvalidRegionError."""
    s = _strip_commas(s)
    try:
        val = int(s)
    except ValueError:
        raise InvalidRegionError(
            f"Invalid {label} value: {s!r}. Expected a non-negative integer."
        )
    if val < 0:
        raise InvalidRegionError(
            f"Invalid {label} value: {val}. Coordinates must be non-negative."
        )
    return val


def parse_region(
    region: str | tuple,
    *,
    one_based: bool = False,
) -> Region:
    """Parse a region string or tuple into a Region.

    Parameters
    ----------
    region
        A region in string form (``"chr1"``, ``"chr1:1000-2000"``,
        ``"chr1:5000"``) or tuple form (``("chr1",)``,
        ``("chr1", 1000, 2000)``, ``("chr1", 5000)``).
    one_based
        If True, interpret coordinates as 1-based inclusive and convert
        to 0-based half-open internally.

    Returns
    -------
    Region
        Always in 0-based, half-open coordinates.

    Raises
    ------
    InvalidRegionError
        If the region is malformed.
    """
    if isinstance(region, str):
        return _parse_string(region, one_based=one_based)
    elif isinstance(region, tuple):
        return _parse_tuple(region, one_based=one_based)
    else:
        raise InvalidRegionError(
            f"Region must be a string or tuple, got {type(region).__name__}"
        )


def _parse_string(region: str, *, one_based: bool) -> Region:
    """Parse a region string."""
    region = region.strip()
    if not region:
        raise InvalidRegionError("Region string is empty.")

    m = _REGION_RE.match(region)
    if m is None:
        raise InvalidRegionError(
            f"Malformed region string: {region!r}. "
            "Expected format: 'contig', 'contig:start-end', or 'contig:position'."
        )

    contig = m.group("contig")
    raw_start = m.group("start")
    raw_end = m.group("end")

    if raw_start is None:
        return Region(contig=contig)

    start = _parse_int(raw_start, "start")

    if raw_end is None:
        if one_based:
            start = start - 1
            end = start + 1
        else:
            end = start + 1
        _validate_start_end(start, end)
        return Region(contig=contig, start=start, end=end)

    end = _parse_int(raw_end, "end")

    if one_based:
        start = start - 1
        # end stays the same: 1-based inclusive end == 0-based exclusive end

    _validate_start_end(start, end)
    return Region(contig=contig, start=start, end=end)


def _parse_tuple(region: tuple, *, one_based: bool) -> Region:
    """Parse a region tuple."""
    if len(region) == 1:
        contig = region[0]
        if not isinstance(contig, str):
            raise InvalidRegionError(
                f"Contig must be a string, got {type(contig).__name__}"
            )
        _validate_contig_name(contig)
        return Region(contig=contig)

    elif len(region) == 2:
        contig, position = region
        if not isinstance(contig, str):
            raise InvalidRegionError(
                f"Contig must be a string, got {type(contig).__name__}"
            )
        _validate_contig_name(contig)
        if not isinstance(position, int):
            raise InvalidRegionError(
                f"Position must be an integer, got {type(position).__name__}"
            )
        if position < 0:
            raise InvalidRegionError(f"Position must be non-negative, got {position}")
        start = position
        if one_based:
            start = start - 1
        end = start + 1
        _validate_start_end(start, end)
        return Region(contig=contig, start=start, end=end)

    elif len(region) == 3:
        contig, start, end = region
        if not isinstance(contig, str):
            raise InvalidRegionError(
                f"Contig must be a string, got {type(contig).__name__}"
            )
        _validate_contig_name(contig)
        if not isinstance(start, int) or not isinstance(end, int):
            raise InvalidRegionError(
                "Start and end must be integers, got "
                f"{type(start).__name__} and {type(end).__name__}"
            )
        if start < 0:
            raise InvalidRegionError(f"Start must be non-negative, got {start}")
        if end < 0:
            raise InvalidRegionError(f"End must be non-negative, got {end}")

        if one_based:
            start = start - 1
            # end stays the same: 1-based inclusive end == 0-based exclusive end

        _validate_start_end(start, end)
        return Region(contig=contig, start=start, end=end)

    else:
        raise InvalidRegionError(
            f"Region tuple must have 1, 2, or 3 elements, got {len(region)}"
        )


def _validate_contig_name(contig: str) -> None:
    """Validate that a contig name matches the allowed pattern."""
    if not _CONTIG_RE.match(contig):
        raise InvalidRegionError(
            f"Invalid contig name: {contig!r}. "
            "Contig names may contain alphanumeric characters, "
            "underscores, hyphens, and periods."
        )


def _validate_start_end(start: int, end: int) -> None:
    """Validate that start >= 0 and start < end."""
    if start < 0:
        raise InvalidRegionError(
            f"Start ({start}) must be non-negative. "
            "This can happen when using one_based=True with position 0."
        )
    if start >= end:
        raise InvalidRegionError(f"Start ({start}) must be less than end ({end}).")
