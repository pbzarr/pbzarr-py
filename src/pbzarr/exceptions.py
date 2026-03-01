"""PBZ exception hierarchy."""


class PbzError(Exception):
    """Base exception for all PBZ errors."""


class ContigNotFoundError(PbzError):
    """Raised when a contig name is not present in the store."""

    def __init__(self, contig: str, available: list[str] | None = None) -> None:
        self.contig = contig
        self.available = available
        msg = f"Contig not found: {contig!r}"
        if available is not None:
            msg += f". Available contigs: {available}"
        super().__init__(msg)


class TrackNotFoundError(PbzError):
    """Raised when a track name is not present in the store."""

    def __init__(self, track: str, available: list[str] | None = None) -> None:
        self.track = track
        self.available = available
        msg = f"Track not found: {track!r}"
        if available is not None:
            msg += f". Available tracks: {available}"
        super().__init__(msg)


class InvalidRegionError(PbzError):
    """Raised when a region string or tuple is malformed or out of bounds."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ColumnNotFoundError(PbzError):
    """Raised when a column name is not present in a track."""

    def __init__(self, column: str, available: list[str] | None = None) -> None:
        self.column = column
        self.available = available
        msg = f"Column not found: {column!r}"
        if available is not None:
            msg += f". Available columns: {available}"
        super().__init__(msg)
