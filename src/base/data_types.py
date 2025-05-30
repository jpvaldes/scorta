from typing import Protocol


class TSpan(Protocol):
    """Protocol for data types representing spans."""

    @property
    def start(self) -> int:
        """The start position of the span."""
        ...

    @start.setter
    def start(self, value: int) -> None:
        """Set the start position of the span."""
        ...

    @property
    def end(self) -> int:
        """The end position of the span."""
        ...

    @end.setter
    def end(self, value: int) -> None:
        """Set the end position of the span."""
        ...


class Span:
    """A span of text.

    A span is a left-closed, right-open interval whose boundaries are integer values.

    Parameters
    ----------
    start : int
        Start position of the span (inclusive).
    end : int
        End position of the span (exclusive).
    """

    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"[{self.start},{self.end})"
