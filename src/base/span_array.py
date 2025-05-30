from typing import Any, List, Sequence, Union

import numpy as np
import numpy.typing as npt

from base.data_types import Span, TSpan


class SpanArray(Sequence[TSpan]):
    """A sequence of spans.

    A **span** is a pair of integers representing a left-closed right-open interval [start, end).

    This class conforms to ``Sequence`` and allows slicing.

    Parameters
    ----------
    start : int, list of int, np.ndarray of int
        The start positions of the spans.
    end : int, list of int, np.ndarray of int
        The end positions of the spans.
    check_ranges : bool, default is True
        Whether to check that all spans are valid (``start <= end``). You can disable this check if you are sure that
        all spans are valid. That will make the creation of the object faster.
    allow_empty_spans : bool, default is True
        Whether to allow empty spans (``start == end``). This parameter is ignored if ``check_ranges`` is
        ``False``.
    """

    def __init__(
        self,
        start: Union[int, List[int], npt.NDArray[np.int64]],
        end: Union[int, List[int], npt.NDArray[np.int64]],
        check_ranges: bool = True,
        allow_empty_spans: bool = True,
    ) -> None:
        if type(start) is not type(end):
            raise ValueError("The start and end positions must have the same type.")

        if isinstance(start, np.ndarray) and isinstance(end, np.ndarray):
            if start.shape == end.shape and start.ndim == 1 and end.ndim == 1:
                self.start = start
                self.end = end
            else:
                raise ValueError("Start and end arrays must have the same one-dimensional shape.")
        elif isinstance(start, list) and isinstance(end, list):
            if len(start) == len(end):
                self.start = np.array(start)
                self.end = np.array(end)
            else:
                raise ValueError("The number of start and end positions must be equal.")
        else:
            self.start = np.array([start])
            self.end = np.array([end])

        if check_ranges:
            self._check_ranges(allow_empty_spans)

        self.sorter = np.lexsort((self.end, self.start))  # e.g., (1,10) < (3,5) < (3,7) < (4,5)
        self.end_sorter = np.argsort(self.end)  # Sorts intervals by end position
        self.max_endpoint = np.zeros(len(self), dtype=int)
        self._augment(self.max_endpoint, 0, len(self))

    def _check_ranges(self, allow_empty_spans: bool) -> None:
        if allow_empty_spans:
            if np.any(self.start > self.end):
                invalid_spans = self.where(self.start > self.end)
                raise ValueError(
                    f"Not all spans are valid. Spans with start > end are not allowed. Invalid spans: {invalid_spans}."
                )
        elif np.any(self.start >= self.end):
            invalid_spans = self.where(self.start >= self.end)
            raise ValueError(
                f"Not all spans are valid. Spans with start >= end are not allowed. Invalid spans: {invalid_spans}."
            )

    def _augment(self, max_endpoint: np.array, low: int, high: int) -> int:
        if low >= high:
            return -1

        mid = (low + high) // 2
        max_left = self._augment(max_endpoint, low, mid)
        max_right = self._augment(max_endpoint, mid + 1, high)
        max_endpoint[mid] = max(max_left, max_right, self.end[self.sorter[mid]])

        return max_endpoint[mid]

    @classmethod
    def from_spans(
        cls,
        spans: Sequence[TSpan],
        check_ranges: bool = True,
        allow_empty_spans: bool = True,
    ) -> "SpanArray":
        if isinstance(spans, SpanArray):
            return spans # TODO: copy?

        start = np.array([span.start for span in spans])
        end = np.array([span.end for span in spans])

        return cls(start, end, check_ranges=check_ranges, allow_empty_spans=allow_empty_spans)

    def __getitem__(self, value: Union[int, slice]) -> Any:
        if isinstance(value, slice):
            return SpanArray(self.start[value], self.end[value], check_ranges=False)

        return Span(self.start[value], self.end[value])

    def __len__(self) -> int:
        return len(self.start)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SpanArray):
            return np.array_equal(
                self.start[self.sorter], other.start[other.sorter]
            ) and np.array_equal(self.end[self.sorter], other.end[other.sorter])

        # TODO: allow comparisons with other types of Sequence[TSpan] objects

        return False

    def __iter__(self) -> Any:
        """Iterate over the spans in the array.

        Yields
        ------
        Span
            The next span in the array. The spans are yielded in the order of their start positions.
        """
        for i in self.sorter:
            yield Span(self.start[i], self.end[i])

    def where(self, condition: npt.NDArray[np.bool_]) -> "SpanArray":
        """Filter the spans in the array based on a boolean mask.

        For example, to filter only spans starting at an even position:

        >>> spans = SpanArray([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
        >>> spans.where(spans.start % 2 == 0)

        Parameters
        ----------
        condition : np.ndarray
            A boolean mask indicating which spans to keep.

        Returns
        -------
        SpanArray
            A new span array containing only the spans where the mask is True.
        """
        return SpanArray(self.start[condition], self.end[condition], check_ranges=False)

    def precedes_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array precede the given span.

        A span ``s1`` **precedes** another span ``s2`` if ``s1.end <= s2.start``. Visually:

        .. code-block:: none

            s1.start     s1.end
            |------------)
                                  |------------)
                                  s2.start     s2.end

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array precede the given span.
        """
        return self.end <= other.start

    def touches_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array touch the given span.

        A span ``s1`` **touches** another span ``s2`` if ``s1.end == s2.start``. Visually:

        .. code-block:: none

            s1.start     s1.end
            |------------)
                         |------------)
                         s2.start     s2.end

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array touch the given span.
        """
        return self.end == other.start

    def left_overlaps_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array overlap with the given span.

        A span ``s1`` **(left-)overlaps** another span ``s2`` if ``s1.start < s2.start < s1.end``. Visually:

        .. code-block:: none

            s1.start     s1.end
            |------------)
                     |------------)
                     s2.start     s2.end

        Note that the relationship is *not* symmetric: if ``s1`` (left-)overlaps with ``s2``, then ``s2`` does not
        (left-)overlap with ``s1``.

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array overlap with the given span.
        """
        return np.logical_and(self.start < other.start, other.start < self.end)

    def starts_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array start at the given span.

        A span ``s1`` **starts** at another span ``s2`` if ``s1.start == s2.start`` and ``s1.end < s2.end``. Visually:

        .. code-block:: none

            s1.start     s1.end
            |------------)
            |--------------------)
            s2.start             s2.end

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array start at the given span.
        """
        return np.logical_and(self.start == other.start, self.end < other.end)

    def contains_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array contain the given span.

        A span ``s1`` **contains** another span ``s2`` if ``s1.start <= s2.start`` and ``s2.end <= s1.end``. Visually:

        .. code-block:: none

            s1.start              s1.end
            |---------------------)
                |----------)
                s2.start   s2.end

        Note that if ``s1`` and ``s2`` are equal, then ``s1`` contains ``s2``.

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array contain the given span.
        """
        return np.logical_and(self.start <= other.start, other.end <= self.end)

    def is_contained_in_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array are contained in the given span.

        A span ``s1`` is **contained in** another span ``s2`` if ``s2.start <= s1.start`` and ``s1.end <= s2.end``.
        Visually:

        .. code-block:: none

                s1.start   s1.end
                |----------)
            s2.start              s2.end
            |---------------------)


        Note that if ``s1`` and ``s2`` are equal, then ``s1`` is contained in ``s2``.

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array are contained in the given span.
        """
        return np.logical_and(other.start <= self.start, self.end <= other.end)

    def ends_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array finish at the given span.

        A span ``s1`` **ends** at another span ``s2`` if ``s1.start < s2.start`` and ``s1.end == s2.end``. Visually:

        .. code-block:: none

                     s1.start     s1.end
                     |------------)
            |---------------------)
            s2.start              s2.end

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array finish at the given span.
        """
        return np.logical_and(self.start < other.start, self.end == other.end)

    def is_disjoint_from_mask(self, other: TSpan) -> npt.NDArray[np.bool_]:
        """Check which spans in the array are disjoint from the given span.

        A span ``s1`` is **disjoint** from another span ``s2`` if ``s1.end <= s2.start`` or ``s2.end <= s1.start``. In
        other words, the two spans do not overlap. Visually:

        .. code-block:: none

            s1.start     s1.end
            |------------)
                                  |------------)
                                  s2.start     s2.end

        Parameters
        ----------
        other : Span
            The span to check against.

        Returns
        -------
        numpy.ndarray
            A boolean mask indicating which spans in the array are disjoint from the given span.
        """
        return np.logical_or(self.end <= other.start, other.end <= self.start)

    def intersects_mask(self, span: TSpan) -> npt.NDArray[np.bool_]:
        """Checks which spans in the array intersect the given span.

        Two spans intersect if they are not disjoint.

        Parameters
        ----------
        span : Span
            The span to intersect with.

        Returns
        -------
        np.ndarray
            A boolean mask indicating which spans in the array intersect the given span.
        """
        return np.logical_and(self.start < span.end, span.start < self.end)

    def intersects(self, span: TSpan) -> "SpanArray":
        """Find all spans in the array that intersect the given span."""
        mask = self.intersects(span)

        return SpanArray(self.start[mask], self.end[mask])

    def contains(self, span: Span) -> "SpanArray":
        """Get all intervals in the array that contain the given span(s)."""
        start_index = np.searchsorted(self.start, span.start, side="right", sorter=self.startsorter)
        end_index = np.searchsorted(self.end, span.end, side="left", sorter=self.end_sorter)

        span_indexes = np.intersect1d(
            self.startsorter[:start_index], self.end_sorter[end_index:], assume_unique=False
        )

        return SpanArray(self.start[span_indexes], self.end[span_indexes])

    def is_contained_in(self, span: TSpan) -> "SpanArray":
        """Get all intervals in the array that are contained in the given span."""
        start_index = np.searchsorted(self.start, span.start, side="left", sorter=self.startsorter)

        if start_index == len(self):
            return SpanArray(np.array([], dtype=np.int64), np.array([], dtype=np.int64))

        end_index = np.searchsorted(self.end, span.end, side="right", sorter=self.end_sorter)

        span_indexes = np.intersect1d(
            self.startsorter[start_index:], self.end_sorter[:end_index], assume_unique=False
        )

        return SpanArray(self.start[span_indexes], self.end[span_indexes])

    def __str__(self) -> str:
        return f"{', '.join([str(span) for span in self])}"

    def __repr__(self) -> str:
        return f"SpanArray({self})"

    def shift(self, by: int) -> None:
        """Shift the intervals by the given amount.

        Parameters
        ----------
        by : int
            The amount to shift the intervals by. The amount may be negative.
        """
        self.start += by
        self.end += by

    @classmethod
    def random(
        cls,
        n: int = 1,
        min_value: int = 0,
        max_value: int = 100,
        min_width: int = 1,
        max_width: int | None = None,
    ) -> "SpanArray":
        """Generate random spans.

        Parameters
        ----------
        n : int, default is 1
            The number of spans to generate.
        min_value : int, default is 0
            The minimum start position of the spans.
        max_value : int, default is 100
            The maximum end position of the spans.
        min_width : int, default is 1
            The minimum width of the spans. If set to 0, the spans can be empty. If set to a negative value, spans that
            are not proper intervals can be generated.
        max_width : int, optional, default is None
            The maximum width of the spans. If not provided, the width is randomly generated.

        Returns
        -------
        SpanArray
            A sequence of random spans.
        """
        if n < 1:
            raise ValueError("The number of spans must be greater than zero.")

        start = np.random.randint(min_value, max_value - min_width, n)
        end = start + np.random.randint(min_width, max_width or max_value - start, n)

        return cls(start, end, check_ranges=False)
