from base.data_types import Span


def test_data_types_span_creation() -> None:
    """A span object must have the expected attributes."""
    span = Span(0, 1)

    assert 0 == span.start
    assert 1 == span.end
