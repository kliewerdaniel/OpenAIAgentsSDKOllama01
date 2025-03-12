from __future__ import annotations

import threading
from typing import Any, Literal

from agents.tracing import Span, Trace, TracingProcessor

TestSpanProcessorEvent = Literal["trace_start", "trace_end", "span_start", "span_end"]


class SpanProcessorForTests(TracingProcessor):
    """
    A simple processor that stores finished spans in memory.
    This is thread-safe and suitable for tests or basic usage.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Dictionary of trace_id -> list of spans
        self._spans: list[Span[Any]] = []
        self._traces: list[Trace] = []
        self._events: list[TestSpanProcessorEvent] = []

    def on_trace_start(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)
            self._events.append("trace_start")

    def on_trace_end(self, trace: Trace) -> None:
        with self._lock:
            # We don't append the trace here, we want to do that in on_trace_start
            self._events.append("trace_end")

    def on_span_start(self, span: Span[Any]) -> None:
        with self._lock:
            # Purposely not appending the span here, we want to do that in on_span_end
            self._events.append("span_start")

    def on_span_end(self, span: Span[Any]) -> None:
        with self._lock:
            self._events.append("span_end")
            self._spans.append(span)

    def get_ordered_spans(self, including_empty: bool = False) -> list[Span[Any]]:
        with self._lock:
            spans = [x for x in self._spans if including_empty or x.export()]
            return sorted(spans, key=lambda x: x.started_at or 0)

    def get_traces(self, including_empty: bool = False) -> list[Trace]:
        with self._lock:
            traces = [x for x in self._traces if including_empty or x.export()]
            return traces

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._traces.clear()
            self._events.clear()

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


SPAN_PROCESSOR_TESTING = SpanProcessorForTests()


def fetch_ordered_spans() -> list[Span[Any]]:
    return SPAN_PROCESSOR_TESTING.get_ordered_spans()


def fetch_traces() -> list[Trace]:
    return SPAN_PROCESSOR_TESTING.get_traces()


def fetch_events() -> list[TestSpanProcessorEvent]:
    return SPAN_PROCESSOR_TESTING._events
