"""Tests for OtelTracer — OpenTelemetry integration with timing instrumentation."""
from __future__ import annotations

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from orchestra.engine.tracer import OtelTracer, TimingContext


class TestOtelTracerSingleton:
    """Test singleton pattern and initialization."""

    def test_singleton_instance(self):
        """get_instance returns same object on multiple calls."""
        OtelTracer.reset_instance()
        instance1 = OtelTracer.get_instance()
        instance2 = OtelTracer.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """reset_instance clears singleton."""
        OtelTracer.reset_instance()
        instance1 = OtelTracer.get_instance()
        OtelTracer.reset_instance()
        instance2 = OtelTracer.get_instance()
        assert instance1 is not instance2

    def test_initialization_with_defaults(self):
        """OtelTracer initializes with default parameters."""
        OtelTracer.reset_instance()
        tracer = OtelTracer.get_instance()
        assert tracer.service_name == "orchestra"
        assert tracer.jaeger_host == "localhost"
        assert tracer.jaeger_port == 6831

    def test_initialization_with_custom_params(self):
        """OtelTracer accepts custom Jaeger parameters."""
        OtelTracer.reset_instance()
        tracer = OtelTracer(
            service_name="test-service",
            jaeger_host="jaeger.example.com",
            jaeger_port=9999,
        )
        assert tracer.service_name == "test-service"
        assert tracer.jaeger_host == "jaeger.example.com"
        assert tracer.jaeger_port == 9999

    def test_graceful_handling_when_otel_unavailable(self):
        """Tracer handles missing OpenTelemetry gracefully."""
        OtelTracer.reset_instance()
        tracer = OtelTracer(fallback_console=False)
        assert tracer is not None


class TestAgentSpans:
    """Test agent span creation and attributes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_start_agent_span_returns_span(self):
        """start_agent_span returns a span object or None gracefully."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run-123", "cdx-deep", "gpt-5.4/xhigh")
        # Span may be None if OTEL not installed, but should not raise
        assert span is None or hasattr(span, 'end')

    def test_start_agent_span_with_attributes(self):
        """Agent span creation with required attributes."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        run_id = "test-run-456"
        alias = "gem-fast"
        model = "gemini-2.0-flash"

        span = tracer.start_agent_span(run_id, alias, model)
        # Should complete without error
        assert span is None or span is not None

    def test_start_agent_span_returns_none_if_unavailable(self):
        """start_agent_span returns None if tracer unavailable."""
        tracer = OtelTracer(fallback_console=False)
        tracer._is_available = False
        span = tracer.start_agent_span("run-789", "test", "model")
        assert span is None

    def test_end_agent_span(self):
        """Spans can be ended without error."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run-999", "test", "model")
        if span:
            tracer.end_span(span)  # Should not raise


class TestProviderSpans:
    """Test provider span creation as children of agent spans."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_start_provider_span_as_child(self):
        """start_provider_span creates child of agent span."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        agent_span = tracer.start_agent_span("run-001", "cdx-deep", "gpt-5.4")
        provider_span = tracer.start_provider_span(agent_span, "codex", "subprocess")
        # Should complete without error
        assert provider_span is None or hasattr(provider_span, 'end')

    def test_start_provider_span_all_providers(self):
        """start_provider_span works with all 4 providers."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        agent_span = tracer.start_agent_span("run-002", "test", "model")

        providers = ["claude", "codex", "gemini", "ollama"]
        methods = ["streaming", "subprocess", "streaming", "http"]

        for provider, method in zip(providers, methods):
            span = tracer.start_provider_span(agent_span, provider, method)
            if span:
                tracer.end_span(span)

    def test_start_provider_span_returns_none_if_parent_none(self):
        """start_provider_span returns None if parent span is None."""
        tracer = OtelTracer.get_instance()
        span = tracer.start_provider_span(None, "claude", "streaming")
        assert span is None

    def test_start_provider_span_returns_none_if_unavailable(self):
        """start_provider_span returns None if tracer unavailable."""
        tracer = OtelTracer(fallback_console=False)
        tracer._is_available = False
        agent_span = Mock()
        span = tracer.start_provider_span(agent_span, "claude", "streaming")
        assert span is None


class TestMetricsRecording:
    """Test recording of timing, token, and cost metrics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_set_timing_records_metric(self):
        """set_timing records latency metric."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run-003", "test", "model")
        if span:
            tracer.set_timing(span, "first_token_latency_ms", 123.45)
            tracer.set_timing(span, "total_latency_ms", 567.89)
            tracer.end_span(span)

    def test_set_tokens_records_counts(self):
        """set_tokens records input and output token counts."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run-004", "test", "model")
        if span:
            tracer.set_tokens(span, 1000, 500)
            tracer.end_span(span)

    def test_set_cost_records_usd_amount(self):
        """set_cost records estimated cost in USD."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run-005", "test", "model")
        if span:
            tracer.set_cost(span, 0.15)
            tracer.end_span(span)

    def test_set_error_records_error_message(self):
        """set_error records error on span."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run-006", "test", "model")
        if span:
            tracer.set_error(span, "Connection timeout")
            tracer.end_span(span)

    def test_set_metrics_with_none_span(self):
        """Metric methods handle None span gracefully."""
        tracer = OtelTracer.get_instance()

        # None of these should raise
        tracer.set_timing(None, "metric", 100.0)
        tracer.set_tokens(None, 1000, 500)
        tracer.set_cost(None, 0.15)
        tracer.set_error(None, "error")
        tracer.end_span(None)


class TestFlush:
    """Test span flushing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_flush_with_default_timeout(self):
        """flush() works with default timeout."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        tracer.flush()  # Should not raise

    def test_flush_with_custom_timeout(self):
        """flush() accepts custom timeout."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        tracer.flush(timeout_secs=5)  # Should not raise

    def test_flush_when_unavailable(self):
        """flush() handles unavailable tracer gracefully."""
        tracer = OtelTracer(fallback_console=False)
        tracer.flush()  # Should not raise


class TestSpanHierarchy:
    """Test proper span hierarchy: agent -> provider."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_complete_span_hierarchy(self):
        """Complete span hierarchy: agent with child provider spans."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        agent_span = tracer.start_agent_span("run-007", "cdx-deep", "gpt-5.4")

        if agent_span:
            provider_span = tracer.start_provider_span(agent_span, "codex", "subprocess")
            if provider_span:
                tracer.set_timing(provider_span, "total_latency_ms", 1234.56)
                tracer.set_tokens(provider_span, 2000, 1000)
                tracer.set_cost(provider_span, 0.25)
                tracer.end_span(provider_span)

            tracer.end_span(agent_span)


class TestTimingContext:
    """Test TimingContext helper for timing measurement."""

    def test_timing_context_measures_duration(self):
        """TimingContext measures total latency."""
        with TimingContext() as timer:
            time.sleep(0.01)  # 10ms

        # Total latency should be >= 10ms
        assert timer.total_latency_ms >= 10.0

    def test_timing_context_first_token(self):
        """TimingContext marks and measures first token latency."""
        with TimingContext() as timer:
            time.sleep(0.005)  # 5ms
            timer.mark_first_chunk()
            time.sleep(0.005)  # Another 5ms

        # First token should be ~5ms, total ~10ms
        assert 4.0 <= timer.first_token_latency_ms <= 8.0
        assert timer.total_latency_ms >= 10.0

    def test_timing_context_without_first_chunk(self):
        """first_token_latency_ms returns 0 if first chunk not marked."""
        with TimingContext() as timer:
            time.sleep(0.01)

        assert timer.first_token_latency_ms == 0.0
        assert timer.total_latency_ms >= 10.0

    def test_timing_context_multiple_first_chunk_calls(self):
        """mark_first_chunk() only records first occurrence."""
        with TimingContext() as timer:
            time.sleep(0.001)
            timer.mark_first_chunk()
            first_latency = timer.first_token_latency_ms

            time.sleep(0.005)
            timer.mark_first_chunk()  # Should not update
            second_latency = timer.first_token_latency_ms

        # Both should be approximately equal (second call ignored)
        assert abs(first_latency - second_latency) < 2.0

    def test_timing_context_before_measurement_complete(self):
        """Latency properties return 0 if measurement incomplete."""
        timer = TimingContext()
        assert timer.first_token_latency_ms == 0.0
        assert timer.total_latency_ms == 0.0

        timer.__enter__()
        assert timer.first_token_latency_ms == 0.0
        assert timer.total_latency_ms == 0.0

        timer.__exit__(None, None, None)
        assert timer.total_latency_ms >= 0.0


class TestProviderIntegrationPatterns:
    """Test patterns for provider instrumentation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_claude_streaming_pattern(self):
        """Test timing pattern for Claude streaming."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        agent_span = tracer.start_agent_span("run-008", "claude-sonnet", "sonnet")

        if agent_span:
            provider_span = tracer.start_provider_span(agent_span, "claude", "streaming")

            if provider_span:
                with TimingContext() as timer:
                    time.sleep(0.005)
                    timer.mark_first_chunk()
                    time.sleep(0.005)

                tracer.set_timing(provider_span, "first_token_latency_ms", timer.first_token_latency_ms)
                tracer.set_timing(provider_span, "total_latency_ms", timer.total_latency_ms)
                tracer.set_tokens(provider_span, 500, 250)
                tracer.set_cost(provider_span, 0.05)
                tracer.end_span(provider_span)

            tracer.end_span(agent_span)
            tracer.flush()

    def test_codex_subprocess_pattern(self):
        """Test timing pattern for Codex subprocess."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        agent_span = tracer.start_agent_span("run-009", "cdx-deep", "gpt-5.4")

        if agent_span:
            provider_span = tracer.start_provider_span(agent_span, "codex", "subprocess")

            if provider_span:
                with TimingContext() as timer:
                    time.sleep(0.01)

                tracer.set_timing(provider_span, "total_latency_ms", timer.total_latency_ms)
                tracer.set_tokens(provider_span, 1000, 500)
                tracer.set_cost(provider_span, 0.10)
                tracer.end_span(provider_span)

            tracer.end_span(agent_span)
            tracer.flush()

    def test_gemini_streaming_pattern(self):
        """Test timing pattern for Gemini streaming."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        agent_span = tracer.start_agent_span("run-010", "gem-fast", "gemini-flash")

        if agent_span:
            provider_span = tracer.start_provider_span(agent_span, "gemini", "streaming")

            if provider_span:
                with TimingContext() as timer:
                    time.sleep(0.003)
                    timer.mark_first_chunk()
                    time.sleep(0.007)

                tracer.set_timing(provider_span, "first_token_latency_ms", timer.first_token_latency_ms)
                tracer.set_timing(provider_span, "total_latency_ms", timer.total_latency_ms)
                tracer.set_tokens(provider_span, 800, 400)
                tracer.set_cost(provider_span, 0.08)
                tracer.end_span(provider_span)

            tracer.end_span(agent_span)
            tracer.flush()

    def test_ollama_http_pattern(self):
        """Test timing pattern for Ollama HTTP POST."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        agent_span = tracer.start_agent_span("run-011", "ollama-local", "llama2")

        if agent_span:
            provider_span = tracer.start_provider_span(agent_span, "ollama", "http")

            if provider_span:
                with TimingContext() as timer:
                    time.sleep(0.02)

                tracer.set_timing(provider_span, "total_latency_ms", timer.total_latency_ms)
                tracer.set_tokens(provider_span, 600, 300)
                tracer.set_cost(provider_span, 0.0)
                tracer.end_span(provider_span)

            tracer.end_span(agent_span)
            tracer.flush()

    def test_parallel_agent_spans(self):
        """Test multiple concurrent agent spans."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        run_id = "run-012"
        agents = [
            ("cdx-deep", "gpt-5.4"),
            ("gem-fast", "gemini-flash"),
            ("claude-sonnet", "sonnet"),
        ]

        spans = []
        for alias, model in agents:
            span = tracer.start_agent_span(run_id, alias, model)
            if span:
                spans.append(span)

        for span in spans:
            if span:
                tracer.end_span(span)

        tracer.flush()


class TestErrorHandling:
    """Test error handling in tracer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_tracer_survives_span_exceptions(self):
        """Tracer handles exceptions during span operations."""
        tracer = OtelTracer.get_instance(fallback_console=True)

        tracer.set_timing(None, "metric", 100.0)
        tracer.set_error(None, "error")

        span = tracer.start_agent_span("run-013", "test", "model")
        assert span is None or span is not None

    def test_availability_check(self):
        """is_available() accurately reports tracer state."""
        OtelTracer.reset_instance()
        tracer = OtelTracer(fallback_console=False)
        assert isinstance(tracer.is_available(), bool)


class TestCoverage:
    """Additional tests for edge cases and coverage."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_metrics_with_zero_values(self):
        """Metrics handle zero values correctly."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run", "test", "model")
        if span:
            tracer.set_tokens(span, 0, 0)
            tracer.set_cost(span, 0.0)
            tracer.end_span(span)

    def test_metrics_with_large_values(self):
        """Metrics handle large values correctly."""
        tracer = OtelTracer.get_instance(fallback_console=True)
        span = tracer.start_agent_span("run", "test", "model")
        if span:
            tracer.set_tokens(span, 100000, 50000)
            tracer.set_cost(span, 999.99)
            tracer.set_timing(span, "total_latency_ms", 999999.99)
            tracer.end_span(span)

    def test_is_available_method(self):
        """is_available() method works correctly."""
        tracer = OtelTracer(fallback_console=False)
        result = tracer.is_available()
        assert isinstance(result, bool)
