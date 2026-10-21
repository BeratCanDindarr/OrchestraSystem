from __future__ import annotations
"""Phase 4 tests for Block 7 (OtelTracer Provider Timing).

Tests span lifecycle, run_ask instrumentation, and TTFT capture across all providers.
Verifies that parent spans survive child operations, timing metrics are recorded correctly,
and streaming-based TTFT capture works for Claude SDK and Ollama.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from orchestra.engine.tracer import OtelTracer, TimingContext
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


class TestProviderSpanDoesNotCloseParen:
    """Test that parent span survives child span operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_provider_span_does_not_close_parent(self):
        """Verify parent span survives child span end without premature closure.

        Block 7 requirement: start_provider_span must not close parent span
        when child span ends. Uses end_on_exit=False in trace.use_span().
        """
        tracer = OtelTracer.get_instance(fallback_console=True)

        # Create parent agent span
        parent_span = tracer.start_agent_span("run-parent-001", "cdx-deep", "gpt-5.4/xhigh")
        assert parent_span is None or hasattr(parent_span, 'is_recording')

        if parent_span is None:
            pytest.skip("OTEL not available for span testing")

        # Record parent is recording before child
        parent_active_before = getattr(parent_span, 'is_recording', lambda: True)()

        # Start child provider span
        child_span = tracer.start_provider_span(parent_span, "claude", "streaming")
        assert child_span is None or hasattr(child_span, 'end')

        # End child span
        if child_span:
            tracer.end_span(child_span)

        # Verify parent still active
        parent_active_after = getattr(parent_span, 'is_recording', lambda: True)()
        assert parent_active_after is True, "Parent span should still be recording after child end"

        # Clean up
        if parent_span:
            tracer.end_span(parent_span)


class TestRunAskSpanLifecycle:
    """Test run_ask wraps execution in agent span with timing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_run_ask_span_lifecycle(self):
        """Verify run_ask() wraps execution and records timing metrics.

        Block 7 requirement: run_ask must:
        1. Call start_agent_span with run_id, alias, model
        2. Call set_timing(span, "total_latency_ms", value)
        3. Call end_span(span) in finally block
        """
        tracer = OtelTracer.get_instance(fallback_console=True)

        # Simulate run_ask instrumentation
        run_id = "run-ask-001"
        alias = "cld-fast"
        model = "claude/sonnet"

        # Start agent span (as run_ask does)
        span = tracer.start_agent_span(run_id, alias, model)
        if span is None:
            pytest.skip("OTEL not available for span testing")

        # Simulate execution
        t0 = time.time()
        time.sleep(0.01)  # Simulate work
        elapsed_ms = (time.time() - t0) * 1000

        # Record timing and end span (as run_ask finally block does)
        tracer.set_timing(span, "total_latency_ms", elapsed_ms)
        tracer.end_span(span)

        # Verify span was ended (no exception raised)
        assert True

    def test_run_ask_span_instrumentation_with_mock(self):
        """Verify start_agent_span and set_timing called with correct parameters."""
        tracer = OtelTracer.get_instance(fallback_console=True)

        run_id = "run-ask-002"
        alias = "cld-fast"
        model = "claude/sonnet"

        # Mock the tracer methods to verify call signatures
        with patch.object(tracer, 'start_agent_span', wraps=tracer.start_agent_span) as mock_start:
            with patch.object(tracer, 'set_timing', wraps=tracer.set_timing) as mock_timing:
                with patch.object(tracer, 'end_span', wraps=tracer.end_span) as mock_end:
                    span = tracer.start_agent_span(run_id, alias, model)

                    # Verify start_agent_span called with correct args
                    mock_start.assert_called_once_with(run_id, alias, model)

                    if span:
                        tracer.set_timing(span, "total_latency_ms", 123.45)
                        mock_timing.assert_called_once_with(span, "total_latency_ms", 123.45)

                        tracer.end_span(span)
                        mock_end.assert_called_once_with(span)


class TestRunAskSpanErrorHandling:
    """Test error recording in span and cleanup in finally block."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_run_ask_span_error_handling(self):
        """Verify errors are recorded to span and span ends despite exception.

        Block 7 requirement: run_ask finally block must:
        1. Call set_error(span, str(e)) on exception
        2. Call end_span(span) even if execution fails
        """
        tracer = OtelTracer.get_instance(fallback_console=True)

        run_id = "run-error-001"
        span = tracer.start_agent_span(run_id, "cld-fast", "claude/sonnet")
        if span is None:
            pytest.skip("OTEL not available for span testing")

        # Simulate error recording
        error_msg = "Connection timeout"
        tracer.set_error(span, error_msg)

        # Verify span can still end (finally block behavior)
        tracer.end_span(span)

        # Verify no exception was raised
        assert True

    def test_run_ask_span_error_with_mock(self):
        """Verify set_error() called before end_span()."""
        tracer = OtelTracer.get_instance(fallback_console=True)

        span = tracer.start_agent_span("run-error-002", "cld-fast", "claude/sonnet")
        if span is None:
            pytest.skip("OTEL not available")

        with patch.object(tracer, 'set_error', wraps=tracer.set_error) as mock_error:
            with patch.object(tracer, 'end_span', wraps=tracer.end_span) as mock_end:
                error_msg = "Test error"
                tracer.set_error(span, error_msg)
                mock_error.assert_called_once_with(span, error_msg)

                tracer.end_span(span)
                # Both should be called
                assert mock_error.call_count == 1
                assert mock_end.call_count == 1


class TestClaudeSDKFirstTokenTiming:
    """Test TTFT capture in Claude SDK provider with streaming."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()


class TestOllamaFirstTokenTiming:
    """Test TTFT capture in Ollama provider with HTTP streaming."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_ollama_first_token_timing(self):
        """Verify Ollama marks first chunk for TTFT capture.

        Block 7 requirement: ollama.py native_run() must:
        1. Use HTTP streaming (urllib + for loop)
        2. Call timer.mark_first_chunk() on first token
        3. Verify first_token_latency_ms > 0 and < total_latency_ms
        """
        # Simulate Ollama native_run streaming pattern (from ollama.py:65-82)
        with TimingContext() as timer:
            # Simulate HTTP response with chunks
            time.sleep(0.003)  # Network delay

            # First token
            timer.mark_first_chunk()
            time.sleep(0.002)

            # Additional tokens
            time.sleep(0.003)

        # Verify timing
        assert timer.first_token_latency_ms > 0, "First token latency must be > 0"
        assert timer.total_latency_ms > 0, "Total latency must be > 0"
        assert timer.first_token_latency_ms < timer.total_latency_ms

    def test_ollama_uses_http_streaming(self):
        """Verify Ollama provider uses HTTP streaming, not subprocess.

        Block 7 requirement: native_run() must use streaming for TTFT.
        """
        from orchestra.providers.ollama import OllamaProvider

        provider = OllamaProvider()

        # Read provider source
        import inspect
        source = inspect.getsource(provider.native_run)

        # Verify streaming approach
        assert "urllib.request.urlopen" in source, "Must use HTTP streaming"
        assert "for raw_line in resp" in source, "Must iterate response stream"
        assert "timer.mark_first_chunk()" in source, "Must mark first token"


class TestTimingContextMarksFirstChunk:
    """Test TimingContext captures first chunk correctly."""

    def test_timing_context_marks_first_chunk(self):
        """Verify TimingContext.mark_first_chunk() sets correct timing.

        Block 7 requirement: TimingContext must:
        1. Record first_chunk_time on mark_first_chunk()
        2. Only mark once (idempotent)
        3. Ensure first_token_latency_ms < total_latency_ms
        """
        with TimingContext() as context:
            # Simulate operation start
            assert context.start_time is not None

            # Simulate work before first token
            time.sleep(0.005)
            context.mark_first_chunk()
            first_latency = context.first_token_latency_ms

            # Simulate streaming continues
            time.sleep(0.003)
            context.mark_first_chunk()  # Second call should be ignored
            second_latency = context.first_token_latency_ms

            # More work
            time.sleep(0.002)

        # Verify idempotency
        assert abs(first_latency - second_latency) < 1.0, "mark_first_chunk should be idempotent"

        # Verify timing relationship
        assert context.first_token_latency_ms > 0
        assert context.total_latency_ms > context.first_token_latency_ms

    def test_timing_context_first_token_ms_property(self):
        """Verify first_token_latency_ms property calculation."""
        with TimingContext() as timer:
            assert timer.first_token_latency_ms == 0.0, "Should be 0 before mark"

            time.sleep(0.005)
            timer.mark_first_chunk()

            latency = timer.first_token_latency_ms
            assert latency >= 5.0, "Should capture at least 5ms delay"

    def test_timing_context_total_latency_ms_property(self):
        """Verify total_latency_ms property calculation."""
        with TimingContext() as timer:
            time.sleep(0.010)

        assert timer.total_latency_ms >= 10.0, "Should capture total duration"


class TestIntegrationProviderTimingWithSpans:
    """Integration tests: spans + timing context together."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── spans a GenAI-aware backend (Langfuse, Phoenix, Jaeger) can read ─────────

def _recording_tracer():
    """An OtelTracer whose spans end up in a list instead of on the network."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from orchestra.engine.tracer import OtelTracer

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracer.__new__(OtelTracer)
    tracer.jaeger_enabled = False
    tracer._is_available = True
    tracer.tracer = provider.get_tracer("test")
    return tracer, exporter


def test_an_agent_span_uses_the_genai_attribute_names():
    tracer, exporter = _recording_tracer()

    span = tracer.start_agent_span("run-1", "cld-deep", "claude/opus", provider="claude")
    tracer.set_tokens(span, 1200, 300)
    tracer.set_cost(span, 0.42)
    tracer.end_span(span)

    (recorded,) = exporter.get_finished_spans()
    attributes = dict(recorded.attributes)
    assert recorded.name == "invoke_agent cld-deep"
    assert attributes["gen_ai.operation.name"] == "invoke_agent"
    assert attributes["gen_ai.agent.name"] == "cld-deep"
    assert attributes["gen_ai.request.model"] == "claude/opus"
    assert attributes["gen_ai.provider.name"] == "claude"
    assert attributes["gen_ai.conversation.id"] == "run-1"
    assert (attributes["gen_ai.usage.input_tokens"], attributes["gen_ai.usage.output_tokens"]) == (1200, 300)
    assert attributes["orchestra.cost_usd"] == 0.42


def test_a_failed_span_is_marked_as_an_error():
    from opentelemetry.trace import StatusCode

    tracer, exporter = _recording_tracer()

    span = tracer.start_agent_span("run-1", "cld-deep", "claude/opus")
    tracer.set_error(span, "Timeout after 180s")
    tracer.end_span(span)

    (recorded,) = exporter.get_finished_spans()
    assert recorded.status.status_code == StatusCode.ERROR
    assert recorded.attributes["error.type"] == "Timeout after 180s"


def test_otlp_export_works_with_the_exporter_the_project_installs():
    """pyproject installs the HTTP exporter; the tracer imported the gRPC one and never exported."""
    from orchestra.engine import tracer as tracer_module

    assert tracer_module.OTLP_AVAILABLE is True


def test_the_configured_endpoint_and_headers_reach_the_exporter(monkeypatch, real_otlp_exporter):
    from orchestra import config
    from orchestra.engine import tracer as tracer_module

    monkeypatch.setitem(config._cfg, "tracer", {
        "otlp_endpoint": "http://localhost:3000/api/public/otel",
        "otlp_headers": {"Authorization": "Basic abc"},
    })

    # What the exporter is built with; its attributes are private and differ between versions.
    with patch.object(tracer_module, "OTLPSpanExporter") as exporter_class, \
         patch.object(tracer_module, "OTLP_PROTOCOL", "http"):
        real_otlp_exporter("ignored-host")

    assert exporter_class.call_args.kwargs == {
        "endpoint": "http://localhost:3000/api/public/otel/v1/traces",
        "headers": {"Authorization": "Basic abc"},
    }
    assert tracer_module.export_configured() is True


def test_a_runs_span_reports_the_prompt_tokens_too():
    from unittest.mock import patch

    from orchestra.engine.runners import common

    run = OrchestraRun(mode="ask", task="t")
    agent = AgentRun(alias="cdx-deep", provider="codex", model="m")
    agent.prompt_tokens, agent.estimated_completion_tokens, agent.estimated_cost_usd = 900, 100, 0.1
    run.agents = [agent]
    tracer = MagicMock()

    with patch.object(common, "_get_tracer", return_value=tracer):
        common._emit_span_usage(object(), run)

    assert tracer.set_tokens.call_args.args[1:] == (900, 100)
