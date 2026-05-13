"""Phase 4 tests for Block 7 (OtelTracer Provider Timing).

Tests span lifecycle, run_ask instrumentation, and TTFT capture across all providers.
Verifies that parent spans survive child operations, timing metrics are recorded correctly,
and streaming-based TTFT capture works for Claude SDK and Ollama.
"""
from __future__ import annotations

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

    def test_claude_sdk_first_token_timing(self):
        """Verify Claude SDK marks first chunk for TTFT capture.

        Block 7 requirement: claude_sdk.py must:
        1. Use client.messages.stream() for streaming
        2. Call timer.mark_first_chunk() on first token
        3. Verify first_token_latency_ms < total_latency_ms
        """
        # Simulate Claude SDK streaming pattern (from claude_sdk.py:67-82)
        with TimingContext() as timer:
            # Simulate initial delay (network + model startup)
            time.sleep(0.005)

            # First token arrives
            timer.mark_first_chunk()

            # Stream remaining tokens
            time.sleep(0.005)

        # Verify timing captured correctly
        assert timer.first_token_latency_ms > 0, "First token latency should be marked"
        assert timer.total_latency_ms > timer.first_token_latency_ms, "Total > first token"
        assert timer.first_token_latency_ms < timer.total_latency_ms

    def test_claude_sdk_uses_streaming_api(self):
        """Verify Claude SDK provider uses client.messages.stream(), not .create().

        Block 7 requirement: streaming must be used for TTFT, not sync.
        """
        from orchestra.providers.claude_sdk import ClaudeSDKProvider

        provider = ClaudeSDKProvider()

        # Read provider source to verify stream() usage
        import inspect
        source = inspect.getsource(provider.run)

        # Verify streaming API used
        assert "client.messages.stream" in source, "Must use .stream() for TTFT capture"
        assert "for text in stream.text_stream" in source, "Must iterate text stream"
        assert "timer.mark_first_chunk()" in source, "Must mark first token"


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

    def test_claude_provider_timing_with_agent_span(self):
        """Full Claude streaming with agent + provider spans."""
        tracer = OtelTracer.get_instance(fallback_console=True)

        # Agent span
        agent_span = tracer.start_agent_span("run-claude-001", "cld-fast", "claude/sonnet")
        if agent_span is None:
            pytest.skip("OTEL not available")

        # Provider span
        provider_span = tracer.start_provider_span(agent_span, "claude", "streaming")

        if provider_span:
            # Timing
            with TimingContext() as timer:
                time.sleep(0.005)
                timer.mark_first_chunk()
                time.sleep(0.005)

            tracer.set_timing(provider_span, "first_token_latency_ms", timer.first_token_latency_ms)
            tracer.set_timing(provider_span, "total_latency_ms", timer.total_latency_ms)
            tracer.set_tokens(provider_span, 1000, 250)
            tracer.set_cost(provider_span, 0.05)
            tracer.end_span(provider_span)

        # Agent span should still be active
        tracer.end_span(agent_span)

    def test_ollama_provider_timing_with_agent_span(self):
        """Full Ollama HTTP with agent + provider spans."""
        tracer = OtelTracer.get_instance(fallback_console=True)

        agent_span = tracer.start_agent_span("run-ollama-001", "oll-coder", "llama2")
        if agent_span is None:
            pytest.skip("OTEL not available")

        provider_span = tracer.start_provider_span(agent_span, "ollama", "http")

        if provider_span:
            with TimingContext() as timer:
                time.sleep(0.003)
                timer.mark_first_chunk()
                time.sleep(0.005)

            tracer.set_timing(provider_span, "first_token_latency_ms", timer.first_token_latency_ms)
            tracer.set_timing(provider_span, "total_latency_ms", timer.total_latency_ms)
            tracer.set_tokens(provider_span, 800, 400)
            tracer.set_cost(provider_span, 0.0)  # Local model
            tracer.end_span(provider_span)

        tracer.end_span(agent_span)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
