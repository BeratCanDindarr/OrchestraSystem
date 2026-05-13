"""OpenTelemetry tracing integration with Jaeger exporter for distributed tracing.

Provides OtelTracer singleton for instrumenting agent runs, capturing first-token
latency, total latency, token counts, and cost attribution per span.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Dict, Any

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    TracerProvider = None
    BatchSpanProcessor = None
    JaegerExporter = None

logger = logging.getLogger(__name__)


class OtelTracer:
    """Singleton OpenTelemetry tracer for agent execution.

    Integrates OpenTelemetry with Jaeger exporter for distributed tracing.
    Captures timing metrics, token counts, and cost attribution per span.

    Span hierarchy:
        run (TraceID)
          ├─ agent.{alias} (first_token_latency_ms, total_latency_ms, tokens_in, tokens_out, cost_usd)
          │   └─ provider.{provider} (subprocess/streaming output, error handling)
    """

    _instance: Optional[OtelTracer] = None
    _lock = threading.Lock()

    def __init__(
        self,
        service_name: str = "orchestra",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        fallback_console: bool = False,
    ):
        """Initialize tracer with Jaeger exporter.

        Args:
            service_name: Service name for Jaeger traces
            jaeger_host: Jaeger agent hostname
            jaeger_port: Jaeger agent port
            fallback_console: Fall back to console exporter if Jaeger unavailable
        """
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.tracer: Optional[Any] = None
        self._is_available = False

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available. Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger"
            )
            return

        try:
            # Try Jaeger exporter first
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )

            trace_provider = TracerProvider()
            trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            trace.set_tracer_provider(trace_provider)

            self.tracer = trace.get_tracer(__name__)
            self._is_available = True
            logger.info(
                f"Initialized OtelTracer with Jaeger exporter ({jaeger_host}:{jaeger_port})"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Jaeger exporter: {e}")
            if fallback_console:
                try:
                    # Fallback: in-memory exporter for testing
                    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
                    from opentelemetry.sdk.trace.export import in_memory_span_exporter

                    in_memory_exporter = in_memory_span_exporter.InMemorySpanExporter()
                    trace_provider = TracerProvider()
                    trace_provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
                    trace.set_tracer_provider(trace_provider)

                    self.tracer = trace.get_tracer(__name__)
                    self._is_available = True
                    logger.info("Initialized OtelTracer with in-memory exporter (testing mode)")
                except Exception as e2:
                    logger.error(f"Failed to initialize fallback exporter: {e2}")

    @classmethod
    def get_instance(cls, **kwargs) -> OtelTracer:
        """Singleton access to OtelTracer.

        Args:
            **kwargs: Arguments passed to __init__ (only used on first call)

        Returns:
            OtelTracer singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = OtelTracer(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def is_available(self) -> bool:
        """Check if tracer is properly initialized."""
        return self._is_available and self.tracer is not None

    def start_agent_span(self, run_id: str, alias: str, model: str):
        """Start span for agent execution.

        Args:
            run_id: Unique run identifier
            alias: Agent alias (e.g., "cdx-deep", "gem-fast")
            model: Model identifier (e.g., "gpt-5.4/xhigh")

        Returns:
            Active span context or None if tracing unavailable
        """
        if not self.is_available():
            return None

        try:
            span = self.tracer.start_span(
                f"agent.{alias}",
                attributes={
                    "run_id": run_id,
                    "alias": alias,
                    "model": model,
                    "span_type": "agent",
                },
            )
            return span
        except Exception as e:
            logger.error(f"Failed to start agent span: {e}")
            return None

    def start_provider_span(self, parent_span, provider: str, method: str):
        """Start child span for provider call.

        Args:
            parent_span: Parent agent span
            provider: Provider name ("codex", "gemini", "claude", "ollama")
            method: Execution method ("sync", "streaming", "subprocess")

        Returns:
            Child span for provider operation or None if tracing unavailable
        """
        if not OTEL_AVAILABLE or not self.is_available() or parent_span is None:
            return None

        try:
            # Safe context management: attach parent but don't auto-close
            ctx = trace.use_span(parent_span, end_on_exit=False)
            ctx.__enter__()  # Attach to context without closing on exit
            child_span = self.tracer.start_as_current_span(
                f"provider.{provider}",
                attributes={
                    "provider": provider,
                    "method": method,
                    "span_type": "provider",
                },
            )
            return child_span
        except Exception as e:
            logger.error(f"Failed to start provider span: {e}")
            return None

    def set_timing(self, span, metric_name: str, value_ms: float):
        """Record latency metric on span.

        Args:
            span: Target span
            metric_name: Metric name (e.g., "first_token_latency_ms", "total_latency_ms")
            value_ms: Value in milliseconds (float)
        """
        if span is None:
            return

        try:
            span.set_attribute(metric_name, value_ms)
        except Exception as e:
            logger.error(f"Failed to set timing metric {metric_name}: {e}")

    def set_tokens(self, span, tokens_in: int, tokens_out: int):
        """Record token counts on span.

        Args:
            span: Target span
            tokens_in: Input token count
            tokens_out: Output token count
        """
        if span is None:
            return

        try:
            span.set_attribute("tokens_in", tokens_in)
            span.set_attribute("tokens_out", tokens_out)
        except Exception as e:
            logger.error(f"Failed to set token counts: {e}")

    def set_cost(self, span, cost_usd: float):
        """Record cost attribution on span.

        Args:
            span: Target span
            cost_usd: Estimated cost in USD
        """
        if span is None:
            return

        try:
            span.set_attribute("cost_usd", cost_usd)
        except Exception as e:
            logger.error(f"Failed to set cost: {e}")

    def set_error(self, span, error_message: str):
        """Record error on span.

        Args:
            span: Target span
            error_message: Error message string
        """
        if span is None:
            return

        try:
            span.set_attribute("error", error_message)
            span.set_attribute("status_code", "ERROR")
        except Exception as e:
            logger.error(f"Failed to set error: {e}")

    def end_span(self, span):
        """End a span.

        Args:
            span: Target span to end
        """
        if span is None:
            return

        try:
            span.end()
        except Exception as e:
            logger.error(f"Failed to end span: {e}")

    def flush(self, timeout_secs: int = 30):
        """Flush pending spans to exporter.

        Args:
            timeout_secs: Timeout in seconds for flush operation
        """
        if not self.is_available():
            return

        try:
            trace.get_tracer_provider().force_flush(timeout_ms=timeout_secs * 1000)
        except Exception as e:
            logger.error(f"Failed to flush tracer: {e}")


class TimingContext:
    """Context manager for capturing operation timing.

    Tracks first-token timing (for streaming) and total latency.
    """

    def __init__(self):
        """Initialize timing context."""
        self.start_time: Optional[float] = None
        self.first_chunk_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        """Enter context, record start time."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, record end time."""
        self.end_time = time.time()

    def mark_first_chunk(self):
        """Mark when first chunk/token is received (for streaming)."""
        if self.first_chunk_time is None:
            self.first_chunk_time = time.time()

    @property
    def first_token_latency_ms(self) -> float:
        """Get first-token latency in milliseconds.

        Returns:
            Latency in ms, or 0 if first chunk not marked.
        """
        if self.start_time is None or self.first_chunk_time is None:
            return 0.0
        return (self.first_chunk_time - self.start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        """Get total latency in milliseconds.

        Returns:
            Latency in ms, or 0 if timing not complete.
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
