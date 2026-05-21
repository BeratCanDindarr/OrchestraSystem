from __future__ import annotations
"""OpenTelemetry tracing integration with OTLP (or Console) exporter for distributed tracing.

Provides OtelTracer singleton for instrumenting agent runs, capturing first-token
latency, total latency, token counts, and cost attribution per span.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, Sequence

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    SimpleSpanProcessor = None
    SpanExporter = None
    SpanExportResult = None

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
    OTLPSpanExporter = None

logger = logging.getLogger(__name__)


class _InMemorySpanExporter(SpanExporter):
    """Fallback exporter that buffers spans in memory when Jaeger is unreachable."""

    def __init__(self):
        """Initialize in-memory span buffer."""
        self._spans: list = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        """Buffer spans in memory.

        Args:
            spans: Sequence of ReadableSpan objects to buffer

        Returns:
            SpanExportResult.SUCCESS
        """
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Clear buffered spans."""
        with self._lock:
            self._spans.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op for in-memory exporter."""
        return True

    def get_spans(self) -> list:
        """Return copy of buffered spans."""
        with self._lock:
            return list(self._spans)


class OtelTracer:
    """Singleton OpenTelemetry tracer for agent execution.

    Integrates OpenTelemetry with OTLP or Console exporter for distributed tracing.
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
        jaeger_enabled: bool = False,
        batch_size: int = 512,
        batch_timeout_ms: int = 5000,
    ):
        """Initialize tracer with OTLP exporter (or Console fallback).

        Args:
            service_name: Service name for traces
            jaeger_host: Jaeger agent/OTLP host
            jaeger_port: Jaeger agent/OTLP port (deprecated for OTLP, kept for compatibility)
            fallback_console: Fall back to console exporter if OTLP unavailable
            jaeger_enabled: Enable Jaeger OTLP export (lazy initialization)
            batch_size: Max spans per export batch
            batch_timeout_ms: Force export after this timeout
        """
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.fallback_console = fallback_console
        self.jaeger_enabled = jaeger_enabled
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.tracer: Optional[Any] = None
        self._is_available = False
        self._exporter: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._trace_provider: Optional[Any] = None
        self._initialized = False

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available. Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
            )
            return

        try:
            from orchestra import config

            if not config.tracing_enabled():
                logger.info("Tracing disabled via config")
                return

            # Override defaults with config values when not explicitly provided
            if not self.jaeger_enabled:
                self.jaeger_enabled = bool(config.tracer_jaeger_enabled())
            self.batch_size = int(config.tracer_batch_size()) if not self.batch_size else self.batch_size
            self.batch_timeout_ms = (
                int(config.tracer_batch_timeout_ms()) if not self.batch_timeout_ms else self.batch_timeout_ms
            )
        except Exception:
            logger.exception("Ignored exception")


        # If Jaeger is enabled, defer exporter creation to _ensure_exporter on first span
        if self.jaeger_enabled:
            return

        # Otherwise set up immediately (console fallback or nothing)
        self._setup_exporter(fallback_console=fallback_console)

    def _setup_exporter(self, fallback_console: bool = False, force_exporter: Any = None):
        """Create exporter, processor, and tracer provider.

        Args:
            fallback_console: Whether to fall back to ConsoleSpanExporter
            force_exporter: Use this exporter instance directly instead of creating one
        """
        exporter = force_exporter

        if exporter is None:
            try:
                if OTLP_AVAILABLE:
                    exporter = OTLPSpanExporter()
            except Exception as e:
                logger.warning(f"Failed to initialize OTLP exporter: {e}")

        if exporter is None and fallback_console:
            try:
                exporter = ConsoleSpanExporter()
            except Exception as e:
                logger.warning(f"Failed to initialize Console exporter: {e}")

        if exporter is None:
            if not fallback_console:
                logger.warning("No trace exporter available and fallback_console=False")
            return

        processor = None
        try:
            if isinstance(exporter, ConsoleSpanExporter):
                processor = SimpleSpanProcessor(exporter)
            else:
                processor = BatchSpanProcessor(
                    exporter,
                    max_export_batch_size=self.batch_size,
                    schedule_delay_millis=self.batch_timeout_ms,
                )
        except Exception as e:
            logger.warning(f"Failed to create span processor: {e}")
            return

        try:
            existing_provider = trace.get_tracer_provider()
            # Use duck-typing check to avoid isinstance() issues when TracerProvider is mocked in tests
            if hasattr(existing_provider, "add_span_processor"):
                trace_provider = existing_provider
                trace_provider.add_span_processor(processor)
            else:
                trace_provider = TracerProvider()
                trace_provider.add_span_processor(processor)
                trace.set_tracer_provider(trace_provider)

            self.tracer = trace.get_tracer(__name__)
            self._is_available = True
            self._exporter = exporter
            self._processor = processor
            self._trace_provider = trace_provider
            logger.info(f"Initialized OtelTracer with {exporter.__class__.__name__}")
        except Exception as e:
            logger.warning(f"Failed to initialize tracer provider: {e}")

    def _ensure_exporter(self) -> bool:
        """Lazily create exporter when jaeger_enabled=True.

        Creates OTLP exporter pointing at the configured Jaeger host.
        If Jaeger is unreachable, falls back to an in-memory exporter.

        Returns:
            True if exporter was successfully created or already exists
        """
        if self._initialized:
            return self._is_available
        self._initialized = True

        if not self.jaeger_enabled:
            return False

        exporter = None
        try:
            if OTLP_AVAILABLE:
                endpoint = f"http://{self.jaeger_host}:4317"
                exporter = OTLPSpanExporter(endpoint=endpoint)
        except Exception as e:
            logger.warning(f"Failed to connect to Jaeger at {self.jaeger_host}:4317: {e}")

        if exporter is None:
            # Fallback: buffer spans in-memory
            try:
                exporter = _InMemorySpanExporter()
                logger.warning("Jaeger unreachable. Falling back to in-memory span buffering.")
            except Exception as e:
                logger.warning(f"Failed to create in-memory fallback exporter: {e}")
                return False

        try:
            processor = BatchSpanProcessor(
                exporter,
                max_export_batch_size=self.batch_size,
                schedule_delay_millis=self.batch_timeout_ms,
            )
            existing_provider = trace.get_tracer_provider()
            # Use duck-typing check to avoid isinstance() issues when TracerProvider is mocked in tests
            if hasattr(existing_provider, "add_span_processor"):
                trace_provider = existing_provider
                trace_provider.add_span_processor(processor)
            else:
                trace_provider = TracerProvider()
                trace_provider.add_span_processor(processor)
                trace.set_tracer_provider(trace_provider)

            self.tracer = trace.get_tracer(__name__)
            self._is_available = True
            self._exporter = exporter
            self._processor = processor
            self._trace_provider = trace_provider
            logger.info(f"Initialized OtelTracer with {exporter.__class__.__name__}")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize tracer provider: {e}")
            return False

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
        if self.jaeger_enabled:
            self._ensure_exporter()

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
        if not OTEL_AVAILABLE or parent_span is None:
            return None

        if self.jaeger_enabled:
            self._ensure_exporter()

        if not self.is_available():
            return None

        try:
            # Safe context management: attach parent but don't auto-close
            ctx = trace.use_span(parent_span, end_on_exit=False)
            ctx.__enter__()  # Attach to context without closing on exit
            child_span = self.tracer.start_span(
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
