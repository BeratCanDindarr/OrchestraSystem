from __future__ import annotations
"""Integration tests for Jaeger exporter in OtelTracer.

Mocks the OTLP exporter to verify batching, timeouts, and span attributes
without requiring a running Jaeger instance.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from orchestra.engine import tracer as tracer_module
from orchestra.engine.tracer import OtelTracer, _InMemorySpanExporter


class TestJaegerExporter:
    """Test Jaeger exporter initialization and span export."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset tracer before each test."""
        OtelTracer.reset_instance()

    def test_jaeger_enabled_triggers_lazy_exporter(self):
        """Verify jaeger_enabled=True defers exporter creation to _ensure_exporter."""
        with patch.object(tracer_module, "OTLP_AVAILABLE", True):
            with patch.object(tracer_module, "OTLPSpanExporter") as mock_exporter_class:
                mock_exporter = Mock()
                mock_exporter_class.return_value = mock_exporter

                tracer = OtelTracer(jaeger_enabled=True, batch_size=512, batch_timeout_ms=5000)
                assert tracer.jaeger_enabled is True

                # Before _ensure_exporter, no exporter should exist
                assert tracer._exporter is None
                assert tracer._is_available is False

                # Trigger lazy initialization
                tracer._ensure_exporter()

                assert tracer._exporter is mock_exporter
                assert tracer._is_available is True
                mock_exporter_class.assert_called_once()

    def test_batch_size_passed_to_processor(self):
        """Verify batch_size is forwarded to BatchSpanProcessor."""
        with patch.object(tracer_module, "OTLP_AVAILABLE", True):
            with patch.object(tracer_module, "OTLPSpanExporter"):
                with patch.object(tracer_module, "BatchSpanProcessor") as mock_processor:
                    mock_processor.return_value = Mock()

                    tracer = OtelTracer(jaeger_enabled=True, batch_size=256, batch_timeout_ms=3000)
                    tracer._ensure_exporter()

                    mock_processor.assert_called_once()
                    _, kwargs = mock_processor.call_args
                    assert kwargs.get("max_export_batch_size") == 256

    def test_batch_timeout_forces_flush(self):
        """Verify batch_timeout_ms is forwarded to BatchSpanProcessor schedule_delay."""
        with patch.object(tracer_module, "OTLP_AVAILABLE", True):
            with patch.object(tracer_module, "OTLPSpanExporter"):
                with patch.object(tracer_module, "BatchSpanProcessor") as mock_processor:
                    mock_processor.return_value = Mock()

                    tracer = OtelTracer(jaeger_enabled=True, batch_size=100, batch_timeout_ms=1000)
                    tracer._ensure_exporter()

                    _, kwargs = mock_processor.call_args
                    assert kwargs.get("schedule_delay_millis") == 1000

    def test_span_contains_expected_attributes(self):
        """Verify spans contain run_id, provider, tokens, cost attributes."""
        with patch.object(tracer_module, "OTLP_AVAILABLE", True):
            with patch.object(tracer_module, "OTLPSpanExporter"):
                with patch.object(tracer_module, "BatchSpanProcessor"):
                    with patch.object(tracer_module, "TracerProvider") as mock_provider_cls:
                        mock_provider = Mock()
                        mock_tracer = Mock()
                        mock_span = Mock()
                        mock_tracer.start_span.return_value = mock_span
                        mock_provider_cls.return_value = mock_provider

                        with patch.object(tracer_module.trace, "get_tracer", return_value=mock_tracer):
                            tracer = OtelTracer(jaeger_enabled=True)
                            tracer._ensure_exporter()

                            span = tracer.start_agent_span("run-123", "cdx-deep", "gpt-5.4")
                            assert span is mock_span

                            mock_tracer.start_span.assert_called_once()
                            _, kwargs = mock_tracer.start_span.call_args
                            assert kwargs["attributes"]["run_id"] == "run-123"
                            assert kwargs["attributes"]["alias"] == "cdx-deep"
                            assert kwargs["attributes"]["model"] == "gpt-5.4"

                            tracer.set_tokens(span, 1000, 500)
                            mock_span.set_attribute.assert_any_call("tokens_in", 1000)
                            mock_span.set_attribute.assert_any_call("tokens_out", 500)

                            tracer.set_cost(span, 0.15)
                            mock_span.set_attribute.assert_any_call("cost_usd", 0.15)

    def test_jaeger_unreachable_fallback(self):
        """Verify fallback to in-memory exporter when Jaeger is unreachable."""
        with patch.object(
            tracer_module, "OTLPSpanExporter", side_effect=Exception("Connection refused")
        ):
            with patch.object(tracer_module, "OTLP_AVAILABLE", True):
                tracer = OtelTracer(jaeger_enabled=True)
                result = tracer._ensure_exporter()

                assert result is True
                assert isinstance(tracer._exporter, _InMemorySpanExporter)

    def test_provider_span_contains_provider_attribute(self):
        """Verify provider span contains provider and method attributes."""
        with patch.object(tracer_module, "OTLP_AVAILABLE", True):
            with patch.object(tracer_module, "OTLPSpanExporter"):
                with patch.object(tracer_module, "BatchSpanProcessor"):
                    with patch.object(tracer_module, "TracerProvider") as mock_provider_cls:
                        mock_provider = Mock()
                        mock_tracer = Mock()
                        mock_parent = Mock()
                        mock_child = Mock()

                        mock_tracer.start_span.return_value = mock_child
                        mock_provider_cls.return_value = mock_provider

                        with patch.object(tracer_module.trace, "get_tracer", return_value=mock_tracer):
                            tracer = OtelTracer(jaeger_enabled=True)
                            tracer._ensure_exporter()

                            child = tracer.start_provider_span(mock_parent, "codex", "subprocess")
                            assert child is mock_child

                            _, kwargs = mock_tracer.start_span.call_args
                            assert kwargs["attributes"]["provider"] == "codex"
                            assert kwargs["attributes"]["method"] == "subprocess"
                            assert kwargs["attributes"]["span_type"] == "provider"

    def test_lazy_initialization_on_first_span(self):
        """Verify _ensure_exporter is called automatically on first span."""
        with patch.object(tracer_module, "OTLP_AVAILABLE", True):
            with patch.object(tracer_module, "OTLPSpanExporter") as mock_exporter_class:
                mock_exporter = Mock()
                mock_exporter_class.return_value = mock_exporter

                tracer = OtelTracer(jaeger_enabled=True)
                assert tracer._initialized is False

                # Mock trace.get_tracer to avoid full OTel setup
                with patch.object(tracer_module.trace, "get_tracer") as mock_get_tracer:
                    mock_get_tracer.return_value = Mock()
                    tracer.start_agent_span("run-001", "test", "model")

                assert tracer._initialized is True
                mock_exporter_class.assert_called_once()

    def test_in_memory_exporter_buffers_spans(self):
        """Verify _InMemorySpanExporter buffers exported spans."""
        exporter = _InMemorySpanExporter()
        mock_span = Mock()

        result = exporter.export([mock_span])
        assert result.name == "SUCCESS"
        assert exporter.get_spans() == [mock_span]

        exporter.shutdown()
        assert exporter.get_spans() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
