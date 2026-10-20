# tests/test_fallback_node_ref.py
from unittest.mock import patch
import pytest
from orchestra.providers import fallback
from orchestra.providers import ollama_backends as ob


def test_no_at_is_backward_compatible():
    provider, effort = fallback.resolve("oll-fast")
    assert provider is not None
