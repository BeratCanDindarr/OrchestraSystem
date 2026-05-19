"""Shared pytest configuration."""
from orchestra import config

# Disable idempotency caching for tests by default to avoid cross-test pollution.
config.set("idempotency", "enabled", False)
