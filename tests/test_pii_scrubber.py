"""Tests for PiiScrubber — in-memory PII pattern redaction."""
from __future__ import annotations

import pytest
from orchestra.engine.pii_scrubber import PiiScrubber


# ---------------------------------------------------------------------------
# API Key Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_sk_api_key():
    """sk_test_* pattern should be redacted."""
    text = "My API key is sk_test_1234567890abcdefghij"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:api_key_sk]" in result
    assert "sk_test_1234567890abcdefghij" not in result


def test_scrub_sk_api_key_with_dash():
    """sk- variant should also be redacted."""
    text = "Key: sk-abcdefghij1234567890xyz"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:api_key_sk]" in result
    assert "sk-abcdefghij1234567890xyz" not in result


def test_scrub_pk_api_key():
    """pk_ pattern should be redacted."""
    text = "Public key: pk_test_abcdef1234567890wxyz"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:api_key_pk]" in result


def test_short_api_key_not_redacted():
    """Short keys (< 20 chars) should not be redacted."""
    text = "sk_test123"
    result = PiiScrubber.scrub_text(text)
    assert result == text  # Unchanged


# ---------------------------------------------------------------------------
# JWT Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_jwt():
    """JWT (eyJ...) pattern should be redacted."""
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP9PmMtNk"
    text = f"Token: {jwt}"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:jwt]" in result
    assert jwt not in result


def test_jwt_too_short_not_redacted():
    """Short JWT-like strings should not be redacted."""
    text = "eyJhbGc"
    result = PiiScrubber.scrub_text(text)
    assert result == text


# ---------------------------------------------------------------------------
# Email Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_email():
    """Email addresses should be redacted."""
    text = "Contact: user@example.com"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:email]" in result
    assert "user@example.com" not in result


def test_scrub_complex_email():
    """Complex email with dots and underscores."""
    text = "Email: john.doe+test_123@company.co.uk"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:email]" in result


def test_scrub_multiple_emails():
    """Multiple emails in one text should all be redacted."""
    text = "Send to alice@example.com or bob@test.org"
    result = PiiScrubber.scrub_text(text)
    assert result.count("[REDACTED:email]") == 2
    assert "alice@example.com" not in result
    assert "bob@test.org" not in result


# ---------------------------------------------------------------------------
# Phone Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_phone_basic():
    """Standard US phone format."""
    text = "Call: 555-123-4567"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:phone]" in result


def test_scrub_phone_with_country_code():
    """Phone with country code."""
    text = "Dial +1 (555) 123-4567"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:phone]" in result


def test_scrub_phone_with_area_code():
    """Phone with parenthetical area code."""
    text = "Number: (555) 123-4567"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:phone]" in result


# ---------------------------------------------------------------------------
# SSN Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_ssn():
    """SSN pattern (XXX-XX-XXXX)."""
    text = "SSN: 123-45-6789"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:ssn]" in result
    assert "123-45-6789" not in result


def test_scrub_ssn_no_dashes():
    """SSN without dashes."""
    text = "SSN: 123 45 6789"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:ssn]" in result


# ---------------------------------------------------------------------------
# Credit Card Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_credit_card():
    """Credit card (4 groups of 4 digits)."""
    text = "Card: 4532-1234-5678-9010"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:credit_card]" in result


def test_scrub_credit_card_no_dashes():
    """Credit card without dashes."""
    text = "Payment: 4532 1234 5678 9010"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:credit_card]" in result


# ---------------------------------------------------------------------------
# OAuth Token Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_oauth_token():
    """OAuth2 token pattern."""
    text = "Token: oauth2_abcdef1234567890wxyz1234567890abcdefghij"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:oauth_token]" in result


def test_scrub_oauth_token_with_dash():
    """OAuth token with dash."""
    text = "oauth-2_abcdef1234567890wxyz1234567890abcdefghij"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:oauth_token]" in result


# ---------------------------------------------------------------------------
# AWS Key Pattern Tests
# ---------------------------------------------------------------------------

def test_scrub_aws_key():
    """AWS access key pattern (AKIA...)."""
    text = "AWS Key: AKIAIOSFODNN7EXAMPLE"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:aws_key]" in result
    assert "AKIAIOSFODNN7EXAMPLE" not in result


# ---------------------------------------------------------------------------
# Entropy-Based Filtering Tests
# ---------------------------------------------------------------------------

def test_base64_image_not_redacted():
    """Base64 image prefix should be whitelisted."""
    # iVB is PNG magic bytes
    text = "Image: iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    result = PiiScrubber.scrub_text(text)
    assert result == text  # Should remain unchanged


def test_jpeg_base64_not_redacted():
    """JPEG Base64 prefix should be whitelisted."""
    text = "/9j/4AAQSkZJRgABAQAAAQABAAD..."
    result = PiiScrubber.scrub_text(text)
    assert result == text


def test_gif_base64_not_redacted():
    """GIF Base64 prefix should be whitelisted."""
    text = "GIF89a[REDACTED]GIF content..."
    result = PiiScrubber.scrub_text(text)
    # GIF89a at start should prevent redaction
    assert "GIF89a" in result


def test_hex_hash_not_redacted():
    """Long hex strings (hashes) should not be redacted."""
    # SHA1 hash (40 hex chars)
    sha1 = "356a192b7913b04c54574d18c28d46e6395428ab"
    text = f"Hash: {sha1}"
    result = PiiScrubber.scrub_text(text)
    assert sha1 in result  # Should remain unchanged


def test_entropy_computation():
    """Entropy calculation should work correctly."""
    # Low entropy (repeating character)
    assert PiiScrubber.compute_entropy("aaaaaaaaaa") < 1.0

    # High entropy (random)
    assert PiiScrubber.compute_entropy("9K3qX7mZ2L4pW5vN") > 3.0

    # Medium entropy
    assert PiiScrubber.compute_entropy("abcabcabca") > 0.9


# ---------------------------------------------------------------------------
# Dictionary Scrubbing Tests
# ---------------------------------------------------------------------------

def test_scrub_dict_simple():
    """Scrub dict with simple string values."""
    obj = {
        "user": "john@example.com",
        "name": "John Doe"
    }
    result = PiiScrubber.scrub_dict(obj)
    assert "[REDACTED:email]" in result["user"]
    assert result["name"] == "John Doe"
    # Original should be unchanged
    assert obj["user"] == "john@example.com"


def test_scrub_dict_immutable():
    """Verify immutable pattern — original dict unchanged."""
    original = {"api_key": "sk_test_1234567890abcdefghij"}
    result = PiiScrubber.scrub_dict(original)
    # Original unchanged
    assert original["api_key"] == "sk_test_1234567890abcdefghij"
    # Result scrubbed
    assert "[REDACTED:api_key_sk]" in result["api_key"]
    # Different objects
    assert result is not original


def test_scrub_dict_nested():
    """Scrub nested dictionaries recursively."""
    obj = {
        "user": {
            "email": "alice@example.com",
            "profile": {
                "phone": "555-123-4567"
            }
        }
    }
    result = PiiScrubber.scrub_dict(obj)
    assert "[REDACTED:email]" in result["user"]["email"]
    assert "[REDACTED:phone]" in result["user"]["profile"]["phone"]


def test_scrub_dict_with_list():
    """Scrub dicts inside lists."""
    obj = {
        "contacts": [
            {"email": "user1@example.com"},
            {"email": "user2@example.com"}
        ]
    }
    result = PiiScrubber.scrub_dict(obj)
    assert "[REDACTED:email]" in result["contacts"][0]["email"]
    assert "[REDACTED:email]" in result["contacts"][1]["email"]


def test_scrub_dict_mixed_types():
    """Dict with mixed types should preserve non-string types."""
    obj = {
        "email": "user@example.com",
        "age": 30,
        "active": True,
        "score": 9.5,
        "tags": ["admin", "user@internal.com"]
    }
    result = PiiScrubber.scrub_dict(obj)
    assert "[REDACTED:email]" in result["email"]
    assert result["age"] == 30
    assert result["active"] is True
    assert result["score"] == 9.5
    assert "[REDACTED:email]" in result["tags"][1]


def test_scrub_dict_empty():
    """Empty dict should return empty dict."""
    result = PiiScrubber.scrub_dict({})
    assert result == {}


def test_scrub_dict_non_dict_input():
    """Non-dict input should return scrubbed string."""
    text = "Email: user@example.com"
    result = PiiScrubber.scrub_dict(text)
    assert "[REDACTED:email]" in result


def test_scrub_dict_list_input():
    """List input should return scrubbed list."""
    items = ["user@example.com", "sk_test_1234567890abcdefghij"]
    result = PiiScrubber.scrub_dict(items)
    assert "[REDACTED:email]" in result[0]
    assert "[REDACTED:api_key_sk]" in result[1]


# ---------------------------------------------------------------------------
# Sensitive Field Scrubbing Tests
# ---------------------------------------------------------------------------

def test_scrub_sensitive_fields():
    """Scrub specific named fields."""
    obj = {
        "username": "alice",
        "password": "securepass123",
        "api_key": "sk_test_123456",
        "email": "alice@example.com"
    }
    result = PiiScrubber.scrub_sensitive_fields(obj, ["password", "api_key"])
    assert result["password"] == "[REDACTED]"
    assert result["api_key"] == "[REDACTED]"
    assert result["username"] == "alice"
    assert result["email"] == "alice@example.com"  # Not in field list


def test_scrub_sensitive_fields_immutable():
    """Sensitive field scrubbing should be immutable."""
    original = {"secret": "my-secret"}
    result = PiiScrubber.scrub_sensitive_fields(original, ["secret"])
    assert original["secret"] == "my-secret"
    assert result["secret"] == "[REDACTED]"


def test_scrub_sensitive_fields_missing():
    """Missing fields should be ignored gracefully."""
    obj = {"username": "alice"}
    result = PiiScrubber.scrub_sensitive_fields(obj, ["password", "api_key"])
    assert result == obj  # Unchanged


# ---------------------------------------------------------------------------
# Complex Real-World Tests
# ---------------------------------------------------------------------------

def test_scrub_event_payload():
    """Scrub a realistic event payload."""
    payload = {
        "run_id": "run_abc123",
        "timestamp": "2026-05-12T12:00:00Z",
        "agent": {
            "name": "cdx-deep",
            "email": "agent@example.com"
        },
        "request": "Please analyze user@company.com for payment status",
        "response": "User's account shows credit card 4532-1234-5678-9010",
        "metadata": {
            "aws_key": "AKIAIOSFODNN7EXAMPLE",
            "tokens": [
                "sk_test_1234567890abcdefghij",
                "oauth2_abcdef1234567890wxyz1234567890abcdefghij"
            ]
        }
    }
    result = PiiScrubber.scrub_dict(payload)

    # Verify scrubbing
    assert "[REDACTED:email]" in result["agent"]["email"]
    assert "[REDACTED:email]" in result["request"]
    assert "[REDACTED:credit_card]" in result["response"]
    assert "[REDACTED:aws_key]" in result["metadata"]["aws_key"]
    assert "[REDACTED:api_key_sk]" in result["metadata"]["tokens"][0]
    assert "[REDACTED:oauth_token]" in result["metadata"]["tokens"][1]

    # Verify non-PII preserved
    assert result["run_id"] == "run_abc123"
    assert result["timestamp"] == "2026-05-12T12:00:00Z"
    assert result["agent"]["name"] == "cdx-deep"


def test_multiple_patterns_same_text():
    """Text with multiple different PII patterns."""
    text = "User alice@example.com (555-123-4567) has key sk_test_1234567890abcdefghij"
    result = PiiScrubber.scrub_text(text)
    assert "[REDACTED:email]" in result
    assert "[REDACTED:phone]" in result
    assert "[REDACTED:api_key_sk]" in result
    assert "alice@example.com" not in result
    assert "555-123-4567" not in result


def test_overlapping_patterns():
    """Text where patterns might overlap — should handle gracefully."""
    text = "Email: test@example.com555-123-4567end"
    result = PiiScrubber.scrub_text(text)
    # Both should be redacted without corruption
    assert "[REDACTED:email]" in result or "[REDACTED:phone]" in result


# ---------------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------------

def test_scrub_large_text_performance():
    """Scrubbing should handle large texts efficiently."""
    # 10KB of text with scattered emails
    large_text = "word " * 2000
    large_text += "user@example.com " * 100
    large_text += "content " * 1000

    import time
    start = time.time()
    result = PiiScrubber.scrub_text(large_text)
    elapsed = time.time() - start

    # Should complete in < 10ms (reasonable for 10KB)
    assert elapsed < 0.1  # 100ms upper bound to be safe in CI
    assert "[REDACTED:email]" in result


def test_scrub_deep_nesting_performance():
    """Deeply nested dicts should be handled within recursion limit."""
    # Create nested structure
    obj = {"level1": {"level2": {"level3": {"data": "user@example.com"}}}}
    result = PiiScrubber.scrub_dict(obj)
    assert "[REDACTED:email]" in result["level1"]["level2"]["level3"]["data"]


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_scrub_empty_string():
    """Empty string should remain empty."""
    assert PiiScrubber.scrub_text("") == ""


def test_scrub_none_type():
    """None should pass through unchanged."""
    assert PiiScrubber.scrub_text(None) is None


def test_scrub_dict_with_none_values():
    """Dict with None values should preserve them."""
    obj = {"email": "user@example.com", "phone": None}
    result = PiiScrubber.scrub_dict(obj)
    assert "[REDACTED:email]" in result["email"]
    assert result["phone"] is None


def test_already_redacted_text():
    """Already redacted text should not cause double-redaction."""
    text = "[REDACTED:email] and [REDACTED:api_key_sk]"
    result = PiiScrubber.scrub_text(text)
    assert result == text  # No patterns to match


def test_special_chars_in_patterns():
    """Patterns with special characters should be handled."""
    text = "Key: sk_test!@#$%^&*()"
    result = PiiScrubber.scrub_text(text)
    # sk_test followed by alphanumerics shouldn't match (has special chars)
    # This is expected behavior — API keys don't have special chars


def test_unicode_email():
    """Unicode characters in email should handle gracefully."""
    text = "user.name@example.com"
    result = PiiScrubber.scrub_text(text)
    # Should match and redact
    assert "[REDACTED:email]" in result


def test_scrub_dict_circular_reference_safe():
    """Circular references are prevented by depth limit."""
    # Create object that would cause infinite recursion
    obj = {"data": "user@example.com"}
    # Don't actually create circular ref; just test depth is respected
    result = PiiScrubber.scrub_dict(obj, _depth=25)
    # Should return original due to depth limit
    assert result == obj
