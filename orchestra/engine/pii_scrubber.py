"""PiiScrubber — Redact sensitive patterns from text/dict before caching or logging."""
from __future__ import annotations

import re
import math
from typing import Any
from collections import Counter


class PiiScrubber:
    """Redact sensitive patterns (API keys, emails, JWTs, etc) from text and dicts.

    Uses regex-based pattern matching with entropy-based filtering to prevent
    false positives on Base64/hashes while catching common API keys and tokens.

    Follows immutable patterns: all methods return new objects, never mutate input.
    """

    # Compiled regex patterns for PII detection
    _PATTERNS = {
        "api_key_sk": re.compile(r"sk[_-][a-zA-Z0-9_-]{15,}"),
        "api_key_pk": re.compile(r"pk[_-][a-z0-9_-]{25,}"),
        "jwt": re.compile(r"eyJ[a-zA-Z0-9_.-]{20,}"),
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "phone": re.compile(r"(?<![a-zA-Z_])\+?1?\s*\(?[0-9]{3}\)?\s*[-]?[0-9]{3}\s*[-]?[0-9]{4}(?![a-zA-Z0-9])"),
        "ssn": re.compile(r"(?<![a-zA-Z_])\d{3}[\-\s]?\d{2}[\-\s]?\d{4}(?![a-zA-Z0-9])"),
        "credit_card": re.compile(r"(?<![a-zA-Z_])\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?![a-zA-Z0-9])"),
        "oauth_token": re.compile(r"oauth[2_-][a-z0-9_-]{35,}"),
        "aws_key": re.compile(r"AKIA[0-9A-Z]{16}\b"),
    }

    # Whitelist patterns for known safe strings
    _WHITELIST_BASE64_PREFIXES = {"iVB", "/9j", "GIF89a", "ffd8ff"}
    _WHITELIST_HEX_PATTERN = re.compile(r"^[a-f0-9]{40,}$")

    @staticmethod
    def compute_entropy(text: str) -> float:
        """Compute Shannon entropy of a string on 0-8 scale.

        Used to identify and whitelist high-entropy strings like hashes
        and Base64 that might match PII patterns but are actually safe.

        Args:
            text: String to analyze

        Returns:
            Entropy value 0-8 (8 = maximum randomness)
        """
        if not text:
            return 0.0

        counts = Counter(text)
        total = len(text)
        entropy = 0.0

        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    @staticmethod
    def _should_redact(matched_text: str, pattern_type: str) -> bool:
        """Determine if a matched pattern should be redacted.

        Applies entropy filtering to avoid false positives on Base64/hashes
        while ensuring API keys are always redacted.

        Args:
            matched_text: The matched string
            pattern_type: Type of pattern matched (api_key_sk, email, etc)

        Returns:
            True if should redact, False to keep unchanged
        """
        # Always redact explicit API keys, emails, and PII numbers
        if pattern_type in {"api_key_sk", "api_key_pk", "email", "oauth_token", "aws_key", "phone", "ssn", "credit_card"}:
            return True

        # For other patterns, apply entropy checks
        if pattern_type == "jwt":
            # JWTs are always redacted (high entropy + eyJ prefix is very specific)
            return True

        # Check whitelist patterns for Base64 image prefixes
        if any(matched_text.startswith(prefix) for prefix in PiiScrubber._WHITELIST_BASE64_PREFIXES):
            return False

        # Check if matches hex hash pattern (lowercase hex, 40+ chars)
        if PiiScrubber._WHITELIST_HEX_PATTERN.match(matched_text):
            return False

        # Apply entropy threshold for remaining patterns
        entropy = PiiScrubber.compute_entropy(matched_text)
        entropy_threshold = 5.0

        return entropy >= entropy_threshold

    @staticmethod
    def scrub_text(text: str) -> str:
        """Replace PII patterns with [REDACTED:type].

        Supported patterns:
        - API keys (sk-*, pk_*)
        - JWTs (eyJ...)
        - Emails
        - Phone numbers
        - SSNs
        - Credit cards
        - OAuth tokens
        - AWS keys

        Args:
            text: String to scrub

        Returns:
            Scrubbed string with patterns replaced
        """
        if not text or not isinstance(text, str):
            return text

        result = text

        # Apply patterns in order, tracking offsets to maintain consistency
        matches = []

        for pattern_type, pattern in PiiScrubber._PATTERNS.items():
            for match in pattern.finditer(result):
                matches.append((match.start(), match.end(), pattern_type, match.group()))

        # Sort by position (reverse) to replace from end to start
        # This prevents offset shifting when making replacements
        matches.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate overlapping matches (keep the one detected first)
        filtered_matches = []
        for start, end, pattern_type, matched_text in matches:
            # Skip if this position was already matched (overlaps with existing)
            overlap = False
            for existing_start, existing_end, _, _ in filtered_matches:
                if not (end <= existing_start or start >= existing_end):
                    overlap = True
                    break

            if not overlap:
                if PiiScrubber._should_redact(matched_text, pattern_type):
                    filtered_matches.append((start, end, pattern_type, matched_text))

        # Apply replacements from end to start (to preserve earlier indices)
        for start, end, pattern_type, _ in filtered_matches:
            replacement = f"[REDACTED:{pattern_type}]"
            result = result[:start] + replacement + result[end:]

        return result

    @staticmethod
    def scrub_dict(obj: dict | Any, _depth: int = 0) -> dict | Any:
        """Recursively scrub all string values in dict.

        Follows immutable pattern: returns new dict with scrubbed values,
        never mutates input.

        Handles:
        - Nested dicts
        - Lists of dicts
        - Strings at any depth
        - Mixed types (non-dicts/lists pass through unchanged)

        Args:
            obj: Dict or value to scrub
            _depth: Internal depth tracker (max 20)

        Returns:
            New dict/object with scrubbed values
        """
        # Prevent infinite recursion
        if _depth > 20:
            return obj

        if not isinstance(obj, dict):
            if isinstance(obj, str):
                return PiiScrubber.scrub_text(obj)
            elif isinstance(obj, list):
                # Recursively scrub list items
                return [PiiScrubber.scrub_dict(item, _depth + 1) for item in obj]
            else:
                # Non-dict, non-list types pass through unchanged
                return obj

        # Create new dict with scrubbed values (immutable)
        scrubbed = {}
        for key, value in obj.items():
            if isinstance(value, str):
                scrubbed[key] = PiiScrubber.scrub_text(value)
            elif isinstance(value, dict):
                scrubbed[key] = PiiScrubber.scrub_dict(value, _depth + 1)
            elif isinstance(value, list):
                scrubbed[key] = [PiiScrubber.scrub_dict(item, _depth + 1) for item in value]
            else:
                # Preserve non-string/dict/list types unchanged
                scrubbed[key] = value

        return scrubbed

    @staticmethod
    def scrub_sensitive_fields(obj: dict, field_names: list[str]) -> dict:
        """Scrub specific named fields in a dict (immutable).

        Useful for quickly redacting known sensitive fields like 'api_key',
        'password', 'token' without full pattern matching.

        Args:
            obj: Dict to scrub
            field_names: List of field names to redact entirely

        Returns:
            New dict with specified fields redacted
        """
        if not isinstance(obj, dict):
            return obj

        scrubbed = dict(obj)
        for field in field_names:
            if field in scrubbed:
                scrubbed[field] = "[REDACTED]"

        return scrubbed
