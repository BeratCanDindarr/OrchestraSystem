"""Optional adapters for external orchestration signals."""
from orchestra.adapters.continuous_learning import (
    get_failure_patterns,
    is_available,
    suggest_mode_adjustment,
)

__all__ = ["get_failure_patterns", "is_available", "suggest_mode_adjustment"]
