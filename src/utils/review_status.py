from typing import Optional

_KNOWN_STATUSES = {
    "APPROVED", "APPROVE_WITH_WARNINGS", "NEEDS_IMPROVEMENT",
    "REJECTED", "FAIL", "CRASH",
}

_ALIASES = {
    "RESOLVED": "APPROVE_WITH_WARNINGS",
    "PASSED": "APPROVED",
    "FAILED": "REJECTED",
}


def normalize_status(status: Optional[str]) -> str:
    if not status:
        return "APPROVE_WITH_WARNINGS"
    upper = str(status).strip().upper()
    if upper in _KNOWN_STATUSES:
        return upper
    if upper in _ALIASES:
        return _ALIASES[upper]
    return "APPROVE_WITH_WARNINGS"
