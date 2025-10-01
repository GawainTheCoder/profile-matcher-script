"""
Utility functions for text processing and URL handling.
"""
import json
import re
import time
from typing import Dict, Any, Optional, List


def append_log(log_handle, payload: Dict[str, Any]) -> None:
    """Append JSON payload to log file handle."""
    if not log_handle:
        return
    try:
        log_handle.write(json.dumps(payload, ensure_ascii=False))
        log_handle.write("\n")
    except Exception:
        # Logging must not break main flow
        pass


def norm(s: str) -> str:
    """Normalize string by stripping whitespace."""
    return (s or "").strip()


def truncate_text(text: str, limit: int = 400) -> str:
    """Truncate text to specified limit with ellipsis."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def canonicalize_linkedin_url(url: str) -> str:
    """Canonicalize LinkedIn URL format."""
    url = (url or "").strip()
    if not url:
        return url
    url = re.sub(r"\s", "", url)
    if not url.lower().startswith("http"):
        url = "https://" + url.lstrip("/")
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    url = url.split("?", 1)[0].split("#", 1)[0]
    if url.endswith("/"):
        url = url[:-1]
    return url


def extract_cctld(url: str) -> str:
    """Extract country code TLD from LinkedIn URL."""
    try:
        m = re.search(r"https?://([a-z]{2,3})\\.linkedin\\.com/", (url or "").lower())
        return m.group(1).upper() if m else ""
    except Exception:
        return ""


def _is_close_token(candidate: str, target: str) -> bool:
    """Check if two tokens are close (edit distance <= 1)."""
    if candidate == target:
        return True
    len_a, len_b = len(candidate), len(target)
    if abs(len_a - len_b) > 1:
        return False
    if len_a == len_b:
        mismatches = sum(1 for ca, cb in zip(candidate, target) if ca != cb)
        return mismatches <= 1
    # Ensure candidate is the longer string
    if len_a < len_b:
        candidate, target = target, candidate
        len_a, len_b = len_b, len_a
    # Now len(candidate) == len(target) + 1
    i = j = 0
    mismatches = 0
    while i < len_a and j < len_b:
        if candidate[i] == target[j]:
            i += 1
            j += 1
        else:
            mismatches += 1
            if mismatches > 1:
                return False
            i += 1
    return True


def close_name_match(name: str, text: str, slug_tokens: Optional[List[str]] = None) -> bool:
    """Check if name appears in text with close matching."""
    if not name:
        return False
    name_lc = name.lower()
    if slug_tokens:
        for token in slug_tokens:
            if _is_close_token(token.lower(), name_lc):
                return True
    tokens = re.findall(r"[a-z]+", text.lower())
    for token in tokens:
        if len(token) >= max(3, len(name_lc) - 1) and _is_close_token(token, name_lc):
            return True
    return False


def is_linkedin_profile(url: str) -> bool:
    """Check if URL is a LinkedIn profile."""
    try:
        from .models import LINKEDIN_DOMAINS
    except ImportError:
        from models import LINKEDIN_DOMAINS
    u = (url or "").lower()
    return any(dom in u for dom in LINKEDIN_DOMAINS)


def pick_confidence(score: int, accept: int, review: int) -> str:
    """Pick confidence level based on score and thresholds."""
    if score >= accept:
        return "High"
    if score >= review:
        return "Medium"
    return "Low"