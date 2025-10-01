"""
Candidate scoring and matching logic for LinkedIn profiles.
"""
import re
from typing import Dict, List, Tuple
try:
    from .models import RowFeatures, GENERIC_NAME_TOKENS
    from .utils import close_name_match
except ImportError:
    from models import RowFeatures, GENERIC_NAME_TOKENS
    from utils import close_name_match


def parse_candidate_last_initial(item: Dict[str, str]) -> str:
    """Best-effort extraction of last-name initial from SERP result title or URL.
    Returns uppercase initial or empty string if unknown or ambiguous.
    """
    title = (item.get("title") or "").strip()
    url = (item.get("link") or "").strip().lower()

    # Try URL slug first: /in/first-last-...
    m = re.search(r"linkedin\.com/(?:in|pub)/([^/?#]+)", url)
    if m:
        slug = m.group(1)
        pieces = [p for p in slug.split("-") if p]
        if len(pieces) >= 2 and pieces[-1].isalpha():
            last_piece = pieces[-1]
            last_lower = last_piece.lower()
            if last_lower not in GENERIC_NAME_TOKENS and not any(token in last_lower for token in ["design", "designer", "web", "landing", "studio", "agency", "growth", "media", "marketing", "page"]):
                return last_piece[0].upper()

    # Fallback: try to parse from SERP title before a dash or pipe
    head = title.split("|")[0]
    head = head.split(" - ")[0].strip()
    tokens = [t for t in re.split(r"\s+", head) if t]
    if len(tokens) >= 2 and tokens[-1].isalpha():
        return tokens[-1][0].upper()
    return ""


def extract_candidate_slug_tokens(url: str) -> List[str]:
    """Extract slug tokens from LinkedIn URL."""
    try:
        m = re.search(r"linkedin\.com/(?:in|pub)/([^/?#]+)", (url or "").lower())
        if not m:
            return []
        slug = m.group(1)
        return [p for p in re.split(r"[-_]+", slug) if p]
    except Exception:
        return []


def score_candidate(item: Dict[str, str], features: RowFeatures) -> Tuple[int, List[str]]:
    """Additive scoring using title/snippet evidence with signal tracking."""
    title_text = item.get("title", "")
    snippet = item.get("snippet", "")
    txt = f"{title_text} {snippet}".lower()
    url_txt = (item.get("link", "") or "").lower()
    signals: List[str] = []
    score = 0

    # Name guard: allow matches if first name appears in snippet/title or URL
    fn = features.first_name
    last_initial = features.last_initial
    fn_lc = (fn or "").strip().lower()
    slug_tokens = extract_candidate_slug_tokens(url_txt)
    fn_in_slug = fn_lc in slug_tokens if fn_lc else False
    if fn_lc:
        name_in_text = fn_lc in txt
        name_in_url = bool(re.search(rf"/(?:in|pub)/[^/]*{re.escape(fn_lc)}", url_txt) or fn_in_slug)
        approximate_name = close_name_match(fn_lc, txt, slug_tokens)
        if not name_in_text and not name_in_url and not approximate_name:
            return -999, ["reject:no-first-name"]

    # Last initial enforcement when confident
    cand_last_initial = parse_candidate_last_initial(item)
    if last_initial and cand_last_initial:
        if cand_last_initial != last_initial:
            return -999, ["reject:last-initial-mismatch"]
        score += 5
        signals.append("last_initial")

    city = (features.city or "").strip().lower()
    country = (features.country or "").strip().lower()
    alt_locations = [loc.lower() for loc in features.location_tokens]
    if city and city in txt:
        score += 5
        signals.append("city")
    elif country and country in txt:
        score += 3
        signals.append("country")
    else:
        for alt in alt_locations:
            if alt and alt in txt:
                score += 3
                signals.append("alt_location")
                break

    # Role/title phrases
    title_phrases = [p.lower() for p in features.title_phrases]
    if any(p in txt for p in title_phrases):
        score += 4
        signals.append("title_phrase")

    skills = [sk.lower() for sk in features.top_skills]
    if any(sk in txt for sk in skills):
        score += 3
        signals.append("skill")

    edu_hits = [ed.lower() for ed in features.schools]
    if any(ed in txt for ed in edu_hits):
        score += 3
        signals.append("education")

    # Company evidence
    for comp in [c.lower() for c in features.companies]:
        if comp and comp in txt:
            score += 3
            signals.append("company")
            break

    desc_hits = [d.lower() for d in features.description_phrases]
    if any(d in txt for d in desc_hits):
        score += 2
        signals.append("description")

    cert_hits = [c.lower() for c in features.certifications]
    if any(cert in txt for cert in cert_hits):
        score += 2
        signals.append("certification")

    if re.search(r"\b\d{1,2}\+?\s+(?:years|yrs)\b", txt):
        score += 2
        signals.append("experience")

    if any(token in txt for token in ["upwork", "freelance", "consultant"]):
        score += 3
        signals.append("freelance")

    return score, signals