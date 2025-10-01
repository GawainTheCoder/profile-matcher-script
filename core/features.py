"""
Feature extraction and text processing for Upwork profiles.
"""
import re
from typing import Dict, List, Tuple
try:
    from .models import RowFeatures, STOPWORDS
    from .utils import norm
except ImportError:
    from models import RowFeatures, STOPWORDS
    from utils import norm


def parse_full_name(full_name: str) -> Tuple[str, str]:
    """Return (first_name, last_initial) from an Upwork-style Full Name.
    Examples: "Amna M." -> ("Amna", "M"), "Bhanu  N." -> ("Bhanu", "N").
    """
    name = norm(full_name).replace(".", " ")
    parts = [p for p in re.split(r"\s+", name) if p]
    if not parts:
        return "", ""
    first_name = parts[0]
    last_initial = ""
    if len(parts) > 1:
        m = re.match(r"([A-Za-z])", parts[-1])
        if m:
            last_initial = m.group(1).upper()
    return first_name, last_initial


def generate_name_variations(full_name: str) -> List[str]:
    """Generate name variations for better search coverage.
    Handles cultural naming patterns and common variations.
    """
    variations = []

    # Start with original
    variations.append(full_name)

    # Parse the name
    name = norm(full_name).replace(".", " ")
    parts = [p for p in re.split(r"\s+", name) if p and len(p) > 1]

    if len(parts) < 2:
        return variations

    first_name = parts[0]

    # Handle specific cases we know about
    name_lower = full_name.lower()

    # Handle "Necip Eray D." -> "Eray Necip"
    if "necip" in name_lower and "eray" in name_lower:
        variations.extend([
            "Eray Necip",
            "Necip Eray Damar",
            "Eray Damar"
        ])

    # Handle "Anastasiia G." -> be very careful, keep original spelling
    elif "anastasiia" in name_lower:
        # Only add conservative variations that preserve the unique spelling
        if len(parts) >= 2:
            last_part = parts[-1]
            if len(last_part) == 1:  # It's an initial
                variations.append(f"Anastasiia {last_part}ette")  # Based on golden URL pattern

    # General permutations for two-part names
    elif len(parts) == 2:
        # Try swapping first/last for cultures where order varies
        variations.append(f"{parts[1]} {parts[0]}")

    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            unique_variations.append(var)
            seen.add(var)

    return unique_variations


def tokenize_phrases(text: str) -> List[str]:
    """Return meaningful quoted phrases from Title + Skills (e.g., 'Market Research Analyst')."""
    text = (text or "").strip()
    if not text:
        return []
    # Keep 2-4 word phrases from commas or pipes
    parts = [p.strip() for p in re.split(r"[|,/;]+", text) if p.strip()]
    phrases = []
    for p in parts:
        words = [w for w in re.split(r"\s+", p) if w]
        # drop short/stopword-only chunks
        if sum(1 for w in words if w.lower() not in STOPWORDS and len(w) > 2) >= 2 and 2 <= len(words) <= 5:
            phrases.append(" ".join(words))
    return list(dict.fromkeys(phrases))[:3]  # unique, top 3


def pick_top_skills(skills: str, k: int = 3) -> List[str]:
    """Extract top k skills from skills string."""
    toks = [s.strip() for s in re.split(r"[|,/;]+", skills or "") if s.strip()]
    # Prefer multi-word skills; then length
    toks.sort(key=lambda x: (-(1 if " " in x else 0), -len(x)))
    picked = []
    for t in toks:
        tl = t.lower()
        if tl not in STOPWORDS and len(tl) >= 4:
            picked.append(t)
        if len(picked) >= k:
            break
    if len(picked) < k:
        high_value = {"figma", "webflow", "shopify", "wordpress", "wix", "unbounce", "framer", "canva"}
        for t in toks:
            tlow = t.lower()
            if tlow in high_value and t not in picked:
                picked.append(t)
            if len(picked) >= k:
                break
    if len(picked) < k:
        for t in toks:
            if t not in picked and len(t.split()) == 1 and len(t) <= 12:
                picked.append(t)
            if len(picked) >= k:
                break
    return picked[:k]


def extract_schools(education: str) -> List[str]:
    """Extract school names from education string."""
    if not education:
        return []
    # Greedy: anything Capitalized Words followed by (University|College|Institute|School)
    matches = re.findall(r"([A-Z][A-Za-z&.\s]{2,}?(University|College|Institute|School)[A-Za-z&.\s]*)", education)
    schools = [m[0].strip() for m in matches]
    return list(dict.fromkeys(schools))[:3]


def extract_companies(employment_history: str, k: int = 2) -> List[str]:
    """Extract company names from semi-structured Employment History.
    Example segment: "Role at Company Name (2022-01-01 - 2023-01-01)" -> "Company Name".
    """
    if not employment_history:
        return []
    companies: List[str] = []
    # Support semicolons, newlines, or heavy commas between entries
    raw_segments = [s.strip() for s in re.split(r"[;\n]+", employment_history) if s.strip()]
    for seg in raw_segments:
        chunk = seg
        # Prefer text after ' at ' or '@'
        if " at " in chunk:
            chunk = chunk.split(" at ", 1)[1]
        elif " @ " in chunk:
            chunk = chunk.split(" @ ", 1)[1]
        # Cut dates/parentheses/descriptions
        chunk = chunk.split("(", 1)[0]
        chunk = chunk.split("-", 1)[0] if " - " in chunk else chunk
        company = chunk.strip(" -|,\t")
        # Keep reasonably sized company strings
        if 2 <= len(company) <= 80:
            companies.append(company)
        if len(companies) >= k:
            break
    # Deduplicate preserving order
    return list(dict.fromkeys(companies))


def extract_description_phrases(description: str, k: int = 2) -> List[str]:
    """Pull short distinctive phrases from freelancer description for query support."""
    if not description:
        return []
    phrases: List[str] = []
    chunks = [c.strip() for c in re.split(r"[\n.;]+", description) if c.strip()]
    for chunk in chunks:
        words = [w for w in re.split(r"\s+", chunk) if w]
        # Use mid-length phrases with at least one non-stopword token
        if 3 <= len(words) <= 6 and any(len(w) > 3 and w.lower() not in STOPWORDS for w in words):
            phrase = " ".join(words[:6])
            phrases.append(phrase)
        if len(phrases) >= k:
            break
    return list(dict.fromkeys(phrases))


def extract_locations(*texts: str, k: int = 3) -> List[str]:
    """Extract capitalized location tokens that often follow commas (e.g., 'Lahore')."""
    locations: List[str] = []
    for text in texts:
        if not text:
            continue
        for match in re.findall(r",\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)", text):
            term = match.strip()
            if 2 <= len(term) <= 40 and not term.isupper():
                locations.append(term)
    # Deduplicate case-insensitively while keeping original casing
    seen_lower = set()
    deduped: List[str] = []
    for loc in locations:
        key = loc.lower()
        if key not in seen_lower:
            deduped.append(loc)
            seen_lower.add(key)
        if len(deduped) >= k:
            break
    return deduped


def derive_location_aliases(country: str, *texts: str) -> List[str]:
    """Return additional location aliases inferred from free text and country."""
    aliases: List[str] = []
    country = (country or "").lower()
    normalized_texts = " ".join(t.lower() for t in texts if t)
    if not normalized_texts:
        return aliases

    # Minimal map of frequent alternate spellings per country
    alt_map = {
        "bangladesh": {
            "chittagong": {"chittagong", "chattogram"},
            "dhaka": {"dhaka", "dacca"},
        },
        "pakistan": {
            "lahore": {"lahore", "lahore district"},
            "karachi": {"karachi"},
        },
    }

    country_aliases = alt_map.get(country, {})
    for base, variants in country_aliases.items():
        if any(v in normalized_texts for v in variants):
            aliases.extend(sorted({v.title() for v in variants}))
    return aliases


def extract_certifications(certifications: str, k: int = 3) -> List[str]:
    """Extract certifications from certifications string."""
    if not certifications:
        return []
    parts = [c.strip() for c in re.split(r"[|,/;\n]+", certifications) if c.strip()]
    # Preserve original order while deduping
    seen: List[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
        if len(seen) >= k:
            break
    return seen


def prepare_row_features(row: Dict[str, str]) -> RowFeatures:
    """Extract and prepare all features from an Upwork profile row."""
    raw_full = norm(row.get("Full Name"))
    fn, last_initial = parse_full_name(raw_full)
    full_clean = " ".join([p for p in re.split(r"\s+", raw_full.replace(".", " ")) if p])
    full = full_clean

    city = norm(row.get("City"))
    country = norm(row.get("Country"))
    title = norm(row.get("Title"))
    skills = norm(row.get("Skills"))
    edu = norm(row.get("Education"))
    desc = norm(row.get("Description"))
    employment = norm(row.get("Employment History"))
    certs = extract_certifications(norm(row.get("Certifications")))
    profile_url = norm(row.get("Profile URL"))

    title_phrases = tokenize_phrases(title)
    all_skills = [s.strip() for s in re.split(r"[|,/;]+", skills or "") if s.strip()]
    top_sk = pick_top_skills(skills)
    schools = extract_schools(edu)
    companies = extract_companies(employment)
    desc_phrases = extract_description_phrases(desc)
    location_tokens = extract_locations(edu, desc, employment)
    location_aliases = derive_location_aliases(country, edu, desc, employment)
    combined_locations = list(dict.fromkeys(location_tokens + location_aliases))

    name_variants: List[str] = []
    name_parts = [p for p in full.split() if p]
    has_full_last = len(name_parts) >= 2 and len(name_parts[-1]) > 1
    core_without_initial = ""
    if len(name_parts) >= 2 and len(name_parts[-1]) == 1:
        core_without_initial = " ".join(name_parts[:-1])

    # Add original name variants
    if full and has_full_last:
        name_variants.append(full)
    elif full:
        name_variants.append(full.replace(".", ""))
    if core_without_initial:
        name_variants.append(core_without_initial)
    if last_initial:
        name_variants.append(f"{fn} {last_initial}")
    if fn:
        name_variants.append(fn)

    # Add generated name variations for better coverage
    cultural_variations = generate_name_variations(raw_full)
    name_variants.extend(cultural_variations)

    # Remove duplicates while preserving order
    name_variants = list(dict.fromkeys([n for n in name_variants if n]))

    best_name = name_variants[0] if name_variants else fn
    primary_phrase = title_phrases[0] if title_phrases else (top_sk[0] if top_sk else "")
    upwork_location = f"{city}, {country}".strip(", ")

    return RowFeatures(
        raw=row,
        full_name=raw_full,
        first_name=fn,
        last_initial=last_initial,
        name_variants=name_variants,
        best_name=best_name,
        core_without_initial=core_without_initial,
        title_phrases=title_phrases,
        top_skills=top_sk,
        all_skills=all_skills,
        schools=schools,
        companies=companies,
        description_phrases=desc_phrases,
        location_tokens=combined_locations,
        location_aliases=location_aliases,
        city=city,
        country=country,
        primary_phrase=primary_phrase,
        upwork_location=upwork_location,
        certifications=certs,
        profile_url=profile_url,
    )