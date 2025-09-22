#!/usr/bin/env python3
"""
upwork_to_linkedin_matcher.py

Row-driven person-resolution from Upwork -> LinkedIn using a SERP provider
(SerpAPI or Serper). Builds queries from each row's fields, enforces last-initial,
scores SERP snippets, and can optionally rerank with OpenAI gpt-5-nano.

Usage (rule-based):
  python upwork_to_linkedin_matcher.py \
      --input latest_people.csv \
      --output linkedin_matches.csv \
      --provider serper \
      --max-queries 6 \
      --results-per-query 5 \
      --min-score 3

Usage (with LLM reranker):
  python upwork_to_linkedin_matcher.py \
      --input latest_people.csv \
      --output latest_people_llm_all.csv \
      --provider serper \
      --max-queries 6 \
      --results-per-query 5 \
      --llm-rerank \
      --llm-model gpt-5-nano-2025-08-07 \
      --llm-top-k 5 \
      --llm-keep-threshold 0.6

Notable flags:
  --no-require-role-signal   Disable role/freelance must-have gate (default gate is ON)
  --accept-threshold 10      High confidence threshold (default 10)
  --review-threshold 7       Medium confidence threshold (default 7)

Env vars:
  SERP_PROVIDER=serper|serpapi  (or use --provider)
  SERPER_API_KEY=...            (for serper.dev)
  SERPAPI_API_KEY=...           (for serpapi.com)
  OPENAI_API_KEY=...            (required only with --llm-rerank)
  OPENAI_ORG_ID=...             (optional, for dashboard attribution)
  OPENAI_PROJECT_ID=...         (optional, for dashboard attribution)
"""
import os, re, csv, time, json, argparse, random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import requests

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(path: Optional[str] = None) -> bool:
        """Minimal .env loader for environments without python-dotenv."""
        env_path = path or os.path.join(os.getcwd(), ".env")
        if not os.path.exists(env_path):
            return False
        loaded = False
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
                loaded = True
        return loaded

# Load environment variables from .env file
load_dotenv()

# ----------------------------- Providers -----------------------------

class SearchProvider:
    def search(self, query: str, num: int = 5, hl: str = "en", gl: str = "us") -> List[Dict[str, str]]:
        """Return list of {link, title, snippet} items."""
        raise NotImplementedError

class SerpAPIProvider(SearchProvider):
    BASE = "https://serpapi.com/search"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("SERPAPI_API_KEY missing.")
        self.api_key = api_key

    def search(self, query: str, num: int = 5, hl: str = "en", gl: str = "us") -> List[Dict[str, str]]:
        params = {
            "engine": "google",
            "q": query,
            "num": num,
            "hl": hl,
            "gl": gl,
            "api_key": self.api_key,
            # Keep payload small; see SerpAPI json_restrictor docs
            "json_restrictor": "organic_results[].{link,title,snippet}"
        }
        resp = requests.get(self.BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in (data.get("organic_results") or []):
            link = item.get("link"); title = item.get("title"); snippet = item.get("snippet")
            if link and title:
                results.append({"link": link, "title": title, "snippet": snippet or ""})
        return results

class SerperProvider(SearchProvider):
    URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("SERPER_API_KEY missing.")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        })

    def search(self, query: str, num: int = 5, hl: str = "en", gl: str = "us") -> List[Dict[str, str]]:
        payload = {"q": query, "num": num, "hl": hl, "gl": gl}
        resp = self.session.post(self.URL, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in (data.get("organic") or []):
            link = item.get("link"); title = item.get("title"); snippet = item.get("snippet", "")
            if link and title:
                results.append({"link": link, "title": title, "snippet": snippet})
        return results

# ------------------------- Utilities & Scoring ------------------------

LINKEDIN_DOMAINS = ("linkedin.com/in", "linkedin.com/pub")

STOPWORDS = set("""
a an and are as at be by for from has have i in is it of on or that the to with your you
""".split())


@dataclass
class RowFeatures:
    raw: Dict[str, str]
    full_name: str
    first_name: str
    last_initial: str
    name_variants: List[str]
    best_name: str
    core_without_initial: str
    title_phrases: List[str]
    top_skills: List[str]
    schools: List[str]
    companies: List[str]
    description_phrases: List[str]
    location_tokens: List[str]
    location_aliases: List[str]
    city: str
    country: str
    primary_phrase: str
    upwork_location: str
    certifications: List[str]
    profile_url: str
    llm_decision: Optional[Dict[str, Any]] = None


@dataclass
class CandidateEvidence:
    candidate_id: str
    score: int
    signals: List[str]
    raw_item: Dict[str, str]
    query: str
    rule_rank: int
    url: str
    slug_tokens: List[str] = field(default_factory=list)
    derived_last_initial: str = ""
    first_name_in_url: bool = False
    rejection_reason: Optional[str] = None
    llm_selected: bool = False
    llm_confidence: Optional[float] = None
    llm_rationale: Optional[str] = None
    llm_rank: Optional[int] = None
    llm_reject_reason: Optional[str] = None

def append_log(log_handle, payload: Dict[str, Any]) -> None:
    if not log_handle:
        return
    try:
        log_handle.write(json.dumps(payload, ensure_ascii=False))
        log_handle.write("\n")
    except Exception:
        # Logging must not break main flow
        pass

def norm(s: str) -> str:
    return (s or "").strip()


def truncate_text(text: str, limit: int = 400) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."

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
    return picked

def extract_schools(education: str) -> List[str]:
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
def parse_candidate_last_initial(item: Dict[str, str]) -> str:
    """Best-effort extraction of last-name initial from SERP result title or URL.
    Returns uppercase initial or empty string if unknown or ambiguous."""
    title = (item.get("title") or "").strip()
    url = (item.get("link") or "").strip().lower()

    # Try URL slug first: /in/first-last-...
    m = re.search(r"linkedin\.com/(?:in|pub)/([^/?#]+)", url)
    if m:
        slug = m.group(1)
        pieces = [p for p in slug.split("-") if p]
        if len(pieces) >= 2 and pieces[-1].isalpha():
            return pieces[-1][0].upper()

    # Fallback: try to parse from SERP title before a dash or pipe
    head = title.split("|")[0]
    head = head.split(" - ")[0].strip()
    tokens = [t for t in re.split(r"\s+", head) if t]
    if len(tokens) >= 2 and tokens[-1].isalpha():
        return tokens[-1][0].upper()
    return ""


def extract_candidate_slug_tokens(url: str) -> List[str]:
    try:
        m = re.search(r"linkedin\.com/(?:in|pub)/([^/?#]+)", (url or "").lower())
        if not m:
            return []
        slug = m.group(1)
        return [p for p in re.split(r"[-_]+", slug) if p]
    except Exception:
        return []

def build_queries(features: RowFeatures, max_queries: int = 6) -> List[str]:
    city = features.city
    country = features.country
    title = norm(features.raw.get("Title"))
    schools = features.schools
    top_sk = features.top_skills
    title_phrases = features.title_phrases
    desc_phrases = features.description_phrases
    companies = features.companies
    name_variants = features.name_variants
    best_name = features.best_name
    fn = features.first_name
    location_hint = city or country
    primary_phrase = features.primary_phrase
    core_without_initial = features.core_without_initial
    location_tokens = features.location_tokens

    ordered_queries: List[str] = []

    def add_query(q: Optional[str]) -> None:
        if q and q.strip():
            ordered_queries.append(q.strip())

    # Tier 1: Full name pairings first (location, education, skills, title)
    primary_names: List[str] = []
    if best_name:
        primary_names.append(best_name)
    for variant in name_variants:
        if variant not in primary_names and len(variant.split()) >= 2:
            primary_names.append(variant)
    for name_opt in primary_names:
        if city:
            add_query(f'site:linkedin.com/in "{name_opt}" "{city}"')
        if country and country.lower() != (city or "").lower():
            add_query(f'site:linkedin.com/in "{name_opt}" "{country}"')
        if schools:
            add_query(f'site:linkedin.com/in "{name_opt}" "{schools[0]}"')
        if top_sk:
            add_query(f'site:linkedin.com/in "{name_opt}" "{top_sk[0]}"')
        if title:
            add_query(f'site:linkedin.com/in "{name_opt}" "{title}"')

    # Tier 2: Golden query - name + strongest role phrase + location
    if best_name and primary_phrase and location_hint:
        add_query(f'site:linkedin.com/in "{best_name}" "{primary_phrase}" "{location_hint}"')
        if city and country and city.lower() != country.lower():
            add_query(f'site:linkedin.com/in "{best_name}" "{primary_phrase}" "{city}" "{country}"')

    # Tier 2: Pair name with individual strong attributes
    secondary_names = name_variants[:2] if len(name_variants) >= 2 else name_variants
    paired_attributes: List[str] = []
    if city:
        paired_attributes.append(city)
    if country and country not in paired_attributes:
        paired_attributes.append(country)
    paired_attributes.extend(location_tokens[:2])
    paired_attributes.extend(title_phrases[:2])
    paired_attributes.extend(top_sk[:2])
    paired_attributes.extend(schools[:2])
    paired_attributes.extend(companies[:2])
    paired_attributes.extend(desc_phrases[:2])

    for name_val in secondary_names:
        for attr in paired_attributes:
            add_query(f'site:linkedin.com/in "{name_val}" "{attr}"')

    # Tier 3: Guards and Upwork persona hints
    if best_name:
        add_query(f'site:linkedin.com/in intitle:"{best_name}"')
    if fn and fn != best_name:
        add_query(f'site:linkedin.com/in intitle:"{fn}"')

    persona_name = best_name or fn
    if persona_name:
        if city:
            add_query(f'"{persona_name}" "Upwork" "{city}" site:linkedin.com/in')
        add_query(f'"{persona_name}" "Upwork" site:linkedin.com/in')

    # Final fallback: first-name combinations for broad recall
    if fn and primary_phrase:
        add_query(f'site:linkedin.com/in "{fn}" "{primary_phrase}"')
    if fn and location_hint:
        add_query(f'site:linkedin.com/in "{fn}" "{location_hint}"')
    for loc in location_tokens[:2]:
        if fn:
            add_query(f'site:linkedin.com/in "{fn}" "{loc}"')

    # If we trimmed a trailing initial, run focused queries on the trimmed core name
    if core_without_initial:
        trimmed = core_without_initial
        if city:
            add_query(f'site:linkedin.com/in "{trimmed}" "{city}"')
        if country and country.lower() != (city or "").lower():
            add_query(f'site:linkedin.com/in "{trimmed}" "{country}"')
        if schools:
            add_query(f'site:linkedin.com/in "{trimmed}" "{schools[0]}"')
        if top_sk:
            add_query(f'site:linkedin.com/in "{trimmed}" "{top_sk[0]}"')

    queries: List[str] = []
    seen = set()
    for q in ordered_queries:
        if q not in seen:
            queries.append(q)
            seen.add(q)
        if len(queries) >= max_queries:
            break
    return queries

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
        if not name_in_text and not name_in_url:
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

def is_linkedin_profile(url: str) -> bool:
    u = (url or "").lower()
    return any(dom in u for dom in LINKEDIN_DOMAINS)

def pick_confidence(score: int, accept: int, review: int) -> str:
    if score >= accept: return "High"
    if score >= review: return "Medium"
    return "Low"

# ---------------------------- LLM Reranker -----------------------------


def build_llm_match_payload(features: RowFeatures, candidates: List[CandidateEvidence]) -> Dict[str, Any]:
    upwork_profile = {
        "full_name": features.full_name,
        "first_name": features.first_name,
        "last_initial": features.last_initial,
        "title": norm(features.raw.get("Title")),
        "skills_text": norm(features.raw.get("Skills"))[:400],
        "top_skills": features.top_skills,
        "city": features.city,
        "country": features.country,
        "location_tokens": features.location_tokens[:5],
        "name_variants": features.name_variants[:5],
        "primary_phrase": features.primary_phrase,
        "schools": features.schools,
        "companies": features.companies,
        "description_phrases": features.description_phrases,
        "certifications": features.certifications,
        "profile_url": features.profile_url,
        "description_excerpt": truncate_text(norm(features.raw.get("Description")), 600),
        "employment_history_excerpt": truncate_text(norm(features.raw.get("Employment History")), 600),
        "education_excerpt": truncate_text(norm(features.raw.get("Education")), 400),
    }

    linkedin_candidates: List[Dict[str, Any]] = []
    for cand in candidates:
        item = cand.raw_item
        linkedin_candidates.append({
            "candidate_id": cand.candidate_id,
            "rule_score": cand.score,
            "rule_rank": cand.rule_rank,
            "url": cand.url,
            "title": item.get("title", ""),
            "snippet": truncate_text(item.get("snippet", ""), 500),
            "query": cand.query,
            "slug_tokens": cand.slug_tokens,
            "derived_last_initial": cand.derived_last_initial,
            "first_name_in_url": cand.first_name_in_url,
            "signals": [s for s in cand.signals if not s.startswith("reject")],
            "raw_signals": cand.signals,
        })

    return {
        "upwork_profile": upwork_profile,
        "linkedin_candidates": linkedin_candidates,
    }


def extract_cctld(url: str) -> str:
    try:
        m = re.search(r"https?://([a-z]{2,3})\\.linkedin\\.com/", (url or "").lower())
        return m.group(1).upper() if m else ""
    except Exception:
        return ""


def parse_responses_json(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    output = data.get("output")
    if isinstance(output, list):
        for block in output:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "message":
                contents = block.get("content")
                if isinstance(contents, list):
                    for item in contents:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") in {"output_text", "reasoning_text", "text"}:
                            text = item.get("text")
                            if isinstance(text, str):
                                try:
                                    return json.loads(text)
                                except json.JSONDecodeError:
                                    continue
            if block.get("type") == "json" and isinstance(block.get("json"), dict):
                return block["json"]
            text = block.get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue

    output_text = data.get("output_text")
    if isinstance(output_text, str):
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            pass

    content = data.get("content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                msg_content = message.get("content")
                if isinstance(msg_content, str):
                    try:
                        return json.loads(msg_content)
                    except json.JSONDecodeError:
                        continue
            text = choice.get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue

    return {}

def llm_rerank_candidates(
    features: RowFeatures,
    candidates: List[CandidateEvidence],
    model: str = "gpt-5-nano",
    keep_threshold: float = 0.6,
    top_k: int = 5,
    mode: str = "select",
) -> List[CandidateEvidence]:
    """Delegate final match selection to the Responses API.

    Returns candidates reordered according to the LLM decision. When the
    model declines the match or confidence is low, the original candidate
    ordering is returned as a fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not candidates:
        return candidates

    candidate_pool = sorted(candidates, key=lambda c: c.score, reverse=True)
    top_candidates = candidate_pool[:max(1, top_k)]

    payload_body = build_llm_match_payload(features, top_candidates)

    schema = {
        "type": "json_schema",
        "name": "match_decision",
        "schema": {
            "type": "object",
            "properties": {
                "best_candidate_id": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"},
                "secondary_candidate_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": []
                },
                "reject_reason": {"type": ["string", "null"], "default": None},
            },
            "required": [
                "best_candidate_id",
                "confidence",
                "rationale",
                "secondary_candidate_ids",
                "reject_reason",
            ],
            "additionalProperties": False,
        },
    }

    system_prompt = (
        "You are a sourcing analyst. Pick the LinkedIn profile that best matches the "
        "Upwork freelancer. Require the same first name (case-insensitive). Enforce last "
        "initial when Upwork provides one. Prefer evidence of matching roles, skills, "
        "companies, locations, certifications, and freelance context. Decline with a "
        "reject reason if none are suitable. Output JSON that follows the provided schema."
    )

    user_prompt = (
        "Upwork + LinkedIn evidence (JSON):\n" +
        json.dumps(payload_body, ensure_ascii=False)
    )

    request_payload = {
        "model": model,
        "max_output_tokens": 900,
        "text": {
            "format": schema,
        },
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "metadata": {
            "source": "upwork_to_linkedin_matcher",
            "row_city": features.city,
            "row_country": features.country,
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    org_id = os.getenv("OPENAI_ORG_ID", "").strip()
    project_id = os.getenv("OPENAI_PROJECT_ID", "").strip()
    if org_id:
        headers["OpenAI-Organization"] = org_id
    if project_id:
        headers["OpenAI-Project"] = project_id

    decision: Dict[str, Any] = {}
    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            data=json.dumps(request_payload),
            timeout=45,
        )
        resp.raise_for_status()
        raw_response = resp.json()
        decision = parse_responses_json(raw_response)
    except requests.HTTPError as exc:
        err_text = ""
        try:
            err_text = exc.response.text
        except Exception:
            err_text = ""
        features.llm_decision = {
            "error": str(exc),
            "error_body": err_text,
            "model": model,
            "request_payload": request_payload,
            "used_candidates": [c.candidate_id for c in top_candidates],
            "timestamp": time.time(),
        }
        return candidate_pool
    except Exception as exc:
        features.llm_decision = {
            "error": str(exc),
            "model": model,
            "request_payload": request_payload,
            "used_candidates": [c.candidate_id for c in top_candidates],
            "timestamp": time.time(),
        }
        return candidate_pool

    if not isinstance(decision, dict):
        decision = {}

    features.llm_decision = {
        "decision": decision,
        "model": model,
        "used_candidates": [c.candidate_id for c in top_candidates],
        "timestamp": time.time(),
        "raw_response": raw_response,
    }

    best_id = decision.get("best_candidate_id")
    confidence = float(decision.get("confidence", 0.0) or 0.0)
    rationale = decision.get("rationale") or ""
    secondary_ids = decision.get("secondary_candidate_ids") or []
    reject_reason = decision.get("reject_reason")

    candidate_lookup = {cand.candidate_id: cand for cand in candidate_pool}
    for cand in candidate_pool:
        cand.llm_selected = False
        cand.llm_confidence = None
        cand.llm_rationale = None
        cand.llm_rank = None
        cand.llm_reject_reason = None

    selected: Optional[CandidateEvidence] = None
    if best_id and best_id in candidate_lookup and confidence >= keep_threshold:
        selected = candidate_lookup[best_id]
        selected.llm_selected = True
        selected.llm_confidence = confidence
        selected.llm_rationale = rationale
        selected.llm_rank = 1

    if selected is None:
        # No confident pick -> optionally annotate reason and fall back
        if reject_reason:
            for cand in candidate_pool:
                cand.llm_reject_reason = reject_reason
        # Attach whatever confidence/rationale we got to top candidate for logging
        if best_id and best_id in candidate_lookup:
            candidate_lookup[best_id].llm_confidence = confidence
            candidate_lookup[best_id].llm_rationale = rationale
        return candidate_pool

    ordered: List[CandidateEvidence] = [selected]
    for idx, cid in enumerate(secondary_ids, start=2):
        cand = candidate_lookup.get(cid)
        if not cand or cand is selected:
            continue
        cand.llm_rank = idx
        cand.llm_rationale = cand.llm_rationale or rationale
        ordered.append(cand)

    for cand in candidate_pool:
        if cand not in ordered:
            ordered.append(cand)

    if mode == "select":
        return ordered[: max(1, len(secondary_ids) + 1)]
    return ordered

# ------------------------------- Runner ------------------------------


def validate_args(args) -> None:
    if args.sleep_min < 0 or args.sleep_max < 0:
        raise ValueError("Sleep intervals must be non-negative")
    if args.sleep_min > args.sleep_max:
        raise ValueError("--sleep-min cannot exceed --sleep-max")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Upwork CSV")
    ap.add_argument("--output", required=True, help="Path to write matches CSV")
    ap.add_argument("--provider", default=os.getenv("SERP_PROVIDER","serper"), choices=["serper","serpapi"])
    ap.add_argument("--accept-threshold", type=int, default=10)
    ap.add_argument("--review-threshold", type=int, default=7)
    ap.add_argument("--max-queries", type=int, default=6)
    ap.add_argument("--results-per-query", type=int, default=5)
    ap.add_argument("--hl", default="en")
    ap.add_argument("--gl", default="us")
    ap.add_argument("--sleep-min", type=float, default=0.8)
    ap.add_argument("--sleep-max", type=float, default=1.8)
    ap.add_argument("--min-score", type=int, default=3, help="Minimum score to include a candidate")
    ap.add_argument("--no-require-role-signal", action="store_true", help="Disable role/freelance must-have gate")
    ap.add_argument("--debug-serp", action="store_true", help="Log queries and raw SERP items for inspection")
    ap.add_argument("--query-log", help="Path to append JSONL diagnostics for queries and candidates")
    args = ap.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))

    if args.provider == "serper":
        provider = SerperProvider(os.getenv("SERPER_API_KEY",""))
    else:
        provider = SerpAPIProvider(os.getenv("SERPAPI_API_KEY",""))

    log_handle = open(args.query_log, "a", encoding="utf-8") if args.query_log else None

    try:
        with open(args.input, newline="", encoding="utf-8") as f, open(args.output, "w", newline="", encoding="utf-8") as out:
            reader = csv.DictReader(f)
            # Simplified output columns - only the essential ones
            output_fieldnames = [
                "upwork_name", "upwork_title", "upwork_location", "upwork_skills",
                "linkedin_url", "linkedin_title", "linkedin_snippet", 
                "match_score", "confidence", "matched_signals", "query_used",
                "llm_selected", "llm_confidence", "llm_rationale", "llm_rank"
            ]
            writer = csv.DictWriter(out, fieldnames=output_fieldnames)
            writer.writeheader()

            for row_idx, row in enumerate(reader, start=1):
                features = prepare_row_features(row)
                queries = build_queries(features, max_queries=args.max_queries)
                candidates: List[CandidateEvidence] = []
                seen_urls = set()
                strong_hit_found = False
                candidate_counter = 0

                append_log(log_handle, {
                    "event": "row_start",
                    "row_index": row_idx,
                    "upwork_name": features.full_name,
                    "query_count": len(queries),
                    "timestamp": time.time(),
                })

                for q in queries:
                    append_log(log_handle, {
                        "event": "query",
                        "row_index": row_idx,
                        "upwork_name": features.full_name,
                        "query": q,
                        "timestamp": time.time(),
                    })
                    if args.debug_serp:
                        print(f"[debug][row {row_idx}] query: {q}", flush=True)
                    try:
                        results = provider.search(q, num=args.results_per_query, hl=args.hl, gl=args.gl)
                    except Exception as e:
                        # keep going on transient errors
                        results = []
                        if args.debug_serp:
                            print(f"[debug][row {row_idx}] query error: {e}", flush=True)
                        append_log(log_handle, {
                            "event": "query_error",
                            "row_index": row_idx,
                            "upwork_name": norm(row.get("Full Name", "")),
                            "query": q,
                            "error": str(e),
                            "timestamp": time.time(),
                        })
                    for item in results:
                        url = item.get("link","")
                        if not is_linkedin_profile(url):
                            continue
                        if url in seen_urls:
                            append_log(log_handle, {
                                "event": "candidate_skipped",
                                "reason": "duplicate_url",
                                "row_index": row_idx,
                                "upwork_name": features.full_name,
                                "query": q,
                                "url": url,
                                "timestamp": time.time(),
                            })
                            continue
                        seen_urls.add(url)
                        score, signals = score_candidate(item, features)
                        rejection_reason = next((s for s in signals if s.startswith("reject:")), None)
                        core_signals = {s for s in signals if not s.startswith("reject")}
                        if args.debug_serp:
                            print(
                                "[debug][row {idx}] candidate score={score} url={url} signals={signals}".format(
                                    idx=row_idx, score=score, url=url, signals=signals
                                ),
                                flush=True,
                            )
                        # Stronger must-have: require role/freelance evidence to include
                        require_role = not args.no_require_role_signal
                        has_role_signal = any(s in signals for s in ["title_phrase","skill","description"])
                        multi_signal = len(core_signals) >= 2
                        has_name_lock = "last_initial" in core_signals

                        allow_candidate = False
                        if score >= args.min_score and (not require_role or has_role_signal) and (multi_signal or score >= args.accept_threshold):
                            allow_candidate = True
                        elif has_name_lock:
                            allow_candidate = True

                        candidate_counter += 1
                        slug_tokens = extract_candidate_slug_tokens(url)
                        candidate = CandidateEvidence(
                            candidate_id=f"cand_{candidate_counter}",
                            score=score,
                            signals=signals,
                            raw_item=item,
                            query=q,
                            rule_rank=0,
                            url=url,
                            slug_tokens=slug_tokens,
                            derived_last_initial=parse_candidate_last_initial(item),
                            first_name_in_url=bool(features.first_name and features.first_name.lower() in slug_tokens),
                            rejection_reason=rejection_reason,
                        )

                        if allow_candidate:
                            candidates.append(candidate)
                            if score >= args.accept_threshold + 2 and len(core_signals) >= 3:
                                strong_hit_found = True

                        append_log(log_handle, {
                            "event": "candidate",
                            "row_index": row_idx,
                            "upwork_name": features.full_name,
                            "query": q,
                            "url": url,
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "score": score,
                            "signals": signals,
                            "accepted": allow_candidate,
                            "provider": args.provider,
                            "candidate_id": candidate.candidate_id,
                            "slug_tokens": candidate.slug_tokens,
                            "derived_last_initial": candidate.derived_last_initial,
                            "first_name_in_url": candidate.first_name_in_url,
                            "rejection_reason": rejection_reason,
                            "timestamp": time.time(),
                        })

                    time.sleep(random.uniform(args.sleep_min, args.sleep_max))
                    if strong_hit_found:
                        break

                # Sort candidates by score (highest first) and assign ranks
                candidates.sort(key=lambda c: c.score, reverse=True)
                for idx, cand in enumerate(candidates, start=1):
                    cand.rule_rank = idx

                # Optional LLM rerank
                # Write one row per candidate (multiple rows per Upwork person)
                upwork_location = features.upwork_location
                upwork_skills = norm(row.get('Skills', ''))

                if candidates:
                    for cand in candidates:
                        score = cand.score
                        signals = cand.signals
                        item = cand.raw_item
                        q = cand.query
                        unique_signals = {s for s in signals if not s.startswith("reject")}
                        if cand.llm_selected:
                            unique_signals.add("llm_selected")
                        elif cand.llm_rank and cand.llm_rank > 1:
                            unique_signals.add(f"llm_rank_{cand.llm_rank}")
                        confidence = pick_confidence(score, args.accept_threshold, args.review_threshold)
                        if cand.llm_selected and cand.llm_confidence is not None:
                            llm_threshold = 0.6
                            if cand.llm_confidence >= max(llm_threshold, 0.85):
                                confidence = "High"
                            elif cand.llm_confidence >= llm_threshold:
                                confidence = "Medium"
                        if cand.llm_reject_reason:
                            unique_signals.add("llm_reject")
                        if confidence == "High" and len(unique_signals) < 3:
                            confidence = "Medium"
                        elif confidence == "Medium" and len(unique_signals) < 2:
                            confidence = "Low"
                        out_row = {
                            "upwork_name": features.full_name,
                            "upwork_title": norm(row.get("Title", "")),
                            "upwork_location": upwork_location,
                            "upwork_skills": upwork_skills[:200] + "..." if len(upwork_skills) > 200 else upwork_skills,  # Truncate long skills
                            "linkedin_url": item.get("link", ""),
                            "linkedin_title": item.get("title", ""),
                            "linkedin_snippet": item.get("snippet", ""),
                            "match_score": score,
                            "confidence": confidence,
                            "matched_signals": ",".join(sorted(unique_signals)),
                            "query_used": q,
                            "llm_selected": "yes" if cand.llm_selected else ("secondary" if cand.llm_rank and cand.llm_rank > 1 else ""),
                            "llm_confidence": f"{cand.llm_confidence:.2f}" if cand.llm_confidence is not None else "",
                            "llm_rationale": truncate_text(cand.llm_rationale, 300) if cand.llm_rationale else (cand.llm_reject_reason or ""),
                            "llm_rank": cand.llm_rank or "",
                        }
                        writer.writerow(out_row)
                else:
                    # Write one row showing no matches found
                    out_row = {
                        "upwork_name": features.full_name,
                        "upwork_title": norm(row.get("Title", "")),
                        "upwork_location": upwork_location,
                        "upwork_skills": upwork_skills[:200] + "..." if len(upwork_skills) > 200 else upwork_skills,
                        "linkedin_url": "",
                        "linkedin_title": "",
                        "linkedin_snippet": "",
                        "match_score": 0,
                        "confidence": "None",
                        "matched_signals": "",
                        "query_used": "",
                        "llm_selected": "",
                        "llm_confidence": "",
                        "llm_rationale": features.llm_decision.get("decision", {}).get("reject_reason") if features.llm_decision else "",
                        "llm_rank": "",
                    }
                    writer.writerow(out_row)
    finally:
        if log_handle:
            log_handle.close()

if __name__ == "__main__":
    main()
