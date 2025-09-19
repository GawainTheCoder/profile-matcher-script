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

def build_queries(row: Dict[str, str], max_queries: int = 6) -> List[str]:
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

    title_phrases = tokenize_phrases(title)
    top_sk = pick_top_skills(skills)
    schools = extract_schools(edu)
    companies = extract_companies(employment)
    desc_phrases = extract_description_phrases(desc)
    location_tokens = extract_locations(edu, desc, employment)
    location_tokens.extend(
        derive_location_aliases(country, edu, desc, employment)
    )
    location_tokens = list(dict.fromkeys(location_tokens))

    # Build name variants from most precise to broader
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
    location_hint = city or country
    primary_phrase = title_phrases[0] if title_phrases else (top_sk[0] if top_sk else "")

    ordered_queries: List[str] = []

    def add_query(q: Optional[str]) -> None:
        if q and q.strip():
            ordered_queries.append(q.strip())

    # Tier 1: Full name pairings first (location, education, skills, title)
    primary_names = []
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

def score_candidate(item: Dict[str, str], row: Dict[str, str]) -> Tuple[int, List[str]]:
    """Additive scoring using title/snippet evidence with signal tracking."""
    title_text = item.get("title", "")
    snippet = item.get("snippet", "")
    txt = f"{title_text} {snippet}".lower()
    url_txt = (item.get("link", "") or "").lower()
    signals: List[str] = []
    score = 0

    # Name guard: allow matches if first name appears in snippet/title or URL
    full = norm(row.get("Full Name"))
    fn, last_initial = parse_full_name(full)
    fn_lc = (fn or "").strip().lower()
    if fn_lc:
        name_in_text = fn_lc in txt
        name_in_url = re.search(rf"/(?:in|pub)/[^/]*{re.escape(fn_lc)}", url_txt)
        if not name_in_text and not name_in_url:
            return -999, ["reject:no-first-name"]

    # Last initial enforcement when confident
    cand_last_initial = parse_candidate_last_initial(item)
    if last_initial and cand_last_initial:
        if cand_last_initial != last_initial:
            return -999, ["reject:last-initial-mismatch"]
        score += 5
        signals.append("last_initial")

    city = (row.get("City") or "").strip().lower()
    country = (row.get("Country") or "").strip().lower()
    alt_locations = [loc.lower() for loc in extract_locations(row.get("Education", ""), row.get("Description", ""), row.get("Employment History", ""))]
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
    title_phrases = [p.lower() for p in tokenize_phrases(row.get("Title", ""))]
    if any(p in txt for p in title_phrases):
        score += 4
        signals.append("title_phrase")

    skills = [sk.lower() for sk in pick_top_skills(row.get("Skills", ""))]
    if any(sk in txt for sk in skills):
        score += 3
        signals.append("skill")

    edu_hits = [ed.lower() for ed in extract_schools(row.get("Education", ""))]
    if any(ed in txt for ed in edu_hits):
        score += 3
        signals.append("education")

    # Company evidence
    for comp in [c.lower() for c in extract_companies(row.get("Employment History", ""))]:
        if comp and comp in txt:
            score += 3
            signals.append("company")
            break

    desc_hits = [d.lower() for d in extract_description_phrases(row.get("Description", ""))]
    if any(d in txt for d in desc_hits):
        score += 2
        signals.append("description")

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

def extract_cctld(url: str) -> str:
    try:
        m = re.search(r"https?://([a-z]{2,3})\\.linkedin\\.com/", (url or "").lower())
        return m.group(1).upper() if m else ""
    except Exception:
        return ""

def llm_rerank_candidates(
    row: Dict[str, str],
    candidates: List[Tuple[int, List[str], Dict[str, str], str]],
    model: str = "gpt-5-nano",
    keep_threshold: float = 0.6,
    top_k: int = 5,
) -> List[Tuple[int, List[str], Dict[str, str], str]]:
    """Use OpenAI Responses API to re-rank and optionally filter candidates.
    Returns a possibly reduced, re-ordered list of candidates.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not candidates:
        return candidates

    # Take top_k by rule-based score first
    top = sorted(candidates, key=lambda x: x[0], reverse=True)[:max(1, top_k)]

    reranked: List[Tuple[float, Tuple[int, List[str], Dict[str, str], str]]] = []

    for score, signals, item, q in top:
        url = item.get("link", "")
        payload = {
            "model": model,
            "input": (
                "Decide if the LinkedIn result likely matches the Upwork freelancer. "
                "Return strict JSON with keys keep (boolean), relevance (0..1), reason (short).\n\n"
                f"Upwork Freelancer:\n"
                f"- Full Name: {norm(row.get('Full Name'))}\n"
                f"- Title: {norm(row.get('Title'))}\n"
                f"- Skills: {norm(row.get('Skills'))}\n"
                f"- Location: {norm(row.get('City'))}, {norm(row.get('Country'))}\n"
                f"- Education: {norm(row.get('Education'))}\n"
                f"- Description: {norm(row.get('Description'))[:300]}\n\n"
                f"LinkedIn Candidate:\n"
                f"- URL: {url}\n"
                f"- ccTLD: {extract_cctld(url)}\n"
                f"- Title: {item.get('title','')}\n"
                f"- Snippet: {item.get('snippet','')}\n\n"
                "Constraints: First name must match and last initial has been pre-validated when present. "
                "Prioritize role/title and location alignment.\n"
                "Respond ONLY with JSON: {\"keep\": true|false, \"relevance\": number 0..1, \"reason\": string}."
            ),
            "temperature": 0,
            "max_output_tokens": 120,
            "store": True,
            "user": (norm(row.get('Full Name')) or "upwork_matcher")[:64],
            "metadata": {
                "source": "upwork_to_linkedin_matcher",
                "row_city": norm(row.get('City')),
                "row_country": norm(row.get('Country')),
            }
            # Structured output can be enabled if supported; keep simple parsing for portability.
        }
        try:
            org_id = os.getenv("OPENAI_ORG_ID", "").strip()
            project_id = os.getenv("OPENAI_PROJECT_ID", "").strip()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if org_id:
                headers["OpenAI-Organization"] = org_id
            if project_id:
                headers["OpenAI-Project"] = project_id
            resp = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # Responses API returns choices-like content; try to extract text
            text = ""
            if isinstance(data, dict):
                # Various SDKs wrap output differently; try generic paths
                text = data.get("output_text") or data.get("content") or ""
                if not text and "output" in data:
                    text = data["output"]
                if not text and "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    text = choice.get("message", {}).get("content", "") or choice.get("text", "")
            decision = {"keep": True, "relevance": 0.5, "reason": ""}
            if text:
                # Attempt to find JSON within the text
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    decision = json.loads(m.group(0))
            rel = float(decision.get("relevance", 0.0))
            keep = bool(decision.get("keep", False))
            if keep and rel >= keep_threshold:
                reranked.append((rel, (score, signals, item, q)))
        except Exception:
            # On any error, fall back to keeping the candidate as-is with neutral relevance
            reranked.append((0.5, (score, signals, item, q)))

    # Sort by LLM relevance then original score
    reranked.sort(key=lambda x: x[0], reverse=True)
    kept = [entry for _, entry in reranked]
    # If nothing passed threshold, fall back to original candidates
    return kept or candidates

# ------------------------------- Runner ------------------------------

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
    ap.add_argument("--llm-rerank", action="store_true", help="Enable LLM-based reranking using gpt-5-nano")
    ap.add_argument("--llm-model", default="gpt-5-nano-2025-08-07")
    ap.add_argument("--llm-top-k", type=int, default=5)
    ap.add_argument("--llm-keep-threshold", type=float, default=0.6)
    ap.add_argument("--debug-serp", action="store_true", help="Log queries and raw SERP items for inspection")
    ap.add_argument("--query-log", help="Path to append JSONL diagnostics for queries and candidates")
    args = ap.parse_args()

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
                "match_score", "confidence", "matched_signals", "query_used"
            ]
            writer = csv.DictWriter(out, fieldnames=output_fieldnames)
            writer.writeheader()

            for row_idx, row in enumerate(reader, start=1):
                queries = build_queries(row, max_queries=args.max_queries)
                candidates = []  # Store all candidates, not just the best one
                seen_urls = set()
                strong_hit_found = False

                append_log(log_handle, {
                    "event": "row_start",
                    "row_index": row_idx,
                    "upwork_name": norm(row.get("Full Name", "")),
                    "query_count": len(queries),
                    "timestamp": time.time(),
                })

                for q in queries:
                    append_log(log_handle, {
                        "event": "query",
                        "row_index": row_idx,
                        "upwork_name": norm(row.get("Full Name", "")),
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
                                "upwork_name": norm(row.get("Full Name", "")),
                                "query": q,
                                "url": url,
                                "timestamp": time.time(),
                            })
                            continue
                        seen_urls.add(url)
                        score, signals = score_candidate(item, row)
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

                        if allow_candidate:
                            candidates.append((score, signals, item, q))
                            if score >= args.accept_threshold + 2 and len(core_signals) >= 3:
                                strong_hit_found = True

                        append_log(log_handle, {
                            "event": "candidate",
                            "row_index": row_idx,
                            "upwork_name": norm(row.get("Full Name", "")),
                            "query": q,
                            "url": url,
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "score": score,
                            "signals": signals,
                            "accepted": allow_candidate,
                            "provider": args.provider,
                            "timestamp": time.time(),
                        })

                    time.sleep(random.uniform(args.sleep_min, args.sleep_max))
                    if strong_hit_found:
                        break

                # Sort candidates by score (highest first)
                candidates.sort(key=lambda x: x[0], reverse=True)

                # Optional LLM rerank
                if args.llm_rerank and candidates:
                    candidates = llm_rerank_candidates(
                        row,
                        candidates,
                        model=args.llm_model,
                        keep_threshold=args.llm_keep_threshold,
                        top_k=args.llm_top_k,
                    )

                # Write one row per candidate (multiple rows per Upwork person)
                upwork_location = f"{norm(row.get('City'))}, {norm(row.get('Country'))}".strip(", ")
                upwork_skills = norm(row.get('Skills', ''))
                
                if candidates:
                    for score, signals, item, q in candidates:
                        unique_signals = {s for s in signals if not s.startswith("reject")}
                        confidence = pick_confidence(score, args.accept_threshold, args.review_threshold)
                        if confidence == "High" and len(unique_signals) < 3:
                            confidence = "Medium"
                        elif confidence == "Medium" and len(unique_signals) < 2:
                            confidence = "Low"
                        out_row = {
                            "upwork_name": norm(row.get("Full Name", "")),
                            "upwork_title": norm(row.get("Title", "")),
                            "upwork_location": upwork_location,
                            "upwork_skills": upwork_skills[:200] + "..." if len(upwork_skills) > 200 else upwork_skills,  # Truncate long skills
                            "linkedin_url": item.get("link", ""),
                            "linkedin_title": item.get("title", ""),
                            "linkedin_snippet": item.get("snippet", ""),
                            "match_score": score,
                            "confidence": confidence,
                            "matched_signals": ",".join(sorted(unique_signals)),
                            "query_used": q
                        }
                        writer.writerow(out_row)
                else:
                    # Write one row showing no matches found
                    out_row = {
                        "upwork_name": norm(row.get("Full Name", "")),
                        "upwork_title": norm(row.get("Title", "")),
                        "upwork_location": upwork_location,
                        "upwork_skills": upwork_skills[:200] + "..." if len(upwork_skills) > 200 else upwork_skills,
                        "linkedin_url": "",
                        "linkedin_title": "",
                        "linkedin_snippet": "",
                        "match_score": 0,
                        "confidence": "None",
                        "matched_signals": "",
                        "query_used": ""
                    }
                    writer.writerow(out_row)
    finally:
        if log_handle:
            log_handle.close()

if __name__ == "__main__":
    main()
