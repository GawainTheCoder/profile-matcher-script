"""
Data models and constants for LinkedIn profile matching.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# LinkedIn profile domains
LINKEDIN_DOMAINS = ("linkedin.com/in", "linkedin.com/pub")

# Stopwords for text processing
STOPWORDS = set("""
a an and are as at be by for from has have i in is it of on or that the to with your you
""".split())

# Generic name tokens to avoid in last name detection
GENERIC_NAME_TOKENS = {
    "design", "designer", "developers", "developer", "develop", "digital", "web",
    "webdesigner", "webdesign", "landing", "page", "pages", "marketing",
    "consultant", "consulting", "consultants", "expert", "experts", "specialist",
    "specialists", "agency", "studio", "growth", "coach", "coaching", "freelancer",
    "freelance", "creative", "creator", "product", "manager", "founder", "owner",
    "ceo", "cto", "cmo", "ux", "ui", "figma", "framer", "shopify", "wordpress",
    "webflow", "consultancy", "services", "solutions",
}

# Country to Google Language code mapping
COUNTRY_TO_GL = {
    "argentina": "ar", "australia": "au", "austria": "at", "bangladesh": "bd",
    "belgium": "be", "brazil": "br", "canada": "ca", "chile": "cl",
    "colombia": "co", "croatia": "hr", "czech republic": "cz", "denmark": "dk",
    "egypt": "eg", "estonia": "ee", "finland": "fi", "france": "fr",
    "germany": "de", "greece": "gr", "hong kong": "hk", "hungary": "hu",
    "india": "in", "indonesia": "id", "ireland": "ie", "israel": "il",
    "italy": "it", "japan": "jp", "kenya": "ke", "latvia": "lv",
    "lithuania": "lt", "luxembourg": "lu", "malaysia": "my", "mexico": "mx",
    "nepal": "np", "netherlands": "nl", "new zealand": "nz", "nigeria": "ng",
    "norway": "no", "pakistan": "pk", "philippines": "ph", "poland": "pl",
    "portugal": "pt", "qatar": "qa", "romania": "ro", "russia": "ru",
    "saudi arabia": "sa", "serbia": "rs", "singapore": "sg", "south africa": "za",
    "spain": "es", "sweden": "se", "switzerland": "ch", "thailand": "th",
    "turkey": "tr", "ukraine": "ua", "united arab emirates": "ae",
    "united kingdom": "uk", "united states": "us", "uruguay": "uy", "vietnam": "vn",
}


@dataclass
class RowFeatures:
    """Extracted and processed features from an Upwork profile row."""
    raw: Dict[str, str]
    full_name: str
    first_name: str
    last_initial: str
    name_variants: List[str]
    best_name: str
    core_without_initial: str
    title_phrases: List[str]
    top_skills: List[str]
    all_skills: List[str]
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
    llm_query_plan: Optional[Dict[str, Any]] = None


@dataclass
class CandidateEvidence:
    """Evidence and metadata for a LinkedIn profile candidate."""
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


def infer_gl(country: str) -> Optional[str]:
    """Infer Google Language code from country name."""
    if not country:
        return None
    key = country.strip().lower()
    if not key:
        return None
    return COUNTRY_TO_GL.get(key)