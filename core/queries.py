"""
Query building logic for LinkedIn profile search.
"""
from typing import List, Optional
try:
    from .models import RowFeatures
    from .utils import norm
except ImportError:
    from models import RowFeatures
    from utils import norm


def build_queries(features: RowFeatures, max_queries: int = 6) -> List[str]:
    """Build search queries from extracted features."""
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
    single_word_skills = [s for s in features.all_skills if len(s.split()) == 1][:3]
    designer_like_skills = [
        s for s in features.all_skills
        if len(s.split()) >= 2 and any(tok in s.lower() for tok in ["design", "designer", "landing page"])
    ][:3]

    for single_skill in single_word_skills:
        for designer_term in designer_like_skills:
            add_query(f'site:linkedin.com/in "{single_skill}" "{designer_term}"')
            if city:
                add_query(f'site:linkedin.com/in "{single_skill}" "{designer_term}" "{city}"')
            elif country:
                add_query(f'site:linkedin.com/in "{single_skill}" "{designer_term}" "{country}"')

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