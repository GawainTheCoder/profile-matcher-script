"""
LLM integration for query generation and candidate reranking.
"""
import json
import os
import re
import time
import requests
from typing import Dict, List, Any, Optional
try:
    from .models import RowFeatures, CandidateEvidence
    from .utils import norm, truncate_text
except ImportError:
    from models import RowFeatures, CandidateEvidence
    from utils import norm, truncate_text


# LLM prompts
LLM_QUERY_PROMPT = (
    "You generate Google search queries that locate a person's LinkedIn profile. "
    "Each query must include (1) the person's first name, (2) their last-name initial if provided, "
    "and (3) the literal token 'site:linkedin.com/in'. Add at least one discriminating descriptor "
    "such as their city, country, role/title phrase, or a distinctive top skill from the provided context. "
    "Construct the query as plain words separated by spaces (no surrounding quotes), e.g., 'Artur R web designer Bali site:linkedin.com/in'. "
    "Prefer descriptors drawn from the provided role_keywords list when available; if 'web designer' appears in that list, ensure at least one query includes the words 'web designer'. "
    "Return strictly JSON: {\"queries\": [\"...\", ...]} with at most 3 queries. "
    "Do not add explanatory text—only the JSON payload."
)


def get_firecrawl_key() -> str:
    """Get Firecrawl API key from environment."""
    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not key:
        key = os.getenv("FIRECRAWL_API_LEY", "").strip()
    return key


def firecrawl_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search using Firecrawl API."""
    api_key = get_firecrawl_key()
    if not api_key:
        return []
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {"query": query}
    try:
        resp = requests.post(
            "https://api.firecrawl.dev/v1/search",
            headers=headers,
            data=json.dumps(payload),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    results = []
    for item in data.get("data", [])[:max_results]:
        link = item.get("url", "")
        title = item.get("title", "")
        snippet = item.get("description", "")
        if link and title:
            results.append({"link": link, "title": title, "snippet": snippet or ""})
    return results


def build_llm_query_context(features: RowFeatures) -> Dict[str, Any]:
    """Build context for LLM query generation."""
    role_keywords: List[str] = []
    if features.primary_phrase:
        role_keywords.append(features.primary_phrase)
    for phrase in features.title_phrases[1:3]:
        if phrase not in role_keywords:
            role_keywords.append(phrase)
    skill_phrases = [s for s in features.all_skills if len(s.split()) >= 2][:5]
    for skill in skill_phrases:
        if skill not in role_keywords:
            role_keywords.append(skill)

    lower_skills = {s.lower(): s for s in features.all_skills}
    if any("web design" in s for s in lower_skills):
        role_keywords.append("web designer")
    if any("landing page" in s for s in lower_skills):
        role_keywords.append("landing page designer")
    if any("ux" in s or "ui" in s for s in lower_skills):
        role_keywords.append("UX UI designer")
    if features.primary_phrase and "designer" not in features.primary_phrase.lower():
        role_keywords.append(f"{features.primary_phrase} specialist")

    deduped_roles: List[str] = []
    for phrase in role_keywords:
        if phrase and phrase not in deduped_roles:
            deduped_roles.append(phrase)

    return {
        "full_name": features.full_name,
        "first_name": features.first_name,
        "last_initial": features.last_initial,
        "city": features.city,
        "country": features.country,
        "title": norm(features.raw.get("Title"))[:160],
        "top_skills": features.top_skills[:5],
        "notable_skills": features.all_skills[:10],
        "primary_phrase": features.primary_phrase,
        "companies": features.companies[:3],
        "description_snippet": truncate_text(norm(features.raw.get("Description")), 240),
        "role_keywords": deduped_roles[:8],
    }


def parse_responses_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse OpenAI Responses API JSON response."""
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


def generate_llm_queries(
    features: RowFeatures,
    model: str = "gpt-5-nano",
    max_queries: int = 3,
    temperature: float = 0.3,
) -> List[str]:
    """Generate search queries using LLM."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not features.first_name:
        return []

    payload_body = build_llm_query_context(features)

    json_schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max_queries,
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }

    request_payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": LLM_QUERY_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Profile context:\n" + json.dumps(payload_body, ensure_ascii=False),
                    }
                ],
            },
        ],
        "max_output_tokens": 600,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "search_queries",
                "schema": json_schema,
            }
        },
        "metadata": {
            "source": "upwork_to_linkedin_matcher.plan",
            "row_city": features.city,
            "row_country": features.country,
        },
    }
    if temperature is not None and not model.startswith("gpt-5-nano"):
        request_payload["temperature"] = temperature
    if model.startswith("gpt-5-nano"):
        request_payload["reasoning"] = {"effort": "low"}

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

    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            data=json.dumps(request_payload),
            timeout=30,
        )
        resp.raise_for_status()
        raw_response = resp.json()
        decision = parse_responses_json(raw_response)
    except Exception as exc:
        features.llm_query_plan = {
            "error": str(exc),
            "model": model,
            "request_payload": request_payload,
            "timestamp": time.time(),
        }
        return []

    if not isinstance(decision, dict):
        decision = {}
    raw_queries = decision.get("queries") or []
    if isinstance(raw_queries, str):
        try:
            raw_queries = json.loads(raw_queries).get("queries", [])
        except Exception:
            raw_queries = []
    cleaned: List[str] = []
    for q in raw_queries:
        if not isinstance(q, str):
            continue
        query = q.strip()
        if not query:
            continue
        if "site:linkedin.com" not in query.lower():
            query = f"site:linkedin.com {query}"
        if "site:linkedin.com/in" not in query.lower():
            query = query.replace("site:linkedin.com", "site:linkedin.com/in")
            if "site:linkedin.com/in" not in query.lower():
                query = f"site:linkedin.com/in {query}"
        if features.first_name.lower() not in query.lower():
            query = f"{features.first_name} {query}".strip()
        if features.last_initial and features.last_initial.lower() not in query.lower():
            query = f"{query} {features.last_initial}".strip()
        query = re.sub(r"\s+", " ", query)
        if query not in cleaned:
            cleaned.append(query)
        if len(cleaned) >= max_queries:
            break

    features.llm_query_plan = {
        "model": model,
        "queries": cleaned,
        "raw_decision": decision,
        "raw_response": raw_response,
        "timestamp": time.time(),
    }

    return cleaned


def build_llm_match_payload(features: RowFeatures, candidates: List[CandidateEvidence]) -> Dict[str, Any]:
    """Build payload for LLM candidate reranking."""
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
        location_match = []
        text_blob = (item.get("title", "") + " " + item.get("snippet", "")).lower()
        if features.city and features.city.lower() in text_blob:
            location_match.append("city")
        if features.country and features.country.lower() in text_blob:
            location_match.append("country")
        skill_hits = []
        for sk in features.top_skills[:5]:
            if sk.lower() in text_blob:
                skill_hits.append(sk)
        name_hits: List[str] = []
        title_text = (item.get("title", "") or "")
        snippet_text = (item.get("snippet", "") or "")
        if features.first_name and features.first_name.lower() in title_text.lower():
            name_hits.append("title")
        if features.first_name and features.first_name.lower() in snippet_text.lower():
            name_hits.append("snippet")
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
            "location_hits": location_match,
            "skill_hits": skill_hits,
            "name_hits": name_hits,
            "rejection_reason": cand.rejection_reason,
        })

    return {
        "upwork_profile": upwork_profile,
        "linkedin_candidates": linkedin_candidates,
    }


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
        "You are a sourcing analyst matching Upwork freelancers to their LinkedIn profiles.\n\n"
        "Your goal: Find the SINGLE BEST match from the candidates, if one exists.\n\n"
        "REQUIREMENTS:\n"
        "- First name MUST match exactly (case-insensitive)\n"
        "- Last initial MUST match when Upwork provides one\n"
        "- Look for supporting evidence: location, skills, role, company, education\n\n"
        "DECISION RULES:\n"
        "- If you find a good match (name + 2+ supporting signals): SELECT IT\n"
        "- If you find a decent match (name + 1 supporting signal): SELECT IT with moderate confidence\n"
        "- If candidates only match the name with no other evidence: REJECT (too risky)\n"
        "- If no candidates match the name requirements: REJECT\n\n"
        "Return the best candidate when confident it's the same person, even if not perfect. "
        "Output JSON following the schema."
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

    # Add reasoning parameter for reasoning models (o1, o3, gpt-5-nano, etc.)
    if model.startswith("o1") or model.startswith("o3") or model.startswith("gpt-5-nano"):
        request_payload["reasoning"] = {"effort": "low"}

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

    # Two-stage verification for low-confidence selections (< 0.7)
    if selected and confidence < 0.7:
        verification_result = verify_llm_selection(features, selected, model, keep_threshold)
        if not verification_result:
            # Verification failed, reject the selection
            selected.llm_selected = False
            selected.llm_reject_reason = "Failed two-stage verification"
            return candidate_pool

    if mode == "select":
        return ordered[: max(1, len(secondary_ids) + 1)]
    return ordered


def verify_llm_selection(
    features: RowFeatures,
    candidate: CandidateEvidence,
    model: str = "gpt-5-nano-2025-08-07",
    threshold: float = 0.6,
) -> bool:
    """
    Two-stage verification: Ask LLM if the selected candidate is actually the same person.
    Returns True if verified, False if rejected.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return True  # Skip verification if no API key

    # Build verification prompt
    verification_prompt = {
        "upwork_name": features.full_name,
        "upwork_location": features.upwork_location,
        "upwork_title": norm(features.raw.get("Title", ""))[:200],
        "upwork_skills": features.top_skills[:10],
        "linkedin_url": candidate.url,
        "linkedin_title": candidate.raw_item.get("title", ""),
        "linkedin_snippet": candidate.raw_item.get("snippet", "")[:400],
    }

    schema = {
        "type": "json_schema",
        "name": "verification_result",
        "schema": {
            "type": "object",
            "properties": {
                "is_same_person": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string"},
            },
            "required": ["is_same_person", "confidence", "reasoning"],
            "additionalProperties": False,
        },
    }

    system_prompt = (
        "You are a verification analyst. Given an Upwork freelancer profile and a selected LinkedIn profile, "
        "determine if they are the SAME person. Require matching first name (case-insensitive). "
        "Consider last initial, location, skills, role alignment, and professional context. "
        "Be conservative - if you're not confident they're the same person, return false."
    )

    user_prompt = (
        "Verify if these profiles are the same person:\n" +
        json.dumps(verification_prompt, ensure_ascii=False)
    )

    request_payload = {
        "model": model,
        "max_output_tokens": 400,
        "text": {"format": schema},
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    }

    # Add reasoning parameter for reasoning models
    if model.startswith("o1") or model.startswith("o3") or model.startswith("gpt-5-nano"):
        request_payload["reasoning"] = {"effort": "low"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            data=json.dumps(request_payload),
            timeout=30,
        )
        resp.raise_for_status()
        raw_response = resp.json()
        decision = parse_responses_json(raw_response)

        is_same = decision.get("is_same_person", False)
        conf = float(decision.get("confidence", 0.0) or 0.0)

        # Pass verification if confidence is above threshold AND marked as same person
        return is_same and conf >= threshold

    except Exception:
        # If verification fails, be conservative and accept original selection
        return True