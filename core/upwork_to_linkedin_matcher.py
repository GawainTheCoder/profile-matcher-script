#!/usr/bin/env python3
"""
upwork_to_linkedin_matcher.py

Row-driven person-resolution from Upwork -> LinkedIn using a SERP provider
(SerpAPI or Serper). Builds queries from each row's fields, enforces last-initial,
and scores SERP snippets. Optionally uses LLM to generate better search queries.

Usage (rule-based):
  python upwork_to_linkedin_matcher.py \
      --input latest_people.csv \
      --output linkedin_matches.csv \
      --provider serper \
      --max-queries 6 \
      --results-per-query 5 \
      --min-score 3

Usage (with LLM query generation):
  python upwork_to_linkedin_matcher.py \
      --input latest_people.csv \
      --output linkedin_matches_llm_queries.csv \
      --provider serper \
      --max-queries 6 \
      --results-per-query 5 \
      --llm-query-plan \
      --llm-query-model gpt-5-nano

Notable flags:
  --no-require-role-signal   Disable role/freelance must-have gate (default gate is ON)
  --accept-threshold 10      High confidence threshold (default 10)
  --review-threshold 7       Medium confidence threshold (default 7)
  --llm-query-plan           Use LLM to generate better search queries

Env vars:
  SERP_PROVIDER=serper|serpapi  (or use --provider)
  SERPER_API_KEY=...            (for serper.dev)
  SERPAPI_API_KEY=...           (for serpapi.com)
  OPENAI_API_KEY=...            (required only with --llm-query-plan)
  OPENAI_ORG_ID=...             (optional, for dashboard attribution)
  OPENAI_PROJECT_ID=...         (optional, for dashboard attribution)
"""
import os
import csv
import time
import random
import argparse
from typing import List

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(path=None) -> bool:
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

# Import modules
from providers import SerpAPIProvider, SerperProvider
from models import CandidateEvidence, infer_gl
from features import prepare_row_features
from queries import build_queries
from scoring import score_candidate, parse_candidate_last_initial, extract_candidate_slug_tokens
from llm import generate_llm_queries, firecrawl_search, get_firecrawl_key
from utils import append_log, norm, canonicalize_linkedin_url, is_linkedin_profile, pick_confidence, truncate_text


def validate_args(args) -> None:
    """Validate command line arguments."""
    if args.sleep_min < 0 or args.sleep_max < 0:
        raise ValueError("Sleep intervals must be non-negative")
    if args.sleep_min > args.sleep_max:
        raise ValueError("--sleep-min cannot exceed --sleep-max")


def main():
    """Main processing function."""
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
    ap.add_argument("--no-score-filter", action="store_true", help="Include all LinkedIn candidates without scoring gate")
    ap.add_argument("--llm-query-plan", action="store_true", help="Use LLM to generate search queries")
    ap.add_argument("--llm-query-model", default="gpt-5-nano", help="LLM model for query generation")
    ap.add_argument("--llm-query-temperature", type=float, default=0.3)
    ap.add_argument("--use-firecrawl", action="store_true", help="Fallback to Firecrawl search when SERP results are weak")
    ap.add_argument("--firecrawl-max-queries", type=int, default=2, help="Max Firecrawl queries per row when enabled")
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
            # Rich output format with all candidates and metadata
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
                queries: List[str] = []
                if args.llm_query_plan:
                    llm_queries = generate_llm_queries(
                        features,
                        model=args.llm_query_model,
                        max_queries=args.max_queries,
                        temperature=args.llm_query_temperature,
                    )
                    for q in llm_queries:
                        if q and q not in queries:
                            queries.append(q)
                    if args.query_log and llm_queries:
                        append_log(log_handle, {
                            "event": "llm_queries",
                            "row_index": row_idx,
                            "upwork_name": features.full_name,
                            "queries": llm_queries,
                            "timestamp": time.time(),
                            "llm_model": args.llm_query_model,
                        })

                fallback_queries = build_queries(features, max_queries=args.max_queries)
                for q in fallback_queries:
                    if len(queries) >= args.max_queries:
                        break
                    if q not in queries:
                        queries.append(q)
                candidates: List[CandidateEvidence] = []
                seen_urls = set()
                strong_hit_found = False
                candidate_counter = 0

                def has_name_match(cand: CandidateEvidence) -> bool:
                    first = (features.first_name or "").lower()
                    if not first:
                        return True
                    if cand.first_name_in_url:
                        return True
                    text = (cand.raw_item.get("title", "") + " " + cand.raw_item.get("snippet", "")).lower()
                    return first in text

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
                    row_gl = infer_gl(features.country) or args.gl
                    try:
                        results = provider.search(q, num=args.results_per_query, hl=args.hl, gl=row_gl)
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
                            "gl": row_gl,
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
                        if args.no_score_filter and score == -999:
                            score = 0
                            signals = [s for s in signals if not s.startswith("reject")]
                            rejection_reason = None
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
                        has_name_lock = "last_initial" in core_signals
                        has_city_signal = "city" in core_signals or "alt_location" in core_signals
                        has_country_signal = "country" in core_signals
                        context_signals = {"title_phrase", "skill", "description", "company", "education", "certification"}
                        has_context_signal = any(s in core_signals for s in context_signals)

                        if args.no_score_filter:
                            allow_candidate = True
                        else:
                            allow_candidate = False
                            if score >= args.min_score:
                                if has_city_signal and (not require_role or has_role_signal or has_context_signal or has_name_lock):
                                    allow_candidate = True
                                elif has_context_signal and (not require_role or has_role_signal or has_name_lock):
                                    allow_candidate = True
                                elif has_name_lock and has_country_signal and has_role_signal:
                                    allow_candidate = True

                            if allow_candidate and core_signals.issubset({"last_initial", "country"}):
                                allow_candidate = False

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
                            if not args.no_score_filter and score >= args.accept_threshold + 2 and len(core_signals) >= 3:
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

                if args.use_firecrawl and (not candidates or not any(has_name_match(c) for c in candidates)):
                    firecrawl_key = get_firecrawl_key()
                    if firecrawl_key:
                        for q in queries[: args.firecrawl_max_queries]:
                            append_log(log_handle, {
                                "event": "firecrawl_query",
                                "row_index": row_idx,
                                "upwork_name": features.full_name,
                                "query": q,
                                "timestamp": time.time(),
                            })
                            for item in firecrawl_search(q, max_results=args.results_per_query):
                                url = item.get("link", "")
                                if not url:
                                    continue
                                if url in seen_urls:
                                    continue
                                score, signals = score_candidate(item, features)
                                rejection_reason = next((s for s in signals if s.startswith("reject:")), None)
                                if args.no_score_filter and score == -999:
                                    score = 0
                                    signals = [s for s in signals if not s.startswith("reject")]
                                    rejection_reason = None
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
                                candidates.append(candidate)
                                seen_urls.add(url)
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
                                    "accepted": True,
                                    "provider": "firecrawl",
                                    "candidate_id": candidate.candidate_id,
                                    "slug_tokens": candidate.slug_tokens,
                                    "derived_last_initial": candidate.derived_last_initial,
                                    "first_name_in_url": candidate.first_name_in_url,
                                    "rejection_reason": rejection_reason,
                                    "timestamp": time.time(),
                                })
                            if candidates and any(has_name_match(c) for c in candidates):
                                break

                # Sort candidates by score (highest first) unless raw output requested
                if not args.no_score_filter:
                    candidates.sort(key=lambda c: c.score, reverse=True)
                for idx, cand in enumerate(candidates, start=1):
                    cand.rule_rank = idx

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
                            "upwork_skills": upwork_skills[:200] + "..." if len(upwork_skills) > 200 else upwork_skills,
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