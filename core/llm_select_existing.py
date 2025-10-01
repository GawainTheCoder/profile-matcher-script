#!/usr/bin/env python3
"""
Run the LLM match selection step on precomputed SERP candidates.

Usage example:
  python llm_select_existing.py \
      --upwork 20_upwork_profiles.csv \
      --candidates test_output_20.csv \
      --output 20_upwork_llm_results.csv \
      --llm-model gpt-5-nano-2025-08-07 \
      --llm-keep-threshold 0.6

The script groups the candidate rows by Upwork freelancer, rebuilds
row features from the original Upwork CSV, and invokes the same
LLM-based selection routine used in `upwork_to_linkedin_matcher.py`.
"""

import argparse
import csv
import json
import time
from typing import Dict, List, Iterable

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models import RowFeatures, CandidateEvidence
from features import prepare_row_features
from scoring import extract_candidate_slug_tokens, parse_candidate_last_initial
from llm import llm_rerank_candidates
from utils import pick_confidence, norm, truncate_text, append_log


def load_upwork_rows(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row.get("Full Name", ""): row for row in reader}


def load_candidate_groups(path: str) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("upwork_name", "").strip()
            if not name:
                continue
            groups.setdefault(name, []).append(row)
    return groups


def build_candidates(rows: Iterable[Dict[str, str]], features: RowFeatures) -> List[CandidateEvidence]:
    candidates: List[CandidateEvidence] = []
    for idx, row in enumerate(rows, start=1):
        url = row.get("linkedin_url", "")
        title = row.get("linkedin_title", "")
        snippet = row.get("linkedin_snippet", "")
        matched_signals = [s for s in (row.get("matched_signals", "") or "").split(",") if s]
        score_str = row.get("match_score", "0")
        try:
            score = int(float(score_str))
        except ValueError:
            score = 0

        slug_tokens = extract_candidate_slug_tokens(url)
        cand = CandidateEvidence(
            candidate_id=f"cand_{idx}",
            score=score,
            signals=matched_signals,
            raw_item={"title": title, "snippet": snippet, "link": url},
            query=row.get("query_used", ""),
            rule_rank=idx,
            url=url,
            slug_tokens=slug_tokens,
            derived_last_initial=parse_candidate_last_initial({"link": url, "title": title}),
            first_name_in_url=bool(features.first_name and features.first_name.lower() in slug_tokens),
            rejection_reason=None,
        )
        candidates.append(cand)

    candidates.sort(key=lambda c: c.score, reverse=True)
    for idx, cand in enumerate(candidates, start=1):
        cand.rule_rank = idx

    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upwork", required=True, help="Original Upwork CSV with full profile data")
    ap.add_argument("--candidates", required=True, help="CSV containing SERP candidates (e.g., test_output_20.csv)")
    ap.add_argument("--output", required=True, help="Path to write LLM-selected CSV")
    ap.add_argument("--accept-threshold", type=int, default=10)
    ap.add_argument("--review-threshold", type=int, default=7)
    ap.add_argument("--llm-model", default="gpt-5-nano-2025-08-07")
    ap.add_argument("--llm-top-k", type=int, default=5)
    ap.add_argument("--llm-keep-threshold", type=float, default=0.6)
    ap.add_argument("--llm-mode", choices=["select", "assist"], default="select")
    ap.add_argument("--query-log", help="Optional JSONL log for diagnostics")
    args = ap.parse_args()

    upwork_rows = load_upwork_rows(args.upwork)
    candidate_groups = load_candidate_groups(args.candidates)

    log_handle = open(args.query_log, "a", encoding="utf-8") if args.query_log else None

    with open(args.output, "w", newline="", encoding="utf-8") as out:
        fieldnames = [
            "upwork_name",
            "upwork_title",
            "upwork_location",
            "upwork_skills",
            "linkedin_url",
            "linkedin_title",
            "linkedin_snippet",
            "match_score",
            "confidence",
            "matched_signals",
            "query_used",
            "llm_selected",
            "llm_confidence",
            "llm_rationale",
            "llm_rank",
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for name, rows in candidate_groups.items():
            raw_upwork = upwork_rows.get(name)
            if not raw_upwork:
                raise KeyError(f"Upwork record for '{name}' not found in {args.upwork}")

            features = prepare_row_features(raw_upwork)
            candidates = build_candidates(rows, features)

            if args.query_log:
                append_log(log_handle, {
                    "event": "row_start",
                    "upwork_name": features.full_name,
                    "candidate_count": len(candidates),
                    "timestamp": time.time(),
                })

            if candidates:
                candidates = llm_rerank_candidates(
                    features,
                    candidates,
                    model=args.llm_model,
                    keep_threshold=args.llm_keep_threshold,
                    top_k=args.llm_top_k,
                    mode=args.llm_mode,
                )

            if args.query_log and features.llm_decision is not None:
                raw_excerpt = ""
                try:
                    raw_excerpt = truncate_text(json.dumps(features.llm_decision.get("raw_response"), ensure_ascii=False), 2000)
                except Exception:
                    raw_excerpt = "<unserializable>"
                append_log(log_handle, {
                    "event": "llm_decision",
                    "upwork_name": features.full_name,
                    "model": features.llm_decision.get("model"),
                    "used_candidates": features.llm_decision.get("used_candidates"),
                    "decision": features.llm_decision.get("decision"),
                    "error": features.llm_decision.get("error"),
                    "error_body": features.llm_decision.get("error_body"),
                    "raw_response": raw_excerpt,
                    "timestamp": features.llm_decision.get("timestamp", time.time()),
                })

            upwork_title = norm(raw_upwork.get("Title"))
            upwork_location = features.upwork_location
            upwork_skills = norm(raw_upwork.get("Skills", ""))
            if len(upwork_skills) > 200:
                upwork_skills = upwork_skills[:200] + "..."

            if candidates:
                # Only output the top LLM-selected candidate (if any)
                top_candidate = None
                for cand in candidates:
                    if cand.llm_selected:
                        top_candidate = cand
                        break

                # If no LLM selection, write empty row
                if not top_candidate:
                    writer.writerow({
                        "upwork_name": features.full_name,
                        "upwork_title": upwork_title,
                        "upwork_location": upwork_location,
                        "upwork_skills": upwork_skills,
                        "linkedin_url": "",
                        "linkedin_title": "",
                        "linkedin_snippet": "",
                        "match_score": 0,
                        "confidence": "None",
                        "matched_signals": "llm_no_selection",
                        "query_used": "",
                        "llm_selected": "no",
                        "llm_confidence": "",
                        "llm_rationale": "No suitable match found above threshold",
                        "llm_rank": "",
                    })
                else:
                    # Write only the top selected candidate
                    cand = top_candidate
                    score = cand.score
                    unique_signals = {s for s in cand.signals if s}
                    unique_signals.add("llm_selected")

                    confidence = pick_confidence(score, args.accept_threshold, args.review_threshold)
                    if cand.llm_confidence is not None:
                        if cand.llm_confidence >= max(args.llm_keep_threshold, 0.85):
                            confidence = "High"
                        elif cand.llm_confidence >= args.llm_keep_threshold:
                            confidence = "Medium"
                    if confidence == "High" and len(unique_signals) < 3:
                        confidence = "Medium"
                    elif confidence == "Medium" and len(unique_signals) < 2:
                        confidence = "Low"

                    writer.writerow({
                        "upwork_name": features.full_name,
                        "upwork_title": upwork_title,
                        "upwork_location": upwork_location,
                        "upwork_skills": upwork_skills,
                        "linkedin_url": cand.url,
                        "linkedin_title": cand.raw_item.get("title", ""),
                        "linkedin_snippet": cand.raw_item.get("snippet", ""),
                        "match_score": score,
                        "confidence": confidence,
                        "matched_signals": ",".join(sorted(unique_signals)),
                        "query_used": cand.query,
                        "llm_selected": "yes",
                        "llm_confidence": f"{cand.llm_confidence:.2f}" if cand.llm_confidence is not None else "",
                        "llm_rationale": truncate_text(cand.llm_rationale, 300) if cand.llm_rationale else "",
                        "llm_rank": "1",
                    })
            else:
                writer.writerow({
                    "upwork_name": features.full_name,
                    "upwork_title": upwork_title,
                    "upwork_location": upwork_location,
                    "upwork_skills": upwork_skills,
                    "linkedin_url": "",
                    "linkedin_title": "",
                    "linkedin_snippet": "",
                    "match_score": 0,
                    "confidence": "None",
                    "matched_signals": "",
                    "query_used": "",
                    "llm_selected": "",
                    "llm_confidence": "",
                    "llm_rationale": "",
                    "llm_rank": "",
                })

    if log_handle:
        log_handle.close()


if __name__ == "__main__":
    main()

