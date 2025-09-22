## Upwork → LinkedIn Matcher

Row-driven person resolution from an Upwork export (CSV) to likely LinkedIn profile URLs using a SERP provider (Serper or SerpAPI), with rule-based scoring and an optional LLM reranker for semantic relevance.

This guide explains:
- What the script does end-to-end
- Required input columns and the output schema
- Configuration (env vars) and CLI flags
- Query strategy, scoring signals, and inclusion gates (last-initial, must-have role/freelance)
- Optional LLM reranking (OpenAI Responses API, gpt-5-nano)
- Limitations, roadmap, and troubleshooting
- How to install and run on your CSV

---

## Overview

Given a CSV of Upwork freelancers, the script:
- Parses each row and derives a search plan using that row’s own fields (no global dictionaries)
- Calls a SERP provider to Google-search only LinkedIn profile URLs
- Scores each candidate result (from SERP title/snippet), retains good ones, and writes a CSV
- Optionally reranks/filters the top candidates per row using an LLM for semantic relevance

Key safeguards:
- First-name guard (must be present in title/snippet)
- Last-initial enforcement (rejects candidates whose last-name initial conflicts with Upwork’s)
- Role/freelance must-have gate (filters off-topic results)

---

## Latest learnings: simple, high-precision strategy

### What to prioritize
- Build tight queries: always use quotes and `site:linkedin.com/in`, and prefer pairing name with one strong attribute at a time (city, country, top title phrase, top skill, one school, one company, or a short description phrase).
- Add SERP noise filters to every query: `-inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`.
- Enforce identity early: require first-name presence in text or URL; when last initial can be inferred, require it to match.
- Keep the inclusion gate: candidate must show role evidence (title phrase, skill, or description phrase) unless the last-initial lock is present.
- Stop early on a strong hit: if a candidate reaches a high score with multiple signals, skip remaining queries for that row to save budget.
- Optionally LLM rerank the top 5 only to reorder/filter borderline cases.

### Prioritized query templates (first 6–8 per row)
- `site:linkedin.com/in "First Last" "City" -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`
- `site:linkedin.com/in "First Last" "Country" -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`
- `site:linkedin.com/in "First Last" "Top Title Phrase" -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`
- `site:linkedin.com/in "First Last" "Top Skill" -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`
- `site:linkedin.com/in "First Last" "School" -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`
- `site:linkedin.com/in "First Last" "Company" -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`
- `site:linkedin.com/in "First" "Top Title Phrase" "City" -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`
- `"First Last" "Upwork" site:linkedin.com/in -inurl:"/jobs/" -inurl:"/learning/" -inurl:"/school/" -inurl:"/company/"`

If the Upwork name is `First LastInitial`, also try the trimmed core name `First Last`.

### Quick wins (recommended tweaks)
- Add SERP noise filters (above) to all generated queries.
- Canonicalize LinkedIn URLs before deduplication (strip tracking params, trailing slashes) so duplicates collapse reliably.
- Strengthen freelance keywords: include `independent`, `self-employed`, `contractor`, `consultant`, `freelancer`.
- Soft ccTLD bonus: add a small score bonus when the LinkedIn subdomain ccTLD matches the row country; use only as a secondary signal.

### Recommended high-precision run
```bash
python3 upwork_to_linkedin_matcher.py \
  --input 20_upwork_profiles.csv \
  --output test_output_20.csv \
  --provider serper \
  --max-queries 8 --results-per-query 8 \
  --accept-threshold 11 --review-threshold 7 --min-score 3 \
  --query-log query_log.jsonl --debug-serp
```

### Lightweight validation loop
- Take a sample of ~20 profiles with known ground truth.
- Run once and compute precision@1 manually.
- If false positives are high: raise `--accept-threshold` or require ≥3 unique signals for High confidence (already enforced), and keep the must-have role gate ON.
- If recall is low: add 1–2 more attribute-pair queries, allow last-initial lock to bypass the must-have role gate, or run the standalone LLM selection step on the SERP output and experiment with a slightly lower keep threshold (e.g., 0.5–0.55).

## Input CSV schema (required columns)

Expected headers (case sensitive):
- **Full Name**: Upwork-style e.g., "Amna M." (only last initial)
- **Title**: e.g., "Market Research Analyst | Competitors Analyst"
- **Description**: long free text; short phrases may be used in queries and for LLM context
- **Country**: e.g., "Pakistan"
- **City**: e.g., "Lahore"
- **Skills**: comma/pipe/semicolon separated
- **English Level**: (unused)
- **Education**: free text; schools are extracted using regex
- **Employment History**: semicolon-separated; company names are extracted heuristically
- **Certifications**: (unused)

Fields primarily used: `Full Name`, `Title`, `Skills`, `City`, `Country`, `Education`, `Description` (LLM), `Employment History`.

---

## Output CSV schema

One row per kept candidate (sorted by score or LLM relevance):
- **upwork_name** (from `Full Name`)
- **upwork_title**
- **upwork_location** ("City, Country")
- **upwork_skills** (truncated to 200 chars)
- **linkedin_url**
- **linkedin_title**
- **linkedin_snippet**
- **match_score** (integer)
- **confidence** (High | Medium | Low)
- **matched_signals** (comma-joined signals)
- **query_used** (the producing Google query)

LLM-specific columns (present when the Responses API step runs):
- **llm_selected** – `yes` for the model’s pick, `secondary` for remaining ranked options, blank if the model abstained
- **llm_confidence** – model-reported 0–1 confidence (string formatted to two decimals)
- **llm_rationale** – trimmed justification from the model (or the rejection reason when no match is chosen)
- **llm_rank** – 1-based ordering assigned by the LLM (populated for secondary options)

If a person yields no candidates after filtering, one placeholder row is written with empty LinkedIn fields and `match_score=0`.

---

## How it searches (queries)

Per row, up to `--max-queries` (deduped) are generated using:
- `site:linkedin.com/in "{first_name}" "{title_phrase}" "{city}"`
- `site:linkedin.com/in "{first_name}" "{top_skill}" "{country}"`
- `"{first_name}" Upwork site:linkedin.com/in ["{city}"]`
- `site:linkedin.com/in intitle:"{first_name}"`
- `site:linkedin.com/in "{Full Name}" ["{city}"]`
- Pivots using one extracted school or company when present

Notes:
- `first_name` is derived from `Full Name`; `last_initial` is only used during scoring.
- Title phrases come from `Title`; top skills come from `Skills`.
- Schools are extracted from `Education`; companies from `Employment History`.
- Short 3–6 word phrases can be extracted from `Description` and used in paired queries when present.

---

## How it scores (signals)

From candidate SERP `title + snippet` (visible text only):

Hard checks (reject):
- Missing first name in text
- Last-initial mismatch (parsed from LinkedIn URL slug or SERP title)

Positive evidence (additive):
- **city** in text: +5
- **country** in text: +3
- **alt_location** (location tokens derived from Education/Description/Employment): +3 (first match)
- **title_phrase** (from Upwork `Title`): +4
- **skill** (from Upwork `Skills`): +3
- **education** (school name from Upwork `Education`): +3
- **company** (from Upwork `Employment History`): +3
- **freelance/Upwork** keywords: +3
- **last_initial_match**: +5

Confidence buckets (from score):
- **High**: ≥ `--accept-threshold` (default 10)
- **Medium**: ≥ `--review-threshold` (default 7)
- **Low**: otherwise ≥ `--min-score` (default 3)

Confidence adjustment based on unique signals:
- High confidence is downgraded to Medium if fewer than 3 unique positive signals are present.
- Medium confidence is downgraded to Low if fewer than 2 unique positive signals are present.

Inclusion gate (precision control):
- Score must be ≥ `--min-score`
- Must-have role/freelance (default): candidate must include at least one of `title_phrase` OR `skill` OR `description` phrase
  - Disable with `--no-require-role-signal` if you prefer higher recall at the risk of off-topic matches

---

## Optional LLM reranker (OpenAI Responses API, `gpt-5-nano`)

The LLM selection step (`llm_select_existing.py`) sends the top `--llm-top-k` candidates per row to the OpenAI Responses API. The payload contains:

- **Upwork profile context** – full name and variants, first/last initial, title, top skills, description phrases, certifications, extracted schools/companies, employment summary, city/country, location aliases
- **Candidate context** – LinkedIn URL, slug tokens, derived last initial, first-name-in-url flag, SERP title/snippet, rule score, producing query, matched signals
- **Guardrails** – first name must align, last initial (when present) must match, favour role/location alignment, allow abstaining when confidence is low

Responses are requested as JSON schema:
```json
{
  "best_candidate_id": "cand_3" | null,
  "confidence": 0.0-1.0,
  "rationale": "short reasoning",
  "secondary_candidate_ids": ["cand_5", "cand_2"],
  "reject_reason": "optional explanation when no match"
}
```

Acceptance and ordering:
- Accept only if `confidence ≥ --llm-keep-threshold` (default 0.6); otherwise fall back to rule ordering
- Return a single candidate in `select` mode, or keep all ranked candidates in `assist` mode (primary + secondary ordering)
- On API errors, the rule-based ordering is preserved

Telemetry & attribution:
- Requests use `store=true` and include metadata (`source`, row city/country) for observability in the OpenAI dashboard
- `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` headers are forwarded when present

Token/cost control:
- Only the top `--llm-top-k` (default 5) candidates are sent
- `max_output_tokens` is capped (900) and `reasoning.effort` is set to `low` for nano-tier models

> **Prompt engineering notes**
> - Injecting the full Upwork profile (skills, employment, education, certifications, description snippets) materially improves discrimination on common names.
> - Providing slug tokens and location aliases helps the model reason about URL structure and geographic hints beyond the snippet.
> - Keep the schema strict (`additionalProperties=false`) so malformed outputs are rejected early.

### LLM-only pass on existing SERP output

Already have a SERP results CSV (e.g., `test_output_20.csv`)? Run the LLM selection step without hitting the search API:

```bash
python3 llm_select_existing.py \
  --upwork 20_upwork_profiles.csv \
  --candidates test_output_20.csv \
  --output 20_upwork_llm_results.csv \
  --llm-model gpt-5-nano-2025-08-07 \
  --llm-keep-threshold 0.6 \
  --query-log query_log_llm.jsonl
```

What happens under the hood:
- Rebuilds the same `RowFeatures` objects from the original Upwork CSV (so the LLM still sees full descriptions, education, employment, certifications, etc.)
- Converts each candidate row into `CandidateEvidence` (rule score, matched signals, slug tokens, derived last initial, producing query)
- Pipes the top `--llm-top-k` candidates per freelancer into the Responses API and annotates the results with `llm_selected`, `llm_confidence`, `llm_rationale`, and `llm_rank`
- Optionally logs the full (truncated) JSON responses for audit/debugging

This is the recommended workflow when you already have a `test_output.csv` from an earlier run and just want to layer the LLM decisioning on top.

---

## Providers & environment variables

Supported providers:
- **serper** (default): `https://google.serper.dev/search` (POST JSON)
- **serpapi**: `https://serpapi.com/search` (GET)

`.env` (auto-loaded via `python-dotenv`):

Search keys:
- `SERP_PROVIDER=serper|serpapi` (or pass `--provider`)
- For Serper: `SERPER_API_KEY=...`
- For SerpAPI: `SERPAPI_API_KEY=...`

LLM keys (only if you plan to run the LLM selection script):
- `OPENAI_API_KEY=...`
- Optional attribution: `OPENAI_ORG_ID=...`, `OPENAI_PROJECT_ID=...`

Python deps:
- `requests`, `python-dotenv`

---

## Install & setup

```bash
cd /path/to/this/folder/linkedin_matcher
python3 -m venv .venv
. .venv/bin/activate
pip install requests python-dotenv
```

Create `.env`:
```bash
# Choose one provider
SERP_PROVIDER=serper
SERPER_API_KEY=YOUR_SERPER_KEY

# Optional LLM reranker
OPENAI_API_KEY=YOUR_OPENAI_KEY
# Optional attribution
# OPENAI_ORG_ID=org_xxx
# OPENAI_PROJECT_ID=proj_xxx
```

---

## CLI flags (common)

- `--input` (required): path to input CSV
- `--output` (required): path to output CSV
- `--provider`: `serper` (default) or `serpapi`
- `--accept-threshold` (default 10), `--review-threshold` (default 7)
- `--max-queries` (default 6), `--results-per-query` (default 5)
- `--hl` (default `en`), `--gl` (default `us`)
- `--sleep-min` (default 0.8), `--sleep-max` (default 1.8)
- `--min-score` (default 3)
- `--no-require-role-signal` (disable must-have role/freelance gate)

---

## How to run

Rule-based only:
```bash
python3 upwork_to_linkedin_matcher.py \
  --input latest_people.csv \
  --output linkedin_matches.csv \
  --provider serper \
  --max-queries 6 --results-per-query 5
```

Add the LLM post-processing step:
```bash
python3 llm_select_existing.py \
  --upwork latest_people.csv \
  --candidates test_output.csv \
  --output latest_people_llm.csv \
  --llm-model gpt-5-nano-2025-08-07 \
  --llm-keep-threshold 0.6
```

Tighten precision further:
- Raise `--min-score` to 7–8
- Keep must-have gate ON and run the LLM selection with a higher `--llm-keep-threshold`

Notes:
- Increase `--sleep-*` for politeness and to avoid rate limits
- Consider region bias via `--gl` per row/country if you localize runs

---

## Tuning recall vs precision

- More recall: lower `--min-score`, disable must-have (`--no-require-role-signal`), raise `--results-per-query`
- More precision: keep must-have ON, raise `--min-score`, and layer the LLM selection pass on top of the SERP CSV (optionally tightening `--llm-keep-threshold` or reducing `--llm-top-k` there)

---

## Limitations

- **SERP snippets**: scoring depends on visible title/snippet; if signals aren’t shown, good candidates can be missed
- **Name ambiguity**: first-name + last initial is weak for very common names; gates + LLM reduce but don’t eliminate edge cases
- **Heuristic extraction**: school/company parsing is best-effort
- **LLM cost/latency**: controlled via the selection script’s `--llm-top-k`; still adds overhead
- **Globalization**: non-English/transliterated names may lower quality; tune `--hl`/`--gl`

---

## Roadmap

- Dynamic `--gl` from row country to improve locality
- ccTLD scoring bonus/penalty (match vs mismatch country)
- Cap top-N candidates per person
- Add caching of SERP results by query
- Demote generic queries when high-precision ones exist
- Optional debug JSONL with LLM decisions per candidate
- Conservative parallelization

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'requests'`
  - Activate venv and `pip install requests python-dotenv`

- No matches for some rows
  - Loosen: `--no-require-role-signal` or lower `--min-score`
  - Or run the LLM selection script with a lower `--llm-keep-threshold` (e.g., 0.5) to let the model surface borderline matches

- Too many irrelevant matches
  - Keep must-have ON, raise `--min-score` (7–8), and run the LLM selection pass with a higher `--llm-keep-threshold`

- See LLM spend/logs on OpenAI platform
  - Ensure `OPENAI_API_KEY` is set; calls are sent with `store=true`, `user`, and `metadata`
  - Optionally set `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` for better attribution

---

## Safety & compliance

Use SERP providers within their ToS and rate limits. Respect local laws and target platform policies. This tool relies on public SERPs and does not crawl LinkedIn pages directly.
