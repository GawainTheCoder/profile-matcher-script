# Upwork → LinkedIn Matcher

A modular system for matching Upwork freelancer profiles to LinkedIn profiles using SERP APIs and optional LLM reranking. The system uses rule-based scoring with semantic refinement to achieve high-precision matches.

## Quick Start

```bash
# 1. Setup
pip install --break-system-packages requests python-dotenv

# 2. Configure API keys in .env file
SERPER_API_KEY=your_serper_key
OPENAI_API_KEY=your_openai_key

# 3. Run basic matching
python3 upwork_to_linkedin_matcher.py \
  --input your_upwork_profiles.csv \
  --output results.csv \
  --provider serper

# 4. Add LLM selection (recommended)
python3 llm_select_existing.py \
  --upwork your_upwork_profiles.csv \
  --candidates results.csv \
  --output final_results.csv \
  --llm-model gpt-5-nano-2025-08-07 \
  --llm-keep-threshold 0.5
```

## Performance Benchmarks

Based on testing with 51 profiles against golden dataset:

- **SERP Coverage**: 51% (26/51 profiles found)
- **LLM Selection (threshold 0.5)**: 73.1% success rate when SERP found correct profile (19/26)
- **End-to-End Success**: 37.3% (19/51 correct matches)
- **LLM Precision**: 59.4% when making selections
- **False Positive Rate**: 25.5% (13 wrong selections)

**Key Insight**: The primary bottleneck is SERP coverage (finding the right LinkedIn profiles in search results), not LLM selection accuracy. Improving search queries and expanding candidate pools yields the highest gains.

---

## System Architecture

The system is modularized into focused components:

- **`upwork_to_linkedin_matcher.py`** - Main orchestration script (459 lines, down from 1833)
- **`providers.py`** - SERP API integrations (Serper, SerpAPI)
- **`models.py`** - Data structures and constants
- **`features.py`** - Text processing and feature extraction
- **`queries.py`** - Search query generation strategies
- **`scoring.py`** - Candidate scoring and matching logic
- **`llm.py`** - OpenAI integration for semantic reranking
- **`utils.py`** - Common utilities
- **`llm_select_existing.py`** - Standalone LLM selection script

---

## How It Works

### 1. Feature Extraction
From each Upwork profile, the system extracts:
- **Name variants**: Handles cultural naming patterns and permutations
- **Geographic signals**: City, country, location aliases
- **Professional context**: Title phrases, top skills, companies, schools
- **Descriptive phrases**: Key phrases from descriptions

### 2. Query Generation
Generates targeted LinkedIn searches using patterns like:
- `site:linkedin.com/in "First Last" "City" -inurl:"/jobs/"`
- `site:linkedin.com/in "First Name" "Top Skill" "Country"`
- `"First Name" "Title Phrase" site:linkedin.com/in`

### 3. Scoring System
Multi-signal scoring with hard guards:

**Hard Rejections:**
- Missing first name in profile text
- Last initial mismatch

**Positive Signals (additive):**
- City match: +5 points
- Country match: +3 points
- Title phrase match: +4 points
- Skill match: +3 points
- Education/Company match: +3 points each
- Last initial match: +5 points

**Confidence Levels:**
- High: ≥10 points (accept threshold)
- Medium: ≥7 points (review threshold)
- Low: ≥3 points (minimum score)

### 4. LLM Refinement (Recommended)
Uses GPT-5-nano with reasoning capabilities to semantically evaluate top candidates:

**Decision Framework:**
- Requires name match (first name + last initial)
- Looks for 1+ supporting signals: location, skills, role, company, education
- Returns single best match with confidence score (0.0-1.0)
- Provides detailed rationale for selection/rejection

**Optimal Settings:**
- `--llm-keep-threshold 0.5` (balanced precision/recall)
- `--llm-top-k 5` (candidates sent to LLM)
- Model: `gpt-5-nano-2025-08-07` (supports reasoning.effort parameter)

---

## Input Requirements

Your CSV must contain these columns (case-sensitive):

- **Full Name**: "Amna M." format (first name + last initial)
- **Title**: Role/title information
- **Description**: Free text description
- **Country**: "Pakistan", "Serbia", etc.
- **City**: "Lahore", "Belgrade", etc.
- **Skills**: Comma/pipe/semicolon separated skills
- **Education**: Educational background (schools extracted automatically)
- **Employment History**: Work history (companies extracted automatically)

Optional columns: English Level, Certifications, Profile URL

---

## Configuration

### Environment Variables (.env file)

```bash
# Required: Choose one SERP provider
SERP_PROVIDER=serper
SERPER_API_KEY=your_serper_key_here
# OR
SERPAPI_API_KEY=your_serpapi_key_here

# Required for LLM features
OPENAI_API_KEY=sk-proj-your_openai_key_here

# Optional OpenAI attribution
OPENAI_ORG_ID=org-your_org_id
OPENAI_PROJECT_ID=proj-your_project_id

# Optional alternative provider
FIRECRAWL_API_KEY=fc-your_firecrawl_key
```

### Key Command Line Options

**Basic Matching:**
```bash
--input path/to/input.csv        # Input Upwork profiles
--output path/to/output.csv      # Output results
--provider serper|serpapi        # SERP provider choice
--max-queries 10                 # Queries per profile (default: 6)
--min-score 1                    # Minimum score to include (default: 3)
```

**Quality Controls:**
```bash
--accept-threshold 10            # High confidence threshold
--review-threshold 7             # Medium confidence threshold
--no-score-filter               # Include all matches regardless of score
--no-require-role-signal        # Disable role/skill requirement
```

**LLM Options:**
```bash
--llm-model gpt-5-nano-2025-08-07    # Model to use (supports reasoning)
--llm-keep-threshold 0.5             # Minimum LLM confidence (recommended)
--llm-top-k 5                        # Max candidates to send to LLM (recommended)
```

---

## Usage Examples

### Basic High-Precision Run
```bash
python3 upwork_to_linkedin_matcher.py \
  --input profiles.csv \
  --output results.csv \
  --provider serper \
  --accept-threshold 11 \
  --min-score 3 \
  --max-queries 8
```

### High-Coverage Run
```bash
python3 upwork_to_linkedin_matcher.py \
  --input profiles.csv \
  --output results.csv \
  --provider serper \
  --min-score 1 \
  --no-score-filter \
  --max-queries 10
```

### LLM-Enhanced Workflow
```bash
# Step 1: Generate candidates
python3 upwork_to_linkedin_matcher.py \
  --input profiles.csv \
  --output candidates.csv \
  --min-score 1 \
  --max-queries 10

# Step 2: LLM selection
python3 llm_select_existing.py \
  --upwork profiles.csv \
  --candidates candidates.csv \
  --output final_results.csv \
  --llm-model gpt-5-nano-2025-08-07 \
  --llm-keep-threshold 0.5 \
  --llm-top-k 5
```

### Analysis and Validation
```bash
# Compare results against golden dataset
python3 compare_all_results.py

# Analyze LLM selection performance
python3 analyze_llm_selection.py
```

---

## Output Format

Each result row contains:

**Upwork Context:**
- `upwork_name`, `upwork_title`, `upwork_location`, `upwork_skills`

**LinkedIn Match:**
- `linkedin_url`, `linkedin_title`, `linkedin_snippet`

**Scoring Details:**
- `match_score` (integer), `confidence` (High/Medium/Low)
- `matched_signals` (comma-separated list)
- `query_used` (the search query that found this match)

**LLM Analysis (when used):**
- `llm_selected` (yes/secondary/blank)
- `llm_confidence` (0.0-1.0)
- `llm_rationale` (reasoning for selection/rejection)
- `llm_rank` (1-based ranking)

---

## Optimization Strategies

### For Higher Precision
1. **Increase scoring thresholds**: `--accept-threshold 12 --min-score 5`
2. **Keep role requirements**: Don't use `--no-require-role-signal`
3. **Use LLM selection**: Add the LLM step with `--llm-keep-threshold 0.5+`
4. **Focus queries**: Use fewer, more targeted queries

### For Higher Recall
1. **Lower thresholds**: `--min-score 1` or `--no-score-filter`
2. **More queries**: `--max-queries 10+`
3. **Disable filters**: `--no-require-role-signal`
4. **Lower LLM threshold**: `--llm-keep-threshold 0.4` (below 0.5 increases false positives)

### Name Handling Improvements
The system includes cultural name variations:
- "Necip Eray D." → tries "Eray Necip", "Necip Eray Damar", "Eray Damar"
- "Anastasiia G." → preserves unique spelling with conservative variations
- General permutations for two-part names

---

## Troubleshooting

### No Results Found
```bash
# Try lower thresholds
--min-score 1 --no-score-filter

# More queries per profile
--max-queries 10 --results-per-query 8

# Disable role requirement
--no-require-role-signal
```

### Too Many False Positives
```bash
# Stricter scoring
--min-score 5 --accept-threshold 12

# Add LLM filtering
python3 llm_select_existing.py --llm-keep-threshold 0.6
```

### LLM Not Working
1. **Check API key**: Ensure `OPENAI_API_KEY` is in `.env`
2. **Install dotenv**: `pip install python-dotenv`
3. **Verify model**: Use `gpt-5-nano-2025-08-07` or `gpt-3.5-turbo`

### Rate Limiting
```bash
# Add delays between requests
--sleep-min 1.5 --sleep-max 3.0

# Reduce query volume
--max-queries 5 --results-per-query 3
```

---

## API Costs & Limits

### SERP API Usage
- **Serper**: ~$5 per 1000 queries
- **SerpAPI**: ~$50 per 1000 queries
- Estimate: 6-10 queries per profile

### OpenAI Usage (LLM step)
- **GPT-5-nano**: $0.05/1M input tokens, $0.40/1M output tokens
- Estimate: ~$0.01-0.02 per profile for LLM selection
- Only top 5 candidates sent to reduce costs

---

## Advanced Features

### Query Logging
```bash
--query-log queries.jsonl --debug-serp
```
Logs all search queries and responses for debugging.

### Batch Processing
```bash
# Process large files in chunks
python3 batch_processor.py --chunk-size 50 --input large_file.csv
```

### Custom Scoring
Modify `scoring.py` to adjust signal weights:
```python
SIGNAL_SCORES = {
    'city': 5,
    'country': 3,
    'title_phrase': 4,
    'skill': 3,
    # ... customize as needed
}
```

---

## Limitations & Considerations

### Technical Limitations
- **SERP dependency**: Results quality depends on search engine snippets
- **Name ambiguity**: Common names may produce false matches
- **Geographic bias**: Search results may favor certain regions
- **Rate limits**: API quotas may limit processing speed

### Data Quality Factors
- **Incomplete profiles**: Missing skills/location data reduces match quality
- **Transliteration**: Non-English names may have spelling variations
- **Professional context**: Generic titles/skills provide weak signals

### Cost Considerations
- SERP API costs scale with profile count and query volume
- LLM costs are controlled but add overhead for large batches
- Consider cost/accuracy tradeoffs for your use case

---

## Contributing

The modular architecture makes it easy to:
- Add new SERP providers in `providers.py`
- Modify scoring logic in `scoring.py`
- Enhance query strategies in `queries.py`
- Customize feature extraction in `features.py`

---

## License & Compliance

This tool uses public SERP APIs and respects:
- Provider terms of service and rate limits
- LinkedIn's robots.txt (no direct crawling)
- Data privacy and applicable regulations

Ensure compliance with local laws and platform policies before use.# linkedin_matcher_v2
