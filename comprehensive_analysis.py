#!/usr/bin/env python3
"""
Comprehensive E2E Analysis: SERP vs LLM Performance
"""
import csv
from typing import Dict, List, Tuple
from collections import defaultdict

def normalize_url(url):
    """Normalize LinkedIn URL for comparison."""
    if not url:
        return ""
    url = url.lower().replace('https://', '').replace('http://', '').replace('www.', '')
    if '.linkedin.com' in url:
        url = url.split('.linkedin.com')[1]
        url = 'linkedin.com' + url
    return url.split('?')[0].split('#')[0].rstrip('/')

def load_golden_dataset() -> Dict[str, str]:
    """Load the golden LinkedIn URLs keyed by Upwork profile names."""
    golden = {}
    with open('/Users/ashterhaider/Downloads/build/perplexity_test/data/51_linkedin.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            golden[row['Full Name']] = row['linkedin_url']
    return golden

def analyze_e2e():
    """Comprehensive E2E analysis."""
    golden = load_golden_dataset()

    # Load SERP results (all candidates)
    serp_results = defaultdict(list)
    with open('/Users/ashterhaider/Downloads/build/perplexity_test/upwork_linkedin_redo/serper_all_results.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['upwork_name']
            if row['linkedin_url'].strip():
                serp_results[name].append(row)

    # Load LLM results
    llm_results = defaultdict(list)
    llm_selections = {}
    with open('/Users/ashterhaider/Downloads/build/perplexity_test/upwork_linkedin_redo/llm_verified_results.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['upwork_name']
            if row['linkedin_url'].strip():
                llm_results[name].append(row)
                # Track LLM selections
                if row.get('llm_selected') == 'yes':
                    if name not in llm_selections:
                        llm_selections[name] = []
                    llm_selections[name].append(row)

    # Analysis counters
    total_profiles = len(golden)
    serp_found_correct = 0
    llm_selected_correct = 0
    llm_selected_wrong = 0
    llm_no_selection = 0
    serp_missing = 0

    print("=" * 80)
    print("🔍 COMPREHENSIVE E2E ANALYSIS: SERP vs LLM Performance")
    print("=" * 80)
    print(f"\nTotal profiles in golden dataset: {total_profiles}\n")

    # Detailed per-profile analysis
    for name, golden_url in golden.items():
        golden_norm = normalize_url(golden_url)

        # Check SERP results
        serp_candidates = serp_results.get(name, [])
        serp_has_correct = any(normalize_url(c['linkedin_url']) == golden_norm for c in serp_candidates)

        # Check LLM selections
        llm_selected = llm_selections.get(name, [])
        llm_selected_correct_profile = False
        llm_selected_wrong_profile = False

        if llm_selected:
            for sel in llm_selected:
                if normalize_url(sel['linkedin_url']) == golden_norm:
                    llm_selected_correct_profile = True
                else:
                    llm_selected_wrong_profile = True

        # Categorize
        if serp_has_correct:
            serp_found_correct += 1
            if llm_selected_correct_profile:
                llm_selected_correct += 1
                status = "✅ SERP FOUND + LLM CORRECT"
            elif llm_selected_wrong_profile:
                llm_selected_wrong += 1
                status = "❌ SERP FOUND + LLM WRONG"
            else:
                llm_no_selection += 1
                status = "⚠️  SERP FOUND + LLM NO SELECTION"
        else:
            serp_missing += 1
            status = "❌ SERP FAILED (golden not in candidates)"

        print(f"\n📋 {name}")
        print(f"   Golden: {golden_url}")
        print(f"   Status: {status}")
        print(f"   SERP candidates: {len(serp_candidates)}")
        print(f"   SERP has correct: {'✅ YES' if serp_has_correct else '❌ NO'}")

        if llm_selected:
            print(f"   LLM selections: {len(llm_selected)}")
            for i, sel in enumerate(llm_selected, 1):
                is_correct = normalize_url(sel['linkedin_url']) == golden_norm
                marker = "🎯" if is_correct else "❌"
                confidence = sel.get('llm_confidence', 'N/A')
                print(f"      {marker} {i}. {sel['linkedin_url']}")
                print(f"         Confidence: {confidence}")
                print(f"         Rationale: {sel.get('llm_rationale', 'N/A')[:100]}...")
        else:
            print(f"   LLM selections: 0 (no selection made)")

        # Show top SERP candidates for context
        if serp_candidates:
            print(f"   📊 Top 5 SERP candidates:")
            for i, candidate in enumerate(serp_candidates[:5], 1):
                is_golden = normalize_url(candidate['linkedin_url']) == golden_norm
                score = candidate.get('match_score', 'N/A')
                confidence = candidate.get('confidence', 'N/A')
                indicator = "🎯" if is_golden else "  "
                print(f"   {indicator} {i}. Score {score:>3} | {confidence:>6} | {candidate['linkedin_url']}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal profiles analyzed: {total_profiles}")
    print(f"\n🔍 SERP Performance:")
    print(f"   - Found correct profile: {serp_found_correct}/{total_profiles} ({serp_found_correct/total_profiles*100:.1f}%)")
    print(f"   - Missing correct profile: {serp_missing}/{total_profiles} ({serp_missing/total_profiles*100:.1f}%)")

    print(f"\n🤖 LLM Selection Performance (when SERP found correct):")
    if serp_found_correct > 0:
        print(f"   - Selected correct: {llm_selected_correct}/{serp_found_correct} ({llm_selected_correct/serp_found_correct*100:.1f}%)")
        print(f"   - Selected wrong: {llm_selected_wrong}/{serp_found_correct} ({llm_selected_wrong/serp_found_correct*100:.1f}%)")
        print(f"   - No selection: {llm_no_selection}/{serp_found_correct} ({llm_no_selection/serp_found_correct*100:.1f}%)")
    else:
        print(f"   - N/A (SERP found 0 correct profiles)")

    print(f"\n🎯 End-to-End Success Rate:")
    print(f"   - Correct matches: {llm_selected_correct}/{total_profiles} ({llm_selected_correct/total_profiles*100:.1f}%)")

    print(f"\n💡 Failure Attribution:")
    print(f"   - SERP failures (bottleneck): {serp_missing}/{total_profiles} ({serp_missing/total_profiles*100:.1f}%)")
    if serp_found_correct > 0:
        llm_failures = llm_selected_wrong + llm_no_selection
        print(f"   - LLM failures (when SERP found): {llm_failures}/{serp_found_correct} ({llm_failures/serp_found_correct*100:.1f}%)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_e2e()
