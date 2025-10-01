#!/usr/bin/env python3
"""
Compare gpt-4o-mini vs gpt-5-nano performance
"""
import csv
from typing import Dict
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
    """Load the golden LinkedIn URLs."""
    golden = {}
    with open('/Users/ashterhaider/Downloads/build/perplexity_test/data/51_linkedin.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            golden[row['Full Name']] = row['linkedin_url']
    return golden

def load_llm_selections(filepath):
    """Load LLM selections from CSV."""
    selections = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['upwork_name']
            if row.get('llm_selected') == 'yes':
                selections[name] = {
                    'url': row['linkedin_url'],
                    'confidence': float(row.get('llm_confidence', 0) or 0),
                    'rationale': row.get('llm_rationale', '')
                }
    return selections

def main():
    golden = load_golden_dataset()

    gpt4_selections = load_llm_selections(
        '/Users/ashterhaider/Downloads/build/perplexity_test/upwork_linkedin_redo/llm_fixed_results.csv'
    )
    gpt5_selections = load_llm_selections(
        '/Users/ashterhaider/Downloads/build/perplexity_test/upwork_linkedin_redo/llm_gpt5nano_results.csv'
    )

    print("=" * 80)
    print("MODEL COMPARISON: gpt-4o-mini vs gpt-5-nano-2025-08-07")
    print("=" * 80)

    gpt4_correct = 0
    gpt5_correct = 0
    gpt4_made_selection = 0
    gpt5_made_selection = 0

    both_correct = 0
    gpt4_only_correct = 0
    gpt5_only_correct = 0
    both_wrong = 0

    differences = []

    for name, golden_url in golden.items():
        golden_norm = normalize_url(golden_url)

        gpt4_sel = gpt4_selections.get(name)
        gpt5_sel = gpt5_selections.get(name)

        gpt4_correct_match = False
        gpt5_correct_match = False

        if gpt4_sel:
            gpt4_made_selection += 1
            gpt4_correct_match = normalize_url(gpt4_sel['url']) == golden_norm
            if gpt4_correct_match:
                gpt4_correct += 1

        if gpt5_sel:
            gpt5_made_selection += 1
            gpt5_correct_match = normalize_url(gpt5_sel['url']) == golden_norm
            if gpt5_correct_match:
                gpt5_correct += 1

        # Track agreement
        if gpt4_correct_match and gpt5_correct_match:
            both_correct += 1
        elif gpt4_correct_match and not gpt5_correct_match:
            gpt4_only_correct += 1
            differences.append({
                'name': name,
                'winner': 'gpt-4o-mini',
                'gpt4': gpt4_sel,
                'gpt5': gpt5_sel,
                'golden': golden_url
            })
        elif gpt5_correct_match and not gpt4_correct_match:
            gpt5_only_correct += 1
            differences.append({
                'name': name,
                'winner': 'gpt-5-nano',
                'gpt4': gpt4_sel,
                'gpt5': gpt5_sel,
                'golden': golden_url
            })
        elif gpt4_sel and gpt5_sel and not gpt4_correct_match and not gpt5_correct_match:
            both_wrong += 1
            # Check if they selected different wrong answers
            if normalize_url(gpt4_sel['url']) != normalize_url(gpt5_sel['url']):
                differences.append({
                    'name': name,
                    'winner': 'both_wrong_different',
                    'gpt4': gpt4_sel,
                    'gpt5': gpt5_sel,
                    'golden': golden_url
                })

    print(f"\n📊 SELECTION STATS:")
    print(f"   gpt-4o-mini made selections: {gpt4_made_selection}/51")
    print(f"   gpt-5-nano made selections:  {gpt5_made_selection}/51")

    print(f"\n✅ CORRECT MATCHES:")
    print(f"   gpt-4o-mini: {gpt4_correct}/{gpt4_made_selection} ({gpt4_correct/gpt4_made_selection*100:.1f}%)" if gpt4_made_selection > 0 else "   gpt-4o-mini: 0/0")
    print(f"   gpt-5-nano:  {gpt5_correct}/{gpt5_made_selection} ({gpt5_correct/gpt5_made_selection*100:.1f}%)" if gpt5_made_selection > 0 else "   gpt-5-nano: 0/0")

    print(f"\n🤝 AGREEMENT:")
    print(f"   Both correct: {both_correct}")
    print(f"   Both wrong: {both_wrong}")
    print(f"   gpt-4o-mini only correct: {gpt4_only_correct}")
    print(f"   gpt-5-nano only correct: {gpt5_only_correct}")

    if differences:
        print(f"\n" + "=" * 80)
        print(f"🔍 DIFFERENCES ({len(differences)} cases):")
        print("=" * 80)

        for i, diff in enumerate(differences, 1):
            print(f"\n{i}. {diff['name']} - Winner: {diff['winner']}")
            print(f"   Golden: {diff['golden']}")

            if diff['gpt4']:
                print(f"   GPT-4o-mini:")
                print(f"      Selected: {diff['gpt4']['url']}")
                print(f"      Confidence: {diff['gpt4']['confidence']:.2f}")
                print(f"      Rationale: {diff['gpt4']['rationale'][:120]}...")
            else:
                print(f"   GPT-4o-mini: No selection")

            if diff['gpt5']:
                print(f"   GPT-5-nano:")
                print(f"      Selected: {diff['gpt5']['url']}")
                print(f"      Confidence: {diff['gpt5']['confidence']:.2f}")
                print(f"      Rationale: {diff['gpt5']['rationale'][:120]}...")
            else:
                print(f"   GPT-5-nano: No selection")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
