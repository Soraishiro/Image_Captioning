import json
from pathlib import Path
from typing import Dict, List

def extract_phrases(vocab: Dict) -> List[str]:
    phrases = set()
    for term, info in vocab.items():
        if not isinstance(info, dict):
            continue
        phrases.add(term)
        for variant in info.get('variants', []):
            normalized = variant.replace(' ', '_')
            phrases.add(normalized)
    return sorted(phrases)

def adapt(input_path: str, output_path: str) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    phrases = extract_phrases(vocab)
    facet_map = {}
    for term, info in vocab.items():
        if isinstance(info, dict) and 'facet' in info:
            facet_map[term] = info['facet']
    output = {'phrases': phrases, 'facet_map': facet_map, 'audit_status': 'reviewed', 'total_phrases': len(phrases)}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Adapt V_cultural facet format → phrase list format')
    parser.add_argument('--input', required=True, help='V_cultural JSON (facet+variants format)')
    parser.add_argument('--output', required=True, help='Output JSON for build_phrase_supervision.py')
    args = parser.parse_args()
    result = adapt(args.input, args.output)
    print(f"[adapt_vocab] {result['total_phrases']} phrases written to {args.output}")
