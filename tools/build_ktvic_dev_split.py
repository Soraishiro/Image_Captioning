from __future__ import annotations
import argparse
import json
import random
from hashlib import sha256
from pathlib import Path
CAPTION_FIELD = 'segment_caption'
RAW_CAPTION_FIELD = 'caption'

def normalize_segment_caption(text: object) -> str:
    return ' '.join(str(text).strip().split())

def resolve_caption_text(annotation: dict) -> str:
    seg = annotation.get(CAPTION_FIELD)
    if seg:
        return normalize_segment_caption(seg)
    return normalize_segment_caption(annotation.get(RAW_CAPTION_FIELD, ''))

def _sha256_of(path: Path) -> str:
    digest = sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _write_json(path: Path, payload: object) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=4, ensure_ascii=False)

def _subset(data: dict, image_ids: set[int]) -> dict:
    images = [im for im in data['images'] if im['id'] in image_ids]
    annotations = [an for an in data['annotations'] if an['image_id'] in image_ids]
    return {'images': images, 'annotations': annotations}

def build_split(train_json: Path, out_dir: Path, seed: int, valid_frac: float) -> dict:
    data = json.load(open(train_json, encoding='utf-8'))
    all_ids = sorted((im['id'] for im in data['images']))
    n_total = len(all_ids)
    if len(set(all_ids)) != n_total:
        raise ValueError('Duplicate image ids in train_data.json — split would be ambiguous.')
    shuffled = list(all_ids)
    random.Random(seed).shuffle(shuffled)
    n_valid = round(n_total * valid_frac)
    valid_ids = set(shuffled[:n_valid])
    train_ids = set(shuffled[n_valid:])
    assert valid_ids.issubset(set(all_ids)), 'dev ids escaped the train id set'
    assert train_ids.isdisjoint(valid_ids), 'train and dev overlap'
    assert train_ids | valid_ids == set(all_ids), 'split does not cover every image'
    assert len(valid_ids) == n_valid, 'dev count mismatch'
    assert len(train_ids) == n_total - n_valid, 'train count mismatch'
    out_dir.mkdir(parents=True, exist_ok=True)
    train_main = _subset(data, train_ids)
    valid_main = _subset(data, valid_ids)
    vocab_captions = [resolve_caption_text(an) for an in data['annotations']]
    train_path = out_dir / 'train_main.json'
    valid_path = out_dir / 'valid_main.json'
    vocab_path = out_dir / 'vi_captions_train_only.json'
    _write_json(train_path, train_main)
    _write_json(valid_path, valid_main)
    _write_json(vocab_path, vocab_captions)
    manifest = {'seed': seed, 'valid_frac': valid_frac, 'source_train_json': str(train_json), 'counts': {'images_total': n_total, 'images_train': len(train_ids), 'images_valid': len(valid_ids), 'annotations_train': len(train_main['annotations']), 'annotations_valid': len(valid_main['annotations']), 'vocab_captions': len(vocab_captions)}, 'sha256': {'source_train_json': _sha256_of(train_json), 'train_main.json': _sha256_of(train_path), 'valid_main.json': _sha256_of(valid_path), 'vi_captions_train_only.json': _sha256_of(vocab_path)}}
    _write_json(out_dir / 'split_manifest.json', manifest)
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description='Build deterministic KTVIC dev split + vocab source.')
    parser.add_argument('--train-json', required=True, type=Path, help='Path to full train_data.json')
    parser.add_argument('--out-dir', required=True, type=Path, help='Output split directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--valid-frac', type=float, default=0.1)
    args = parser.parse_args()
    manifest = build_split(args.train_json, args.out_dir, args.seed, args.valid_frac)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
