import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np

def load_vocab(vocab_file: str) -> Set[str]:
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'phrases' in data:
        return set(data['phrases'])
    return set(data.keys())

def build_positive_sets(ann_file: str, vocab: Set[str]) -> Dict[int, List[str]]:
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_caps: Dict[int, List[str]] = {}
    for ann in data.get('annotations', []):
        cap = ann.get('segment_caption') or ann.get('caption', '')
        img_caps.setdefault(ann['image_id'], []).append(cap)
    positives: Dict[int, List[str]] = {}
    for img_id, captions in img_caps.items():
        img_p = []
        caps_norm = [c.replace('_', ' ').lower() for c in captions]
        for phrase in vocab:
            pat = phrase.replace('_', ' ')
            if any((pat in c for c in caps_norm)):
                img_p.append(phrase)
        if img_p:
            positives[img_id] = img_p
    return positives

def build_token_masks(ann_file: str, vocab: Set[str], max_len: int=25) -> Dict[int, List[int]]:
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_first: Dict[int, str] = {}
    for ann in data.get('annotations', []):
        if ann['image_id'] not in img_first:
            img_first[ann['image_id']] = ann.get('segment_caption') or ann.get('caption', '')
    masks: Dict[int, List[int]] = {}
    for img_id, caption in img_first.items():
        tokens = caption.split()
        tokens_norm = [t.replace('_', ' ').lower() for t in tokens]
        positions = []
        for phrase in vocab:
            pw = phrase.replace('_', ' ').split()
            n = len(pw)
            target = ' '.join(pw).lower()
            matched = False
            for i in range(len(tokens_norm) - n + 1):
                if ' '.join(tokens_norm[i:i + n]) == target:
                    for j in range(i, i + n):
                        if j < max_len and j not in positions:
                            positions.append(j)
                    matched = True
                    break
            if not matched and n > 1:
                for i, tok in enumerate(tokens_norm):
                    if tok == target and i < max_len and (i not in positions):
                        positions.append(i)
        if positions:
            masks[img_id] = sorted(positions)
    return masks

def encode_phrases(phrases: List[str], model_name: str='vinai/phobert-base') -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print('[build_supervision] WARNING: transformers not available, returning zeros. Install: pip install transformers torch')
        return np.zeros((len(phrases), 768), dtype=np.float32)
    segmented = [p.replace(' ', '_') for p in phrases]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    for i in range(0, len(segmented), 32):
        batch = segmented[i:i + 32]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt')
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls)
    return np.concatenate(embeddings, axis=0)

def main():
    import argparse
    p = argparse.ArgumentParser(description='Build LCQM phrase supervision')
    p.add_argument('--vocab_file', type=Path, required=True)
    p.add_argument('--train_annotations', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, default=Path('data/'))
    args = p.parse_args()
    vocab = load_vocab(str(args.vocab_file))
    positives = build_positive_sets(str(args.train_annotations), vocab)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / 'phrase_positives_train.json', 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in positives.items()}, f, ensure_ascii=False, indent=2)
    token_masks = build_token_masks(str(args.train_annotations), vocab)
    with open(args.output_dir / 'token_masks_train.json', 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in token_masks.items()}, f, ensure_ascii=False)
    phrase_list = sorted(vocab)
    emb = encode_phrases(phrase_list)
    np.save(args.output_dir / 'phrase_embeddings.npy', emb)
    with open(args.output_dir / 'phrase_list.json', 'w', encoding='utf-8') as f:
        json.dump(phrase_list, f, ensure_ascii=False, indent=2)
    print(f'[build_supervision] {len(positives)} images, {len(phrase_list)} phrases → {args.output_dir}')
if __name__ == '__main__':
    main()
