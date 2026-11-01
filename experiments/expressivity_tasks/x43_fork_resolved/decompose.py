"""Join verified corpus-order xsa labels to saved per-target predictions.

This does not derive labels. It requires a separately receipted, successfully
built index with an explicit orientation/chi convention.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(nll, correct, fork):
    if not (len(nll) == len(correct) == len(fork)):
        raise ValueError('length mismatch')
    if fork.dtype != np.bool_ or not np.isfinite(nll).all() or (nll < 0).any():
        raise ValueError('invalid labels/loss')
    total = float(nll.sum(dtype=np.float64))
    result = dict(targets=len(nll), total_nll=total,
                  fork_density=float(fork.mean()) if len(fork) else None)
    for name, mask in [('fork', fork), ('forced', ~fork)]:
        values = nll[mask]
        result[name] = dict(positions=int(mask.sum()),
            mean_nll=float(values.mean()) if len(values) else None,
            total_nll=float(values.sum(dtype=np.float64)),
            accuracy=float(correct[mask].mean()) if len(values) else None,
            fraction_total_nll=float(values.sum(dtype=np.float64)/total) if total else None)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', type=Path, required=True, help='npz: corpus-order bool fork array')
    p.add_argument('--provenance', type=Path, required=True, help='index provenance JSON')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).parent
    provenance = json.loads(args.provenance.read_text())
    digest = hashlib.sha256((root/'input.txt').read_bytes()).hexdigest()
    if provenance.get('corpus_sha256') != digest or provenance.get('status') != 'passed':
        raise ValueError('unverified index/corpus')
    for key in ('xsa_binary_sha256', 'index_sha256', 'chi_definition', 'orientation', 'label_derivation'):
        if not provenance.get(key):
            raise ValueError('missing index evidence: '+key)
    if provenance.get('labels_sha256') != hashlib.sha256(args.labels.read_bytes()).hexdigest():
        raise ValueError('unverified labels')
    with np.load(args.labels, allow_pickle=False) as labels:
        fork = labels['fork']
    if fork.dtype != np.bool_ or fork.shape != (len((root/'input.txt').read_bytes()),):
        raise ValueError('labels must cover all corpus positions in natural byte order')
    rows = []
    for arm in ('emender', 'transformer'):
        with np.load(root/f'{arm}_per_position.npz', allow_pickle=False) as evaluation:
            positions = evaluation['position']
            split = int(.9*len(fork))
            if not np.array_equal(positions, np.arange(split+1, len(fork))):
                raise ValueError('evaluation target coverage mismatch')
            rows.append(dict(arm=arm, **summarize(evaluation['nll'], evaluation['correct'], fork[positions])))
    args.output.write_text(json.dumps(dict(provenance=provenance, models=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
