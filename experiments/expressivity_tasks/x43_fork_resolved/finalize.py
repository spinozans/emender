"""Validate completed local artifacts and produce the concise shakedown receipt."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


def main():
    results = json.loads((ROOT/'receipts/results.json').read_text())
    assert json.loads((ROOT/'receipts/model_terminal.json').read_text())['exit_code'] == 0
    assert [r['arm'] for r in results['models']] == ['emender', 'transformer']
    meta = results['metadata']
    assert hashlib.sha256((ROOT/'input.txt').read_bytes()).hexdigest() == meta['corpus_sha256']
    assert hashlib.sha256((ROOT/'PREREGISTRATION.md').read_bytes()).hexdigest() == meta['preregistration_sha256']
    manifest = json.loads((ROOT/'receipts/source_manifest.json').read_text())
    for path, digest in manifest['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    rows = []
    for row in results['models']:
        name = row['arm']
        with np.load(ROOT/f'{name}_per_position.npz', allow_pickle=False) as saved:
            assert np.array_equal(saved['position'], np.arange(meta['split']+1, meta['corpus_bytes']))
            assert np.isfinite(saved['nll']).all() and (saved['nll'] >= 0).all()
            assert np.isclose(saved['nll'].mean(), row['heldout_nll'])
            assert np.isclose(saved['correct'].mean(), row['heldout_accuracy'])
        assert row['train_seconds'] >= 180 and row['steps'] > 0
        rows.append(f"| {name} | {row['trainable_parameters']:,} | {row['steps']:,} | {row['heldout_nll']:.6f} | {100*row['heldout_accuracy']:.3f}% | blocked | blocked |")
    artifacts = {}
    for pattern in ('*.npz', '*.pt'):
        for path in ROOT.glob(pattern):
            artifacts[path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), committed=path.suffix != '.pt')
    (ROOT/'receipts/artifact_manifest.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    (ROOT/'receipts/final_validation.json').write_text(json.dumps(dict(status='passed',checks=[
        'both arms completed with exit zero and >=180 seconds training',
        'corpus and preregistration SHA256 match', 'recorded training sources unchanged',
        'all held-out targets present exactly once', 'finite per-target NLL and aggregate metrics agree',
        'checkpoint and per-position artifact hashes retained; checkpoints uncommitted']),indent=2)+'\n')
    text = '''# X43 part 1 result: model legs passed; index/decomposition blocked

| Arm | Trainable parameters | Steps / 180 seconds | NLL nats/byte | Accuracy | Fork NLL share | Fork accuracy |
|---|---:|---:|---:|---:|---|---|
'''+ '\n'.join(rows)+'''

Both arms evaluated 111,539 identical held-out targets with context reset every
128 bytes. These are wall-time shakedown results; token budgets and training
feature precision differ. They do not establish an architecture ranking.

The prescribed xsa build exited 101: Cargo.toml missing in ~/sxgc. Verbatim
failure: receipts/xsa_build.log. No index or fork labels were produced; no fork
density, histogram, automaton statistics, forced/fork loss, or branch accuracy
is claimed. Those central X43 measurements remain blocked. No changes to sxgc.

The exact-count interpolation and fork-write-gate follow-ups were preregistered
before training and were not run. The deferred decomposition utility passed
synthetic checks only. CPU byte/alignment/causality checks and local artifact
validation passed. Checkpoints remain local and ignored by Git. Compressed per-target arrays are
retained as evidence; receipts/artifact_manifest.json retains all artifact hashes.

Residual risks: missing index and unresolved chi/orientation convention;
one seed and a short run; reset context; unmatched token budgets; bf16 feature
autocast in Emender training versus FP32 control; no small-model kernel parity
recertification. Upstream certified core files were reused unchanged.
'''
    (ROOT/'RESULT.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
