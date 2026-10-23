"""Freeze X27 size predictions, actual parameter counts and fresh max-bin gates."""
from datetime import datetime, timezone
import json
import socket
import subprocess
import sys

import torch

from experiments.expressivity_tasks.run_small_architecture_separators import (
    ROOT, COMPARISON_ROOT, CARDS, CONFIGS, build_model, link_datasets,
)
from experiments.expressivity_tasks.run_merrill26_separators import digest, write_json


def register():
    if (ROOT / 'registration.json').exists():
        raise ValueError('X27 registration already exists; do not overwrite')
    link_datasets()
    prior_path = ROOT.parent / 'results_stage25_recipe_controlled' / 'registration.json'
    x25 = json.loads(prior_path.read_text())
    counts, smokes = {}, {}
    for name in CARDS:
        arm, task, _ = name.split('_')
        row = json.loads((ROOT / 'receipts' / f'smoke_{arm}_{task}.json').read_text())
        if (row['status'] != 'passed' or row['optimizer_updates'] != 0
                or row['n'] != 300 or row['tokens'] != 90904):
            raise ValueError('unqualified max-bin smoke: ' + name)
        torch.manual_seed(0)
        model = build_model(arm, task)
        total = sum(p.numel() for p in model.parameters())
        if not 200000 <= total <= 800000 or row['parameters'] != total:
            raise ValueError('small-scale parameter band or smoke count mismatch: ' + name)
        layer = model.layers[0]
        counts[name] = {'constructor_size': CONFIGS[arm], 'vocab_size': model.vocab_size,
            'mlp_ratio': model.mlp_ratio, 'mlp_hidden_dim': model.mlp_hidden_dim,
            'parameters': total, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'module_parameters': {key: sum(p.numel() for p in module.parameters())
                                  for key, module in model.named_children()},
            'parameter_shapes': {key: list(value.shape) for key, value in model.named_parameters()},
            'effective_recurrence': ({'heads': layer.n_heads, 'state': layer.n_state,
                'key_dim': layer.key_dim, 'value_dim': layer.value_dim,
                'state_activation': layer.state_activation, 'precision': layer.recurrent_state_precision,
                'fused_inference': layer.fused_inference} if arm != 'gdn2' else
                {'heads': layer.num_heads, 'key_head_dim': layer.actual_head_dim,
                 'value_head_dim': layer.gdn.v_proj.out_features // layer.num_heads,
                 'allow_neg_eigval': layer.allow_neg_eigval, 'short_conv': layer.gdn.use_short_conv}),
            'optimizer_updates': 0}
        smokes[name] = row
    write_json(ROOT / 'receipts' / 'parameter_counts.json', counts)
    qualified_sources = {name: sha for name, sha in x25['source_sha256'].items() if name.startswith('ndm/')}
    for name, sha in {**qualified_sources, **x25['dependency_sha256']}.items():
        if digest(name) != sha:
            raise ValueError('X25 qualified engine/dependency changed: ' + name)
    sources = set(qualified_sources) | {
        'experiments/expressivity_tasks/run_small_architecture_separators.py',
        'experiments/expressivity_tasks/run_recipe_controlled_separators.py',
        'experiments/expressivity_tasks/run_e97_softplus_graph.py',
        'experiments/expressivity_tasks/run_merrill26_separators.py',
        'experiments/expressivity_tasks/tasks/merrill26_separators.py',
        'scripts/launch_small_architecture_separators.sh',
        'scripts/watch_small_architecture_separators.py',
        'scripts/register_small_architecture_separators.py',
        'scripts/watch_merrill26_separators.py', 'scripts/gpu_lease.sh',
        'tests/test_small_architecture_separators.py', str(prior_path),
        str(ROOT.parent / 'results_stage26_replication' / 'registration.json')}
    sources.update(str(p) for p in ROOT.rglob('*') if p.is_file() and 'datasets' not in p.parts)
    dependencies = dict(x25['dependency_sha256'])
    dataset_receipts = {str(p.resolve()): digest(p)
                        for p in (COMPARISON_ROOT / 'datasets' / 'graph').glob('*.json')}
    dependencies.update(dataset_receipts)
    reg = {'task': 'X27', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python': sys.version, 'optimizer_updates_before_registration': 0,
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'protocol': str(ROOT / 'REGISTRATION.md'), 'runs': list(CARDS), 'cards': CARDS,
        'queue': 'card6 first; wait for idle/unleased X25 cards0 and1; no preemption',
        'corrections': {'concession_19': 'batch128 task directive corrected by supervisor to X25 batch64; X26 batch128 limits direct numeric architecture attribution'},
        'predictions': {
            'size-binding': 'small-ours >=99% ID and BOTH OOD where corresponding big-ours fail; size implicated, train-small then transplant/distill is a future production hypothesis',
            'architecture-binding': 'small-ours fail while X26 RNN learns/generalizes; architecture-class favored over size alone, cross-recipe limitations disclosed',
            'nothing-learns': 'small-ours and X26 small-theirs fail; task/encoding suspect, new native-tokenization control registration required, no automatic bug claim',
            'other': 'exact counts all cells; ID failure separate from OOD degradation; mixed outcomes retained; errors not scientific verdicts; missing bins right-censored'},
        'recipe': x25['recipe'], 'architecture': counts,
        'comparison': {'large': str(prior_path), 'small_theirs': 'results_stage26_replication',
                       'datasets': str(COMPARISON_ROOT / 'datasets'), 'dataset_receipts': dataset_receipts},
        'max_bin_zero_update_smokes': smokes, 'kernel_match': x25['kernel_match'],
        'source_sha256': {name: digest(name) for name in sorted(sources)},
        'dependency_sha256': dependencies}
    write_json(ROOT / 'registration.json', reg)


if __name__ == '__main__':
    register()
