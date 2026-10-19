#!/usr/bin/env python3
"""Read-only receipt census; stdout is JSON, --table emits the per-run Markdown table.

Never imports trainers, loads checkpoints, follows symlinks, or writes volume roots.
Sample schedules are NOT learning-rate schedules. Unknowns remain unknown.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from functools import lru_cache

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [Path(f'/mnt/nvme{i}n1/erikg/diloco_8gpu') for i in (1, 2)] + [
    Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')]
PRUNE = {'worktree', '.git', '.cache', '__pycache__', 'node_modules', 'repo-cache',
         'llama.cpp-fork', 'sources', 'shards', 'authority', 'packs', 'data', 'episodes',
         'debug-work', 'work', 'work-retry', 'replay-full', 'replay-retry',
         'pool', 'pending', 'completed', 'failed'}
IDENTITIES = {'args.json', 'launch.json', 'recipe.json', 'command.txt', 'run.sh'}
LOGS = {'training.jsonl', 'console.log'}


def online_family(path):
    s = str(path)
    for part, label in [('bank-pilot2', '48 bank pilot/shakeout'),
                        ('bank-merge-shakeout', '48 bank pilot/shakeout'),
                        ('bank-singlelearner-v1', '50 single-learner A'),
                        ('bank-singlelearner-B', '51 single-learner B'),
                        ('bank', '49 multi-lane bank'),
                        ('smoke', '52 PG smoke/screens'),
                        ('F7-probe', '53 F7/F8 staged controls'),
                        ('F7-staged', '53 F7/F8 staged controls'),
                        ('F8-staged', '53 F7/F8 staged controls')]:
        if f'/{part}/' in s:
            return label
    if '/e97-rl-loop-v1/training/' in s:
        return '47 online correction SFT cycles'
    return '53 other online controls'
EVAL_NAMES = {'summary.json', 'verdict-receipt.json', 'measurement-receipt.json',
              'audit.json', 'result.json', 'results.json', 'metrics.json', 'metadata.json',
              'wash-build-summary.json'}


def walk(root):
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in PRUNE and not d.startswith('triton') and 'worktree' not in d
                         and not d.startswith('rank-') and not Path(base, d).is_symlink())
        for name in sorted(files):
            yield Path(base, name)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_root(path):
    return path.parent.parent if path.parent.name in {'identity', 'logs'} else path.parent


def log_snapshot(path):
    # Hash exactly the bytes parsed, even if another process appends a live stream.
    raw = path.read_bytes()
    rows = []
    for line in raw.decode(errors='replace').splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest(), len(raw)


def scalar_metrics(value, prefix='', depth=0):
    """Keep score/identity scalars, not prompts, answers, per-case rows or private text."""
    out = {}
    if depth > 7 or not isinstance(value, dict):
        return out
    for k, v in value.items():
        key = f'{prefix}.{k}' if prefix else k
        if k in {'rows', 'outcomes', 'per_episode', 'cases', 'disagreements', 'source_inventory'}:
            continue
        if isinstance(v, dict):
            out.update(scalar_metrics(v, key, depth + 1))
        elif isinstance(v, (int, float, bool)) or v is None:
            out[key] = v
        elif isinstance(v, list) and all(isinstance(x, (int, float, bool)) for x in v):
            out[key] = v
        elif isinstance(v, str) and re.search(
                r'checkpoint|sha256|verdict|interpretation|status|optimizer|schedule|loss|mode|source_commit', k):
            out[key] = v
    return out


@lru_cache(maxsize=None)
def historical_source(commit, filename):
    if not commit or not re.fullmatch(r'[0-9a-f]{7,40}', commit):
        return ''
    result = subprocess.run(['git', 'show', f'{commit}:{filename}'], cwd=REPO,
                            capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ''


def classify(config, command, start):
    policy = start.get('sft_precision', {})
    opt = config.get('optimizer', policy.get('optimizer'))
    source = config.get('source_commit', start.get('source_commit'))
    sft = ('train_e97_4b_pi_sft.py' in command or
           any(t in str(start.get('trainer', '')) for t in ('rl_train_step', 'rl_policy_gradient_step')))
    # Start records of older SFTs have no trainer field, but have this SFT-only identity.
    sft = sft or ('parent_checkpoint' in start and 'start_update' in start)
    filename = 'scripts/train_e97_4b_pi_sft.py' if sft else 'train.py'
    code = historical_source(source, filename)
    evidence = f'{source}:{filename}' if code else 'receipt only; historical source unavailable'
    sf = opt in ('schedulefree', 'bf16-sr-candidate') or (sft and 'AdamWScheduleFree' in code)
    if sf:
        klass = 'AdamWScheduleFree'
        if policy.get('optimizer') == 'bf16-sr-candidate' or '--optimizer-precision bf16-sr-candidate' in command:
            klass = 'ScheduleFreeSRCandidate'
        elif '--offload-schedulefree-state' in command or 'pinned-cpu' == start.get('optimizer_state_storage'):
            klass = 'CPUOffloadAdamWScheduleFree'
        # These production trainers have no external schedule; verify the historical implementation.
        anneal = bool(re.search(r'--(?:lr-schedule|lr_schedule)\s+(?:cosine|linear)', command))
        return klass, 'annealed-SF' if anneal else 'constant-lr-SF', evidence
    if opt == 'adamw':
        return 'AdamW', 'annealed' if 'lr_scale_at(' in code else 'unknown', evidence
    return str(opt or 'unknown'), 'unknown', evidence


def family(root):
    """Disjoint production/research-context families; exports remain separately visible."""
    s, name = str(root), root.name
    if '/emender/runs/' in s or '/gdn2_mlp' in s:
        return '03 older E97/GDN2 pretrain context'
    if 'lr_interference_scan' in s:
        return '02 local pretrain LR-interference scans'
    if 'e97_4b_pi_instruction_local/runs/' in s:
        checks = [
            ('pi-sft-local-', '04 initial Pi SFT q/u chain'),
            ('pi-finalization-', '05 final-only repair'),
            ('pi-live-aligned-repair-', '06 balanced live-aligned repair'),
            ('pi-compositional-', '07 compositional and recovery repairs'),
            ('pi-recover-read-', '07 compositional and recovery repairs'),
            ('pi-cumulative-', '07 compositional and recovery repairs'),
            ('pi-v2-', '08 v2 path fidelity'), ('pi-v3-', '09 v3 on-policy/action/retention'),
            ('post-broad-', '17 post-broad exact/focused repairs'),
            ('broad-dual-prompt-64k-', '11 broad 64K'),
            ('broad-', '10 broad 4K'), ('pi-sft-64k-', '12 64K systems probes'),
            ('mixed-document-', '13 mixed-document boundary probes'),
            ('scaled-document-', '15 scaled-document LR5e-5' if 'lr5e5' in name else '14 scaled-document LR1e-6'),
            ('general-continuation-', '16 general continuation'),
            ('experimental-', '18 experimental correction')]
        for pattern, label in checks:
            if pattern in name:
                return label
    if 'e97_4b_systematic_representation' in s:
        return '19 systematic action/private representation'
    checks = [
        ('private-analysis-lr5e5-screen', '20 private-analysis LR5e-5 screen'),
        ('native-lr-screen', '21 native smoke/LR screen'),
        ('native-real-data-smoke', '21 native smoke/LR screen'),
        ('native-training-50m-agent-lr1e5', '22 native 880-update programme'),
        ('grounding-correction-v1-train', '23 grounded correction'),
        ('grounded-expansion-v1-train', '24 grounded expansion'),
        ('representation-bridge-v1-train', '25 representation bridge'),
        ('pi-native-curriculum-training', '26 Pi-native curriculum'),
        ('pi-native-repair-training', '27 Pi-native repair1'),
        ('pi-native-repair2-training', '28 Pi-native repair2'),
        ('pi-native-repair3-training', '29 Pi-native repair3'),
        ('pi-native-repair4-training', '30 Pi-native repair4'),
        ('pi-native-repair5-', '31 Pi-native repair5 arc'),
        ('pi-native-repair6-', '32 Pi-native repair6 arc'),
        ('pi-native-repair7-', '33 Pi-native repair7 arc'),
        ('pi-native-repair8-', '34 Pi-native repair8 arc'),
        ('pi-native-repair9-dose-screen', '38 repair9 dose screen'),
        ('pi-native-repair9r-full-arc-v3', '37 repair9r v3 arc'),
        ('pi-native-repair9r-full-arc-', '36 repair9r arc'),
        ('pi-native-repair9-full-', '35 repair9 full arc'),
        ('pi-native-throughput-profile', '44 throughput controls'),
        ('e97-throughput-fix', '44 throughput controls'),
        ('e97-e1-chat-agent', '39 E1 chat-agent'),
        ('e97-e2-chat-agent', '40 E2 chat-agent'),
        ('e97-e3-chat-agent', '41 E3 chat-agent'),
        ('e97-e4-corpus-arc', '42 E4 corpus arc'),
        ('e97-e3-sfparam-screen', '43 SF parameter screen'),
        ('wash-v1', '45 wash-1'), ('wash-v2', '46 wash-2')]
    for pattern, label in checks:
        if pattern in s:
            return label
    if '/diloco_8gpu/' in s:
        return '01 local/Frontier 4B pretrain and exports'
    return 'UNMAPPED ' + name


def extract_run(root, paths):
    config, command, receipts = {}, '', []
    for path in sorted(paths):
        if path.name not in IDENTITIES:
            continue
        receipts.append({'path': str(path), 'sha256': digest(path)})
        try:
            if path.suffix == '.json':
                data = load(path)
                if isinstance(data, dict):
                    config.update(data)
            else:
                command += '\n' + path.read_text(errors='replace')
        except (ValueError, OSError):
            pass
    starts, steps, used_logs, completions, checkpoints = [], {}, [], [], []
    for path in sorted(paths, key=lambda p: p.name != 'training.jsonl'):
        if path.name not in LOGS:
            continue
        local, log_sha, log_bytes = log_snapshot(path)
        if not any(r.get('event') in {'start', 'step'} for r in local):
            continue
        used_logs.append({'path': str(path), 'sha256': log_sha, 'snapshot_bytes': log_bytes})
        for row in local:
            if row.get('event') == 'start':
                starts.append(row)
            if row.get('event') == 'complete':
                completions.append(scalar_metrics(row))
            if row.get('event') == 'checkpoint':
                checkpoints.append(scalar_metrics(row))
            if row.get('event') == 'step' and 'update' in row:
                steps.setdefault(row['update'], row)
        if steps:  # console and JSONL commonly duplicate the same stream
            break
    start = starts[-1] if starts else {}
    effective = {**config, **start}
    for flag in ('lr', 'steps', 'warmup-steps', 'weight-decay', 'context-size'):
        match = re.search(r'--' + flag + r'(?:\s+|=)([0-9.eE+-]+)', command)
        if match:
            effective.setdefault(flag.replace('-', '_'), float(match[1]))
    klass, regime, basis = classify(effective, command, start)
    ordered = [steps[k] for k in sorted(steps)]
    losses = [r['loss'] for r in ordered if isinstance(r.get('loss'), (int, float))]
    window = min(32, len(losses))
    return {'root': str(root), 'family': family(root), 'optimizer_class': klass, 'regime': regime,
            'source_basis': basis, 'lr': effective.get('lr'),
            'warmup_steps': effective.get('warmup_steps', start.get('sft_precision', {}).get('warmup_steps')),
            'planned_steps': effective.get('steps'), 'observed_updates': len(steps),
            'update_range': [min(steps), max(steps)] if steps else [],
            'dose_input_tokens': sum(r.get('global_tokens', 0) for r in ordered) or None,
            'dose_target_tokens': sum(r.get('global_targets', 0) for r in ordered) or None,
            'loss_first': losses[0] if losses else None, 'loss_last': losses[-1] if losses else None,
            'loss_first32_mean': sum(losses[:window])/window if window else None,
            'loss_last32_mean': sum(losses[-window:])/window if window else None,
            'config': scalar_metrics(effective), 'identities': receipts, 'logs': used_logs,
            'completion_receipts': completions, 'checkpoint_receipts': checkpoints,
            'outcomes': []}


def online_census(paths):
    groups = {}
    for path in paths:
        rows, log_sha, log_bytes = log_snapshot(path)
        starts = [r for r in rows if r.get('event') == 'start']
        if not starts:
            continue
        start = starts[0]
        name = online_family(path)
        group = groups.setdefault(name, dict(family=name, logs=[], regimes=Counter(),
            lrs=Counter(), observed_updates=0, dose_input_tokens=0, dose_target_tokens=0,
            loss_first_sum=0., loss_last_sum=0., loss_streams=0, dose_rows_logged=0))
        opt, regime, basis = classify(start, '', start)
        # PG logs record hyperparameters under parameters rather than SFT precision policy.
        params = start.get('parameters', {})
        # Objective differs even when PG reuses the historical SFT optimizer builder.
        objective = 'policy-gradient' if 'rl_policy_gradient_step' in str(start.get('trainer', '')) else 'masked-SFT'
        lr = start.get('lr', params.get('lr', params.get('learning_rate')))
        steps = [r for r in rows if r.get('event') == 'step']
        losses = [r['loss'] for r in steps if isinstance(r.get('loss'), (int, float))]
        group['logs'].append(dict(path=str(path), sha256=log_sha, snapshot_bytes=log_bytes, optimizer=str(opt),
                                 regime=regime, objective=objective, source_basis=basis, lr=lr, observed_updates=len(steps)))
        group['regimes'][regime] += 1
        group['lrs'][str(lr)] += 1
        group['observed_updates'] += len(steps)
        group['dose_rows_logged'] += sum('global_tokens' in r and 'global_targets' in r for r in steps)
        for field, key in [('dose_input_tokens', 'global_tokens'), ('dose_target_tokens', 'global_targets')]:
            group[field] += sum(r.get(key, 0) for r in steps)
        if losses:
            group['loss_first_sum'] += losses[0]
            group['loss_last_sum'] += losses[-1]
            group['loss_streams'] += 1
    return list(groups[k] for k in sorted(groups))


def census(roots):
    groups, evaluations, errors, online_logs = {}, [], [], []
    for root in roots:
        if not root.exists():
            errors.append(f'missing root: {root}')
            continue
        for path in walk(root):
            if path.name == 'log.jsonl' and '/e97-rl-loop-v1/' in str(path):
                online_logs.append(path)
            if path.name in IDENTITIES | LOGS:
                groups.setdefault(run_root(path), []).append(path)
            # Per-task bank rollouts are not model-level outcome receipts.
            rollout = any(f'/{name}/' in str(path) for name in
                          ('bank', 'bank-singlelearner-B', 'bank-singlelearner-v1', 'F7-probe'))
            if path.name == 'results.csv' and 'lr_interference_scan' in str(path):
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                evaluations.append(dict(path=str(path), sha256=digest(path), metrics={'scan_rows': rows}))
            receipt = ('/bank-heldout-reads/' in str(path) and path.name.endswith('.json')
                       and 'receipt' in path.name)
            if (path.name in EVAL_NAMES or receipt) and not rollout and path.stat().st_size < 20_000_000:
                try:
                    value = load(path)
                except (ValueError, OSError) as exc:
                    errors.append(f'{path}: {exc}')
                    continue
                metrics = scalar_metrics(value)
                if metrics:
                    evaluations.append({'path': str(path), 'sha256': digest(path), 'metrics': metrics})
    runs = []
    for root, paths in sorted(groups.items()):
        # Exclude evaluation run.sh and unexecuted launch plans; retain args-only pretrain/export receipts.
        if not any(p.name in LOGS or p.name in {'args.json', 'launch.json', 'recipe.json'} for p in paths):
            continue
        record = extract_run(root, paths)
        if not record['observed_updates'] and not any(p.name == 'args.json' or
                (p.name == 'launch.json' and p.parent.name == 'identity') for p in paths):
            continue
        for evaluation in evaluations:
            refs = [v for k, v in evaluation['metrics'].items()
                    if 'checkpoint' in k and isinstance(v, str)]
            if any(v.startswith(str(root) + '/') for v in refs):
                record['outcomes'].append(evaluation['path'])
        runs.append(record)
    return {'schema': 'emender-retroactive-regime-audit-v1', 'roots': list(map(str, roots)),
            'regime_counts': dict(Counter(r['regime'] for r in runs)),
            'runs': runs, 'evaluations': evaluations, 'online_families': online_census(online_logs),
            'errors': errors}


def table(data):
    print('| # | Run root | Optimizer / regime | LR; warmup | observed updates (range); planned | Input / target dose | Loss first32 → last32 | Exact-path outcome links |')
    print('|---:|---|---|---|---|---|---|---|')
    for n, r in enumerate(data['runs'], 1):
        def fmt(v):
            return '?' if v is None else f'{v:.6g}' if isinstance(v, float) else str(v)
        print(f"| {n} | `{r['root']}` | {r['optimizer_class']} / {r['regime']} | {fmt(r['lr'])}; {fmt(r['warmup_steps'])} | {r['observed_updates']} {r['update_range']}; {fmt(r['planned_steps'])} | {fmt(r['dose_input_tokens'])} / {fmt(r['dose_target_tokens'])} | {fmt(r['loss_first32_mean'])} → {fmt(r['loss_last32_mean'])} | {len(r['outcomes'])} (JSON paths) |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path)
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--from-json', type=Path)
    args = parser.parse_args()
    data = load(args.from_json) if args.from_json else census(args.root or DEFAULT_ROOTS)
    if args.table:
        table(data)
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
