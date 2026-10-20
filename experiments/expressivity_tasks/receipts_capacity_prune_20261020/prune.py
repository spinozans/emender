"""Attended emergency prune; only explicit --apply deletes pre-manifested weights.

Stage2b override: historical best-chain membership alone no longer pins a weight.
Routine inventories are records, not independent downstream evaluation pins.
Other roots keep best chains, legacy points, boundaries and latest; X5c keeps
all independently referenced receipt weights. Active roots are never pruned.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / 'experiments/expressivity_tasks'
OUT = Path(__file__).resolve().parent
TARGETS = ['2b', '5c_growth_8m', '2', '2c', '3_8m', '5b_growth_8m',
           '5_instrument', '4_13b', '6_truecur_8m']
MATCHED = {1000, 4000, 8000, 12000, 16000, 18269, *range(10000, 60001, 10000)}
ROUTINE = {'checkpoints.json', 'validation.json', 'training.jsonl', 'pruning.jsonl',
           'postcompletion_capacity_prune.json', 'checkpoint_inventory.json'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entry_hash(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, obj):
    with path.open('w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def best_chain(history):
    best = None
    chain = set()
    for row in history:
        if not all(k in row for k in ('accuracy', 'loss', 'step')):
            raise ValueError('Unrecognized validation row')
        key = (-row['accuracy'], row['loss'], row['step'])
        if best is None or key < best:
            best = key
            chain.add(row['step'])
    return chain


def references():
    # Scan tracked repository text, all stage-root receipts/reports, and the
    # operator's fix-fanout reports/reviews. Never scan checkpoint payloads.
    paths = {REPO / p for p in subprocess.check_output(
        ['git', 'ls-files'], cwd=REPO, text=True).splitlines()}
    for root in BASE.glob('results_invertible_stage*'):
        paths.update(p for p in root.rglob('*') if p.is_file() and p.suffix != '.pt')
    scratch = Path('/home/erikg/emender-scratch/fix-fanout')
    paths.update(scratch.glob('*report*.md'))
    paths.update(scratch.glob('*review*.md'))
    paths.update(scratch.glob('*manifest*.json'))
    hashes, names, sources, excluded = {}, {}, [], []
    for p in sorted(paths):
        if not p.is_file() or p.is_symlink() or OUT in p.parents:
            continue
        if p.name in ROUTINE:
            excluded.append(str(p))
            continue
        if p.suffix not in {'.json', '.jsonl', '.txt', '.log', '.md', '.typ', '.py', '.sh'}:
            continue
        try:
            text = p.read_text()
        except (UnicodeError, OSError):
            continue
        sources.append({'path': str(p), 'sha256': hashlib.sha256(text.encode()).hexdigest()})
        for h in set(re.findall(r'\b[0-9a-f]{64}\b', text)):
            hashes.setdefault(h, []).append(str(p))
        for match in re.finditer(r'step_[A-Za-z0-9_.-]+\.pt', text):
            prefix = re.search(r'([A-Za-z0-9_./-]+/)$', text[max(0, match.start() - 512):match.start()])
            name = (prefix[1] if prefix else '') + match[0]
            names.setdefault(name, []).append(str(p))
    return hashes, names, sources, excluded


def pin_sources(path, sha, hashes, names):
    sources = set(hashes.get(sha, []))
    # Bare filenames pin conservatively across runs; qualified paths match
    # only that run, preventing ordinary step_001000 names from false matches.
    for name, refs in names.items():
        if str(path).endswith('/' + name.lstrip('/')):
            sources.update(refs)
    return sorted(sources)


def root_plan(root, label, hashes, names, processes):
    active = [s for s in processes.splitlines() if str(root) in s]
    plan = {'root': str(root.relative_to(REPO)), 'active_processes': active,
            'removed': [], 'retained': [], 'ambiguities': []}
    prior = read_json(BASE / 'results_invertible_stage2b/receipts/postcompletion_capacity_prune.json')
    stops = set(prior['broadcast_x2b_stop_steps'])
    for run in sorted(p for p in root.iterdir() if p.is_dir() and p.name != 'receipts'):
        weights = sorted(run.glob('*.pt'))
        if not weights:
            continue
        ledger = read_json(run / 'checkpoints.json', {})
        by_path = {Path(c['path']).name: c for c in ledger.values() if isinstance(c, dict) and 'path' in c}
        result = read_json(run / 'result.json', read_json(run / 'pilot_result.json', {}))
        validation = read_json(run / 'validation.json', [])
        known = result.get('status') == 'complete'
        required = set()
        if label != '5c_growth_8m':
            required |= MATCHED
            if label == '2b':
                required |= stops
            else:
                try:
                    required |= best_chain(validation)
                except (ValueError, TypeError, KeyError):
                    known = False
            for key in ('step', 'stop_step', 'completed_updates', 'total_steps'):
                if isinstance(result.get(key), int):
                    required.add(result[key])
            # Curriculum entry/exit/best pins also appear in result.json; retain
            # boundary steps explicitly even if the receipt has no path/hash.
            for stage in result.get('stages', []):
                for key in ('entry_step', 'exit_step'):
                    if isinstance(stage.get(key), int):
                        required.add(stage[key])
        max_step = max(int(re.match(r'step_(\d+)', p.name)[1]) for p in weights)
        for p in weights:
            stat = p.lstat()
            cp = by_path.get(p.name, {})
            sha = cp.get('sha256', '')
            step = int(re.match(r'step_(\d+)', p.name)[1])
            reasons = []
            refs = pin_sources(p, sha, hashes, names)
            if refs:
                reasons.append('independent receipt/report/path/hash reference')
            if step in required:
                reasons.append('registered retained step class')
            if step == max_step:
                reasons.append('latest/final checkpoint')
            if label == '3_8m' and run.name.startswith('one_step_'):
                reasons.append('one-step diagnostic: retain entire run')
            if active:
                reasons.append('active root: no external retention race')
            if not known:
                reasons.append('ambiguous/incomplete run: fail-closed')
            if p.is_symlink() or stat.st_dev != REPO.stat().st_dev or stat.st_nlink != 1:
                reasons.append('ambiguous link/device: fail-closed')
            if not re.fullmatch('[0-9a-f]{64}', sha) or (cp.get('bytes', stat.st_size) != stat.st_size):
                reasons.append('missing/mismatched checkpoint ledger: fail-closed')
            entry = {'path': str(p.relative_to(REPO)), 'bytes': stat.st_size,
                     'allocated_bytes': stat.st_blocks * 512, 'checkpoint_sha256': sha,
                     'checkpoint_sha256_source': str((run / 'checkpoints.json').relative_to(REPO)),
                     'device': stat.st_dev, 'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}
            if reasons:
                entry.update(reasons=reasons, pin_sources=refs)
                plan['retained'].append(entry)
                if any('ambiguous' in s or 'mismatched' in s for s in reasons):
                    plan['ambiguities'].append(entry['path'])
            else:
                entry['reason'] = ('supervisor emergency override: historical best-chain inventory only'
                                   if label == '2b' else 'non-required intermediate rolling checkpoint')
                entry['manifest_entry_sha256'] = entry_hash(entry)
                plan['removed'].append(entry)
    plan['planned_freed_bytes'] = sum(e['bytes'] for e in plan['removed'])
    return plan


def evidence(root):
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix != '.pt'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if any((OUT / (label + '.json')).exists() for label in TARGETS):
        raise RuntimeError('Existing manifests: archive dry-run plans before applying; never overwrite execution receipts')
    hashes, names, sources, excluded = references()
    write_json(OUT / 'reference_scan.json', {'sources': sources, 'routine_records_excluded': excluded})
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    summary = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'initial_free_bytes': shutil.disk_usage(REPO).free, 'applied': args.apply,
               'authority': 'taskCAPACITY-prune.md plus live supervisor emergency override 2026-10-20',
               'emergency': 'Root 100%, live run Errno 28 data loss, W3 production arms blocked',
               'override': 'Stage2b historical best-chain membership alone is no longer a pin. Keep final/best, legacy/matched, broadcast/own stops and all downstream receipt/report/evaluation pins. Routine checkpoints.json, validation.json and historical prune inventories are existence records, not independent pins.',
               'superseded_required_set_manifest': 'experiments/expressivity_tasks/results_invertible_stage2b/receipts/postcompletion_capacity_prune.json',
               'practical_gate': 'Supervisor revised >=500G target to >=200G gate with margin; >=300G preferred',
               'targets': {}}
    for label in TARGETS:
        root = BASE / ('results_invertible_stage' + label)
        manifest_path = OUT / (label + '.json')
        if args.apply and manifest_path.exists():
            raise RuntimeError(f'Refusing to overwrite a prior plan: {manifest_path}')
        processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
        plan = root_plan(root, label, hashes, names, processes)
        before = evidence(root)
        plan['non_checkpoint_evidence_sha256'] = before
        plan['registered_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        plan['mode'] = 'apply' if args.apply else 'dry-run'
        write_json(manifest_path, plan)  # durable BEFORE any unlink
        freed = 0
        if args.apply:
            current = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
            if plan['removed'] and any(str(root) in line for line in current.splitlines()):
                raise RuntimeError(f'Root became active after registration: {root}')
            with (OUT / (label + '.execution.jsonl')).open('x') as log:
                for e in plan['removed']:
                    p = REPO / e['path']
                    st = p.lstat()
                    assert not p.is_symlink() and st.st_nlink == 1
                    assert (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) == (e['device'], e['inode'], e['bytes'], e['mtime_ns'])
                    p.unlink()
                    log.write(json.dumps({'path': e['path'], 'manifest_entry_sha256': e['manifest_entry_sha256'], 'removed_bytes': e['bytes']}) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                    freed += e['bytes']
            after = evidence(root)
            changed = [p for p, h in before.items() if after.get(p) != h]
            # Active roots are retained in full and their evidence may advance.
            if not plan['active_processes']:
                assert not changed, changed
            for e in plan['retained']:
                assert (REPO / e['path']).exists(), e['path']
            plan['validation'] = {'removed_absent': all(not (REPO / e['path']).exists() for e in plan['removed']),
                                  'retained_present': True, 'non_checkpoint_evidence_unchanged': not changed,
                                  'active_evidence_changes': changed if plan['active_processes'] else []}
            write_json(manifest_path, plan)
        summary['targets'][label] = {'manifest': str(manifest_path.relative_to(REPO)),
                                     'removed_count': len(plan['removed']), 'freed_bytes': freed,
                                     'planned_freed_bytes': plan['planned_freed_bytes'],
                                     'retained_count': len(plan['retained']), 'ambiguities': plan['ambiguities']}
        summary['final_free_bytes'] = shutil.disk_usage(REPO).free
        write_json(OUT / 'summary.json', summary)
        print(label, json.dumps(summary['targets'][label]), flush=True)
    print(subprocess.check_output(['df', '-h', '/'], text=True), flush=True)


if __name__ == '__main__':
    main()
