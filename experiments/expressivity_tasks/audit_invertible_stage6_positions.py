"""CPU-only acceptance audit of frozen X6D receipts and retained boundary evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.expressivity_tasks import eval_invertible_stage6_positions as diagnostic


def model_digest(state):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode() + b'\0')
        h.update(str(value.dtype).encode() + b'\0')
        h.update(str(tuple(value.shape)).encode() + b'\0')
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def audit(evidence=diagnostic.EVIDENCE):
    import torch
    inventory = diagnostic.read_json(evidence / 'inventory.json')
    summary = diagnostic.read_json(evidence / 'evaluation_summary.json')
    receipts = {}
    for item in summary['receipts']:
        assert diagnostic.digest(item['path']) == item['sha256']
        receipt = diagnostic.read_json(Path(item['path']))
        comparison = diagnostic.reconcile(receipt['aggregate'], receipt['positions'], receipt['recorded'])
        assert comparison == receipt['reconciliation']
        assert comparison['recorded_counts_match'] and abs(comparison['recorded_loss_delta']) <= 1e-6
        assert all(row['count'] == 512 and row['out_of_range'] == 0 for row in receipt['positions'])
        for source, sha in receipt['sources'].items():
            assert diagnostic.digest(source) == sha
        receipts[item['path']] = receipt
    for alias in summary['aliases']:
        receipt = receipts[alias['receipt']]
        comparison = diagnostic.reconcile(receipt['aggregate'], receipt['positions'], alias['recorded'])
        assert comparison == alias['reconciliation']
        assert comparison['recorded_counts_match'] and abs(comparison['recorded_loss_delta']) <= 1e-6
    result = {'audited_utc': diagnostic.utc(), 'source': str(Path(__file__)),
              'source_sha256': diagnostic.digest(__file__), 'gpu_re_evaluations': 0,
              'receipts': len(receipts), 'aliases': len(summary['aliases']),
              'maximum_absolute_loss_delta': max(abs(r['reconciliation']['recorded_loss_delta']) for r in receipts.values()),
              'boundary_checks': [], 'pruned_entries': [], 'training_source_vs_future_fix': {}}
    for name, info in inventory['runs'].items():
        assert diagnostic.digest(info['metadata']) == info['sha256']
        meta = diagnostic.read_json(Path(info['metadata']))
        config = meta['config']
        training_hash = next(v for k, v in config['source_sha256'].items() if k.endswith('run_invertible_stage6_truecur_8m.py'))
        result['training_source_vs_future_fix'][name] = {'training_sha256': training_hash,
            'future_launch_sha256': diagnostic.digest(diagnostic.runner.__file__),
            'different': training_hash != diagnostic.digest(diagnostic.runner.__file__)}
        data_path = Path(meta['snapshot_data_path'])
        assert diagnostic.digest(data_path) == meta['data_receipt']['sha256']
        cp_by_path = {job['checkpoint']['path']: job['checkpoint'] for job in meta['jobs']}
        for cp in cp_by_path.values():
            assert diagnostic.digest(cp['snapshot_path']) == cp['sha256']
        for missing in meta['missing_entry_checkpoints']:
            assert missing['pruning_events']
            result['pruned_entries'].append({'name': name, **missing})
        for boundary in meta['boundaries']:
            assert boundary['recorded_metrics_exactly_equal']
            entry = boundary['entry_checkpoint']
            carried = boundary['carried_checkpoint']
            record = {'name': name, **boundary,
                      'direct_positional_evidence': boundary['entry_payload_present'] or boundary['payload_sha_equal']}
            if boundary['entry_payload_present']:
                entry_raw = torch.load(cp_by_path[entry['path']]['snapshot_path'], map_location='cpu', weights_only=False)
                carried_raw = torch.load(cp_by_path[carried['path']]['snapshot_path'], map_location='cpu', weights_only=False)
                record['entry_model_sha256'] = model_digest(entry_raw['model'])
                record['carried_model_sha256'] = model_digest(carried_raw['model'])
                record['model_tensors_equal'] = entry_raw['model'].keys() == carried_raw['model'].keys() and all(
                    torch.equal(v, carried_raw['model'][k]) for k, v in entry_raw['model'].items())
                assert record['model_tensors_equal']
                def find(cp):
                    return next(r for r in receipts.values() if r['name'] == name and
                                r['checkpoint']['sha256'] == cp['sha256'] and r['pool_depth'] == boundary['from_depth'])
                record['positional_rows_equal'] = find(entry)['positions'] == find(carried)['positions']
                assert record['positional_rows_equal']
                del entry_raw, carried_raw
            else:
                assert boundary['entry_pruning_events']
            result['boundary_checks'].append(record)
    result['boundaries_total'] = len(result['boundary_checks'])
    result['direct_boundaries'] = sum(b['direct_positional_evidence'] for b in result['boundary_checks'])
    result['aggregate_only_boundaries'] = result['boundaries_total'] - result['direct_boundaries']
    result['passed'] = True
    diagnostic.write_json(evidence / 'audit.json', result)
    print(f"PASS: {result['receipts']} positional receipts + {result['aliases']} aliases; "
          f"loss delta <= {result['maximum_absolute_loss_delta']}; "
          f"{result['boundaries_total']} equal-metric boundaries, {result['direct_boundaries']} direct; "
          f"{len(result['pruned_entries'])} pruned T1-T16 entries; no GPU rerun")


if __name__ == '__main__':
    audit()
