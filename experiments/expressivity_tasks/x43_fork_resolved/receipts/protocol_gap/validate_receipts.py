"""Independent receipt completeness checks; no GPU or model execution."""
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
checked = []
for context in (128,256):
    pair = []
    for slots in (48,257,2049):
        label = f'stress{slots}_t{context}_fixed'
        row = json.loads((root/f'{label}.json').read_text())
        terminal = json.loads((root/f'{label}_terminal.json').read_text())
        assert terminal['exit_code'] == 0
        assert row['status'] == 'measured' and len(row['rows']) == 5
        assert row['actual_parameters'] == 1301052902
        assert row['T'] == context and row['B'] == 16 and row['slots'] == slots
        assert row['peak_occupancy'] == min(context,slots)
        assert row['forward_graph_read_trace_byte_exact']
        assert row['source_sha256'] == hashlib.sha256((root/f'{label}_source.py.txt').read_bytes()).hexdigest()
        assert row['median_ms'] == statistics.median(r['total_ms'] for r in row['rows'][2:])
        assert 0 < row['max_memory_bytes'] <= row['max_reserved_bytes'] < row['device_total_bytes']
        assert row['inputs']['documents_index_prefix_matched']
        pair.append(row['inputs'])
        checked.append(label)
    assert pair[0] == pair[1] == pair[2], 'different source inputs across sibling capacities'
(root/'validation.json').write_text(json.dumps(dict(status='passed',checked=checked,
    notes='Six accepted measurements; five full steps each; medians independently recomputed, source snapshot hashes and matched source-document receipts verified.'),indent=2)+'\n')
print('PASS: six complete, matched, hash-bound measurements')
