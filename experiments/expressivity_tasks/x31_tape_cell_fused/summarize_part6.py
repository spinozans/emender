"""Summarize the existing registered targets from part 6 receipts."""
import json
from pathlib import Path
from .train_gates import require_parity

ROOT=Path(__file__).resolve().parent


def main():
    def read(name):return json.loads((ROOT/name).read_text())
    require_parity(ROOT/'parity_part6.json')
    parity=read('parity_part6.json')
    cost=read('cost_part6.json')
    append=read('append_cost_part6.json')
    storage=read('storage_part6.json')
    checkpoint=next((x for x in cost['measurements'] if x['engine']=='x31_checkpoint'),{})
    remaining=[]
    if cost['gates'].get('s11776_speedup_ge_10') is not True:
        remaining.append('Large-store throughput: O(S) metadata vectors and spills remain inside one program per store; three launches per token, dense state allocation and no token-group amortization remain.')
    if cost['gates'].get('heads_ratio_le_2') is not True:
        remaining.append('Head scaling: scoring batches heads across the launch grid, but each query still pads 15 unused MMA rows; recurrent metadata work, dense base-state allocation and memory traffic scale with heads. There is no shared multi-query GEMM.')
    if cost['gates'].get('checkpoint_overhead_le_5pct') is not True:
        remaining.append('Checkpoint time: changed-bit capture scans vector spills, stages proposed values, packs ordered groups, synchronizes to size/compact logs, clones replay state and rebases addresses. A faster forward exposes these costs.')
    result=dict(full_parity=True,parity_tests=parity['tests'],
        forward_scope=cost['scope'],timing_gates=cost['gates'],
        x30_speedups=cost['x30_speedups'],heads_ratios=cost['heads16_to_1_by_slots'],
        checkpoint=dict(observed_fractional_time_delta=cost['checkpoint_overhead'],
            timing_gate=cost['gates']['checkpoint_overhead_le_5pct'],
            base_state_bytes=checkpoint.get('base_state_bytes'),
            reserved_bytes=checkpoint.get('journal_reserved_bytes'),
            recorded_entries=checkpoint.get('journal_recorded_entries'),
            used_record_bytes=None if not checkpoint else 20*checkpoint['journal_recorded_entries'],
            private_forward_replay_copy_bytes=checkpoint.get('private_replay_copy_bytes'),
            storage_independence_gate=storage['storage_independence_gate'],
            budget_scaling=storage['measurements'],
            storage_scope=storage['note']),
        append_phase_gates={str(x['slots']):dict(speedup=x['speedup'],gate=x['gate']) for x in append['measurements']},
        collection_errors=cost['errors'],residual_risks=remaining+[
            'Base/replay state, dense death diagnostics and transient capture staging remain O(S). Log storage independence does not assert total training memory independence.',
            'Retained journal capacity is at most twice the largest store event count per store; unequal store activity can waste a factor of stores. Capture reservation bounds potential live events and is released after compaction.',
            'Learning was not rerun in part 6; MQAR and ladder results remain historical part 5 measurements.'],
        learning_reference='summary_part5.json; not rerun in part 6',
        all_registered_targets_passed=all(x is True for x in cost['gates'].values())
            and storage['storage_independence_gate']
            and all(x['gate'] for x in append['measurements']))
    (ROOT/'summary_part6.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
