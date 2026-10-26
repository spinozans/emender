"""Summarize unchanged registered targets from part 7 receipts."""
import json
from pathlib import Path
from .train_gates import require_parity

ROOT=Path(__file__).resolve().parent


def main():
    def read(name):return json.loads((ROOT/name).read_text())
    require_parity(ROOT/'parity_part7.json')
    parity=read('parity_part7.json')
    cost=read('cost_part7.json')
    append=read('append_cost_part7.json')
    storage=read('storage_part7.json')
    checkpoint=next((x for x in cost['measurements'] if x['engine']=='x31_checkpoint'),{})
    remaining=[]
    if cost['gates'].get('heads_ratio_le_2') is not True:
        remaining.append('Head scaling still includes dense H*S native state allocation/initialization and diagnostics. Empty-origin reads use a reachable-row bound; unknown imports retain full-grid reads. Grouped MMA computes cross-head scores and retains only the same-store diagonal because head stores have distinct keys; this does not remove off-diagonal arithmetic or H*S traffic. Four launches per token remain.')
    if cost['gates'].get('checkpoint_overhead_le_5pct') is not True:
        remaining.append('Capture still serializes ordered mutation groups with atomics/barriers, reserves a conservative fixed capacity, rebases addresses across that capacity, clones nonpayload metadata and copies private replay payload at allocated rows. Imported nonempty states retain dense replay copies. Final materialization captures live rows once per chunk.')
    ratios=cost['heads16_to_1_by_slots']
    gates=dict(cost['gates'])
    gates['heads_ratio_le_2']=all(ratios.get(str(s)) is not None and ratios[str(s)]<=2 for s in (736,11776))
    rows=[dict(slots=x['slots'],heads=x['heads'],wall_ms_per_token=x['wall_ms_per_token'])
          for x in cost['measurements'] if x['engine']=='x31']
    result=dict(full_parity=True,parity_tests=parity['tests'],
        forward_scope=cost['scope'],timing_gates=gates,
        x30_speedups=cost['x30_speedups'],heads_ratios=ratios,head_measurements=rows,
        checkpoint=dict(observed_fractional_time_delta=cost['checkpoint_overhead'],
            timing_gate=gates['checkpoint_overhead_le_5pct'],
            base_state_bytes=checkpoint.get('base_state_bytes'),
            reserved_bytes=checkpoint.get('journal_reserved_bytes'),
            recorded_entries=checkpoint.get('journal_recorded_entries'),
            private_forward_replay_copy_bytes=checkpoint.get('private_replay_copy_bytes'),
            private_replay_note='Address-space bytes, not bytes copied. Empty imports copy only allocated payload rows; other fields and nonempty imports clone densely.',
            storage_independence_gate=storage['storage_independence_gate'],
            budget_scaling=storage['measurements'],storage_scope=storage['note']),
        append_phase_gates={str(x['slots']):dict(speedup=x['speedup'],gate=x['gate']) for x in append['measurements']},
        collection_errors=cost['errors'],residual_risks=remaining+[
            'The steady recurrent path holds scalar metadata, but the existing cancellable heap is O(log S), and imported future birth groups can walk O(S). Fresh import initializes O(S) metadata once per chunk. A timer-wheel/bounded birth-group replacement is not delivered.',
            'Fixed log capacity bounds potential events and is not guaranteed within twice the actual largest event count. Unknown imported live bounds conservatively reserve against S; empty-origin continuations carry a growing host upper bound without inspecting device values.',
            'Base/replay address space and dense diagnostics remain O(S); no total-memory-independence or measured peak-memory claim.',
            'Learning was not rerun; part 5 remains historical learning evidence.'],
        learning_reference='summary_part5.json; not rerun in part 7',
        all_registered_targets_passed=all(x is True for x in gates.values())
            and storage['storage_independence_gate'] and all(x['gate'] for x in append['measurements']))
    (ROOT/'summary_part7.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
