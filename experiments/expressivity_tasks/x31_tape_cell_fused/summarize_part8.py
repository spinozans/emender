"""Summarize unchanged registered targets from part 8 receipts."""
import json
from pathlib import Path
from .train_gates import require_parity

ROOT=Path(__file__).resolve().parent


def main():
    def read(name):return json.loads((ROOT/name).read_text())
    require_parity(ROOT/'parity_part8.json')
    parity=read('parity_part8.json')
    cost=read('cost_part8.json')
    append=read('append_cost_part8.json')
    storage=read('storage_part8.json')
    checkpoint=next((x for x in cost['measurements'] if x['engine']=='x31_checkpoint'),{})
    remaining=[]
    if cost['gates'].get('heads_ratio_le_2') is not True:
        remaining.append('Dense public native payload initialization and death diagnostics remain H*S. Same-store score reduction now skips cross-head outputs; empty-origin metadata imports initialize only a reachable prefix. See structural_cost_part8.json for measured native-initialization cost; this is a component cost, not proof that every alternative representation must fail.')
    if cost['gates'].get('checkpoint_overhead_le_5pct') is not True:
        remaining.append('Mutation capture still emits before-image records and updates a single-writer device count. Atomics and journal-only barriers are removed; replay copies and address rebasing are deferred to backward. Reverse groups preserve repeated-address ordering. The matched capture ablation is in structural_cost_part8.json; no irreducible hardware floor is claimed.')
    ratios=cost['heads16_to_1_by_slots']
    gates=dict(cost['gates'])
    gates['heads_ratio_le_2']=all(ratios.get(str(s)) is not None and ratios[str(s)]<=2 for s in (736,11776))
    structural=read('structural_cost_part8.json')
    native16=next(x['wall_ms_per_token'] for x in structural['measurements'] if x.get('component')=='native_zero_initialization' and x['heads']==16)
    h1=next(x['wall_ms_per_token'] for x in cost['measurements'] if x['engine']=='x31' and x['slots']==11776 and x['heads']==1)
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
            private_forward_replay_copy_bytes=0,
            retained_replay_address_space_bytes=checkpoint.get('private_replay_copy_bytes'),
            private_replay_note='Forward retains owners without copying or rebasing. A consuming continuation freezes replay owners before mutation; backward clones and rebases on consumption.',
            storage_independence_gate=storage['storage_independence_gate'],
            budget_scaling=storage['measurements'],storage_scope=storage['note']),
        append_phase_gates={str(x['slots']):dict(speedup=x['speedup'],gate=x['gate']) for x in append['measurements']},
        heads_ratio_le_2_by_slots={str(s):ratios[str(s)]<=2 for s in (736,11776)},
        head_scaling_floor=dict(native_h16_ms_per_token=native16,
            allowed_total_h16_ms_per_token=2*h1,
            native_component_to_h1_ratio=native16/h1,
            scope='Measured current dense initialization component, not a universal hardware lower bound.'),
        structural_measurements=dict(receipt='structural_cost_part8.json',
            measurements=[{k:x[k] for k in ('component','heads','wall_ms_per_token','cuda_ms_per_token','capture') if k in x} for x in structural['measurements']],
            capture_fractional_delta=structural['capture_fractional_delta'],note=structural['note']),
        collection_errors=cost['errors'],residual_risks=remaining+[
            'The heap is O(log live), bounded by cumulative appends for empty-origin streams, but arbitrary imports can reach O(log S). Imported future/tied birth walks and birth-index construction retain O(S) work; no timer wheel or bounded birth-group replacement is delivered.',
            'Fixed log capacity bounds potential events and is not guaranteed within twice the actual largest event count. Unknown imported live bounds conservatively reserve against S; empty-origin continuations carry a growing host upper bound without inspecting device values.',
            'Base/replay address space and dense diagnostics remain O(S); no total-memory-independence or measured peak-memory claim.',
            'Public in-place edits to retained replay owners raise on backward. Consuming continuations freeze them; this adds a deferred copy to that continuation. Backward still clones and scans journal capacity.',
            'Final checkpoint timing passes narrowly; the superseded pre-private-trace collection measured +5.036%. Timing variability matters at this threshold.',
            'Final materialization now serializes slot tiles within each store to preserve the single-writer invariant; arbitrary dense imported-state performance was not measured.',
            'Learning was not rerun; part 5 remains historical learning evidence.'],
        learning_reference='summary_part5.json; not rerun in part 8',
        all_registered_targets_passed=all(x is True for x in gates.values())
            and storage['storage_independence_gate'] and all(x['gate'] for x in append['measurements']))
    (ROOT/'summary_part8.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
