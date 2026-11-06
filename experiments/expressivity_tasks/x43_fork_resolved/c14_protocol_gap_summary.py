"""Rebuild the protocol-shape ratio and conditional precision receipt, CPU only."""
import json
from pathlib import Path
from .c14_phase5r6_audit import audit

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / 'receipts/protocol_gap'


def summarize(context, unbounded_slots=257):
    arms = {name: json.loads((RECEIPTS / f'stress{slots}_t{context}_fixed.json').read_text())
            for name, slots in [('unbounded', unbounded_slots), ('budgeted', 48)]}
    traffic = {name: audit(tokens=context, peak=peak)['one_pass_payload_bytes']
               for name, peak in [('unbounded', 257), ('budgeted', 48)]}
    result = dict(context=context, batch=16, unbounded_physical_slots=unbounded_slots,
                  observed_peaks={name: arm.get('peak_occupancy') for name, arm in arms.items()},
                  status={name: arm['status'] for name, arm in arms.items()},
                  fp32_logical_payload_bytes_per_head_sequence=traffic,
                  bf16_logical_payload_bytes_per_head_batch={k: v*16/2 for k,v in traffic.items()},
                  traffic_ratio=traffic['unbounded']/traffic['budgeted'],
                  memory={name: {key: arm.get(key) for key in
                                ['max_memory_bytes', 'max_reserved_bytes', 'device_total_bytes']}
                          for name, arm in arms.items()})
    for name, arm in arms.items():
        memory = result['memory'][name]
        memory['allocator_margin_bytes'] = arm['device_total_bytes'] - arm['max_reserved_bytes']
        memory['idealized_all_allocated_bytes_halved'] = arm['max_memory_bytes']/2
        # Only the floating K/V/strength state, excluding integer metadata.
        memory['one_head_batch_fp32_store_state_bytes'] = 16*arm['slots']*(64+128+1)*4
        memory['one_head_batch_bf16_store_state_bytes'] = 16*arm['slots']*(64+128+1)*2
    if all(a['status'] == 'measured' for a in arms.values()):
        times = {name: arm['median_ms'] for name, arm in arms.items()}
        ratio = times['unbounded']/times['budgeted']
        result.update(full_step_ms=times, steps_per_second={k: 1000/v for k,v in times.items()},
                      full_step_unbounded_over_budgeted=ratio,
                      small_regime_reference_ratio=13/8.14,
                      small_regime_reference_scope='Operator-supplied approximate 2.13M comparison; not remeasured here.')
        # Explicit two-point conditional fit; not a precision benchmark.
        a = (times['unbounded']-times['budgeted'])/(traffic['unbounded']-traffic['budgeted'])
        c = times['budgeted'] - a*traffic['budgeted']
        result['conditional_bf16_fit'] = dict(common_ms=c, ms_per_logical_byte=a,
            physically_nonnegative=(a >= 0 and c >= 0))
        if a >= 0 and c >= 0:
            bf = {k: c+a*v/2 for k,v in traffic.items()}
            result['conditional_bf16_fit'].update(full_step_ms=bf,
                unbounded_over_budgeted=bf['unbounded']/bf['budgeted'])
    return result


def main():
    result = dict(shapes=[summarize(t) for t in (128, 256)],
        oversized_capacity_control=[summarize(t,2049) for t in (128,256)],
        precision_status='analytic only; no BF16 store implementation or benchmark',
        limitations=[
            'R5 one-stream-insertion empty-state fixture reaches 128/256, not 257, at these contexts.',
            'Logical payload is not measured HBM traffic. Two timing points exactly fit C+A*M but do not validate it; different replay launch costs and shape padding violate a pure traffic interpretation.',
            'Conditional BF16 fit assumes all modeled payload cost halves while common cost is unchanged; this is a scenario, not a certified full-step prediction.',
            'Idealized peak/2 assumes every allocated byte halves, including dense/optimizer/backward storage and metadata; it is an optimistic envelope, not expected store-only BF16 memory.',
            'For store-only BF16, total peak is P-F/2, with F the simultaneously resident FP32 store floating bytes. F is not measured; total device peak and OOM margin cannot be uniquely forecast.',
            'One-head state byte counts are capacity arithmetic, not aggregate graph-pool accounting. Reserved allocator memory does not scale linearly with dtype.',
        ])
    (RECEIPTS/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
