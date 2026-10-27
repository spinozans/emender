"""Receipt gate shortfalls and decomposition; no lower-bound waiver."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent
load=lambda name:json.loads((R/name).read_text())
s=load('summary.json')
fresh=load('part6_final_fresh.json')
by={(r['arm'],r['tokens'],r['slots']):r for r in fresh}
long,base=by['tape_lm',4096,2944],by['base_lm',4096,2944]
continued=[]
for label in ('live17','live256','live1024','live2800','near_budget'):
    receipt=load(f'part6_hot_{label}.json')
    by_mode={(r['mode'],r['slots']):r for r in receipt['results']}
    for g in receipt['gates']:
        a,b=(by_mode[g['mode'],slots] for slots in g['slots'])
        if g['status']=='failed':
            continued.append(dict(label=label,mode=g['mode'],slots=g['slots'],occupancy_matched=g['occupancy_matched'],
                medians=[r['median_ms'] for r in (a,b)],
                compiler=[r['compiler'] for r in (a,b)],
                ratios=g['ratios'],
                required_savings_ms={metric:max(0,b['median_ms'][metric]-1.5*a['median_ms'][metric])
                                     for metric in ('step_ms','replay_ms')},
                row_pitch=[r['pool']['allocated_rows'] for r in (a,b)],
                float_lane_width=[1<<(max(1,r['pool']['allocated_rows'])-1).bit_length() for r in (a,b)],
                logical_reduction_lane_width=[1<<(r['slots']-1).bit_length() for r in (a,b)],
                interpretation='Current full-logical reduction, native forward and layout cost; not an irreducible floor'))
traffic={str(t):{k:v for k,v in by['tape_lm',t,2944]['live_traffic'].items()
                         if k in ('scope','total_live_slot_tokens','forward_payload_logical_bytes',
                                  'replay_payload_logical_bytes','allocations','frees','key_adjoint_tracked')}
         for t in (512,4096)}
result=dict(structural_floor_established=False,gate_waivers=[],
    long=dict(tape_wall_ms=long['median_ms'],base_wall_ms=base['median_ms'],
              ratio=long['median_ms']/base['median_ms'],limit=2,
              required_wall_savings_ms=long['median_ms']-2*base['median_ms'],
              event_components=long['component_median_ms'],compiler=long['reverse_compiler_stats']),
    logical_traffic=traffic,
    traffic_interpretation='Logical operations only; no DRAM counters, hardware bandwidth limit or GEMM lower-bound proof',
    continued_failed_pairs=continued,
    profile_decomposition=s['decomposition'],
    inclusive_shortfall=[dict(label=d['label'],inclusive=d['inclusive'],limit=10,
                              excess_activities=max(0,d['inclusive']-10),custom=d['custom'])
                         for d in s['decomposition']],
    memory_bound='Per-store |entry union allocations| <= min(S, entry_live + allocation_events) <= min(S, entry_live + T); exact row pitch max union, minimum one sentinel row',
    lifetime=dict(original_growth_bytes_per_chunk=40844800,journal_bulk_bytes=30561280,
                  remaining_owner_trace_context_bytes=40844800-30561280,
                  candidate_standalone_allocated_series=[9319424]*7))
(R/'part6_failure_arithmetic.json').write_text(json.dumps(result,indent=2)+'\n')
