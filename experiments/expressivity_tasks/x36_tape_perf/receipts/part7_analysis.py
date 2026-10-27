"""Reproducible part-7 diagnostic arithmetic; no floor waivers."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent

def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report=dict(status='FAILED: requested performance closure not achieved',
                implementation='Device journal decisions suppress dead packed value-adjoint traffic; named dense scatter is absent from capacity workload',
                scatter=dict(native_gradient_requests=[False,False,False],dense_unpack_calls_per_chunk=0,
                             dense_unpack_calls_per_token=0,maximum_removable_scatter_ms=0,
                             event_path='reverse_program: Cases/Matched/Alloc -> packed GValues/GKeys -> DVout/DKout',
                             native_path='native_grad: needs_input_grad guard before unpack; unpack, if requested, runs once per chunk',
                             inactive_rows='No native gradient tensor is requested in capacity measurements. For native-gradient consumers the existing unpack preserves output adjoints on untouched key/value rows and applies +0 to cold strength rows. Never zero arbitrary nonzero output adjoints.'),
                sources={name:sha(PROJECT/name) for name in ('adjoint_pool.py','backward.py','replay_setup.py','forward.py','kernels.py')},
                fixed_live_work=[],capacity=[],long_context={},fresh_gates=[],residual_risks=[])
    n,t,dk,dv=256,512,64,128
    for s in (2944,5888,11776,8192,8224):
        bs=1<<(s-1).bit_length()
        report['fixed_live_work'].append(dict(slots=s,stores=n,tokens=t,
            forward_tile_probes=n*t*((s+31)//32),
            forward_tile_formula='N*T*ceil(S/32), saturated logical reach, even when only 17 rows are live',
            forward_nonempty_tiles={str(live):min(live,(s+31)//32) for live in (17,256,1024,2800)},
            forward_padded_score_value_flops={str(live):n*t*min(live,(s+31)//32)*32*2*(dk+dv) for live in (17,256,1024,2800)},
            tile_count_basis='Uniform linspace entries in this protocol: when the row gap is below 32 every tile is populated; above 32 each row occupies a distinct tile. Source evaluates 32 float lanes per populated tile.',
            logical_reduction_width=bs,
            semantic_full_reductions_per_store_token=dk+4,
            reduction_operand_lanes=n*t*(dk+4)*bs,
            reduction_note='DPop, maximum, denominator, dp sum and DK query-gradient sums. Counts are semantic operand lanes, not executed ISA counts or measured bandwidth.',
            public_freed_bytes=n*t*s,
            public_freed_note='int8 zeros backing; public transpose may require another equally sized allocation/copy; not a gradient scatter',
            pool_float_bytes_live17=n*17*(dv+1)*4,
            pool_float_bytes_live256=n*256*(dv+1)*4,
            suppressed_value_adjoint_logical_bytes={str(live):n*t*live*dv*8 for live in (17,256,1024,2800)}))
    for live in (17,256,1024,2800):
        name=f'part7_live{live}.json'
        data=read(name)
        for r in data['results']:
            assert r['native_gradient_requests']==[False]*3 and r['dense_native_unpack_calls']==0
            assert r['pool']['allocated_rows']==live
        by={(r['mode'],r['slots']):r for r in data['results']}
        old={(r['mode'],r['slots']):r for r in read(f'part6_hot_live{live}.json')['results']}
        for gate in data['gates']:
            a,b=[by[gate['mode'],s] for s in gate['slots']]
            meda,medb=a['median_ms'],b['median_ms']
            forward_delta=medb['forward_ms']-meda['forward_ms']
            replay_delta=medb['replay_ms']-meda['replay_ms']
            report['capacity'].append(dict(live=live,mode=gate['mode'],slots=gate['slots'],
                source=name,status=gate['status'],ratios=gate['ratios'],
                medians=[meda,medb],
                versus_part6_ratios=[{k:r['median_ms'][k]/old[r['mode'],r['slots']]['median_ms'][k] for k in ('step_ms','replay_ms')} for r in (a,b)],
                required_savings_ms={k:max(0,medb[k]-1.5*meda[k]) for k in ('step_ms','replay_ms')},
                device_delta_ms=dict(forward=forward_delta,replay=replay_delta),
                attribution='Native unpack accounts for exactly zero. Forward probes/metadata and full-logical reverse reductions remain capacity dependent. Timings of their subexpressions are not causally isolated; no irreducible-floor waiver.'))
    for tokens in (512,4096):
        records=read(f'part7_fresh_t{tokens}.json')
        base=next(r for r in records if r['arm']=='base_lm')
        tape=next(r for r in records if r['arm']=='tape_lm')
        assert base['status']==tape['status']=='completed'
        ratio=tape['median_ms']/base['median_ms']
        report['fresh_gates'].append(dict(tokens=tokens,ratio=ratio,limit=3 if tokens==512 else 2,
            status='passed' if ratio<=(3 if tokens==512 else 2) else 'failed',
            wall_ms=dict(base=base['median_ms'],tape=tape['median_ms'])))
        if tokens==4096:
            short=next(r for r in read('part7_fresh_t512.json') if r['arm']=='tape_lm')
            pairs=[r['live_traffic']['total_live_slot_tokens'] for r in (short,tape)]
            profiles=[r['profile'] for r in (short,tape)]
            kernel_components=[{k:v['component_device_busy_ms'] for k,v in p.items() if k.startswith('x36::')} for p in profiles]
            kernel_busy=[sum(sum(v.values()) for v in p.values()) for p in kernel_components]
            scan=[p['x36::scan_forward'].get('scan_kernel',0) for p in kernel_components]
            reverse=[p['x36::replay_backward'].get('reverse_kernel',0) for p in kernel_components]
            report['long_context']=dict(tokens=[512,4096],token_ratio=8,occupied_pairs=pairs,
                occupied_pair_ratio=pairs[1]/pairs[0],mean_live=[pairs[0]/(n*512),pairs[1]/(n*4096)],
                payload_read_flops=[p*2*(dk+dv) for p in pairs],
                read_flop_formula='2*(DK+DV)*occupied_pairs: score dot + weighted value sum only; excludes exp, decay, backward, dense padding and scalar recurrence',
                flops_per_input_token=[pairs[0]*2*(dk+dv)/(64*512),pairs[1]*2*(dk+dv)/(64*4096)],
                per_token_note='Live occupancy grows 8x. Flat useful cost is per live-slot/token pair, not per token for this workload. Kernels use vector reductions; this is not a GEMM hardware-floor claim.',
                required_wall_savings_ms=max(0,tape['median_ms']-2*base['median_ms']),
                native_unpack_savings_ms=0,profile=tape.get('profile',{}),
                scan_kernel_busy_ms=scan,reverse_kernel_busy_ms=reverse,
                scan_ns_per_occupied_pair=[ms*1e6/p for ms,p in zip(scan,pairs)],
                reverse_ns_per_occupied_pair=[ms*1e6/p for ms,p in zip(reverse,pairs)],
                non_scan_reverse_device_busy_ms=[busy-f-r for busy,f,r in zip(kernel_busy,scan,reverse)],
                timing_scope='Separate profiles: GPU busy times are not additive wall samples. No causal wall savings attributed to launch count or telemetry.',
                status='No calibrated compute/launch lower bound; residual remains failed if wall gate fails')
    profile=read('part7_profile.json')['results'][-1]['profile']
    report['continued_profile']=profile
    report['peak_fixed_live_allocated_bytes']=max(r['median_ms']['peak_allocated_bytes'] for live in (17,256,1024,2800) for r in read(f'part7_live{live}.json')['results'])
    report['exact_host_sizing_constraint']=dict(transfers=2,
        first='Forward host journal reservation and cold-tail specialization depend on device entry metadata',
        second='Backward host torch.empty shapes depend on device union maximum after recorded allocations',
        arithmetic='two serial host shape dependencies in the present eager exact-size implementation; 2 > required 0',
        scope='Conditional implementation constraint, not a hardware lower bound. Removing transfers requires changing the exact-allocation/dispatch strategy; no fix implemented.')
    report['residual_risks']=[
        'Targeted dense cold-gradient pass does not execute in this workload; the implemented event-consumer mask removes different, packed adjoint traffic.',
        'Sync removal and launch fusion not implemented.',
        'Remaining capacity costs identified by source/operation counts and forward/replay times, not isolated into causal subkernel timings.',
        'No calibrated GEMM/launch/telemetry floor. Failed gates receive no floor waiver.',
        'Fresh full capacity sweep and historical loss curves carried from part 6; fresh specializations do not execute the pooled event-consumer optimization.',
        'contact_supervisor tool unavailable; the false named premise could not be escalated through Intercom.'
    ]
    (ROOT/'part7_arithmetic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','fresh_gates','residual_risks')},indent=2))

if __name__=='__main__':main()
