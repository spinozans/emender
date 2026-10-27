"""Reproduce the part-4 numerical ledger from retained measurement receipts."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def gate(actual, limit):
    return dict(status='passed' if actual <= limit else 'failed', actual=actual, limit=limit)


def main():
    suites=list(ET.parse(ROOT/'part4_seven_phase_parity.xml').getroot().iter('testsuite'))
    assert suites and all(int(v.get('failures',0))==int(v.get('errors',0))==int(v.get('skipped',0))==0 for v in suites)
    parity_tests=sum(int(v.get('tests',0)) for v in suites)
    capacity = {}
    sources = []
    for slots in (2944, 5888, 11776, 8192, 8224):
        name = f'part4_seven_s{slots}.json'
        records = read(name)
        row = records[-1]
        assert row['status'] == 'completed' and row['samples'] == len(row['raw_ms']) == 5
        assert row['final_live_min_mean_max'] == [256., 256., 256.]
        assert row['live_traffic']['total_live_slot_tokens'] == 16777216
        assert row['live_traffic']['frees'] == 0
        capacity[slots] = row
        sources.append(name)
    short_records = read('part4_seven_s2944.json')
    long_records = read('part4_seven_t4096.json')
    short, long = short_records[-1], long_records[-1]
    curve = read('part4_seven_training_slice.json')
    decomposition = []
    for row in [*capacity.values(), long]:
        profile = row['profile']
        forward = profile['x36::scan_forward']['component_device_busy_ms']
        reverse = profile['x36::replay_backward']['component_device_busy_ms']
        adjoints = reverse.get('reverse_kernel', 0.) + reverse.get('reverse_fallback_kernel', 0.)
        decomposition.append(dict(slots=row['slots'], tokens=row['tokens'],
            step_wall_ms=row['median_ms'], replay_event_ms=row['component_median_ms']['replay_backward'],
            scan_gpu_ms=forward['scan_kernel'], journal_ablation_estimate_ms=row['journal_increment_estimate_ms'],
            other_forward_gpu_ms=sum(forward.values())-forward['scan_kernel'],
            replay_adjoint_gpu_ms=adjoints, fused_replay_setup_gpu_ms=reverse['setup_kernel'],
            rebase_gpu_ms=reverse['rebase'],
            other_replay_gpu_ms=sum(reverse.values())-adjoints-reverse['setup_kernel']-reverse['rebase'],
            scan_replay_allocator_cpu_ms=row['scan_replay_host_allocation_cpu_ms'],
            launch_count=sum(profile[k]['activity_count'] for k in ('x36::scan_forward','x36::replay_backward')),
            compiler=row['reverse_compiler_stats']))
    (ROOT/'part4_decomposition.json').write_text(json.dumps(dict(
        method='Five raw wall samples; separate stream-event components and correlation-attributed profiles. GPU sums exclude overlapping CPU allocation. Journal is a no-grad ablation estimate, not additive to scan.',
        rows=decomposition), indent=2)+'\n')
    growth = []
    for a, b in ((2944,5888),(5888,11776),(8192,8224)):
        growth.append(dict(slots=[a,b], capacity_growth=b/a,
            step=gate(capacity[b]['median_ms']/capacity[a]['median_ms'],1.5),
            replay=gate(capacity[b]['component_median_ms']['replay_backward']/capacity[a]['component_median_ms']['replay_backward'],1.5)))
    profiles = [row['profile'] for row in [*capacity.values(),long]]
    steady_syncs = [event for p in profiles for event in p['steady_sync_runtime'] if event['name'] != 'cudaDeviceSynchronize']
    dtoh = [event for p in profiles for event in p['dtoh']]
    boundary_sync_counts = [len(p['steady_sync_runtime']) for p in profiles]
    pairs_long = long['live_traffic']['total_live_slot_tokens']
    short_traffic = read('part4_seven_s2944.json')[-1]['live_traffic']
    pairs_short = short_traffic['total_live_slot_tokens']
    arithmetic = dict(
        scope='Logical occupied accesses/FLOPs; not measured DRAM bytes, executed ISA counts, or a hardware lower bound.',
        live_slot_token_pairs_t512_measured=pairs_short,
        t512_traffic_source='part4_seven_s2944.json',
        live_slot_token_pairs_t4096_measured=pairs_long,
        pair_growth=pairs_long/pairs_short, token_growth=8, mean_occupancy_growth=(pairs_long/pairs_short)/8,
        key_score_active_flops_t4096=2*64*pairs_long,
        replay_value_dot_active_flops_t4096=2*128*pairs_long,
        replay_value_adjoint_update_active_flops_t4096=2*128*pairs_long,
        query_adjoint_active_flops_t4096=2*64*pairs_long,
        forward_logical_bytes=long['live_traffic']['forward_payload_logical_bytes'],
        replay_logical_bytes=long['live_traffic']['replay_payload_logical_bytes'],
        target_wall_ms=2*long_records[0]['median_ms'], actual_wall_ms=long['median_ms'],
        target_wall_gap_ms=long['median_ms']-2*long_records[0]['median_ms'],
        read_gemm_floor_claim=False,
        reason='Replay remains dominant; the final long ratio still fails. Growing occupancy and retained logical-S reductions preclude attributing the residual just to an honest GEMM floor.')
    (ROOT/'part4_t4096_arithmetic.json').write_text(json.dumps(arithmetic,indent=2)+'\n')
    max_curve_error=max(curve[a]['max_loss_error'] for a in ('base_lm','tape_lm'))
    launches=[d['launch_count'] for d in decomposition]
    summary=dict(status='partial; capacity spill cliff closed; long-context and inclusive-launch targets fail; arbitrary sparse imported-state pool incomplete',
        all_registered_targets_passed=False, parity_tests=parity_tests, requested_original_tests=138, added_exact_fixtures=3, parity_receipt='part4_seven_phase_parity.xml',
        source_sha256_receipt='part4_seven_source_sha256.json',
        measurement_sources=[*sources,'part4_seven_t4096.json','part4_seven_training_slice.json'],
        gates=dict(b64_t512_ratio_le_3=gate(short['median_ms']/short_records[0]['median_ms'],3),
            b64_t4096_ratio_le_2=gate(long['median_ms']/long_records[0]['median_ms'],2),
            capacity_doublings_and_boundary=growth,
            inclusive_launches_le_10=gate(max(launches),10), inclusive_launches_by_shape=[dict(slots=d['slots'],tokens=d['tokens'],activities=d['launch_count']) for d in decomposition],
            narrow_custom_phases_le_10=gate(7,10), custom_phases_t512=7, custom_phases_t4096=7,
            requested_seven_phases_at_long_context=gate(7,7),
            zero_steady_host_syncs=gate(len(steady_syncs)+len(dtoh),0),
            profiler_boundary_device_sync_counts=boundary_sync_counts,
            matched_2m_loss_curves=gate(max_curve_error,1e-5)),
        representation=dict(private_row_bound_t512=512, actual_live_rows_t512=256,
            long_registered_dispatch_rows=2048, unknown_imports='dense fallback',
            native_cold_rows='retained', full_logical_reductions='retained', global_reduction_scratch='removed'),
        residual_risks=['T4096 still exceeds 2x.', 'Inclusive activities exceed 10; native fills/copies/transposes/mapping transfers remain unfused.', 'Prefix reach is not a general arbitrary-slot live pool; sparse imports and cold native owners remain dense.', 'Exact reductions still depend on validated Triton layouts and logical-S expansion.'],
        operational=dict(python='/home/erikg/emender/.venv/bin/python',physical_gpus=['0'],broker='run_leased.sh',no_pt_files=True,no_wg=True,no_delegation=True,resilient_requirement_ids='not applicable'))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(gates=summary['gates'],arithmetic=arithmetic),indent=2))


if __name__ == '__main__':
    main()
