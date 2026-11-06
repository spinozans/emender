"""Recompute Phase-3 arithmetic from immutable training and controlled timing receipts."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT/'receipts/scalability_phase3'


def main():
    for name in ('train', 'curve'):
        assert json.loads((R/f'{name}_terminal.json').read_text())['exit_code'] == 0
    extent_terminal = json.loads((R/'extent_terminal.json').read_text())
    result_path = R/'sustained/result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        with gzip.open(str(result_path)+'.gz', 'rt') as f:
            result = json.load(f)
    assert result['status'] == 'completed_active_budget' and result['active_seconds'] >= 600
    warmup = last = None
    histogram = collections.Counter()
    head_histograms = collections.defaultdict(collections.Counter)
    peaks = collections.defaultdict(int)
    checkpoints = []
    occupancy_trace = []
    path = R/'sustained/steps.jsonl'
    opener = path.open if path.exists() else lambda: gzip.open(str(path)+'.gz', 'rt')
    with opener() as f:
        for line in f:
            row = json.loads(line)
            if last:
                assert row['step'] == last['step']+1 and row['targets'] > last['targets']
                assert row['active_seconds'] > last['active_seconds']
            if row['step'] == 20:
                warmup = row
            for layer, heads in enumerate(row['telemetry']):
                for h in heads:
                    key = f'{layer}/{h["head"]}'
                    peaks[key] = max(peaks[key], h['peak_occupancy'])
                    assert h['capacity_merges'] == h['refused_insertions'] == 0
                    for stream, array in h['occupancy']:
                        histogram[stream+array] += 1
                        head_histograms[key][stream+array] += 1
            if row['step'] % 25 == 0:
                occupancy_trace.append(dict(active_seconds=row['active_seconds'], heads={f'{l}/{h}': max(x['peak_occupancy'] for x in heads if x['head'] == h) for l,heads in enumerate(row['telemetry']) for h in sorted({x['head'] for x in heads})}))
            if warmup and (not checkpoints or row['active_seconds']-checkpoints[-1]['active_seconds'] >= 100):
                checkpoints.append({k: row[k] for k in ('step', 'targets', 'active_seconds')})
            last = row
    assert last['step'] == result['steps'] and last['targets'] == result['targets']
    seconds = last['active_seconds']-warmup['active_seconds']
    steps = last['step']-warmup['step']
    targets = last['targets']-warmup['targets']
    rate = steps/seconds
    target_rate = targets/seconds
    def mean(h):
        return sum(k*v for k,v in h.items())/sum(h.values())
    curve = json.loads((R/'fixed_curve/result.json').read_text())
    rows = [r for r in curve['rows'] if r['mode'] == 'bounded']
    assert [r['peak_live'] for r in rows] == [48,110,128,257]
    assert all(r['tokens'] == 257 and r['reference_exact'] and r['repeat_exact'] for r in rows)
    x = np.array([r['peak_live'] for r in rows], dtype=float)
    y = np.array([r['median_ms'] for r in rows])
    slope, intercept = np.polyfit(x,y,1)
    predicted = slope*x+intercept
    r2 = 1-float(np.sum((y-predicted)**2)/np.sum((y-y.mean())**2))
    curve_out = dict(points=[dict(live=r['peak_live'],median_ms=r['median_ms'],min_ms=min(r['native_ms']),max_ms=max(r['native_ms']),spills=r['n_spills']) for r in rows],
                     slope_us_per_live_slot_per_B16_T257_head=slope*1000,
                     intercept_ms=intercept,r_squared=r2,
                     marginal_ns_per_byte_per_live_slot=slope*1e6/(16*257),
                     scope='affine descriptive fit over four controlled append-only schedules at fixed physical BS512; not a measured long-context training law')
    # Parameter-proportional throughput is an explicit scenario, never an estimate
    # certified by a small-model measurement. Dense utilization is unmeasured.
    proportional = []
    for p in (150_000_000,1_300_000_000):
        factor = p/result['trainable_parameters']
        q = target_rate/factor
        proportional.append(dict(parameters=p,parameter_factor=factor,assumed_targets_per_second=q,
                                 targets_per_2700_active_seconds=q*2700,
                                 seconds_for_2B_targets=2e9/q,days_for_2B_targets=2e9/q/86400))
    # Forward+checkpoint recomputation only. Replay, dense work, data, telemetry,
    # optimizer and receipts are absent. B1 may take nearly B16 invocation time.
    store_scenarios = []
    for layers, context, width in ((12,512,768),(22,1024,2048)):
        tile = 2*context
        for heads in (1,2,8):
            for batch_scaling in ('ideal_per_stream','same_invocation_latency_B1'):
                batch_factor = 16 if batch_scaling == 'ideal_per_stream' else 1
                sec_per_byte = y[-1]/1000/257/batch_factor * layers*heads*2*(tile/512)
                store_scenarios.append(dict(layers=layers,backbone_width=width,context=context,heads=heads,
                    assumed_BS=tile,BS_multiplier=tile/512,batch_scaling=batch_scaling,
                    assumed_seconds_per_byte_native_forward_and_recompute_only=sec_per_byte,
                    hypothetical_bytes_in_2700_native_seconds=2700/sec_per_byte,
                    hypothetical_native_days_for_2B=2e9*sec_per_byte/86400))
    extent = json.loads((R/'extent_probe.json').read_text())
    extent_scenarios = []
    for r in extent['rows']:
        if r.get('status') == 'failed' or r['tokens'] not in (512,1024):
            continue
        layers = 12 if r['tokens'] == 512 else 22
        for heads in (1,2,8):
            sec = r['median_ms']/1000/r['tokens']*layers*heads*2
            extent_scenarios.append(dict(layers=layers,heads=heads,context=r['tokens'],
                controlled_live=257,native_forward_plus_recompute_seconds_per_byte=sec,
                hypothetical_bytes_per_2700_native_seconds=2700/sec,
                hypothetical_native_days_for_2B=2e9*sec/86400,
                interpretation='direct measured B1 head time multiplied by layers/heads/two passes; controlled zero-value schedule, excludes replay/dense/data/telemetry/optimizer, not a full-model rate or certified bound'))
    scale_projection = None
    proxy = next((r for r in extent['rows'] if r['tokens'] == 512 and 'median_ms' in r), None)
    if proxy:
        assumed_head_ms = proxy['median_ms']*4
        per_byte = assumed_head_ms/1000/1024*22*2*2
        scale_projection = dict(status='conditional_extrapolation_not_qualified',layers=22,heads=2,context=1024,
            basis='measured B1/T512/BS1024 at peak257; double T and double BS => assumed4x invocation time, unchanged cell64/128 widths',
            assumed_native_head_ms=assumed_head_ms,native_forward_recompute_seconds_per_byte=per_byte,
            hypothetical_native_days_for_2B=per_byte*2e9/86400,
            hypothetical_native_byte_rate=1/per_byte,
            required_byte_rate_for_2B_in_86400=2e9/86400,
            qualification='Excludes dense/replay/data/telemetry/optimizer; natural long-context schedule and BS2048 runtime unmeasured; not a certified bound')
    out = dict(status='measured_C14_only_scale_tractability_unestablished',
               sustained=dict(steps=result['steps'],targets=result['targets'],active_seconds=result['active_seconds'],
                              wall_seconds=result['wall_seconds'],excluded_receipt_checkpoint_seconds=result['excluded_receipt_checkpoint_seconds'],
                              active_plus_receipt_checkpoint_seconds=result['active_seconds']+result['excluded_receipt_checkpoint_seconds'],
                              postwarm_steps=steps,postwarm_targets=targets,postwarm_seconds=seconds,
                              postwarm_steps_per_second=rate,postwarm_targets_per_second=target_rate,
                              hypothetical_C14_steps_per_2700_active=rate*2700,
                              hypothetical_C14_targets_per_2700_active=target_rate*2700,
                              hypothetical_C14_wall_seconds_per_2700_active=2700*(1+result['excluded_receipt_checkpoint_seconds']/result['active_seconds'])+(result['wall_seconds']-result['active_seconds']-result['excluded_receipt_checkpoint_seconds']),
                              postwarm_seconds_per_target=1/target_rate,mean_live=mean(histogram),head_mean_live={k:mean(h) for k,h in head_histograms.items()},
                              head_peaks=dict(peaks),live_histogram=dict(sorted(histogram.items())),interval_endpoints=checkpoints),
               conditional_1p3B_projection=scale_projection,
               optional_extent_abort=json.loads((R/'extent_abort.json').read_text()) if (R/'extent_abort.json').exists() else None,
               extent_terminal=extent_terminal,extent_probe=extent,measured_extent_native_only_scenarios=extent_scenarios,
               cost_curve=curve_out,parameter_proportional_scenarios=proportional,native_only_scenarios=store_scenarios,
               assumptions=['Same RTX6000Ada; fixed cell dimensions256/key64/value128 at all backbone widths.',
                            'Native scenarios assume cost proportional to BS beyond measured512; this is unvalidated and residual spills may worsen it.',
                            'Native time assumed proportional to tokens and layers/heads; either ideal B16-to-B1 throughput or equal invocation latency; neither is a hardware bound.',
                            'Parameter-proportional scenario assumes total cost proportional to trainable parameters, unchanged bytes/update and dense utilization; not a scale forecast.',
                            'Natural occupancy at scale is unknown; theoretical maximum2T, cap2T+1; physical tile nextpow2(min(cap,2T)) regardless of measured live count.',
                            'Historical final_pure config supplies 2B targets/86400 seconds budget class, not a registered unbounded launch. Required throughput23148.148 targets/s.',
                            'No measurement bridges raw-byte C14 to GPT-2 tokenized historical results; no cross-unit throughput comparison.'],
               logical_value_stage_bytes_per_target_N2_L22_T1024=1024*2048*2*22*2,
               memory_notes=dict(raw_native_state_bytes_per_B1_head_T1024=800*2049,
                                 FP32_parameters_gradients_two_Adam_moments_at_1p3B_bytes=16*1_300_000_000,
                                 scope='logical staging traffic is not measured HBM traffic; state allocation excludes replay/graphs/activations; optimizer estimate excludes autocast buffers'),
               next_target='Long-context physical BS1024/2048 spilling and single-stream CTA underutilization, followed by full width768/2048 measured training; no search launched')
    (R/'projection.json').write_text(json.dumps(out,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x,y,'o-',label='row-key kernel, measured medians')
    ax.plot(x,predicted,'--',label=f'affine fit: {intercept:.2f} ms + {slope*1000:.2f} us/live')
    ax.set(xlabel='Peak live rows (controlled append schedule)',ylabel='Native head invocation ms',title='B16 / T257 / cap257 / BS512, 20 samples')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R/'occupancy_cost.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for key in peaks:
        ax.plot([x['active_seconds'] for x in occupancy_trace], [x['heads'][key] for x in occupancy_trace], label='layer/head '+key, linewidth=.8)
    ax.set(xlabel='Active training seconds',ylabel='Batch peak live slots',title='Natural training occupancy, sampled every25 steps')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R/'occupancy_trace.svg')
    for path in R.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__':
    main()
