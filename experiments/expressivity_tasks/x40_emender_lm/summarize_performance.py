"""Regenerate part-3 tables from raw receipts; failed gates stay failed."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from .performance import write

ROOT=Path(__file__).resolve().parent
R=ROOT/'receipts'


def read(path):return json.loads(path.read_text())


def ledger(folder):
    events=json.loads(gzip.decompress((R/folder/'trace.json.gz').read_bytes()))['traceEvents']
    runtime=Counter(e['name'] for e in events if e.get('cat') in ('cuda_runtime','cuda_driver'))
    device=Counter(e['cat'] for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset'))
    return dict(runtime_launches={k:v for k,v in runtime.items() if 'Launch' in k},
        total_host_launches=sum(v for k,v in runtime.items() if 'Launch' in k),device_activities=dict(device),
        total_device_activities=sum(device.values()))


def main():
    base=read(R/'part3_baseline/wall.json');exact=read(R/'part3_exact_decay/wall.json')
    graph=read(R/'part3_graph_final/wall.json');training=read(R/'part3_slice/summary.json')
    scale=read(R/'part3_scale_final/parity_T512.json');bf=read(R/'part3_bf16/gate.json')
    curve=ROOT/'receipts/ignition_v1/curve.jsonl';v1=[json.loads(x) for x in curve.read_text().splitlines()]
    quality=read(R/'ignition_v1/quality_3052.json')['validation']
    controls=[]
    for shape in ('pilot','final'):
        for t in (512,2048):
            raw=read(R/'part3_transformer_scale'/f'{shape}_transformer_T{t}.json')
            controls.append({k:raw[k] for k in ('parameters','tokens_per_second','peak_allocated','peak_reserved','measured_updates','config')})
    amortization={shape:controls[i+1]['tokens_per_second']/controls[i]['tokens_per_second'] for shape,i in (('150M',0),('1.3B',2))}
    launches={name:ledger(folder) for name,folder in (('baseline','part3_baseline'),('exact_decay','part3_exact_decay'),('graphs','part3_graph_final'))}
    gates=[dict(name='B64/T512 update <=2.5s',status='passed' if graph['median_seconds']<=2.5 else 'failed',value=graph['median_seconds']),
        dict(name='5M matched training loss <=1e-5',status=training['status'],value=training['max_loss_difference']),
        dict(name='scale policy parity before throughput',status=scale['status'],error=scale.get('error')),
        dict(name='pure scale throughput/control >=0.5',status='not-run',reason='scale parity failed'),
        dict(name='pure T2048/T512 tokens/s >=2',status='not-run',reason='scale parity failed'),
        dict(name='bf16 bitwise store parity',status=bf['status'],enabled=bf['enabled'])]
    summary=dict(status='partial: scale parity failed; no scale-studyability closure',baseline=base,exact_decay=exact,
        graphs=graph,training_slice=training,launches=launches,gates=gates,transformer_controls=controls,transformer_T2048_over_T512=amortization,
        ignition_v1=dict(steps=len(v1),last_step=v1[-1]['step'],tokens=v1[-1]['tokens'],held_out_ppl=quality['perplexity'],
            curve_sha256=hashlib.sha256(curve.read_bytes()).hexdigest(),quality_sha256=hashlib.sha256((R/'ignition_v1/quality_3052.json').read_bytes()).hexdigest()))
    write(R/'part3_summary.json',summary)
    manifest={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ('x31_tape_cell_fused','x34_lm_pilot','x36_tape_perf','x37_head_differentiation','x38_diversity_objective','x39_memory_usage','x40_emender_lm','x42_pile_scale')
        for p in (ROOT.parent/directory).glob('*.py')}
    write(R/'part3_sources.json',manifest)
    table='\n'.join(f"| {'150M' if c['config']['width']==768 else '1.3B'} | {c['config']['context']} | {c['parameters']:,} | {c['tokens_per_second']:.1f} | {c['peak_allocated']/2**30:.3f} / {c['peak_reserved']/2**30:.3f} |" for c in controls)
    launch_table='\n'.join(f"| {name} | {item['total_host_launches']} | {item['runtime_launches'].get('cudaGraphLaunch',0)} | {item['total_device_activities']} |" for name,item in launches.items())
    report=f'''# X40 part 3 — 8.03s to {graph['median_seconds']:.3f}s, scale closure blocked

B64/T512 ignition-v1 performance is repaired: unchanged v1 **{base['median_seconds']:.3f}s/update**,
exact-decay optimization **{exact['median_seconds']:.3f}s**, per-chunk graphs
**{graph['median_seconds']:.3f}s** ({base['median_seconds']/graph['median_seconds']:.2f}x faster than v1).
The registered <=2.5s gate passes. **The complete performance program remains
incomplete:** X42-policy B1/T512 pressure parity fails, so pure scale timing,
pure/control ratios and T2048 pure amortization are not measured. No waiver.

## Decomposition before changes

[DECOMPOSITION.md](DECOMPOSITION.md) and raw baseline receipts precede every fix.
Of the profiled device time, scan uses 5240.911ms and reverse 2651.728ms;
fill/copy/journal bookkeeping is small. The composition already contained the
X36 persistent core, tiled causal reads and journal replay. Repeated exact
rounded decay and repeated unit-strength death-horizon evaluation were the
engineering debt, particularly at the 2048/8192 horizons. The X36 homogeneous
43ms lane price was not the same policy workload.

Exact unit-reference tables contain repeated FP32 multiplications, never pow.
Unit death scheduling is computed once; forward reads advance by one rounded
multiply with mutation timestamps; reverse uses the table for reference 1 and
retains the original exact fallback for all other references. Scan device busy
time falls 5240.911→214.287ms; reverse 2651.728→658.357ms in separate profiles.
Those joint changes explain the measured gain; no per-change causal split is
claimed. X36 replay was ported locally to keep every upstream source read-only.

## Graph results and parity

Per-policy chunk forward/backward sequences use make_graphed_callables. Address
maps are device-authored, eliminating transient pinned-host mapping transfers.
Input shape/dtype/device/gradient flags are checked; dense LM, clipping, AdamW
and health observations remain eager. Runner opt-in: `--cuda-graphs`.
Whole-LM capture attempts failed (OOM, then implicit-stream
capture errors); their failure receipts remain. Per-chunk capture passes loss
and every parameter gradient bitwise for two different B64/T512 batches.

| Dispatch | Host launches, including graphs | Graph launches | Device activities, including fills/copies |
|---|---:|---:|---:|
{launch_table}

Graphs reduce host launch calls but add device copies/gradient bookkeeping.
Five synchronized wall samples per mode are retained; graph versus optimized
eager is {graph['median_seconds']/exact['median_seconds']:.3f}x, with no established graph speed gain.
Peak graph allocation/reservation: {graph['peak_allocated']/2**30:.3f}/{graph['peak_reserved']/2**30:.3f}GiB.
Graphs stay opt-in; FP32 exact-decay optimization is the default. Baseline and
initial eager measurements use physical card 1; final graph, control and slice
measurements use broker-approved card 5, the same RTX6000 Ada SKU. Clock/thermal
conditions are not calibrated, so small timing differences are not established gains.

## Scale probes: failed pure gate, independent transformer controls

X42 pilot_pure/final_pure configurations contain **one** four-head memory lane
and MLPs elsewhere. They are not six-memory-block X40 scaled uniformly.
The additional production-shape X42 gate fails before any pure timing:
142 pressure elements differ (maximum 0.0166666731); full diagnostic in
`receipts/part3_scale_final/parity_T512.json`.
Values, native state, routing decisions and input gradients pass before the
pressure assertion. This is a public-telemetry parity failure, not a numerical
training-gradient failure; it is still a failed gate. The diagnostic compares
both pressure arrays against an independent 128-token case-window calculation.
No protected X31/X42 source was changed to bypass it.

Independent validated X42 transformer probes completed 200 measured optimizer
updates each, three warmups, microbatch 1, one microbatch per update, autocast
bf16 dense layers, checkpointing as configured. These are synthetic repeated
nonpadding-token engineering probes, **not** the configured 256-microbatch
accumulation or trained quality endpoints. Both allocated and reserved peaks
are measured. Pure/control ratios and pure amortization remain unmeasured.

| Transformer shape | T | Parameters | Tokens/s | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|
{table}

Transformer T2048/T512 throughput ratios are {amortization["150M"]:.3f}x at 150M
and {amortization["1.3B"]:.3f}x at 1.3B. These do not establish pure-stack amortization.

## Precision and matched training

bf16 key/value store fails the initial production B64/T512 forward bitwise
gate. Its raw per-policy mismatch counts are in part3_bf16/gate.json. No bf16
performance or gradient claim is made after forward failure. Store remains
FP32; `--store-dtype bf16` fails closed before training.

The final graph-enabled slice completes 153 updates / 5,013,504 tokens in
{training['seconds']:.3f}s: **{training['tokens_per_second']:.1f} tokens/s**, maximum
loss difference versus ignition-v1 **{training['max_loss_difference']:.9g}**
(tolerance 1e-5). Initialization, batches, AdamW, clipping and gates match v1;
no graph warmup counts as training exposure. Per-step differences and source
hashes are receipted. No .pt files or checkpoint-dependent claims.

The original v1 curve has 3052 updates / 100,007,936 tokens and final held-out
PPL 10.37446235 versus the cited reference 6.8564176. These are quality context,
not a new optimized full-training endpoint or new transformer quality run.
The prior part-2 report remains in REPORT.md.

## Validation, reproduction and residual work

The combined run passes 183 tests (179 upstream fixtures plus 4 local); a
subsequent local run passes 5 including the new precision guard. See
part3_validation/tests.log/XML for the X31/X36 fixtures routed through the
local reverse port, x40_tests.log for local architecture/gate tests, and all
production policy/graph parity JSONs. All GPU commands use run_leased.sh and
only the approved interpreter. Commands and their actual exit statuses are
retained in part3_validation/commands.json; expected gate failures remain
labelled failed. Raw profiles are gzip archives with full launch ledgers.

Reproduce optimized profiling with `performance --certify --output
FRESH_DIR`; graph certification/timing with `graph_probe --output FRESH_DIR`;
scale gates with `scale_probe --output FRESH_DIR`; independent controls with
`scale_probe --arm transformer --output FRESH_DIR`; precision with `bf16_gate
--output FRESH_DIR`; matched slice with `train_slice --graph --output FRESH_DIR`.
Use each as `run_leased.sh -m experiments.expressivity_tasks.x40_emender_lm.MODULE`.
`summarize_performance` regenerates this report. Baseline receipts bind the
pre-change source hashes; rerunning that baseline requires the pre-change
implementation with the new instrumentation, not the current optimized model.

Remaining blockers: resolve the X42 pressure/oracle discrepancy without
weakening certification; then run 200 pure updates at both shapes/contexts and
compute ratios/amortization. bf16 parity is failed. Graphs have no established
speed advantage and consume more memory. Long-horizon capacity regimes are
not certified by the short TinyStories curve. The local replay copy requires
maintenance against upstream changes. No pure-scale studyability claim is made.

No WG, workers, subagents, Slurm/Frontier, protected-source edits or .pt files.
Resilient DiLoCo requirement IDs do not apply to these single-GPU probes.
AGENTS.md/CLAUDE.md are unchanged. contact_supervisor/intercom is not exposed in
available tool metadata; no substitute service was started. Work is reported
as partial, with failures retained, rather than attested complete.
'''
    (ROOT/'REPORT_PART3.md').write_text(report)

if __name__=='__main__':main()
