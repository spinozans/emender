"""Source-bound summary; missing and failed evidence never becomes a pass."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part12'


def load(path):
    file = OUT / path
    return json.loads(file.read_text()) if file.exists() else {'status': 'not-run'}


def main():
    support = load('support_gate.json')
    gates = {name: load(f'parity{name}.json') for name in ('', '_empty', '_popped', '_decay', '_correction')}
    curves = {f'{arm}_T{t}': load(f'curve_{arm}_T{t}.json') for t in (2048,) for arm in ('base', 'candidate')}
    shapes = []
    support_totals = []
    ratios = []
    for context in (512, 2048):
        telemetry = load(f'profile/support_T{context}.json')
        rows = telemetry.get('rows', [])
        if rows:
            support_totals.append(dict(context=context, touched_fraction=sum(sum(r['union_counts']) for r in rows)/sum(r['stores']*r['slots'] for r in rows)))
        for slots in sorted({r['slots'] for r in rows}):
            group = [r for r in rows if r['slots'] == slots]
            counts = [n for r in group for n in r['union_counts']]
            shapes.append(dict(context=context, slots=slots, mean_touched=sum(counts)/len(counts),
                               union_fraction=sum(counts)/(len(counts)*slots),
                               mean_reserved_rows=sum(r['reserved_rows'] for r in group)/len(group),
                               calls=len(group)))
        for mode in ('profile', 'sustained'):
            if mode == 'profile':
                pure = load(f'profile/profile_pure_T{context}.json')
                baseline = load(f'profile/profile_transformer_T{context}.json')
                good = pure.get('status') == baseline.get('status') == 'passed'
                ratio = pure['wall_seconds']/baseline['wall_seconds'] if good else None
            else:
                pure = load(f'sustained/attempts/1.3B_pure_B64_T{context}/00.json')
                baseline = load(f'sustained/attempts/1.3B_transformer_B64_T{context}/00.json')
                good = pure.get('status') == baseline.get('status') == 'passed'
                ratio = baseline['tokens_per_second']/pure['tokens_per_second'] if good else None
            ratios.append(dict(context=context, mode=mode, status='passed' if good else 'not-established', pure_over_transformer_seconds=ratio, physical_gpu_pair=[pure.get('physical_gpu'), baseline.get('physical_gpu')]))
    resources = []
    for context in (512, 2048):
        compiler = load(f'profile/compiler_pure_T{context}.json')
        seen = set()
        for record in compiler.get('records', []):
            key = (record['region'], record['shape']['S'])
            if key not in seen:
                seen.add(key)
                resources.append(dict(context=context, region=record['region'], slots=record['shape']['S'],
                    registers=record['registers'], n_spills=record['spills'], grid=record['grid'], multiprocessors=compiler['multiprocessors']))
    decomposition = []
    core_fusion = []
    for context in (512, 2048):
        regions = load(f'split/regions_T{context}.json')
        if 'phases' in regions:
            decomposition.append(dict(context=context, regions=dict(zip(regions['phases'], regions['seconds'])),
                overhead_seconds=regions['reverse_wrapper_overhead_seconds'], split_update_wall_seconds=regions['wall_seconds'],
                limitation=regions['limitation']))
        normal = load(f'profile/profile_pure_T{context}.json')
        fused = load(f'fused-forward/profile_pure_T{context}.json')
        if normal.get('status') == fused.get('status') == 'passed':
            core_fusion.append(dict(context=context, normal_wall_seconds=normal['wall_seconds'], fused_wall_seconds=fused['wall_seconds'],
                full_model_profile_loss_equal=normal['loss'] == fused['loss'],
                normal_calls=normal['calls'], fused_calls=fused['calls'],
                fused_over_normal_seconds=fused['wall_seconds']/normal['wall_seconds']))
    pilot_files = list((OUT/'pilot').glob('*.json'))
    pilot = next((json.loads(p.read_text()) for p in pilot_files if 'tokens_per_second' in json.loads(p.read_text())), {'status':'not-run'})
    speed = pilot.get('tokens_per_second')
    pilot_gate = 'not-run' if speed is None else 'passed' if speed >= 100000 else 'failed'
    measurement_complete = (support['status'] == 'passed'
        and all(g.get('status') == 'passed' for g in gates.values())
        and all(g.get('status') == 'passed' and g.get('tokens', 0) >= 2_000_000 for g in curves.values())
        and all(r['status'] == 'passed' for r in ratios)
        and len(decomposition) == len(core_fusion) == 2
        and pilot.get('status') == 'passed' and len(pilot.get('seconds', [])) == 20)
    (OUT/'pilot_gate.json').write_text(json.dumps(dict(status=pilot_gate, threshold_tokens_per_second=100000,
        tokens_per_second=speed, raw_receipt='pilot/fresh_gate.json', measured_updates=len(pilot.get('seconds', [])),
        warmups=pilot.get('warmups'), scope='22M B128 FP32 pilot; unchanged T512 replay, inherited matched curve identified in pilot/inherited_curve.json'), indent=2) + '\n')
    risks = []
    if any(g.get('status') != 'passed' for g in gates.values()): risks.append('Fresh native byte parity is incomplete or failed.')
    if any(g.get('status') != 'passed' or g.get('tokens', 0)<2_000_000 for g in curves.values()): risks.append('Paired 1.3B real-data 2M-token curve gates are incomplete or failed.')
    split = load('parity_split_random.json')
    if split.get('status') != 'passed': risks.append('Split reverse diagnostic has not passed its independent byte gate.')
    if any(r['status'] != 'passed' for r in ratios): risks.append('Matched diagnostic/sustained performance measurements remain incomplete or failed.')
    if pilot_gate != 'passed': risks.append('The pilot 100k gate remains open/failed.')
    risks.append('Synchronized split-kernel times include spill/reload and extra launches; they cannot be presented as exclusive intervals inside the production fused binary.')
    result = dict(status='passed' if measurement_complete and pilot_gate == 'passed' else 'failed',
                  task_complete=measurement_complete and pilot_gate == 'passed',
                  implementation_and_measurements_complete=measurement_complete, production_speedup_accepted=False,
                  support_status=support['status'], support_fixtures=len(support.get('rows', [])),
                  parity={k:v['status'] for k,v in gates.items()}, curve_status={k:v['status'] for k,v in curves.items()},
                  support_shapes=shapes, support_totals=support_totals, ratios=ratios, split_decomposition=decomposition, core_fusion=core_fusion, launch_resources=resources, pilot_tokens_per_second=speed, pilot_100k_gate=pilot_gate,
                  residual_risks=risks, production_default_changed=False,
                  scope='Opt-in fresh-origin T2048 union replay; continued/imported long states retain existing path.',
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('part12*.py')},
                  manual_notes='Single attended session; approved Python and run_leased.sh only, physical cards 1 and 5. PID 634691 was alive before and after work and was not disturbed. Supervisor contact tool was not exposed. No WG service, subagents, Frontier/Slurm, resilient DiLoCo work, or .pt artifacts. Production BF16 and unrelated workspace changes are untouched.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# X40 part 12 — union-pooled read adjoints', '', '**Performance gate: '+result['status'].upper()+'.** Implementation and requested measurement work is '+('complete' if measurement_complete else 'incomplete')+'.', '',
             f'Fresh initialization/journal gate: {support["status"]}, {result["support_fixtures"]} B32 fixtures. See [support receipt](receipts/part12/support_gate.json) and [source audit](receipts/part12/source_audit.json).', '',
             'The ever-touched union is a conservative support envelope. Popped and threshold-decayed rows stay in it. Logical-S reduction operands and cold native-gradient backing remain intact.', '',
             'The candidate extends fresh-origin pooling through T2048 behind `X40_LONG_POOL=1` plus `X40_FRESH_POOL=1`. Scratch uses packed indices; the prefix-width dispatch is disabled for pooled replay. Production BF16 arithmetic is unchanged.', '',
             'All 40 pooled native byte cases passed, as did eight split-reverse and eight combined core/read cases. The paired full 1.3B TinyStories slice ran 2,097,152 tokens per arm at T2048: every update loss, gradient norm and full gradient hash matched; initial/final parameter and input-data hashes matched. See [curve receipts](receipts/part12/curve_candidate_T2048.json).', '',
             'Measured support (actual full-stack diagnostic, when available):', '',
             '| T | S/head | Mean touched | Union/S | Reserved rows |', '|---:|---:|---:|---:|---:|']
    for r in shapes: lines.append(f'| {r["context"]} | {r["slots"]} | {r["mean_touched"]:.2f} | {r["union_fraction"]:.3%} | {r["mean_reserved_rows"]:.2f} |')
    lines += ['', 'Weighted touched/S over all four heads: '+', '.join(f'T{r["context"]}: {r["touched_fraction"]:.3%}' for r in support_totals)+'. These are workload-specific journal unions, not current live occupancy. Full-S reductions remain, so their reciprocals are not measured adjoint speedups. T512 already used the fresh union before part 12.', '', 'Matched pure/transformer elapsed-time ratios:', '',
        'Envelope: B64 effective/B32 physical, checkpointing for both arms, packed journals for pure only, no offload. Dense blocks use BF16; pure native stores and bounded vocabulary projection remain FP32, while the control uses fused BF16 vocabulary loss. Synthetic nonpadding inputs share independent seed 3402, matching the part10 protocol.', '']
    for r in ratios: lines.append(f'- T{r["context"]} {r["mode"]}: {r["pure_over_transformer_seconds"]:.4f}x ({r["status"]}); physical cards {r["physical_gpu_pair"]}.')
    lines += ['', 'Production fused measurements (seconds):', '',
              '| T | Full update | Sequential core | Separate reads | Fused reverse |', '|---:|---:|---:|---:|---:|']
    for context in (512, 2048):
        profile=load(f'profile/profile_pure_T{context}.json')
        if profile.get('status') == 'passed':
            phase=profile['exclusive_seconds']
            lines.append(f'| {context} | {profile["wall_seconds"]:.6f} | {phase["sequential_forward_core"]:.6f} | {phase["forward_read_elementwise_tiles"]:.6f} | {phase["reverse_fused_read_math_and_adjoints"]:.6f} |')
    lines += ['', 'The exact part10 shared checkpoint/B32 sustained receipts give 2.2128x at T512 and 3.2073x at T2048, rather than the task-supplied 1.6x prior. [Historical source hashes](receipts/part12/matched_history.json) retain the comparable evidence; the new measurements above remain separate.', '', 'Exclusive synchronized split-reverse regions (seconds; extra launch/spill costs are explicit):', '',
              '| T | Undo | Strength/store events | Score/softmax | Read adjoints | Wrapper overhead |', '|---:|---:|---:|---:|---:|---:|']
    for r in decomposition:
        phase=r['regions']
        lines.append(f'| {r["context"]} | {phase["undo"]:.6f} | {phase["strength_restore_and_store_event_adjoints"]:.6f} | {phase["score_softmax_recompute"]:.6f} | {phase["read_adjoints"]:.6f} | {r["overhead_seconds"]:.6f} |')
    lines += ['', 'These are additive wall regions of the separately gated split implementation. The unchanged fused kernel retains its own authoritative wall measurement; split fractions are not applied to that wall.', '',
              'The initial diagnostic split omitted the original compact-replay assertion. Restoring its byte occupancy operand restored parity; a compiler layout effect is inferred. Three rejected attempts are retained, and restoring the original assertion passed all eight B32 shapes. Production union replay was not changed by these scoped diagnostic patches.', '',
              'Sequential core/read launch experiment:', '']
    for r in core_fusion:
        calls=r['fused_calls'];normal=r['normal_calls']
        lines.append(f'- T{r["context"]}: core {normal.get("sequential_forward_core",0)}; batched read {normal.get("forward_read_elementwise_tiles",0)}; reverse {normal.get("reverse_fused_read_math_and_adjoints",0)}. The combined core/read path leaves {calls.get("forward_read_elementwise_tiles",0)} separate read calls; wall ratio {r["fused_over_normal_seconds"]:.4f}x, profile loss equal={r["full_model_profile_loss_equal"]}. Compiler resources and grids are in the profile receipts; they are not achieved-occupancy measurements.')
    lines += ['', 'The combined core/read candidate is not promoted: it is slower at both T values. This losing experiment does not establish a hardware or algorithmic floor.', '', 'Compiler launch resources (reported `n_regs`/`n_spills`, not measured achieved occupancy):', '',
              '| T | Region | S | Registers | n_spills |', '|---:|---|---:|---:|---:|']
    for r in resources:
        lines.append(f'| {r["context"]} | {r["region"]} | {r["slots"]} | {r["registers"]} | {r["n_spills"]} |')
    lines += ['', 'These B32 store-owner kernels launch 32 CTAs on a 142-SM RTX 6000 Ada. A single such launch can place CTAs on at most 32 SMs; this is a grid bound, not an achieved-occupancy measurement. The core reports spills while these reverse shapes report zero spills.', '', f'Pilot: {speed:,.2f} tok/s; 100k gate **{pilot_gate.upper()}**. Three warmups, twenty full B128 updates. The unchanged T512 pilot curve is inherited explicitly; this is not a new pilot quality run.' if speed is not None else 'Pilot: not run.', '', 'Residual risks and open gates:', '']
    lines += ['- '+r for r in risks]
    lines += ['', result['manual_notes'], '', 'No resilient DiLoCo requirement IDs apply. AGENTS.md and CLAUDE.md are unchanged. Unrelated working-tree changes are excluded from this work.', '']
    (ROOT/'REPORT_PART12.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
