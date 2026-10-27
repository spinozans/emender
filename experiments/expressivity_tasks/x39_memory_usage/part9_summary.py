"""Assemble four-arm science from preserved receipts, without re-auditing history."""
import json
import math
import statistics
from pathlib import Path
from hashlib import sha256

ROOT=Path(__file__).resolve().parent
REC=ROOT/'receipts'

def load(name):return json.loads((REC/name).read_text())
def sha(path):return sha256(path.read_bytes()).hexdigest()
def heads(values):return '/'.join('undefined' if x is None else f'{x:.4f}' for x in values)
def save(name,value):(REC/name).write_text(json.dumps(value,indent=2)+'\n')

def main():
    training=load('part9_semantic_full_training.json')
    semantic=load('part9_semantic_full_evaluation.json')
    parity=load('part9_semantic_parity.json')
    parameters=load('part9_semantic_full_parameters.json')
    witness=load('part9_gate_witness.json')
    learned=load('part9_learned_parameters.json')
    assert witness['status']=='passed' and learned['shared_W_in_off_identity_support_norm']>0
    assert training['status']=='completed' and training['step']==3052 and training['tokens']==100007936
    assert parity['status']=='passed'
    curve_path=ROOT/'runs/semantic_full/curve.jsonl'
    curve=[json.loads(x) for x in curve_path.read_text().splitlines()]
    assert len(curve)==3052
    assert [x['step'] for x in curve]==list(range(1,3053))
    assert all(x['tokens']==x['step']*32768 and x['replay_node']=='ReplayBackward' and all(math.isfinite(x[k]) for k in ('loss','ce','grad_norm','gate_mean')) for x in curve)
    control=load('part6_control_evaluation.json')
    rows=[]
    for tag in ('decay','gate_0.01_full','poverty','poverty_rich','semantic_full'):
        e=semantic if tag=='semantic_full' else load(f'part7_{tag}_evaluation.json')
        contexts=[]
        for a,b in zip(e['contexts'],control['contexts']):
            assert a['context']==b['context']
            def average(key):
                return [sum(v)/len(v) for v in zip(*(w[0][key] for w in a['telemetry']))]
            contexts.append(dict(T=a['context'],context_perplexity=a['perplexity'],context_ratio=a['perplexity']/b['perplexity'],
                **{k:average(k) for k in ('allocation_rate_per_token_head','pressure_fraction_per_head','mean_live_per_head','write_enabled_fraction_per_head','append_request_rate_per_token_head','sigmoid_gate_strength_per_head')}))
        band=1.05 if tag=='decay' else 1.02
        slopes=e['occupancy_log_log_slopes_per_head']
        quality=e['perplexity_ratio']<=band
        screen=all(s is not None and s<=.5 for s in slopes)
        rows.append(dict(arm=tag,tokens=e['tokens'],budget=e['budget'],perplexity=e['validation']['perplexity'],ratio=e['perplexity_ratio'],
            registered_quality_band=band,quality_pass=quality,registered_strong_slope_max=.5,slopes=slopes,
            strong_screen=screen,capacity_capped=tag=='poverty',contexts=contexts,
            checkpoint_sha256=e['checkpoint_sha256'],stories=len(e['stories']),depth_bins=e['depth_bins']))
    poor,rich=rows[2:4]
    assert poor['checkpoint_sha256']==rich['checkpoint_sha256']
    sem=rows[-1]
    verdict=('strong occupancy compression screen passes with registered quality' if sem['strong_screen'] and sem['quality_pass'] else
             'occupancy is sublinear, but registered strong-compression screen fails' if all(s is not None and s<1 for s in sem['slopes']) and sem['quality_pass'] else
             'registered quality band fails' if not sem['quality_pass'] else 'no all-head sublinear occupancy result')
    result=dict(status='four arms trained and evaluated',rows=rows,semantic_verdict=verdict,
        training=dict(updates=len(curve),tokens=curve[-1]['tokens'],replay_and_finite_checks=True,curve_sha256=sha(curve_path),physical_gpu=training['physical_gpu']),
        parity=parity,parameters=parameters,optimized_gate_witness=witness,learned_parameters=learned,
        provenance='Parts 7–8 disk snapshot/cached loaded-code distinction stands as receipted; no retrospective certification.',
        residual_risks=['One training seed; only two packed windows per length, 16 stories and seven depth probes.',
            'Fixed-window perplexity band does not establish general quality or retention beyond 4096.',
            'Semantic architecture changes frozen features and total parameters; widened base is a parameter receipt, not a newly trained control.',
            'No trained wire-only or semantic-only ablation isolates the causal contribution of learned W_in.',
            'Hard detached routing has no allocation-count gradient; match-evidence is detached and gate cost optimizes the wired sigmoid.',
            'Historical disk snapshots cannot establish cached loaded-code identity; existing qualification remains final.'])
    save('part9_adjudication.json',result)
    # Standalone vector curve: raw per-update CE and registered checkpoint PPL.
    checkpoints=[]
    for p in REC.glob('part9_semantic_full_*.json'):
        suffix=p.stem.removeprefix('part9_semantic_full_')
        if suffix.isdigit():
            d=json.loads(p.read_text());checkpoints.append((d['tokens'],d['validation']['perplexity']))
    checkpoints.sort()
    def points(values,low,high):
        return ' '.join(f'{60+850*x/100007936:.2f},{250-190*(y-low)/(high-low):.2f}' for x,y in values)
    ce=[(x['tokens'],x['ce']) for x in curve]
    maxce=max(y for _,y in ce)
    maxp=max(y for _,y in checkpoints)
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="590" viewBox="0 0 960 590">
<rect width="960" height="590" fill="white"/>
<g font-family="sans-serif" font-size="15" fill="#222">
<text x="60" y="25">X39 arm 4′ — batch 64, λ=.01, 100,007,936 tokens</text>
<text x="60" y="48">Raw training cross entropy (range 0–{maxce:.2f})</text>
<path d="M60 60V250H910" fill="none" stroke="#777"/>
<polyline points="{points(ce,0,maxce)}" fill="none" stroke="#2463ab" stroke-width="1"/>
<text x="60" y="278">0</text><text x="830" y="278">100M tokens</text>
<g transform="translate(0,290)"><text x="60" y="25">Fixed-window validation perplexity (range 0–{maxp:.2f})</text>
<path d="M60 60V250H910" fill="none" stroke="#777"/>
<polyline points="{points(checkpoints,0,maxp)}" fill="none" stroke="#b65023" stroke-width="2"/>
<text x="60" y="278">0</text><text x="830" y="278">100M tokens</text></g></g></svg>'''
    (REC/'part9_semantic_curve.svg').write_text(svg)
    table='| Arm | Registered PPL ratio / slope | Measured PPL / ratio | Per-head slope | Disposition |\n|---|---|---|---|---|\n'
    names=['1 decay horizon128','2 gate cost .01','3 poverty S128','3 same weights S2944',"4′ shared semantic + match + cost .01"]
    for name,row in zip(names,rows):
        disposition=('capacity-capped; rich transfer fails' if row['capacity_capped'] else
            'quality passes; strong slope screen passes' if row['quality_pass'] and row['strong_screen'] else
            'quality passes; strong slope screen fails' if row['quality_pass'] else 'quality band fails')
        table+=f"| {name} | ≤{row['registered_quality_band']:.2f} / ≤.5 | {row['perplexity']:.8f} / {row['ratio']:.8f} | {heads(row['slopes'])} | {disposition} |\n"
    usage='| Arm | T | Allocations/token/head | Pressure fraction/head | Mean occupancy/head | Context PPL/control |\n|---|---|---|---|---|---|\n'
    for row in rows:
        for c in row['contexts']:
            usage+=f"| {row['arm']} | {c['T']} | {heads(c['allocation_rate_per_token_head'])} | {heads(c['pressure_fraction_per_head'])} | {heads(c['mean_live_per_head'])} | {c['context_ratio']:.6f} |\n"
    depth_table='| Depth bin | Control accuracy | Arm 4′ accuracy | Arm 4′ valid−corrupted logprob |\n|---|---|---|---|\n'
    for label,d in semantic['depth_bins'].items():
        depth_table+=f"| {label} | {control['depth_bins'][label]['accuracy']:.4f} | {d['accuracy']:.4f} | {d['valid_mean_logprob']-d['corrupted_mean_logprob']:.6f} |\n"
    story_peaks=[statistics.median(story['telemetry'][0]['peak_live_per_head'][h] for story in semantic['stories']) for h in range(4)]
    story_max=[max(story['telemetry'][0]['peak_live_per_head'][h] for story in semantic['stories']) for h in range(4)]
    result['semantic_story_median_peak_per_head']=story_peaks
    result['semantic_story_max_peak_per_head']=story_max
    save('part9_adjudication.json',result)
    report=f'''# X39 part 9 — four trained arms; compression receipts

**Arm 4′ completed first in part 9:** 3052 B64 updates, 100,007,936 target
tokens, λ=.01, seed 3401. Its fixed-window perplexity is
**{sem['perplexity']:.8f}**, ratio **{sem['ratio']:.8f}** to B64 control
6.8564175640. Per-head log-log peak-occupancy slopes at
T=512/1024/2048/4096 are **{heads(sem['slopes'])}**.
The registered verdict is **{verdict}**. These finite-window occupancy
measurements do not independently establish semantic information content.

{table}

All training endpoints above are 100,007,936 tokens. Quality and slope bands
are pre-registered acceptance screens, not predictions. Poverty poor/rich are
one trained arm at two deployment budgets, with identical checkpoint hash.
Part-7/8 provenance qualifications stand exactly as receipted: recorded hashes
identify disk snapshots and cannot conclusively identify cached bytecode.
This final synthesis adds no retrospective certification or reconciliation.

## Compression conclusions

1. **Decay is the built-in occupancy regulator.** A 128-token untouched-slot
   fade horizon and one birth per two tokens yield the observed equilibrium
   of 64 slots/head. Occupancy slope is approximately zero by construction,
   while allocation remains .5/token/head. This is expiration of old state,
   not evidence of learned semantic deduplication. `part7_decay_shape.json`
   preserves 128-token NLL bins and context ratios; full position curves and
   occupancy profiles are in `part7_decay_evaluation.json`.
2. **Gate cost changes the constant without demonstrating a different
   asymptotic exponent.** Allocations fall to about .14–.30/token/head at
   T4096, but measured slopes .9294–.9602 fail the ≤.5 strong-compression
   screen. The finite-range estimates are near one; they do not prove an
   exact asymptotic exponent. The .01 operating point passes its 2% quality
   band; .1 failed the matched 20M selection pilot.
3. **Poverty thrift is situational, not dispositional in this experiment.**
   The same weights give slopes approximately 0 at S128 and 1 at S2944.
   At S128 pressure rises .5→.9375 over these lengths; the flat curve is a
   capacity cap. Rich-budget allocations return to .5/token/head. Strong
   compression does not transfer to rich deployment.
4. **Arm 4′: {verdict}.** Shared W_in, detached recurrent match evidence and
   gate cost are now trained rather than inferred from the hand-set witness.
   Peak occupancies over the registered lengths are 9–14 slots/head, far below
   S2944; pressure is zero. At T4096 allocations are .00256–.00305/token/head.
   This supports the registered conceptual-cache occupancy hypothesis over
   these lengths, with the retained-information limits below.
   The table reports its measured outcome without substituting retrieval
   success or a capacity cap for conceptual-cache evidence.

## Arm 4′ execution and evidence

`part9_semantic_parity.json` records independent sequential X31 state/evidence
and exact gate-decision parity at B64/H4/K64/V128/S2944/T512. Full-scan
forward/state/trace/pressure and gradients match X31 bitwise, and ReplayBackward
is asserted. The evidence score comparison allows rtol 2e-5 / atol 2e-6 for
PyTorch versus tiled reduction. This is a targeted production-shape test,
not exhaustive adversarial coverage. An initial harness attempt used an
incorrect trace field name and failed before training; its console is retained.
The corrected gate passed before the accepted run. A post-training replay
recheck explicitly compares every trace field, including the informational
matched index and POP-fired bit; it removes the harness's conditional field
skipping. Exact pre-training harness bytes and its passing certificate are
preserved as `runs/semantic_full/pretraining_parity.py` and
`part9_semantic_parity_pretraining.json`. Model/training code is unchanged.
Existing CPU suite: 11 passed.

`part9_gate_witness.json` reruns both required alias ablations on the actual
optimized wire at corrected READ scale: semantic+wired gives novel/repeat
allocation 1/0; identity+wired and semantic+unwired give 1/1. This remains a
hand-set circuit witness, not a trained-language ablation. The learned-parameter
receipt confirms a single shared 64×256 W_in moved from initialization
(distance {learned['shared_W_in_distance_from_init']:.6f}); its off-identity
support norm {learned['shared_W_in_off_identity_support_norm']:.6f} shows learning
beyond simple decay. Match coefficients finish at approximately −1.785.

Every accepted update asserts ReplayBackward and finite loss/gradient norm.
Raw CE/loss/gate-strength/norm records are `runs/semantic_full/curve.jsonl`;
checkpoint validation receipts are `part9_semantic_full_<step>.json` and the
standalone curve is `part9_semantic_curve.svg`. Training, checkpoint hash,
source/config/data fingerprints and exact pre-training protocol bytes are
retained. Physical broker card {training['physical_gpu']} was used; no forbidden
or certified-held card was requested. Checkpoint weights remain local and
ignored; no .pt is committed.

Parameter receipt: semantic {parameters['actual_arm']['total']:,} total /
{parameters['actual_arm']['trainable']:,} trainable; widened base
{parameters['parameter_matched_base']['total']:,} total, relative total gap
{parameters['relative_total_difference']:.8g}. The widened base is untrained;
the registered B64 quality comparator remains the previously trained control.
Architecture and initialization are specified before training in
`runs/semantic_full/registered_protocol.md`. Existing append error placement
is preserved. The cost applies to the actual wired sigmoid.

`part9_semantic_full_evaluation.json` retains two windows at every length,
full position NLL, per-head occupancy profiles, successful allocations,
requests, write-enable and sigmoid strengths, pressure/R-CAP/W64, first 16
story profiles with truncation flags, and seven depth probes. These are the
same probes as the control. `part9_adjudication.json` derives the final table.

{usage}

Arm 4′'s 16-story median peak occupancy per head is **{heads(story_peaks)}**;
maximum is **{heads(story_max)}**. The complete per-token profiles, story
lengths and truncation flags are retained in its evaluation receipt.
Depth bins use very few paired prompts; these measurements do not by
themselves establish general recall or semantic information preservation.

{depth_table}

## Structural-lessons ledger, parts 1–9

| Part | Receipted lesson | Limit |
|---|---|---|
| 1 | Append amplitude scales stored values after routing; it is not allocation probability. | A differentiable gate does not imply differentiable allocation count. |
| 2 | Penalizing continuous write-enable sigmoid can cross the hard threshold. | Synthetic controller evidence does not choose a language-model lambda. |
| 3 | Scarcity supplies loss through capacity-limited recall; semantic retrieval alone leaves writes enabled. | Retrieval and write suppression are separate circuit properties. |
| 4 | Hand-set semantic keys plus read-match wire suppress alias-repeat writes; both ablations fail. | Constructive witness is not trained natural-text evidence; its historical score scale differs. |
| 5 | Missing directive/control and optimized wire were explicit execution blockers in archived work. | An unrun arm is neither a null nor a compression result. |
| 6 | Exact-exposure B64 control resolves pilot/full quality denominators. | B8 control was not optimizer-update matched. |
| 7 | Decay equilibrium, gate-cost coefficient reduction, poor/rich transfer failure measured. | One seed; original source qualification retained. |
| 8 | Disk snapshot hashes do not identify already cached bytecode; current oracle parity passes. | This distinction is the final provenance disposition. |
| 9 | Optimized recurrent match wire certified, trained and evaluated with shared W_in. | {verdict}; finite windows do not prove general conceptual memory. |

Residual risks: {' '.join(result['residual_risks'])} Rich pressure zero is
not an improvement over control, which also has zero pressure here. A
low slope caused by information loss is not by itself successful compression.

---

'''
    path=ROOT/'REPORT.md'
    old=path.read_text()
    if old.startswith('# X39 part 9 —'):
        old=old.split('\n---\n\n',1)[1]
    path.write_text(report+old)
    receipt_path=ROOT/'RECEIPTS.md'
    prior=receipt_path.read_text()
    if prior.startswith('# Part 9 final receipt'):
        prior=prior.split('\n---\n\n',1)[1]
    receipt_path.write_text('# Part 9 final receipt — all four trained arms\n\n'+table+
        '\nArm 4′ completed 3052 updates / 100,007,936 target tokens. '+verdict+'.\n'+
        'Evidence: part9_adjudication.json, part9_semantic_full_evaluation.json, '+
        'part9_semantic_parity.json, raw curve and checkpoint validation receipts. '+
        'Historical provenance qualifications stand as receipted; no retrospective certification.\n\n---\n\n'+prior)
    print(json.dumps(dict(semantic_verdict=verdict,ppl=sem['perplexity'],ratio=sem['ratio'],slopes=sem['slopes'])),flush=True)

if __name__=='__main__':main()
