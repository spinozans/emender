"""Derive X39 part-7 adjudication from immutable machine receipts."""
import json
import math
from pathlib import Path
from hashlib import sha256

ROOT = Path(__file__).resolve().parent
REC = ROOT/'receipts'
def read(name):
    return json.loads((REC/name).read_text())
def write(name, value):
    (REC/name).write_text(json.dumps(value, indent=2)+'\n')
def digest(path):
    return sha256(path.read_bytes()).hexdigest()

control = read('part6_control_evaluation.json')
selection = read('part7_lambda_selection.json')
cost = selection['selected']
tags = ['decay'] + ([f'gate_{cost:g}_full'] if cost is not None else []) + ['poverty', 'poverty_rich']
rows = []
for tag in tags:
    result = read(f'part7_{tag}_evaluation.json')
    contexts = []
    for a,b in zip(result['contexts'],control['contexts']):
        assert a['context'] == b['context']
        def average(key):
            metrics = [window[0][key] for window in a['telemetry']]
            return [sum(head)/len(head) for head in zip(*metrics)]
        contexts.append(dict(context=a['context'], perplexity=a['perplexity'],
            control_perplexity=b['perplexity'], perplexity_ratio=a['perplexity']/b['perplexity'],
            allocation_rate_per_token_head=average('allocation_rate_per_token_head'),
            pressure_fraction_per_head=average('pressure_fraction_per_head'),
            write_enabled_fraction_per_head=average('write_enabled_fraction_per_head'),
            mean_live_per_head=average('mean_live_per_head')))
    band = 1.05 if tag == 'decay' else 1.02
    slopes = result['occupancy_log_log_slopes_per_head']
    ratio = result['perplexity_ratio']
    row = dict(arm=tag, tokens=result['tokens'], budget=result['budget'],
        perplexity=result['validation']['perplexity'], perplexity_ratio=ratio,
        quality_band=band, quality_pass=ratio <= band,
        more_than_5pct_quality_cost=ratio > 1.05, slopes=slopes,
        slope_quality_screen_per_head=[s is not None and s <= .5 and ratio <= band for s in slopes],
        capacity_capped=tag == 'poverty',
        strong_compression_per_head=[s is not None and s <= .5 and ratio <= band and tag != 'poverty' for s in slopes],
        contexts=contexts, depth_bins=result['depth_bins'],
        story_count=len(result['stories']), checkpoint_sha256=result['checkpoint_sha256'])
    rows.append(row)
write('part7_adjudication.json',dict(status='partial: semantic arm not run', rows=rows,
    lambda_selection=selection, semantic=read('part7_semantic_blocker.json')))
poor=next(row for row in rows if row['arm']=='poverty')
rich_check=next(row for row in rows if row['arm']=='poverty_rich')
assert poor['checkpoint_sha256'] == rich_check['checkpoint_sha256']
assert poor['budget'] == 128 and rich_check['budget'] == 2944

decay = read('part7_decay_evaluation.json')
shape=[]
for a,b in zip(decay['contexts'],control['contexts']):
    bins=[]
    for start in range(0,a['context'],128):
        end=min(start+128,a['context'])
        an=sum(a['per_position_nll'][start:end])/(end-start)
        bn=sum(b['per_position_nll'][start:end])/(end-start)
        bins.append(dict(start=start,end=end,nll=an,control_nll=bn,delta_nll=an-bn))
    shape.append(dict(context=a['context'],perplexity_ratio=a['perplexity']/b['perplexity'],bins=bins))
write('part7_decay_shape.json',dict(bin_tokens=128, contexts=shape,
    raw_per_position_nll='part7_decay_evaluation.json and part6_control_evaluation.json'))

# The registered decision is over the swept set. Interpolation describes the
# quality boundary; it never invents a new trained operating point.
pilots=selection['rows']
crossings=[]
for left,right in zip(pilots,pilots[1:]):
    lo,hi=left['perplexity'],right['perplexity']
    threshold=selection['threshold']
    if lo != hi and min(lo,hi) <= threshold <= max(lo,hi):
        weight=(threshold-lo)/(hi-lo)
        crossings.append(dict(bracket=[left['cost'],right['cost']],
            log_lambda_interpolated=10**(math.log10(left['cost'])+weight*(math.log10(right['cost'])-math.log10(left['cost'])))))
write('part7_lambda_interpolation.json',dict(matched_tokens=selection['matched_tokens'],
    threshold=selection['threshold'], crossings=crossings, selected_swept_lambda=cost,
    caveat='Descriptive log-lambda linear-PPL interpolation only; no extrapolation and no interpolated lambda trained. Endpoint validations use identical exact exposure.'))

sources=[]
for run in (ROOT/'runs').iterdir():
    reg=run/'registration.json'
    if not reg.exists() or run.name == 'control_b64':
        continue
    fp=json.loads(reg.read_text())
    protocol=run/'registered_protocol.md'
    if not protocol.exists():
        protocol.write_bytes((ROOT/'RECEIPTS.md').read_bytes())
    mismatches=[]
    for name,expected in fp['sources'].items():
        path=protocol if name == 'x39_memory_usage/RECEIPTS.md' else ROOT.parent/name
        current=digest(path)
        if current != expected:
            mismatches.append(dict(source=name, registered_sha256=expected, current_sha256=current))
    curve=[json.loads(line) for line in (run/'curve.jsonl').read_text().splitlines()]
    assert len(curve) == fp['steps']
    assert [r['step'] for r in curve] == list(range(1,fp['steps']+1))
    assert all(r['replay_node']=='ReplayBackward' and math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in curve)
    control_first=json.loads((ROOT/'runs/control_b64/curve.jsonl').read_text().splitlines()[0])['loss']
    if fp['arm'] == 'gate':
        assert curve[0]['ce'] == control_first
    sources.append(dict(run=run.name, source_hashes_stable=not mismatches,
        source_mismatches=mismatches, updates=len(curve),
        first_ce=curve[0]['ce'], control_first_ce=control_first,
        initial_ce_matches_control=curve[0]['ce'] == control_first,
        replay_and_finite_checks=True, last_tokens=curve[-1]['tokens']))
provenance_failed=any(row['source_mismatches'] for row in sources)
write('part7_integrity.json',dict(status='failed-source-provenance' if provenance_failed else 'passed', runs=sources,
    interpretation='All accepted runs shared one Python process. External on-disk X36 edits occurred during that process. Cached imports do not establish a complete loaded-source attestation; later disk fingerprints cannot certify loaded versions. No upstream files restored or edited by this task.'))
adjudication=read('part7_adjudication.json')
adjudication.update(status='provisional: source-provenance failure; semantic arm not run',
    source_provenance_passed=not provenance_failed)
for row in adjudication['rows']:
    row['certified_compression_per_head']=[screen and not provenance_failed for screen in row['strong_compression_per_head']]
write('part7_adjudication.json',adjudication)
def slopes_text(row):
    return '/'.join('undefined' if s is None else f'{s:.4f}' for s in row['slopes'])

table = '| Arm | Pre-registered | Measured PPL / control ratio | Per-head slope | Disposition |\n|---|---|---|---|---|\n'
names={'decay':'1 decay, horizon 128', 'poverty':'3 poverty, S128',
       'poverty_rich':'3 same weights, S2944'}
for row in rows:
    label=names.get(row['arm'],f'2 gate cost, lambda {cost}')
    requirement=f"100M; PPL ratio <= {row['quality_band']}; slope <= .5"
    disposition=('capacity-capped; rich transfer is the disposition test' if row['capacity_capped'] else
        'provisional compression screen passes; source attestation fails' if all(row['strong_compression_per_head']) and provenance_failed else
        'strong compression with passing quality' if all(row['strong_compression_per_head']) else
        'quality passes; strong compression fails' if row['quality_pass'] else 'quality band fails')
    table+=f"| {label} | {requirement} | {row['perplexity']:.8f} / {row['perplexity_ratio']:.8f} | {slopes_text(row)} | {disposition} |\n"
table+='| 4\' semantic keys + match wire | Operating lambda; 100M; parity + ReplayBackward | Not run | Unmeasured | Optimized recurrent wire absent; incomplete, not a trained null |\n'

pilot_table='| Lambda | PPL at 20,021,248 tokens | Ratio to control | Within 2% |\n|---|---|---|---|\n'
for row in pilots:
    pilot_table+=f"| {row['cost']} | {row['perplexity']:.8f} | {row['perplexity']/selection['reference']:.8f} | {row['perplexity'] <= selection['threshold']} |\n"

usage_table='| Arm | T | Allocation/token/head (two-window mean) | Pressure fraction/head | Context PPL / control |\n|---|---|---|---|---|\n'
for row in rows:
    for c in row['contexts']:
        allocation='/'.join(f'{v:.4f}' for v in c['allocation_rate_per_token_head'])
        pressure='/'.join(f'{v:.4f}' for v in c['pressure_fraction_per_head'])
        usage_table+=f"| {row['arm']} | {c['context']} | {allocation} | {pressure} | {c['perplexity_ratio']:.6f} |\n"

rich=next(r for r in rows if r['arm']=='poverty_rich')
transfer=('No strong compression transfers to rich deployment.' if not any(rich['strong_compression_per_head'])
          else 'Rich deployment has at least one head passing the compression/quality screen; inspect the per-head receipt.')
lessons='''| Structural lesson | Trained evidence and limit |
|---|---|
| Decay bounds retained memory without reducing writes | Every registered length peaks at 64 slots/head; slope about zero, allocation .5/token/head, zero rich pressure; short-context PPL costs .975%. This is recent retention, not semantic deduplication. |
| Write-enable cost reduces the allocation coefficient | Selected lambda .01 gives slopes .9294–.9602: fewer writes, but the <=.5 strong-compression criterion fails in all heads. PPL costs .798%. |
| Capacity caps can mimic compression | S128 occupancy must be interpreted alongside pressure and the same trained weights at S2944; low capped slope alone is not learned compression. |
| Semantic retrieval is not trained deduplication evidence | Prior constructive CPU wire passes its two ablations, but no optimized recurrent wire or trained semantic result exists. |
'''

summary=f'''## Part 7 measured result — three trainings complete; provenance failed; fourth unrun

**All new scientific measurements are provisional.** Final integrity checking
detected external edits to X36 `backward.py` and `replay_setup.py` during the
single Python training process. Initial decay registration matched the B64
control's hashes for these modules, but later registrations saw changed disk
bytes. Cached Python imports suggest that much loaded code stayed unchanged;
this is not a complete loaded-source attestation. The unchanged-source gate
**failed**. `receipts/part7_integrity.json` retains each registered/current hash
mismatch. No training is claimed fully certified from this evidence. Upstream
files were read-only for this task and were not restored or modified.

Arms 1–3 completed sequentially at B64, seed 3401, context 512, each with
100,007,936 target tokens. The three lambda pilots each completed 20,021,248
tokens. Corrected runs held broker card 0 exclusively; certified cards 2/3/4
and forbidden card 7 were excluded by the unchanged X39 wrapper.
**The four-arm task is incomplete:** arm 4' was not trained. This is an
implementation blocker, not a scientific null from a trained semantic model.

{table}

The control is 6.8564175640 at 100M and 14.6701488443 at the exact pilot
exposure. The selected lambda is **{cost}**, the largest swept value passing
the registered threshold {selection['threshold']:.8f}. The full run starts
independently; it is not a continuation of its pilot.

{pilot_table}

`receipts/part7_lambda_interpolation.json` records descriptive log-lambda/PPL
boundary interpolation between bracketing pilots, with no extrapolation and
no invented unswept operating point. Validation checkpoint receipts and every
training loss/gate mean are retained as curves. The repeated .01 full-run
20M validation is 14.7947889612, versus the pilot's 14.7895112197; both pass.
Same-seed CUDA training is not claimed to be bit-identical across executions.

### Synthesis and structural-lessons ledger

The provisional decay measurements show non-capacity-limited sublinear retained memory at an acceptable
short-context quality cost. Its unchanged .5 allocation rate distinguishes
expiration from selective writing. Gate cost reduces writes substantially,
but does not produce the registered strong sublinear scaling. {transfer}
The poor/rich evaluations share the exact checkpoint hash, changing only the
deployment budget. Pressure zero in a rich budget is not itself an improvement:
the control also has zero pressure over these lengths.

{lessons}

{usage_table}

Full occupancy profiles (mean, peak, final), allocation/request/enable/sigmoid
rates, per-head pressure/R-CAP/W64 statistics, first 16 story profiles with
truncation flags, seven depth probes and all per-position context NLL are in
the per-arm `part7_*_evaluation.json` receipts. `part7_decay_shape.json` gives
128-token NLL bins and matched context ratios; raw position curves remain in
the evaluation receipts. The decay context ratios are about 1.0099/.9611/
.9104/.9283 at T=512/1024/2048/4096: longer-context degradation relative to
control is not observed on these registered windows.

### Validation, provenance and remaining work

Every invocation passed a fresh exact-configuration B64/H4/K64/V128/T512
comparison against certified X31: forward values, final state, routing traces
and gradients bitwise. This is the targeted production-shape gate, not a new
claim that the entire adversarial suite was rerun. Parameter receipts match
the B64 control's 21,913,872 total / 21,847,052 trainable tape parameters
against base_lm's 21,914,119 total. Every accepted update asserts ReplayBackward
and finite loss/gradient norm. All 11 existing X39 CPU tests pass.
`part7_integrity.json` verifies uninterrupted step/exposure sequences but
**fails unchanged-source provenance**. Each run retains the exact
`registered_protocol.md` bytes used for its fingerprint.

An initial implementation attempt was stopped after 460 updates / 15,073,280
tokens because its RNG-consuming model-construction order differed from the
control. It is archived under `receipts/part7_aborted_initialization`, excluded
from all results, and replaced by fresh runs matching control construction
order. Every gate run's first-step CE exactly equals the control's first CE.
No checkpoint files are committed. Pre-existing upstream working-tree edits
remain untouched; their source hashes identify the implementations used.

Arm 4' still requires an optimized recurrent pre-read-match-to-write-enable
lane and its exact oracle/replay certification. `part4_match.py` is explicitly
a sequential CPU witness; its 1/sqrt(K) evidence scale differs from certified
READ's 2/sqrt(K). The archived report also leaves error-term placement
unresolved. No architecture was silently substituted. Discovery by tool name
and description exposed no contact_supervisor/Intercom capability; no alternate
orchestration service was started. `part7_semantic_blocker.json` records this
remaining work. A title claiming four trained arms would misstate the result.

Residual risks: failed source provenance from concurrent external changes,
one training seed, small registered validation/probe samples,
same-seed numerical variation, and no learned semantic-arm evidence. Capacity
limits can destroy information while flattening slopes. Reported bands are
fixed-window comparisons and do not establish general language quality or
longer-than-4096 retention.
'''

old_report=(ROOT/'REPORT.md').read_text()
history='\n## Historical reports (superseded where arms 1–3 are described as unrun)\n\n'
if history in old_report:
    old_report=old_report.split(history,1)[1]
(ROOT/'REPORT.md').write_text('# X39 part 7 — provisional three-arm measurements; source gate failed\n\n'+summary+
    history+old_report)
old_protocol=(ROOT/'RECEIPTS.md').read_text()
if '## Part 7 measured dispositions (post-run)' in old_protocol:
    old_protocol=old_protocol.split('\n',1)[0]+'\n\n## Part 6 amendment'+old_protocol.split('## Part 6 amendment',1)[1]
first,rest=old_protocol.split('\n',1)
(ROOT/'RECEIPTS.md').write_text(first+'\n\n## Part 7 measured dispositions (post-run)\n\n'+table+
    '\n**Provisional measurements: unchanged-source attestation failed.** External X36 backward/replay_setup edits occurred during the single training process; `receipts/part7_integrity.json` retains the mismatches. These rows are not fully certified compression receipts.\n\n'+
    '\nThe operator part-7 instruction authorizes the ordered usage runs and exact B64 comparisons. '
    'Original pre-training registration bytes are retained in each run. '
    'The selection rule and quality bands were not changed after results. '
    'Full trained adjudication: `receipts/part7_adjudication.json`.\n\n'+pilot_table+
    '\n### Part 7 structural-lessons ledger\n\n'+lessons+'\n'+transfer+'\n\n'+rest)
print(json.dumps(dict(rows=[{k:r[k] for k in ('arm','perplexity','perplexity_ratio','slopes')} for r in rows],integrity=sources),indent=2))
if provenance_failed:
    raise SystemExit(2)
