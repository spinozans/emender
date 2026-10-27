"""Rebuild X37 human report from retained JSON; exact project Python only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'receipts'
summary=json.loads((R/'summary.json').read_text())
lines=['# X37 — head differentiation mechanism test', '',
       'X34 control was reused without training. Three new arms each received '
       '100,007,936 TinyStories tokens on the optimized X36 part-4 integrated replay lane (gated b30edb8b snapshot).', '',
       '| Arm | Init identical stories | Final identical stories | Mean final peak/head | PPL | vs control | CUDA event step ms |',
       '|---|---:|---:|---|---:|---:|---:|']
control=summary['control']
lines.append(f"| a: X34 reused | — | 16/16 | {control['primary']['mean_peak_per_head']} | "
             f"{summary['control_validation']['perplexity']:.5f} | 1.00000x | historical |")
for arm,row in summary['arms'].items():
    lines.append(f"| {arm} | {row['initial_primary']['identical_stories']}/16 | "
                 f"{row['primary']['identical_stories']}/16 | {row['primary']['mean_peak_per_head']} | "
                 f"{row['validation']['perplexity']:.5f} | {row['perplexity_ratio_to_control']:.5f}x | "
                 f"{row['training']['median_gpu_step_ms']:.3f} |")
lines += ['', 'Primary endpoint is the same first 16 held-out stories as X34; '
          'all raw per-story occupancy/routing/pressure data are in eval_* receipts. '
          'Identical peaks are a coarse statistic and do not establish identical computation.', '',
          '**Source audit changes the mechanism interpretation.** X34 head rows '
          'are already unequal at initialization (control.json receipts every projection) and at the trained endpoint (control_checkpoint_audit.json). Copying the full W_k matrix into W_q aligns queries with keys within a head; it does not replicate the independently initialized head slices. '
          'The control uses learned transformer hidden states directly, not a token-only '
          'frozen feature table. F12g write corrections select age-1 slots, not a '
          'write-content argmax. Thus the purported identical-parameter permutation '
          'fixed point is not established by the 16/16 occupancy receipt. '
          'X37 tests the registered new interventions; it cannot prove escape '
          'from an exact weight-symmetry fixed point that the control does not occupy. '
          'From an empty store with continuous writing, nonbinding capacity, no strength deaths and nonzero correction residuals, '
          'F12g alternates APPEND and age-1 CORRECTION, yielding ceil(T/2) peak-live '
          'independently of the head geometry. Pops/deaths, write-disable events and '
          'capacity pressure can break that occupancy pattern; read differences alone '
          'do not establish an escape from it.', '',
          '| Arm | Frozen+learned total | Trainable | Matched base total | Discrepancy |',
          '|---|---:|---:|---:|---:|']
for arm in ('b','c','d'):
    r=json.loads((R/f'parameters_{arm}.json').read_text())
    lines.append(f"| {arm} | {r['arm']['total']} | {r['arm']['trainable']} | "
                 f"{r['base_lm']['total']} | {100*r['relative_total_difference']:.7f}% |")
lines += ['', 'Each matched base count is from a separately widened replacement MLP '
          'in the same six-block skeleton; these bases were not trained. Frozen '
          'embeddings are counted. The original X34 base is not claimed to match '
          'the enlarged frozen tables.', '',
          '| Arm | Final R-CONFLICT per head | R-NOMATCH per head | R-CAP per head | Mean pressure per head |',
          '|---|---|---|---|---|']
findings={}
for arm in ('b','c','d'):
    final=json.loads((R/f'eval_{arm}_3052.json').read_text())
    telemetry=[s['telemetry'][0] for s in final['stories']]
    routing={name:[sum(t['routing_per_head'][h][name] for t in telemetry) for h in range(4)]
             for name in ('R-CONFLICT','R-NOMATCH','R-CAP')}
    pressure=[sum(t['pressure_mean_per_head'][h] for t in telemetry)/16 for h in range(4)]
    lines.append(f"| {arm} | {routing['R-CONFLICT']} | {routing['R-NOMATCH']} | {routing['R-CAP']} | {pressure} |")
    means=final['primary']['mean_peak_per_head']
    findings[arm]=dict(routing_per_head=routing,mean_story_pressure_per_head=pressure,
        peak_means_nondecreasing_with_budget=all(a<=b for a,b in zip(means,means[1:])),
        routing_mix_diverged=any(len(set(v))>1 for v in routing.values()))
lines += ['', '| Arm | Final W_in distance from init per head | Pairwise distance range | Monotonic at registered evaluations |',
          '|---|---|---|---|']
for arm,row in summary['arms'].items():
    win=row['W_in']; pair=list(win['pairwise_distance'].values())
    span=f'{min(pair):.6f}–{max(pair):.6f}' if pair else 'N/A'
    lines.append(f"| {arm} | {win['distance_from_init']} | {span} | {row['W_in_distance_monotonic_per_head']} |")
lines += ['', 'All W_in layers start with the same identity slice; geometry is independently '
          'seeded. Pairwise W_in distances therefore measure acquired separation, '
          'not a nonzero random initial offset. Nonmonotonic movements remain reported. '
          'Gradient norms are taken after clipping at evaluation boundaries.', '',
          f"Winner by final 8192-token validation NLL: **{summary['winner']}**. "
          'The complete X34 depth/context/story suite is in winner_probe.json.', '',
          'The combined X31 oracle/X36 regression suite passed 141 tests before training; '
          'four X37 tests verify frozen ownership, nonzero distinct W_in gradients, '
          'ReplayBackward for b/c/d, and exact independent-head composition. '
          'The initial test collection error and its successful correction are retained. '
          'After concurrent upstream edits triggered the boundary gate, all 109 '
          'X31 oracle tests and four X37 tests passed again under the pinned '
          'source loader (113 total). Its initial Triton source-lookup isolation '
          'failure occurred before any c optimizer update; both that failure '
          'and the corrected source/linecache validation are retained. '
          'No numerical gate was relaxed.', '',
          'Actual CUDA-event step samples are in runs/*/curve.jsonl. They exclude '
          'input transfer before the start event and evaluations/checkpoint saves. '
          'Host dispatch timings are asynchronous and are not wall-step latencies. '
          'train_* receipts retain loop wall time (including evaluation/save boundaries, excluding initial setup and hashing), hardware and broker-granted card. '
          'Synchronization occurs at reporting boundaries, not for each step; '
          'X37 does not claim a new zero-sync profiler certification.', '',
          'Residual risks: '+ ' '.join(summary['caveats'])+' '
          'Batch size changes the number of optimizer updates by 8x; quality comparisons '
          'cannot isolate head geometry from that optimization confound. '
          'Story telemetry is 16 deterministic examples, not a population confidence interval. '
          'Policy ordering is partly imposed by budgets; decay horizons exceed many stories. '
          'Geometry is frozen at width 64; this is one registered design, not a hyperparameter sweep.', '',
          'No imported X31/X34/X36 sources were edited by X37. Concurrent X36 work '
          'changed the shared working tree and correctly triggered the arm-boundary gate '
          'after b. Continued c/d execution used the exact byte-verified source archive '
          'through receipts/pinned/run_leased.sh; the completed b checkpoint received '
          'no further optimizer updates. pinned_execution.json records source drift; '
          'training_console.txt retains the boundary refusal. gate_sources.json '
          'pins the runtime bytes for all arms. '
          'No model checkpoints are committed. No WG machinery, delegated agents, '
          'Frontier/Slurm or resilient behavior was used. Resilient requirement IDs: N/A. '
          'AGENTS.md and CLAUDE.md were checked synchronized. '
          'No callable contact_supervisor tool was exposed; coordination attempts and '
          'source-audit discoveries are recorded in receipts/coordination.txt.', '']
(R/'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
(ROOT/'REPORT.md').write_text('\n'.join(lines))

predictions={
    'primary_control':dict(expected='16/16 identical',actual=control['primary']['identical_stories'],passed=control['primary']['identical_stories']==16),
    'primary_b':dict(expected='modest fall below 16/16',actual=summary['arms']['b']['primary']['identical_stories'],passed=summary['arms']['b']['primary']['identical_stories']<16),
    'primary_c':dict(expected='<8/16',actual=summary['arms']['c']['primary']['identical_stories'],passed=summary['arms']['c']['primary']['identical_stories']<8),
    'primary_d_immediate':dict(expected='immediate structured occupancy differentiation',
        initial_identical=summary['arms']['d']['initial_primary']['identical_stories'],
        initial_means=summary['arms']['d']['initial_primary']['mean_peak_per_head'],
        final_means=summary['arms']['d']['primary']['mean_peak_per_head']),
    'routing':{a:findings[a]['routing_mix_diverged'] for a in ('b','c','d')},
    'quality_within_2pct':{a:r['quality_within_2pct'] for a,r in summary['arms'].items()},
    'quality_over_5pct_worse':{a:r['quality_over_5pct_worse'] for a,r in summary['arms'].items()},
    'W_in':{a:dict(all_heads_moved=all(v>0 for v in summary['arms'][a]['W_in']['distance_from_init']),
                  all_pairs_diverged=all(v>0 for v in summary['arms'][a]['W_in']['pairwise_distance'].values()),
                  sampled_monotonic=summary['arms'][a]['W_in_distance_monotonic_per_head']) for a in ('c','d')},
    'quality_comparison_exact_token_match':False,
    'quality_confounds':summary['caveats'],
}
d=predictions['primary_d_immediate']
d['initial_ordered']=all(x<=y for x,y in zip(d['initial_means'],d['initial_means'][1:]))
d['final_ordered']=all(x<=y for x,y in zip(d['final_means'],d['final_means'][1:]))
d['passed']=d['initial_identical']<16 and d['initial_ordered'] and d['final_ordered'] and len(set(d['final_means']))>1
(R/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')

headline=(f"Primary final identical-peak counts: b {summary['arms']['b']['primary']['identical_stories']}/16, "
          f"c {summary['arms']['c']['primary']['identical_stories']}/16, "
          f"d {summary['arms']['d']['primary']['identical_stories']}/16 "
          f"(d init {summary['arms']['d']['initial_primary']['identical_stories']}/16). "
          f"{sum(r['quality_over_5pct_worse'] for r in summary['arms'].values())}/3 new arms "
          'are >5% worse in near-matched-token perplexity than X34. '
          'W_in separation and occupancy differentiation are separate outcomes.')
report=(ROOT/'REPORT.md').read_text()
first,rest=report.split('\n\n',1)
report=first+'\n\n'+headline+'\n\n'+rest
report+='\nPrediction adjudication (see receipts/predictions.json):\n\n'
for key in ('primary_control','primary_b','primary_c','primary_d_immediate'):
    report+=f"- {key}: {'met' if predictions[key]['passed'] else 'FAILED'}; expected {predictions[key]['expected']}.\n"
for arm in ('b','c','d'):
    report+=f"- {arm} quality within 2%: {'met' if predictions['quality_within_2pct'][arm] else 'FAILED'}; >5% worse: {predictions['quality_over_5pct_worse'][arm]}.\n"
(ROOT/'REPORT.md').write_text(report)

probe=json.loads((R/'winner_probe.json').read_text())
assert probe['arm']==summary['winner']
with (ROOT/'REPORT.md').open('a') as report:
    report.write('\nWinner probe outcomes (same zero-shot suite as X34):\n\n')
    report.write('| Depth bin | Pairs | Discrimination accuracy |\n|---|---:|---:|\n')
    for name,row in probe['depth_bins'].items():
        report.write(f"| {name} | {row['pairs']} | {row['accuracy']:.4f} |\n")
    report.write('\n| Context | Windows | Perplexity |\n|---|---:|---:|\n')
    for row in probe['context']:
        report.write(f"| {row['context']} | {row['windows']} | {row['perplexity']:.5f} |\n")
    report.write('\nOne training seed was run per arm. The registered outcomes and '
                 'source audit support these specific mechanism tests; they do not '
                 'establish broad specialization or quality equivalence.\n')

timing={}
for arm in ('b','c','d'):
    curve=[json.loads(line) for line in (ROOT/'runs'/arm/'curve.jsonl').read_text().splitlines()]
    values=sorted(row['gpu_step_ms'] for row in curve[2:])
    timing[arm]=dict(samples=len(values),excluded_warmup_steps=2,
                    median_cuda_event_step_ms=(values[(len(values)-1)//2]+values[len(values)//2])/2,
                    p05_cuda_event_step_ms=values[int(.05*(len(values)-1))],
                    p95_cuda_event_step_ms=values[int(.95*(len(values)-1))],
                    mean_cuda_event_step_ms=sum(values)/len(values),
                    loop_wall_seconds=summary['arms'][arm]['training']['elapsed_seconds'],
                    notes='CUDA stream event elapsed time, not GPU busy time; excludes input copy before event, eval and saves.')
(R/'timing_summary.json').write_text(json.dumps(timing,indent=2)+'\n')

with (ROOT/'REPORT.md').open('a') as report:
    relative=timing['d']['median_cuda_event_step_ms']/timing['c']['median_cuda_event_step_ms']
    report.write(f"\nThe heterogeneous-policy d arm measured **{relative:.3f}x** c's median "
                 'CUDA-event step duration. It uses four independent scalar-policy '
                 'scans and different budgets/pop/death behavior; these contributions '
                 'were not independently profiled or causally separated. The '
                 'operator estimate of quick minutes per arm did not hold for d. '
                 'The full registered token budget was retained without upstream '
                 'patches or reduced batching.\n')

with (ROOT/'REPORT.md').open('a') as report:
    report.write('\nAdditional prediction adjudication:\n\n')
    for arm in ('b','c','d'):
        status='met' if predictions['routing'][arm] else 'FAILED'
        report.write(f'- {arm} routing-mix divergence: {status}.\n')
    for arm in ('c','d'):
        row=predictions['W_in'][arm]
        ok=row['all_heads_moved'] and row['all_pairs_diverged'] and all(row['sampled_monotonic'])
        report.write(f"- {arm} W_in movement, pairwise separation and sampled monotonic growth: {'met' if ok else 'FAILED'}.\n")
