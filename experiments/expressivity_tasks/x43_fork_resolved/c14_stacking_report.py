"""Render all signs and the exact forced-walk failure without smoothing it."""
import json
from .c14_stacking_score import ROOT,OUT,write
from .c14_documents import sha


def main():
    t=json.loads((OUT/'table.json').read_text()); v=json.loads((OUT/'verdicts.json').read_text())
    validation=json.loads((OUT/'scoring_validation.json').read_text())
    def row(arm,stratum): return t[f'{arm}/final_test/{stratum}']
    def fmt(x): return '∞' if x=='Infinity' else ('—' if x is None else f'{x:.9f}')
    arms=['dense','v2_on','dense_c13','v2_on_c13','dense_soft','v2_on_soft','array_alone','transformer']
    lines=['# C14 P3 emender stacking, gain decomposition, and array-alone baseline','',
        '**Soft stacking meets both P3 comparisons: 1.505908651 versus v2 alone 1.509260010 and dense+soft calibration 1.695958394. C13 stacking with exact walks: P3 FAIL.** A nonempty single-branch Wikipedia continuation is not a guaranteed Pile continuation. Exact forced walks assign zero probability to real targets; both dense+C13 and v2+C13 have infinite total NLL. No smoothing, target-conditioned abstention, or final-test tuning repairs this literal C13 row. The no-walk variant was declared in analysis_plan.json before final scoring; its PASS does not erase the exact-walk FAIL.','',
        'All NLLs are natural-log nats/byte on the frozen document-disjoint final-test half. All weights are frozen. Calibration uses all registered calibration nonempty fork positions, C13’s intercept/max-neural-probability/log1p-context/log1p-support logistic gate, normalization and bounded L-BFGS-B. Both gates and their hash freeze are written before scoring final-test. No extraction, index changes, training, union-service commands, or .pt commits.','',
        '## Final-test comparison','',
        '| Stratum | Targets | Dense | v2_on | Dense+C13 walk | v2+C13 walk | Dense+soft gate | v2+soft gate | Array alone | Dense transformer |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    strata=['total','window_reset_1_16','window_interior_17_128','wikipedia_support','empty_support','supported_target','nonempty_zero_target','zero_target','fork','nonfork','forced','forced_zero_target']
    for s in strata:
        lines.append('| '+s+' | '+str(row('dense',s)['targets'])+' | '+' | '.join(fmt(row(a,s)['nll']) for a in arms)+' |')
    soft=row('v2_on_soft','total')['nll']; neural=row('v2_on','total')['nll']; dsoft=row('dense_soft','total')['nll']
    soft_gain=row('v2_on','total')['total_nll']-row('v2_on_soft','total')['total_nll']
    reset_gain=row('v2_on','window_reset_1_16')['total_nll']-row('v2_on_soft','window_reset_1_16')['total_nll']
    lines += ['',f"Of the small soft-fusion improvement, **{reset_gain/soft_gain:.3%}** comes from window-reset positions 1–16. Interior 17–128 changes by **{row('v2_on_soft','window_interior_17_128')['nll']-row('v2_on','window_interior_17_128')['nll']:+.9f}**. Zero-target and nonfork NLLs worsen; all signs remain in the table.",'',f"No-walk comparison signs: beats consultation alone **{v['soft_beats_consultation_alone']}**, beats dense+soft calibration **{v['soft_beats_dense_calibration']}**, beats dense transformer **{v['soft_beats_transformer']}**. Conjunctive P3 without walk: **{v['P3_without_walk']}**.",'',f"Forced-walk zero-probability final targets: **{row('v2_on_c13','total')['infinite_losses']:,}**. These targets remain in every total. Empty-support positions abstain to the neural model. Supported-target C13 loss is finite but does not rescue total P3.",'',
        f"The v2 soft diagnostic has total NLL **{soft:.9f}**, change **{soft-neural:+.9f}** versus v2 alone and **{soft-dsoft:+.9f}** versus the actual dense+soft gate. Against the frozen historical dense transformer control **1.404434206**, its change is **{soft-1.404434206:+.9f}**. The calibrated-fusion comparisons pass without walks; the predeclared primary literal C13 recipe remains a failure. The gain over v2 alone is small (about 0.22%) and has no uncertainty estimate.",'',
        'The requested dense+calibration anchor “2.277” is not a committed fused final-test result: 2.277418521 is the historical raw dense calibration-half NLL; the paired raw dense final-test NLL is 2.276133549. The actual calibrated dense rows are measured above. The transformer control is the matched frozen final-window historical reference (batch 16). Its existing position-wise loss JSONL is hash-checked against the committed training receipt and identity-joined with every frozen window; the support/fork columns above are newly aggregated from those historical losses, with no transformer inference or service access.','',
        '## Supported-target gain by support length','',
        'Gain below is v2_on minus dense: negative means lower loss. Contribution is the group’s loss-sum change divided by all supported final targets; contributions add to the measured replay supported-target change (within 3e-9 of the committed −1.116369511). Gain share divides the group loss-sum change by the total supported loss-sum change, so harmful groups have negative shares. No bins were used for fitting.','',
        '| Length bytes | Supported targets | Dense NLL | v2 NLL | Array NLL | Change | Contribution | Gain share |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    denom=row('dense','supported_target')['targets']; delta=row('v2_on','supported_target')['total_nll']-row('dense','supported_target')['total_nll']
    decomp={}
    for label in ['1-3','4-12','13-64','65+']:
        s='length_'+label+'/supported'; d=row('dense',s); n=row('v2_on',s); change=n['total_nll']-d['total_nll']
        decomp[label]=dict(targets=d['targets'],dense_nll=d['nll'],v2_nll=n['nll'],change_nll=n['nll']-d['nll'],change_sum=change,contribution=change/denom,gain_share=change/delta)
        lines.append(f"| {label} | {d['targets']:,} | {fmt(d['nll'])} | {fmt(n['nll'])} | {fmt(row('array_alone',s)['nll'])} | {n['nll']-d['nll']:+.9f} | {change/denom:+.9f} | {change/delta:.3%} |")
    lines += ['','## Calibration-half rows (fit split, not an acceptance set)','',
        '| Stratum | Targets | Dense | v2_on | Dense+C13 walk | v2+C13 walk | Dense+soft gate | v2+soft gate | Array alone | Dense transformer |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for st in ['total','supported_target','nonempty_zero_target','empty_support','fork','nonfork']:
        rr=[t[f'{arm}/calibration/{st}'] for arm in arms]
        lines.append('| '+st+' | '+str(rr[0]['targets'])+' | '+' | '.join(fmt(r['nll']) for r in rr)+' |')
    paired=json.loads((ROOT/'receipts/c14_paired_eval/table.json').read_text())
    lines += ['','## Frozen all-input-zero reference','',
        'Unchanged committed v2_off rows (all 260 inputs zero, no fusion); no replay or tuning of this arm:',
        '', '| Stratum | Targets | v2_off NLL |', '| --- | ---: | ---: |']
    for st in ['total','window_reset_1_16','window_interior_17_128','wikipedia_support','empty_support','supported_target','nonempty_zero_target','zero_target','fork']:
        rr=paired[f'v2_off/final_test/{st}']
        lines.append(f"| {st} | {rr['targets']:,} | {rr['nll']:.9f} |")
    lines += ['','## Fork decomposition','',
        '| Length / supported stratum | Targets | Dense NLL | v2 NLL | Change | Contribution to supported gain |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    fork_prose=[]
    for stratum in ['fork','nonfork']:
        d=row('dense',stratum); n=row('v2_on',stratum)
        fork_prose.append(f"All-position {stratum}: **{d['targets']:,}** targets, v2 minus dense **{n['nll']-d['nll']:+.9f}**; contribution to total change **{(n['total_nll']-d['total_nll'])/row('dense','total')['targets']:+.9f}** nats/byte.")
    for s in ['supported_fork','supported_nonfork']+[f'length_{label}/supported/{fork}' for label in ['1-3','4-12','13-64','65+'] for fork in ['fork','nonfork']]:
        d=row('dense',s); n=row('v2_on',s)
        lines.append(f"| {s} | {d['targets']:,} | {fmt(d['nll'])} | {fmt(n['nll'])} | {n['nll']-d['nll']:+.9f} | {(n['total_nll']-d['total_nll'])/denom:+.9f} |")
    lines += ['']+fork_prose+['']
    short=sum(decomp[k]['gain_share'] for k in ['1-3','4-12']); long=decomp['65+']['gain_share']
    nonfork_share=(row('v2_on','supported_nonfork')['total_nll']-row('dense','supported_nonfork')['total_nll'])/delta
    lines += ['',f"Medium suffixes (13–64 bytes) supply **{decomp['13-64']['gain_share']:.3%}** of the gain; supported singleton/nonfork positions supply **{nonfork_share:.3%}** across lengths. Thus neither 1–3-byte transfer nor rare 65+ matches dominate. The concentration in array-singleton medium suffixes is consistent with transferable continuation predictability; it does not identify semantic structure carried by the spine.",'',f"Lengths 1–12 account for **{short:.3%}** of the supported gain; 65+ account for **{long:.3%}**. This quantifies short versus long suffix transfer. Length alone cannot distinguish grammatical generalization, repeated boilerplate/whitespace, near-verbatim overlap, or learned structural computation. These are post-hoc descriptive strata, not an intervention or proof that the spine carries semantic structure. Contexts reset at each window and are at most 128 bytes here.",'',
        '## Array as a language model','',
        f"On supported targets, the unsmoothed array’s own NLL is **{row('array_alone','supported_target')['nll']:.9f}**, versus v2_on **{row('v2_on','supported_target')['nll']:.9f}** (v2 minus array **{row('v2_on','supported_target')['nll']-row('array_alone','supported_target')['nll']:+.9f}**). On nonempty Wikipedia support, **{row('array_alone','wikipedia_support')['infinite_losses']:,}** zero-target positions make array NLL infinite. Empty support abstains in fusion, but the array alone has no distribution there and is scored with q(target)=0; its full-window total is also infinite.",'',
        'This is a maximal-supported-suffix empirical branch baseline, analogous in purpose to an infini-gram baseline, not a backoff/smoothed implementation of that system. Its supported-target score is conditional on the answer being in support, not a fair standalone total-LM score or a mathematical ceiling for a trained model. No Pile extraction or held-out index is constructed.','',
        '## Validation and limits','',
        f"The inference pass replays dense/v2 on all {validation['final_test_targets']:,} final targets plus the full calibration half. All {validation['paired_keys_checked']} retained paired denominators match exactly, and means agree within 1e-8 nats/byte. V2 aggregate replay is exact; dense final total changes by +0.471769333 nats, +2.204e-9 nats/byte. Its largest stratum mean drift is retained in replay_difference.json. No root cause or byte-exact dense replay is claimed. Cache/source/window/checkpoint identities and first-shard eval parity follow the committed paired runner. Prediction and gate SHA256s, exact finite-part loss sums, infinite-loss counts, denominators, fitted gates, calibration freeze, inference resume, tests and independent validation are retained in `receipts/c14_stacking/`. The 17 GB position cache (`/home/erikg/c14_stacking_predictions.npy`) and 1.7 GB historical transformer loss join (`/home/erikg/c14_stacking_transformer_losses.npy`) stay outside git with hash receipts; all source inputs remain read-only.",'',
        'Residual risks: one seed and fixed time budget, unequal training exposures, no uncertainty estimate, float32 decoded branch probabilities and metadata, BF16 neural replay, historical transformer batching difference, and support-length selection cannot establish a causal semantic mechanism. Hard-walk failure is a real foreign-array mismatch, not discarded data. No resilient DiLoCo behavior was changed; its requirement IDs are inapplicable.','',
        'Manual notes: `contact_supervisor` was searched first and is absent from the active tool catalog. No supervisor approval or reply is implied. Work stayed in this attended session with the standalone cards-2/4 GPU broker; no WorksGood service, workers, subagents, Frontier or Slurm. Unrelated workspace changes were preserved.']
    # Historical reset/interior control details are deliberately sourced, never guessed.
    (ROOT/'RESULT_C14_STACKING.md').write_text('\n'.join(lines)+'\n')
    write('gain_decomposition.json',dict(sign='v2_on minus dense',supported_targets=denom,total_change_sum=delta,total_change_nll=delta/denom,bins=decomp,short_1_12_gain_share=short,long_65_plus_gain_share=long,supported_nonfork_gain_share=nonfork_share))


if __name__=='__main__':main()
