"""Independently check paired aggregate partitions and produce honest P1/P2."""
import json
from pathlib import Path
import time
from .c14_paired_eval import ROOT, OUT, CACHE, EXPECTED, WINDOWS, JOURNAL, rows, write
from .c14_documents import sha


def main():
    state=json.loads((OUT/'resume.json').read_text())
    table=json.loads((OUT/'table.json').read_text())
    assert state['status']=='complete'
    assert (state['windows'],state['targets'])==EXPECTED[:2]
    assert state['config']['windows_sha256']==sha(Path('/home/erikg/c14_dense_bytes/eval_windows.jsonl'))==WINDOWS
    assert state['config']['journal_sha256']==sha(CACHE/'journal.jsonl')==JOURNAL
    assert state['config']['checkpoint_hashes']==dict(
        dense='7a2598c8c1e77f349be6be7af1d59ece7300a5345cb3dab1420e0274d2523019',
        v1='5499d02420774829fa23ee17b232f06789d292ab815f12ae039e671793485261',
        v2='6b12a443249f0464808ba3d52274d373902ad43ea40913ce347cdf1c1af9811b')
    for path,receipt in json.loads((OUT/'frozen_input_hashes.json').read_text()).items():
        assert sha(ROOT.parents[2]/path)==receipt['sha256'],path
    sums=state['sums']; modes=state['config']['modes']
    def cell(arm,half,stratum): return table['/'.join((arm,half,stratum))]
    expected={}
    with Path('/home/erikg/c14_dense_bytes/eval_windows.jsonl').open() as f:
        for line in f:
            d=json.loads(line); h=d['half']; n=d['targets']
            for half in ('all',h):
                for name,count in [('total',n),('window_reset_1_16',min(n,16)),('window_interior_17_128',max(0,n-16))]:
                    key=half+'/'+name; expected[key]=expected.get(key,0)+count
    checks=0
    for key,(s,n) in sums.items():
        assert table[key]==dict(total_nll=s,targets=n,nll=s/n if n else None)
        assert n>=0 and s>=0
    for arm in modes:
        for half in ('all','calibration','final_test'):
            assert cell(arm,half,'agreement')==cell(arm,half,'supported_target')
            assert cell(arm,half,'argmax_agreement')['targets']<=cell(arm,half,'supported_target')['targets']
            assert cell(arm,half,'fork')['targets']<=cell(arm,half,'wikipedia_support')['targets']
            for name in ('total','window_reset_1_16','window_interior_17_128'):
                assert cell(arm,half,name)['targets']==expected[half+'/'+name]
            for a,b,c in [('window_reset_1_16','window_interior_17_128','total'),
                          ('wikipedia_support','empty_support','total'),
                          ('supported_target','nonempty_zero_target','wikipedia_support'),
                          ('empty_support','nonempty_zero_target','zero_target'),
                          ('supported_target','zero_target','total')]:
                x,y,z=[cell(arm,half,k) for k in (a,b,c)]
                assert x['targets']+y['targets']==z['targets']
                assert abs(x['total_nll']+y['total_nll']-z['total_nll'])<1e-4
                checks+=1
            for name in [k.split('/')[-1] for k in table if k.startswith('dense/'+half+'/')]:
                assert cell(arm,half,name)['targets']==cell('dense',half,name)['targets']
                allc,cal,final=[cell(arm,h,name) for h in ('all','calibration','final_test')]
                assert allc['targets']==cal['targets']+final['targets']
                assert abs(allc['total_nll']-cal['total_nll']-final['total_nll'])<1e-4
    covered=sum(r['covered'] for r in rows(CACHE/'journal.jsonl') if r['event']=='shard_complete')
    assert cell('dense','all','wikipedia_support')['targets']==covered
    for label,h in state['config']['checkpoint_hashes'].items():
        if label=='dense': p=Path('/home/erikg/c14_dense_bytes/emender_dense.pt')
        else:
            folder='c14_consultation_emender'+('_v2' if label=='v2' else '')
            p=Path(json.loads((ROOT/'receipts'/folder/'training.json').read_text())['checkpoint'])
        assert sha(p)==h
    historical=json.loads((ROOT/'receipts/c14_dense/training.json').read_text())['models'][0]
    dense_replay={}
    for half in ('calibration','final_test'):
        c=cell('dense',half,'total'); old=historical['heldout'][half]
        assert c['targets']==old['targets']
        delta=c['nll']-old['nll']
        assert abs(delta)<.002,delta
        dense_replay[half]=dict(recomputed=c['nll'],historical=old['nll'],difference=delta,
            note='Same frozen weights/targets, batch up to128 vs historical16; no bit-exact loss replay claim')
    verdicts={}
    for label in ('v1','v2'):
        gain={name:cell('dense','final_test',name)['nll']-cell(label+'_on','final_test',name)['nll']
              for name in ('total','wikipedia_support','empty_support','supported_target',
                           'nonempty_zero_target','zero_target','agreement','fork')}
        clauses=dict(total_gain_positive=gain['total']>0,
            wikipedia_support_gain_exceeds_zero_target=gain['wikipedia_support']>gain['zero_target'],
            supported_target_gain_exceeds_zero_target=gain['supported_target']>gain['zero_target'],
            agreement_gain_exceeds_zero_target=gain['agreement']>gain['zero_target'])
        difference=cell(label+'_off','final_test','total')['nll']-cell('dense','final_test','total')['nll']
        verdicts[label]=dict(primary=label=='v2',gain_dense_minus_device_on=gain,
            P1_clauses=clauses,P1='PASS' if all(clauses.values()) else 'FAIL',
            coverage_support_minus_empty_gain=gain['wikipedia_support']-gain['empty_support'],
            supported_target_minus_nonempty_zero_target_gain=gain['supported_target']-gain['nonempty_zero_target'],
            P2_device_off_minus_dense=difference,P2_absolute_difference=abs(difference),
            P2='PASS' if abs(difference)<=.02 else 'FAIL')
    write('verdicts.json',dict(final_test_only=True,positive_gain='dense NLL minus device-on NLL',
          P1_rule='directional total gain AND larger gains on nonempty Wikipedia support and supported-target/branch-agreement than all zero-target positions; separate support/empty and nonempty-zero-target comparisons retained',
          P2_rule='absolute device-off minus dense final-test NLL <=.02 nats/byte',arms=verdicts))
    write('validation.json',dict(status='passed',partition_sum_checks=checks,
        expected_window_targets=expected,coverage_from_cache_journal=covered,
        all_modes_identical_denominators=True,checkpoint_hashes_unchanged=True,
        frozen_identity_verified=True,dense_replay=dense_replay,
        table_sha256=sha(OUT/'table.json'),runner_sha256=sha(ROOT/'c14_paired_eval.py'),
        scope='Independent frozen-window denominators, all aggregate partition identities, five modes, cache coverage and checkpoint hashes; no significance test or native query rerun'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve=[r for r in state['curve'] if r['final_nll']['dense'] is not None]
    x=[(r['targets']-expected['calibration/total'])/1e6 for r in curve]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for arm in modes:
        axes[0].plot(x,[r['final_nll'][arm] for r in curve],label=arm)
    for label in ('v1','v2'):
        axes[1].plot(x,[r['final_nll']['dense']-r['final_nll'][label+'_on'] for r in curve],label=label)
        axes[2].plot(x,[r['final_nll'][label+'_off']-r['final_nll']['dense'] for r in curve],label=label)
    axes[0].set_ylabel('Cumulative NLL (nats/byte)')
    axes[1].set_ylabel('Dense minus device-on (positive = gain)')
    axes[2].set_ylabel('Device-off minus dense')
    for value in (-.02,.02): axes[2].axhline(value,color='grey',linestyle='--',linewidth=.8)
    for ax in axes:
        ax.set_xlabel('Final-test targets evaluated (millions)'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('C14 Emender: frozen final-test cumulative paired evaluation')
    fig.savefig(OUT/'curves.svg',metadata={'Date':None,'Creator':'Emender C14 paired eval'})
    plt.close(fig)
    svg=OUT/'curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    lines=['# C14 Emender paired evaluation at Pile scale','',
       'V2 is primary; v1 is retained. NLL averages are natural-log nats/byte; total NLL sums are nats. Weights are frozen: no retraining. '
       'The primary verdict uses the preregistered document-disjoint final-test half. Calibration and combined rows are reported separately.','',
       '; '.join(f"{label}: P1 **{v['P1']}**, P2 **{v['P2']}**" for label,v in verdicts.items())+'.','',
       f"Frozen cache: {EXPECTED[0]:,} windows / {EXPECTED[1]:,} targets, {EXPECTED[2]:,} bytes; "
       f"windows SHA256 `{WINDOWS}`, journal SHA256 `{JOURNAL}`.", '',
       'All shard hashes, contiguity and recorded extraction parity attestations pass. Every eval position passes '
       'the c14_verify_sum compensated/sequential float32 entropy oracle. This reuses extraction attestations; '
       'no fresh native extractor query rerun is claimed. Feature assembly uses the shared v2 decoder.','']
    names=['total','window_reset_1_16','window_interior_17_128','wikipedia_support','empty_support',
            'supported_target','nonempty_zero_target','zero_target','agreement','argmax_agreement','fork']
    labels=['Total','Reset 1–16','Interior 17–128','Wikipedia support','Empty support',
            'Supported target','Nonempty zero-target','All zero-target','Branch agreement','Argmax agreement (supplemental)','Fork']
    for half in ('final_test','calibration','all'):
        lines += [f'## {half}', '', '| Arm/mode | Total NLL sum | '+' | '.join(labels)+' |',
                  '| --- | ---: | '+' | '.join(['---:']*len(names))+' |']
        for arm in modes:
            values=[f"{cell(arm,half,name)['nll']:.9f}" if cell(arm,half,name)['targets'] else 'n/a' for name in names]
            lines += ['| '+arm+' | '+f"{cell(arm,half,'total')['total_nll']:.6f}"+' | '+' | '.join(values)+' |']
        lines += ['', '| Stratum | Targets (identical in all five modes) |','| --- | ---: |']
        for name,label in zip(names,labels): lines += [f"| {label} | {cell('dense',half,name)['targets']:,} |"]
        lines += ['']
    lines += ['## Preregistered predictions: all signs retained','',
       'Positive gain means dense minus consultation device-on NLL. Wikipedia support means nonempty '
       'nonterminal outgoing counts; supported target means q(target)>0. All zero-target includes empty '
       'and nonempty zero-target. Registered branch agreement means positive target count '
       '(c14_coverage.py branch_agreement_supported), identical to supported-target here; '
       'argmax agreement is supplemental. These overlapping diagnostics never filter the total loss.','',
       '| Arm | Total gain | Supported-target gain | Zero-target gain | Agreement gain | Coverage gain (support − empty) | Supported-target − nonempty-zero-target gain | P1 | Off − dense | P2 |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    for label,v in verdicts.items():
        g=v['gain_dense_minus_device_on']
        values=[g['total'],g['supported_target'],g['zero_target'],g['agreement'],v['coverage_support_minus_empty_gain'],v['supported_target_minus_nonempty_zero_target_gain']]
        lines += ['| '+label+' | '+' | '.join(f'{x:+.9f}' for x in values)+f" | {v['P1']} | {v['P2_device_off_minus_dense']:+.9f} | {v['P2']} |"]
    lines += ['', '| Final-test stratum | v1 gain | v2 gain |','| --- | ---: | ---: |']
    for name in ('total','wikipedia_support','empty_support','supported_target','nonempty_zero_target','zero_target','agreement','fork'):
        lines += [f"| {name} | {verdicts['v1']['gain_dense_minus_device_on'][name]:+.9f} | {verdicts['v2']['gain_dense_minus_device_on'][name]:+.9f} |"]
    lines += ['', 'P1 requires lower total final-test NLL and larger nonempty-support, supported-target and branch-agreement gains than '
         'all zero-target gains; the separate support/empty and nonempty-zero-target comparisons above retain '
         'the coverage interpretation. P2 requires absolute device-off minus dense NLL at most .02. '
         'Device-off zeros every one of the 260 features at every position, with evaluation dropout disabled.','',
         '## Matched active training budget and unequal exposure','',
         '| Arm | Steps | Training targets | Active seconds |','| --- | ---: | ---: | ---: |']
    for arm in ('dense','v1','v2'):
        r=state['config']['training'][arm]
        lines += [f"| {arm} | {r['steps']:,} | {r.get('targets',r.get('actual_bytes_targets')):,} | {r['active_seconds']:.6f} |"]
    lines += ['', 'All three checkpoints use the same Emender architecture with 2,133,700 trainable parameters, seed 3401, '
        'training batch 16/context 128, BF16 and AdamW 3e-4/decay .01/clip 1. '
        'Consultation training uses independent keep=.5 jointly over all 260 scalars without rescaling. '
        'Evaluation uses BF16 autocast, float32 per-target losses and float64 sums. '
        'Each source document is digest-checked against frozen offsets; every cached window identity and '
        'length is matched to the frozen window file. Windows reset independently and tail padding is masked. '
        'Inference batches contain at most 128 windows versus historical dense evaluation batch 16. '
        'The independent validation receipt retains the small dense replay difference; no byte-exact historical loss claim.', '',
        f"Evaluation process inference-loop time: {state['elapsed_seconds']:.3f}s; includes cache reads, decode, source checks, all five modes and aggregate receipts; "
        'excludes imports, model loading, offsets scan, preflight full cache audit, compiler setup and broker lease wait. '
        'No extraction or training cost is hidden in this measurement. GPU broker card '+state['config']['gpu']+'.','',
        '## Receipts and limits','',
        '`receipts/c14_paired_eval/` retains exact sums/denominators (`table.json`), running cumulative '
        'NLL curves and exact resume state (`resume.json`, `curves.svg`), cache identity/oracle audit, checkpoint and '
        'training-exposure hashes, verdicts and independent validation. `frozen_input_hashes.json` binds '
        '19 original committed protocols, training curves/configs, the v2 primary amendment and decoder proofs. '
        '`transformer_progress.json` is a '
        'read-only live-service snapshot with conditional projections; the service was not modified.','',
        'One seed and one fixed time budget; different training exposure is measured, not controlled away. '
        'No confidence intervals, matched-step causal claim, P3, coverage swap, calibration fit or transformer '
        'quality verdict. Early exploratory inference had a nonfinite loss and nondeterministic loss bits; '
        'failed logs and repeat diagnostics remain in MANUAL_NOTES.md. The adopted run requires '
        'deterministic algorithms and a fixed cuBLAS workspace, checks first-shard loss-bit parity '
        'against normal eval, and fails closed on every nonfinite loss. These checks do not prove '
        'exhaustive native-kernel determinism. No resilient DiLoCo work was performed; its gap-matrix IDs do not apply. '
        'No .pt files are committed. The enabled tool catalog has no contact_supervisor; no supervisor reply is implied.']
    (ROOT/'RESULT_C14_PAIRED_EVAL.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verdicts,indent=2))


if __name__=='__main__': main()
