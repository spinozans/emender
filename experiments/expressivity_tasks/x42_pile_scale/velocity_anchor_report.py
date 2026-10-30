"""Render honest pending/completed/failed 45-minute anchor rows."""
import argparse
import json
from pathlib import Path
from .velocity_state import atomic_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/velocity_baseline_anchors'

def length_report(output):
    from .velocity_length_sweep import SCOPE
    rows=[]
    for arm in ('transformer','gdn2','pure'):
        path=output/arm/'length_sweep.json'
        failure=output/(arm+'_failure.json')
        receipt=json.loads(path.read_text()) if path.exists() else {}
        rows.append(dict(arm=arm,status=receipt.get('status','failed' if failure.exists() else 'queued'),
                         source=str(path),measurements=receipt.get('rows',[])))
    complete=all(r['status']=='completed' for r in rows)
    atomic_json(output/'length_table.json',dict(status='completed' if complete else 'incomplete',
        scope=SCOPE,protocol='B64/T2048 seed3401, 2700 active train seconds; c06 optimizer; two reset windows per eval length',
        rows=rows,registered_score_unchanged=True))
    lines=['# Three-family LM length generalization','',SCOPE,'',
        '| family | status | T | NLL | perplexity | PPL / own T512 |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        if not row['measurements']:lines.append(f"| {row['arm']} | {row['status']} | pending | pending | pending | pending |")
        for m in row['measurements']:
            lines.append(f"| {row['arm']} | {row['status']} | {m['context']} | {m['nll']} | {m['perplexity']} | {m['ppl_ratio_vs_T512']} |")
    lines+=['','Incomplete/failed rows are not measurements. No three-family result is claimed until all arms complete.',
            'X40 window discipline: two reset contiguous batch-1 windows starting i*T; coverage differs by T.',
            'No resilient DiLoCo behavior changed; requirement IDs not applicable.']
    (output/'LENGTH_TABLE.md').write_text('\n'.join(lines)+'\n')


def report(output=OUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    length_report(output)
    reference=output/'pure/result.json'
    rows=[]
    for arm,path in [('pure emender c06 (fresh anchor)',reference),
                     ('transformer',output/'transformer/result.json'),
                     ('gdn2',output/'gdn2/result.json')]:
        failure=path.parent.parent/(path.parent.name+'_failure.json')
        if path.exists():
            r=json.loads(path.read_text())
            rows.append(dict(arm=arm,status=r['status'],score=r['score'],nll=r['nll_heldout'],
                tokens=r['tokens_consumed'],tokens_per_second=r['tokens_per_second'],
                train_seconds=r['train_seconds'],source=str(path),occ_frac=r.get('occ_frac'),
                degeneracy_flags=r.get('degeneracy_flags')))
        else:rows.append(dict(arm=arm,status='failed' if failure.exists() else 'queued',source=str(failure if failure.exists() else path)))
    atomic_json(output/'anchor.json',dict(rows=rows,protocol='B64/T2048, seed3401, contiguous pile24 train prefix, 2700 active train seconds; 24 reset held-out T2048 windows',
        score_asymmetry='Baselines score=NLL only; emender adds 0.01*occ_frac and degeneracy penalties. Compare NLL column.',
        capacity='Baselines match c06 150003984 trainable parameters within 0.1%; c06 total 227317104 includes frozen geometry. GPT2 vocabulary 50257.',
        fresh_emender='Fresh c06 policy rerun on card 0, including additional length sweep.'))
    lines=['# X42 velocity baseline anchors','',
        '45 minutes of active training per arm; evaluation and setup excluded. Whole updates may overshoot.',
        'Baselines match c06 trainable capacity (150,003,984); c06 also has 77,313,120 frozen parameters.',
        'Same GPT-2 vocabulary 50,257, fixed pile24 prefix, B64/T2048, seed 3401, c06 LR/warmup, 24 held-out reset windows.',
        'Baselines use part8 bf16 compute and fused vocabulary CE; emender retains certified FP32 store/loss.',
        'Baseline score = NLL; emender adds 0.01*occ_frac plus degeneracy penalties. Raw NLL is the direct comparison.',
        '', '| arm | status | score | raw NLL | tokens seen | tok/s |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+' | '.join(str(row.get(k,'pending')) for k in ('arm','status','score','nll','tokens','tokens_per_second'))+' |')
    lines+=['','Queued/failed rows are not measurements. Results are not yet evidence that 5.37 is strong.',
        'No resilient DiLoCo behavior is changed; resilient requirement IDs are not applicable.']
    (output/'ANCHOR.md').write_text('\n'.join(lines)+'\n')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));a=p.parse_args();report(a.output)
