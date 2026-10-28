"""Summarize immutable completed-v0 receipts without loading weights."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT/'receipts'
def read(name):return json.loads((R/name).read_text())
def heads(telemetry):
    return [[[dict((k,h[k]) for k in ('mean_live_per_head','peak_live_per_head','final_live_per_head','gate_mean','match_mean','error_mean','routing_per_head')) for h in b] for b in window] for window in telemetry]
def main():
    quality=[]
    for f in R.glob('quality_*.json'):
        q=json.loads(f.read_text());quality.append(dict(step=q['step'],tokens=q['tokens'],nll=q['validation']['nll'],perplexity=q['validation']['perplexity']))
    e=read('evaluation.json');curve=[json.loads(s) for s in (R/'curve.jsonl').read_text().splitlines()]
    assert len(curve)==3052 and curve[-1]['tokens']==100007936
    x39=ROOT.parent/'x39_memory_usage/receipts/part7_aborted_initialization/run/curve.jsonl'
    old=[json.loads(s) for s in x39.read_text().splitlines()]
    result=dict(status='completed-null',training=read('training.json'),curve=dict(rows=len(curve),first=curve[0],last=curve[-1]),
        quality=sorted(quality,key=lambda x:x['step']),final_telemetry=heads(read('quality_3052.json')['validation']['telemetry']),
        contexts=[dict(context=r['context'],perplexity=r['perplexity'],nll_bins128=r['nll_bins128']) for r in e['contexts']],
        long_context_ppl_ratio=e['long_context_ppl_ratio'],slopes=e['slopes_per_block_head'],depth=read('depth.json'),
        x39_closed_gate=dict(path=str(x39.relative_to(ROOT.parent)),rows=len(old),gate_means=sorted(set(r['gate_mean'] for r in old))),
        source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in R.iterdir() if f.name in ('curve.jsonl','training.json','evaluation.json','depth.json','quality_3052.json')})
    (R/'part2_v0_summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
