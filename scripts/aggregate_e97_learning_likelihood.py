#!/usr/bin/env python3
"""Fail-closed complete-shard aggregation for read-only likelihood probes."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from ndm.e97_atomic import publish_bytes_no_replace


def aggregate(root,probes_path,world):
    probes_bytes=probes_path.read_bytes(); probes=json.loads(probes_bytes)
    expected={e['id'] for e in probes['examples']}
    paths=sorted(root.glob('rank-*.json'))
    if len(paths)!=world: raise ValueError('incomplete shard set')
    reports=[json.loads(p.read_text()) for p in paths]
    shared=('schema','checkpoint_sha256','weight_mode','probes_sha256','probe_source_sha256','world_size')
    identity={k:reports[0][k] for k in shared}
    if identity['probes_sha256']!=hashlib.sha256(probes_bytes).hexdigest() or identity['world_size']!=world:
        raise ValueError('probe/world mismatch')
    if identity['schema']!='emender-e97-learning-likelihood-shard-v1': raise ValueError('invalid schema')
    results=[]
    for rank,report in enumerate(reports):
        if {k:report[k] for k in shared}!=identity or report['rank']!=rank or report['device']!=f'cuda:{rank}':
            raise ValueError('mixed identity or invalid rank/device routing')
        expected_rank=[e['id'] for e in probes['examples'][rank::world]]
        if [r['id'] for r in report['results']]!=expected_rank: raise ValueError('shard membership mismatch')
        results.extend(report['results'])
    if len(results)!=len(expected) or {r['id'] for r in results}!=expected: raise ValueError('duplicate/missing examples')
    groups=defaultdict(list)
    for row in results: groups[row['split']+'/'+row['source']].append(row)
    summary={}
    for key,rows in groups.items():
        n=len(rows); item={'examples':n,'full_prefix_serving_top1_matches':sum(r['full_prefix_serving_top1_equal'] for r in rows),
                          'mean_marker_logprob':{m:sum(r['markers'][m]['logprob_sum'] for r in rows)/n for m in probes['marker_candidates']}}
        supervised=[r for r in rows if r['target_tokens']]
        if supervised:
            denominator=sum(r['target_tokens'] for r in supervised)
            item['serving_target_token_weighted_nll']=sum(r['serving_target_nll_mean']*r['target_tokens'] for r in supervised)/denominator
            item['boundary_forward_target_token_weighted_nll']=sum(r['boundary_forward_target_nll_mean']*r['target_tokens'] for r in supervised)/denominator
            item['first_target_top1_matches']=sum(r['serving_first_target_rank']==1 for r in supervised)
        if not all(math.isfinite(v) for v in item['mean_marker_logprob'].values()): raise ValueError('nonfinite aggregate')
        summary[key]=item
    return {'schema':'emender-e97-learning-likelihood-summary-v1','training_eligible':False,
            'identity':identity,'examples':len(results),'by_source':summary,'results':results,
            'shards':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            'scope':'descriptive likelihood/parity diagnostic, not a capability pass or training authorization'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--probes',type=Path,required=True); p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    result=aggregate(args.input_root,args.probes,args.world_size)
    publish_bytes_no_replace(args.output,(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print(json.dumps({'output':str(args.output),'by_source':result['by_source']}),flush=True)

if __name__=='__main__': main()
