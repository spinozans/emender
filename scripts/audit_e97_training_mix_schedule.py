#!/usr/bin/env python3
"""Materialize a frozen smoke schedule on CPU and account for each source."""
import argparse
from collections import Counter
import json
from pathlib import Path
from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset,SFTSamplerIdentity,sha256
from ndm.e97_atomic import publish_bytes_no_replace


def audit(args):
    manifest=json.loads((args.authority/'manifest.json').read_text())
    meta=manifest['outputs']['metadata'];path=args.authority/Path(meta['path']).name
    if sha256(path)!=meta['sha256']:raise ValueError('metadata identity')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    identity=SFTSamplerIdentity(authority_manifest_sha256=args.authority_sha256,
        pack_manifest_sha256=args.pack_sha256,sampler_key=args.sampler_key,
        data_world_size=args.world_size,context_size=args.context_size)
    steps=[dict(update=i+1,rank_sample_ids=[None]*args.world_size,global_tokens=0,global_targets=0,
                source_targets=Counter(),source_tokens=Counter()) for i in range(args.steps)]
    occurrences=Counter()
    for rank in range(args.world_size):
        data=MaskedSFTPackedDataset(args.authority,args.packs,identity=identity,rank=rank,sampler_mode='epoch-permutation')
        try:
            for cursor,step in enumerate(steps):
                pack=data.packs[data.pack_id_at(cursor)];start=int(pack['record_offset'])
                record_ids=data.pack_record_ids[start:start+int(pack['record_count'])]
                tokens,loss,valid,reset,lengths,counts=data.get_boundary_aware_batch(1)
                if int(loss.sum())!=int(counts.sum()) or (loss&reset[:,1:]).any() or (loss&~valid[:,1:]).any():
                    raise ValueError('materialized mask accounting')
                step['rank_sample_ids'][rank]=list(data.last_batch_sample_ids)
                step['global_tokens']+=int(lengths.sum());step['global_targets']+=int(counts.sum())
                for i in record_ids:
                    row=rows[int(i)];name=row['source'];occurrences[int(i)]+=1
                    step['source_targets'][name]+=row['targets'];step['source_tokens'][name]+=row['tokens']
        finally:data.close()
    totals=Counter()
    for step in steps:
        if sum(step['source_targets'].values())!=step['global_targets'] or sum(step['source_tokens'].values())!=step['global_tokens']:
            raise ValueError('per-source totals do not match actual batch')
        totals.update(step['source_targets'])
    report=dict(schema='emender-e97-training-mix-schedule-v1',status='passed',
        authority_manifest_sha256=args.authority_sha256,pack_manifest_sha256=args.pack_sha256,
        sampler_key=args.sampler_key,world_size=args.world_size,context_size=args.context_size,
        steps=steps,source_target_totals=dict(totals),
        actual_source_target_fractions={k:v/sum(totals.values()) for k,v in totals.items()},
        unique_records=len(occurrences),record_occurrences=sum(occurrences.values()),
        scope='precomputed and CPU-materialized schedule; runtime must match sample IDs and counts')
    publish_bytes_no_replace(args.output,(json.dumps(report,indent=2,sort_keys=True)+'\n').encode(),mode=0o400)


def main():
    p=argparse.ArgumentParser()
    for name in ('authority','packs','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('authority-sha256','pack-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--sampler-key',type=int,default=974117);p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--context-size',type=int,default=65536);p.add_argument('--steps',type=int,default=8)
    a=p.parse_args()
    if min(a.world_size,a.context_size,a.steps)<=0:raise ValueError('positive dimensions required')
    audit(a)


if __name__=='__main__':main()
