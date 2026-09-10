#!/usr/bin/env python3
"""Rebuild a source-native, full-trajectory candidate without changing old data."""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct

import pyarrow.parquet as pq
import tiktoken
from scripts.audit_e97_open_swe_semantics import publish,sha
from scripts.audit_e97_open_swe_admission_metadata import repo_identity
from scripts.e97_open_swe_native_codec import (PROFILE,compact,source_payload,render,encode,vocabulary,problem_split)

INDEX=struct.Struct('<QQQB7x')
SOURCE_FIELDS=['trajectory_id','instance_id','repo','license','language','messages','tools']
LICENSES={'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause'}


def rename_no_replace(stage,destination):
    fn=ctypes.CDLL(None,use_errno=True).renameat2
    fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(stage),-100,os.fsencode(destination),1):
        error=ctypes.get_errno();raise OSError(error,os.strerror(error),str(destination))
    fd=os.open(Path(destination).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def load_inputs(args):
    for path,digest in [(args.authority/'manifest.json',args.manifest_sha256),(args.cohort,args.cohort_sha256),
                        (args.protected_manifest,args.protected_sha256)]:
        if sha(path)!=digest:raise ValueError('input identity mismatch: '+str(path))
    manifest=json.loads((args.authority/'manifest.json').read_text())
    if manifest['training_eligible'] is not False:raise ValueError('expected original nonadmitted candidate')
    items=json.loads(args.cohort.read_text());cohort={x['identity']:x for x in items}
    if len(cohort)!=len(items) or len(cohort)!=args.expected_trajectories:raise ValueError('cohort coverage')
    card=args.raw_root/'README.md'
    if sha(card)!=manifest['dataset_card_sha256']:raise ValueError('card mismatch')
    files=[]
    for entry in sorted(manifest['input_files'],key=lambda d:d['name']):
        found=list(args.raw_root.rglob(entry['name']))
        if len(found)!=1 or found[0].is_symlink():raise ValueError('ambiguous shard')
        path=found[0]
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('shard mismatch')
        files.append((path,entry))
    protected=json.loads(args.protected_manifest.read_text())
    excluded={repo_identity(v['url']) for v in protected['repositories'].values()}
    if None in excluded:raise ValueError('invalid protected repository identity')
    return manifest,cohort,files,excluded


def iter_selected(files,cohort,columns):
    seen=set()
    for path,_ in files:
        index=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8,columns=columns,use_threads=False):
            for row in batch.to_pylist():
                at=index;index+=1;tid='open-swe:'+str(row['trajectory_id'])
                if tid not in cohort:continue
                expected=cohort[tid]
                if tid in seen or (path.name,at)!=(expected['source_file'],expected['source_row']):raise ValueError('row identity mismatch')
                seen.add(tid);yield tid,row,expected
        print('NATIVE_BUILD_SHARD '+path.name,flush=True)
    if seen!=set(cohort):raise ValueError('missing source trajectories')


def build(args):
    out=args.output;stage=out.with_name(out.name+'.partial')
    if out.exists() or stage.exists():raise FileExistsError('refuse to overwrite final or failed stage')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage.mkdir(mode=0o700)
    original,cohort,files,protected=load_inputs(args)
    reserved=set();keys={}
    for tid,row,old in iter_selected(files,cohort,['trajectory_id','instance_id','repo']):
        repo=repo_identity(row['repo']);instance=row['instance_id']
        if repo is None or not isinstance(instance,str) or not instance:raise ValueError('invalid problem identity')
        keys[tid]=(repo,instance)
        if int(old['split'])==1:reserved.add(keys[tid])
    enc=tiktoken.get_encoding('p50k_base');lengths,tokenizer_sha=vocabulary(enc)
    names=['tokens.bin','loss_mask.bin','records.idx','records.jsonl','source_messages.jsonl','exclusions.jsonl']
    handles={n:(stage/n).open('xb') for n in names}
    totals=Counter();splits={'0':Counter(),'1':Counter()};reasons=Counter();offset=0
    try:
        for tid,row,old in iter_selected(files,cohort,SOURCE_FIELDS+['resolved']):
            payload=source_payload(row);source_sha=hashlib.sha256(compact(payload).encode()).hexdigest()
            common={'trajectory_identity':tid,'source_file':old['source_file'],'source_row':old['source_row'],
                    'source_sha256':source_sha,'problem_key':list(keys[tid])}
            try:
                if row['resolved']!=1:raise ValueError('source_not_reported_resolved')
                if row['license'] not in LICENSES:raise ValueError('declared_license_not_allowed')
                if keys[tid][0] in protected:raise ValueError('protected_repository')
                pieces,counts=render(row,enc)
                ids,mask,text=encode(pieces,enc,lengths)
            except ValueError as error:
                reason=str(error);reasons[reason]+=1
                handles['exclusions.jsonl'].write((compact({**common,'reason':reason})+'\n').encode())
                continue
            split=problem_split(*keys[tid],reserved);n=len(ids);target=int(mask.sum())
            metadata={**common,'record_index':totals['records'],'offset':offset,'tokens':n,'targets':target,
                      'split':split,'sha256':hashlib.sha256(text.encode()).hexdigest(),**counts}
            handles['tokens.bin'].write(ids.tobytes());handles['loss_mask.bin'].write(mask.tobytes())
            handles['records.idx'].write(INDEX.pack(offset,n,target,split))
            handles['records.jsonl'].write((compact(metadata)+'\n').encode())
            handles['source_messages.jsonl'].write((compact({'trajectory_identity':tid,'source':payload})+'\n').encode())
            totals.update(records=1,tokens=n,targets=target,**counts)
            splits[str(split)].update(records=1,tokens=n,targets=target)
            offset+=n
            if totals['records']%500==0:print('NATIVE_BUILD_PROGRESS '+compact(dict(totals)),flush=True)
    finally:
        for handle in handles.values():handle.flush();os.fsync(handle.fileno());handle.close()
    if totals['records']+sum(reasons.values())!=len(cohort):raise ValueError('coverage reconciliation')
    sources={}
    for name in ('build_e97_open_swe_native_sft.py','e97_open_swe_native_codec.py','validate_e97_open_swe_native_sft.py'):
        content=Path(__file__).with_name(name).read_bytes();p=stage/name;p.write_bytes(content)
        sources[name]=hashlib.sha256(content).hexdigest()
    manifest={'schema':'emender-open-swe-source-native-candidate-v1','profile':PROFILE,'training_eligible':False,
        'admission_blockers':['source-native runtime execution qualification','complete licensing and protected-overlap admission','training recipe and loss/optimizer integration'],
        'dataset_id':original.get('dataset_id','nvidia/Open-SWE-Traces'),'tokenizer':'p50k_base',
        'tokenizer_vocabulary_sha256':tokenizer_sha,'tiktoken_version':tiktoken.__version__,
        'maximum_record_tokens':65536,'maximum_private_analysis_tokens':2048,'maximum_private_analysis_bytes':65536,
        'record_policy':'one complete source trajectory per recurrent record; no truncation or window resets',
        'split_policy':{'key':'normalized repository + original instance_id','hash_namespace':'e97-native-problem-split-v1',
                        'hash_validation_percent':5,'reserved_original_validation_problems':[list(k) for k in sorted(reserved)],
                        'independent_holdout':False,'prior_lineage_exposure':'existing previously used Open-SWE candidate family; regrouping does not remove prior exposure'},
        'channel_policy':'source reasoning_content and think thoughts private; source content and finish.message public; original think flag retained',
        'transport_omissions':'tool-call transport IDs and raw argument JSON formatting are in source_messages.jsonl, not generated targets; decoded argument values are exact',
        'terminal_separator_supervised':False,'counts':dict(totals),'split_counts':splits,'exclusion_counts':dict(reasons),
        'exclusion_policy':'whole trajectory; first failing reason; not overlapping reason counts',
        'source_cohort':{'path':str(args.cohort),'sha256':args.cohort_sha256,'trajectories':len(cohort)},
        'parent_authority':{'path':str(args.authority/'manifest.json'),'sha256':args.manifest_sha256},
        'protected_manifest':{'path':str(args.protected_manifest),'sha256':args.protected_sha256},
        'protected_repositories':sorted(protected),'raw_card_sha256':original['dataset_card_sha256'],
        'raw_files':[entry for _,entry in files],'builder_sources':sources,
        'outputs':{n:{'path':n,'bytes':(stage/n).stat().st_size,'sha256':sha(stage/n)} for n in names}}
    publish(stage/'manifest.json',manifest)
    from scripts.validate_e97_open_swe_native_sft import validate
    verdict=validate(stage,args.raw_root)
    publish(stage/'validation.json',verdict)
    # Retain old inputs and recheck all bound source bytes before publication.
    load_inputs(args)
    for p in stage.iterdir():
        if not p.is_file() or p.is_symlink():raise ValueError('unexpected staged entry')
        with p.open('rb') as f:os.fsync(f.fileno())
        p.chmod(0o400)
    fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    stage.chmod(0o500);rename_no_replace(stage,out)
    print('NATIVE_REBUILD_COMPLETE '+compact({'output':str(out),'counts':dict(totals),'splits':splits,'exclusions':dict(reasons),
        'manifest_sha256':sha(out/'manifest.json'),'training_eligible':False}),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    for name in ('authority','raw-root','cohort','protected-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','cohort-sha256','protected-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--expected-trajectories',type=int,default=10905)
    build(p.parse_args())


if __name__=='__main__':main()
