#!/usr/bin/env python3
"""Validate native records, exact source semantics, masks, coverage and split groups."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from scripts.audit_e97_open_swe_semantics import sha
from scripts.audit_e97_open_swe_admission_metadata import repo_identity
from scripts.e97_open_swe_native_codec import (PROFILE,compact,source_payload,semantic_message,
    decode_record,render,vocabulary,problem_split,strict_json)

INDEX=struct.Struct('<QQQB7x')


def validate(root,raw_root):
    root=Path(root);raw_root=Path(raw_root)
    manifest_sha=sha(root/'manifest.json');m=json.loads((root/'manifest.json').read_text())
    assert m['schema']=='emender-open-swe-source-native-candidate-v1' and m['profile']==PROFILE
    assert m['training_eligible'] is False
    for item in m['outputs'].values():
        p=root/item['path'];assert p.parent==root and not p.is_symlink()
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    for name,digest in m['builder_sources'].items():assert sha(root/name)==digest
    for key in ('source_cohort','parent_authority','protected_manifest'):
        assert sha(Path(m[key]['path']))==m[key]['sha256']
    cohort_list=json.loads(Path(m['source_cohort']['path']).read_text())
    cohort={r['identity']:r for r in cohort_list};assert len(cohort)==len(cohort_list)
    enc=tiktoken.get_encoding(m['tokenizer']);lengths,tokenizer_sha=vocabulary(enc)
    assert tokenizer_sha==m['tokenizer_vocabulary_sha256']
    reserved={tuple(k) for k in m['split_policy']['reserved_original_validation_problems']}
    counts=Counter();splits={'0':Counter(),'1':Counter()};groups={};sources={};positions={}
    with (root/'records.jsonl').open() as metadata,(root/'source_messages.jsonl').open() as archive, \
         (root/'tokens.bin').open('rb') as tokens,(root/'loss_mask.bin').open('rb') as masks, \
         (root/'records.idx').open('rb') as index:
        for line in metadata:
            record=json.loads(line);source=json.loads(next(archive));tid=record['trajectory_identity']
            assert tid not in sources and tid==source['trajectory_identity'] and tid in cohort
            payload=source['source'];digest=hashlib.sha256(compact(payload).encode()).hexdigest()
            assert digest==record['source_sha256']
            entry=index.read(INDEX.size);assert len(entry)==INDEX.size
            offset,n,targets,split=INDEX.unpack(entry)
            assert (offset,n,targets,split)==(record['offset'],record['tokens'],record['targets'],record['split'])
            assert offset==counts['tokens'] and 0<n<=65536 and 0<targets<=n and split in (0,1)
            assert record['record_index']==counts['records']
            data=tokens.read(n*4);mask_bytes=masks.read(n);assert len(data)==n*4 and len(mask_bytes)==n
            ids=np.frombuffer(data,dtype='<u4');mask=np.frombuffer(mask_bytes,dtype=np.uint8)
            assert np.all(mask<=1) and int(mask.sum())==targets and mask[0]==mask[-1]==0
            text=enc.decode_bytes(ids.tolist()).decode()
            assert hashlib.sha256(text.encode()).hexdigest()==record['sha256']
            assert enc.encode_ordinary(text)==ids.tolist()
            tools,messages,ranges=decode_record(text)
            assert tools==[strict_json(s) if isinstance(s,str) else s for s in payload['tools']]
            assert messages==[semantic_message(r) for r in payload['messages']]
            # Check semantic fields and target ranges independently of builder masks.
            bounds=np.r_[0,np.cumsum(lengths[ids])];expected=np.zeros(n,dtype=np.uint8)
            for start,end in ranges:
                lo=int(np.searchsorted(bounds,start));hi=int(np.searchsorted(bounds,end))
                assert bounds[lo]==start and bounds[hi]==end
                expected[lo:hi]=1
                assert ids[lo]==enc.encode_ordinary('Analysis')[0]
            assert np.array_equal(mask,expected)
            pieces,stats=render(payload,enc)
            assert ''.join(s for s,_ in pieces)==text
            for k,v in stats.items():assert record[k]==v
            key=(repo_identity(payload['repo']),payload['instance_id'])
            assert list(key)==record['problem_key'] and split==problem_split(*key,reserved)
            assert groups.setdefault(key,split)==split
            assert record['source_file']==cohort[tid]['source_file'] and record['source_row']==cohort[tid]['source_row']
            sources[tid]=digest;positions[tid]=(record['source_file'],record['source_row'])
            counts.update(records=1,tokens=n,targets=targets,**stats)
            splits[str(split)].update(records=1,tokens=n,targets=targets)
            if counts['records']%1000==0:print('NATIVE_VALIDATE_RECORDS '+str(counts['records']),flush=True)
        assert not archive.read() and not tokens.read(1) and not masks.read(1) and not index.read(1)
    included=set(sources);reasons=Counter()
    for line in (root/'exclusions.jsonl').read_text().splitlines():
        r=json.loads(line);tid=r['trajectory_identity'];assert tid not in sources and tid in cohort
        assert (r['source_file'],r['source_row'])==(cohort[tid]['source_file'],cohort[tid]['source_row'])
        sources[tid]=r['source_sha256'];positions[tid]=(r['source_file'],r['source_row']);reasons[r['reason']]+=1
    assert set(sources)==set(cohort)
    assert dict(counts)==m['counts'] and splits==m['split_counts'] and dict(reasons)==m['exclusion_counts']
    assert sha(raw_root/'README.md')==m['raw_card_sha256']
    raw_seen=set();raw_reserved=set()
    for file in m['raw_files']:
        candidates=list(raw_root.rglob(file['name']));assert len(candidates)==1
        path=candidates[0];assert not path.is_symlink() and path.stat().st_size==file['bytes'] and sha(path)==file['sha256']
        row_number=0
        columns=['trajectory_id','instance_id','repo','license','language','messages','tools','resolved']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8,columns=columns,use_threads=False):
            for row in batch.to_pylist():
                at=row_number;row_number+=1;tid='open-swe:'+str(row['trajectory_id'])
                if tid not in cohort:continue
                assert tid not in raw_seen and positions[tid]==(path.name,at)
                assert hashlib.sha256(compact(source_payload(row)).encode()).hexdigest()==sources[tid]
                key=(repo_identity(row['repo']),row['instance_id'])
                if int(cohort[tid]['split'])==1:raw_reserved.add(key)
                if tid in included:
                    assert row['resolved']==1 and key[0] not in m['protected_repositories']
                    assert row['license'] in {'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause'}
                raw_seen.add(tid)
        assert sha(path)==file['sha256']
    assert raw_seen==set(cohort) and raw_reserved==reserved
    for item in m['outputs'].values():assert sha(root/item['path'])==item['sha256']
    assert sha(root/'manifest.json')==manifest_sha
    return {'schema':'emender-open-swe-native-validation-v1','status':'passed','manifest_sha256':manifest_sha,
        'records':counts['records'],'tokens':counts['tokens'],'targets':counts['targets'],'excluded':sum(reasons.values()),
        'source_trajectories_verified':len(raw_seen),'complete_message_roundtrip':True,'exact_tool_argument_values':True,
        'all_masks_checked':True,'cross_split_problems':0,'unique_problems':len(groups),'training_eligible':False,
        'scope':'source reconstruction and serialization only; not execution replay, licensing admission, independent holdout, or model capability'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(validate(a.root,a.raw_root),sort_keys=True),flush=True)
