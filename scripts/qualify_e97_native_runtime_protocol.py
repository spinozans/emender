#!/usr/bin/env python3
"""Bounded real-data training/runtime framing qualification; executes no tools."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tiktoken
from scripts.audit_e97_open_swe_semantics import sha,publish
from scripts.e97_open_swe_native_codec import native_turn,semantic_message,compact
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode


def qualify(dataset,manifest_sha256,output,count=32):
    if type(count) is not int or count<=0:raise ValueError('invalid_sample_count')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,mode=0o700)
    if sha(dataset/'manifest.json')!=manifest_sha256:raise ValueError('dataset_identity')
    manifest=json.loads((dataset/'manifest.json').read_text())
    for name in ('records.jsonl','source_messages.jsonl','tokens.bin'):
        if sha(dataset/name)!=manifest['outputs'][name]['sha256']:raise ValueError('payload_identity')
    records=[json.loads(line) for line in (dataset/'records.jsonl').read_text().splitlines()]
    train=[r for r in records if r['split']==0]
    selected=sorted(train,key=lambda r:hashlib.sha256(('native-protocol-qualification-v1\0'+r['trajectory_identity']).encode()).digest())[:count]
    if len(selected)!=count:raise ValueError('insufficient_training_records')
    selected={r['trajectory_identity']:r for r in selected};enc=tiktoken.get_encoding('p50k_base')
    policy={'schema':'emender-native-protocol-panel-v1','dataset_manifest_sha256':manifest_sha256,
            'selection':'lowest SHA256(native-protocol-qualification-v1 NUL trajectory_identity), train split only',
            'identities':sorted(selected),'commands_executed':False,'model_loaded':False,'independent_holdout':False}
    publish(output/'panel.json',policy)
    reports=[]
    with (dataset/'source_messages.jsonl').open() as archive,(dataset/'tokens.bin').open('rb') as tokens:
        for line in archive:
            item=json.loads(line);tid=item['trajectory_identity']
            if tid not in selected:continue
            record=selected[tid];source=item['source']
            if hashlib.sha256(compact(source).encode()).hexdigest()!=record['source_sha256']:raise ValueError('source_identity')
            tokens.seek(record['offset']*4);data=tokens.read(record['tokens']*4)
            if len(data)!=record['tokens']*4:raise ValueError('short_record')
            all_ids=np.frombuffer(data,dtype='<u4').tolist();full=enc.decode_bytes(all_ids).decode()
            if hashlib.sha256(full.encode()).hexdigest()!=record['sha256']:raise ValueError('record_identity')
            session=NativeEpisode(source['tools'],enc);turns=0;prefix_tokens=0
            for message in source['messages']:
                if message['role']=='assistant':
                    prompt=session.prompt();body=native_turn(message);ids=enc.encode_ordinary(prompt)
                    if not full.startswith(prompt+body) or all_ids[:len(ids)]!=ids:raise ValueError('runtime_prefix_mismatch')
                    turn=session.accept_generated_turn(body);m=semantic_message(message)
                    if semantic_message(turn.source_message())!=m:raise ValueError('action_roundtrip')
                    expected=[]
                    if m['content'] is not None:expected.append({'kind':'commentary','text':m['content']})
                    if m['name']=='finish':expected.append({'kind':'final','text':m['arguments']['message']})
                    if turn.public_events()!=expected:raise ValueError('public_channel_mismatch')
                    expected_call={'name':m['name'],'arguments':m['arguments']} if m['name'] in ('execute_bash','str_replace_editor') else None
                    if turn.backend_call()!=expected_call:raise ValueError('backend_argument_mismatch')
                    turns+=1;prefix_tokens+=len(ids)
                else:session.append_source_message(message)
            if not session.finished or session.text()+'\x1e'!=full:raise ValueError('complete_history_mismatch')
            reports.append({'trajectory_identity':tid,'turns':turns,'prefix_tokens_checked':prefix_tokens,'status':'passed'})
            print('NATIVE_PROTOCOL_TRAJECTORY_PASSED '+str(len(reports)),flush=True)
    if {r['trajectory_identity'] for r in reports}!=set(selected):raise ValueError('coverage')
    for name in ('records.jsonl','source_messages.jsonl','tokens.bin'):
        if sha(dataset/name)!=manifest['outputs'][name]['sha256']:raise ValueError('payload_changed')
    if sha(dataset/'manifest.json')!=manifest_sha256:raise ValueError('manifest_changed')
    result={'schema':'emender-native-runtime-protocol-qualification-v1','status':'passed',
            'dataset_manifest_sha256':manifest_sha256,'panel_sha256':sha(output/'panel.json'),
            'trajectories':len(reports),'assistant_boundaries':sum(r['turns'] for r in reports),
            'prefix_tokens_checked':sum(r['prefix_tokens_checked'] for r in reports),'reports':reports,
            'training_eligible':False,'execution_qualified':False,'model_loaded':False,
            'scope':'training/runtime serialization and public/private routing only; no tool execution or model capability'}
    publish(output/'summary.json',result)
    print('NATIVE_PROTOCOL_QUALIFICATION_PASSED '+str(result['assistant_boundaries']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trajectories',type=int,default=32);a=p.parse_args()
    qualify(a.dataset,a.manifest_sha256,a.output,a.trajectories)
