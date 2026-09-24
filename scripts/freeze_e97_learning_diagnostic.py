#!/usr/bin/env python3
"""Freeze read-only likelihood probes from exactly consumed u64 packs.

Not a training authority or an independent holdout. No model evaluation occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import tiktoken
from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity, sha256
from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_agent_protocol import serialize_pi_messages

AUTH=Path('/mnt/nvme2n1/erikg/sft/e97-4b-systematic-representation-stage-50m-v1/private-analysis')
PACK=AUTH/'packs-65536-boundary-epoch-v1'
SCREEN=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/private-analysis-lr5e5-screen-v1')
TRAINING=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/training_source_identity_systematic_v1/worktree')


def first_target(tokens,mask,*,max_prefix=4096,max_target=64):
    starts=np.flatnonzero(mask)
    if len(starts)==0: return None
    start=int(starts[0])
    if not 1<=start<=max_prefix: return None
    stop=start
    while stop<len(mask) and mask[stop] and stop-start<max_target: stop+=1
    return tokens[:start].tolist(),tokens[start:stop].tolist(),start


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    receipt_path=SCREEN/'runs/e97-private-lr5e5-v1-u64/terminal/checkpoint.reload.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['checkpoint_sha256']=='12b2140ff4d299c219af0981ab978ccf2c3135984567941ca244fb57544521b1'
    assert receipt['sft_updates']==64 and receipt['world_size']==8 and receipt['sampler_mode']=='epoch-permutation'
    module=Path(__file__).resolve().parents[1]/'ndm/data/masked_sft_dataset.py'
    assert sha256(module)==sha256(TRAINING/'ndm/data/masked_sft_dataset.py'), 'sampler source differs from executed training snapshot'
    identity=SFTSamplerIdentity(receipt['authority_manifest_sha256'],receipt['pack_manifest_sha256'],974121,8,65536)
    data=MaskedSFTPackedDataset(AUTH,PACK,identity=identity,rank=0,sampler_mode='epoch-permutation')
    metadata_info=data.authority_manifest['outputs']['metadata']
    metadata_path=AUTH/metadata_info['path']
    assert metadata_path.parent==AUTH and sha256(metadata_path)==metadata_info['sha256']
    rows=[json.loads(line) for line in metadata_path.open()]
    encoding=tiktoken.get_encoding('p50k_base')
    quotas={'agent':12,'conversation':6,'core-retention':6}
    selected=Counter(); source_targets=Counter(); source_inputs=Counter(); occurrences=Counter()
    seen=set(); unique_records=set(); unique_agent_trajectories=set(); examples=[]; consumed=[]
    tokens_total=targets_total=0
    for update in range(64):
        for rank in range(8):
            # Only rank changes; epoch permutation is shared across ranks.
            data.rank=rank
            pack_id=data.pack_id_at(update); pack=data.packs[pack_id]
            count=int(pack['record_count']); offset=int(pack['record_offset'])
            record_ids=data.pack_record_ids[offset:offset+count]
            tokens_total+=int(pack['tokens']); targets_total+=int(pack['targets'])
            consumed.append({'update':update+1,'rank':rank,'pack_id':pack_id,'sample_id':data.sample_id(update)})
            for raw_id in record_ids:
                record_id=int(raw_id); row=rows[record_id]; rec=data.records[record_id]
                source=row['source']; length=int(rec['tokens']); base=int(rec['offset'])
                assert int(rec['split'])==0 and length==row['tokens'] and int(rec['targets'])==row['targets']
                mask=data.masks[base:base+length]; ids=data.tokens[base:base+length]
                target_count=int(mask[1:].sum())
                source_targets[source]+=target_count; source_inputs[source]+=length; occurrences[source]+=1
                unique_records.add(record_id)
                if source=='agent': unique_agent_trajectories.add(row['trajectory_identity'])
                if selected[source]>=quotas.get(source,0): continue
                uniqueness=row.get('trajectory_identity') or row['identity_sha256']
                if uniqueness in seen: continue
                item=first_target(ids,mask)
                if item is None: continue
                prefix,target,start=item
                prefix_text=encoding.decode(prefix); target_text=encoding.decode(target)
                assert encoding.encode_ordinary(prefix_text)==prefix, 'prefix tokenization mismatch'
                if source=='agent':
                    assert prefix_text.endswith('Assistant:\n') and target_text.startswith('Analysis: ')
                examples.append({'id':f'consumed-{source}-{record_id:08d}','split':'consumed-training',
                    'source':source,'record_id':record_id,'record_identity':row['identity_sha256'],
                    'trajectory_identity':row.get('trajectory_identity'),'update':update+1,'rank':rank,'pack_id':pack_id,
                    'record_token_offset':base,'target_start_in_record':start,
                    'prefix_tokens':prefix,'target_tokens':target})
                selected[source]+=1; seen.add(uniqueness)
    assert tokens_total==receipt['sft_total_tokens'] and targets_total==receipt['assistant_target_tokens'], 'consumed budget mismatch'
    assert dict(selected)==quotas, f'insufficient eligible prefixes: {selected}; consumed records={occurrences}; consumed targets={source_targets}'
    panel_path=SCREEN/'dev-panel.json'
    assert sha256(panel_path)=='a29094fa9934d83c7f89ce6f4f4cf93fbaa9982aa295bec78684616896d0cb90'
    panel=json.loads(panel_path.read_text())
    for task in panel['tasks']:
        text=serialize_pi_messages([{'role':'system','content':panel['system']},{'role':'user','content':task['user']}],private_analysis=True)
        examples.append({'id':'development-'+task['id'],'split':'consumed-development','source':task['kind'],
                         'prefix_tokens':encoding.encode_ordinary(text),'target_tokens':[]})
    result={'schema':'emender-e97-readonly-learning-probes-v1','training_eligible':False,
        'purpose':'read-only fitting, prompt-conditioning and train/serve parity diagnostic; not a new capability gate',
        'selection':'first eligible records in update/rank/pack order; unique agent trajectories; prefix <=4096 tokens, first <=64 supervised targets; no model-output selection',
        'scope':'prefix likelihood only; not complete-turn generation, full packed backward parity, or independent held-out evaluation',
        'budgets':{'consumed_training_updates':64,'data_world_size':8,'input_tokens':tokens_total,'target_tokens':targets_total,
                   'record_occurrences':dict(occurrences),'unique_source_records':len(unique_records),'unique_agent_trajectories':len(unique_agent_trajectories),
                   'source_target_tokens':dict(source_targets),'source_input_tokens':dict(source_inputs)},
        'sampler_identity':identity.to_metadata(),'sampler_mode':'epoch-permutation',
        'sampler_source_sha256':sha256(module),'checkpoint_sha256':receipt['checkpoint_sha256'],
        'reload_receipt_sha256':sha256(receipt_path),'development_panel_sha256':sha256(panel_path),
        'marker_candidates':['Analysis: ','Action: ','Final: ','The '],
        'checkpoints':['parent-train','prior-lr2e6-train','u64-lr5e5-train','u64-lr5e5-saved'],
        'consumed_packs':consumed,'examples':examples,'builder_sha256':sha256(Path(__file__))}
    payload=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    publish_bytes_no_replace(args.output,payload,mode=0o600)
    print(json.dumps({'output':str(args.output),'sha256':hashlib.sha256(payload).hexdigest(),'examples':len(examples),'selected_training':dict(selected),'budgets':result['budgets']}),flush=True)
    data.close()

if __name__=='__main__': main()
