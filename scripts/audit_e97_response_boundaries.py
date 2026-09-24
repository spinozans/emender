#!/usr/bin/env python3
"""Read-only response-boundary alignment audit over actual consumed u64 packs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import tiktoken
from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset,SFTSamplerIdentity,sha256
from ndm.e97_atomic import publish_bytes_no_replace
from scripts.freeze_e97_learning_diagnostic import AUTH,PACK,TRAINING


def main():
    p=argparse.ArgumentParser(); p.add_argument('--probes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    assert sha256(args.probes)=='4b6ec6efae0fe7d9a132ffeae3a680d2ab81320aa43b8b997118c1f1900a3370'
    probes=json.loads(args.probes.read_text()); root=Path(__file__).resolve().parents[1]
    source={}
    for name in ['ndm/data/masked_sft_dataset.py','ndm/models/ladder_lm.py','scripts/train_e97_4b_pi_sft.py']:
        assert sha256(root/name)==sha256(TRAINING/name),f'executed source mismatch: {name}'
        source[name]=sha256(root/name)
    identity=SFTSamplerIdentity.from_metadata(probes['sampler_identity'])
    data=MaskedSFTPackedDataset(AUTH,PACK,identity=identity,rank=0,sampler_mode=probes['sampler_mode'])
    info=data.authority_manifest['outputs']['metadata']; metadata=AUTH/info['path']
    assert metadata.parent==AUTH and sha256(metadata)==info['sha256']
    rows=[json.loads(line) for line in metadata.open()]
    enc=tiktoken.get_encoding('p50k_base'); counts=Counter(); examples=[]
    wanted={e['record_id'] for e in probes['examples'] if e['split']=='consumed-training' and e['source']=='agent'}
    for consumed in probes['consumed_packs']:
        data.rank=consumed['rank']; pack_id=data.pack_id_at(consumed['update']-1)
        assert pack_id==consumed['pack_id']
        tokens,loss_mask,valid,reset,length,targets,_=data.pack_at_with_boundaries(pack_id)
        assert int(loss_mask.sum())==targets
        descriptor=data.packs[pack_id]; offset=int(descriptor['record_offset']); n=int(descriptor['record_count'])
        record_ids=data.pack_record_ids[offset:offset+n]; spans=data.record_spans_at(pack_id)
        for raw_id,(start,end) in zip(record_ids,spans):
            record_id=int(raw_id); row=rows[record_id]
            if row['source']!='agent': continue
            counts['agent_records']+=1
            rec=data.records[record_id]; base=int(rec['offset']); m=data.masks[base:base+int(rec['tokens'])]
            starts=np.flatnonzero((m==1)&np.concatenate(([True],m[:-1]==0)))
            for j,local in enumerate(starts):
                position=start+int(local)
                text=enc.decode(tokens[position:min(end,position+16)].tolist())
                if text=='\x1e': counts['separator_only_runs']+=1; continue
                assert text.startswith('Analysis: '), 'unexpected agent target-run opening'
                before=enc.decode(tokens[max(start,position-16):position].tolist())
                assert before.endswith('Assistant:\n'), 'missing assistant boundary'
                assert position>start and loss_mask[position-1] and valid[position-1] and valid[position] and not reset[position]
                # Actual LadderLM forward: inp=x[:-1], target=x[1:], so the
                # prediction at position-1 sees only the prefix, not Analysis.
                assert int(tokens[1:][position-1])==int(tokens[position])==enc.encode_ordinary('Analysis')[0]
                counts['analysis_target_runs']+=1
                if j==0 and record_id in wanted:
                    examples.append({'record_id':record_id,'pack_id':pack_id,'rank':consumed['rank'],'update':consumed['update'],
                         'record_start_in_pack':start,'target_token_position_in_pack':position,'prediction_position':position-1,
                         'loss_mask_at_prediction':bool(loss_mask[position-1]),'reset_before_target':bool(reset[position]),
                         'token_window':[{'position':k,'id':int(tokens[k]),'text':enc.decode([int(tokens[k])])} for k in range(max(start,position-5),min(end,position+3))]})
        counts['packs']+=1
    assert counts['packs']==512 and counts['agent_records']==169 and len(examples)==12
    result={'schema':'emender-e97-response-boundary-audit-v1','status':'passed','training_eligible':False,
            'source_sha256':source,'probe_sha256':sha256(args.probes),'counts':dict(counts),'examples':examples,
            'scope':'actual consumed pack response headers, shifted labels, valid/reset/loss masks and executed source identity; not production fused backward or optimizer correctness'}
    publish_bytes_no_replace(args.output,(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print(json.dumps({'counts':dict(counts),'examples':examples[:1],'sha256':sha256(args.output)}),flush=True)

if __name__=='__main__': main()
