"""Attach immutable precision provenance to final rows without changing raw attempts."""
import hashlib
import json
import shutil
from .part8_surface import ROOT,OUT


def main():
    source=ROOT/'receipts/part3_bf16/gate.json'
    proof=json.loads(source.read_text())
    assert proof['status']=='failed' and not proof['enabled']
    shutil.copyfile(source,OUT/'bf16_store_rejected.json')
    for path in (OUT/'surface').glob('*.json'):
        row=json.loads(path.read_text())
        row['precision_evidence']=dict(emender_native_store='fp32; bf16 store failed production parity',
            baselines='bf16 dense and fused vocabulary projection, fp32 master parameters and AdamW',
            rejected_bf16_store_receipt='receipts/part8/bf16_store_rejected.json',
            rejected_bf16_store_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            asymmetry='Intentional: report ratios against best-practice bf16 baselines, without equalizing store precision.')
        row['common_allocator_request']={name:'expandable_segments:True' for name in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}
        row['allocator_observation_note']='Both aliases were set by the sequential driver for every attempt. This v2 probe did not capture allocator_settings per attempt; no per-attempt observed allocator assertion is claimed.'
        path.write_text(json.dumps(row,indent=2)+'\n')

if __name__=='__main__':main()
