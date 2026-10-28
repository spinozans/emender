"""Read-only completion attestation; in-progress builds are never restarted."""
import argparse
import json
import time
from pathlib import Path
from .data import digest


def attest(root):
    root=Path(root)
    progress=json.loads((root/'progress.json').read_text())
    receipt=dict(snapshot_unix=time.time(),root=str(root),progress=progress,
                 monitor_command=f'cat {root}/progress.json',builder_pid=progress.get('pid'))
    if not (root/'manifest.json').exists() or progress['status']!='complete':
        return dict(receipt,status='not_complete',final_attestation=False,
                    next_action='After progress.status=complete, rerun this module; do not restart the builder.')
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['complete']
    assert digest(root/'tokenizer.json')==manifest['tokenizer_sha256']
    assert digest(root/'documents.jsonl')==manifest['document_index_sha256']
    counts=dict(train=0,valid=0);documents=dict(train=0,valid=0);boundaries=dict(train=None,valid=None)
    last=-1
    with (root/'documents.jsonl').open() as f:
        for line in f:
            row=json.loads(line);split=row['split'];source=row['source_index']
            assert source>last;last=source
            assert row['token_start']==counts[split] and row['tokens']>0
            counts[split]+=row['tokens'];documents[split]+=1
            if boundaries[split] is None:
                boundaries[split]=dict(first_source_index=source)
            boundaries[split]['last_source_index']=source
    assert counts==manifest['tokens']==progress['tokens']
    assert documents==manifest['split_documents']==progress['split_documents']
    assert sum(documents.values())==manifest['documents']
    assert boundaries==manifest['split_boundaries']
    assert manifest['source_interval']==dict(first_source_index=0,last_source_index=manifest['source_documents_scanned']-1)
    assert last<=manifest['source_interval']['last_source_index']
    for split in counts:
        assert (root/f'{split}.bin').stat().st_size==2*counts[split]
        assert counts[split]==progress['target_tokens'][split]
        assert digest(root/f'{split}.bin')==manifest['split_sha256'][split]
    return dict(receipt,status='attested',final_attestation=True,manifest=manifest,
                checks='tokenizer/index/split SHA256; exact token/document counts; strict source order; contiguous per-split offsets; boundaries; registered token budgets')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    try:
        result=attest(a.data)
    except Exception as exc:
        Path(a.output).write_text(json.dumps(dict(status='failed',error=repr(exc)),indent=2)+'\n');raise
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])

if __name__=='__main__':
    main()
