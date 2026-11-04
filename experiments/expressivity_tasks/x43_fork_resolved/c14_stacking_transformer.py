"""Join committed transformer per-position losses to frozen C14 windows (CPU)."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from .c14_documents import sha
from .c14_stacking_predict import ROOT,OUT,EXPECTED,WINDOWS
from .c14_stacking_score import write

PRED=Path('/home/erikg/c14_stacking_transformer_losses.npy')
SOURCE=Path('/home/erikg/c14_dense_bytes/transformer_eval.jsonl')
WINDOW_PATH=Path('/home/erikg/c14_dense_bytes/eval_windows.jsonl')


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    start=time.monotonic()
    training=json.loads((ROOT/'receipts/c14_dense/training.json').read_text())['models'][1]
    assert training['arm']=='transformer' and sha(WINDOW_PATH)==WINDOWS
    assert sha(SOURCE)==training['eval_sha256']
    loss=np.lib.format.open_memmap(PRED,mode='w+',dtype=np.float32,shape=(EXPECTED[1],))
    offset=windows=0; sums={}; digest=hashlib.sha256()
    with SOURCE.open('rb') as source, WINDOW_PATH.open() as frozen:
        for line,window_line in itertools.zip_longest(source,frozen):
            assert line is not None and window_line is not None
            digest.update(line); r=json.loads(line); w=json.loads(window_line)
            assert (r['document'],r['start'])==(w['document'],w['start'])
            nll=np.array(r['nll'],dtype=np.float32)
            assert len(nll)==w['targets'] and np.isfinite(nll).all()
            loss[offset:offset+len(nll)]=nll
            for st,m in [('total',np.ones(len(nll),bool)),('window_reset_1_16',np.arange(len(nll))<16),('window_interior_17_128',np.arange(len(nll))>=16)]:
                key=w['half']+'/'+st; total,n=sums.setdefault(key,[0.,0])
                sums[key]=[total+float(nll[m].astype(np.float64).sum()),n+int(m.sum())]
            offset+=len(nll); windows+=1
            if windows%500000==0: print('TRANSFORMER_JOIN',windows,offset,flush=True)
    assert offset==EXPECTED[1] and windows==EXPECTED[0]
    assert digest.hexdigest()==training['eval_sha256']
    loss.flush()
    for half in ['calibration','final_test']:
        total,n=sums[half+'/total']
        assert n==training['heldout'][half]['targets']
        assert abs(total/n-training['heldout'][half]['nll'])<1e-9
    write('transformer_identity.json',dict(path=str(PRED),sha256=sha(PRED),source=str(SOURCE),
        source_sha256=training['eval_sha256'],windows_sha256=WINDOWS,windows=windows,targets=offset,
        historical_batch=16,new_model_inference=False,source_sha256_rechecked_after_join=True,
        sums=sums,code_sha256=sha(__file__),seconds=time.monotonic()-start))
    print('TRANSFORMER_COMPLETE',offset,flush=True)


if __name__=='__main__':main()
