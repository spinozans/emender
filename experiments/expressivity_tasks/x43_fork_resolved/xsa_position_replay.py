"""Verify the serving implementation independently against frozen test scores."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from .xsa_position_predict import PositionalDecoder
from .xsa_natural_position import OFFSETS, flagship_identity


def main(exe, bundle, out):
    torch.set_num_threads(4)
    docs=json.loads((bundle/'documents.json').read_text())['test']
    ref=json.loads((bundle/'result.json').read_text())['doc_nll']['positional']
    times=[];errors=[];losses=[]
    with open(flagship_identity()['source'],'rb') as raw:
        for i,d in enumerate(docs):
            raw.seek(d['offset']);text=raw.read(d['length'])
            if hashlib.sha256(text).hexdigest()!=d['sha256']:raise ValueError('source mismatch')
            # Hash the weights and executable at construction, outside timed steps.
            decoder=PositionalDecoder(exe,bundle);prev=0;total=0.
            try:
                for t in OFFSETS:
                    start=time.perf_counter_ns();decoder.observe(text[prev:t]);p,_=decoder.predict();times.append((time.perf_counter_ns()-start)/1e6);prev=t
                    total-=np.log(p[text[t]])
            finally:decoder.close()
            mean=float(total/len(OFFSETS));errors.append(abs(mean-ref[i]));losses.append(mean)
    if max(errors)>1e-5:raise ValueError(f'serving/scoring mismatch: {max(errors)}')
    r=dict(documents=len(docs),targets=len(times),nll=float(np.mean(losses)),max_document_nll_error=max(errors),elapsed_ms_quantiles=dict(zip(('p50','p95','p99','max'),map(float,np.quantile(times,[.5,.95,.99,1])))),scope='CPU debug XSA, full neural prediction and transport, teacher-forced stride 7; includes observe (first 256 bytes), scheduled synchronous seal and cold first step per document; excludes process launch and weight load. Not one-token generation or large-base timing.')
    out.write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('exe',type=Path);a.add_argument('bundle',type=Path);a.add_argument('out',type=Path);o=a.parse_args();main(o.exe,o.bundle,o.out)
