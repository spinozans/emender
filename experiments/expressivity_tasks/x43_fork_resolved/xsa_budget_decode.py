"""Generate bytes with a retained budgeted-read cell (one stateful session)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from . import xsa_budget_data as D
from .xsa_budget_run import Policy


class Decoder:
    def __init__(self,bundle,seed):
        bind=json.loads((bundle/'binding.json').read_text());ck=bundle/f'{seed}.pt';rec=json.loads((bundle/f'{seed}_training.json').read_text());self.price=json.loads((bundle/f'{seed}_result.json').read_text())['selected_price']
        if D.digest(D.EXE)!=bind['engine'] or D.digest(ck)!=rec['sha'] or D.digest(Path(D.__file__))!=bind['data']:raise ValueError('inference identity mismatch')
        session=D.Session
        if (bundle/'recovery.json').is_file():
            from . import xsa_budget_cap_recovery as recovery
            rec=json.loads((bundle/'recovery.json').read_text())
            if rec['cap']!=8192 or rec['wrapper_sha']!=D.digest(Path(recovery.__file__)):raise ValueError('recovery identity mismatch')
            session=recovery.FullWindow
        self.base,_=D.models();self.net=Policy().eval();self.net.load_state_dict(torch.load(ck,weights_only=True));self.s=session(D.EXE);self.ctx=b'';self.t=0

    def close(self):self.s.close()

    def append(self,data):
        for i in range(0,len(data),65536):
            part=data[i:i+65536];new=self.t+len(part)
            if self.s.rpc('append '+part.hex())!={'visible':new}:raise RuntimeError('offset mismatch')
            if new//512>self.t//512:self.s.rpc('seal')
            self.ctx=(self.ctx+part)[-192:];self.t=new

    @torch.no_grad()
    def predict(self):
        if self.t<16:raise ValueError('need >=16 observed bytes')
        calls=Counter()
        def rpc(cmd): calls[cmd.split()[0]]+=1;return self.s.rpc(cmd)
        x=torch.tensor([list(self.ctx[-16:])]);z=self.base.encoder(self.base.embedding(x).flatten(1));q=self.base.head(z).double().softmax(1)[0].numpy();h3=D.branch(rpc,self.ctx[-3:]);keys,m=D.menu(self.ctx);k=len(keys);f=torch.zeros(1,k,14);f[0,:,:8]=torch.from_numpy(m);f[0,:,8:11]=torch.tensor(D.features(h3,q));j=int(self.net.pick(self.net.estimate(z,f),torch.ones(1,k,dtype=torch.bool),self.price));h=D.action(rpc,j,keys,self.ctx,self.t);f[0,j,11:]=torch.tensor(D.features(h,q));w=self.net.weights(z,f)[0,j].numpy();p=w[0]*q
        for weight,c in zip(w[1:],(h3,h)):
            v=q.copy()
            for b,n in c.items():v[b]+=n
            p+=weight*v/(1+c.total())
        if not np.isclose(p.sum(),1):raise ValueError('unnormalized')
        return p,dict(action=j,available=k,anchor_hex=keys[j][0].hex() if j>=2 else None,work=dict(calls))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('bundle',type=Path);ap.add_argument('prefix',type=Path);ap.add_argument('--seed',type=int,default=3401);ap.add_argument('--steps',type=int,default=32);a=ap.parse_args();torch.set_num_threads(4);dec=Decoder(a.bundle,a.seed);trace=[];out=bytearray()
    try:
        dec.append(a.prefix.read_bytes())
        for _ in range(a.steps):
            start=time.perf_counter_ns();p,row=dec.predict();byte=int(p.argmax());dec.append(bytes([byte]));out.append(byte);trace.append(dict(**row,byte=byte,ms=(time.perf_counter_ns()-start)/1e6))
    finally:dec.close()
    print(json.dumps(dict(generated_hex=out.hex(),trace=trace),indent=2))
