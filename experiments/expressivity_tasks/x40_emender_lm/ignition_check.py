"""Bounded CPU counterexample / broker-only GPU ignition confirmation."""
import argparse,json,os
from pathlib import Path
import torch
from .model import LM,POLICIES,require_replays
from ..x30_tape_cell.naive_reference import tape_scan,init_state
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(3401)
    model=LM(128,ignition=True)
    rows=[]
    for b,block in enumerate(model.blocks):
        c=block.cell;x=torch.randn(1,16,256)*100
        gate=(c.W_write(x)+c.match_coef*1e6+c.error_coef*1e6).sigmoid()
        assert (gate>.5).all() and not c.W_write.weight.requires_grad
        assert not c.match_coef.requires_grad and not c.error_coef.requires_grad
        rows.append(dict(block=b,min_gate=float(gate.min()),max_gate=float(gate.max())))
    cpu=[]
    for h,cfg in enumerate(POLICIES):
        k=torch.eye(64)[:16].reshape(1,16,1,64).contiguous()
        v=torch.randn(1,16,1,128);beta=torch.ones(1,16,1)
        out,state,trace=tape_scan(k,k,v,beta,torch.zeros_like(beta),
            torch.zeros_like(beta),init_state(1,cfg,"cpu"),cfg,append_err_coef=torch.zeros(1))
        assert state.live.sum()>0 and trace.w_on.all()
        cpu.append(dict(head=h,novel_tokens=16,enabled=16,final_live=int(state.live.sum())))
    result=dict(status='passed',blocks=rows,cpu_novel_content=cpu,
        limitation='CPU uses eager X30 ISA reference, not certification of the optimized X40 kernel; GPU parity remains mandatory.')
    if a.gpu:
        assert os.environ.get('GPU_LEASE_HELD') in ('0','1','5','6')
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ['GPU_LEASE_HELD']
        from tokenizers import Tokenizer
        from ..x34_lm_pilot import data
        from .parity import certify
        result['short_parity']=certify(batch=2,context=16)
        vocab=Tokenizer.from_file(str(data.RUNS/'tokenizer.json')).get_vocab_size()
        ids=data.batch('train',0,batch_size=1)[0][:,:32].cuda()
        torch.manual_seed(3401)
        original=LM(vocab).cuda().eval()
        with torch.no_grad():
            _,initial=original(ids)
        result['v0_initial_actual_data']=[[dict(gate=h['gate_mean'],match=h['match_mean'],peak=h['peak_live_per_head'][0]) for h in b] for b in initial]
        del original
        torch.manual_seed(3401)
        model=LM(vocab,ignition=True).cuda().eval()
        logits,telemetry=model(ids);loss=logits.square().mean()
        count=require_replays(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        lanes=[]
        for b,block in enumerate(telemetry):
            for h,m in enumerate(block):
                assert m['peak_live_per_head'][0]>0
                assert m['gate_mean']>.5
                lanes.append(dict(block=b,head=h,gate=m['gate_mean'],peak=m['peak_live_per_head'][0]))
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
        opt.step();opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            _,after=model(ids)
        assert all(h['gate_mean']>.5 and h['peak_live_per_head'][0]>0 for b in after for h in b)
        result['after_one_optimizer_step']='all 24 lanes enabled and live'
        result.update(gpu=dict(physical_card=os.environ['GPU_LEASE_HELD'],replay_nodes=count,lanes=lanes,batch=1,context=32))
    path=ROOT/'receipts'/('part2_gpu_ignition.json' if a.gpu else 'part2_cpu_ignition.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
