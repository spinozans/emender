"""Bounded projection chunk sizing under the unchanged X36 component tolerances."""
import argparse
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from ..x36_tape_perf.loss import ChunkedLoss

ROOT=Path(__file__).resolve().parent/'receipts/part6'


def main():
    p=argparse.ArgumentParser();p.add_argument('--chunk',type=int,choices=[4096,16384],required=True);a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(6401)
    h=(torch.randn(32768,256,device='cuda')*.1).requires_grad_()
    w=(torch.randn(32768,256,device='cuda')*.1).requires_grad_()
    targets=torch.randint(32768,(32768,),device='cuda')
    result=dict(status='failed',chunk=a.chunk,tokens=32768,width=256,vocab=32768,precision='fp32',
                loss_atol=1e-6,loss_rtol=1e-6,gradient_atol=2e-7,gradient_rtol=2e-6,
                scope='Existing X36 test_bounded_projection_matches_dense tolerances; native gradients retain their independent zero-tolerance gates.')
    try:
        dense=F.cross_entropy(F.linear(h,w),targets)
        reference=torch.autograd.grad(dense,(h,w))
        bounded=ChunkedLoss.apply(h,w,targets,a.chunk)
        gradients=torch.autograd.grad(bounded,(h,w))
        result.update(loss_absolute_difference=float((dense-bounded).abs()),
                      hidden_gradient_max_difference=float((reference[0]-gradients[0]).abs().max()),
                      weight_gradient_max_difference=float((reference[1]-gradients[1]).abs().max()))
        torch.testing.assert_close(bounded,dense,atol=1e-6,rtol=1e-6)
        for x,y in zip(reference,gradients):torch.testing.assert_close(x,y,atol=2e-7,rtol=2e-6)
        result['status']='passed'
    except Exception as exc:result['error']=repr(exc);raise
    finally:
        result.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        (ROOT/f'loss_chunk{a.chunk}_parity.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
