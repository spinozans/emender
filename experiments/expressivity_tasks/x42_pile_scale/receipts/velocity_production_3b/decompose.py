import json,time
from pathlib import Path
import torch
from experiments.expressivity_tasks.x42_pile_scale import velocity_train as runner,velocity_candidate as candidate
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
rows=[];phase='setup'
def timed(label,original):
    def call(*a,**k):
        torch.cuda.synchronize();start=time.monotonic()
        try:return original(*a,**k)
        finally:
            torch.cuda.synchronize();rows.append(dict(component=label,seconds=time.monotonic()-start,phase=phase))
    return call
candidate.scan=timed('head_scan_forward',candidate.scan)
candidate.Candidate.loss=timed('loss_forward',candidate.Candidate.loss)
torch.Tensor.backward=timed('backward',torch.Tensor.backward)
torch.optim.AdamW.step=timed('optimizer',torch.optim.AdamW.step)
original=runner.evaluate
def evaluate(*a,**k):
    global phase
    phase='evaluation';return original(*a,**k)
runner.evaluate=evaluate
reg='experiments/expressivity_tasks/x42_pile_scale/velocity_registration.json'
runner.verify_registration(reg);phase='training'
try:result=runner.train(candidate.Policy(),reg,'experiments/expressivity_tasks/x42_pile_scale/runs/velocity_decompose_v1',seconds=120,eval_windows=2)
finally:atomic_json('experiments/expressivity_tasks/x42_pile_scale/receipts/velocity_production_decomposition.json',dict(method='synchronized wall timers; nested scan times overlap loss_forward; engineering budget 120s; eval windows 2',rows=rows))
