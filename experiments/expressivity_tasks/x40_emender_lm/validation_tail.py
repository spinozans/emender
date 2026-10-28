"""Sequential broker child commands, retaining expected gate failures as failures."""
import json
import os
from pathlib import Path
import subprocess
from .performance import write

ROOT=Path(__file__).resolve().parent
PYTHON='/home/erikg/emender/.venv/bin/python'


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):raise RuntimeError('lease required')
    receipts=ROOT/'receipts'
    commands=[
        (['-m','pytest','-q',str(ROOT/'tests')],receipts/'part3_validation/x40_tests.log',0),
        (['-m','experiments.expressivity_tasks.x40_emender_lm.scale_probe','--output',str(receipts/'part3_scale_final')],receipts/'part3_scale_failure/final_console.txt',1),
        (['-m','experiments.expressivity_tasks.x40_emender_lm.scale_probe','--arm','transformer','--output',str(receipts/'part3_transformer_scale')],receipts/'part3_validation/transformer_scale.log',0),
        (['-m','experiments.expressivity_tasks.x40_emender_lm.bf16_gate','--output',str(receipts/'part3_bf16')],receipts/'part3_validation/bf16.log',1),
        (['-m','experiments.expressivity_tasks.x40_emender_lm.train_slice','--graph','--output',str(receipts/'part3_slice')],receipts/'part3_validation/slice.log',0),
        (['-m','experiments.expressivity_tasks.x40_emender_lm.graph_probe','--output',str(receipts/'part3_graph_final')],receipts/'part3_validation/graph_final.log',0),
    ]
    results=[]
    for args,logpath,expected in commands:
        with logpath.open('x') as log:run=subprocess.run([PYTHON,*args],stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(command=[PYTHON,*args],exit_code=run.returncode,expected_exit=expected,
            result='passed' if run.returncode==0 else 'failed'))
        write(receipts/'part3_validation/commands.json',results)
        print(json.dumps(results[-1]),flush=True)
        if run.returncode!=expected:raise RuntimeError(f'unexpected exit: {args}')

if __name__=='__main__':main()
