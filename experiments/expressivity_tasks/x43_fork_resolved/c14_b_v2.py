"""Isolated B1-v2 receipts/checkpoint, using the unchanged registered protocol."""
import sys
import json
from pathlib import Path
from . import c14_b_train as train


def configure():
    train.OUT=train.ROOT/'receipts/c14_b_v2'
    train.CHECKPOINT=Path('/home/erikg/c14_dense_bytes/emender_c14_b1_v2_resume.pt')
    train.CODE=(*train.CODE,'c14_b_batched_replay.py','c14_b_replay_graph.py','c14_b_v2.py')


def main():
    configure()
    phase=sys.argv.pop(1)
    if phase=='validate':
        from . import validate_c14_b_enabled as validator
        validator.OUT=train.OUT;validator.CODE=train.CODE
        validator.main()
        gate=json.loads((train.OUT/'launch_gates.json').read_text())
        gate['implementation']='Unchanged B-only Triton forward; batched PyTorch autograd replay with fused tensor orchestration and CUDA graph launch capture; reentrant block checkpoint'
        gate['limitations']=['No production store kernel or X40 reverse-kernel changes; scalar eager store remains independent oracle.',
            'Native/replay output agreement is checked on every backward at the standing 2e-5 tolerance; cache answer bits and disabled dense outputs remain exact.',
            'Captured buffers are overwritten per call and slot state is reset per window; returned gradients are copied before buffer reuse.']
        train.write('launch_gates.json',gate)
        print(json.dumps({'v2_implementation':gate['implementation']}),flush=True)
    elif phase=='train':train.main()
    elif phase=='eval':
        from . import c14_b_eval as evaluation
        evaluation.OUT=train.OUT
        evaluation.main()
    else:raise ValueError('phase must be validate, train or eval')

if __name__=='__main__':main()
