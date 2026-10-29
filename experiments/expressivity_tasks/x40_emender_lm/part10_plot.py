"""Standalone plots of paired empirical loss and exclusive diagnostic phases."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def main():
    curves=[[json.loads(line) for line in (OUT/f'curve_{arm}.jsonl').read_text().splitlines()] for arm in ('fp32','bf16')]
    assert len(curves[0])==len(curves[1])==153
    x=[r['tokens']/1e6 for r in curves[0]]
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True,layout='constrained')
    for rows,label in zip(curves,('FP32 reference','BF16 read / FP32 surrogate replay')):axes[0].plot(x,[r['loss'] for r in rows],label=label,lw=1.2)
    axes[0].set_ylabel('Training loss');axes[0].legend();axes[0].set_title('Matched 5M-token pilot slice; candidate is not certified')
    axes[1].plot(x,[abs(a['loss']-b['loss']) for a,b in zip(*curves)],lw=1)
    axes[1].set_xlabel('Tokens (millions)');axes[1].set_ylabel('Absolute loss difference')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'loss_curves.svg');plt.close(fig)
    phases=[('sequential_forward_core','Sequential core'),('forward_read_elementwise_tiles','Batched read'),('reverse_fused_read_math_and_adjoints','Fused reverse'),('reverse_setup_and_teardown','Reverse setup'),('initial_checkpoint_dense_and_native_setup','Initial dense/setup'),('checkpoint_recompute_dense_and_native_setup','Recompute dense/setup'),('backward_dense_loss_and_replay_setup','Dense backward/loss'),('forward_dense_loss_and_native_setup','Forward/loss'),('optimizer','Optimizer'),('clip','Clip')]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    bottoms=[0.,0.]
    for phase,label in phases:
        vals=[]
        for context in (512,2048):
            r=json.loads((OUT/f'profile_pure_T{context}.json').read_text());vals.append(r['exclusive_seconds'][phase]/r['wall_seconds']*100)
        ax.bar(['T512','T2048'],vals,bottom=bottoms,label=label)
        bottoms=[a+b for a,b in zip(bottoms,vals)]
    ax.set_ylabel('Instrumented wall (%)');ax.set_title('1.3B B64/B32 checkpoint diagnostic; fused reverse is not subdivided')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8);fig.savefig(OUT/'diagnostic_phases.svg');plt.close(fig)

    for p in OUT.glob('*.svg'):p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')

if __name__=='__main__':main()
