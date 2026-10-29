"""Standalone completed scale-surface figure; no missing-arm interpolation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parent/'receipts/part6'


def main():
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['surface_status']=='passed' and summary['matched_ratios']==12
    scales=('22M','150M','1.3B');x=(22e6,150e6,1.3e9)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True,layout='constrained')
    for axis,(context,batch) in zip(axes.flat,((512,64),(512,128),(2048,64),(2048,128))):
        rows=[next(r for r in summary['surface'] if r['scale']==scale and r['context']==context and r['batch']==batch) for scale in scales]
        for arm,color in (('pure','#1769aa'),('transformer','#b65408')):
            rates=[row[arm]['tokens_per_second'] for row in rows]
            axis.plot(x,rates,'o-',color=color,label=arm)
        axis.set_xscale('log');axis.set_yscale('log');axis.grid(True,alpha=.2)
        axis.set_title(f'T{context} / effective B{batch}')
        axis.set_xticks(x,scales);axis.set_ylabel('Sustained training tokens/s')
        ratio=axis.twinx()
        rates=[row['ratio_to_transformer'] for row in rows]
        ratio.plot(x,rates,'s--',color='#333333',alpha=.75,label='pure / transformer')
        if context==512 and batch==64:first_ratio=ratio
        ratio.set_ylim(0,max(1,max(rates)*1.1));ratio.set_ylabel('Pure / transformer ratio')
        for a,b in zip(x,rates):ratio.annotate(f'{b:.3f}',(a,b),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[0,0].legend(handles=list(axes[0,0].lines)+list(first_ratio.lines),loc='upper right',fontsize=8)
    for axis in axes[1]:axis.set_xlabel('Pure trainable scale; control matches pure total capacity')
    fig.suptitle('X40 part 6: all 24 scale arms / 12 matched ratios\nPhysical B32 accumulation; three warmups + 20 complete training updates\nPure 1.3B/T2048 uses the verified expandable CUDA allocator',fontsize=13)
    fig.savefig(ROOT/'scale_surface.svg')
    fig.savefig(ROOT/'scale_surface.png',dpi=180)
    (ROOT/'plot_renderer.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,source='summary.json',arms=24,ratios=12,
        protocol='bf16 dense blocks; fp32 native state and bounded vocabulary projection; block checkpoint, weighted microbatch accumulation, clip + AdamW once per effective update'),indent=2)+'\n')


if __name__=='__main__':main()
