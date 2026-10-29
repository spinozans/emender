"""Render ratio curves directly from validated summary; gaps remain gaps."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .part8_surface import OUT


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,(context,batch) in zip(axes.flat,((512,64),(512,128),(2048,64),(2048,128))):
        rows=[r for r in summary['ratios'] if r['context']==context and r['batch']==batch]
        for key,label in (('pure_over_transformer','Pure / transformer'),('pure_over_gdn2','Pure / GDN2'),('gdn2_over_transformer','GDN2 / transformer')):
            ax.plot(range(3),[float('nan') if r[key] is None else r[key] for r in rows],marker='o',label=label)
        ax.axhline(1,color='gray',ls=':',lw=1)
        ax.set_title(f'T{context}, B{batch}')
        ax.set_xticks(range(3),('22M','150M','1.3B'))
        ax.grid(alpha=.25);ax.set_ylabel('Sustained training throughput ratio')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('X40 v2: fp32 native emender vs bf16 best-practice baselines\nScale labels are pure trainable counts; capacity matches use total parameters')
    fig.tight_layout();fig.savefig(OUT/'ratio_curves.svg');fig.savefig(OUT/'ratio_curves.png',dpi=160);plt.close(fig)
    svg=OUT/'ratio_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

if __name__=='__main__':main()
