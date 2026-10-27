"""Summarize completed smoke evidence; never infer the full-run verdict."""
import json
import statistics
from .data import ROOT


def main():
    arms={arm:json.loads((ROOT/f'receipts/smoke_{arm}.json').read_text())
          for arm in ('base_lm','tape_lm')}
    result={'scope':'part 1 smoke only','arms':{}}
    for arm,receipt in arms.items():
        curve=receipt['curve']
        assert [r['step'] for r in curve]==list(range(1,receipt['steps']+1))
        result['arms'][arm]=dict(status=receipt['status'],tokens=receipt['tokens'],steps=receipt['steps'],
            first_loss=curve[0]['loss'],last_loss=curve[-1]['loss'],
            validation=[dict(step=r['step'],tokens=r['tokens'],nll=r['validation']['nll'],
                             perplexity=r['validation']['perplexity']) for r in curve if 'validation' in r],
            elapsed_seconds=receipt['elapsed_seconds'],
            median_step_seconds=statistics.median(r['seconds'] for r in curve[1:]))
    a,b=arms['base_lm'],arms['tape_lm']
    result['matched_tokens']=a['tokens']==b['tokens']
    assert a['fingerprint']['data']==b['fingerprint']['data']
    assert a['fingerprint']['tokenizer']==b['fingerprint']['tokenizer']
    assert a['fingerprint']['protocol']==b['fingerprint']['protocol']
    result['measured_training_time_ratio_tape_over_base']=(
        result['arms']['tape_lm']['median_step_seconds']/result['arms']['base_lm']['median_step_seconds'])
    metrics=[r['telemetry'][0] for r in b['curve']]
    result['tape_telemetry']=dict(peak_live_per_head=[max(m['peak_live_per_head'][h] for m in metrics) for h in range(4)],
        pressure_max=max(m['pressure_max'] for m in metrics),
        pressure_fraction_mean=statistics.mean(m['pressure_fraction'] for m in metrics),
        replay_asserted_every_step=all(r['replay_node']=='ReplayBackward' for r in b['curve']))
    (ROOT/'receipts/smoke_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import os,sys
    os.environ['MPLCONFIGDIR']=str(ROOT/'runs/mplconfig')
    sys.path.append(str(ROOT/'runs/plot_deps'))
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(json.dumps(result,indent=2))
        print('SVG skipped: install plot_requirements.txt under runs/plot_deps')
        return
    matplotlib.use('Agg')
    import os,sys
    os.environ['MPLCONFIGDIR']=str(ROOT/'runs/mplconfig')
    sys.path.append(str(ROOT/'runs/plot_deps'))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for arm,receipt in arms.items():
        rows=receipt['curve']
        axes[0].plot([r['tokens']/1e6 for r in rows],[r['loss'] for r in rows],label=arm,alpha=.7)
        val=result['arms'][arm]['validation']
        axes[1].plot([r['tokens']/1e6 for r in val],[r['perplexity'] for r in val],label=arm,marker='o')
    axes[0].set_ylabel('Training NLL (nat/token)')
    axes[1].set_ylabel('Held-out perplexity')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Millions of training tokens');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('X34 smoke: engineering evidence, no full-run verdict')
    fig.tight_layout();fig.savefig(ROOT/'receipts/loss_curves.svg',metadata={'Date':None})
    svg=ROOT/'receipts/loss_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
