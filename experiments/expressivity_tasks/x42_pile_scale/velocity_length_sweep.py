"""Additional eval only: X40 context_probe window discipline on the Pile stream."""
import math
import torch
from .velocity_train import window, telemetry

LENGTHS = (512, 1024, 2048, 4096)
SCOPE = ('Proxy scale: 150M trainable parameters, 45-minute active training, Pile slice; '
         'NOT TinyStories and NOT the terminal 1.3B comparison. '
         'First apples-to-apples GDN2-vs-emender LM length receipt only when all arms complete.')

@torch.no_grad()
def sweep(model, valid, *, pure=False, windows=2, window_fn=window):
    if windows < 1 or len(valid) < windows * max(LENGTHS) + 1:
        raise ValueError('insufficient held-out tokens for complete sweep')
    model.eval()
    rows = []
    for context in LENGTHS:
        losses, occupancy = [], []
        for i in range(windows):
            x, y = window_fn(valid, i * context, context)
            loss = float(model.loss(x, y))
            if not math.isfinite(loss) or loss < 0:
                raise ValueError('invalid length-sweep NLL')
            losses.append(loss)
            if pure:
                occupancy.append([{k: r[k] for k in ('occ_frac', 'gate_mean', 'event_density',
                    'ever_touched_fraction', 'peak_live_per_head')} for r in telemetry(model)])
        nll = sum(losses) / windows
        rows.append(dict(context=context, windows=windows, tokens_evaluated=windows*context,
                         nll=nll, perplexity=math.exp(nll), telemetry=occupancy))
    for row in rows:
        row.update(nll_delta_vs_T512=row['nll']-rows[0]['nll'],
                   ppl_ratio_vs_T512=row['perplexity']/rows[0]['perplexity'])
    return dict(status='completed', score_eligible=False, scope=SCOPE, rows=rows,
                methodology='X40 context_probe: two reset batch-1 contiguous windows per length, '
                'starts i*T on the same held-out prefix; all next-token positions scored. '
                'Token coverage differs by length. No recurrent state carried between windows.')
