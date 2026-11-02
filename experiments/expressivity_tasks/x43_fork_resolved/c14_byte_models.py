"""Construct registered tier-1 byte heads on CPU; no training and no GPU lease."""
import json
from pathlib import Path
import sys
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c11 import Boosted
from experiments.expressivity_tasks.x43_fork_resolved.c6 import count

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def main():
    from .c14_documents import sha
    rows = []; target = 2_000_000
    for arm in ('emender', 'transformer'):
        torch.manual_seed(3401)
        model = Boosted(arm, target)
        if arm == 'emender':
            target = count(model.head.base)
        rows.append(dict(arm=arm, modes=['dense_zero_input', 'wikipedia_dropout'],
                         base_parameters=count(model.head.base), parameters=count(model),
                         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         shapes={k:list(v.shape) for k,v in model.named_parameters()}))
    sources = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().is_relative_to(REPO/'experiments') and Path(path).suffix == '.py':
            path = Path(path).resolve()
            sources[str(path.relative_to(REPO))] = sha(path)
    sources[str(Path(__file__).resolve().relative_to(REPO))] = sha(__file__)
    result = dict(status='registered', device='cpu', seed=3401, training_started=False,
                  torch=torch.__version__, models=rows, sources=sources)
    (ROOT/'receipts/c14_byte/model_shapes.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({r['arm']:r['parameters'] for r in rows}))


if __name__ == '__main__':
    main()
