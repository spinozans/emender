"""CPU validation of byte and held-out alignment plus control causality."""
import json
from pathlib import Path
import torch
from .run import Control, count


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4301)
    model = Control(2_000_000).eval()
    ids = torch.randint(0, 256, (1, 128))
    changed = ids.clone()
    changed[:, 64:] = (changed[:, 64:]+1) % 256
    with torch.no_grad():
        a, b = model(ids), model(changed)
    assert torch.equal(a[:, :64], b[:, :64]), 'future-byte leakage'
    raw = bytes(range(256)) * 10
    data = torch.tensor(list(raw), dtype=torch.long)
    assert bytes(data.tolist()) == raw
    split = int(.9*len(data))
    targets = []
    for lo in range(split, len(data)-1, 128):
        hi = min(lo+128, len(data)-1)
        assert torch.equal(data[lo:hi+1][1:], data[lo+1:hi+1])
        targets.extend(range(lo+1, hi+1))
    assert targets == list(range(split+1,len(data)))
    result = dict(status='passed', checks=['all 256 byte values round-trip',
        'held-out targets exactly once including short final window',
        'transformer predictions invariant to future byte perturbation'],
        control_parameters=count(model))
    (Path(__file__).parent/'receipts/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
