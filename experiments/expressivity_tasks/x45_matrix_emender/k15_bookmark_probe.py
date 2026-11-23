"""Short, single-card checkpoint integration probe. Output goes outside the checkout."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from . import k15_probe_composition as incumbent
from .velocity_ladder import _build_k10
from .k15_composition import convert_model_to_composition
from .k15_bookmarks import BookmarkK15
from .protocol_correction import _require_lease
from ..x42_pile_scale.lattice import Stream, flagship_identity, evaluate


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(out):
    assert _require_lease() in ('0', '1') and os.environ['OMP_NUM_THREADS'] == '4'
    state = Path('/mnt/nvme2n1/erikg/xsa_anchor_replays_v1/k15/final.pt')
    exe = Path.home()/'sxgc/xsa/target/debug/xsa'
    identity = flagship_identity()
    stream = Stream(identity['source'], identity['manifest'])
    anchor = json.loads(state.with_name('anchor.json').read_text())
    assert digest(state) == anchor['sha256'] == 'accb73ce3ad0cbe5b88d439dad8073d17a0ab3ec25a9f3f52b6cfdf7f968ecd0'
    assert digest(incumbent.__file__) == anchor['engine_sha256'] == 'e769a26157ca3da36569258289476f0c1e74e9b2e33e3d187e8b185076349d2c'
    assert digest(exe) == '87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    base, *_ = _build_k10(dict(incumbent._tuned_winner()), 3401)
    base = convert_model_to_composition(base).cuda()
    base.load_state_dict(torch.load(state, map_location='cpu', weights_only=True))
    net = BookmarkK15(base, exe).cuda()
    def emit(**fields):
        print(json.dumps(fields), flush=True)
    emit(anchor=anchor, identity=identity, exe=digest(exe), params=sum(p.numel() for p in net.parameters()))
    score = evaluate(base, stream, 'matrix_ops', 'calibration')
    emit(anchor_evaluation=score)
    assert score['score'] == anchor['anchor'] == 1.3811104467936925
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-6, fused=True)
    used = set()
    for doc, start, x, y in stream.windows('train', context=128):
        if len(x) != 128 or doc in used:
            continue
        used.add(doc)
        ids, targets = (torch.tensor(row, device='cuda')[None] for row in (x,y))
        step = len(used)
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if step == 1:
                with torch.no_grad():
                    assert torch.equal(net(ids, doors=False)[0], base.head(base.features(ids)))
            forced = [(0,1),(1,0),(0,2),(1,2),(0,3),(0,1),(1,2)] if step == 1 else None
            logits, trace, probs, writes = net(ids, forced=forced, audit=step==1)
            loss = F.cross_entropy(logits[:,32:].flatten(0,1), targets[:,32:].flatten())
            if step == 1:
                writes[0].retain_grad()
                objective = loss
            else:
                # Score-function estimator: decrease expected NLL. Constant baseline.
                objective = loss + (loss.detach()-2) * torch.stack(probs).sum()
        opt.zero_grad(); objective.backward()
        delayed = float(writes[0].grad.norm()) if step == 1 else None
        if step == 1:
            assert delayed > 0 and torch.isfinite(writes[0].grad).all()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize(); elapsed = time.perf_counter()-t
        emit(step=step, doc=doc, start=start, bytes_sha=hashlib.sha256(bytes(x.tolist()+[int(y[-1])])).hexdigest(),
             forced=bool(forced), delayed_gradient=delayed, norm=float(grad), nll=float(loss.detach()),
             calls=[(r['slot'],r['op'],len(r['calls'])) for r in trace], seconds=elapsed,
             tokens_per_second=128/elapsed)
        if forced:
            (out/'forced_trace.json').write_text(json.dumps(trace,indent=2))
        if step == 4:
            path=out/'integrated.pt';tmp=path.with_suffix('.partial')
            torch.save(net.state_dict(),tmp);os.replace(tmp,path);emit(checkpoint=digest(path))
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            net.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                online, chosen, _, _ = net(ids, audit=True)
                differences = []
                for end in (17, 33, 49, 65, 81, 97, 113):
                    prefix, events, _, _ = net(ids[:,:end], audit=True)
                    assert events == chosen[:len(events)]
                    differences.append(float((online[:,end-1]-prefix[:,-1]).abs().max()))
                emit(prefix_replay_max_abs=differences, greedy_trace=[(e['slot'], e['op']) for e in chosen])
                assert max(differences) <= 0.0625  # one bf16 ulp around logit magnitude 8
                for doors in (False, True):
                    net(ids, doors=doors);torch.cuda.synchronize();t=time.perf_counter()
                    for _ in range(8):
                        net(ids, doors=doors)
                    torch.cuda.synchronize();emit(forward_tokens_s=1024/(time.perf_counter()-t),doors=doors)
            return
    raise RuntimeError('insufficient independent full windows')

if __name__ == '__main__':
    main(Path(sys.argv[1]))
