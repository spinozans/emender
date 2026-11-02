"""X43 isolated policy-mask probe; upstream model and kernels are unchanged."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from ..x42_pile_scale import velocity_candidate as vc
from .run import ROOT, count, logits
from .decompose import summarize

OUT = ROOT/'receipts/part3'
ARMS = ('baseline', 'bernoulli_half', 'bernoulli_chi', 'fork')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@contextmanager
def masked_writes(mask, audit=None):
    """Keep patch alive through checkpoint recomputation and backward."""
    original = vc.scan

    def scan(k, q, v, beta, pop, append, cfg, coef, wire, match, error, policy):
        if mask.shape != wire.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('mask must be bool [batch,time] aligned to consumed bytes')
        wire = wire.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        result = original(k, q, v, beta, pop, append, cfg, coef, wire, match, error, policy)
        if audit is not None:
            trace = result[0][2]
            events = trace.case != 0
            forbidden = events & ~mask.unsqueeze(-1)
            audit.append(dict(events=int(events.sum()), positions=events.numel(),
                              forbidden=int(forbidden.sum())))
            if forbidden.any():
                raise AssertionError('write at forbidden position')
        return result
    with patch.object(vc, 'scan', scan):
        yield


def make_model():
    torch.manual_seed(4301)
    return vc.Candidate(vc.Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.),
                        vocab=256, d=256, layers=2, target=2_000_000).cuda()


def metrics(nll, correct, pos, fork):
    return dict(overall_nll=float(nll.mean(dtype=np.float64)),
                accuracy=float(correct.mean()), **summarize(nll, correct, fork[pos]))


def evaluate(model, data, fork, arm, chi, name):
    model.eval()
    rng = torch.Generator(device='cuda').manual_seed(4304)
    split = int(.9*len(data))
    losses, correct, positions, telemetry, audits = [], [], [], [], []
    eligible, inputs = 0, 0
    with torch.no_grad():
        for lo in range(split, len(data)-1, 128):
            hi = min(lo+128, len(data)-1)
            ix = torch.arange(lo, hi, device='cuda')[None]
            mask = get_mask(arm, ix, fork, chi, rng)
            eligible += int(mask.sum()); inputs += mask.numel()
            with masked_writes(mask, audits):
                pred = logits(model, data[lo:hi][None], 'emender')[0]
            truth = data[lo+1:hi+1]
            losses.append(F.cross_entropy(pred, truth, reduction='none').cpu().numpy())
            correct.append((pred.argmax(-1) == truth).cpu().numpy())
            positions.append(np.arange(lo+1, hi+1))
            telemetry.append(dict(inputs=hi-lo, layers=[b.cell.telemetry for b in model.blocks]))
    nll, acc, pos = map(np.concatenate, (losses, correct, positions))
    assert np.array_equal(pos, np.arange(split+1, len(data)))
    np.savez_compressed(OUT/f'{name}_per_position.npz', position=pos, nll=nll, correct=acc)
    f = fork.cpu().numpy()
    midpoint = len(pos)//2
    result = dict(full=metrics(nll, acc, pos, f),
                  final_test=metrics(nll[midpoint:], acc[midpoint:], pos[midpoint:], f),
                  eligibility_density=eligible/inputs,
                  forbidden_writes=sum(a['forbidden'] for a in audits))
    # Weight by window length, independently for every layer/head.
    result['telemetry'] = [[{key: sum(r['inputs']*r['layers'][l][h][key] for r in telemetry)/inputs
                            for key in ('event_density','gate_mean','occ_frac','prewrite_occ_frac',
                                        'ever_touched_fraction')}
                           for h in range(2)] for l in range(2)]
    dump(OUT/f'{name}_evaluation.json', result)
    return result


def get_mask(arm, ix, fork, chi, rng):
    if arm == 'fork':
        return fork[ix]
    if arm == 'baseline':
        return torch.ones_like(ix, dtype=torch.bool)
    return torch.rand(ix.shape, device=ix.device, generator=rng) < (.5 if arm == 'bernoulli_half' else chi)


def validate(data, fork):
    model = make_model().eval()
    ids = data[:16][None]
    ones = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        direct = logits(model, ids, 'emender')
        with masked_writes(ones):
            same = logits(model, ids, 'emender')
        torch.testing.assert_close(direct, same, rtol=0, atol=0)
        audit = []
        with masked_writes(~ones, audit):
            zero = logits(model, ids, 'emender')
        assert sum(a['events'] for a in audit) == 0 and torch.isfinite(zero).all()
        audit = []
        alternating = (torch.arange(16, device='cuda') % 2 == 0)[None]
        with masked_writes(alternating, audit):
            logits(model, ids, 'emender')
        assert sum(a['forbidden'] for a in audit) == 0
    model.train()
    mask = fork[torch.arange(16, device='cuda')[None]]
    with masked_writes(mask):
        loss = model.loss(ids, data[1:17][None]); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(not p.requires_grad for b in model.blocks for module in
               (b.cell.W_write,b.cell.W_pop,b.cell.W_append) for p in module.parameters())
    dump(OUT/'tests.json', dict(status='passed', checks=[
        'all-true mask bitwise equals unchanged forward', 'all-false mask has zero write events',
        'alternating mask never writes at forbidden positions',
        'masked checkpoint backward finite', 'management parameters remain frozen',
        'mask indexed by consumed byte; eval labels indexed by predicted byte']))
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.register:
        paths = [ROOT/'PREREGISTRATION_PART3.md', Path(__file__),
                 ROOT/'run_write_gate_leased.sh', ROOT/'input.txt',
                 ROOT/'receipts/part1c/labels.npz',
                 ROOT/'receipts/part1c/decomposition.json',
                 ROOT/'emender_per_position.npz', ROOT/'transformer_per_position.npz']
        dump(OUT/'registration.json', dict(registered_unix=time.time(),
             artifacts={str(p.relative_to(ROOT)):sha(p) for p in paths},
             training_seconds=180, matched_step=1000, arms=list(ARMS)))
        return
    registration = json.loads((OUT/'registration.json').read_text())
    for path, digest in registration['artifacts'].items():
        assert sha(ROOT/path) == digest, path
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4'), 'broker lease required'
    torch.set_num_threads(1)
    raw = (ROOT/'input.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    data = torch.tensor(list(raw), dtype=torch.long, device='cuda')
    fork_np = np.load(ROOT/'receipts/part1c/labels.npz')['fork']
    assert sha(ROOT/'receipts/part1c/labels.npz') == '294b56a92e38aa723ff2867afac2c25dd0d980c3ef4e548ef1fb3833b449bd35'
    fork = torch.tensor(fork_np, device='cuda')
    split = int(.9*len(data)); chi = float(fork_np[:split].mean())
    validate(data, fork)
    if args.validate_only:
        return
    rows = []
    for arm in ARMS:
        model = make_model().train()
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=.01)
        rng = torch.Generator(device='cuda').manual_seed(4301)
        mask_rng = torch.Generator(device='cuda').manual_seed(4303)
        steps, curve, train_eligible = 0, [], torch.zeros((), device='cuda')
        start = time.monotonic()
        while time.monotonic()-start < 180:
            starts = torch.randint(0, split-128, (16,), device='cuda', generator=rng)
            ix = starts[:,None]+torch.arange(129, device='cuda')
            batch = data[ix]
            mask = get_mask(arm, ix[:,:-1], fork, chi, mask_rng)
            train_eligible += mask.sum()
            opt.zero_grad(set_to_none=True)
            with masked_writes(mask):
                loss = model.loss(batch[:,:-1], batch[:,1:])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); steps += 1
            if steps == 1000:
                # CPU copy is charged to the same time budget in every arm.
                torch.save(model.state_dict(), OUT/f'{arm}_step1000.pt')
            if steps == 1 or steps % 500 == 0:
                curve.append(dict(step=steps, seconds=time.monotonic()-start, nll=float(loss.detach())))
                print(arm, curve[-1], flush=True)
        torch.cuda.synchronize()
        seconds = time.monotonic()-start
        torch.save(model.state_dict(), OUT/f'{arm}.pt')
        row = dict(arm=arm, steps=steps, train_seconds=seconds,
                   tokens_per_second=steps*16*128/seconds, trainable_parameters=count(model),
                   training_eligibility_density=float(train_eligible)/(steps*16*128), curve=curve,
                   checkpoint_sha256=sha(OUT/f'{arm}.pt'))
        # Persist training completion before first evaluation.
        dump(OUT/f'{arm}_training.json', row)
        row['evaluation'] = evaluate(model, data, fork, arm, chi, arm)
        if steps >= 1000:
            model.load_state_dict(torch.load(OUT/f'{arm}_step1000.pt', weights_only=True))
            row['matched_step1000'] = evaluate(model, data, fork, arm, chi, f'{arm}_step1000')
        rows.append(row)
        dump(OUT/'results.json', dict(registration_sha256=sha(OUT/'registration.json'),
             gpu=os.environ['GPU_LEASE_HELD'], training_chi=chi, models=rows))
        del model, opt
        torch.cuda.empty_cache()
    comparisons = []
    for arm in ('emender','transformer'):
        ev = np.load(ROOT/f'{arm}_per_position.npz')
        mid = len(ev['position'])//2
        comparisons.append(dict(arm=arm, full=metrics(ev['nll'],ev['correct'],ev['position'],fork_np),
            final_test=metrics(ev['nll'][mid:],ev['correct'][mid:],ev['position'][mid:],fork_np)))
    dump(OUT/'retained_comparisons.json', comparisons)


if __name__ == '__main__':
    main()
