"""Matched X28E stage-3 ladder fine-tune on the X39 B64 control checkpoint.

Control arm of the registered ignition-v1 depth probe. The protocol mirrors
experiments/expressivity_tasks/x40_emender_lm/depth.py exactly (30000
updates, AdamW lr 1e-4 wd 1e-4, B64 permutation epochs dropping the short
remainder, seed 0, validation every 500 with latest-best selection,
final-only binary margin BCE at the EOS position, S=2944 slot budgets), with
two identified deviations forced by the control architecture:

  * the model is the X39 exact-exposure B64 hybrid control checkpoint
    (x39_memory_usage/runs/control_b64/latest.pt, held-out PPL 6.8564176),
    loaded by sha256-verified checkpoint identity — there is no X40
    checkpoint, and none is created here;
  * the replay-graph check is the X31 presence check used by the X39
    control (replay_present) instead of the pure stack's 24-node count.

The registration receipt (depth_control_registration.json) is written
before any fine-tune compute. Launch only through run_leased.sh.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..x28e_memory_head.generators import stage3_ladder as ladder
from ..x34_lm_pilot.cli import replay_present
from ..x36_tape_perf.model import LM
from .runner import write

ROOT = Path(__file__).resolve().parent
X39 = ROOT.parent / 'x39_memory_usage'
CHECKPOINT = X39 / 'runs' / 'control_b64' / 'latest.pt'
X28E = ROOT.parent / 'results_x28e_memory_head'
UPDATES = 30000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = [Path(__file__), ROOT / 'depth.py',
              ROOT.parent / 'x28e_memory_head/generators/stage3_ladder.py',
              ROOT.parent / 'x36_tape_perf/model.py']
    return {str(p.relative_to(ROOT.parent.parent)): sha(p) for p in paths}


class ControlProbe(nn.Module):
    """Mirror of depth.Probe on the hybrid control LM."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        # X28E stage-local scalar-token embedding; no BPE merging of a/b.
        self.lm.embed = nn.Embedding(5, 256).cuda()
        self.binary = nn.Linear(256, 2).cuda()
        cell = lm.blocks[2].cell
        assert cell.cfg.n_heads == 4 and cell.cfg.budget == 2944, \
            f'control tape cell must be the trained S=2944 4-head cell, got {cell.cfg}'

    def features(self, ids):
        x = self.lm.embed(ids)
        for block in self.lm.blocks:
            x, _ = block(x, ids)
        return self.lm.norm(x)

    def forward(self, ids, lengths):
        features = self.features(ids)
        return self.binary(features[torch.arange(len(ids), device=ids.device), lengths - 1])


@torch.no_grad()
def evaluate(probe, pool):
    probe.eval()
    pred = []
    xs, ys, lengths = pool[:3]
    for start in range(0, len(xs), 8):
        ln = lengths[start:start + 8]
        ids = torch.from_numpy(xs[start:start + 8, :int(ln.max())].copy()).cuda()
        logits = probe(ids, torch.from_numpy(ln).cuda())
        pred.extend((logits[:, 1] > logits[:, 0]).long().cpu().tolist())
    return ladder.binary_report(ys, np.asarray(pred))


def registration(held):
    x28e_path = X28E / 'e97_native_seed0' / 'final_eval.json'
    x28e = json.loads(x28e_path.read_text())
    return dict(
        status='registered_before_control_fine_tune',
        purpose='Matched-transformer head-to-head arm for the registered X40 v1 '
                'depth probe (the paper depth-comparability receipt).',
        physical_gpu=held,
        registered_prediction=dict(
            text='Predict pure model balanced accuracy>.6 at4x and9x where the '
                 'fine-tuned transformer <=.55. Honest null: no separation, '
                 'chance OOD accuracy, or collapse despite in-range learning.',
            source='x40_emender_lm/RECEIPTS.md depth-probe paragraph, registered '
                   'before v1 execution; sha256=' + sha(ROOT / 'RECEIPTS.md'),
            subject='the ignition-v1 pure model arm (depth.json, produced by the '
                    'v1 run process; not retrained here)'),
        control_arm_role='No new score prediction is registered for the control '
            'arm itself; it exists to satisfy the registration requirement of an '
            'identified fine-tuned transformer receipt under the exact update '
            'protocol. The X28E precedent for attention-native arms is OOD '
            'collapse (e97_native 4x/9x = .5/.5, accept recall 0).',
        protocol=dict(
            updates=UPDATES, batch=64,
            optimizer='AdamW lr 1e-4 weight_decay 1e-4',
            permutations='epoch permutations dropping short remainder, rng '
                        'np.random.default_rng(0), torch.manual_seed(0)',
            supervision='final-only binary margin BCE at the EOS position',
            head='new Embedding(5,256) replacing the LM embed; Linear(256,2) '
                 'on the final hidden state; whole model fine-tuned',
            slot_budgets='S=2944 per head, native to the control tape cell '
                         '(matched to the pure probe)',
            validation='every 500 updates on the fixed validation pool, '
                       'latest-best (>=) selection',
            deviations=[ 'matched model is the X39 exact-exposure B64 hybrid '
                          'control (tape_lm: five attention blocks + one X31 '
                          'tape cell at block 2), not an X40 pure stack',
                        'replay check is X31 replay_present, not the pure '
                          'stack 24-node count']),
        checkpoint_identity=dict(
            path=str(CHECKPOINT.relative_to(ROOT.parent.parent)),
            sha256=sha(CHECKPOINT), step=3052, tokens=100007936,
            held_out_ppl=6.8564175640,
            x39_training_receipt='x39_memory_usage/receipts/'
                                 'part6_control_training.json (status completed)'),
        comparators=dict(
            x28e_e97_native=dict(
                path=str(x28e_path.relative_to(ROOT.parent.parent)),
                sha256=sha(x28e_path), selected_update=x28e['selected_update'],
                bins={k: x28e[k]['report']['balanced_accuracy']
                      for k in ladder.BIN_ORDER}),
            pure_arm='experiments/expressivity_tasks/x40_emender_lm/receipts/'
                     'ignition_v1/depth.json (in progress in the v1 run process)'),
        sources=fingerprint())


def main():
    held = os.environ.get('GPU_LEASE_HELD')
    if held not in ('0', '1', '5', '6') or os.environ.get('CUDA_VISIBLE_DEVICES') != held:
        raise RuntimeError('single broker lease required on 0/1/5/6')
    os.environ['X40_RECEIPT_DIR'] = str(ROOT / 'receipts' / 'ignition_v1')
    write('depth_control_registration.json', registration(held))
    training = json.loads((X39 / 'receipts' / 'part6_control_training.json').read_text())
    saved = torch.load(CHECKPOINT, map_location='cuda', weights_only=False)
    assert training['status'] == 'completed' and training['step'] == 3052
    assert saved['step'] == 3052 and saved['fingerprint'] == training['fingerprint'], \
        'checkpoint does not match the completed X39 control training receipt'
    vocab = saved['model']['embed.weight'].shape[0]
    model = LM('tape_lm', vocab).cuda()
    model.load_state_dict(saved['model'])
    del saved
    torch.manual_seed(0)
    probe = ControlProbe(model)
    train = ladder.ladder_train_pool()
    val = ladder.ladder_validation_pool()
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=1e-4)
    best = -1.
    weights = None
    selected = None
    rows = []
    update = 0
    while update < UPDATES:
        permutation = rng.permutation(len(train[0]))
        for start in range(0, len(permutation) - 63, 64):
            idx = permutation[start:start + 64]
            ln = train[2][idx]
            ids = torch.from_numpy(train[0][idx, :int(ln.max())].copy()).cuda()
            labels = torch.from_numpy(train[1][idx]).cuda()
            probe.train()
            opt.zero_grad(set_to_none=True)
            logits = probe(ids, torch.from_numpy(ln).cuda())
            loss = F.binary_cross_entropy_with_logits(logits[:, 1] - logits[:, 0],
                                                     labels.float())
            replay_present(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            update += 1
            if update % 500 == 0:
                report = evaluate(probe, val)
                metric = report['balanced_accuracy']
                rows.append(dict(update=update, report=report))
                if metric >= best:
                    best = metric
                    selected = update
                    weights = {k: v.detach().cpu().clone()
                               for k, v in probe.state_dict().items()}
                write('depth_control_validation.json', dict(updates=update, validation=rows))
                print(json.dumps(dict(update=update, balanced_accuracy=metric)), flush=True)
            if update >= UPDATES:
                break
    probe.load_state_dict(weights)
    bins = {name: evaluate(probe, ladder.ladder_bin_pool(name)) for name in ladder.BIN_ORDER}
    x28e_path = X28E / 'e97_native_seed0' / 'final_eval.json'
    x28e = json.loads(x28e_path.read_text())
    write('depth_control.json', dict(
        status='completed', updates=UPDATES, selected_update=selected, bins=bins,
        supervision='final EOS binary margin BCE', budget=2944,
        model='X39 exact-exposure B64 hybrid control (tape_lm), checkpoint '
              'sha256=' + sha(CHECKPOINT),
        train_hash=ladder.ladder_hash('train', *train[:3]),
        validation_trajectory_best=best,
        comparison=dict(
            x28e_e97_native_fine_tuned_baseline={
                k: x28e[k]['report']['balanced_accuracy'] for k in ladder.BIN_ORDER},
            pure_ignition_v1_arm='pending: depth.json is written by the still-running '
                                 'v1 process; this control receipt is complete and '
                                 'the head-to-head completes when that file lands'),
        honest_null_note='Nulls are publishable: if the pure arm shows no '
            'separation, chance OOD accuracy, or collapse despite in-range '
            'learning, the registered prediction fails and stays failed.'))


if __name__ == '__main__':
    main()
