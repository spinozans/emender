"""X7: scratch T16 only, precision pilot; no ladder or carry supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import types

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage3_8m import (
    build_model as base_model, digest, frozen_pools, padded_forward, finish_metrics,
)
from experiments.expressivity_tasks.run_invertible_stage5_curriculum_8m import checkpoint
from experiments.expressivity_tasks.run_invertible_stage6_truecur_8m import prune
from experiments.expressivity_tasks.eval_invertible_stage6_positions import position_counts
from experiments.expressivity_tasks.run_invertible_stage2 import metric_counts
from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage7_deepfirst_8m')
DEPTH = 16
CEILING = 500000
EVAL_EVERY = 500
NAMES = ('e97_bf16_seed0', 'e97_fp32_seed0', 'gdn2_fp32_seed0')
PRECISION = {
    'e97_bf16_seed0': 'X6 legacy: bf16 projections/gates/output/replay checkpoints/scratch/state gradients; live scan accumulator already fp32; fp32 parameters',
    'e97_fp32_seed0': 'existing fp32 state knob: fp32 initial/final state, replay checkpoints/scratch/state gradients; live scan accumulator already fp32; fixed BH1/NW4 geometry; bf16 projections/gates/output, fp32 parameters; discarded eval workspace remains legacy',
    'gdn2_fp32_seed0': 'pilot sequential adapter: bf16 projected/convolved q/k/v and raw gate logits promoted to fp32 for q/k normalization, gate transforms, delta read/write/state/readout arithmetic and autograd; recurrence output cast back to bf16 before unchanged output norm/gate/projection; fp32 parameters; stock accumulator was already fp32',
}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def fp32_gdn_recurrence(q, k, v, g, beta, scale=None, initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                        state_v_first=False, cu_seqlens=None, use_gate_in_kernel=False,
                        A_log=None, dt_bias=None, **kwargs):
    """Stock GDN delta equations in fp32, only for independent fixed-length X7 batches."""
    import torch
    import torch.nn.functional as F
    if cu_seqlens is not None or kwargs:
        raise ValueError('X7 adapter only supports the registered fixed-length stock call')
    dtype = q.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, g, beta = [a.float() for a in (q, k, v, g, beta)]
        if use_qk_l2norm_in_kernel:
            # FLA l2norm uses rsqrt(sum(x*x) + 1e-6), not F.normalize's clamp.
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        if use_gate_in_kernel:
            g = -A_log.float().exp() * F.softplus(g + dt_bias.float())
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid() * (2. if allow_neg_eigval else 1.)
        elif allow_neg_eigval:
            raise ValueError('negative eigenvalues require sigmoid beta')
        scale = k.shape[-1] ** -.5 if scale is None else scale
        state = torch.zeros(q.shape[0], q.shape[2], k.shape[-1], v.shape[-1],
                            device=q.device, dtype=torch.float32)
        if initial_state is not None:
            state = initial_state.float().transpose(-1, -2) if state_v_first else initial_state.float()
        outputs = []
        for t in range(q.shape[1]):
            state = state * g[:, t].exp()[..., None, None]
            delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
            state = state + k[:, t, :, :, None] * delta[..., None, :]
            outputs.append((state * (q[:, t] * scale)[..., None]).sum(-2))
        output = torch.stack(outputs, dim=1).to(dtype)
        final = state.transpose(-1, -2).contiguous() if state_v_first else state
        return output, final if output_final_state else None


def install_gdn_adapter(gdn):
    """Keep upstream forward bytecode; replace only its two recurrence lookups, per instance.

    A private globals copy avoids modifying FLA globally or copying its projection,
    convolution, cache and output processing. State-dict keys remain unchanged.
    """
    forward = gdn.forward.__func__
    namespace = dict(forward.__globals__)
    for name in ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'):
        if name not in namespace:
            raise ValueError('unsupported FLA forward: missing ' + name)
        namespace[name] = fp32_gdn_recurrence
    gdn.forward = types.MethodType(types.FunctionType(forward.__code__, namespace,
        forward.__name__, forward.__defaults__, forward.__closure__), gdn)


def build_model(arm, precision):
    model = base_model(arm, 61)
    if arm == 'e97':
        for layer in model.layers:
            layer.recurrent_state_precision = 'legacy' if precision == 'bf16' else 'fp32'
            # Both arms use their training recurrence at evaluation; the fp32 knob
            # rejects eager eval, and asymmetric backends would confound the pilot.
            layer.fused_inference = True
    elif precision == 'fp32':
        for layer in model.layers:
            install_gdn_adapter(layer.gdn)
    else:
        raise ValueError('unregistered arm')
    return model


def forward_batch(model, batch):
    import torch
    x, y, mask = [torch.from_numpy(a).cuda() for a in batch]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = padded_forward(model, x)
    return logits, y, mask


def evaluate(model, pool):
    import torch
    import torch.nn.functional as F
    model.eval()
    totals, loss_sum, predictions, losses = {}, 0., [], []
    with torch.no_grad():
        for start in range(0, 512, 32):
            batch = tuple(a[start:start + 32] for a in pool)
            logits, y, mask = forward_batch(model, batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite evaluation logits')
            pred = logits.argmax(-1).cpu().numpy()
            predictions.append(pred)
            per_target = F.cross_entropy(logits[mask].float(), y[mask], reduction='none')
            loss_sum += per_target.sum().item()
            losses.append(per_target.reshape(32, DEPTH).cpu().numpy())
            for key, value in metric_counts(pred, batch[1], batch[2], 61).items():
                totals[key] = totals.get(key, 0) + value
    positions = position_counts(np.concatenate(predictions), pool[1], pool[2], np.concatenate(losses))
    assert sum(p['correct'] for p in positions) == totals['correct']
    return {'aggregate': finish_metrics(totals, loss_sum, 61), 'positions': positions,
            'perfect_prefix_depth': next((p['driver_position'] - 1 for p in positions if p['correct'] != 512), DEPTH)}


def run(args):
    import torch
    import torch.nn.functional as F
    import schedulefree
    name = f'{args.arm}_{args.precision}_seed{args.seed}'
    if name not in NAMES:
        raise ValueError('unregistered run')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    sources = [__file__, 'ndm/recurrent_precision.py', 'ndm/models/e88_fla_hybrid.py',
               'ndm/models/fla_gated_delta.py', 'ndm/triton/e88_triton_forward.py',
               'ndm/triton/e88_triton_backward.py', 'ndm/triton/e88_triton_optimized.py',
               'experiments/expressivity_tasks/run_invertible_stage3_8m.py',
               'experiments/expressivity_tasks/run_invertible_stage5_curriculum_8m.py',
               'experiments/expressivity_tasks/run_invertible_stage6_truecur_8m.py',
               'experiments/expressivity_tasks/eval_invertible_stage6_positions.py',
               'experiments/expressivity_tasks/tasks/invertible_quadratic.py']
    import fla
    fla_root = Path(fla.__file__).parent
    fla_paths = ['layers/gated_deltanet.py', 'modules/l2norm.py', 'ops/gated_delta_rule/chunk.py',
                 'ops/gated_delta_rule/chunk_fwd.py', 'ops/gated_delta_rule/gate.py',
                 'ops/gated_delta_rule/fused_recurrent.py', 'ops/gated_delta_rule/wy_fast.py',
                 'ops/common/chunk_delta_h.py', 'ops/common/chunk_o.py']
    config = {**vars(args), 'name': name, 'drivers': DEPTH, 'p': 61, 'vocab': 3721,
              'depths': [DEPTH], 'initialization': 'from scratch; no checkpoint loaded',
              'ladder': False, 'carry_supervision': False, 'stagnation_kill': False,
              'safety_ceiling': CEILING, 'stop_policy': '500K updates safety ceiling only; no threshold stop',
              'eval_every': EVAL_EVERY, 'eval_examples': 512, 'batch': 32, 'lr': .001,
              'optimizer': 'AdamWScheduleFree', 'betas': [.9, .95], 'weight_decay': .01,
              'clip': 1., 'warmup': 0, 'precision_scope': PRECISION[name],
              'training': 'fresh procedural batches; frozen validation never sampled for training',
              'retention': 'X6 bounded pruning: entry, best and latest below400GiB; terminal retained',
              'padding': 'X3 right pad to16; T16 has 18 tokens, padded to32',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'host': os.uname().nodename, 'started_at': time.time(), 'torch_version': torch.__version__,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {p: digest(p) for p in sources},
              'fla_source_sha256': {str(fla_root / p): digest(fla_root / p) for p in fla_paths},
              'eval_backend': 'sequential Triton for both E97 arms (X6 eval was eager); fp32 sequential adapter for GDN2'}
    pool = frozen_pools(args.seed, DEPTH, 61)['val']
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T16_{i}': a for i, a in enumerate(pool)})
    write_json(out / 'data_receipt.json', {'sha256': digest(out / 'frozen_validation.npz'),
        'array_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in pool],
        'examples': 512, 'targets': int(pool[2].sum()), 'selection': 'exact X3/X6 T16 validation only'})
    torch.manual_seed(args.seed)
    model = build_model(args.arm, args.precision).cuda()
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    assert config['trainable_parameters'] == {'e97': 9900800, 'gdn2': 9887056}[args.arm]
    write_json(out / 'config.json', config)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=.001, betas=(.9, .95), weight_decay=.01)
    assert not opt.state
    rng, task = np.random.default_rng(args.seed), InvertibleQuadraticTask(61)
    history, checkpoints, best, entry = [], {}, None, None
    started = time.monotonic()
    with (out / 'training.jsonl').open('a', buffering=1) as log:
        for step in range(CEILING + 1):
            if step:
                opt.train()
                model.train()
                opt.zero_grad(set_to_none=True)
                logits, y, mask = forward_batch(model, task.generate_batch(32, DEPTH + 2, rng))
                loss = F.cross_entropy(logits[mask].float(), y[mask])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                log.write(json.dumps({'step': step, 'depth': DEPTH, 'loss': loss.item(),
                    'grad_norm': norm.item(), 'elapsed_s': time.monotonic() - started}) + '\n')
                del logits, loss, y, mask
            if step % EVAL_EVERY and step != CEILING:
                continue
            opt.eval()
            metrics = evaluate(model, pool)
            current = metrics['aggregate']
            is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
            cp = None
            if is_best or step % 1000 == 0 or step == CEILING:
                cp = checkpoint(out, model, opt, rng, step, DEPTH, step, config)
                checkpoints[cp['path']] = cp
                latest = cp['path']
            if is_best:
                best = {**current, 'step': step, 'checkpoint': cp}
            if step == 0:
                entry = cp['path']
            row = {'step': step, 'depth': DEPTH, **metrics, 'checkpoint': cp,
                   'best_accuracy': best['accuracy'], 'elapsed_s': time.monotonic() - started,
                   'decision': 'safety_ceiling' if step == CEILING else 'continue'}
            history.append(row)
            write_json(out / 'validation.json', history)
            write_json(out / 'checkpoints.json', checkpoints)
            print(json.dumps({'name': name, **row}), flush=True)
            prune(out, checkpoints, {entry, best['checkpoint']['path'], latest})
    write_json(out / 'result.json', {'name': name, 'status': 'complete', 'total_steps': CEILING,
        'stop_reason': 'safety_ceiling', 'best': best, 'final': history[-1],
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['e97', 'gdn2'], required=True)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], required=True)
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
