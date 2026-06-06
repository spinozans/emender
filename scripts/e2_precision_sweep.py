#!/usr/bin/env python3
"""E2 — Precision sweep (fp64 / fp32 / bf16), SERIAL S5 eval of the e88-linear winner.

Tests H1 (serial rounding as a useful per-step nonlinearity) from
paper/review/PRECISION_NONLINEARITY_RESEARCH.md §7-E2.

What this does (REAL training + REAL eval, no mocks):
  * Builds the e88-linear CMA winner config verbatim from
    results/s5_symmetric_20260603/winners/e88-linear.args.json
    (dim=256, depth=5, n_heads=38, n_state=32, linear_state=1, use_gate=1,
     lr=0.0026571..., schedule-free AdamW, S5 permutation, seq_len=128, batch=32).
  * TRAINS one model per (seed, train_dtype) at a chosen precision.
      - fp32 : params fp32, NO autocast            (the eval-only base model)
      - bf16 : params fp32 + bf16 autocast         (matches the published recipe)
      - fp64 : params float64, NO autocast         (training infeasible on Ada;
               supported here only for completeness / short runs)
  * EVALUATES the SAME trained serial model at THREE eval precisions
    {fp64, fp32, bf16} over the length grid T in {128,256,512,1024}, 8 batches/len.
    The eval ALWAYS runs through the dtype-driven PyTorch serial recurrence
    fallback in ndm/models/e88_fla_hybrid.py (the fast CUDA/Triton paths are
    bf16-and-training only), so only the arithmetic dtype changes between arms.

Interpretation (stated in the doc):
  * acc RISES as eval precision DROPS (bf16 > fp32 > fp64): rounding helps -> H1.
  * acc flat / rises WITH precision (fp64 >= fp32 >= bf16): not rounding-driven;
    fp64-serial ~ idealized-linear proxy -> H1 disfavored.

GPUs: set CUDA_VISIBLE_DEVICES externally (this experiment uses ONLY 2,3).
One process == one (seed, train_dtype) job on one visible GPU.

Decay note (honest): the Mamba-style decay is computed in fp32 inside the E88
layer (A_log.float() / softplus(.float())) regardless of arm, then cast to the
working dtype. That fp32 decay step is CONSTANT across the three eval arms, so it
does not confound the precision comparison; the accumulating state recurrence
(where H1's per-step rounding lives) runs in the target dtype.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from ndm.models.hybrid_ladder import HybridLadderLM
from experiments.expressivity_tasks.tasks import ALL_TASKS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINNER = os.path.join(
    REPO,
    'experiments/expressivity_tasks/results/s5_symmetric_20260603/winners/e88-linear.args.json',
)

DTYPES = {'fp64': torch.float64, 'fp32': torch.float32, 'bf16': torch.bfloat16}
EVAL_DTYPES = ['fp64', 'fp32', 'bf16']
EVAL_LENGTHS = [128, 256, 512, 1024]


def build_model(p, vocab_size, device):
    model = HybridLadderLM(
        vocab_size=vocab_size,
        dim=int(p['dim']), depth=int(p['depth']),
        layer_pattern=['E88'],
        layer_kwargs=[{'linear_state': bool(p['linear_state']),
                       'use_gate': bool(p['use_gate'])}],
        n_state=int(p['n_state']), n_heads=int(p['n_heads']),
        expansion=1.0,
    ).to(device)
    return model


def evaluate(model, task, B, T, n_batches, rng, device, eval_dtype):
    """Serial eval at a fixed arithmetic dtype.

    fp64/fp32 : model is cast to that dtype, NO autocast.
    bf16      : model params stay fp32, bf16 autocast ON (matches published eval).
    """
    model.eval()
    correct = total = 0
    losses = []
    use_autocast = (eval_dtype == 'bf16')
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, mask = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inp).to(device)
            y = torch.from_numpy(tgt).to(device)
            m = torch.from_numpy(mask).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += ((preds == y) & m).sum().item()
            total += m.sum().item()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   y.view(-1), reduction='none').view_as(m)
            losses.append((loss * m).sum().item() / max(m.sum().item(), 1))
    return correct / max(total, 1), float(np.mean(losses))


def sweep_eval(model, task, batch_size, seq_len, n_batches, rng, device):
    """Eval the SAME model at all three precisions over the length grid.

    Returns {eval_dtype: {T: {acc, loss, B_eval}}}. Casts the model in-place per
    arm and restores fp32 at the end. fp32 master weights are snapshotted so the
    fp64<->fp32 round-trip is loss-free.
    """
    master = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = {}
    for ed in EVAL_DTYPES:
        if ed == 'fp64':
            model.double()
        elif ed == 'fp32':
            model.float()
        else:  # bf16 -> params fp32, autocast handles the cast
            model.float()
        out[ed] = {}
        for T in EVAL_LENGTHS:
            B_eval = batch_size
            if T > 4 * seq_len:
                B_eval = max(2, batch_size // (T // (4 * seq_len)))
            try:
                acc, loss = evaluate(model, task, B_eval, T, n_batches, rng, device, ed)
                out[ed][str(T)] = {'acc': float(acc), 'loss': float(loss),
                                   'B_eval': int(B_eval)}
                print(f"    [{ed}] T={T:>5d} (B={B_eval}): acc={acc:.4f} loss={loss:.4f}",
                      flush=True)
            except Exception as e:
                out[ed][str(T)] = {'error': f'{type(e).__name__}: {e}'}
                print(f"    [{ed}] T={T}: ERROR {type(e).__name__}: {e}", flush=True)
        # restore fp32 master before next arm so no precision is lost downstream
        model.float()
        model.load_state_dict(master)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--train_dtype', choices=['fp32', 'bf16', 'fp64'], default='fp32')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--eval_interval', type=int, default=2000)
    ap.add_argument('--n_batches', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save_ckpt', default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'CUDA required'
    device = 'cuda'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    w = json.load(open(WINNER))
    p = w['params']
    task = ALL_TASKS['s5_permutation']()
    print(f"Task: {task.name} vocab={task.vocab_size} "
          f"random_baseline={task.random_baseline_acc():.4f}", flush=True)

    model = build_model(p, task.vocab_size, device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"Pattern={model.actual_pattern} params={n_params:,} "
          f"train_dtype={args.train_dtype} steps={args.steps}", flush=True)

    train_dt = DTYPES[args.train_dtype]
    if args.train_dtype == 'fp64':
        model.double()
    elif args.train_dtype == 'fp32':
        model.float()
    # bf16: keep fp32 params, autocast in the loop
    use_autocast = (args.train_dtype == 'bf16')

    import schedulefree
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=float(p['lr']), weight_decay=0.01, betas=(0.9, 0.95))

    log = {'experiment': 'E2_precision_sweep', 'arm': 'e88-linear',
           'task': task.name, 'pattern': model.actual_pattern,
           'dim': int(p['dim']), 'depth': int(p['depth']),
           'n_heads': int(p['n_heads']), 'n_state': int(p['n_state']),
           'lr': float(p['lr']), 'linear_state': int(p['linear_state']),
           'use_gate': int(p['use_gate']), 'seed': args.seed,
           'params': n_params, 'train_dtype': args.train_dtype,
           'steps': args.steps, 'seq_len': 128, 'batch_size': 32,
           'random_baseline_acc': task.random_baseline_acc(),
           'gpu': torch.cuda.get_device_name(0),
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'train_curve': []}

    seq_len, batch = 128, 32
    t0 = time.time()
    model.train()
    optimizer.train()
    for step in range(args.steps):
        inp, tgt, mask = task.generate_batch(batch, seq_len, rng)
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            logits = model(x)
        loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   y.view(-1), reduction='none').view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            optimizer.eval()
            # quick in-loop probe at the TRAINING dtype (T=128 only)
            acc, eloss = evaluate(model, task, batch, seq_len, 4, rng, device,
                                  args.train_dtype)
            optimizer.train()
            model.train()
            el = time.time() - t0
            print(f"  step {step:>6d} loss={loss.item():.4f} "
                  f"eval_acc[{args.train_dtype}]={acc:.4f} ({el:.0f}s)", flush=True)
            log['train_curve'].append({'step': step, 'train_loss': float(loss.item()),
                                       'probe_acc': float(acc), 'elapsed_s': float(el)})

    log['train_elapsed_s'] = float(time.time() - t0)
    optimizer.eval()

    if args.save_ckpt:
        model.float()
        torch.save({'state_dict': model.state_dict(), 'config': p,
                    'seed': args.seed, 'train_dtype': args.train_dtype}, args.save_ckpt)
        print(f"Saved checkpoint -> {args.save_ckpt}", flush=True)

    # === The precision sweep: SAME model, three eval dtypes ===
    print(f"\n=== Precision sweep (serial eval @ fp64/fp32/bf16) seed={args.seed} "
          f"train_dtype={args.train_dtype} ===", flush=True)
    model.float()
    sweep = sweep_eval(model, task, batch, seq_len, args.n_batches, rng, device)
    log['precision_sweep'] = sweep
    log['sweep_elapsed_s'] = float(time.time() - t0 - log['train_elapsed_s'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(log, open(args.out, 'w'), indent=2)
    print(f"Saved -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
