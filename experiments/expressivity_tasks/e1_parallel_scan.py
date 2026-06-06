"""E1 - Parallel/associative-scan vs serial execution of e88-linear on S5.

Tests H1/H2 (serial-precision/order artifact). e88-linear is an input-dependent
LINEAR-in-state recurrence (linear_state=1):

    S_t = decay_t * S_{t-1} + outer(v_t - S_{t-1}^T k_norm_t, k_norm_t)

which is an AFFINE map of the matrix state:  S_t = A_t S_{t-1} + B_t  with
    A_t = decay_t I - k_norm_t k_norm_t^T   (n x n, shared over value columns)
    B_t = k_norm_t (x) v_t
Affine maps compose associatively, so the recurrence is a genuine Blelloch
associative scan (implemented in E88FLAHybrid._scan_recurrence /._affine_scan
as a Hillis-Steele doubling scan). This script evaluates the SAME trained
weights two ways -- serial time-loop vs associative scan -- at identical dtype,
on identical eval batches, across the length grid {128,256,512,1024}.

Modes:
  verify : build winner config (random or lightly-trained), assert serial==scan
           up to FP reassociation in fp32. Validates the scan is faithful.
  run    : train winner config on S5 (20000 steps, one seed), then dual-path
           length-extrap eval (serial + scan) at bf16 (matches winner eval) and
           fp32, on identical batches. Writes one JSON with both paths.

REAL training + eval only. No mocks.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ndm.models.hybrid_ladder import HybridLadderLM
from ndm.models.e88_fla_hybrid import E88FLAHybrid
from experiments.expressivity_tasks.tasks import ALL_TASKS


# Winner config: results/s5_symmetric_20260603/winners/e88-linear.args.json
WINNER = dict(dim=256, depth=5, n_heads=38, n_state=32, expansion=1.0,
              lr=0.0026571129141058, linear_state=1, use_gate=1)


def set_recurrence_mode(model, mode):
    """Set e88_recurrence_mode on every E88 layer. mode in {'serial','scan'}.

    'serial' -> the eager PyTorch time-loop (the path train_hybrid.py uses).
    'scan'   -> the associative-scan path.
    Both force the eager projection path so the ONLY difference is the
    state-recurrence execution order (identical weights, identical dtype).
    """
    n = 0
    for m in model.modules():
        if isinstance(m, E88FLAHybrid):
            m.e88_recurrence_mode = mode
            n += 1
    return n


def build_model(task, device):
    e88_kwargs = {'linear_state': bool(WINNER['linear_state']),
                  'use_gate': bool(WINNER['use_gate'])}
    model = HybridLadderLM(
        vocab_size=task.vocab_size,
        dim=WINNER['dim'], depth=WINNER['depth'],
        layer_pattern=['E88'],
        layer_kwargs=[e88_kwargs],
        n_state=WINNER['n_state'], n_heads=WINNER['n_heads'],
        expansion=WINNER['expansion'],
        rank=None, use_triton_e88=False,
    ).to(device)
    return model


def forward_mode(model, x, mode, autocast, device):
    set_recurrence_mode(model, mode)
    use_ac = (device == 'cuda') and autocast
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_ac):
            logits = model(x)
    return logits


def eval_on_batches(model, task, batches, mode, autocast, device):
    """Evaluate a fixed list of (inp,tgt,mask) batches in a given recurrence
    mode + dtype. Returns (acc, loss, logits_first_batch)."""
    model.eval()
    correct = total = 0
    losses = []
    first_logits = None
    for i, (inp, tgt, mask) in enumerate(batches):
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        logits = forward_mode(model, x, mode, autocast, device)
        if i == 0:
            first_logits = logits.float().cpu()
        preds = logits.argmax(dim=-1)
        correct += ((preds == y) & m).sum().item()
        total += m.sum().item()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               y.view(-1), reduction='none').view_as(m)
        losses.append((loss * m).sum().item() / max(m.sum().item(), 1))
    return correct / max(total, 1), float(np.mean(losses)), first_logits


def dual_eval(model, task, seq_len, eval_lengths, n_batches, device,
              data_seed=20260604):
    """For each T and each dtype, run serial and scan on IDENTICAL batches.
    Returns a dict keyed by dtype_tag -> T -> metrics."""
    out = {}
    for autocast, tag in [(True, 'bf16'), (False, 'fp32')]:
        out[tag] = {}
        for T in eval_lengths:
            B_eval = 32
            if T > 4 * seq_len:
                B_eval = max(2, 32 // (T // (4 * seq_len)))
            # Fixed batches: identical inputs for both paths.
            rng = np.random.default_rng(data_seed + T)
            batches = [task.generate_batch(B_eval, T, rng) for _ in range(n_batches)]

            acc_s, loss_s, lg_s = eval_on_batches(
                model, task, batches, 'serial', autocast, device)
            acc_p, loss_p, lg_p = eval_on_batches(
                model, task, batches, 'scan', autocast, device)

            # Logit agreement on the first batch.
            diff = (lg_s - lg_p).abs()
            denom = lg_s.abs().mean().clamp_min(1e-8)
            pred_agree = (lg_s.argmax(-1) == lg_p.argmax(-1)).float().mean().item()
            out[tag][str(T)] = {
                'B_eval': int(B_eval),
                'n_batches': int(n_batches),
                'acc_serial': float(acc_s),
                'acc_scan': float(acc_p),
                'acc_delta': float(acc_s - acc_p),
                'loss_serial': float(loss_s),
                'loss_scan': float(loss_p),
                'mean_abs_logit_diff': float(diff.mean().item()),
                'max_abs_logit_diff': float(diff.max().item()),
                'rel_logit_diff': float((diff.mean() / denom).item()),
                'pred_agreement': float(pred_agree),
            }
            print(f"  [{tag}] T={T:>5d} B={B_eval:>2d}  serial_acc={acc_s:.4f}  "
                  f"scan_acc={acc_p:.4f}  d={acc_s-acc_p:+.4f}  "
                  f"|dlogit|={diff.mean().item():.3e}  agree={pred_agree:.4f}",
                  flush=True)
    return out


def cmd_verify(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    task = ALL_TASKS['s5_permutation']()
    model = build_model(task, device)
    n_e88 = sum(isinstance(m, E88FLAHybrid) for m in model.modules())
    print(f"Built winner model: {n_e88} E88 layers, "
          f"{sum(p.numel() for p in model.parameters()):,} params", flush=True)

    rng = np.random.default_rng(123)
    worst = 0.0
    for T in [128, 256, 512]:
        inp, tgt, mask = task.generate_batch(8, T, rng)
        x = torch.from_numpy(inp).to(device)
        # fp32 (no autocast): scan must equal serial up to FP reassociation.
        lg_serial = forward_mode(model, x, 'serial', autocast=False, device=device).float()
        lg_scan = forward_mode(model, x, 'scan', autocast=False, device=device).float()
        diff = (lg_serial - lg_scan).abs()
        rel = (diff.mean() / lg_serial.abs().mean().clamp_min(1e-8)).item()
        agree = (lg_serial.argmax(-1) == lg_scan.argmax(-1)).float().mean().item()
        worst = max(worst, diff.max().item())
        print(f"  fp32 T={T:>4d}: mean|d|={diff.mean().item():.3e}  "
              f"max|d|={diff.max().item():.3e}  rel={rel:.3e}  "
              f"pred_agree={agree:.4f}", flush=True)
    # fp32 affine scan should match serial to ~1e-4 (rank-1 vs matmul order).
    ok = worst < 1e-2
    print(f"\nVERIFY {'PASS' if ok else 'FAIL'}: worst max|logit diff| (fp32) = {worst:.3e}",
          flush=True)
    return 0 if ok else 1


def cmd_run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    task = ALL_TASKS['s5_permutation']()
    print(f"Task: {task.name}, vocab_size={task.vocab_size}", flush=True)

    model = build_model(task, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}  pattern={model.actual_pattern}", flush=True)

    import schedulefree
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=WINNER['lr'], weight_decay=0.01, betas=(0.9, 0.95))

    log = {'task': task.name, 'seed': args.seed, 'params': n_params,
           'winner': WINNER, 'steps_total': args.steps,
           'seq_len': args.seq_len, 'batch_size': args.batch_size,
           'random_baseline_acc': task.random_baseline_acc(), 'train_log': []}

    t0 = time.time()
    eval_interval = max(50, args.steps // 20)
    model.train()
    set_recurrence_mode(model, 'serial')  # training uses the standard path
    optimizer.train()
    for step in range(args.steps):
        inp, tgt, mask = task.generate_batch(args.batch_size, args.seq_len, rng)
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model(x)
        loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   y.view(-1), reduction='none').view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == args.steps - 1:
            optimizer.eval()
            model.eval()
            acc = correct = total = 0
            with torch.no_grad():
                for _ in range(4):
                    i2, t2, m2 = task.generate_batch(args.batch_size, args.seq_len, rng)
                    xx = torch.from_numpy(i2).to(device)
                    yy = torch.from_numpy(t2).to(device)
                    mm = torch.from_numpy(m2).to(device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                        lg = model(xx)
                    correct += ((lg.argmax(-1) == yy) & mm).sum().item()
                    total += mm.sum().item()
            acc = correct / max(total, 1)
            optimizer.train()
            model.train()
            set_recurrence_mode(model, 'serial')
            el = time.time() - t0
            print(f"  step {step:>5d}  train_loss={loss.item():.4f}  "
                  f"eval_acc={acc:.4f}  ({el:.0f}s)", flush=True)
            log['train_log'].append({'step': step, 'train_loss': float(loss.item()),
                                     'eval_acc': float(acc), 'elapsed_s': float(el)})

    log['train_elapsed_s'] = float(time.time() - t0)
    print(f"\nTraining done in {log['train_elapsed_s']:.0f}s. Dual-path eval:", flush=True)

    optimizer.eval()
    model.eval()
    log['dual_eval'] = dual_eval(
        model, task, args.seq_len, args.eval_lengths,
        args.eval_lengths_n_batches, device)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.label}.json')
    json.dump(log, open(out_path, 'w'), indent=2)
    print(f"Saved to {out_path}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('verify')
    pr = sub.add_parser('run')
    pr.add_argument('--seed', type=int, required=True)
    pr.add_argument('--steps', type=int, default=20000)
    pr.add_argument('--seq_len', type=int, default=128)
    pr.add_argument('--batch_size', type=int, default=32)
    pr.add_argument('--eval_lengths', type=int, nargs='+', default=[128, 256, 512, 1024])
    pr.add_argument('--eval_lengths_n_batches', type=int, default=8)
    pr.add_argument('--label', required=True)
    pr.add_argument('--output_dir',
                    default='experiments/expressivity_tasks/results/e1_parallel_scan')
    args = ap.parse_args()
    if args.cmd == 'verify':
        sys.exit(cmd_verify(args))
    else:
        sys.exit(cmd_run(args))


if __name__ == '__main__':
    main()
