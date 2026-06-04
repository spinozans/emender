"""Train a HybridLadderLM (per-layer architecture) on a task. Mirrors
train_task.py but uses HybridLadderLM directly so we can pass layer_pattern.

Usage:
    python train_hybrid.py --task parity --layer_pattern E88 fla-gdn \\
        --dim 128 --depth 4 --steps 500 --label hybrid_parity
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ndm.models.hybrid_ladder import HybridLadderLM
from experiments.expressivity_tasks.tasks import ALL_TASKS


def evaluate(model, task, B, T, n_batches, rng, device):
    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, mask = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inp).to(device)
            y = torch.from_numpy(tgt).to(device)
            m = torch.from_numpy(mask).to(device)
            use_autocast = device == 'cuda' and not getattr(model, 'disable_autocast', False)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += ((preds == y) & m).sum().item()
            total += m.sum().item()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                    y.view(-1), reduction='none').view_as(m)
            losses.append((loss * m).sum().item() / max(m.sum().item(), 1))
    return correct / max(total, 1), float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(ALL_TASKS.keys()))
    ap.add_argument('--layer_pattern', nargs='+', required=True,
                    help='List of layer levels, e.g. E88 fla-gdn')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--n_state', type=int, default=16)
    ap.add_argument('--rank', type=int, default=None)
    ap.add_argument('--expansion', type=float, default=1.0)
    ap.add_argument('--use_triton_e88', action='store_true',
                    help='Route E88 layers through the Triton fwd/bwd kernels.')
    # E88 structural (BL-1) knobs. Default None = use the E88 layer's own
    # constructor defaults (unchanged behavior for run_separation_suite and the
    # §6 probes). When set, they are forwarded per-layer to E88-family layers so
    # the S5-symmetric CMA-ES can search them (paper/review/S5_SYMMETRIC_PROTOCOL.md
    # §D.2: linear_state and use_gate become searched knobs for E88).
    ap.add_argument('--linear_state', type=int, default=None, choices=[0, 1],
                    help='E88 state nonlinearity: 0=tanh, 1=linear. Default: '
                         "layer default (tanh). Forwarded only to E88-family layers.")
    ap.add_argument('--use_gate', type=int, default=None, choices=[0, 1],
                    help='E88 output gate: 0=off, 1=on. Default: HybridLadderLM '
                         'default. Forwarded only to E88-family layers.')
    ap.add_argument('--decay_mode', type=str, default=None,
                    choices=['mamba', 'simple', 'none', 'constant'],
                    help='E88 recurrence decay mode. mamba=input-dependent '
                         'exp decay (default); constant=learned per-head '
                         'constant (input-INDEPENDENT transition, eigenvalues '
                         'in (0,1)); none=identity (eigenvalue 1); '
                         'simple=input-dependent sigmoid. Forwarded only to '
                         'E88-family layers. Used by the E5 input-dependence '
                         'ablation (paper/review/E5_ABLATE_INPUTDEP.md).')
    ap.add_argument('--m2rnn_q_heads', type=int, default=None)
    ap.add_argument('--m2rnn_k_heads', type=int, default=None)
    ap.add_argument('--m2rnn_v_heads', type=int, default=None)
    ap.add_argument('--m2rnn_f_heads', type=int, default=None)
    ap.add_argument('--m2rnn_g_heads', type=int, default=None)
    ap.add_argument('--m2rnn_weight_heads', type=int, default=None)
    ap.add_argument('--m2rnn_normalize_qk', action='store_true',
                    help='L2-normalize M2RNN query/key vectors before recurrence')
    ap.add_argument('--m2rnn_no_residual', action='store_true',
                    help='Disable M2RNN D*v direct residual path')
    ap.add_argument('--m2rnn_freeze_state_weight', action='store_true',
                    help='Keep M2RNN state_weight fixed at identity')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'schedulefree'])
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label', required=True)
    ap.add_argument('--output_dir', default='experiments/expressivity_tasks/results')
    ap.add_argument('--eval_lengths', type=int, nargs='+', default=None,
                    help='If set, after training, eval at each of these T values '
                         '(Délétang length-extrapolation protocol). Records per-T '
                         "accuracy under log['length_extrap'].")
    ap.add_argument('--eval_interval', type=int, default=None,
                    help='Steps between S5 eval logging. Default: max(50, steps//20). '
                         'Set to 100 for dense candidate-budget calibration curves.')
    ap.add_argument('--eval_lengths_n_batches', type=int, default=8,
                    help='Number of eval batches per length in --eval_lengths.')
    ap.add_argument('--disable_autocast', action='store_true',
                    help='Run forward passes without bf16 autocast. Useful for '
                         'exact algorithmic tasks and fair comparison to M2RNN, '
                         'whose current expressivity path already disables autocast.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build task
    task_kwargs = {}
    if args.task == 'modular_counter':       task_kwargs['K'] = args.K
    elif args.task == 'dyck':                 task_kwargs['max_depth'] = args.K
    elif args.task == 'dyck2':                task_kwargs['max_depth'] = args.K
    elif args.task == 'fsm_tracking':         task_kwargs['n_states'] = args.K
    elif args.task == 'selective_copy':       task_kwargs['n_to_copy'] = args.K
    elif args.task == 'assoc_recall':         task_kwargs['n_pairs'] = args.K
    elif args.task in ('overwrite_recall', 'reset_recall'):
        task_kwargs['n_keys'] = args.K
    elif args.task == 'keyed_fsm_memory':
        task_kwargs['n_keys'] = args.K
        task_kwargs['n_states'] = args.K
    task = ALL_TASKS[args.task](**task_kwargs)
    print(f"Task: {task.name}, vocab_size={task.vocab_size}", flush=True)

    # Build hybrid model
    m2_kwargs = {}
    if args.m2rnn_q_heads is not None: m2_kwargs['num_q_heads'] = args.m2rnn_q_heads
    if args.m2rnn_k_heads is not None: m2_kwargs['num_k_heads'] = args.m2rnn_k_heads
    if args.m2rnn_v_heads is not None: m2_kwargs['num_v_heads'] = args.m2rnn_v_heads
    if args.m2rnn_f_heads is not None: m2_kwargs['num_f_heads'] = args.m2rnn_f_heads
    if args.m2rnn_g_heads is not None: m2_kwargs['num_g_heads'] = args.m2rnn_g_heads
    if args.m2rnn_weight_heads is not None: m2_kwargs['num_weight_heads'] = args.m2rnn_weight_heads
    if args.m2rnn_normalize_qk: m2_kwargs['normalize_qk'] = True
    if args.m2rnn_no_residual: m2_kwargs['use_residual'] = False
    if args.m2rnn_freeze_state_weight: m2_kwargs['state_weight_trainable'] = False
    # M2RNN raw-write state-nonlinearity ablation: --linear_state drops the tanh
    # in Z = tanh(h W + k v^T) -> Z = h W + k v^T (analogue of E88 linear_state).
    if args.linear_state is not None:
        m2_kwargs['linear_state'] = bool(args.linear_state)
    # E88-family structural overrides (only applied when explicitly passed).
    e88_kwargs = {}
    if args.linear_state is not None:
        e88_kwargs['linear_state'] = bool(args.linear_state)
    if args.use_gate is not None:
        e88_kwargs['use_gate'] = bool(args.use_gate)
    if args.decay_mode is not None:
        e88_kwargs['decay_mode'] = args.decay_mode

    def _layer_kw(level):
        if level in ('m2rnn', 'm2rnn-paper'):
            return dict(m2_kwargs)
        if isinstance(level, str) and level.startswith('E88'):
            return dict(e88_kwargs)
        return {}

    layer_kwargs = [_layer_kw(level) for level in args.layer_pattern]
    model = HybridLadderLM(
        vocab_size=task.vocab_size,
        dim=args.dim, depth=args.depth,
        layer_pattern=args.layer_pattern,
        layer_kwargs=layer_kwargs,
        n_state=args.n_state, n_heads=args.n_heads,
        expansion=args.expansion,
        rank=args.rank,
        use_triton_e88=args.use_triton_e88,
    ).to(device)
    if args.disable_autocast:
        model.disable_autocast = True
    print(f"Pattern: {model.actual_pattern}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)

    if args.optimizer == 'schedulefree':
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
        print(f"Using schedule-free AdamW (lr={args.lr})", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"Using vanilla AdamW (lr={args.lr})", flush=True)

    log = {'task': task.name, 'pattern': model.actual_pattern, 'dim': args.dim, 'depth': args.depth,
           'seq_len': args.seq_len, 'batch_size': args.batch_size, 'lr': args.lr,
           'seed': args.seed, 'params': n_params,
           'disable_autocast': bool(args.disable_autocast),
           'use_triton_e88': bool(args.use_triton_e88),
           'linear_state': args.linear_state,
           'use_gate': args.use_gate,
           'decay_mode': args.decay_mode,
           'random_baseline_acc': task.random_baseline_acc(),
           'steps': []}

    t0 = time.time()
    eval_interval = args.eval_interval if args.eval_interval is not None else max(50, args.steps // 20)
    model.train()
    if hasattr(optimizer, 'train'): optimizer.train()
    for step in range(args.steps):
        inp, tgt, mask = task.generate_batch(args.batch_size, args.seq_len, rng)
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        use_autocast = device == 'cuda' and not getattr(model, 'disable_autocast', False)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            logits = model(x)
        loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                    y.view(-1), reduction='none').view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == args.steps - 1:
            if hasattr(optimizer, 'eval'): optimizer.eval()
            acc, eval_loss = evaluate(model, task, args.batch_size, args.seq_len, 4, rng, device)
            if hasattr(optimizer, 'train'): optimizer.train()
            elapsed = time.time() - t0
            print(f"  step {step:>5d}  train_loss={loss.item():.4f}  eval_acc={acc:.4f}  eval_loss={eval_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            log['steps'].append({'step': step, 'train_loss': float(loss.item()),
                                  'eval_acc': float(acc), 'eval_loss': float(eval_loss),
                                  'elapsed_s': float(elapsed)})
            model.train()

    if hasattr(optimizer, 'eval'): optimizer.eval()
    acc, eval_loss = evaluate(model, task, args.batch_size, args.seq_len, 16, rng, device)
    log['final_acc'] = float(acc); log['final_loss'] = float(eval_loss)
    log['elapsed_total_s'] = float(time.time() - t0)
    print(f"\nFINAL: acc={acc:.4f}  loss={eval_loss:.4f}  baseline={task.random_baseline_acc():.4f}", flush=True)

    # Length-extrapolation eval (Délétang protocol): test at lengths the
    # model never trained on. A model that learned the algorithm
    # extrapolates; a model that memorized the training-length
    # distribution does not.
    if args.eval_lengths is not None:
        log['length_extrap'] = {}
        # Use a smaller per-batch B at very long T to avoid OOM.
        for T_eval in args.eval_lengths:
            B_eval = args.batch_size
            # Cap memory: scale batch down for very long sequences.
            if T_eval > 4 * args.seq_len:
                B_eval = max(2, args.batch_size // (T_eval // (4 * args.seq_len)))
            try:
                acc_T, loss_T = evaluate(
                    model, task, B_eval, T_eval,
                    args.eval_lengths_n_batches, rng, device,
                )
                print(f"  length_extrap T={T_eval:>5d} (B={B_eval}): "
                      f"acc={acc_T:.4f}  loss={loss_T:.4f}", flush=True)
                log['length_extrap'][str(T_eval)] = {
                    'acc': float(acc_T),
                    'loss': float(loss_T),
                    'B_eval': int(B_eval),
                }
            except Exception as e:
                print(f"  length_extrap T={T_eval}: ERROR {type(e).__name__}: {e}",
                      flush=True)
                log['length_extrap'][str(T_eval)] = {'error': str(e)}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.label}.json')
    json.dump(log, open(out_path, 'w'), indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
