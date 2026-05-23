"""Generic trainer: one (model, task) pair, fixed steps, report accuracy.

Uses LadderLM from ndm.models for the backbone (handles E88/FLA-GDN/etc.
uniformly with proper prenorm + residual).

Output written to {output_dir}/{label}.json with task accuracy curve.

Usage:
    python train_task.py --task parity --model E88 --dim 256 --depth 4 \\
        --steps 5000 --seq_len 256 --batch_size 64 --label parity_e88
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'elman', 'cuda'))

import torch
import torch.nn.functional as F

from experiments.expressivity_tasks.tasks import ALL_TASKS


def build_model(level, dim, depth, vocab_size, **kwargs):
    """Build a small model for the given level."""
    if level in ('m2rnn', 'm2rnn-paper'):
        from ndm.models.m2rnn_baseline import M2RNNLM

        n_state = kwargs.get('n_state', 16)
        paper_shape = level == 'm2rnn-paper'
        model = M2RNNLM(
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            n_heads=kwargs.get('n_heads', 4),
            n_state=n_state,
            expansion=kwargs.get('expansion', 1.0),
            paper_shape=paper_shape,
            k_head_dim=kwargs.get('k_head_dim', 64 if paper_shape else None),
            v_head_dim=kwargs.get('v_head_dim', n_state if paper_shape else None),
            num_q_heads=kwargs.get('num_q_heads', 1 if paper_shape else None),
            num_k_heads=kwargs.get('num_k_heads', 1 if paper_shape else None),
            num_v_heads=kwargs.get('num_v_heads', None),
            num_f_heads=kwargs.get('num_f_heads', None),
            num_g_heads=kwargs.get('num_g_heads', None),
            num_weight_heads=kwargs.get('num_weight_heads', None),
            use_residual=kwargs.get('use_residual', True),
            state_weight_trainable=kwargs.get('state_weight_trainable', True),
            use_conv=kwargs.get('use_conv', paper_shape),
            d_conv=kwargs.get('d_conv', 4),
            output_norm=kwargs.get('output_norm', paper_shape),
            normalize_qk=kwargs.get('normalize_qk', False),
            gradient_clipping=kwargs.get('gradient_clipping', 1.0 if paper_shape else None),
        )
        # XMA's Triton M2RNN path currently has a bf16-autocast compile edge
        # case in this small harness. Keep these expressivity runs in fp32.
        model.disable_autocast = True
        return model

    if level == 'mamba2':
        from ndm.models.mamba2_baseline import Mamba2LM

        return Mamba2LM(
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            d_state=kwargs.get('mamba_d_state', 64),
            expand=kwargs.get('mamba_expand', 2),
        )

    from ndm.models import LadderLM
    # Common kwargs
    common = dict(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
    )

    if level == 'E88':
        common.update(
            n_heads=kwargs.get('n_heads', 8),
            n_state=kwargs.get('n_state', 16),
            expansion=1.0,
            use_gate=1,
            gate_activation='silu',
        )
    elif level == 'E91':
        common.update(
            n_heads=kwargs.get('n_heads', 8),
            n_state=kwargs.get('n_state', 16),
            rank=kwargs.get('rank', None),  # None defaults to n_state in E91MatMat
            use_gate=True,
            gate_activation='silu',
        )
    elif level == 'E92':
        common.update(
            n_heads=kwargs.get('n_heads', 8),
            n_state=kwargs.get('n_state', 16),
        )
    elif level == 'E93':
        common.update(
            n_state=kwargs.get('n_state', 16),
            # m_state defaults to dim if not specified; user can override via n_heads*n_state convention
        )
    elif level == 'E93a_no_decay':
        common.update(
            n_state=kwargs.get('n_state', 16),
        )
    elif level == 'fla-gdn':
        common.update(
            expansion=kwargs.get('expansion', 2),
            n_heads=kwargs.get('n_heads', 4),
        )
    elif level == 'llama':
        common.update(
            n_heads=kwargs.get('n_heads', 4),
            expansion=kwargs.get('expansion', 4),
        )

    return LadderLM(level=level, **common)


def evaluate_accuracy(model, task, B, T, n_batches, rng, device):
    """Compute task accuracy over n_batches batches."""
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            inputs_np, targets_np, mask_np = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inputs_np).to(device)
            y = torch.from_numpy(targets_np).to(device)
            m = torch.from_numpy(mask_np).to(device)
            use_autocast = device == 'cuda' and not getattr(model, 'disable_autocast', False)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                logits = model(x)  # [B, T, V]
            preds = logits.argmax(dim=-1)
            correct += ((preds == y) & m).sum().item()
            total += m.sum().item()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                y.view(-1),
                reduction='none',
            ).view_as(m)
            loss = (loss * m).sum().item() / max(m.sum().item(), 1)
            losses.append(loss)
    return correct / max(total, 1), np.mean(losses)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build task
    task_kwargs = {}
    if args.task == 'modular_counter':
        task_kwargs['K'] = args.K
    elif args.task == 'dyck':
        task_kwargs['max_depth'] = args.K
    elif args.task == 'fsm_tracking':
        task_kwargs['n_states'] = args.K
    elif args.task == 'selective_copy':
        task_kwargs['n_to_copy'] = args.K
    elif args.task == 'assoc_recall':
        task_kwargs['n_pairs'] = args.K
    elif args.task in ('overwrite_recall', 'reset_recall'):
        task_kwargs['n_keys'] = args.K
    elif args.task == 'keyed_fsm_memory':
        task_kwargs['n_keys'] = args.K
        task_kwargs['n_states'] = args.K

    task = ALL_TASKS[args.task](**task_kwargs)
    print(f"Task: {task.name}, vocab_size={task.vocab_size}", flush=True)

    # Build model
    extra_model_kwargs = {}
    if args.n_heads is not None: extra_model_kwargs['n_heads'] = args.n_heads
    if args.n_state is not None: extra_model_kwargs['n_state'] = args.n_state
    if args.expansion is not None: extra_model_kwargs['expansion'] = args.expansion
    if args.rank is not None: extra_model_kwargs['rank'] = args.rank
    if args.m2rnn_q_heads is not None: extra_model_kwargs['num_q_heads'] = args.m2rnn_q_heads
    if args.m2rnn_k_heads is not None: extra_model_kwargs['num_k_heads'] = args.m2rnn_k_heads
    if args.m2rnn_v_heads is not None: extra_model_kwargs['num_v_heads'] = args.m2rnn_v_heads
    if args.m2rnn_f_heads is not None: extra_model_kwargs['num_f_heads'] = args.m2rnn_f_heads
    if args.m2rnn_g_heads is not None: extra_model_kwargs['num_g_heads'] = args.m2rnn_g_heads
    if args.m2rnn_weight_heads is not None: extra_model_kwargs['num_weight_heads'] = args.m2rnn_weight_heads
    if args.m2rnn_normalize_qk:
        extra_model_kwargs['normalize_qk'] = True
    if args.m2rnn_no_residual:
        extra_model_kwargs['use_residual'] = False
    if args.m2rnn_freeze_state_weight:
        extra_model_kwargs['state_weight_trainable'] = False
    model = build_model(args.model, args.dim, args.depth, task.vocab_size, **extra_model_kwargs)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, dim={args.dim}, depth={args.depth}, params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    log = {'task': task.name, 'model': args.model, 'dim': args.dim, 'depth': args.depth,
           'seq_len': args.seq_len, 'batch_size': args.batch_size, 'lr': args.lr,
           'seed': args.seed, 'params': n_params,
           'random_baseline_acc': task.random_baseline_acc(),
           'steps': []}

    t0 = time.time()
    eval_interval = max(50, args.steps // 20)

    model.train()
    for step in range(args.steps):
        inputs_np, targets_np, mask_np = task.generate_batch(args.batch_size, args.seq_len, rng)
        x = torch.from_numpy(inputs_np).to(device)
        y = torch.from_numpy(targets_np).to(device)
        m = torch.from_numpy(mask_np).to(device)

        use_autocast = device == 'cuda' and not getattr(model, 'disable_autocast', False)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            logits = model(x)
        loss_per = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            y.view(-1),
            reduction='none',
        ).view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == args.steps - 1:
            acc, eval_loss = evaluate_accuracy(model, task, args.batch_size, args.seq_len,
                                                4, rng, device)
            elapsed = time.time() - t0
            print(f"  step {step:>5d}  train_loss={loss.item():.4f}  eval_acc={acc:.4f}  eval_loss={eval_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            log['steps'].append({'step': step, 'train_loss': float(loss.item()),
                                  'eval_acc': float(acc), 'eval_loss': float(eval_loss),
                                  'elapsed_s': float(elapsed)})
            model.train()

    # Final eval (more batches for stable estimate)
    acc, eval_loss = evaluate_accuracy(model, task, args.batch_size, args.seq_len,
                                        16, rng, device)
    log['final_acc'] = float(acc)
    log['final_loss'] = float(eval_loss)
    log['elapsed_total_s'] = float(time.time() - t0)
    print(f"\nFINAL: acc={acc:.4f}  loss={eval_loss:.4f}  baseline={task.random_baseline_acc():.4f}", flush=True)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(ALL_TASKS.keys()))
    ap.add_argument('--model', required=True, help='Level for LadderLM (E88, fla-gdn, mamba2, llama)')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=None)
    ap.add_argument('--n_state', type=int, default=None)
    ap.add_argument('--expansion', type=float, default=None)
    ap.add_argument('--rank', type=int, default=None, help='Rank for E91 matrix-matrix update')
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
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--K', type=int, default=5, help='Task-specific K (mod base, max depth, n_states, n_to_copy, n_pairs)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label', required=True)
    ap.add_argument('--output_dir', default='experiments/expressivity_tasks/results')
    args = ap.parse_args()
    train(args)
