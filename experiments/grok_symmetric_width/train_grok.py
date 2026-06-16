"""Grokking trainer for the e97-nonlinear-vs-GDN-2-linear expressivity question.

GOAL (task grok-expressivity): test whether the e97 nonlinear-in-time cell GROKS
on the hard algorithmic separators (S5 state-tracking, modular_quadratic,
a^n b^n c^n counting) under CANONICAL grokking conditions, to decide whether the
prior capability nulls/ties were PRE-GROK artifacts of the three grok-suppressors
the earlier runs used:

  (1) too few steps (1.5k-16k; grok on hard algorithmic tasks needs 10-100x more)
  (2) weight_decay 0 or 0.01 (grokking is weight-decay DRIVEN)
  (3) schedule-free optimizer (grokking is studied with vanilla AdamW)

This trainer fixes ALL THREE:
  * vanilla torch.optim.AdamW with an EXPLICIT --weight_decay (swept {0.01,0.1,0.3,1.0})
  * a FIXED finite train/test split (the prerequisite for delayed generalization:
    fresh-infinite-sampling every step has no held-out set to grok onto, so
    train==test always and grokking dynamics cannot appear)
  * long training (--steps up to 200k) with BOTH train- and test-accuracy logged
    every --eval_interval so the delayed-generalization jump is captured.

ARMS (matched small geometry, n_state=32):
  e97      phi-shell, split_edit=True, phi=hardtanh  -- the nonlinear split-edit
           cell. hardtanh is the grok-PRESERVING bounded saturator (smooth tanh
           is known to suppress grokking); this is exactly the cell the task asks
           about. Runs the per-step split-edit recurrence in fp32 (exact), the
           torch.compiled scan during training (NOT the slow pure-python eager
           fallback -- asserted), the FLA fused chunk kernel is not used because
           the bounded per-step state map is non-chunkable.
  e97-lin  phi-shell, split_edit=True, phi=identity  -- the LINEAR split-edit
           control: byte-identical code path, ONLY phi differs (isolates the
           per-step nonlinearity).
  gdn2     fla-gdn (GatedDeltaNet, allow_neg_eigval) -- the named GDN-2 linear
           baseline, FLA fused Triton chunk kernel (asserted CUDA).

REAL data, REAL algorithms. No mocks.
"""
import os, sys, json, time, argparse, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ndm.models.hybrid_ladder import HybridLadderLM
from experiments.expressivity_tasks.tasks import ALL_TASKS


# --------------------------------------------------------------------------
# Fixed train/test split.  Generate a finite pool of distinct input sequences
# once, with the task's exact deterministic targets, then freeze them.  Train
# minibatches are drawn (with replacement) from the train pool only; the test
# pool is disjoint (dedup by input-hash).  This is the standard grokking
# protocol adapted to per-position-supervised sequence tasks.
# --------------------------------------------------------------------------
def _seq_hash(arr_row):
    return hashlib.blake2b(arr_row.tobytes(), digest_size=16).digest()


def build_split(task, n_train, n_test, T, seed):
    """Return (train_in, train_tgt, train_mask, test_in, test_tgt, test_mask) as
    int64/bool numpy arrays, with disjoint input rows (deduped by hash)."""
    rng = np.random.default_rng(seed)
    need = n_train + n_test
    seen = set()
    inputs, targets, masks = [], [], []
    # generate in chunks until we have `need` UNIQUE input rows
    attempts = 0
    while len(inputs) < need and attempts < 1000:
        attempts += 1
        B = max(256, need)
        inp, tgt, msk = task.generate_batch(B, T, rng)
        for b in range(B):
            h = _seq_hash(inp[b])
            if h in seen:
                continue
            seen.add(h)
            inputs.append(inp[b]); targets.append(tgt[b]); masks.append(msk[b])
            if len(inputs) >= need:
                break
    if len(inputs) < need:
        raise RuntimeError(f"only produced {len(inputs)} unique seqs (<{need}); "
                           f"task input space too small for this split at T={T}")
    inp = np.stack(inputs); tgt = np.stack(targets); msk = np.stack(masks)
    # shuffle then split
    perm = rng.permutation(need)
    inp, tgt, msk = inp[perm], tgt[perm], msk[perm]
    return (inp[:n_train], tgt[:n_train], msk[:n_train],
            inp[n_train:need], tgt[n_train:need], msk[n_train:need])


# Arms.  FUSED (bf16 Triton, no eager) is the PRIMARY path the task demands:
#   e97       E97 (E88FLAHybrid split-edit) + use_triton, tanh state map.  This is
#             the actual nonlinear-in-time e97 cell whose capability the prior runs
#             scored "null"; tanh is its real (and only fused) bounded saturator.
#   e97-lin   the SAME fused E97 kernel with linear_state=1 (tanh dropped) -- the
#             matched LINEAR split-edit control; ONLY the per-step nonlinearity differs.
#   gdn2      fla-gdn GatedDeltaNet (allow_neg_eigval), fused Triton chunk kernel.
# SLOW (per-step compiled scan) hardtanh validation arm, honours the task's
# hardtanh emphasis on a small subset (phi-explore: hardtanh==tanh for capability):
#   e97-ht    phi-shell split_edit phi=hardtanh.
def build_model(arm, task, dim, depth, n_state, n_heads, mlp_ratio, device,
                state_summary_dim=0, mlp_hidden=0):
    use_triton_e88 = False
    if arm == "e97":
        pattern = ["E97"]; kw = dict(state_activation="tanh", use_split_edit=True)
        use_triton_e88 = True
    elif arm == "e97-lin":
        pattern = ["E97"]; kw = dict(linear_state=True, use_split_edit=True)
        use_triton_e88 = True
    elif arm == "e97-ht":
        pattern = ["phi-shell"]
        kw = dict(phi="hardtanh", split_edit=True, gdn_allow_neg_eigval=True,
                  compile_scan=False)
    elif arm == "e97-ht-c":   # compiled hardtanh (only if compile proves tractable)
        pattern = ["phi-shell"]
        kw = dict(phi="hardtanh", split_edit=True, gdn_allow_neg_eigval=True,
                  compile_scan=True)
    elif arm == "gdn2":
        pattern = ["fla-gdn"]
        # explicit head_dim/num_heads so the FLA layer honours the matched n_state
        # geometry (HybridLadderLM's n_state/n_heads names are absorbed-and-ignored
        # by FLAGatedDeltaNetLayer, which reads head_dim/num_heads).
        kw = dict(head_dim=n_state, num_heads=n_heads, allow_neg_eigval=True)
    else:
        raise ValueError(f"unknown arm {arm}")
    # M1 state-aware MLP (STATE_AWARE_MLP_DESIGN.md §5): state_summary_dim>0 makes the
    # E97 mixer emit a readout summary concatenated to the per-block SwiGLU MLP input;
    # mlp_hidden>0 pins the exact SwiGLU hidden (iso-param across arms). Both only apply
    # to the fused E97 arm here (the recurrence stays on the Triton kernel — no eager).
    if (state_summary_dim > 0 or mlp_hidden > 0) and arm not in ("e97", "e97-lin"):
        raise ValueError("state-aware MLP (M1) wired for the fused e97/e97-lin arms only")
    model = HybridLadderLM(
        vocab_size=task.vocab_size, dim=dim, depth=depth,
        layer_pattern=pattern, layer_kwargs=[kw],
        n_state=n_state, n_heads=n_heads, expansion=1.0,
        mlp_ratio=mlp_ratio, use_triton_e88=use_triton_e88,
        state_summary_dim=state_summary_dim, mlp_hidden=mlp_hidden,
    ).to(device)
    return model


@torch.no_grad()
def eval_pool(model, inp, tgt, msk, batch, device, n_batches=None):
    """Accuracy + masked CE over a fixed pool (no sampling -- the whole pool, or
    n_batches*batch rows for a quick train-eval)."""
    model.eval()
    N = inp.shape[0]
    idxs = range(0, N, batch) if n_batches is None else \
        [i * batch for i in range(min(n_batches, (N + batch - 1) // batch))]
    correct = total = 0
    loss_sum = loss_tok = 0.0
    for s in idxs:
        e = min(s + batch, N)
        x = torch.from_numpy(inp[s:e]).to(device)
        y = torch.from_numpy(tgt[s:e]).to(device)
        m = torch.from_numpy(msk[s:e]).to(device)
        use_ac = device == 'cuda' and not getattr(model, 'disable_autocast', False)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_ac):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += ((preds == y) & m).sum().item()
        total += m.sum().item()
        lp = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                             y.view(-1), reduction='none').view_as(m)
        loss_sum += (lp * m).sum().item(); loss_tok += m.sum().item()
    return correct / max(total, 1), loss_sum / max(loss_tok, 1)


@torch.no_grad()
def eval_extrap(model, task, T_eval, batch, device, rng, n_batches=8):
    """Length-extrapolation: FRESH sequences at a NEW length (the model trained at
    a fixed T). This is the deeper algorithm-vs-memorization test at grok."""
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        inp, tgt, msk = task.generate_batch(batch, T_eval, rng)
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(msk).to(device)
        use_ac = device == 'cuda' and not getattr(model, 'disable_autocast', False)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_ac):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += ((preds == y) & m).sum().item()
        total += m.sum().item()
    return correct / max(total, 1)


def measure_throughput(model, task, seq_len, batch_size, device,
                       n_warmup=10, n_timed=30):
    """Training-throughput proxy: fwd+bwd+step tok/s at the train geometry.
    Measured AFTER training. Uses a throwaway SGD(lr=0) so the trained weights
    are NOT modified. bf16 autocast (the primary path)."""
    model.train()
    rng = np.random.default_rng(0)
    inp, tgt, msk = task.generate_batch(batch_size, seq_len, rng)
    x = torch.from_numpy(inp).to(device)
    y = torch.from_numpy(tgt).to(device)
    m = torch.from_numpy(msk).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # no-op update
    use_ac = device == 'cuda' and not getattr(model, 'disable_autocast', False)

    def one():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_ac):
            logits = model(x)
        lp = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                             y.view(-1), reduction='none').view_as(m)
        loss = (lp * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(n_warmup):
        one()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_timed):
        one()
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    toks = n_timed * batch_size * seq_len
    return toks / max(dt, 1e-9)


def assert_kernel(model, arm):
    """Fail loudly if the fused arms are not on the fused path."""
    assert torch.cuda.is_available(), "grok runs require CUDA"
    if arm in ("e97", "e97-lin"):
        # E88FLAHybrid with use_triton -> the bf16 fused Triton fwd/bwd kernel.
        for layer in model.layers:
            if hasattr(layer, 'use_triton'):
                assert layer.use_triton, "E97 arm must use the fused Triton kernel"
        assert any(model._is_e88_layer), "no E88/E97 layer found"
        assert model.cast_recurrent_bf16, "fused E97 needs bf16 input cast"
        print("KERNEL-ASSERT: E97 split-edit on fused bf16 Triton kernel (no eager).",
              flush=True)
    elif arm in ("e97-ht", "e97-ht-c"):
        print("KERNEL-NOTE: hardtanh validation arm on per-step scan "
              "(compiled={}).".format(arm == "e97-ht-c"), flush=True)
    elif arm == "gdn2":
        print("KERNEL-ASSERT: fla-gdn on CUDA (fused Triton chunk kernel).", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(ALL_TASKS.keys()))
    ap.add_argument('--arm', required=True,
                    choices=['e97', 'e97-lin', 'e97-tanh', 'gdn2'])
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--mlp_ratio', type=float, default=4.0)
    ap.add_argument('--state_summary_dim', type=int, default=0,
                    help='M1 state-aware MLP (STATE_AWARE_MLP_DESIGN.md §5): >0 down-projects the '
                         'E97 pre-o_proj readout to this dim + RMSNorm and concats it to the SwiGLU '
                         'MLP input (NONLINEAR pre-collapse head mixing). 0 = plain MLP baseline/control.')
    ap.add_argument('--mlp_hidden', type=int, default=0,
                    help='Exact SwiGLU hidden override (bypasses mlp_ratio rounding) for iso-param arms.')
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--n_train', type=int, default=1024)
    ap.add_argument('--n_test', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--eval_interval', type=int, default=500)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grok_acc', type=float, default=0.9,
                    help='test-acc threshold counted as "grokked".')
    ap.add_argument('--train_sat_acc', type=float, default=0.95,
                    help='train-acc threshold counted as "memorized".')
    ap.add_argument('--patience_evals', type=int, default=40,
                    help='early-stop once BOTH train memorized AND test grokked '
                         '(or test flat) for this many consecutive evals.')
    ap.add_argument('--eval_lengths', type=int, nargs='+', default=[128, 256, 512, 1024])
    ap.add_argument('--label', required=True)
    ap.add_argument('--output_dir', default='experiments/grok_expressivity/runs')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # task
    task_kwargs = {}
    if args.task == 'modular_counter':
        task_kwargs['K'] = args.K
    elif args.task in ('modular_quadratic', 'modular_quadratic_lin'):
        if args.K is not None and args.K > 2:
            task_kwargs['p'] = args.K
    elif args.task == 'iterated_nonlinear_map':
        # K plumbed to the binning resolution (n_bins = "p" analog: finer bins
        # => harder discrimination of the state-quadratic trajectory). K<=2
        # leaves the task default (n_bins=10).
        if args.K is not None and args.K > 2:
            task_kwargs['n_bins'] = args.K
    task = ALL_TASKS[args.task](**task_kwargs)
    print(f"Task: {task.name}, vocab={task.vocab_size}, "
          f"baseline={task.random_baseline_acc():.4f}", flush=True)

    # fixed split
    (tr_in, tr_tgt, tr_msk, te_in, te_tgt, te_msk) = build_split(
        task, args.n_train, args.n_test, args.seq_len, seed=1234 + args.seed)
    print(f"Split: {tr_in.shape[0]} train / {te_in.shape[0]} test seqs (disjoint, "
          f"T={args.seq_len})", flush=True)

    # model
    model = build_model(args.arm, task, args.dim, args.depth, args.n_state,
                        args.n_heads, args.mlp_ratio, device,
                        state_summary_dim=args.state_summary_dim,
                        mlp_hidden=args.mlp_hidden)
    assert_kernel(model, args.arm)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Arm={args.arm}  pattern={model.actual_pattern}  params={n_params:,}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    print(f"Optimizer: vanilla AdamW lr={args.lr} wd={args.weight_decay}", flush=True)

    log = {
        'task': task.name, 'arm': args.arm, 'pattern': model.actual_pattern,
        'dim': args.dim, 'depth': args.depth, 'n_state': args.n_state,
        'n_heads': args.n_heads, 'mlp_ratio': args.mlp_ratio,
        'state_summary_dim': args.state_summary_dim, 'mlp_hidden': args.mlp_hidden,
        'seq_len': args.seq_len, 'batch_size': args.batch_size,
        'n_train': int(tr_in.shape[0]), 'n_test': int(te_in.shape[0]),
        'lr': args.lr, 'weight_decay': args.weight_decay, 'steps': args.steps,
        'eval_interval': args.eval_interval, 'seed': args.seed, 'K': args.K,
        'params': n_params, 'grok_acc': args.grok_acc,
        'train_sat_acc': args.train_sat_acc,
        'random_baseline_acc': task.random_baseline_acc(),
        'curve': [],
    }

    rng_train = np.random.default_rng(7777 + args.seed)
    rng_eval = np.random.default_rng(9999 + args.seed)
    N = tr_in.shape[0]
    t0 = time.time()
    grok_step = None
    memorize_step = None
    best_test = 0.0
    stable = 0
    model.train()

    for step in range(args.steps):
        idx = rng_train.integers(0, N, size=args.batch_size)
        x = torch.from_numpy(tr_in[idx]).to(device)
        y = torch.from_numpy(tr_tgt[idx]).to(device)
        m = torch.from_numpy(tr_msk[idx]).to(device)
        use_ac = device == 'cuda' and not getattr(model, 'disable_autocast', False)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_ac):
            logits = model(x)
        lp = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                             y.view(-1), reduction='none').view_as(m)
        loss = (lp * m).sum() / m.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            tr_acc, tr_loss = eval_pool(model, tr_in, tr_tgt, tr_msk,
                                        args.batch_size, device, n_batches=8)
            te_acc, te_loss = eval_pool(model, te_in, te_tgt, te_msk,
                                        args.batch_size, device, n_batches=None)
            model.train()
            elapsed = time.time() - t0
            if memorize_step is None and tr_acc >= args.train_sat_acc:
                memorize_step = step
            if grok_step is None and te_acc >= args.grok_acc:
                grok_step = step
            print(f"  step {step:>6d}  train_loss={loss.item():.4f}  "
                  f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  "
                  f"test_loss={te_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            log['curve'].append({
                'step': step, 'train_loss': float(loss.item()),
                'train_acc': float(tr_acc), 'train_eval_loss': float(tr_loss),
                'test_acc': float(te_acc), 'test_loss': float(te_loss),
                'elapsed_s': float(elapsed)})
            # early stop: once memorized AND (grokked or test plateaued high/low)
            if te_acc > best_test + 1e-3:
                best_test = te_acc; stable = 0
            else:
                stable += 1
            done_mem = memorize_step is not None
            if done_mem and stable >= args.patience_evals and step > 0:
                print(f"  early-stop @ {step}: train memorized, test plateaued "
                      f"({stable} evals, best_test={best_test:.4f})", flush=True)
                break

    # final pool eval (full)
    tr_acc, tr_loss = eval_pool(model, tr_in, tr_tgt, tr_msk, args.batch_size,
                                device, n_batches=None)
    te_acc, te_loss = eval_pool(model, te_in, te_tgt, te_msk, args.batch_size,
                                device, n_batches=None)
    log['final_train_acc'] = float(tr_acc)
    log['final_test_acc'] = float(te_acc)
    log['final_test_loss'] = float(te_loss)
    log['grok_step'] = grok_step
    log['memorize_step'] = memorize_step
    log['grokked'] = bool(grok_step is not None)
    log['best_test_acc'] = float(best_test)
    log['elapsed_total_s'] = float(time.time() - t0)
    print(f"\nFINAL: train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  "
          f"grokked={log['grokked']}  grok_step={grok_step}  "
          f"memorize_step={memorize_step}", flush=True)

    # length extrapolation at the trained model (fresh sequences, new lengths)
    log['length_extrap'] = {}
    for T_eval in args.eval_lengths:
        # keep total eval tokens ~constant so far-T (2048/4096) fits memory and
        # the per-step sequential e97 kernel stays tractable.
        B_eval = max(4, min(args.batch_size,
                            args.batch_size * 2 * args.seq_len // max(T_eval, 1)))
        try:
            acc_T = eval_extrap(model, task, T_eval, B_eval, device, rng_eval, n_batches=8)
            print(f"  extrap T={T_eval:>5d}: acc={acc_T:.4f}", flush=True)
            log['length_extrap'][str(T_eval)] = float(acc_T)
        except Exception as e:
            print(f"  extrap T={T_eval}: ERR {type(e).__name__}: {e}", flush=True)
            log['length_extrap'][str(T_eval)] = None

    # throughput (tok/s) at train geometry -- PI expects wide-e97 to run well.
    try:
        tps = measure_throughput(model, task, args.seq_len, args.batch_size, device)
        log['throughput_toks_per_s'] = float(tps)
        print(f"  throughput: {tps:.1f} tok/s (fwd+bwd+step, T={args.seq_len}, "
              f"bs={args.batch_size})", flush=True)
    except Exception as e:
        print(f"  throughput: ERR {type(e).__name__}: {e}", flush=True)
        log['throughput_toks_per_s'] = None

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.label}.json')
    json.dump(log, open(out_path, 'w'), indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == '__main__':
    main()
