#!/usr/bin/env python3
"""Write-strength diagnostic for the mlp-mem cell (task nlmem-capability).

The capability battery finds mlp-mem at CHANCE on parity / linear-recall / MQAR. Before
concluding "no capability", rule out the obvious confound: the mlp-mem memory may simply
be UNDER-WRITING (default eta bias -1 -> softplus(-1)*eta_max ~ 0.31 write strength;
gamma bias 4 -> 0.98 retention), so the recurrent memory contributes ~nothing and the
model falls back to conv+MLP (which cannot track parity -> chance).

This sweeps the inner-LR cap `mlp_mem_eta_max` (write strength) and trains the SAME
HybridLadderLM mlp-mem stack on parity (state-tracking) and assoc_recall (content recall).
If a stronger write makes the cell functional, the battery null is a tuning artifact and
must be re-run; if mlp-mem stays at chance even with strong writes, the failure is a real
property of the cell. REAL tasks, no mocks. Lease a GPU via gpu-broker.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from ndm.models.hybrid_ladder import HybridLadderLM
from experiments.expressivity_tasks.tasks import ALL_TASKS


def run(task_name, eta_max, gamma_bias, steps, lr, dev, seed=42, dim=256, depth=4,
        n_heads=4, n_state=32, hidden=32, T=128, B=32):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    task = ALL_TASKS[task_name]()
    lk = dict(mlp_mem_hidden=hidden, mlp_mem_eta_max=eta_max, mlp_mem_ckpt=16)
    model = HybridLadderLM(
        vocab_size=task.vocab_size, dim=dim, depth=depth,
        layer_pattern=['mlp-mem'] * depth, layer_kwargs=[lk] * depth,
        n_state=n_state, n_heads=n_heads, expansion=1.0, mlp_ratio=2.0).to(dev)
    model.disable_autocast = True
    # optionally lower the retention-gate bias so the memory forgets faster / writes net-harder
    if gamma_bias is not None:
        for lyr in model.layers:
            if hasattr(lyr, 'gamma_proj'):
                torch.nn.init.constant_(lyr.gamma_proj.bias, gamma_bias)
    import schedulefree
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.01,
                                         betas=(0.9, 0.95))
    model.train(); opt.train()
    for step in range(steps):
        inp, tgt, mask = task.generate_batch(B, T, rng)
        x = torch.from_numpy(inp).to(dev); y = torch.from_numpy(tgt).to(dev)
        m = torch.from_numpy(mask).to(dev)
        logits = model(x)
        lp = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1),
                             reduction='none').view_as(m)
        loss = (lp * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    # eval
    model.eval(); opt.eval()
    with torch.no_grad():
        cor = tot = 0
        for _ in range(8):
            inp, tgt, mask = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inp).to(dev); y = torch.from_numpy(tgt).to(dev)
            m = torch.from_numpy(mask).to(dev)
            pred = model(x).argmax(-1)
            cor += ((pred == y) & m).sum().item(); tot += m.sum().item()
    return cor / max(tot, 1), task.random_baseline_acc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--tasks', nargs='+', default=['parity', 'assoc_recall'])
    ap.add_argument('--eta_max', type=float, nargs='+', default=[1.0, 4.0, 16.0])
    ap.add_argument('--gamma_bias', type=float, default=None,
                    help='override gamma_proj bias init (default None=layer default 4.0)')
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dev = 'cuda'
    print(f"write-strength diag: steps={args.steps} lr={args.lr} "
          f"eta_max={args.eta_max} gamma_bias={args.gamma_bias}")
    for task in args.tasks:
        for em in args.eta_max:
            t0 = time.time()
            acc, base = run(task, em, args.gamma_bias, args.steps, args.lr, dev)
            print(f"  {task:14s} eta_max={em:5.1f}  acc={acc:.4f}  (baseline {base:.3f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == '__main__':
    main()
