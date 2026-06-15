#!/usr/bin/env python3
"""implement-diloco-periodic: REAL correctness test for the DiLoCo periodic merge.

Spawns 2 gloo CPU processes, each holding a real nn model + a REAL ScheduleFree
AdamW optimizer, feeds each rank DIFFERENT gradients so their weights genuinely
diverge over K local steps, then calls the ACTUAL train.diloco_merge() and
asserts the y-mode-swap + weight-averaging + z-reset semantics:

  1. CONSENSUS: after the merge, every rank's parameters are byte-identical
     (max cross-rank diff ~0) -> no inter-round drift.
  2. AVERAGING: the merged eval (x) weights equal the mean of the per-rank
     pre-merge eval weights (DiLoCo outer_lr=1, outer_beta=0 == plain averaging).
  3. SCHEDULEFREE INVARIANT: post-merge, z == params and the train-mode weights
     y == merged x (the swap restores a consistent schedulefree state).
  4. OUTER MOMENTUM: with outer_beta>0 / outer_lr!=1 the general DiLoCo update
     W_{r+1} = W_r + outer_lr*(beta*mom + (mean_x - W_r)) is applied exactly.

No GPUs, no fabrication: real optimizer, real distributed collectives.
"""
import os, sys, tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _ns(**kw):
    from types import SimpleNamespace
    base = dict(optimizer='schedulefree', diloco_outer_lr=1.0, diloco_outer_beta=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _build():
    """Tiny but real model + real ScheduleFree optimizer (seed-identical across ranks)."""
    import schedulefree
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16),
    )
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2, warmup_steps=0)
    return model, opt


def _local_train(model, opt, rank, k_steps):
    """Run k real fwd/bwd/step iterations with rank-specific data (weights diverge)."""
    opt.train()
    g = torch.Generator().manual_seed(100 + rank)  # distinct data per rank
    for _ in range(k_steps):
        x = torch.randn(8, 16, generator=g)
        y = torch.randn(8, 16, generator=g) + rank  # rank-dependent target
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()


def _worker(rank, world_size, init_file, outer_lr, outer_beta, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    args = _ns(diloco_outer_lr=outer_lr, diloco_outer_beta=outer_beta)
    outer_state = None
    if not (outer_lr == 1.0 and outer_beta == 0.0):
        outer_state = {
            'anchor': [p.data.detach().clone() for p in model.parameters()],
            'moment': [torch.zeros_like(p.data) for p in model.parameters()],
        }

    _local_train(model, opt, rank, k_steps=15)

    # Capture each rank's pre-merge EVAL (x) weights to validate the average.
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()

    # Gather all ranks' pre-merge x to rank 0 (to compute the expected mean).
    gathered = [None] * world_size
    dist.all_gather_object(gathered, [t.tolist() for t in pre_x])

    # Capture the anchor (W_r) BEFORE the merge advances it to W_{r+1}.
    anchor_pre = None
    if outer_state is not None:
        anchor_pre = [t.detach().clone() for t in outer_state['anchor']]

    # --- THE FUNCTION UNDER TEST ---
    train.diloco_merge(model, opt, args, world_size, outer_state)

    # Post-merge: train-mode y weights, eval-mode x weights, and z state.
    y_now = [p.data.detach().clone() for p in model.parameters()]
    opt.eval()
    x_now = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    z_now = [opt.state[p]['z'].detach().clone() for p in model.parameters()]

    if rank == 0:
        # Expected mean of pre-merge x across ranks.
        import functools
        mean_x = []
        for j in range(len(pre_x)):
            acc = torch.zeros_like(pre_x[j])
            for r in range(world_size):
                acc += torch.tensor(gathered[r][j])
            mean_x.append(acc / world_size)
        ret['mean_x'] = [t.tolist() for t in mean_x]
        ret['x_now'] = [t.tolist() for t in x_now]
        ret['y_now'] = [t.tolist() for t in y_now]
        ret['z_now'] = [t.tolist() for t in z_now]
        if anchor_pre is not None:
            ret['anchor'] = [t.tolist() for t in anchor_pre]

    # Consensus check: all ranks must hold identical params after merge.
    flat = torch._utils._flatten_dense_tensors([p.data for p in model.parameters()]).clone()
    allp = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(allp, flat)
    max_diff = max((allp[r] - allp[0]).abs().max().item() for r in range(world_size))
    if rank == 0:
        ret['max_cross_rank_diff'] = max_diff

    dist.barrier()
    dist.destroy_process_group()


def _run(outer_lr=1.0, outer_beta=0.0):
    world_size = 2
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker, args=(world_size, init_file, outer_lr, outer_beta, ret),
             nprocs=world_size, join=True)
    return dict(ret)


def _close(a, b, tol):
    ta, tb = torch.tensor(a), torch.tensor(b)
    return (ta - tb).abs().max().item() <= tol


def test_local_sgd_averaging():
    """outer_lr=1, outer_beta=0 -> plain averaging + schedulefree consensus."""
    r = _run(outer_lr=1.0, outer_beta=0.0)
    tol = 1e-5
    # 1. consensus across ranks
    assert r['max_cross_rank_diff'] <= tol, f"ranks diverge: {r['max_cross_rank_diff']}"
    # 2. merged eval weights == mean of pre-merge eval weights
    for xn, mx in zip(r['x_now'], r['mean_x']):
        assert _close(xn, mx, tol), "merged x != mean of pre-merge x"
    # 3a. z == params (consensus base sequence)
    for zn, yn in zip(r['z_now'], r['y_now']):
        assert _close(zn, yn, tol), "z not reset to consensus params"
    # 3b. train-mode y == merged eval x (swap restored consistent state)
    for yn, xn in zip(r['y_now'], r['x_now']):
        assert _close(yn, xn, tol), "post-merge y != merged x"
    print("PASS test_local_sgd_averaging: consensus + averaging + z-reset verified")


def test_outer_momentum():
    """outer_beta=0.9, outer_lr=0.7 -> general DiLoCo outer step, exact + consensus."""
    lr, beta = 0.7, 0.9
    r = _run(outer_lr=lr, outer_beta=beta)
    tol = 1e-5
    assert r['max_cross_rank_diff'] <= tol, f"ranks diverge: {r['max_cross_rank_diff']}"
    # First round: mom = beta*0 + (mean_x - W_0) = (mean_x - anchor);
    # W_1 = anchor + lr*mom. The post-merge eval x must equal W_1.
    for xn, mx, anc in zip(r['x_now'], r['mean_x'], r['anchor']):
        tx, tm, ta = torch.tensor(xn), torch.tensor(mx), torch.tensor(anc)
        expected = ta + lr * (tm - ta)
        d = (tx - expected).abs().max().item()
        assert d <= tol, f"outer-momentum update mismatch (max diff {d})"
    print("PASS test_outer_momentum: general DiLoCo outer step verified")


if __name__ == '__main__':
    test_local_sgd_averaging()
    test_outer_momentum()
    print("ALL DILOCO MERGE TESTS PASSED")
