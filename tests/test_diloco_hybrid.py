#!/usr/bin/env python3
"""diloco-loss-parity-longhorizon: REAL correctness test for the DiLoCo HYBRID
(per-step DDP WITHIN islands + DiLoCo periodic averaging ACROSS islands).

The hybrid relies on one invariant: when intra-island ranks are kept bit-identical
(which torch DDP guarantees by all-reducing gradients within the island every step),
a GLOBAL all-reduce mean over ALL ranks equals the per-ISLAND mean. So the existing
global train.diloco_merge() is the correct cross-island outer step, unchanged.

This test makes intra-island ranks identical the same way DDP would (identical data
=> identical trajectory => bit-identical weights), then runs the ACTUAL merge and
asserts:
  1. CONSENSUS: all ranks byte-identical after the merge.
  2. ISLAND-MEAN: the merged weights equal the mean of the DISTINCT per-island
     weights (not the per-rank mean) — i.e. each island counts ONCE, not island_size
     times, even though the all-reduce sums every rank.
  3. INTRA-ISLAND IDENTITY held pre-merge (sanity: the DDP-equivalent invariant).

4 gloo CPU processes = 2 islands x 2 ranks. Real model, real ScheduleFree optimizer,
real distributed collectives. No GPUs, no fabrication.
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
    import schedulefree
    torch.manual_seed(0)  # identical init on every rank
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16),
    )
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2, warmup_steps=0)
    return model, opt


def _local_train(model, opt, island_id, k_steps):
    """Train with ISLAND-shared data: ranks in the same island see identical batches
    -> identical trajectories -> bit-identical weights (exactly what intra-island DDP
    produces by gradient all-reduce). Distinct islands diverge."""
    opt.train()
    g = torch.Generator().manual_seed(100 + island_id)  # per-ISLAND, not per-rank
    for _ in range(k_steps):
        x = torch.randn(8, 16, generator=g)
        y = torch.randn(8, 16, generator=g) + island_id
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()


def _worker(rank, world_size, island_size, init_file, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    island_id = rank // island_size
    model, opt = _build()
    args = _ns(diloco_outer_lr=1.0, diloco_outer_beta=0.0)

    _local_train(model, opt, island_id, k_steps=15)

    # Pre-merge EVAL (x) weights.
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()

    # Gather every rank's pre-merge x so rank 0 can form the ISLAND mean and verify
    # intra-island identity.
    gathered = [None] * world_size
    dist.all_gather_object(gathered, [t.tolist() for t in pre_x])

    # --- THE FUNCTION UNDER TEST: GLOBAL merge across all ranks ---
    train.diloco_merge(model, opt, args, world_size, None)

    opt.eval()
    x_now = [p.data.detach().clone() for p in model.parameters()]
    opt.train()

    if rank == 0:
        n_islands = world_size // island_size
        # 3. intra-island identity (pre-merge): rank r and its island-mate identical.
        intra_max = 0.0
        for isl in range(n_islands):
            base = gathered[isl * island_size]
            for r in range(isl * island_size, (isl + 1) * island_size):
                for a, b in zip(gathered[r], base):
                    intra_max = max(intra_max,
                                    (torch.tensor(a) - torch.tensor(b)).abs().max().item())
        ret['intra_island_max_diff'] = intra_max
        # 2. expected = mean of the DISTINCT per-island weights (one rep per island).
        island_mean = []
        for j in range(len(pre_x)):
            acc = torch.zeros_like(pre_x[j])
            for isl in range(n_islands):
                acc += torch.tensor(gathered[isl * island_size][j])
            island_mean.append(acc / n_islands)
        ret['island_mean'] = [t.tolist() for t in island_mean]
        ret['x_now'] = [t.tolist() for t in x_now]

    # 1. consensus across ALL ranks.
    flat = torch._utils._flatten_dense_tensors([p.data for p in model.parameters()]).clone()
    allp = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(allp, flat)
    if rank == 0:
        ret['max_cross_rank_diff'] = max(
            (allp[r] - allp[0]).abs().max().item() for r in range(world_size))

    dist.barrier()
    dist.destroy_process_group()


def _run(world_size=4, island_size=2):
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker, args=(world_size, island_size, init_file, ret),
             nprocs=world_size, join=True)
    return dict(ret)


def _close(a, b, tol):
    return (torch.tensor(a) - torch.tensor(b)).abs().max().item() <= tol


def test_hybrid_global_merge_is_island_mean():
    r = _run(world_size=4, island_size=2)
    tol = 1e-5
    assert r['intra_island_max_diff'] <= tol, \
        f"intra-island ranks not identical pre-merge: {r['intra_island_max_diff']}"
    assert r['max_cross_rank_diff'] <= tol, \
        f"ranks diverge after merge: {r['max_cross_rank_diff']}"
    for xn, im in zip(r['x_now'], r['island_mean']):
        assert _close(xn, im, tol), "global merge != per-island mean"
    print("PASS test_hybrid_global_merge_is_island_mean: "
          "intra-island identity + consensus + global-merge==island-mean verified")


if __name__ == '__main__':
    test_hybrid_global_merge_is_island_mean()
    print("ALL DILOCO HYBRID TESTS PASSED")
