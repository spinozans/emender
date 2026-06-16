#!/usr/bin/env python3
"""implement-diloco-periodic: REAL correctness test for the DiLoCo periodic merge.

Spawns 2 gloo CPU processes, each holding a real nn model + a REAL ScheduleFree
AdamW optimizer, feeds each rank DIFFERENT gradients so their weights genuinely
diverge over K local steps, then calls the ACTUAL train.diloco_merge() and
asserts the y-mode-swap + full ScheduleFree-state averaging semantics:

  1. CONSENSUS: after the merge, every rank's parameters are byte-identical
     (max cross-rank diff ~0) -> no inter-round drift.
  2. AVERAGING: the merged eval (x) weights equal the mean of the per-rank
     pre-merge eval weights and z equals the mean pre-merge z.
  3. SCHEDULEFREE INVARIANT: post-merge, weight_sum/k/lr_max are preserved, and
     train-mode y is rebuilt from the merged x/z pair. The SF averaging
     denominator is never reset.
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


def _loss(model, x, y):
    with torch.no_grad():
        return ((model(x) - y) ** 2).mean().item()


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

    # Capture each rank's pre-merge EVAL (x) and base (z) weights to validate
    # full-state averaging.
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    pre_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]
    group_pre = {
        'k': opt.param_groups[0]['k'],
        'weight_sum': opt.param_groups[0]['weight_sum'],
        'lr_max': opt.param_groups[0]['lr_max'],
    }

    # Gather all ranks' pre-merge x/z to rank 0 (to compute expected means).
    gathered = [None] * world_size
    dist.all_gather_object(gathered, {
        'x': [t.tolist() for t in pre_x],
        'z': [t.tolist() for t in pre_z],
        'group': group_pre,
    })

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
        # Expected means of pre-merge x and z across ranks.
        mean_x = []
        mean_z = []
        for j in range(len(pre_x)):
            acc = torch.zeros_like(pre_x[j])
            z_acc = torch.zeros_like(pre_z[j])
            for r in range(world_size):
                acc += torch.tensor(gathered[r]['x'][j])
                z_acc += torch.tensor(gathered[r]['z'][j])
            mean_x.append(acc / world_size)
            mean_z.append(z_acc / world_size)
        ret['mean_x'] = [t.tolist() for t in mean_x]
        ret['mean_z'] = [t.tolist() for t in mean_z]
        ret['x_now'] = [t.tolist() for t in x_now]
        ret['y_now'] = [t.tolist() for t in y_now]
        ret['z_now'] = [t.tolist() for t in z_now]
        ret['group_pre_merge'] = gathered[0]['group']
        if anchor_pre is not None:
            ret['anchor'] = [t.tolist() for t in anchor_pre]
        ret['group_after_merge'] = {
            'k': opt.param_groups[0]['k'],
            'weight_sum': opt.param_groups[0]['weight_sum'],
            'lr_max': opt.param_groups[0]['lr_max'],
        }

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


def _worker_dynamic(rank, world_size, init_file, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    args = _ns(diloco_outer_lr=1.0, diloco_outer_beta=0.0)

    # Real local training with rank-specific data creates different SF x/z
    # histories and a nonzero pre-merge averaging denominator.
    _local_train(model, opt, rank, k_steps=25)
    pre_weight_sum = opt.param_groups[0]['weight_sum']
    pre_k = opt.param_groups[0]['k']

    g = torch.Generator().manual_seed(999)
    probe_x = torch.randn(16, 16, generator=g)
    probe_y = torch.randn(16, 16, generator=g)
    pre_loss = _loss(model, probe_x, probe_y)

    train.diloco_merge(model, opt, args, world_size, None)
    merge_weight_sum = opt.param_groups[0]['weight_sum']
    merge_k = opt.param_groups[0]['k']
    merged_loss = _loss(model, probe_x, probe_y)
    opt.eval()
    merged_xz_diff = max(
        (p.data - opt.state[p]['z']).abs().max().item() for p in model.parameters()
    )
    opt.train()

    # First post-merge step. The old reset band-aid made the next ckp1 equal 1,
    # collapsing eval x onto the recent z. The principled merge preserves
    # weight_sum, so the next update keeps x as a long-run average.
    step_x = torch.randn(16, 16, generator=g)
    step_y = torch.randn(16, 16, generator=g)
    opt.zero_grad()
    step_loss = ((model(step_x) - step_y) ** 2).mean()
    step_loss.backward()
    opt.step()
    post_loss = _loss(model, probe_x, probe_y)

    opt.eval()
    post_xz_diff = 0.0
    for p in model.parameters():
        post_xz_diff = max(post_xz_diff, (p.data - opt.state[p]['z']).abs().max().item())
    opt.train()

    gathered = [None] * world_size
    dist.all_gather_object(gathered, {
        'pre_weight_sum': pre_weight_sum,
        'merge_weight_sum': merge_weight_sum,
        'post_weight_sum': opt.param_groups[0]['weight_sum'],
        'pre_k': pre_k,
        'merge_k': merge_k,
        'pre_loss': pre_loss,
        'merged_loss': merged_loss,
        'post_loss': post_loss,
        'merged_xz_diff': merged_xz_diff,
        'post_xz_diff': post_xz_diff,
    })
    if rank == 0:
        ret['dynamic'] = gathered

    dist.barrier()
    dist.destroy_process_group()


def _run_dynamic():
    world_size = 2
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker_dynamic, args=(world_size, init_file, ret),
             nprocs=world_size, join=True)
    return dict(ret)['dynamic']


def _worker_full_average(rank, world_size, init_file, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    args = _ns(diloco_outer_lr=1.0, diloco_outer_beta=0.0)

    rows = []
    for round_idx in range(4):
        _local_train(model, opt, rank, k_steps=10)
        pre_weight_sum = opt.param_groups[0]['weight_sum']
        train.diloco_merge(model, opt, args, world_size, None)
        opt.eval()
        xz_diff = max(
            (p.data - opt.state[p]['z']).abs().max().item() for p in model.parameters()
        )
        x_norm = sum(p.data.float().norm().item() for p in model.parameters())
        z_norm = sum(opt.state[p]['z'].float().norm().item() for p in model.parameters())
        opt.train()
        rows.append({
            'round': round_idx,
            'pre_weight_sum': pre_weight_sum,
            'merge_weight_sum': opt.param_groups[0]['weight_sum'],
            'k': opt.param_groups[0]['k'],
            'xz_diff': xz_diff,
            'x_norm': x_norm,
            'z_norm': z_norm,
        })

    gathered = [None] * world_size
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        ret['full_average'] = gathered

    dist.barrier()
    dist.destroy_process_group()


def _run_full_average():
    world_size = 2
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker_full_average, args=(world_size, init_file, ret),
             nprocs=world_size, join=True)
    return dict(ret)['full_average']


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
    # 3a. merged z == mean pre-merge z, not a reset to x/y.
    for zn, mz in zip(r['z_now'], r['mean_z']):
        assert _close(zn, mz, tol), "merged z != mean of pre-merge z"
    # 3b. train-mode y is rebuilt from merged x/z.
    beta1 = 0.9
    for yn, xn, zn in zip(r['y_now'], r['x_now'], r['z_now']):
        ty, tx, tz = torch.tensor(yn), torch.tensor(xn), torch.tensor(zn)
        expected_y = tx * beta1 + tz * (1.0 - beta1)
        assert (ty - expected_y).abs().max().item() <= tol, "post-merge y != SF train interpolation"
    assert r['group_after_merge']['weight_sum'] == r['group_pre_merge']['weight_sum'], (
        "SF weight_sum should be preserved across merge"
    )
    assert r['group_after_merge']['weight_sum'] > 0.0, "SF weight_sum was reset"
    assert r['group_after_merge']['k'] == 15, "Adam/SF step clock should be preserved"
    assert r['group_after_merge']['k'] == r['group_pre_merge']['k'], "SF k changed at merge"
    assert r['group_after_merge']['lr_max'] == r['group_pre_merge']['lr_max'], "SF lr_max changed at merge"
    print("PASS test_local_sgd_averaging: consensus + full SF-state averaging verified")


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


def test_schedulefree_dynamic_loss_continuity():
    """Real train -> merge -> train regression for ScheduleFree-DiLoCo dynamics."""
    rows = _run_dynamic()
    for i, row in enumerate(rows):
        assert row['pre_weight_sum'] > 0.0, f"rank {i} never built SF average"
        assert row['merge_weight_sum'] == row['pre_weight_sum'], (
            f"rank {i} merge did not preserve SF averaging denominator"
        )
        assert row['post_weight_sum'] > row['merge_weight_sum'], (
            f"rank {i} SF averaging mass was not monotonic after merge"
        )
        assert row['merge_k'] == row['pre_k'], f"rank {i} merge changed SF k"
        assert torch.isfinite(torch.tensor(row['pre_loss']))
        assert torch.isfinite(torch.tensor(row['merged_loss']))
        assert torch.isfinite(torch.tensor(row['post_loss']))
        baseline = max(row['pre_loss'], row['merged_loss'], 1e-12)
        assert row['post_loss'] <= baseline * 1.5 + 1e-5, (
            f"rank {i} post-merge loss spike: pre={row['pre_loss']:.6g} "
            f"merged={row['merged_loss']:.6g} post={row['post_loss']:.6g}"
        )
        assert row['merged_xz_diff'] > 1e-8, (
            f"rank {i} merge collapsed full-history x onto recent z"
        )
        assert row['post_xz_diff'] > 1e-8, (
            f"rank {i} first post-merge step re-anchored x to z"
        )
    print("PASS test_schedulefree_dynamic_loss_continuity: no post-merge spike")


def test_schedulefree_full_trajectory_average_survives_merges():
    """Multiple real-rank merges keep x as a long-run average instead of z reset."""
    rank_rows = _run_full_average()
    for rank, rows in enumerate(rank_rows):
        last_weight_sum = 0.0
        for row in rows:
            assert row['merge_weight_sum'] == row['pre_weight_sum'], (
                f"rank {rank} round {row['round']} reset weight_sum"
            )
            assert row['merge_weight_sum'] > last_weight_sum, (
                f"rank {rank} round {row['round']} weight_sum not monotonic"
            )
            assert row['xz_diff'] > 1e-8, (
                f"rank {rank} round {row['round']} x collapsed to z after merge"
            )
            assert torch.isfinite(torch.tensor(row['x_norm']))
            assert torch.isfinite(torch.tensor(row['z_norm']))
            last_weight_sum = row['merge_weight_sum']
    print("PASS test_schedulefree_full_trajectory_average_survives_merges")


def test_single_rank_diloco_merge_is_byte_identical_noop():
    """world_size=1 returns before SF swaps or state mutation."""
    import train
    model, opt = _build()
    _local_train(model, opt, rank=0, k_steps=3)
    before_params = [p.detach().clone() for p in model.parameters()]
    before_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]
    before_group = dict(opt.param_groups[0])

    sync_s = train.diloco_merge(model, opt, _ns(), world_size=1, outer_state=None)

    assert sync_s == 0.0
    for p, before in zip(model.parameters(), before_params):
        assert torch.equal(p, before), "single-rank merge changed live parameter bytes"
    for p, before in zip(model.parameters(), before_z):
        assert torch.equal(opt.state[p]['z'], before), "single-rank merge changed z bytes"
    for key in ('weight_sum', 'k', 'lr_max', 'scheduled_lr', 'train_mode'):
        assert opt.param_groups[0][key] == before_group[key], f"single-rank merge changed {key}"
    print("PASS test_single_rank_diloco_merge_is_byte_identical_noop")


if __name__ == '__main__':
    test_local_sgd_averaging()
    test_outer_momentum()
    test_schedulefree_dynamic_loss_continuity()
    test_schedulefree_full_trajectory_average_survives_merges()
    test_single_rank_diloco_merge_is_byte_identical_noop()
    print("ALL DILOCO MERGE TESTS PASSED")
